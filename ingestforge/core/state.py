"""
Processing State Management for IngestForge.

This module tracks document processing progress and maintains state across
application restarts. It enables recovery from failures, progress reporting,
and status dashboards.

Architecture Context
--------------------
State management is used by the Pipeline orchestrator to track documents
through processing stages:

    PENDING → SPLITTING → EXTRACTING → CHUNKING → ENRICHING → INDEXING → COMPLETED
                                                                          ↓
                                                                       FAILED

State is persisted to JSON, allowing:
- Resume processing after crashes
- Retry failed documents
- Report progress to users
- Avoid reprocessing completed documents

State Hierarchy
---------------
    ProcessingState (project-level)
    ├── project_name: str
    ├── total_documents: int
    ├── total_chunks: int
    └── documents: Dict[str, DocumentState]
                              ↓
                        DocumentState (per-document)
                        ├── document_id: str
                        ├── source_file: str
                        ├── status: ProcessingStatus
                        ├── started_at, completed_at
                        ├── total_pages, processed_pages
                        ├── total_chunks, indexed_chunks
                        ├── chapters: List[str]
                        └── chunk_ids: List[str]

Key Components
--------------
**ProcessingStatus**
    Enumeration of document processing stages. Progress flows through:
    PENDING → SPLITTING → EXTRACTING → CHUNKING → ENRICHING → INDEXING → COMPLETED
    Any stage can transition to FAILED.

**DocumentState**
    Tracks a single document's processing lifecycle:
    - File metadata (size, hash for deduplication)
    - Processing progress (pages processed, chunks created)
    - Timing (started_at, completed_at)
    - Error information on failure

**ProcessingState**
    Aggregates state for all documents in a project:
    - Query methods: get_pending_documents(), get_failed_documents()
    - Statistics: total_documents, total_chunks, total_embeddings
    - Persistence: save()/load() to JSON file

**StateManager**
    High-level API with automatic persistence:
    - Context manager for document processing
    - Automatic save on context exit
    - Error capture and state update

Usage Pattern
-------------
The StateManager provides a context manager for clean state tracking:

    manager = StateManager(state_file, project_name="my-kb")

    with manager.document(doc_id, source_file) as doc:
        doc.status = ProcessingStatus.CHUNKING
        doc.total_chunks = 42
        # ... processing ...
        doc.status = ProcessingStatus.COMPLETED

    # State automatically saved on context exit
    # If exception occurs, doc.fail(error) is called

For batch operations, direct access is also available:

    state = ProcessingState.load(state_file)
    pending = state.get_pending_documents()
    for doc in pending:
        process_document(doc)
    state.save(state_file)

Design Decisions
----------------
1. **JSON persistence**: Human-readable, easy debugging, no dependencies.
2. **Context manager pattern**: Ensures state is always saved, even on error.
3. **Document-level granularity**: Can resume partially processed batches.
4. **Hash-based deduplication**: Detect if document was already processed.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Type
from types import TracebackType

try:
    from filelock import FileLock, Timeout as FileLockTimeout

    HAS_FILELOCK = True
except ImportError:
    HAS_FILELOCK = False
    FileLock = None  # type: ignore[misc,assignment]
    FileLockTimeout = Exception  # type: ignore[misc,assignment]


class ProcessingStatus(str, Enum):
    """Document processing status."""

    PENDING = "pending"
    SPLITTING = "splitting"
    EXTRACTING = "extracting"
    CHUNKING = "chunking"
    ENRICHING = "enriching"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class DocumentState:
    """State of a document being processed."""

    document_id: str
    source_file: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None

    # Processing progress
    total_pages: int = 0
    processed_pages: int = 0
    total_chunks: int = 0
    indexed_chunks: int = 0

    # Split chapters
    chapters: List[str] = field(default_factory=list)

    # Chunk IDs
    chunk_ids: List[str] = field(default_factory=list)

    # Metadata
    title: Optional[str] = None
    author: Optional[str] = None
    file_size_bytes: int = 0
    file_hash: Optional[str] = None

    def start_processing(self) -> None:
        """Mark document as started."""
        self.status = ProcessingStatus.SPLITTING
        self.started_at = datetime.now().isoformat()

    def complete(self) -> None:
        """Mark document as completed."""
        self.status = ProcessingStatus.COMPLETED
        self.completed_at = datetime.now().isoformat()

    def fail(self, error: str) -> None:
        """Mark document as failed."""
        self.status = ProcessingStatus.FAILED
        self.error_message = error
        self.completed_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentState":
        """Create from dictionary."""
        data = data.copy()
        if "status" in data:
            data["status"] = ProcessingStatus(data["status"])
        return cls(**data)


@dataclass
class ProcessingState:
    """Overall processing state for IngestForge."""

    project_name: str = "default"
    documents: Dict[str, DocumentState] = field(default_factory=dict)
    last_updated: Optional[str] = None

    # Statistics
    total_documents: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0

    def add_document(self, doc_state: DocumentState) -> None:
        """Add or update a document state."""
        self.documents[doc_state.document_id] = doc_state
        self.total_documents = len(self.documents)
        self._update_timestamp()

    def get_document(self, document_id: str) -> Optional[DocumentState]:
        """Get document state by ID."""
        return self.documents.get(document_id)

    def get_pending_documents(self) -> List[DocumentState]:
        """Get all pending documents."""
        return [
            doc
            for doc in self.documents.values()
            if doc.status == ProcessingStatus.PENDING
        ]

    def get_failed_documents(self) -> List[DocumentState]:
        """Get all failed documents."""
        return [
            doc
            for doc in self.documents.values()
            if doc.status == ProcessingStatus.FAILED
        ]

    def get_completed_documents(self) -> List[DocumentState]:
        """Get all completed documents."""
        return [
            doc
            for doc in self.documents.values()
            if doc.status == ProcessingStatus.COMPLETED
        ]

    def get_in_progress_documents(self) -> List[DocumentState]:
        """Get all in-progress documents."""
        in_progress_statuses = {
            ProcessingStatus.SPLITTING,
            ProcessingStatus.EXTRACTING,
            ProcessingStatus.CHUNKING,
            ProcessingStatus.ENRICHING,
            ProcessingStatus.INDEXING,
        }
        return [
            doc for doc in self.documents.values() if doc.status in in_progress_statuses
        ]

    def update_statistics(self) -> None:
        """Update aggregate statistics."""
        self.total_documents = len(self.documents)
        self.total_chunks = sum(doc.total_chunks for doc in self.documents.values())
        self.total_embeddings = sum(
            doc.indexed_chunks
            for doc in self.documents.values()
            if doc.status == ProcessingStatus.COMPLETED
        )
        self._update_timestamp()

    def _update_timestamp(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "project_name": self.project_name,
            "documents": {
                doc_id: doc.to_dict() for doc_id, doc in self.documents.items()
            },
            "last_updated": self.last_updated,
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_embeddings": self.total_embeddings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingState":
        """Create from dictionary."""
        documents = {}
        for doc_id, doc_data in data.get("documents", {}).items():
            documents[doc_id] = DocumentState.from_dict(doc_data)

        return cls(
            project_name=data.get("project_name", "default"),
            documents=documents,
            last_updated=data.get("last_updated"),
            total_documents=data.get("total_documents", 0),
            total_chunks=data.get("total_chunks", 0),
            total_embeddings=data.get("total_embeddings", 0),
        )

    def save(self, state_file: Path, lock_timeout: float = 10.0) -> None:
        """
        Save state to JSON file with atomic write and file locking.

        Uses file locking to prevent race conditions when multiple processes
        write to the same state file (e.g., during parallel processing).

        Args:
            state_file: Path to the state JSON file
            lock_timeout: Maximum time to wait for lock (seconds)
        """
        state_file = Path(state_file)
        lock_file = state_file.with_suffix(".json.lock")
        temp_file = state_file.with_suffix(".json.tmp")

        def _write_atomic() -> None:
            """Write to temp file then atomic rename."""
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            # Atomic rename (on POSIX; Windows may need retry)
            try:
                temp_file.replace(state_file)
            except OSError:
                # Windows fallback: remove then rename
                if state_file.exists():
                    os.remove(state_file)
                temp_file.rename(state_file)

        if HAS_FILELOCK:
            lock = FileLock(lock_file, timeout=lock_timeout)
            try:
                with lock:
                    _write_atomic()
            except FileLockTimeout:
                print(
                    f"Warning: Could not acquire lock for {state_file} after {lock_timeout}s"
                )
                # Fallback to non-locked write
                _write_atomic()
        else:
            # No filelock available, write directly
            _write_atomic()

    @classmethod
    def load(cls, state_file: Path, lock_timeout: float = 10.0) -> "ProcessingState":
        """
        Load state from JSON file with file locking.

        Uses file locking to prevent reading while another process is writing.

        Args:
            state_file: Path to the state JSON file
            lock_timeout: Maximum time to wait for lock (seconds)

        Returns:
            Loaded ProcessingState or empty state if file doesn't exist
        """
        state_file = Path(state_file)
        if not state_file.exists():
            return cls()

        def _read_state() -> "ProcessingState":
            """Read and parse state file."""
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)

        try:
            return cls._load_with_lock(state_file, lock_timeout, _read_state)
        except Exception as e:
            print(f"Warning: Could not load state from {state_file}: {e}")
            return cls()

    @classmethod
    def _load_with_lock(
        cls,
        state_file: Path,
        lock_timeout: float,
        read_func: Callable[[], "ProcessingState"],
    ) -> "ProcessingState":
        """Load state with file locking.

        Args:
            state_file: State file path
            lock_timeout: Lock timeout seconds
            read_func: Function to read state

        Returns:
            ProcessingState
        """
        if not HAS_FILELOCK:
            return read_func()

        lock_file = state_file.with_suffix(".json.lock")
        lock = FileLock(lock_file, timeout=lock_timeout)
        try:
            with lock:
                return read_func()
        except FileLockTimeout:
            print(
                f"Warning: Could not acquire lock for {state_file} after {lock_timeout}s"
            )
            return read_func()


class StateManager:
    """
    Manages processing state with automatic persistence.

    Usage:
        manager = StateManager(state_file)
        with manager.document(doc_id, source_file) as doc:
            doc.status = ProcessingStatus.CHUNKING
            doc.total_chunks = 42
        # State is automatically saved on context exit
    """

    def __init__(self, state_file: Path, project_name: str = "default") -> None:
        self.state_file = state_file
        self.state = ProcessingState.load(state_file)
        self.state.project_name = project_name

    def save(self) -> None:
        """Save current state."""
        self.state.update_statistics()
        self.state.save(self.state_file)

    def get_or_create_document(
        self, document_id: str, source_file: str
    ) -> DocumentState:
        """Get existing document state or create new one."""
        if document_id in self.state.documents:
            return self.state.documents[document_id]

        doc_state = DocumentState(
            document_id=document_id,
            source_file=source_file,
        )
        self.state.add_document(doc_state)
        return doc_state

    def update_document(self, doc_state: DocumentState) -> None:
        """Update document state and save."""
        self.state.add_document(doc_state)
        self.save()

    class DocumentContext:
        """Context manager for document processing."""

        def __init__(self, manager: "StateManager", doc_state: DocumentState) -> None:
            self.manager = manager
            self.doc_state = doc_state

        def __enter__(self) -> DocumentState:
            return self.doc_state

        def __exit__(
            self,
            exc_type: Optional[Type[BaseException]],
            exc_val: Optional[BaseException],
            exc_tb: Optional[TracebackType],
        ) -> Literal[False]:
            if exc_type is not None:
                self.doc_state.fail(str(exc_val))
            self.manager.update_document(self.doc_state)
            return False

    def document(self, document_id: str, source_file: str) -> DocumentContext:
        """
        Get document context manager.

        Usage:
            with manager.document("doc_123", "file.pdf") as doc:
                doc.status = ProcessingStatus.CHUNKING
        """
        doc_state = self.get_or_create_document(document_id, source_file)
        return self.DocumentContext(self, doc_state)
