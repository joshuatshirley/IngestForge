"""
Base Interfaces for Storage Backends.

This module defines the ChunkRepository interface that all storage backends must
implement. This abstraction allows swapping storage backends (JSONL, ChromaDB,
PostgreSQL, etc.) without changing the rest of the application.

TASK-017 Migration Status
--------------------------
**Current State**: Migrating from ChunkRecord to IFChunkArtifact
**Status**: Active migration (backward compatible)
**Timeline**: ChunkRecord support will be removed in v2.0

Migration Guide for Storage Backend Implementers:
1. Accept both ChunkRecord and IFChunkArtifact (use ChunkInput type)
2. Use normalize_to_chunk_record() for internal conversion
3. Consider implementing get_chunk_artifact() for artifact-based retrieval
4. Preserve lineage metadata in _lineage_* fields
5. Test with both legacy ChunkRecord and new IFChunkArtifact inputs

Architecture Context
--------------------
Storage is accessed through the ChunkRepository interface:

    ┌─────────────────┐     ┌─────────────────┐
    │   Pipeline      │     │   Retriever     │
    │   (indexing)    │     │   (searching)   │
    └────────┬────────┘     └────────┬────────┘
             │                       │
             └───────────┬───────────┘
                         │
              ┌──────────┴──────────┐
              │   ChunkRepository   │
              │   (abstract base)   │
              └──────────┴──────────┘
                         │
         ┌───────────────┼───────────────┐
         ↓               ↓               ↓
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │  JSONL  │    │ ChromaDB│    │ Future  │
    │ Backend │    │ Backend │    │ Backend │
    └─────────┘    └─────────┘    └─────────┘

Key Data Structures
-------------------
**IFChunkArtifact** (Preferred)
    Modern artifact-based chunk representation with:
    - Full lineage tracking (parent_id, root_artifact_id, lineage_depth)
    - Provenance chain (list of processor IDs)
    - Content hashing for integrity verification
    - Pydantic validation for type safety

**ChunkRecord** (Deprecated - use IFChunkArtifact)
    Legacy chunk format. Will be removed in v2.0.
    Convert to IFChunkArtifact using:
        artifact = IFChunkArtifact.from_chunk_record(record)

**SearchResult**
    Returned by search operations. Contains:
    - chunk_id: Unique identifier
    - content: The text content
    - score: Relevance score (0-1)
    - document_id: Parent document
    - source_location: Citation information
    - metadata: Entities, concepts, etc.

    Can be converted to IFChunkArtifact using to_artifact() method.

**ParentChunkMapping**
    Links child chunks to parent chunks for context expansion.
    Used by ParentRetriever to fetch surrounding context.

Interface Contract
------------------
Core Methods (all accept ChunkInput = Union[ChunkRecord, IFChunkArtifact]):
- add_chunk(): Store single chunk (accepts ChunkInput)
- add_chunks(): Store multiple chunks (accepts List[ChunkInput])
- get_chunk(): Retrieve by ID (returns ChunkRecord for backward compatibility)
- get_chunk_artifact(): Retrieve by ID (returns IFChunkArtifact) [OPTIONAL]
- get_chunks_by_document(): Get all chunks for a document
- delete_chunk(): Remove a chunk
- delete_document(): Remove all chunks for a document
- search(): Keyword-based search
- search_semantic(): Vector similarity search (optional)
- count(): Get total chunk count
- clear(): Remove all data

Optional Enhancement Methods:
- get_chunk_artifact(): Return IFChunkArtifact instead of ChunkRecord
- get_artifacts_by_document(): Return List[IFChunkArtifact]

Implementing a Custom Backend
-----------------------------
    from ingestforge.storage.base import ChunkRepository, ChunkInput, normalize_to_chunk_record

    class PostgreSQLRepository(ChunkRepository):
        def add_chunk(self, chunk: ChunkInput) -> bool:
            # Convert to ChunkRecord for storage
            record = normalize_to_chunk_record(chunk)
            # Insert into PostgreSQL with pgvector
            ...

        def add_chunks(self, chunks: List[ChunkInput]) -> int:
            # Convert all to ChunkRecords
            records = [normalize_to_chunk_record(c) for c in chunks]
            # Batch insert
            ...

        def search_semantic(self, embedding, top_k) -> List[SearchResult]:
            # Use pgvector similarity search
            ...
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

from ingestforge.chunking.semantic_chunker import ChunkRecord

if TYPE_CHECKING:
    from ingestforge.core.pipeline.artifacts import IFChunkArtifact

# Type alias for storage input (ChunkRecord or IFChunkArtifact)
ChunkInput = Union[ChunkRecord, "IFChunkArtifact"]


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


# ===========================================================================
# Tag Sanitization Utilities (ORG-002)
# ===========================================================================

# Constants for tag validation
MAX_TAG_LENGTH = 32
MAX_TAGS_PER_CHUNK = 50
TAG_PATTERN = re.compile(r"^[a-z0-9]+$")

# Constant for artifact detection
MAX_LINEAGE_FIELDS = 5


def normalize_to_chunk_record(item: "ChunkInput") -> ChunkRecord:
    """
    Convert storage input to ChunkRecord.

    TASK-017: Core normalization function for backward compatibility.
    Used by all storage backends to accept both ChunkRecord and IFChunkArtifact.

    Rule #1: Early return for common case
    Rule #4: Function < 60 lines
    Rule #7: Explicit return type
    Rule #9: Complete type hints

    Args:
        item: Either a ChunkRecord or IFChunkArtifact

    Returns:
        ChunkRecord (passed through or converted)

    Raises:
        TypeError: If item is neither ChunkRecord nor IFChunkArtifact

    Example:
        def add_chunk(self, chunk: ChunkInput) -> bool:
            record = normalize_to_chunk_record(chunk)
            # Now work with ChunkRecord internally
            ...
    """
    # Fast path: already a ChunkRecord
    if isinstance(item, ChunkRecord):
        return item

    # Check if it's an IFChunkArtifact (duck typing to avoid import)
    if hasattr(item, "to_chunk_record") and callable(item.to_chunk_record):
        return item.to_chunk_record()

    # Fallback: log warning and try to construct
    _Logger.get().warning(f"Unknown chunk type: {type(item).__name__}")
    raise TypeError(
        f"Expected ChunkRecord or IFChunkArtifact, got {type(item).__name__}"
    )


def normalize_to_artifact(item: "ChunkInput") -> "IFChunkArtifact":
    """
    Convert storage input to IFChunkArtifact.

    TASK-017: Core normalization function for artifact-based processing.
    Complement to normalize_to_chunk_record() for new artifact-first workflows.

    Rule #1: Early return for common case
    Rule #4: Function < 60 lines
    Rule #7: Explicit return type
    Rule #9: Complete type hints

    Args:
        item: Either a ChunkRecord or IFChunkArtifact

    Returns:
        IFChunkArtifact (passed through or converted)

    Raises:
        TypeError: If item is neither ChunkRecord nor IFChunkArtifact

    Example:
        def get_chunk_artifact(self, chunk_id: str) -> Optional[IFChunkArtifact]:
            record = self.get_chunk(chunk_id)
            if record is None:
                return None
            return normalize_to_artifact(record)
    """
    from ingestforge.core.pipeline.artifacts import IFChunkArtifact

    # Fast path: already an IFChunkArtifact
    if isinstance(item, IFChunkArtifact):
        return item

    # Check if it's a ChunkRecord
    if isinstance(item, ChunkRecord):
        return IFChunkArtifact.from_chunk_record(item)

    # Check if it has from_chunk_record method (duck typing)
    if hasattr(item, "chunk_id") and hasattr(item, "content"):
        return IFChunkArtifact.from_chunk_record(item)

    # Fallback: log warning and raise
    _Logger.get().warning(f"Unknown chunk type: {type(item).__name__}")
    raise TypeError(
        f"Expected ChunkRecord or IFChunkArtifact, got {type(item).__name__}"
    )


def sanitize_tag(tag: str) -> str:
    """
    Sanitize a tag to conform to storage requirements.

    Rule #7: Input sanitization
    Rule #1: Early returns

    Tags are:
    - Converted to lowercase
    - Stripped of whitespace
    - Non-alphanumeric characters removed
    - Truncated to MAX_TAG_LENGTH (32) characters

    Args:
        tag: Raw tag string to sanitize

    Returns:
        Sanitized tag string

    Raises:
        ValueError: If tag is empty after sanitization
    """
    if not tag:
        raise ValueError("Tag cannot be empty")

    # Lowercase and strip
    clean = tag.lower().strip()

    # Remove non-alphanumeric characters
    clean = re.sub(r"[^a-z0-9]", "", clean)

    # Truncate to max length
    clean = clean[:MAX_TAG_LENGTH]

    if not clean:
        raise ValueError(f"Tag '{tag}' has no valid characters after sanitization")

    return clean


def validate_tag(tag: str) -> Tuple[bool, str]:
    """
    Validate a tag without modifying it.

    Rule #1: Early returns
    Rule #9: Type hints

    Args:
        tag: Tag to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not tag:
        return False, "Tag cannot be empty"

    if len(tag) > MAX_TAG_LENGTH:
        return False, f"Tag exceeds maximum length of {MAX_TAG_LENGTH}"

    if not TAG_PATTERN.match(tag):
        return False, "Tag must contain only lowercase letters and numbers"

    return True, ""


def parse_tags_json(metadata: Dict[str, Any]) -> List[str]:
    """Parse tags_json from metadata safely.

    Rule #1: Extracted to reduce nesting in chromadb search/crud operations.
    Rule #5: Returns empty list on parse failure (logged at call site).

    Args:
        metadata: Chunk metadata dictionary

    Returns:
        List of tags, or empty list if tags_json missing or invalid
    """
    import json

    if "tags_json" not in metadata:
        return []

    try:
        tags = json.loads(metadata["tags_json"])
        return tags if isinstance(tags, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def chunk_has_tag(metadata: Dict[str, Any], tag: str) -> bool:
    """Check if chunk metadata contains a specific tag.

    Rule #1: Extracted helper to reduce nesting in search operations.

    Args:
        metadata: Chunk metadata dictionary
        tag: Sanitized tag to check for

    Returns:
        True if tag is present in chunk's tags
    """
    tags = parse_tags_json(metadata)
    return tag in tags


@dataclass(slots=True)
class SearchResult:
    """Result from a search operation.

    Results Visualization data model.
    - AC: Entity badges (via metadata)
    - AC: Citation display (via source_location)
    - AC: Relevance score (via score)

    Uses slots=True to reduce memory overhead for large result sets.
    """

    chunk_id: str
    content: str
    score: float
    document_id: str
    section_title: str
    chunk_type: str
    source_file: str
    word_count: int

    # Additional metadata
    metadata: Dict[str, Any] = field(default=None)

    # Source location for citations
    source_location: Optional[Dict[str, Any]] = field(default=None)
    page_start: Optional[int] = field(default=None)
    page_end: Optional[int] = field(default=None)

    # Library/collection name
    library: str = field(default="default")

    # Author/Contributor identity (TICKET-301)
    author_id: Optional[str] = field(default=None)
    author_name: Optional[str] = field(default=None)

    def validate(self) -> bool:
        """
        Validate that all required fields are correctly populated.

        Returns:
            True if valid, raises ValueError otherwise.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        if not self.chunk_id:
            raise ValueError("chunk_id is required and cannot be empty")
        if not self.content:
            raise ValueError("content is required and cannot be empty")
        if not isinstance(self.score, (int, float)):
            raise ValueError(f"score must be numeric, got {type(self.score).__name__}")
        if self.score < 0 or self.score > 1:
            raise ValueError(f"score must be between 0 and 1, got {self.score}")
        if not self.document_id:
            raise ValueError("document_id is required and cannot be empty")
        if not isinstance(self.word_count, int) or self.word_count < 0:
            raise ValueError(
                f"word_count must be a non-negative integer, got {self.word_count}"
            )
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "score": self.score,
            "document_id": self.document_id,
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "source_file": self.source_file,
            "word_count": self.word_count,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "library": self.library,
            "metadata": self.metadata or {},
            # Author identity (TICKET-301)
            "author_id": self.author_id,
            "author_name": self.author_name,
        }
        if self.source_location:
            result["source_location"] = self.source_location
        return result

    def to_dict_with_annotations(
        self,
        annotation_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Convert to dictionary with annotations included (ORG-004).

        Rule #4: Separate rendering from retrieval

        Args:
            annotation_manager: AnnotationManager instance for fetching annotations

        Returns:
            Dictionary with base result plus annotations list
        """
        result = self.to_dict()

        if annotation_manager is not None:
            try:
                annotations = annotation_manager.get_for_chunk(self.chunk_id)
                result["annotations"] = [
                    {
                        "annotation_id": ann.annotation_id,
                        "text": ann.text,
                        "created_at": ann.created_at,
                    }
                    for ann in annotations
                ]
            except Exception as e:
                _Logger.get().warning(
                    f"Failed to load annotations for chunk {self.chunk_id}: {e}"
                )
                result["annotations"] = []
        else:
            result["annotations"] = []

        return result

    @classmethod
    def from_chunk(
        cls,
        chunk: ChunkRecord,
        score: float = 0.0,
    ) -> "SearchResult":
        """Create SearchResult from ChunkRecord."""
        # Get source_location dict if present
        source_loc = None
        if chunk.source_location:
            source_loc = chunk.source_location.to_dict()

        return cls(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            score=score,
            document_id=chunk.document_id,
            section_title=chunk.section_title,
            chunk_type=chunk.chunk_type,
            source_file=chunk.source_file,
            word_count=chunk.word_count,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            source_location=source_loc,
            library=chunk.library,
            # Author identity (TICKET-301)
            author_id=getattr(chunk, "author_id", None),
            author_name=getattr(chunk, "author_name", None),
            metadata={
                "embedding": chunk.embedding is not None,
                "entities": chunk.entities,
                "concepts": chunk.concepts,
                "quality_score": chunk.quality_score,
            },
        )

    def to_artifact(self) -> "IFChunkArtifact":
        """
        Convert SearchResult to IFChunkArtifact.

        h: Enables retrieval layer to return artifacts.
        Rule #4: Function < 60 lines
        Rule #7: Explicit return type
        Rule #9: Complete type hints

        Returns:
            IFChunkArtifact with all SearchResult data preserved.
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        # Build metadata from SearchResult fields
        artifact_metadata: Dict[str, Any] = {
            "section_title": self.section_title,
            "chunk_type": self.chunk_type,
            "source_file": self.source_file,
            "word_count": self.word_count,
            "library": self.library,
            "retrieval_score": self.score,  # Preserve retrieval score
        }

        # Add optional fields
        if self.page_start is not None:
            artifact_metadata["page_start"] = self.page_start
        if self.page_end is not None:
            artifact_metadata["page_end"] = self.page_end
        if self.source_location:
            artifact_metadata["source_location"] = self.source_location
        if self.author_id:
            artifact_metadata["author_id"] = self.author_id
        if self.author_name:
            artifact_metadata["author_name"] = self.author_name

        # Merge existing metadata (may contain entities, concepts, etc.)
        if self.metadata:
            for key, value in self.metadata.items():
                if key not in artifact_metadata:
                    artifact_metadata[key] = value

        # Extract lineage info from metadata if present (from g storage)
        parent_id = artifact_metadata.pop("_lineage_parent_id", None)
        root_artifact_id = artifact_metadata.pop("_lineage_root_id", None)
        lineage_depth = artifact_metadata.pop("_lineage_depth", 0)
        provenance = artifact_metadata.pop("_lineage_provenance", [])

        return IFChunkArtifact(
            artifact_id=self.chunk_id,
            document_id=self.document_id,
            content=self.content,
            metadata=artifact_metadata,
            parent_id=parent_id,
            root_artifact_id=root_artifact_id,
            lineage_depth=lineage_depth,
            provenance=provenance if provenance else ["retrieval"],
        )


class ChunkRepository(ABC):
    """
    Abstract base class for chunk storage backends.

    TASK-017: ChunkRecord → IFChunkArtifact Migration
    ==================================================

    Current Migration Status
    ------------------------
    - Phase: Active migration (backward compatible)
    - ChunkRecord: DEPRECATED, will be removed in v2.0
    - IFChunkArtifact: PREFERRED for all new code
    - All methods accept both types via ChunkInput type alias

    Migration Timeline
    ------------------
    - v1.x: Both ChunkRecord and IFChunkArtifact supported
    - v2.0: ChunkRecord removed, IFChunkArtifact only

    For Backend Implementers
    -------------------------
    **Input Methods** (accept chunks):
        - add_chunk(chunk: ChunkInput) -> bool
        - add_chunks(chunks: List[ChunkInput]) -> int

        Use normalize_to_chunk_record() to convert inputs:
            record = normalize_to_chunk_record(chunk)

    **Output Methods** (return chunks):
        Legacy (returns ChunkRecord):
            - get_chunk() - DEPRECATED
            - get_chunks_by_document() - DEPRECATED
            - get_chunks_by_tag() - DEPRECATED
            - get_unread_chunks() - DEPRECATED
            - get_parent_chunk() - DEPRECATED
            - get_child_chunks() - DEPRECATED

        Modern (returns IFChunkArtifact):
            - get_chunk_artifact() - PREFERRED
            - get_artifacts_by_document() - PREFERRED
            - get_artifacts_by_tag() - PREFERRED
            - get_unread_artifacts() - PREFERRED
            - get_parent_artifact() - PREFERRED
            - get_child_artifacts() - PREFERRED

    **Default Implementations**:
        All artifact-returning methods have default implementations that
        convert ChunkRecords to IFChunkArtifacts. You can override these
        for more efficient artifact-native storage.

    Lineage Preservation
    --------------------
    When storing IFChunkArtifacts, preserve lineage metadata:
        - _lineage_parent_id: artifact.parent_id
        - _lineage_root_id: artifact.root_artifact_id
        - _lineage_depth: artifact.lineage_depth
        - _lineage_provenance: artifact.provenance
        - _content_hash: artifact.content_hash

    These fields are automatically handled by to_chunk_record().

    Example Implementation
    ----------------------
        class MyBackend(ChunkRepository):
            def add_chunk(self, chunk: ChunkInput) -> bool:
                # Accept both types, convert to ChunkRecord
                record = normalize_to_chunk_record(chunk)
                # Store using your backend
                self._store(record)
                return True

            def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
                # Return ChunkRecord for backward compatibility
                return self._fetch(chunk_id)

            # Optional: override for efficient artifact storage
            def get_chunk_artifact(self, chunk_id: str) -> Optional[IFChunkArtifact]:
                # Return artifact directly if stored natively
                return self._fetch_artifact(chunk_id)

    JPL Compliance
    --------------
    - Rule #4: All methods < 60 lines
    - Rule #7: All return values checked
    - Rule #9: 100% type hints (ChunkInput, IFChunkArtifact)
    - Rule #10: Content hash verification via _content_hash field

    See Also
    --------
    - normalize_to_chunk_record(): Convert ChunkInput → ChunkRecord
    - normalize_to_artifact(): Convert ChunkInput → IFChunkArtifact
    - IFChunkArtifact.from_chunk_record(): Convert ChunkRecord → IFChunkArtifact
    - IFChunkArtifact.to_chunk_record(): Convert IFChunkArtifact → ChunkRecord
    """

    @abstractmethod
    def add_chunk(self, chunk: "ChunkInput") -> bool:
        """
        Add a single chunk to storage.

        g: Accepts both ChunkRecord and IFChunkArtifact.

        Args:
            chunk: Chunk to add (ChunkRecord or IFChunkArtifact)

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: List["ChunkInput"]) -> int:
        """
        Add multiple chunks to storage.

        g: Accepts lists containing ChunkRecord or IFChunkArtifact.

        Args:
            chunks: Chunks to add (list of ChunkRecord or IFChunkArtifact)

        Returns:
            Number of chunks successfully added
        """
        pass

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        """
        Get a chunk by ID (returns legacy ChunkRecord).

        DEPRECATED: Use get_chunk_artifact() for new code.
        This method returns ChunkRecord for backward compatibility.
        Will be removed in v2.0.

        Args:
            chunk_id: Chunk identifier

        Returns:
            ChunkRecord or None if not found

        See Also:
            get_chunk_artifact(): Returns IFChunkArtifact (preferred)
        """
        pass

    def get_chunk_artifact(self, chunk_id: str) -> Optional["IFChunkArtifact"]:
        """
        Get a chunk by ID (returns IFChunkArtifact).

        TASK-017: New artifact-based retrieval method.
        Preferred over get_chunk() for new code.

        Default implementation converts ChunkRecord to IFChunkArtifact.
        Backends may override for more efficient artifact-native storage.

        Args:
            chunk_id: Chunk identifier

        Returns:
            IFChunkArtifact or None if not found

        Example:
            artifact = repo.get_chunk_artifact("chunk-123")
            if artifact:
                print(f"Lineage depth: {artifact.lineage_depth}")
                print(f"Provenance: {artifact.provenance}")
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        record = self.get_chunk(chunk_id)
        if record is None:
            return None

        return IFChunkArtifact.from_chunk_record(record)

    @abstractmethod
    def verify_chunk_exists(self, chunk_id: str) -> bool:
        """
        Check if a chunk ID exists in storage (Fast check).

        Args:
            chunk_id: Chunk identifier

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]:
        """
        Get all chunks for a document (returns legacy ChunkRecords).

        DEPRECATED: Use get_artifacts_by_document() for new code.
        Returns ChunkRecords for backward compatibility.
        Will be removed in v2.0.

        Args:
            document_id: Document identifier

        Returns:
            List of ChunkRecords

        See Also:
            get_artifacts_by_document(): Returns List[IFChunkArtifact] (preferred)
        """
        pass

    def get_artifacts_by_document(self, document_id: str) -> List["IFChunkArtifact"]:
        """
        Get all chunks for a document (returns IFChunkArtifacts).

        TASK-017: New artifact-based retrieval method.
        Preferred over get_chunks_by_document() for new code.

        Default implementation converts ChunkRecords to IFChunkArtifacts.
        Backends may override for more efficient artifact-native storage.

        Args:
            document_id: Document identifier

        Returns:
            List of IFChunkArtifacts

        Example:
            artifacts = repo.get_artifacts_by_document("doc-123")
            for artifact in artifacts:
                print(f"Chunk {artifact.artifact_id}: depth={artifact.lineage_depth}")
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        records = self.get_chunks_by_document(document_id)
        return [IFChunkArtifact.from_chunk_record(r) for r in records]

    @abstractmethod
    def delete_chunk(self, chunk_id: str) -> bool:
        """
        Delete a chunk.

        Args:
            chunk_id: Chunk to delete

        Returns:
            True if deleted
        """
        pass

    @abstractmethod
    def delete_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.

        Args:
            document_id: Document to delete

        Returns:
            Number of chunks deleted
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 10,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.

        Args:
            query: Search query
            top_k: Number of results
            library_filter: If provided, only return chunks from this library

        Returns:
            List of SearchResult
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """
        Get total number of chunks.

        Returns:
            Chunk count
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all stored chunks."""
        pass

    @staticmethod
    def validate_search_params(
        query: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
    ) -> None:
        """
        Validate common search parameters.

        Args:
            query: Text query (for keyword search)
            query_embedding: Embedding vector (for semantic search)
            top_k: Number of results to return

        Raises:
            ValueError: If parameters are invalid
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > 1000:
            raise ValueError(f"top_k must be <= 1000, got {top_k}")
        if query is not None and not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if query_embedding is not None:
            if not isinstance(query_embedding, list):
                raise ValueError(
                    f"query_embedding must be a list, got {type(query_embedding).__name__}"
                )
            if len(query_embedding) == 0:
                raise ValueError("query_embedding cannot be empty")

    def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using embedding vector.

        Optional - not all backends support semantic search.
        Backends that support vector search (e.g., ChromaDB) should override this.
        Backends that only support keyword search (e.g., JSONL) may leave this
        as-is to raise NotImplementedError.

        Args:
            query_embedding: Query embedding vector
            top_k: Number of results
            library_filter: If provided, only return chunks from this library

        Returns:
            List of SearchResult
        """
        raise NotImplementedError("Semantic search not supported by this backend")

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        return {
            "total_chunks": self.count(),
        }

    def get_libraries(self) -> List[str]:
        """
        Get list of unique library names in storage.

        Returns:
            List of library names. Always includes "default".
        """
        return ["default"]

    def count_by_library(self, library_name: str) -> int:
        """
        Count chunks in a specific library.

        Args:
            library_name: Library to count

        Returns:
            Number of chunks in the library
        """
        return 0

    def delete_by_library(self, library_name: str) -> int:
        """
        Delete all chunks in a library.

        Args:
            library_name: Library to delete

        Returns:
            Number of chunks deleted
        """
        raise NotImplementedError("Library deletion not supported by this backend")

    def reassign_library(self, old_library: str, new_library: str) -> int:
        """
        Move all chunks from one library to another.

        Args:
            old_library: Source library name
            new_library: Target library name

        Returns:
            Number of chunks moved
        """
        raise NotImplementedError("Library reassignment not supported by this backend")

    def move_document_to_library(self, document_id: str, new_library: str) -> int:
        """
        Move all chunks of a specific document to a different library.

        Args:
            document_id: Document to move
            new_library: Target library name

        Returns:
            Number of chunks updated
        """
        raise NotImplementedError("Document library move not supported by this backend")

    # Read/Unread tracking methods (ORG-001)
    def mark_read(self, chunk_id: str, status: bool = True) -> bool:
        """
        Mark a chunk as read or unread.

        Args:
            chunk_id: Unique identifier of the chunk to update
            status: True to mark as read, False to mark as unread

        Returns:
            True if successful, False if chunk not found or update failed

        Raises:
            ValueError: If chunk_id is empty or None
        """
        raise NotImplementedError("Read/unread tracking not supported by this backend")

    def get_unread_chunks(
        self,
        library_filter: Optional[str] = None,
    ) -> List[ChunkRecord]:
        """
        Get all chunks marked as unread (returns legacy ChunkRecords).

        DEPRECATED: Use get_unread_artifacts() for new code.
        Returns ChunkRecords for backward compatibility.

        Args:
            library_filter: If provided, only return chunks from this library

        Returns:
            List of unread ChunkRecords

        See Also:
            get_unread_artifacts(): Returns List[IFChunkArtifact] (preferred)
        """
        raise NotImplementedError("Read/unread tracking not supported by this backend")

    def get_unread_artifacts(
        self,
        library_filter: Optional[str] = None,
    ) -> List["IFChunkArtifact"]:
        """
        Get all chunks marked as unread (returns IFChunkArtifacts).

        TASK-017: New artifact-based retrieval method.
        Preferred over get_unread_chunks() for new code.

        Default implementation converts ChunkRecords to IFChunkArtifacts.
        Backends may override for more efficient artifact-native storage.

        Args:
            library_filter: If provided, only return chunks from this library

        Returns:
            List of unread IFChunkArtifacts

        Example:
            unread = repo.get_unread_artifacts()
            print(f"You have {len(unread)} unread chunks")
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        records = self.get_unread_chunks(library_filter)
        return [IFChunkArtifact.from_chunk_record(r) for r in records]

    # Tagging methods (ORG-002)
    def add_tag(self, chunk_id: str, tag: str) -> bool:
        """
        Add a tag to a chunk.

        Rule #7: Input sanitization - tag is sanitized before storage
        Rule #1: Early returns

        Args:
            chunk_id: Unique identifier of the chunk
            tag: Tag to add (will be sanitized: lowercase, alphanumeric, max 32 chars)

        Returns:
            True if tag was added, False if chunk not found or tag already exists

        Raises:
            ValueError: If chunk_id or tag is empty, or if chunk already has 50 tags
        """
        raise NotImplementedError("Tagging not supported by this backend")

    def remove_tag(self, chunk_id: str, tag: str) -> bool:
        """
        Remove a tag from a chunk.

        Args:
            chunk_id: Unique identifier of the chunk
            tag: Tag to remove (will be sanitized before lookup)

        Returns:
            True if tag was removed, False if chunk not found or tag doesn't exist

        Raises:
            ValueError: If chunk_id or tag is empty
        """
        raise NotImplementedError("Tagging not supported by this backend")

    def get_chunks_by_tag(
        self,
        tag: str,
        library_filter: Optional[str] = None,
    ) -> List[ChunkRecord]:
        """
        Get all chunks with a specific tag (returns legacy ChunkRecords).

        DEPRECATED: Use get_artifacts_by_tag() for new code.
        Returns ChunkRecords for backward compatibility.

        Args:
            tag: Tag to filter by (will be sanitized)
            library_filter: If provided, only return chunks from this library

        Returns:
            List of ChunkRecords with the specified tag (empty list if none found)

        See Also:
            get_artifacts_by_tag(): Returns List[IFChunkArtifact] (preferred)
        """
        raise NotImplementedError("Tagging not supported by this backend")

    def get_artifacts_by_tag(
        self,
        tag: str,
        library_filter: Optional[str] = None,
    ) -> List["IFChunkArtifact"]:
        """
        Get all chunks with a specific tag (returns IFChunkArtifacts).

        TASK-017: New artifact-based retrieval method.
        Preferred over get_chunks_by_tag() for new code.

        Default implementation converts ChunkRecords to IFChunkArtifacts.
        Backends may override for more efficient artifact-native storage.

        Args:
            tag: Tag to filter by (will be sanitized)
            library_filter: If provided, only return chunks from this library

        Returns:
            List of IFChunkArtifacts with the specified tag

        Example:
            artifacts = repo.get_artifacts_by_tag("important")
            for artifact in artifacts:
                print(f"{artifact.artifact_id}: {artifact.content[:50]}")
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        records = self.get_chunks_by_tag(tag, library_filter)
        return [IFChunkArtifact.from_chunk_record(r) for r in records]

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags in the storage.

        Returns:
            Sorted list of unique tags
        """
        raise NotImplementedError("Tagging not supported by this backend")

    # Parent-child chunk mapping methods
    def set_parent_mapping(
        self,
        child_chunk_id: str,
        parent_chunk_id: str,
    ) -> bool:
        """
        Map a child chunk to its parent chunk.

        Args:
            child_chunk_id: ID of the smaller/child chunk
            parent_chunk_id: ID of the larger/parent chunk

        Returns:
            True if mapping was set successfully
        """
        raise NotImplementedError("Parent mapping not supported by this backend")

    def get_parent_chunk(self, child_chunk_id: str) -> Optional[ChunkRecord]:
        """
        Get the parent chunk for a given child chunk (returns legacy ChunkRecord).

        DEPRECATED: Use get_parent_artifact() for new code.
        Returns ChunkRecord for backward compatibility.

        Args:
            child_chunk_id: ID of the child chunk

        Returns:
            Parent ChunkRecord or None if no parent exists

        See Also:
            get_parent_artifact(): Returns IFChunkArtifact (preferred)
        """
        raise NotImplementedError("Parent mapping not supported by this backend")

    def get_parent_artifact(self, child_chunk_id: str) -> Optional["IFChunkArtifact"]:
        """
        Get the parent chunk for a given child chunk (returns IFChunkArtifact).

        TASK-017: New artifact-based retrieval method.
        Preferred over get_parent_chunk() for new code.

        Default implementation converts ChunkRecord to IFChunkArtifact.
        Backends may override for more efficient artifact-native storage.

        Args:
            child_chunk_id: ID of the child chunk

        Returns:
            Parent IFChunkArtifact or None if no parent exists

        Example:
            parent = repo.get_parent_artifact("chunk-123")
            if parent:
                print(f"Parent context: {parent.content}")
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        record = self.get_parent_chunk(child_chunk_id)
        if record is None:
            return None

        return IFChunkArtifact.from_chunk_record(record)

    def get_child_chunks(self, parent_chunk_id: str) -> List[ChunkRecord]:
        """
        Get all child chunks for a given parent chunk (returns legacy ChunkRecords).

        DEPRECATED: Use get_child_artifacts() for new code.
        Returns ChunkRecords for backward compatibility.

        Args:
            parent_chunk_id: ID of the parent chunk

        Returns:
            List of child ChunkRecords

        See Also:
            get_child_artifacts(): Returns List[IFChunkArtifact] (preferred)
        """
        raise NotImplementedError("Parent mapping not supported by this backend")

    def get_child_artifacts(self, parent_chunk_id: str) -> List["IFChunkArtifact"]:
        """
        Get all child chunks for a given parent chunk (returns IFChunkArtifacts).

        TASK-017: New artifact-based retrieval method.
        Preferred over get_child_chunks() for new code.

        Default implementation converts ChunkRecords to IFChunkArtifacts.
        Backends may override for more efficient artifact-native storage.

        Args:
            parent_chunk_id: ID of the parent chunk

        Returns:
            List of child IFChunkArtifacts

        Example:
            children = repo.get_child_artifacts("parent-123")
            print(f"Found {len(children)} child chunks")
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        records = self.get_child_chunks(parent_chunk_id)
        return [IFChunkArtifact.from_chunk_record(r) for r in records]

    def expand_to_parent(
        self,
        results: List["SearchResult"],
        deduplicate: bool = True,
    ) -> List["SearchResult"]:
        """
        Expand search results to include parent chunk content.

        For each result, retrieves the parent chunk (if any) and returns
        the parent content instead of the child content for context expansion.

        Args:
            results: Original search results (child chunks)
            deduplicate: If True, remove duplicate parents

        Returns:
            List of SearchResults with parent content where available
        """
        raise NotImplementedError("Parent expansion not supported by this backend")


@dataclass
class ParentChunkMapping:
    """Stores the mapping between child and parent chunks."""

    child_chunk_id: str
    parent_chunk_id: str
    document_id: str
    child_position: int  # Position of child within parent (0-indexed)
    total_children: int  # Total number of children for this parent

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "child_chunk_id": self.child_chunk_id,
            "parent_chunk_id": self.parent_chunk_id,
            "document_id": self.document_id,
            "child_position": self.child_position,
            "total_children": self.total_children,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParentChunkMapping":
        """Create from dictionary."""
        return cls(
            child_chunk_id=data["child_chunk_id"],
            parent_chunk_id=data["parent_chunk_id"],
            document_id=data["document_id"],
            child_position=data.get("child_position", 0),
            total_children=data.get("total_children", 1),
        )
