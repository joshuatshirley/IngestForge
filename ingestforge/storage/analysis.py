"""Analysis Storage - Store and retrieve analysis results.

Provides storage for analysis results (themes, arguments, comprehension, etc.)
that links back to source documents and makes analyses searchable.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Key Components
--------------
**AnalysisRecord**
    Dataclass representing a stored analysis result:
    - analysis_id: Unique identifier
    - analysis_type: Type (theme, argument, comprehension, etc.)
    - content: The analysis text (searchable)
    - source_document: Original document path
    - source_chunks: IDs of analyzed chunks
    - confidence: Analysis confidence score
    - metadata: Structured data (themes list, etc.)

**AnalysisStorage**
    Storage class for analysis results:
    - store_analysis(): Store and return ID
    - get_analysis(): Retrieve by ID
    - search_analyses(): Semantic search
    - list_analyses(): List all
    - delete_analysis(): Remove analysis
    - get_analyses_for_document(): Get all for a doc"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class _Logger:
    """Lazy logger holder."""

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


# =============================================================================
# Analysis Types
# =============================================================================

ANALYSIS_TYPES = [
    "theme",  # Theme extraction results
    "argument",  # Argument analysis results
    "comprehension",  # Comprehension explanations
    "literary",  # Literary analysis (characters, symbols, etc.)
    "code",  # Code analysis results
    "summary",  # Document summaries
    "character",  # Character analysis
    "symbol",  # Symbol analysis
    "arc",  # Story arc analysis
]

# =============================================================================
# AnalysisRecord Dataclass
# =============================================================================


@dataclass
class AnalysisRecord:
    """Stored analysis result linking back to source content.

    Attributes:
        analysis_id: Unique ID for this analysis
        analysis_type: Type (theme, argument, comprehension, etc.)
        content: The analysis text (searchable)
        source_document: Original document path
        source_chunks: IDs of chunks that were analyzed
        confidence: Analysis confidence score
        metadata: Additional structured data (themes list, etc.)
        created_at: ISO timestamp
        library: Library for organization
        title: Optional title for the analysis
    """

    analysis_id: str
    analysis_type: str
    content: str
    source_document: str
    source_chunks: List[str] = field(default_factory=list)
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    library: str = "default"
    title: str = ""

    def __post_init__(self) -> None:
        """Set default timestamp if not provided."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "analysis_id": self.analysis_id,
            "analysis_type": self.analysis_type,
            "content": self.content,
            "source_document": self.source_document,
            "source_chunks": self.source_chunks,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "library": self.library,
            "title": self.title,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisRecord":
        """Create from dictionary."""
        return cls(
            analysis_id=data.get("analysis_id", ""),
            analysis_type=data.get("analysis_type", ""),
            content=data.get("content", ""),
            source_document=data.get("source_document", ""),
            source_chunks=data.get("source_chunks", []),
            confidence=data.get("confidence", 0.8),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", ""),
            library=data.get("library", "default"),
            title=data.get("title", ""),
        )

    @staticmethod
    def generate_id() -> str:
        """Generate a unique analysis ID."""
        return f"analysis_{uuid.uuid4().hex[:12]}"


# =============================================================================
# AnalysisStorage Class
# =============================================================================


class AnalysisStorage:
    """Storage for analysis results using ChromaDB.

    Stores analyses as searchable chunks with special chunk_type prefixes.
    """

    COLLECTION_NAME = "ingestforge_analyses"

    def __init__(
        self,
        persist_directory: Path,
        embedding_function: Any = None,
    ) -> None:
        """Initialize analysis storage.

        Args:
            persist_directory: Path for persistent storage
            embedding_function: Optional custom embedding function
        """
        self.persist_directory = persist_directory
        self._embedding_function = embedding_function
        self._client = None
        self._collection = None

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Any:
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb

                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required for analysis storage. "
                    "Install with: pip install chromadb"
                )
        return self._client

    def _create_embedding_function(self) -> Any:
        """Create embedding function using sentence-transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            from chromadb import EmbeddingFunction, Documents, Embeddings

            model = SentenceTransformer("all-MiniLM-L6-v2")

            class CustomEmbeddingFunction(EmbeddingFunction):
                def __init__(self, st_model: SentenceTransformer):
                    self._model = st_model

                def __call__(self, input: Documents) -> Embeddings:
                    embeddings = self._model.encode(list(input), convert_to_numpy=True)
                    return embeddings.tolist()

            return CustomEmbeddingFunction(model)
        except Exception as e:
            _Logger.get().warning(f"Could not create embedding function: {e}")
            return None

    @property
    def collection(self) -> Any:
        """Get or create collection."""
        if self._collection is None:
            if self._embedding_function is None:
                self._embedding_function = self._create_embedding_function()

            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def store_analysis(self, record: AnalysisRecord) -> str:
        """Store an analysis record.

        Args:
            record: AnalysisRecord to store

        Returns:
            analysis_id of stored record
        """
        try:
            # Prepare metadata for ChromaDB (only simple types)
            metadata = {
                "analysis_type": record.analysis_type,
                "source_document": record.source_document,
                "source_chunks_json": json.dumps(record.source_chunks),
                "confidence": record.confidence,
                "metadata_json": json.dumps(record.metadata),
                "created_at": record.created_at,
                "library": record.library,
                "title": record.title,
                "chunk_type": f"analysis:{record.analysis_type}",
            }

            self.collection.add(
                ids=[record.analysis_id],
                documents=[record.content],
                metadatas=[metadata],
            )

            _Logger.get().info(f"Stored analysis: {record.analysis_id}")
            return record.analysis_id

        except Exception as e:
            _Logger.get().error(f"Failed to store analysis: {e}")
            raise

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisRecord]:
        """Retrieve analysis by ID.

        Args:
            analysis_id: Analysis identifier

        Returns:
            AnalysisRecord or None if not found
        """
        try:
            result = self.collection.get(
                ids=[analysis_id],
                include=["documents", "metadatas"],
            )

            if not result["ids"]:
                return None

            return self._result_to_record(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
            )

        except Exception as e:
            _Logger.get().error(f"Failed to get analysis: {e}")
            return None

    def _result_to_record(
        self,
        analysis_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> AnalysisRecord:
        """Convert ChromaDB result to AnalysisRecord."""
        source_chunks = []
        if "source_chunks_json" in metadata:
            try:
                source_chunks = json.loads(metadata["source_chunks_json"])
            except json.JSONDecodeError as e:
                _Logger.get().debug(f"Failed to parse source_chunks_json: {e}")

        analysis_metadata = {}
        if "metadata_json" in metadata:
            try:
                analysis_metadata = json.loads(metadata["metadata_json"])
            except json.JSONDecodeError as e:
                _Logger.get().debug(f"Failed to parse metadata_json: {e}")

        return AnalysisRecord(
            analysis_id=analysis_id,
            analysis_type=metadata.get("analysis_type", ""),
            content=content,
            source_document=metadata.get("source_document", ""),
            source_chunks=source_chunks,
            confidence=metadata.get("confidence", 0.8),
            metadata=analysis_metadata,
            created_at=metadata.get("created_at", ""),
            library=metadata.get("library", "default"),
            title=metadata.get("title", ""),
        )

    def search_analyses(
        self,
        query: str,
        k: int = 10,
        analysis_type: Optional[str] = None,
        library: Optional[str] = None,
    ) -> List[AnalysisRecord]:
        """Semantic search across analyses.

        Args:
            query: Search query
            k: Number of results
            analysis_type: Filter by analysis type
            library: Filter by library

        Returns:
            List of matching AnalysisRecords
        """
        try:
            where_filter = None

            # Build filter conditions
            conditions = []
            if analysis_type:
                conditions.append({"analysis_type": analysis_type})
            if library:
                conditions.append({"library": library})

            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

            result = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_filter,
                include=["documents", "metadatas"],
            )

            records = []
            if result["ids"] and result["ids"][0]:
                for i, analysis_id in enumerate(result["ids"][0]):
                    record = self._result_to_record(
                        analysis_id,
                        result["documents"][0][i],
                        result["metadatas"][0][i],
                    )
                    records.append(record)

            return records

        except Exception as e:
            _Logger.get().error(f"Failed to search analyses: {e}")
            return []

    def list_analyses(
        self,
        analysis_type: Optional[str] = None,
        library: Optional[str] = None,
        limit: int = 100,
    ) -> List[AnalysisRecord]:
        """List all analyses with optional filters.

        Args:
            analysis_type: Filter by type
            library: Filter by library
            limit: Maximum results

        Returns:
            List of AnalysisRecords
        """
        try:
            where_filter = None

            conditions = []
            if analysis_type:
                conditions.append({"analysis_type": analysis_type})
            if library:
                conditions.append({"library": library})

            if len(conditions) == 1:
                where_filter = conditions[0]
            elif len(conditions) > 1:
                where_filter = {"$and": conditions}

            result = self.collection.get(
                where=where_filter,
                limit=limit,
                include=["documents", "metadatas"],
            )

            records = []
            for i, analysis_id in enumerate(result["ids"]):
                record = self._result_to_record(
                    analysis_id,
                    result["documents"][i],
                    result["metadatas"][i],
                )
                records.append(record)

            # Sort by created_at descending
            records.sort(key=lambda r: r.created_at, reverse=True)
            return records

        except Exception as e:
            _Logger.get().error(f"Failed to list analyses: {e}")
            return []

    def delete_analysis(self, analysis_id: str) -> bool:
        """Delete an analysis.

        Args:
            analysis_id: Analysis to delete

        Returns:
            True if deleted
        """
        try:
            self.collection.delete(ids=[analysis_id])
            _Logger.get().info(f"Deleted analysis: {analysis_id}")
            return True
        except Exception as e:
            _Logger.get().error(f"Failed to delete analysis: {e}")
            return False

    def get_analyses_for_document(
        self,
        doc_path: str,
    ) -> List[AnalysisRecord]:
        """Get all analyses for a document.

        Args:
            doc_path: Document path

        Returns:
            List of analyses for the document
        """
        try:
            result = self.collection.get(
                where={"source_document": doc_path},
                include=["documents", "metadatas"],
            )

            records = []
            for i, analysis_id in enumerate(result["ids"]):
                record = self._result_to_record(
                    analysis_id,
                    result["documents"][i],
                    result["metadatas"][i],
                )
                records.append(record)

            return records

        except Exception as e:
            _Logger.get().error(f"Failed to get analyses for document: {e}")
            return []

    def refresh_analysis(
        self,
        analysis_id: str,
        llm_client: Any,
    ) -> Optional[AnalysisRecord]:
        """Re-run an analysis with the current LLM.

        Args:
            analysis_id: Analysis to refresh
            llm_client: LLM client for regeneration

        Returns:
            Updated AnalysisRecord or None on failure
        """
        # Get existing analysis
        existing = self.get_analysis(analysis_id)
        if not existing:
            _Logger.get().error(f"Analysis not found: {analysis_id}")
            return None

        # This requires access to the original chunks and analysis logic
        # For now, we return the existing record with a note
        _Logger.get().warning(
            "Full refresh requires re-running the original analysis command. "
            "Consider using the appropriate analysis command with --store flag."
        )
        return existing

    def count(self) -> int:
        """Get total number of analyses."""
        try:
            return self.collection.count()
        except Exception as e:
            _Logger.get().error(f"Failed to count analyses: {e}")
            return 0

    def count_by_type(self, analysis_type: str) -> int:
        """Count analyses by type."""
        try:
            result = self.collection.get(
                where={"analysis_type": analysis_type},
                include=[],
            )
            return len(result["ids"])
        except Exception as e:
            _Logger.get().error(
                f"Failed to count analyses by type '{analysis_type}': {e}"
            )
            return 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        stats = {
            "total_analyses": self.count(),
            "by_type": {},
        }

        for atype in ANALYSIS_TYPES:
            count = self.count_by_type(atype)
            if count > 0:
                stats["by_type"][atype] = count

        return stats


# =============================================================================
# Factory Function
# =============================================================================


def get_analysis_storage(
    persist_directory: Optional[Path] = None,
) -> AnalysisStorage:
    """Get analysis storage instance.

    Args:
        persist_directory: Storage path (defaults to .data/chromadb)

    Returns:
        AnalysisStorage instance
    """
    if persist_directory is None:
        persist_directory = Path.cwd() / ".data" / "chromadb"

    return AnalysisStorage(persist_directory=persist_directory)
