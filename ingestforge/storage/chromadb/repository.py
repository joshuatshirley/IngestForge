"""
Main ChromaDB Repository class.

Provides semantic search using ChromaDB vector database.
"""

from pathlib import Path
from typing import Any

from ingestforge.storage.base import ChunkRepository
from ingestforge.storage.chromadb.crud import ChromaDBCRUDMixin
from ingestforge.storage.chromadb.search import ChromaDBSearchMixin
from ingestforge.storage.chromadb.management import ChromaDBManagementMixin
from ingestforge.storage.chromadb.lifecycle import ChromaDBLifecycleMixin


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


class ChromaDBRepository(
    ChromaDBCRUDMixin,
    ChromaDBSearchMixin,
    ChromaDBManagementMixin,
    ChromaDBLifecycleMixin,
    ChunkRepository,
):
    """
    ChromaDB-based chunk storage with vector search.

    Stores embeddings in ChromaDB for efficient semantic search.

    Phase 4 (Rule #4): Inherits from mixin classes to reduce file size from 993 to <300 lines.
    """

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "ingestforge_chunks",
        embedding_function: Any = None,
        enable_multi_vector: bool = False,
    ):
        """
        Initialize ChromaDB repository.

        Args:
            persist_directory: Path for persistent storage
            collection_name: Name of the collection
            embedding_function: Optional custom embedding function
            enable_multi_vector: Enable multi-vector search with questions collection

        Raises:
            AssertionError: If persist_directory is invalid
        """
        # Validate persist_directory
        assert persist_directory is not None, "persist_directory cannot be None"
        assert isinstance(
            persist_directory, Path
        ), f"persist_directory must be a Path, got {type(persist_directory).__name__}"

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self._embedding_function = embedding_function
        self.enable_multi_vector = enable_multi_vector

        # Lazy-loaded client and collection
        self._client = None
        self._collection = None
        self._questions_collection = None

        # Ensure directory exists
        self.persist_directory.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> Any:
        """Lazy-load ChromaDB client."""
        if self._client is None:
            try:
                import chromadb

                # Use PersistentClient for automatic persistence (ChromaDB 0.4.x+)
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
            except ImportError:
                raise ImportError(
                    "chromadb is required for ChromaDB storage. "
                    "Install with: pip install chromadb"
                )
        return self._client

    def _create_embedding_function(self) -> Any:
        """Create a custom embedding function using sentence-transformers directly.

        Workaround for ChromaDB's built-in function having torch meta tensor issues.

        Note: CustomEmbeddingFunction is defined inline here rather than in a
        separate embeddings module because:
        1. It requires ChromaDB's EmbeddingFunction base class (tight coupling)
        2. It's only used by this repository class
        3. Moving it would require additional imports and module structure
        """
        try:
            from sentence_transformers import SentenceTransformer
            from chromadb import EmbeddingFunction, Documents, Embeddings

            model = SentenceTransformer("all-MiniLM-L6-v2")

            class CustomEmbeddingFunction(EmbeddingFunction):
                """Custom embedding function wrapping sentence-transformers."""

                def __init__(self, st_model: SentenceTransformer):
                    self._model = st_model

                def __call__(self, input: Documents) -> Embeddings:
                    """Embed a list of documents."""
                    embeddings = self._model.encode(list(input), convert_to_numpy=True)
                    return embeddings.tolist()

            return CustomEmbeddingFunction(model)
        except Exception as e:
            _Logger.get().warning(f"Could not create custom embedding function: {e}")
            return None

    @property
    def collection(self) -> Any:
        """Get or create collection."""
        if self._collection is None:
            # Create custom embedding function if needed
            if self._embedding_function is None:
                self._embedding_function = self._create_embedding_function()

            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

        return self._collection

    @property
    def questions_collection(self) -> Any:
        """Get or create questions collection for multi-vector search."""
        if self._questions_collection is None and self.enable_multi_vector:
            # Create custom embedding function if needed
            if self._embedding_function is None:
                self._embedding_function = self._create_embedding_function()

            self._questions_collection = self.client.get_or_create_collection(
                name=f"{self.collection_name}_questions",
                embedding_function=self._embedding_function,
                metadata={"hnsw:space": "cosine"},
            )

        return self._questions_collection
