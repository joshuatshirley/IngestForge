"""
Semantic vector search retriever.

Uses embedding similarity for semantic search.
"""

from typing import Any, List, Optional

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.storage.base import ChunkRepository, SearchResult


logger = get_logger(__name__)


class SemanticRetriever:
    """
    Semantic search using embedding vectors.

    Requires a storage backend that supports vector search.
    """

    def __init__(
        self,
        config: Config,
        storage: ChunkRepository,
    ):
        """
        Initialize semantic retriever.

        Args:
            config: IngestForge configuration
            storage: Storage backend (must support semantic search)
        """
        self.config = config
        self.storage = storage
        self._embedding_model = None

    @property
    def embedding_model(self) -> Any:
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                model_name = self.config.enrichment.embedding_model
                self._embedding_model = SentenceTransformer(model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers required for semantic search. "
                    "Install with: pip install sentence-transformers"
                )
        return self._embedding_model

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for query.

        Args:
            query: Search query

        Returns:
            Embedding vector
        """
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        return embedding.tolist()

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_query_embedding: bool = True,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results
            use_query_embedding: Generate query embedding (vs text search)
            library_filter: If provided, only return chunks from this library

        Returns:
            List of SearchResult
        """
        if use_query_embedding:
            # Generate query embedding
            query_embedding = self.embed_query(query)

            # Search using embedding
            try:
                return self.storage.search_semantic(
                    query_embedding,
                    top_k=top_k,
                    library_filter=library_filter,
                    **kwargs,
                )
            except NotImplementedError:
                # Fallback to text search
                logger.warning(
                    "Storage doesn't support semantic search, using text search"
                )
                return self.storage.search(
                    query, top_k=top_k, library_filter=library_filter, **kwargs
                )
        else:
            return self.storage.search(
                query, top_k=top_k, library_filter=library_filter, **kwargs
            )
