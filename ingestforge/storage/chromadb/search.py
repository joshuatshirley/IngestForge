"""
Search Operations Mixin for ChromaDB.

Handles semantic search, multi-vector search, and ranking.
"""

import time
from typing import Any, Dict, List, Optional

from ingestforge.storage.base import SearchResult, sanitize_tag, chunk_has_tag


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


class ChromaDBSearchMixin:
    """
    Mixin providing search operations for ChromaDB repository.

    Extracted from chromadb.py to reduce file size (Phase 4, Rule #4).
    """

    def search(
        self,
        query: str,
        top_k: int = 10,
        document_filter: Optional[str] = None,
        library_filter: Optional[str] = None,
        tag_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results
            document_filter: Filter by document ID
            library_filter: Filter by library name
            tag_filter: Filter by tag (ORG-002)

        Returns:
            List of SearchResult
        """
        start_time = time.perf_counter()
        try:
            # Sanitize tag filter if provided
            clean_tag = sanitize_tag(tag_filter) if tag_filter else None

            # Build where clause
            where = self._build_where_clause(document_filter, library_filter)

            # Fetch more results if tag filtering (filter in memory)
            fetch_k = top_k * 3 if clean_tag else top_k

            result = self.collection.query(
                query_texts=[query],
                n_results=fetch_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            results = self._process_search_results(result, clean_tag, top_k)

            latency_ms = (time.perf_counter() - start_time) * 1000
            _Logger.get().debug(
                f"Search completed: {len(results)} results in {latency_ms:.2f}ms"
            )
            return results
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            _Logger.get().error(f"Search failed after {latency_ms:.2f}ms: {e}")
            return []

    def _process_search_results(
        self, result: Dict[str, Any], clean_tag: Optional[str], top_k: int
    ) -> List[SearchResult]:
        """Process search results with optional tag filtering.

        Args:
            result: ChromaDB query result
            clean_tag: Sanitized tag filter or None
            top_k: Maximum results to return

        Returns:
            List of SearchResult
        """
        results = []
        if not (result["ids"] and result["ids"][0]):
            return results

        for i, chunk_id in enumerate(result["ids"][0]):
            metadata = result["metadatas"][0][i]

            # Apply tag filter if specified (ORG-002)
            if clean_tag and not chunk_has_tag(metadata, clean_tag):
                continue

            # Convert distance to similarity score
            distance = result["distances"][0][i]
            score = 1.0 - distance

            chunk = self._metadata_to_chunk(
                chunk_id,
                result["documents"][0][i],
                metadata,
            )
            results.append(SearchResult.from_chunk(chunk, score))

            # Stop if we have enough results after tag filtering
            if len(results) >= top_k:
                break

        return results

    def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        document_filter: Optional[str] = None,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using embedding vector.

        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            document_filter: Filter by document ID
            library_filter: Filter by library
            **kwargs: Additional arguments

        Returns:
            List of search results sorted by relevance

        Rule #4: <60 lines - Uses helper functions
        """
        start_time = time.perf_counter()
        try:
            # Build query filters using helper (Rule #1: DRY)
            where = self._build_where_clause(document_filter, library_filter)

            # Execute ChromaDB query
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )

            # Process results using helper
            results = self._process_query_results(result)

            latency_ms = (time.perf_counter() - start_time) * 1000
            _Logger.get().debug(
                f"Semantic search completed: {len(results)} results in {latency_ms:.2f}ms"
            )
            return results

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            _Logger.get().error(f"Semantic search failed after {latency_ms:.2f}ms: {e}")
            return []

    def _build_where_clause(
        self, document_filter: Optional[str], library_filter: Optional[str]
    ) -> Optional[dict]:
        """
        Build where clause for ChromaDB query.

        Args:
            document_filter: Document ID filter
            library_filter: Library filter

        Returns:
            ChromaDB where clause or None

        Rule #4: Helper function - Extracted from search functions
        Rule #1: DRY - Used by multiple search methods
        """
        conditions = []
        if document_filter:
            conditions.append({"document_id": document_filter})
        if library_filter:
            conditions.append({"library": library_filter})

        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {"$and": conditions}
        return None

    def _process_query_results(self, result: dict) -> List[SearchResult]:
        """
        Process ChromaDB query results into SearchResult objects.

        Args:
            result: Raw ChromaDB query result

        Returns:
            List of SearchResult objects

        Rule #4: Helper function - Extracted from search_semantic
        """
        results = []

        if result["ids"] and result["ids"][0]:
            for i, chunk_id in enumerate(result["ids"][0]):
                # Convert distance to similarity score
                distance = result["distances"][0][i]
                score = 1.0 - distance

                # Convert to chunk object
                chunk = self._metadata_to_chunk(
                    chunk_id,
                    result["documents"][0][i],
                    result["metadatas"][0][i],
                )

                results.append(SearchResult.from_chunk(chunk, score))

        return results

    def _search_both_collections(
        self, query: str, top_k: int, where: Optional[dict]
    ) -> tuple:
        """Search both content and questions collections.

        Rule #4: No large functions - Extracted from search_multi_vector

        Returns:
            Tuple of (content_results, question_results)
        """
        # Search content collection
        content_results = self.collection.query(
            query_texts=[query],
            n_results=top_k * 2,  # Get more candidates for fusion
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Search questions collection
        question_results = self.questions_collection.query(
            query_texts=[query],
            n_results=top_k * 2,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return content_results, question_results

    def search_multi_vector(
        self,
        query: str,
        top_k: int = 10,
        content_weight: float = 1.0,
        question_weight: float = 0.8,
        document_filter: Optional[str] = None,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search using multi-vector approach: content + hypothetical questions.

        Rule #4: Function <60 lines (refactored to 35 lines)

        Combines results from both the content collection and questions collection
        using Reciprocal Rank Fusion (RRF) for robust score-agnostic merging.

        Args:
            query: Search query
            top_k: Number of final results to return
            content_weight: Weight for content collection results (default 1.0)
            question_weight: Weight for questions collection results (default 0.8)
            document_filter: Filter by document ID
            library_filter: Filter by library name

        Returns:
            List of SearchResult fused from both collections
        """
        if not self.enable_multi_vector or not self.questions_collection:
            # Fall back to regular search
            return self.search(query, top_k, document_filter, library_filter, **kwargs)

        try:
            # Build where clause
            where = self._build_where_clause(document_filter, library_filter)

            # Search both collections
            content_results, question_results = self._search_both_collections(
                query, top_k, where
            )

            # Fuse results using RRF
            return self._fuse_multi_vector_rrf(
                content_results,
                question_results,
                top_k,
                content_weight,
                question_weight,
            )

        except Exception as e:
            _Logger.get().error(f"Multi-vector search failed: {e}")
            # Fall back to regular search
            return self.search(query, top_k, document_filter, library_filter, **kwargs)

    def _fuse_multi_vector_rrf(
        self,
        content_results: Dict[str, Any],
        question_results: Dict[str, Any],
        top_k: int,
        content_weight: float = 1.0,
        question_weight: float = 0.8,
        k: int = 60,
    ) -> List[SearchResult]:
        """
        Fuse multi-vector results using weighted Reciprocal Rank Fusion.

        RRF formula: score = sum(weight / (k + rank)) for each ranking list

        Args:
            content_results: Results from content collection
            question_results: Results from questions collection
            top_k: Maximum results to return
            content_weight: Weight for content rankings
            question_weight: Weight for question rankings
            k: RRF constant (default 60)

        Returns:
            Fused and re-ranked results
        """
        rrf_scores: Dict[str, float] = {}
        chunk_data: Dict[str, tuple] = {}

        self._process_content_rankings(
            content_results, rrf_scores, chunk_data, content_weight, k
        )
        self._process_question_rankings(
            question_results, rrf_scores, chunk_data, question_weight, k
        )

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
        return self._build_multi_vector_results(ranked, chunk_data)

    def _process_content_rankings(
        self,
        content_results: Dict[str, Any],
        rrf_scores: Dict[str, float],
        chunk_data: Dict[str, tuple],
        weight: float,
        k: int,
    ):
        """Process content results and update RRF scores."""
        if not (content_results["ids"] and content_results["ids"][0]):
            return

        for rank, chunk_id in enumerate(content_results["ids"][0], 1):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + weight / (k + rank)
            if chunk_id not in chunk_data:
                idx = rank - 1
                chunk_data[chunk_id] = (
                    content_results["documents"][0][idx],
                    content_results["metadatas"][0][idx],
                )

    def _process_question_rankings(
        self,
        question_results: Dict[str, Any],
        rrf_scores: Dict[str, float],
        chunk_data: Dict[str, tuple],
        weight: float,
        k: int,
    ):
        """Process question results and map back to parent chunks."""
        if not (question_results["ids"] and question_results["ids"][0]):
            return

        for rank, question_id in enumerate(question_results["ids"][0], 1):
            idx = rank - 1
            metadata = question_results["metadatas"][0][idx]
            parent_chunk_id = metadata.get("parent_chunk_id")

            if not parent_chunk_id:
                continue

            rrf_scores[parent_chunk_id] = rrf_scores.get(
                parent_chunk_id, 0
            ) + weight / (k + rank)

            if parent_chunk_id not in chunk_data:
                chunk = self.get_chunk(parent_chunk_id)
                if chunk:
                    chunk_data[parent_chunk_id] = (
                        chunk.content,
                        self._chunk_to_metadata(chunk),
                    )

    def _build_multi_vector_results(
        self,
        ranked: List[tuple],
        chunk_data: Dict[str, tuple],
    ) -> List[SearchResult]:
        """Build SearchResult list from ranked multi-vector scores."""
        results = []
        for chunk_id, rrf_score in ranked:
            if chunk_id not in chunk_data:
                continue

            document, metadata = chunk_data[chunk_id]
            chunk = self._metadata_to_chunk(chunk_id, document, metadata)
            result = SearchResult.from_chunk(chunk, rrf_score)
            result.metadata = result.metadata or {}
            result.metadata["fusion_method"] = "multi_vector_rrf"
            results.append(result)

        return results
