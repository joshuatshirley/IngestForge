"""Cross-Encoder Reranker.

Re-scores retrieval candidates using a high-precision local cross-encoder.
Follows NASA JPL Power of Ten rules:
- Rule #1: Simple control flow (no recursion)
- Rule #2: Fixed upper bounds (max 50 candidates)
- Rule #4: Short functions (<60 lines)
- Rule #5: Assertions at entry points
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.storage.base import SearchResult

logger = get_logger(__name__)

# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================

MAX_CANDIDATES = 50
MAX_TOP_K = 100
MIN_TERM_LENGTH = 3
TITLE_WEIGHT = 0.30
PHRASE_WEIGHT = 0.20
COVERAGE_WEIGHT = 0.40
ORIGINAL_WEIGHT = 0.10


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class RerankReport:
    """Report from reranking operation.

    Attributes:
        query: The search query used for reranking.
        method: The reranking method used (cross-encoder, semantic, fallback).
        input_count: Number of results before reranking.
        output_count: Number of results after reranking.
        rank_changes: Number of position changes in ranking.
        time_ms: Time taken for reranking in milliseconds.
    """

    query: str
    method: str
    input_count: int
    output_count: int
    rank_changes: int = 0
    time_ms: float = 0.0

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.input_count >= 0, "input_count cannot be negative"
        assert self.output_count >= 0, "output_count cannot be negative"
        assert self.output_count <= self.input_count, "output cannot exceed input"
        assert self.time_ms >= 0, "time_ms cannot be negative"


# =============================================================================
# Reranker Class
# =============================================================================


class Reranker:
    """Logic for high-precision result re-scoring.

    Supports three reranking methods:
    - cross-encoder: High precision using CrossEncoder model
    - semantic: Using sentence embeddings similarity
    - fallback: Term-based scoring (no external dependencies)

    JPL Rule #4: Modular design with focused methods.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize reranker.

        Args:
            model_name: Optional cross-encoder model name.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._cross_encoder: Any = None  # None=not loaded, False=unavailable
        self._semantic_model: Any = None  # None=not loaded, False=unavailable

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
        method: str = "auto",
    ) -> List[SearchResult]:
        """Re-score and sort the provided results.

        JPL Rule #1: Linear flow with early returns.
        JPL Rule #2: Bounded input size.

        Args:
            query: The search query.
            results: List of search results to rerank.
            top_k: Maximum number of results to return.
            method: Reranking method (auto, cross-encoder, semantic, fallback).

        Returns:
            Reranked list of SearchResult objects.
        """
        # JPL Rule #5: Validate inputs
        if not results:
            return []
        if not query:
            return results[:top_k]

        # JPL Rule #2: Bound inputs
        candidates = results[:MAX_CANDIDATES]
        top_k = min(top_k, MAX_TOP_K)

        # Select and execute reranking method
        if method == "auto":
            return self._rerank_auto(query, candidates, top_k)
        elif method == "cross-encoder":
            return self._rerank_with_fallback(
                query, candidates, top_k, self._rerank_cross_encoder
            )
        elif method == "semantic":
            return self._rerank_with_fallback(
                query, candidates, top_k, self._rerank_semantic
            )
        else:
            return self._rerank_fallback(query, candidates, top_k)

    # -------------------------------------------------------------------------
    # Private Methods - Method Selection
    # -------------------------------------------------------------------------

    def _rerank_auto(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Auto-select best available reranking method.

        Priority: cross-encoder > semantic > fallback
        """
        # Try cross-encoder first
        if self._cross_encoder is not False:
            if self._cross_encoder is not None:
                return self._rerank_cross_encoder(query, results, top_k)
            # Try to load cross-encoder
            if self._try_load_cross_encoder():
                return self._rerank_cross_encoder(query, results, top_k)

        # Try semantic model
        if self._semantic_model is not False:
            if self._semantic_model is not None:
                return self._rerank_semantic(query, results, top_k)
            # Try to load semantic model
            if self._try_load_semantic():
                return self._rerank_semantic(query, results, top_k)

        # Fallback to term-based reranking
        return self._rerank_fallback(query, results, top_k)

    def _rerank_with_fallback(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        primary_method: Any,
    ) -> List[SearchResult]:
        """Try primary method, fall back on failure."""
        try:
            # Check if model is available
            if primary_method == self._rerank_cross_encoder:
                if self._cross_encoder is False:
                    return self._rerank_fallback(query, results, top_k)
                if self._cross_encoder is None and not self._try_load_cross_encoder():
                    return self._rerank_fallback(query, results, top_k)
            elif primary_method == self._rerank_semantic:
                if self._semantic_model is False:
                    return self._rerank_fallback(query, results, top_k)
                if self._semantic_model is None and not self._try_load_semantic():
                    return self._rerank_fallback(query, results, top_k)

            return primary_method(query, results, top_k)
        except Exception as e:
            logger.debug(f"Reranking failed: {e}, using fallback")
            return self._rerank_fallback(query, results, top_k)

    # -------------------------------------------------------------------------
    # Private Methods - Model Loading
    # -------------------------------------------------------------------------

    def _try_load_cross_encoder(self) -> bool:
        """Try to load cross-encoder model. Returns True on success."""
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self.model_name)
            logger.info(f"Loaded cross-encoder: {self.model_name}")
            return True
        except ImportError:
            logger.debug("sentence-transformers not installed")
            self._cross_encoder = False
            return False
        except Exception as e:
            logger.debug(f"Failed to load cross-encoder: {e}")
            self._cross_encoder = False
            return False

    def _try_load_semantic(self) -> bool:
        """Try to load semantic embedding model. Returns True on success."""
        try:
            from sentence_transformers import SentenceTransformer

            self._semantic_model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded semantic model: all-MiniLM-L6-v2")
            return True
        except ImportError:
            logger.debug("sentence-transformers not installed")
            self._semantic_model = False
            return False
        except Exception as e:
            logger.debug(f"Failed to load semantic model: {e}")
            self._semantic_model = False
            return False

    # -------------------------------------------------------------------------
    # Private Methods - Reranking Implementations
    # -------------------------------------------------------------------------

    def _rerank_cross_encoder(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Rerank using cross-encoder model."""
        pairs = [(query, r.content) for r in results]
        scores = self._cross_encoder.predict(pairs)

        # Update scores and metadata
        for i, score in enumerate(scores):
            results[i].score = float(score)
            results[i].metadata = results[i].metadata or {}
            results[i].metadata["rerank_method"] = "cross-encoder"

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Cross-encoder reranked {len(results)} candidates")
        return results[:top_k]

    def _rerank_semantic(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Rerank using semantic similarity."""
        import numpy as np

        query_embedding = self._semantic_model.encode(query)
        contents = [r.content for r in results]
        content_embeddings = self._semantic_model.encode(contents)

        # Compute cosine similarities
        similarities = np.dot(content_embeddings, query_embedding) / (
            np.linalg.norm(content_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # Update scores and metadata
        for i, score in enumerate(similarities):
            results[i].score = float(score)
            results[i].metadata = results[i].metadata or {}
            results[i].metadata["rerank_method"] = "semantic"

        results.sort(key=lambda x: x.score, reverse=True)
        logger.info(f"Semantic reranked {len(results)} candidates")
        return results[:top_k]

    def _rerank_fallback(
        self, query: str, results: List[SearchResult], top_k: int
    ) -> List[SearchResult]:
        """Rerank using term-based scoring (no external dependencies).

        Scoring formula:
        - 40% term coverage
        - 30% title match
        - 20% exact phrase match
        - 10% original score preservation
        """
        if not query or not query.strip():
            # No query, return original order
            for r in results:
                r.metadata = r.metadata or {}
                r.metadata["rerank_method"] = "fallback"
            return results[:top_k]

        # Extract query terms (filter short terms)
        query_lower = query.lower()
        query_terms = [t for t in query_lower.split() if len(t) >= MIN_TERM_LENGTH]

        if not query_terms:
            # All terms filtered, return original order
            for r in results:
                r.metadata = r.metadata or {}
                r.metadata["rerank_method"] = "fallback"
            return results[:top_k]

        # Score each result
        for result in results:
            content_lower = result.content.lower()
            title_lower = (result.section_title or "").lower()

            # Term coverage score (40%)
            matched = sum(1 for t in query_terms if t in content_lower)
            coverage_score = matched / len(query_terms) if query_terms else 0

            # Title match score (30%)
            title_matched = sum(1 for t in query_terms if t in title_lower)
            title_score = title_matched / len(query_terms) if query_terms else 0

            # Exact phrase match score (20%)
            phrase_score = 1.0 if query_lower in content_lower else 0.0

            # Original score preservation (10%)
            original_score = result.score

            # Combined score
            result.score = (
                COVERAGE_WEIGHT * coverage_score
                + TITLE_WEIGHT * title_score
                + PHRASE_WEIGHT * phrase_score
                + ORIGINAL_WEIGHT * original_score
            )
            result.metadata = result.metadata or {}
            result.metadata["rerank_method"] = "fallback"

        results.sort(key=lambda x: x.score, reverse=True)
        logger.debug(f"Fallback reranked {len(results)} candidates")
        return results[:top_k]
