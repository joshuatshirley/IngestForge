"""
Hybrid Retrieval: BM25 + Semantic Search Fusion.

Combines keyword-based (BM25) and vector-based (semantic) retrieval
for higher recall and precision than either method alone. BM25 catches
exact terminology matches; semantic search finds conceptually similar
content even when wording differs.

Architecture Context
--------------------
HybridRetriever is the default retrieval strategy. It sits between
the QueryPipeline (which calls ``retriever.search()``) and the two
underlying retrieval engines:

    QueryPipeline.process(query)
            ↓
    HybridRetriever.search(query)
        ├── BM25Retriever.search()      ← keyword matching (Okapi BM25)
        ├── SemanticRetriever.search()   ← vector similarity (cosine)
        └── _fuse_weighted() or _fuse_rrf()  ← score fusion
            ↓
    Ranked results with hybrid scores

Fusion Methods
--------------
- **Weighted**: Normalize both score sets to [0,1], then combine with
  configurable weights (default 0.4 BM25 + 0.6 semantic).
- **RRF (Reciprocal Rank Fusion)**: Position-based fusion that is
  robust to score distribution differences. Uses the formula:
  ``score = sum(1 / (k + rank))`` across both ranking lists.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.retrieval.bm25 import BM25Retriever
from ingestforge.retrieval.semantic import SemanticRetriever
from ingestforge.storage.base import ChunkRepository, SearchResult

logger = get_logger(__name__)
DEFAULT_SEARCH_TIMEOUT_SECONDS = 5.0
MAX_SEARCH_TIMEOUT_SECONDS = 60.0


@dataclass
class HybridSearchResult(SearchResult):
    """Extended search result with hybrid scores."""

    bm25_score: float = 0.0
    semantic_score: float = 0.0
    fusion_method: str = "weighted"


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 and semantic search.

    Features:
    - Parallel execution of both search methods
    - Weighted score fusion
    - Reciprocal Rank Fusion (RRF) option
    - Configurable weights
    """

    def __init__(
        self,
        config: Config,
        storage: ChunkRepository,
        bm25_retriever: Optional[BM25Retriever] = None,
        semantic_retriever: Optional[SemanticRetriever] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            config: IngestForge configuration
            storage: Storage backend
            bm25_retriever: Optional pre-configured BM25 retriever
            semantic_retriever: Optional pre-configured semantic retriever
        """
        self.config = config
        self.storage = storage

        # Initialize retrievers
        self._bm25 = bm25_retriever
        self._semantic = semantic_retriever

        # Weights from config
        self.bm25_weight = config.retrieval.hybrid.bm25_weight
        self.semantic_weight = config.retrieval.hybrid.semantic_weight

        # Validate weights on initialization
        self._check_weight_consistency()

        # Authority boost configuration (check if exists in config)
        self.authority_enabled = getattr(
            getattr(config.retrieval, "authority", None), "enabled", False
        )
        self.authority_patterns = getattr(
            getattr(config.retrieval, "authority", None), "patterns", None
        )

    def _check_weight_consistency(self, strict: bool = False) -> None:
        """
        Validate that BM25 and semantic weights sum to 1.0.

        TECH-REF-003.2: Fusion Weight Validator

        Args:
            strict: If True, raise AssertionError instead of warning.

        Raises:
            AssertionError: If strict=True and weights don't sum to 1.0.
        """
        weight_sum = self.bm25_weight + self.semantic_weight
        tolerance = 0.001  # Allow small floating-point tolerance

        if abs(weight_sum - 1.0) > tolerance:
            msg = (
                f"Hybrid weights sum to {weight_sum:.3f}, expected 1.0. "
                f"BM25={self.bm25_weight}, Semantic={self.semantic_weight}"
            )
            if strict:
                raise AssertionError(msg)
            logger.warning(msg)

    @property
    def bm25(self) -> BM25Retriever:
        """Lazy-load BM25 retriever."""
        if self._bm25 is None:
            self._bm25 = BM25Retriever(self.config, self.storage)
        return self._bm25

    @property
    def semantic(self) -> SemanticRetriever:
        """Lazy-load semantic retriever."""
        if self._semantic is None:
            self._semantic = SemanticRetriever(self.config, self.storage)
        return self._semantic

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_parallel: bool = True,
        fusion_method: str = "weighted",
        library_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        query_intent: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Execute multi-stage hybrid search.

        Metadata filtering support.

        Pipeline:
        1. (Optional) Domain Classification
        2. (Optional) HyDE Augmentation
        3. Hybrid Search (BM25 + Semantic)
        4. (Optional) Cross-Encoder Reranking

        Args:
            query: User search query.
            top_k: Number of results to return.
            use_parallel: Whether to run BM25 and Semantic searches in parallel.
            fusion_method: Method to combine results ("weighted" or "rrf").
            library_filter: Optional library/project identifier filter.
            metadata_filter: Optional complex metadata filters (e.g. date, type).
            query_intent: Optional intent classification for weight adjustment.
            **kwargs: Additional configuration overrides.
        """
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        effective_query = query

        # Domain Awareness (Task [QUERY-002])
        domain_strategy = None
        if kwargs.get("enable_domain_routing", True):
            from ingestforge.query.domain_classifier import QueryDomainClassifier

            classifier = QueryDomainClassifier()
            domain_strategy = classifier.get_query_strategy(query)
            logger.debug(f"Detected domain strategy: {domain_strategy.name}")

        # Stage 1: HyDE Augmentation (Task 12.1.1)
        if kwargs.get("use_hyde", False):
            from ingestforge.retrieval.hyde import HyDEGenerator
            from ingestforge.llm.factory import get_llm_client

            llm = get_llm_client(self.config)
            generator = HyDEGenerator(llm)
            effective_query = generator.generate_hypothetical_doc(query)

        # Stage 2: Core Hybrid Search
        # Apply intent-aware weights AND domain modifiers
        with self._temporary_weights(query_intent, domain_strategy):
            # Execute searches and fuse (candidate_k increased for reranking)
            candidate_k = top_k * 5 if kwargs.get("use_rerank", True) else top_k * 3
            bm25_results, semantic_results = self._execute_searches(
                effective_query,
                candidate_k,
                use_parallel,
                library_filter,
                metadata_filter=metadata_filter,
                **kwargs,
            )

            # Fuse results
            fused = self._fuse_results(
                bm25_results, semantic_results, candidate_k, fusion_method
            )

            # Field Boosting (Task [BOOST-002])
            if domain_strategy and domain_strategy.boost_fields:
                from ingestforge.retrieval.rescorer import MetadataRescorer

                rescorer = MetadataRescorer()
                fused = rescorer.rescore(fused, effective_query, domain_strategy)

            results = self._apply_authority_boost(fused)

        # Stage 3: Cross-Encoder Reranking (Task 12.2.1)
        if kwargs.get("use_rerank", True) and results:
            from ingestforge.retrieval.reranker import Reranker

            reranker = Reranker()
            results = reranker.rerank(query, results, top_k=top_k)

        return results[:top_k]

    def _temporary_weights(
        self, query_intent: Optional[str], domain_strategy: Optional[Any] = None
    ):
        """
        Context manager for temporary intent-aware weight override.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        from contextlib import contextmanager

        @contextmanager
        def _weight_context():
            original_bm25 = self.bm25_weight
            original_semantic = self.semantic_weight

            # Apply intent weights if provided
            if query_intent:
                from ingestforge.retrieval.weight_profiles import get_profile

                profile = get_profile(query_intent)
                self.bm25_weight = profile.bm25_weight
                self.semantic_weight = profile.semantic_weight

            # Apply domain-specific modifiers (BM25 vs Semantic balance)
            if domain_strategy:
                self.bm25_weight *= domain_strategy.bm25_modifier
                self.semantic_weight *= domain_strategy.semantic_modifier

                # Re-normalize to ensure they sum to 1.0 (or keep the relative balance)
                total = self.bm25_weight + self.semantic_weight
                if total > 0:
                    self.bm25_weight /= total
                    self.semantic_weight /= total

                logger.debug(
                    f"Domain-optimized weights ({domain_strategy.name}): "
                    f"BM25={self.bm25_weight:.2f}, "
                    f"Semantic={self.semantic_weight:.2f}"
                )

            try:
                yield
            finally:
                self.bm25_weight = original_bm25
                self.semantic_weight = original_semantic

        return _weight_context()

    def _execute_searches(
        self,
        query: str,
        candidate_k: int,
        use_parallel: bool,
        library_filter: Optional[str],
        metadata_filter: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Tuple[List[SearchResult], List[SearchResult]]:
        """
        Execute BM25 and semantic searches.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: Search query
            candidate_k: Number of candidates to retrieve
            use_parallel: Execute in parallel
            library_filter: Library filter
            metadata_filter: Optional metadata filter
            **kwargs: Additional search arguments

        Returns:
            Tuple of (bm25_results, semantic_results)
        """
        if use_parallel:
            return self._search_parallel(
                query,
                candidate_k,
                library_filter=library_filter,
                metadata_filter=metadata_filter,
                **kwargs,
            )

        bm25_results = self.bm25.search(
            query,
            top_k=candidate_k,
            library_filter=library_filter,
            metadata_filter=metadata_filter,
            **kwargs,
        )
        semantic_results = self.semantic.search(
            query,
            top_k=candidate_k,
            library_filter=library_filter,
            metadata_filter=metadata_filter,
            **kwargs,
        )
        return bm25_results, semantic_results

    def _fuse_results(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
        top_k: int,
        fusion_method: str,
    ) -> List[SearchResult]:
        """
        Fuse BM25 and semantic results using specified method.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            bm25_results: BM25 search results
            semantic_results: Semantic search results
            top_k: Number of results to return
            fusion_method: "weighted" or "rrf"

        Returns:
            Fused search results
        """
        if fusion_method == "rrf":
            return self._fuse_rrf(bm25_results, semantic_results, top_k)
        return self._fuse_weighted(bm25_results, semantic_results, top_k)

    def _search_parallel(
        self,
        query: str,
        top_k: int,
        library_filter: Optional[str] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        timeout_seconds: float = DEFAULT_SEARCH_TIMEOUT_SECONDS,
        **kwargs,
    ) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Execute BM25 and semantic search concurrently via thread pool.

        TECH-REF-003.1: Parallel Search Circuit Breaker

        Uses 2 threads because both retrievers are I/O-bound (disk/network).
        Falls back gracefully if either retriever returns empty results or times out.
        Circuit breaker prevents hangs by enforcing a timeout (default 5s).

        Args:
            query: Search query
            top_k: Number of results
            library_filter: Optional library filter
            metadata_filter: Optional metadata filter
            timeout_seconds: Max time to wait for each retriever (circuit breaker).
                Default 5s, max 60s (Rule #2).
            **kwargs: Additional search arguments

        Returns:
            Tuple of (bm25_results, semantic_results) - partial results if timeout
        """
        timeout_seconds = min(timeout_seconds, MAX_SEARCH_TIMEOUT_SECONDS)
        bm25_results = []
        semantic_results = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            bm25_future = executor.submit(
                self.bm25.search,
                query,
                top_k,
                library_filter=library_filter,
                metadata_filter=metadata_filter,
                **kwargs,
            )
            semantic_future = executor.submit(
                self.semantic.search,
                query,
                top_k,
                library_filter=library_filter,
                metadata_filter=metadata_filter,
                **kwargs,
            )

            bm25_results, semantic_results = self._collect_search_results(
                bm25_future, semantic_future, timeout_seconds
            )

        return bm25_results, semantic_results

    def _collect_search_results(
        self,
        bm25_future: Any,
        semantic_future: Any,
        timeout_seconds: float,
    ) -> Tuple[List[SearchResult], List[SearchResult]]:
        """Collect results from parallel search futures.

        Args:
            bm25_future: BM25 search future
            semantic_future: Semantic search future
            timeout_seconds: Timeout in seconds

        Returns:
            Tuple of (bm25_results, semantic_results)
        """
        bm25_results = []
        semantic_results = []

        # Use timeout as a simple circuit breaker
        for future in as_completed(
            [bm25_future, semantic_future], timeout=timeout_seconds
        ):
            try:
                result = future.result(timeout=timeout_seconds)
                if future == bm25_future:
                    bm25_results = result
                else:
                    semantic_results = result
            except TimeoutError:
                retriever_name = "BM25" if future == bm25_future else "Semantic"
                logger.warning(
                    f"{retriever_name} search timed out after {timeout_seconds}s"
                )
            except Exception as e:
                retriever_name = "BM25" if future == bm25_future else "Semantic"
                logger.warning(f"{retriever_name} search failed: {e}")

        return bm25_results, semantic_results

    def _apply_authority_boost(
        self,
        results: List[SearchResult],
    ) -> List[SearchResult]:
        """Apply authority-based score boosting to results.

        Documents with higher authority levels (e.g., primary sources) get
        boosted scores, causing them to rank higher in final results.

        Args:
            results: List of SearchResult objects to boost

        Returns:
            Same list with scores adjusted by authority boost, re-sorted
        """
        if not self.authority_enabled or not results:
            return results

        from ingestforge.retrieval.authority import apply_authority_boost_to_results

        return apply_authority_boost_to_results(
            results,
            authority_patterns=self.authority_patterns,
        )

    def _fuse_weighted(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        """Fuse results using weighted score combination.

        Algorithm:
        1. Normalize BM25 scores to [0,1] by dividing by max score
        2. Normalize semantic scores to [0,1] (same approach)
        3. For each unique chunk: fused = w_bm25 * bm25 + w_sem * semantic
        4. Chunks appearing in only one list get 0.0 for the missing score
        5. Sort by fused score, return top_k

        Args:
            bm25_results: Ranked results from BM25 retriever.
            semantic_results: Ranked results from semantic retriever.
            top_k: Maximum results to return.

        Returns:
            Fused and re-ranked SearchResult list.
        """
        bm25_scores = self._normalize_scores(bm25_results)
        semantic_scores = self._normalize_scores(semantic_results)

        fused_scores = self._combine_weighted_scores(bm25_scores, semantic_scores)
        ranked = sorted(fused_scores.items(), key=lambda x: -x[1][0])[:top_k]

        result_map = self._build_result_map(bm25_results, semantic_results)
        return self._build_weighted_results(ranked, result_map)

    def _normalize_scores(self, results: List[SearchResult]) -> Dict[str, float]:
        """Normalize scores to [0,1] range."""
        if not results:
            return {}

        max_score = max(r.score for r in results)
        if max_score <= 0:
            return {}

        return {r.chunk_id: r.score / max_score for r in results}

    def _combine_weighted_scores(
        self,
        bm25_scores: Dict[str, float],
        semantic_scores: Dict[str, float],
    ) -> Dict[str, Tuple[float, float, float]]:
        """Combine BM25 and semantic scores with weights."""
        all_chunks = set(bm25_scores.keys()) | set(semantic_scores.keys())
        fused_scores: Dict[str, Tuple[float, float, float]] = {}

        for chunk_id in all_chunks:
            bm25 = bm25_scores.get(chunk_id, 0.0)
            semantic = semantic_scores.get(chunk_id, 0.0)
            fused = self.bm25_weight * bm25 + self.semantic_weight * semantic
            fused_scores[chunk_id] = (fused, bm25, semantic)

        return fused_scores

    def _build_result_map(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
    ) -> Dict[str, SearchResult]:
        """Build map of chunk_id to SearchResult."""
        result_map = {}
        for r in bm25_results + semantic_results:
            if r.chunk_id not in result_map:
                result_map[r.chunk_id] = r
        return result_map

    def _build_weighted_results(
        self,
        ranked: List[Tuple[str, Tuple[float, float, float]]],
        result_map: Dict[str, SearchResult],
    ) -> List[SearchResult]:
        """Build SearchResult list from ranked weighted scores."""
        results = []
        for chunk_id, (fused, bm25, semantic) in ranked:
            base_result = result_map.get(chunk_id)
            if not base_result:
                continue

            result = SearchResult(
                chunk_id=chunk_id,
                content=base_result.content,
                score=fused,
                document_id=base_result.document_id,
                section_title=base_result.section_title,
                chunk_type=base_result.chunk_type,
                source_file=base_result.source_file,
                word_count=base_result.word_count,
                page_start=base_result.page_start,
                page_end=base_result.page_end,
                source_location=base_result.source_location,
                metadata={
                    "bm25_score": bm25,
                    "semantic_score": semantic,
                    "fusion_method": "weighted",
                },
            )
            results.append(result)

        return results

    def _fuse_rrf(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
        top_k: int,
        k: int = 60,
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion (RRF).

        RRF is score-distribution agnostic — it only uses rank positions,
        making it robust when BM25 and semantic scores have very different
        distributions. The constant k (default 60) dampens the influence
        of high-ranked items; higher k = more uniform weighting.

        Formula per chunk: ``rrf_score = sum(1 / (k + rank_i))`` across
        each ranking list where the chunk appears.

        Reference: Cormack, Clarke & Buettcher (2009) — "Reciprocal Rank
        Fusion outperforms Condorcet and individual Rank Learning Methods"

        Args:
            bm25_results: BM25-ranked results (1-indexed internally).
            semantic_results: Semantic-ranked results.
            top_k: Maximum results to return.
            k: RRF constant (default 60). Higher = more uniform weighting.

        Returns:
            Fused and re-ranked SearchResult list.
        """
        rrf_scores = self._calculate_rrf_scores(bm25_results, semantic_results, k)
        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]

        result_map = self._build_result_map(bm25_results, semantic_results)
        return self._build_rrf_results(ranked, result_map, k)

    def _calculate_rrf_scores(
        self,
        bm25_results: List[SearchResult],
        semantic_results: List[SearchResult],
        k: int,
    ) -> Dict[str, float]:
        """Calculate RRF scores from ranked result lists."""
        rrf_scores: Dict[str, float] = {}

        # BM25 ranks
        for rank, result in enumerate(bm25_results, 1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)

        # Semantic ranks
        for rank, result in enumerate(semantic_results, 1):
            chunk_id = result.chunk_id
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)

        return rrf_scores

    def _build_rrf_results(
        self,
        ranked: List[Tuple[str, float]],
        result_map: Dict[str, SearchResult],
        k: int,
    ) -> List[SearchResult]:
        """Build SearchResult list from ranked RRF scores."""
        results = []
        for chunk_id, rrf_score in ranked:
            base_result = result_map.get(chunk_id)
            if not base_result:
                continue

            result = SearchResult(
                chunk_id=chunk_id,
                content=base_result.content,
                score=rrf_score,
                document_id=base_result.document_id,
                section_title=base_result.section_title,
                chunk_type=base_result.chunk_type,
                source_file=base_result.source_file,
                word_count=base_result.word_count,
                page_start=base_result.page_start,
                page_end=base_result.page_end,
                source_location=base_result.source_location,
                metadata={
                    "fusion_method": "rrf",
                    "rrf_k": k,
                },
            )
            results.append(result)

        return results
