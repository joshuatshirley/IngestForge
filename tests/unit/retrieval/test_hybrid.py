"""
Tests for Hybrid Retriever.

This module tests hybrid retrieval combining BM25 and semantic search.

Test Strategy
-------------
- Focus on fusion logic (weighted and RRF)
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Mock BM25 and semantic retrievers
- Test score normalization and combination

Organization
------------
- TestHybridRetrieverInit: Initialization and lazy loading
- TestScoreNormalization: _normalize_scores function
- TestWeightedFusion: Weighted score fusion
- TestRRFFusion: Reciprocal Rank Fusion
- TestSearch: Main search method
- TestIntentAwareWeights: Intent-based weight override
"""

from unittest.mock import Mock

import pytest

from ingestforge.core.config import Config
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.storage.base import SearchResult


# ============================================================================
# Test Helpers
# ============================================================================


def make_search_result(chunk_id: str, score: float, content: str = "") -> SearchResult:
    """Create a test SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content or f"Content for {chunk_id}",
        score=score,
        document_id="test_doc",
        section_title="Test Section",
        chunk_type="text",
        source_file="test.txt",
        word_count=10,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestHybridRetrieverInit:
    """Tests for HybridRetriever initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_hybrid_retriever_with_storage(self):
        """Test creating HybridRetriever with storage."""
        config = Config()
        storage = Mock()

        retriever = HybridRetriever(config, storage)

        assert retriever.config is config
        assert retriever.storage is storage
        assert retriever._bm25 is None
        assert retriever._semantic is None

    def test_create_hybrid_retriever_with_custom_retrievers(self):
        """Test creating HybridRetriever with custom retrievers."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        assert retriever._bm25 is bm25_mock
        assert retriever._semantic is semantic_mock

    def test_lazy_load_bm25_retriever(self):
        """Test lazy loading of BM25 retriever."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        # Access bm25 property
        bm25 = retriever.bm25

        # Should create BM25Retriever
        assert bm25 is not None
        assert retriever._bm25 is bm25
        # Second access returns same instance
        assert retriever.bm25 is bm25

    def test_lazy_load_semantic_retriever(self):
        """Test lazy loading of semantic retriever."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        # Access semantic property
        semantic = retriever.semantic

        # Should create SemanticRetriever
        assert semantic is not None
        assert retriever._semantic is semantic
        # Second access returns same instance
        assert retriever.semantic is semantic


class TestScoreNormalization:
    """Tests for score normalization.

    Rule #4: Focused test class - tests _normalize_scores only
    """

    def test_normalize_scores_basic(self):
        """Test basic score normalization."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        results = [
            make_search_result("chunk_1", 10.0),
            make_search_result("chunk_2", 5.0),
            make_search_result("chunk_3", 2.5),
        ]

        normalized = retriever._normalize_scores(results)

        # Max score is 10.0, so normalize to [0,1]
        assert normalized["chunk_1"] == 1.0
        assert normalized["chunk_2"] == 0.5
        assert normalized["chunk_3"] == 0.25

    def test_normalize_scores_empty_list(self):
        """Test normalizing empty results list."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        normalized = retriever._normalize_scores([])

        assert normalized == {}

    def test_normalize_scores_all_zero(self):
        """Test normalizing when all scores are zero."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        results = [
            make_search_result("chunk_1", 0.0),
            make_search_result("chunk_2", 0.0),
        ]

        normalized = retriever._normalize_scores(results)

        # Should return empty dict for all-zero scores
        assert normalized == {}


class TestWeightedFusion:
    """Tests for weighted score fusion.

    Rule #4: Focused test class - tests weighted fusion
    """

    def test_combine_weighted_scores_both_present(self):
        """Test combining scores when chunk appears in both lists."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.4
        retriever.semantic_weight = 0.6

        bm25_scores = {"chunk_1": 1.0}
        semantic_scores = {"chunk_1": 1.0}

        fused = retriever._combine_weighted_scores(bm25_scores, semantic_scores)

        # Fused = 0.4 * 1.0 + 0.6 * 1.0 = 1.0
        assert fused["chunk_1"][0] == 1.0
        assert fused["chunk_1"][1] == 1.0  # BM25 score
        assert fused["chunk_1"][2] == 1.0  # Semantic score

    def test_combine_weighted_scores_bm25_only(self):
        """Test combining when chunk only in BM25 results."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.4
        retriever.semantic_weight = 0.6

        bm25_scores = {"chunk_1": 1.0}
        semantic_scores = {}

        fused = retriever._combine_weighted_scores(bm25_scores, semantic_scores)

        # Fused = 0.4 * 1.0 + 0.6 * 0.0 = 0.4
        assert fused["chunk_1"][0] == 0.4
        assert fused["chunk_1"][1] == 1.0  # BM25 score
        assert fused["chunk_1"][2] == 0.0  # Semantic score (missing)

    def test_combine_weighted_scores_semantic_only(self):
        """Test combining when chunk only in semantic results."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.4
        retriever.semantic_weight = 0.6

        bm25_scores = {}
        semantic_scores = {"chunk_1": 1.0}

        fused = retriever._combine_weighted_scores(bm25_scores, semantic_scores)

        # Fused = 0.4 * 0.0 + 0.6 * 1.0 = 0.6
        assert fused["chunk_1"][0] == 0.6
        assert fused["chunk_1"][1] == 0.0  # BM25 score (missing)
        assert fused["chunk_1"][2] == 1.0  # Semantic score

    def test_fuse_weighted_end_to_end(self):
        """Test complete weighted fusion workflow."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.5
        retriever.semantic_weight = 0.5

        bm25_results = [
            make_search_result("chunk_1", 10.0),
            make_search_result("chunk_2", 5.0),
        ]
        semantic_results = [
            make_search_result("chunk_2", 8.0),
            make_search_result("chunk_3", 4.0),
        ]

        fused = retriever._fuse_weighted(bm25_results, semantic_results, top_k=5)

        # chunk_1: BM25=1.0 (10/10), Sem=0.0 → 0.5 * 1.0 + 0.5 * 0.0 = 0.5
        # chunk_2: BM25=0.5 (5/10), Sem=1.0 (8/8) → 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        # chunk_3: BM25=0.0, Sem=0.5 (4/8) → 0.5 * 0.0 + 0.5 * 0.5 = 0.25
        # Ranked: chunk_2 (0.75), chunk_1 (0.5), chunk_3 (0.25)
        assert len(fused) == 3
        assert fused[0].chunk_id == "chunk_2"
        assert fused[1].chunk_id == "chunk_1"
        assert fused[2].chunk_id == "chunk_3"


class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion.

    Rule #4: Focused test class - tests RRF fusion
    """

    def test_calculate_rrf_scores_basic(self):
        """Test basic RRF score calculation."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        bm25_results = [
            make_search_result("chunk_1", 10.0),
            make_search_result("chunk_2", 5.0),
        ]
        semantic_results = [
            make_search_result("chunk_2", 8.0),
            make_search_result("chunk_1", 4.0),
        ]

        k = 60
        rrf_scores = retriever._calculate_rrf_scores(bm25_results, semantic_results, k)

        # chunk_1: rank 1 in BM25, rank 2 in semantic
        # RRF = 1/(60+1) + 1/(60+2) = 1/61 + 1/62 ≈ 0.0164 + 0.0161 = 0.0325
        # chunk_2: rank 2 in BM25, rank 1 in semantic
        # RRF = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.0161 + 0.0164 = 0.0325
        assert abs(rrf_scores["chunk_1"] - 0.0325) < 0.001
        assert abs(rrf_scores["chunk_2"] - 0.0325) < 0.001

    def test_calculate_rrf_scores_only_one_list(self):
        """Test RRF when chunk appears in only one list."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        bm25_results = [make_search_result("chunk_1", 10.0)]
        semantic_results = [make_search_result("chunk_2", 8.0)]

        k = 60
        rrf_scores = retriever._calculate_rrf_scores(bm25_results, semantic_results, k)

        # chunk_1: only in BM25 rank 1 → 1/(60+1) ≈ 0.0164
        # chunk_2: only in semantic rank 1 → 1/(60+1) ≈ 0.0164
        assert abs(rrf_scores["chunk_1"] - 0.0164) < 0.001
        assert abs(rrf_scores["chunk_2"] - 0.0164) < 0.001

    def test_fuse_rrf_end_to_end(self):
        """Test complete RRF fusion workflow."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        bm25_results = [
            make_search_result("chunk_1", 10.0),
            make_search_result("chunk_2", 5.0),
        ]
        semantic_results = [
            make_search_result("chunk_2", 8.0),
            make_search_result("chunk_3", 4.0),
        ]

        fused = retriever._fuse_rrf(bm25_results, semantic_results, top_k=5, k=60)

        # chunk_1: rank 1 in BM25 only → 1/61
        # chunk_2: rank 2 in BM25, rank 1 in semantic → 1/62 + 1/61
        # chunk_3: rank 2 in semantic only → 1/62
        # chunk_2 should rank highest
        assert len(fused) == 3
        assert fused[0].chunk_id == "chunk_2"
        # Verify metadata
        assert fused[0].metadata["fusion_method"] == "rrf"
        assert fused[0].metadata["rrf_k"] == 60


class TestSearch:
    """Tests for main search method.

    Rule #4: Focused test class - tests search() only
    """

    def test_search_sequential_weighted(self):
        """Test sequential search with weighted fusion."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        # Mock retriever responses
        bm25_mock.search.return_value = [make_search_result("chunk_1", 10.0)]
        semantic_mock.search.return_value = [make_search_result("chunk_2", 8.0)]

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)
        retriever.bm25_weight = 0.5
        retriever.semantic_weight = 0.5

        results = retriever.search("test query", top_k=5, use_parallel=False)

        # Should call both retrievers
        bm25_mock.search.assert_called_once()
        semantic_mock.search.assert_called_once()
        # Should return fused results
        assert len(results) == 2

    def test_search_with_rrf_fusion(self):
        """Test search with RRF fusion method."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        bm25_mock.search.return_value = [make_search_result("chunk_1", 10.0)]
        semantic_mock.search.return_value = [make_search_result("chunk_2", 8.0)]

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        results = retriever.search(
            "test query", top_k=5, use_parallel=False, fusion_method="rrf"
        )

        assert len(results) == 2
        # Verify RRF metadata
        assert results[0].metadata["fusion_method"] == "rrf"

    def test_search_respects_top_k(self):
        """Test search respects top_k parameter."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        # Return many results
        bm25_mock.search.return_value = [
            make_search_result(f"chunk_{i}", 10.0 - i) for i in range(10)
        ]
        semantic_mock.search.return_value = [
            make_search_result(f"chunk_{i}", 8.0 - i * 0.5) for i in range(10)
        ]

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        results = retriever.search("test query", top_k=3, use_parallel=False)

        # Should return only top 3
        assert len(results) == 3

    def test_search_passes_library_filter(self):
        """Test search passes library_filter to retrievers."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        bm25_mock.search.return_value = []
        semantic_mock.search.return_value = []

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        retriever.search(
            "test query", top_k=5, use_parallel=False, library_filter="test_lib"
        )

        # Verify library_filter passed to both
        bm25_call_kwargs = bm25_mock.search.call_args[1]
        semantic_call_kwargs = semantic_mock.search.call_args[1]
        assert bm25_call_kwargs["library_filter"] == "test_lib"
        assert semantic_call_kwargs["library_filter"] == "test_lib"


class TestIntentAwareWeights:
    """Tests for intent-aware weight override.

    Rule #4: Focused test class - tests query_intent parameter
    """

    def test_search_with_intent_overrides_weights(self):
        """Test search with query_intent temporarily overrides weights."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        bm25_mock.search.return_value = [make_search_result("chunk_1", 10.0)]
        semantic_mock.search.return_value = [make_search_result("chunk_2", 8.0)]

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)
        original_bm25 = retriever.bm25_weight
        original_semantic = retriever.semantic_weight

        # Search with factual intent (high BM25 weight)
        results = retriever.search(
            "test query", top_k=5, use_parallel=False, query_intent="factual"
        )

        # Weights should be restored after search
        assert retriever.bm25_weight == original_bm25
        assert retriever.semantic_weight == original_semantic
        # Results should be returned
        assert len(results) == 2

    def test_search_with_literary_intent(self):
        """Test search with literary intent (high semantic weight)."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        bm25_mock.search.return_value = [make_search_result("chunk_1", 10.0)]
        semantic_mock.search.return_value = [make_search_result("chunk_2", 8.0)]

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        results = retriever.search(
            "test query", top_k=5, use_parallel=False, query_intent="literary"
        )

        # Should apply literary profile weights during search
        assert len(results) == 2

    def test_search_with_unknown_intent_uses_default(self):
        """Test search with unknown intent uses default profile."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        bm25_mock.search.return_value = [make_search_result("chunk_1", 10.0)]
        semantic_mock.search.return_value = []

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)
        original_weights = (retriever.bm25_weight, retriever.semantic_weight)

        results = retriever.search(
            "test query", top_k=5, use_parallel=False, query_intent="unknown_intent"
        )

        # Weights should be restored
        assert (retriever.bm25_weight, retriever.semantic_weight) == original_weights
        assert len(results) == 1


class TestBuildResultMap:
    """Tests for result map building.

    Rule #4: Focused test class - tests _build_result_map
    """

    def test_build_result_map_no_duplicates(self):
        """Test building result map from unique results."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        bm25_results = [make_search_result("chunk_1", 10.0)]
        semantic_results = [make_search_result("chunk_2", 8.0)]

        result_map = retriever._build_result_map(bm25_results, semantic_results)

        assert len(result_map) == 2
        assert "chunk_1" in result_map
        assert "chunk_2" in result_map

    def test_build_result_map_with_duplicates(self):
        """Test building result map with duplicate chunks."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        # Same chunk in both lists
        bm25_results = [make_search_result("chunk_1", 10.0)]
        semantic_results = [make_search_result("chunk_1", 8.0)]

        result_map = retriever._build_result_map(bm25_results, semantic_results)

        # Should keep first occurrence
        assert len(result_map) == 1
        assert "chunk_1" in result_map


class TestCircuitBreaker:
    """Tests for parallel search circuit breaker (TECH-REF-003.1).

    Rule #4: Focused test class - tests timeout behavior
    """

    def test_parallel_search_uses_default_timeout(self):
        """Test parallel search uses 5s default timeout."""
        from ingestforge.retrieval.hybrid import DEFAULT_SEARCH_TIMEOUT_SECONDS

        assert DEFAULT_SEARCH_TIMEOUT_SECONDS == 5.0

    def test_parallel_search_timeout_capped_at_max(self):
        """Test timeout is capped at MAX_SEARCH_TIMEOUT_SECONDS."""

        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        bm25_mock.search.return_value = []
        semantic_mock.search.return_value = []

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        # Call with timeout > max - should be capped
        retriever._search_parallel("test", 10, timeout_seconds=120.0)

        # Both searches should have been called (timeout was capped, not rejected)
        bm25_mock.search.assert_called_once()
        semantic_mock.search.assert_called_once()

    def test_parallel_search_returns_partial_results_on_one_failure(self):
        """Test parallel search returns partial results when one retriever fails."""
        config = Config()
        storage = Mock()
        bm25_mock = Mock()
        semantic_mock = Mock()

        # BM25 returns results, semantic raises exception
        bm25_mock.search.return_value = [make_search_result("chunk_1", 10.0)]
        semantic_mock.search.side_effect = RuntimeError("Semantic search failed")

        retriever = HybridRetriever(config, storage, bm25_mock, semantic_mock)

        bm25_results, semantic_results = retriever._search_parallel(
            "test", 10, timeout_seconds=5.0
        )

        # Should still get BM25 results
        assert len(bm25_results) == 1
        # Semantic should be empty due to error
        assert len(semantic_results) == 0


class TestWeightValidator:
    """Tests for fusion weight validator (TECH-REF-003.2).

    Rule #4: Focused test class - tests weight validation
    """

    def test_weight_check_passes_for_sum_one(self):
        """Test weight check passes when weights sum to 1.0."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.4
        retriever.semantic_weight = 0.6

        # Should not raise or warn (valid weights)
        retriever._check_weight_consistency(strict=True)

    def test_weight_check_strict_raises_on_invalid(self):
        """Test strict mode raises AssertionError for invalid weights."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.5
        retriever.semantic_weight = 0.6  # Sum = 1.1

        with pytest.raises(AssertionError, match="expected 1.0"):
            retriever._check_weight_consistency(strict=True)

    def test_weight_check_non_strict_warns(self):
        """Test non-strict mode logs warning instead of raising."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        retriever.bm25_weight = 0.3
        retriever.semantic_weight = 0.3  # Sum = 0.6

        # Should not raise - just logs warning
        retriever._check_weight_consistency(strict=False)

    def test_weight_check_tolerates_floating_point_error(self):
        """Test weight check allows small floating-point tolerance."""
        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)
        # These should sum to 1.0 but floating point may differ slightly
        retriever.bm25_weight = 0.1 + 0.1 + 0.1 + 0.1  # 0.4
        retriever.semantic_weight = 0.6

        # Should pass strict mode despite potential floating-point error
        retriever._check_weight_consistency(strict=True)


# ============================================================================
# RRF Performance Tests
# ============================================================================


class TestRRFPerformance:
    """
    AC: Performance - Scoring takes < 10ms for 100 candidates.

    Rule #4: Focused test class - tests RRF performance
    """

    def test_rrf_scoring_100_candidates_under_10ms(self):
        """
        AC: RRF scoring for 100 candidates completes in < 10ms.

        This validates the performance requirement from the acceptance criteria.
        """
        import time

        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        # Create 100 BM25 results
        bm25_results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                content=f"Content {i}",
                score=1.0 - (i * 0.01),
                document_id=f"doc_{i % 10}",
                section_title=f"Section {i}",
                chunk_type="text",
                source_file=f"file_{i}.txt",
                word_count=10,
            )
            for i in range(100)
        ]

        # Create 100 semantic results (different order)
        semantic_results = [
            SearchResult(
                chunk_id=f"chunk_{99 - i}",
                content=f"Content {99 - i}",
                score=1.0 - (i * 0.01),
                document_id=f"doc_{(99 - i) % 10}",
                section_title=f"Section {99 - i}",
                chunk_type="text",
                source_file=f"file_{99 - i}.txt",
                word_count=10,
            )
            for i in range(100)
        ]

        # Measure RRF scoring time
        start = time.perf_counter()
        rrf_scores = retriever._calculate_rrf_scores(
            bm25_results, semantic_results, k=60
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Assert performance requirement
        assert elapsed_ms < 10, f"RRF scoring took {elapsed_ms:.2f}ms, expected < 10ms"
        assert len(rrf_scores) == 100, "Should have scores for all 100 chunks"

    def test_full_rrf_fusion_100_candidates_under_10ms(self):
        """
        AC: Full RRF fusion for 100 candidates completes in < 10ms.

        Tests the complete _fuse_rrf() method including result building.
        """
        import time

        config = Config()
        storage = Mock()
        retriever = HybridRetriever(config, storage)

        # Create 100 BM25 results
        bm25_results = [
            SearchResult(
                chunk_id=f"chunk_{i}",
                content=f"Content for chunk {i}",
                score=1.0 - (i * 0.01),
                document_id=f"doc_{i % 10}",
                section_title=f"Section {i}",
                chunk_type="text",
                source_file=f"file_{i}.txt",
                word_count=15,
            )
            for i in range(100)
        ]

        # Create 100 semantic results (shuffled)
        semantic_results = [
            SearchResult(
                chunk_id=f"chunk_{(i * 7) % 100}",
                content=f"Content for chunk {(i * 7) % 100}",
                score=1.0 - (i * 0.01),
                document_id=f"doc_{((i * 7) % 100) % 10}",
                section_title=f"Section {(i * 7) % 100}",
                chunk_type="text",
                source_file=f"file_{(i * 7) % 100}.txt",
                word_count=15,
            )
            for i in range(100)
        ]

        # Measure full RRF fusion time
        start = time.perf_counter()
        fused = retriever._fuse_rrf(bm25_results, semantic_results, top_k=100, k=60)
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Assert performance requirement
        assert (
            elapsed_ms < 10
        ), f"Full RRF fusion took {elapsed_ms:.2f}ms, expected < 10ms"
        assert len(fused) <= 100, "Should return at most 100 results"
        assert all(r.metadata["fusion_method"] == "rrf" for r in fused)


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - HybridRetriever init: 5 tests (storage, custom retrievers, lazy loading x2)
    - Score normalization: 3 tests (basic, empty, all zero)
    - Weighted fusion: 4 tests (both present, BM25 only, semantic only, end-to-end)
    - RRF fusion: 3 tests (basic, one list, end-to-end)
    - Search: 4 tests (sequential, RRF, top_k, library filter)
    - Intent-aware weights: 3 tests (factual, literary, unknown)
    - Build result map: 2 tests (no duplicates, with duplicates)

    Total: 24 tests

Design Decisions:
    1. Focus on fusion logic correctness (weighted vs RRF)
    2. Mock BM25 and semantic retrievers to isolate hybrid logic
    3. Test score normalization thoroughly
    4. Test intent-aware weight override
    5. Test top_k limiting and library filtering
    6. Simple, clear tests that verify fusion works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - HybridRetriever initialization with storage and optional retrievers
    - Lazy loading of BM25 and semantic retrievers
    - Score normalization to [0,1] range
    - Weighted score fusion (w_bm25 * bm25 + w_sem * semantic)
    - RRF score calculation (1/(k+rank) per ranking list)
    - Sequential search execution
    - RRF vs weighted fusion methods
    - Top-k result limiting
    - Library filtering parameter passing
    - Intent-aware weight profile override
    - Temporary weight changes with restore
    - Result map building with duplicate handling

Justification:
    - Hybrid retrieval is critical for accuracy (combines keyword + concept)
    - Fusion logic needs verification with known inputs
    - Intent-aware weighting improves retrieval quality
    - Score normalization critical for fair fusion
    - Simple tests verify hybrid system works correctly
"""
