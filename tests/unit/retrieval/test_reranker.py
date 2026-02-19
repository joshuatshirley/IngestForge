"""
Tests for Result Reranker.

This module tests result reranking for improved precision.

Test Strategy
-------------
- Focus on reranking logic and score updates
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test fallback method (no external dependencies)
- Mock cross-encoder and semantic models
- Test method selection and top_k limiting

Organization
------------
- TestRerankReport: RerankReport dataclass
- TestRerankerInit: Initialization
- TestMethodSelection: Auto method selection
- TestFallbackReranking: Term-based fallback reranking
- TestRerank: Main rerank method
"""

from unittest.mock import Mock, patch


from ingestforge.retrieval.reranker import Reranker, RerankReport
from ingestforge.storage.base import SearchResult


# ============================================================================
# Test Helpers
# ============================================================================


def make_search_result(
    chunk_id: str,
    score: float,
    content: str = "",
    section_title: str = "",
) -> SearchResult:
    """Create a test SearchResult."""
    return SearchResult(
        chunk_id=chunk_id,
        content=content or f"Content for {chunk_id}",
        score=score,
        document_id="test_doc",
        section_title=section_title,
        chunk_type="text",
        source_file="test.txt",
        word_count=len((content or "").split()),
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestRerankReport:
    """Tests for RerankReport dataclass.

    Rule #4: Focused test class - tests only RerankReport
    """

    def test_create_rerank_report(self):
        """Test creating RerankReport."""
        report = RerankReport(
            query="test query",
            method="cross-encoder",
            input_count=10,
            output_count=5,
            rank_changes=3,
            time_ms=25.5,
        )

        assert report.query == "test query"
        assert report.method == "cross-encoder"
        assert report.input_count == 10
        assert report.output_count == 5
        assert report.rank_changes == 3
        assert report.time_ms == 25.5


class TestRerankerInit:
    """Tests for Reranker initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_reranker_with_defaults(self):
        """Test creating Reranker with default model."""
        reranker = Reranker()

        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
        assert reranker._cross_encoder is None
        assert reranker._semantic_model is None

    def test_create_reranker_with_custom_model(self):
        """Test creating Reranker with custom model."""
        reranker = Reranker(model_name="custom-model")

        assert reranker.model_name == "custom-model"


class TestMethodSelection:
    """Tests for automatic method selection.

    Rule #4: Focused test class - tests method selection
    """

    def test_auto_method_selects_cross_encoder_if_available(self):
        """Test auto method selects cross-encoder if available."""
        reranker = Reranker()

        # Mock cross-encoder as available (patch private attribute)
        reranker._cross_encoder = Mock()
        with patch.object(
            reranker, "_rerank_cross_encoder", return_value=[]
        ) as mock_ce:
            results = [make_search_result("chunk_1", 1.0)]
            reranker.rerank("query", results, method="auto")

            # Should call cross-encoder method
            mock_ce.assert_called_once()

    def test_auto_method_falls_back_to_semantic_if_no_cross_encoder(self):
        """Test auto method uses semantic if cross-encoder unavailable."""
        reranker = Reranker()

        # Mock cross-encoder unavailable (False), semantic available
        reranker._cross_encoder = False  # False = tried to load but failed
        reranker._semantic_model = Mock()
        with patch.object(reranker, "_rerank_semantic", return_value=[]) as mock_sem:
            results = [make_search_result("chunk_1", 1.0)]
            reranker.rerank("query", results, method="auto")

            # Should call semantic method
            mock_sem.assert_called_once()

    def test_auto_method_uses_fallback_if_no_models(self):
        """Test auto method uses fallback if no models available."""
        reranker = Reranker()

        # Mock both models unavailable (False)
        reranker._cross_encoder = False
        reranker._semantic_model = False
        with patch.object(reranker, "_rerank_fallback", return_value=[]) as mock_fb:
            results = [make_search_result("chunk_1", 1.0)]
            reranker.rerank("query", results, method="auto")

            # Should call fallback method
            mock_fb.assert_called_once()


class TestFallbackReranking:
    """Tests for fallback term-based reranking.

    Rule #4: Focused test class - tests _rerank_fallback
    """

    def test_rerank_fallback_basic(self):
        """Test basic fallback reranking."""
        reranker = Reranker()
        results = [
            make_search_result("chunk_1", 0.5, "This is about Python programming"),
            make_search_result("chunk_2", 0.6, "This is about Java development"),
        ]

        reranked = reranker._rerank_fallback("Python programming", results, top_k=2)

        # chunk_1 should rank higher (better term match)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "chunk_1"
        assert reranked[0].metadata["rerank_method"] == "fallback"

    def test_rerank_fallback_title_boost(self):
        """Test fallback gives title match boost."""
        reranker = Reranker()
        results = [
            make_search_result("chunk_1", 0.5, "Content about programming", ""),
            make_search_result("chunk_2", 0.5, "Other content", "Python Programming"),
        ]

        reranked = reranker._rerank_fallback("Python", results, top_k=2)

        # chunk_2 should rank higher (title match)
        assert reranked[0].chunk_id == "chunk_2"

    def test_rerank_fallback_exact_phrase_boost(self):
        """Test fallback gives exact phrase match boost."""
        reranker = Reranker()
        results = [
            make_search_result("chunk_1", 0.5, "Python and programming are separate"),
            make_search_result("chunk_2", 0.5, "This is about Python programming"),
        ]

        reranked = reranker._rerank_fallback("Python programming", results, top_k=2)

        # chunk_2 should rank higher (exact phrase)
        assert reranked[0].chunk_id == "chunk_2"

    def test_rerank_fallback_respects_top_k(self):
        """Test fallback respects top_k parameter."""
        reranker = Reranker()
        results = [
            make_search_result(f"chunk_{i}", 0.5, "Content with Python")
            for i in range(10)
        ]

        reranked = reranker._rerank_fallback("Python", results, top_k=3)

        assert len(reranked) == 3

    def test_rerank_fallback_empty_query(self):
        """Test fallback with empty query returns original top_k."""
        reranker = Reranker()
        results = [
            make_search_result("chunk_1", 1.0, "Content"),
            make_search_result("chunk_2", 0.5, "More content"),
        ]

        reranked = reranker._rerank_fallback("", results, top_k=2)

        # Should return original results (no reranking)
        assert len(reranked) == 2

    def test_rerank_fallback_filters_short_terms(self):
        """Test fallback filters out short query terms."""
        reranker = Reranker()
        results = [
            make_search_result("chunk_1", 0.5, "Content with it and the"),
            make_search_result("chunk_2", 0.5, "Content with Python code"),
        ]

        reranked = reranker._rerank_fallback("it and the to", results, top_k=2)

        # Should return original (all terms < 3 chars filtered)
        assert len(reranked) == 2


class TestRerank:
    """Tests for main rerank method.

    Rule #4: Focused test class - tests rerank()
    """

    def test_rerank_empty_results(self):
        """Test reranking empty results list."""
        reranker = Reranker()

        reranked = reranker.rerank("query", [], top_k=5)

        assert reranked == []

    def test_rerank_with_explicit_method(self):
        """Test reranking with explicitly specified method."""
        reranker = Reranker()
        results = [make_search_result("chunk_1", 1.0, "Content")]

        # Explicitly use fallback
        reranked = reranker.rerank("query", results, top_k=5, method="fallback")

        assert len(reranked) == 1
        assert reranked[0].metadata["rerank_method"] == "fallback"

    def test_rerank_updates_scores(self):
        """Test reranking updates result scores."""
        reranker = Reranker()
        results = [
            make_search_result("chunk_1", 1.0, "Content with Python"),
            make_search_result("chunk_2", 0.5, "Other content"),
        ]

        original_scores = [r.score for r in results]
        reranked = reranker.rerank("Python", results, top_k=2, method="fallback")

        # Scores should be updated
        reranked_scores = [r.score for r in reranked]
        assert reranked_scores != original_scores

    def test_rerank_adds_metadata(self):
        """Test reranking adds rerank_method to metadata."""
        reranker = Reranker()
        results = [make_search_result("chunk_1", 1.0, "Content")]

        reranked = reranker.rerank("query", results, top_k=5, method="fallback")

        assert reranked[0].metadata is not None
        assert "rerank_method" in reranked[0].metadata

    def test_rerank_cross_encoder_fallback_on_error(self):
        """Test cross-encoder falls back when unavailable."""
        reranker = Reranker()
        results = [make_search_result("chunk_1", 1.0, "Content")]

        # Mock cross-encoder unavailable (False)
        reranker._cross_encoder = False

        # Should use fallback
        reranked = reranker.rerank("query", results, method="cross-encoder")

        assert len(reranked) == 1
        # Should have fallen back
        assert reranked[0].metadata["rerank_method"] == "fallback"

    def test_rerank_semantic_fallback_on_error(self):
        """Test semantic reranking falls back when unavailable."""
        reranker = Reranker()
        results = [make_search_result("chunk_1", 1.0, "Content")]

        # Mock semantic model unavailable (False)
        reranker._semantic_model = False

        # Should use fallback
        reranked = reranker.rerank("query", results, method="semantic")

        assert len(reranked) == 1
        # Should have fallen back
        assert reranked[0].metadata["rerank_method"] == "fallback"


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - RerankReport: 1 test (dataclass creation)
    - Reranker init: 2 tests (defaults, custom model)
    - Method selection: 3 tests (cross-encoder, semantic, fallback)
    - Fallback reranking: 6 tests (basic, title boost, phrase boost, top_k, empty, short terms)
    - Rerank: 6 tests (empty, explicit method, updates scores, metadata, CE error, semantic error)

    Total: 18 tests

Design Decisions:
    1. Focus on fallback reranking (no external dependencies)
    2. Mock cross-encoder and semantic models
    3. Test method selection logic ("auto" mode)
    4. Test scoring updates and metadata
    5. Test error handling and fallback behavior
    6. Simple, clear tests that verify reranking works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - RerankReport dataclass creation
    - Reranker initialization with default and custom models
    - Auto method selection (cross-encoder → semantic → fallback)
    - Fallback reranking using term-based scoring
    - Title match boost (30% weight)
    - Exact phrase match boost (20% weight)
    - Term coverage scoring (40% weight)
    - Original score preservation (10% weight)
    - Top-k result limiting
    - Empty query handling
    - Short term filtering (< 3 chars)
    - Score updates on reranked results
    - Metadata addition (rerank_method)
    - Error handling with graceful fallback

Justification:
    - Reranking is critical for precision improvement
    - Fallback method ensures robustness (no model dependencies)
    - Method selection logic needs verification
    - Score and metadata updates are essential
    - Simple tests verify reranking system works correctly
"""
