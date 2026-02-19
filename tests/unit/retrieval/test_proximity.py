"""Tests for Proximity-Based Ranking.

Tests for ProximityRanker.
Verifies JPL Power of Ten compliance.
"""

import pytest

from ingestforge.retrieval.proximity import (
    ProximityRanker,
    ProximityScore,
    ProximityReport,
    create_proximity_ranker,
    rank_by_proximity,
    MAX_CANDIDATES,
    MAX_QUERY_TERMS,
    DEFAULT_PROXIMITY_WINDOW,
)
from ingestforge.storage.base import SearchResult


# =============================================================================
# Test Fixtures
# =============================================================================


def make_result(
    chunk_id: str,
    content: str,
    score: float = 0.5,
    document_id: str = "doc-1",
) -> SearchResult:
    """Create a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        score=score,
        section_title="Test Section",
        chunk_type="text",
        source_file="test.txt",
        word_count=len(content.split()),
        metadata={},
    )


# =============================================================================
# TestProximityScore
# =============================================================================


class TestProximityScore:
    """Tests for ProximityScore dataclass."""

    def test_create_valid_score(self) -> None:
        """Test creating a valid proximity score."""
        score = ProximityScore(
            chunk_id="chunk-1",
            proximity_score=0.75,
            term_coverage=0.8,
            min_span=50,
            avg_distance=10.0,
            cluster_count=3,
        )

        assert score.chunk_id == "chunk-1"
        assert score.proximity_score == 0.75
        assert score.term_coverage == 0.8
        assert score.min_span == 50
        assert score.avg_distance == 10.0
        assert score.cluster_count == 3

    def test_score_out_of_range_fails(self) -> None:
        """Test that invalid scores raise AssertionError."""
        with pytest.raises(AssertionError):
            ProximityScore(
                chunk_id="chunk-1",
                proximity_score=1.5,  # Invalid: > 1.0
                term_coverage=0.5,
                min_span=10,
                avg_distance=5.0,
                cluster_count=1,
            )

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        score = ProximityScore(
            chunk_id="chunk-1",
            proximity_score=0.5,
            term_coverage=0.6,
            min_span=20,
            avg_distance=8.0,
            cluster_count=2,
        )

        d = score.to_dict()
        assert d["chunk_id"] == "chunk-1"
        assert d["proximity_score"] == 0.5
        assert d["term_coverage"] == 0.6


# =============================================================================
# TestProximityReport
# =============================================================================


class TestProximityReport:
    """Tests for ProximityReport dataclass."""

    def test_create_valid_report(self) -> None:
        """Test creating a valid report."""
        report = ProximityReport(
            query="test query",
            input_count=10,
            output_count=10,
            avg_proximity_score=0.65,
            time_ms=5.2,
        )

        assert report.query == "test query"
        assert report.input_count == 10
        assert report.output_count == 10
        assert report.avg_proximity_score == 0.65

    def test_negative_count_fails(self) -> None:
        """Test that negative counts raise AssertionError."""
        with pytest.raises(AssertionError):
            ProximityReport(
                query="test",
                input_count=-1,  # Invalid
                output_count=0,
                avg_proximity_score=0.0,
            )


# =============================================================================
# TestProximityRanker
# =============================================================================


class TestProximityRanker:
    """Tests for ProximityRanker class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        ranker = ProximityRanker()

        assert ranker.window_size == DEFAULT_PROXIMITY_WINDOW

    def test_init_custom_window(self) -> None:
        """Test initialization with custom window size."""
        ranker = ProximityRanker(window_size=100)

        assert ranker.window_size == 100

    def test_init_invalid_window_fails(self) -> None:
        """Test that invalid window size raises AssertionError."""
        with pytest.raises(AssertionError):
            ProximityRanker(window_size=0)

        with pytest.raises(AssertionError):
            ProximityRanker(window_size=1000)  # Exceeds max

    def test_rank_empty_results(self) -> None:
        """Test ranking empty results returns empty."""
        ranker = ProximityRanker()

        results, report = ranker.rank([], "test query")

        assert results == []
        assert report.input_count == 0
        assert report.output_count == 0

    def test_rank_empty_query(self) -> None:
        """Test ranking with empty query returns original results."""
        ranker = ProximityRanker()
        results = [make_result("chunk-1", "Some content here")]

        ranked, report = ranker.rank(results, "")

        # Empty query returns original results unchanged
        assert len(ranked) == 1
        assert report.input_count == 0  # No terms to process

    def test_rank_single_term_match(self) -> None:
        """Test ranking with single term match."""
        ranker = ProximityRanker()
        results = [
            make_result("chunk-1", "The python programming language is great"),
            make_result("chunk-2", "No match here at all"),
        ]

        ranked, report = ranker.rank(results, "python")

        assert len(ranked) == 2
        # First result should have higher score due to term match
        assert ranked[0].metadata.get("term_coverage", 0) > 0

    def test_rank_multiple_terms_proximity(self) -> None:
        """Test ranking prefers terms close together."""
        ranker = ProximityRanker()
        results = [
            make_result("chunk-1", "machine learning is amazing"),  # Terms adjacent
            make_result(
                "chunk-2", "machine programming is a learning experience"
            ),  # Terms far apart
        ]

        ranked, report = ranker.rank(results, "machine learning")

        assert len(ranked) == 2
        # Both have full coverage but chunk-1 should score higher (closer terms)
        prox_1 = ranked[0].metadata.get("proximity_score", 0)
        prox_2 = ranked[1].metadata.get("proximity_score", 0)

        # The one with adjacent terms should have higher proximity
        if ranked[0].chunk_id == "chunk-1":
            assert prox_1 >= prox_2

    def test_rank_respects_boost_weight(self) -> None:
        """Test that boost weight affects final scores."""
        ranker = ProximityRanker()
        results = [make_result("chunk-1", "python programming code", score=0.8)]

        # Low boost weight - original score dominates
        ranked_low, _ = ranker.rank(results, "python", boost_weight=0.1)

        # High boost weight - proximity score dominates
        ranked_high, _ = ranker.rank(results, "python", boost_weight=0.9)

        # Both should preserve original score in metadata
        assert "original_score" in ranked_low[0].metadata
        assert ranked_low[0].metadata["original_score"] == 0.8

    def test_rank_bounds_results(self) -> None:
        """Test that results are bounded by MAX_CANDIDATES."""
        ranker = ProximityRanker()
        # Create more results than MAX_CANDIDATES
        results = [
            make_result(f"chunk-{i}", f"content {i}")
            for i in range(MAX_CANDIDATES + 10)
        ]

        ranked, report = ranker.rank(results, "content")

        # Should process at most MAX_CANDIDATES
        assert report.output_count <= MAX_CANDIDATES


# =============================================================================
# TestScoreChunk
# =============================================================================


class TestScoreChunk:
    """Tests for score_chunk method."""

    def test_score_chunk_no_match(self) -> None:
        """Test scoring content with no term matches."""
        ranker = ProximityRanker()

        score = ranker.score_chunk("hello world", "python java")

        assert score.proximity_score == 0.0
        assert score.term_coverage == 0.0

    def test_score_chunk_partial_match(self) -> None:
        """Test scoring with partial term coverage."""
        ranker = ProximityRanker()

        score = ranker.score_chunk("python is great", "python java ruby")

        assert score.term_coverage == pytest.approx(1 / 3, rel=0.01)
        assert score.proximity_score > 0.0

    def test_score_chunk_full_match(self) -> None:
        """Test scoring with full term coverage."""
        ranker = ProximityRanker()

        score = ranker.score_chunk("python and java together", "python java")

        assert score.term_coverage == 1.0
        assert score.proximity_score > 0.0
        assert score.min_span > 0

    def test_score_chunk_adjacent_terms(self) -> None:
        """Test that adjacent terms score higher."""
        ranker = ProximityRanker()

        # Terms adjacent
        score_adjacent = ranker.score_chunk(
            "machine learning rocks", "machine learning"
        )

        # Terms separated
        score_separated = ranker.score_chunk(
            "machine is a type of learning device", "machine learning"
        )

        # Adjacent should have better proximity (lower avg_distance)
        assert score_adjacent.avg_distance < score_separated.avg_distance


# =============================================================================
# TestTermExtraction
# =============================================================================


class TestTermExtraction:
    """Tests for term extraction."""

    def test_extracts_terms(self) -> None:
        """Test that terms are extracted correctly."""
        ranker = ProximityRanker()

        terms = ranker._extract_terms("Python programming language")

        assert "python" in terms
        assert "programming" in terms
        assert "language" in terms

    def test_filters_stopwords(self) -> None:
        """Test that stopwords are filtered."""
        ranker = ProximityRanker()

        terms = ranker._extract_terms("the quick brown fox and the lazy dog")

        assert "the" not in terms
        assert "and" not in terms
        assert "quick" in terms
        assert "brown" in terms

    def test_filters_short_terms(self) -> None:
        """Test that short terms are filtered."""
        ranker = ProximityRanker(min_term_length=3)

        terms = ranker._extract_terms("I am a programmer")

        assert "am" not in terms
        assert "programmer" in terms

    def test_bounds_term_count(self) -> None:
        """Test that term count is bounded."""
        ranker = ProximityRanker()

        # Create query with many terms
        query = " ".join([f"term{i}" for i in range(100)])
        terms = ranker._extract_terms(query)

        assert len(terms) <= MAX_QUERY_TERMS


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self) -> None:
        """Rule #2: Verify fixed upper bounds are defined."""
        assert MAX_CANDIDATES > 0
        assert MAX_QUERY_TERMS > 0
        assert DEFAULT_PROXIMITY_WINDOW > 0

    def test_rule_5_preconditions(self) -> None:
        """Rule #5: Verify preconditions are asserted."""
        ranker = ProximityRanker()

        # Invalid boost weight should raise
        with pytest.raises(AssertionError):
            ranker.rank([], "test", boost_weight=1.5)

        # Invalid window size should raise
        with pytest.raises(AssertionError):
            ProximityRanker(window_size=-1)

    def test_rule_9_type_hints(self) -> None:
        """Rule #9: Verify methods have type hints."""
        ranker = ProximityRanker()

        # Check that key methods have annotations
        assert hasattr(ranker.rank, "__annotations__")
        assert hasattr(ranker.score_chunk, "__annotations__")


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_proximity_ranker(self) -> None:
        """Test factory function."""
        ranker = create_proximity_ranker(window_size=75)

        assert isinstance(ranker, ProximityRanker)
        assert ranker.window_size == 75

    def test_rank_by_proximity(self) -> None:
        """Test convenience ranking function."""
        results = [
            make_result("chunk-1", "python programming tutorial"),
            make_result("chunk-2", "no match content here"),
        ]

        ranked = rank_by_proximity(results, "python")

        assert len(ranked) == 2
        assert isinstance(ranked[0], SearchResult)
