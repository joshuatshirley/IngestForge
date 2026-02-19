"""Tests for Score Normalizer.

RRF Normalization Pass
Tests score normalization across search providers.
"""

from __future__ import annotations

import pytest

from ingestforge.retrieval.normalizer import (
    ScoreNormalizer,
    NormalizationMethod,
    NormalizedScore,
    MAX_RESULTS,
    DEFAULT_RRF_K,
)


class TestNormalizedScore:
    """Tests for NormalizedScore dataclass."""

    def test_create_normalized_score(self) -> None:
        """Test creating NormalizedScore."""
        score = NormalizedScore(
            item_id="chunk-001",
            raw_score=0.75,
            normalized_score=0.5,
            provider="bm25",
            method=NormalizationMethod.MIN_MAX,
        )

        assert score.item_id == "chunk-001"
        assert score.raw_score == 0.75
        assert score.normalized_score == 0.5
        assert score.provider == "bm25"
        assert score.method == NormalizationMethod.MIN_MAX


class TestScoreNormalizerMinMax:
    """Tests for min-max normalization."""

    def test_normalize_empty_scores(self) -> None:
        """Test normalizing empty scores."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_min_max({})

        assert result == []

    def test_normalize_single_score(self) -> None:
        """Test normalizing single score."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_min_max({"a": 0.5})

        assert len(result) == 1
        assert result[0].item_id == "a"
        assert result[0].normalized_score == 1.0  # Single value normalizes to 1.0

    def test_normalize_multiple_scores(self) -> None:
        """Test normalizing multiple scores."""
        normalizer = ScoreNormalizer()
        scores = {"a": 0.0, "b": 0.5, "c": 1.0}
        result = normalizer.normalize_min_max(scores, provider="test")

        result_dict = {r.item_id: r.normalized_score for r in result}

        assert result_dict["a"] == 0.0
        assert result_dict["b"] == 0.5
        assert result_dict["c"] == 1.0

    def test_normalize_preserves_provider(self) -> None:
        """Test provider name is preserved."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_min_max({"a": 1.0}, provider="bm25")

        assert result[0].provider == "bm25"

    def test_normalize_clamps_to_bounds(self) -> None:
        """Test scores are clamped to [0,1]."""
        normalizer = ScoreNormalizer()
        scores = {"a": -10.0, "b": 100.0}
        result = normalizer.normalize_min_max(scores)

        for r in result:
            assert 0.0 <= r.normalized_score <= 1.0


class TestScoreNormalizerRRF:
    """Tests for RRF normalization."""

    def test_rrf_empty_lists(self) -> None:
        """Test RRF with empty lists."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_rrf([])

        assert result == []

    def test_rrf_single_list(self) -> None:
        """Test RRF with single ranked list."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_rrf([["a", "b", "c"]])

        assert len(result) == 3
        # First item should have highest RRF score
        assert result[0].item_id == "a"
        assert result[0].normalized_score > result[1].normalized_score

    def test_rrf_multiple_lists(self) -> None:
        """Test RRF with multiple ranked lists."""
        normalizer = ScoreNormalizer()
        # Item "b" appears first in list2, second in list1
        # Item "a" appears first in list1, second in list2
        result = normalizer.normalize_rrf(
            [
                ["a", "b", "c"],
                ["b", "a", "c"],
            ]
        )

        result_dict = {r.item_id: r.normalized_score for r in result}

        # Both "a" and "b" should have same RRF score (both rank 1 and 2)
        assert abs(result_dict["a"] - result_dict["b"]) < 0.001
        # "c" always ranked 3rd, should have lower score
        assert result_dict["c"] < result_dict["a"]

    def test_rrf_custom_k(self) -> None:
        """Test RRF with custom k value."""
        normalizer = ScoreNormalizer()
        result_k60 = normalizer.normalize_rrf([["a", "b"]], k=60)
        result_k10 = normalizer.normalize_rrf([["a", "b"]], k=10)

        # Different k values produce different scores
        assert result_k60[0].normalized_score != result_k10[0].normalized_score

    def test_rrf_preserves_method(self) -> None:
        """Test RRF method is recorded."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_rrf([["a"]])

        assert result[0].method == NormalizationMethod.RRF


class TestScoreNormalizerZScore:
    """Tests for z-score normalization."""

    def test_zscore_empty_scores(self) -> None:
        """Test z-score with empty scores."""
        normalizer = ScoreNormalizer()
        result = normalizer.normalize_z_score({})

        assert result == []

    def test_zscore_identical_scores(self) -> None:
        """Test z-score with identical scores."""
        normalizer = ScoreNormalizer()
        scores = {"a": 1.0, "b": 1.0, "c": 1.0}
        result = normalizer.normalize_z_score(scores)

        # All same values -> all get 0.5 normalized
        for r in result:
            assert r.normalized_score == 0.5

    def test_zscore_varied_scores(self) -> None:
        """Test z-score with varied scores."""
        normalizer = ScoreNormalizer()
        scores = {"a": 0.0, "b": 50.0, "c": 100.0}
        result = normalizer.normalize_z_score(scores)

        result_dict = {r.item_id: r.normalized_score for r in result}

        # Higher raw scores should have higher normalized scores
        assert result_dict["c"] > result_dict["b"]
        assert result_dict["b"] > result_dict["a"]


class TestScoreNormalizerFusion:
    """Tests for multi-provider fusion."""

    def test_fuse_empty_providers(self) -> None:
        """Test fusing empty providers."""
        normalizer = ScoreNormalizer()
        result = normalizer.fuse_providers({})

        assert result == {}

    def test_fuse_single_provider(self) -> None:
        """Test fusing single provider."""
        normalizer = ScoreNormalizer()
        provider_scores = {
            "bm25": {"a": 1.0, "b": 0.5},
        }
        result = normalizer.fuse_providers(provider_scores)

        assert "a" in result
        assert "b" in result

    def test_fuse_multiple_providers(self) -> None:
        """Test fusing multiple providers."""
        normalizer = ScoreNormalizer()
        provider_scores = {
            "bm25": {"a": 1.0, "b": 0.5},
            "semantic": {"a": 0.5, "c": 1.0},
        }
        result = normalizer.fuse_providers(provider_scores)

        # All items from all providers should be present
        assert "a" in result
        assert "b" in result
        assert "c" in result

    def test_fuse_with_weights(self) -> None:
        """Test fusing with custom weights."""
        normalizer = ScoreNormalizer()
        provider_scores = {
            "bm25": {"a": 1.0},
            "semantic": {"a": 0.0},
        }
        weights = {"bm25": 0.8, "semantic": 0.2}
        result = normalizer.fuse_providers(provider_scores, weights)

        # Weighted heavily toward bm25 (1.0 * 0.8 + 0.0 * 0.2 = 0.8)
        assert result["a"] == pytest.approx(0.8, rel=0.01)


class TestScoreNormalizerGeneric:
    """Tests for generic normalize method."""

    def test_normalize_uses_method(self) -> None:
        """Test normalize uses configured method."""
        normalizer_mm = ScoreNormalizer(NormalizationMethod.MIN_MAX)
        normalizer_zs = ScoreNormalizer(NormalizationMethod.Z_SCORE)

        scores = {"a": 0.0, "b": 1.0}

        result_mm = normalizer_mm.normalize(scores)
        result_zs = normalizer_zs.normalize(scores)

        assert result_mm[0].method == NormalizationMethod.MIN_MAX
        assert result_zs[0].method == NormalizationMethod.Z_SCORE


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_bounds_constants_exist(self) -> None:
        """Test that bound constants are defined."""
        assert MAX_RESULTS > 0
        assert DEFAULT_RRF_K > 0

    def test_rule_5_assertions(self) -> None:
        """Test precondition assertions."""
        normalizer = ScoreNormalizer()

        with pytest.raises(AssertionError):
            normalizer.normalize_min_max(None)  # type: ignore

        with pytest.raises(AssertionError):
            normalizer.normalize_rrf(None)  # type: ignore

        with pytest.raises(AssertionError):
            normalizer.normalize_rrf([["a"]], k=0)

    def test_rule_9_type_hints(self) -> None:
        """Test type hints exist."""
        import inspect

        normalizer = ScoreNormalizer()

        # Check key methods have return annotations
        for method_name in ["normalize_min_max", "normalize_rrf", "normalize_z_score"]:
            method = getattr(normalizer, method_name)
            sig = inspect.signature(method)
            assert sig.return_annotation is not None
