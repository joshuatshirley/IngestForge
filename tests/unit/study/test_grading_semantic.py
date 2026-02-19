"""Tests for grading_semantic module (NLP-002.1).

Tests the semantic answer scorer:
- GradingThresholds validation
- SemanticScore dataclass
- SemanticGrader with mock embeddings
- Fallback string similarity
"""

import pytest
import numpy as np
from unittest.mock import Mock

from ingestforge.study.grading_semantic import (
    GradeLevel,
    SemanticScore,
    GradingThresholds,
    SemanticGrader,
    grade_answer_semantic,
)


class TestGradeLevel:
    """Test GradeLevel enum."""

    def test_all_levels_defined(self) -> None:
        """All grade levels should be defined."""
        assert GradeLevel.EXACT
        assert GradeLevel.ACCEPTED
        assert GradeLevel.CLOSE
        assert GradeLevel.WRONG


class TestSemanticScore:
    """Test SemanticScore dataclass."""

    def test_exact_match_is_correct(self) -> None:
        """Exact match should be correct."""
        score = SemanticScore(
            expected="Paris",
            actual="Paris",
            similarity=1.0,
            grade=GradeLevel.EXACT,
            is_exact=True,
            is_accepted=True,
        )

        assert score.is_correct
        assert score.is_exact

    def test_accepted_is_correct(self) -> None:
        """Accepted synonym should be correct."""
        score = SemanticScore(
            expected="automobile",
            actual="car",
            similarity=0.85,
            grade=GradeLevel.ACCEPTED,
            is_accepted=True,
        )

        assert score.is_correct
        assert not score.is_exact

    def test_close_not_correct(self) -> None:
        """Close answer should not count as correct."""
        score = SemanticScore(
            expected="photosynthesis",
            actual="respiration",
            similarity=0.55,
            grade=GradeLevel.CLOSE,
            is_close=True,
        )

        assert not score.is_correct
        assert score.is_close

    def test_wrong_not_correct(self) -> None:
        """Wrong answer should not be correct."""
        score = SemanticScore(
            expected="Paris",
            actual="London",
            similarity=0.3,
            grade=GradeLevel.WRONG,
        )

        assert not score.is_correct
        assert not score.is_close


class TestGradingThresholds:
    """Test GradingThresholds validation."""

    def test_default_thresholds(self) -> None:
        """Default thresholds should be valid."""
        thresholds = GradingThresholds()

        errors = thresholds.validate()
        assert len(errors) == 0

    def test_accept_threshold_out_of_range(self) -> None:
        """Should reject threshold > 1.0."""
        thresholds = GradingThresholds(accept_threshold=1.5)

        errors = thresholds.validate()
        assert len(errors) > 0
        assert "accept_threshold" in errors[0]

    def test_close_threshold_exceeds_accept(self) -> None:
        """Should reject close > accept."""
        thresholds = GradingThresholds(
            accept_threshold=0.5,
            close_threshold=0.8,
        )

        errors = thresholds.validate()
        assert len(errors) > 0
        assert "close_threshold" in errors[0]

    def test_custom_valid_thresholds(self) -> None:
        """Custom valid thresholds should pass."""
        thresholds = GradingThresholds(
            accept_threshold=0.8,
            close_threshold=0.6,
        )

        errors = thresholds.validate()
        assert len(errors) == 0


class TestSemanticGraderInit:
    """Test SemanticGrader initialization."""

    def test_default_init(self) -> None:
        """Should initialize with defaults."""
        grader = SemanticGrader()

        assert grader.model_name == "all-MiniLM-L6-v2"
        assert grader.case_sensitive is False

    def test_custom_model(self) -> None:
        """Should accept custom model name."""
        grader = SemanticGrader(model_name="all-mpnet-base-v2")

        assert grader.model_name == "all-mpnet-base-v2"

    def test_invalid_thresholds_rejected(self) -> None:
        """Should reject invalid thresholds."""
        thresholds = GradingThresholds(accept_threshold=1.5)

        with pytest.raises(ValueError, match="Invalid thresholds"):
            SemanticGrader(thresholds=thresholds)

    def test_case_sensitive_option(self) -> None:
        """Should accept case_sensitive option."""
        grader = SemanticGrader(case_sensitive=True)

        assert grader.case_sensitive is True


class TestSemanticGraderExactMatch:
    """Test exact match detection."""

    def test_exact_match_same_case(self) -> None:
        """Identical strings should be exact match."""
        grader = SemanticGrader()
        score = grader.grade_answer("Paris", "Paris")

        assert score.is_exact
        assert score.similarity == 1.0

    def test_exact_match_different_case(self) -> None:
        """Case-insensitive exact match."""
        grader = SemanticGrader(case_sensitive=False)
        score = grader.grade_answer("Paris", "paris")

        assert score.is_exact
        assert score.similarity == 1.0

    def test_case_sensitive_no_exact_match(self) -> None:
        """Case-sensitive mode should not be exact match for different case."""
        grader = SemanticGrader(case_sensitive=True)
        grader._model_available = False  # Force fallback to test case sensitivity

        score = grader.grade_answer("Paris", "paris")

        # Should NOT be exact match when case differs in case-sensitive mode
        assert not score.is_exact
        # But may still have some similarity via fallback
        assert score.similarity < 1.0

    def test_whitespace_handling(self) -> None:
        """Should handle whitespace in answers."""
        grader = SemanticGrader()
        score = grader.grade_answer("  Paris  ", "Paris")

        assert score.is_exact


class TestSemanticGraderWithMockModel:
    """Test grading with mocked sentence transformer."""

    def test_synonym_acceptance(self) -> None:
        """Should accept synonyms with high similarity."""
        mock_model = Mock()
        # Return embeddings that are very similar
        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],  # "automobile"
                [0.95, 0.1, 0.0],  # "car" - very similar
            ]
        )

        grader = SemanticGrader()
        grader._model = mock_model
        grader._model_available = True

        score = grader.grade_answer("automobile", "car")

        # Should be accepted due to high similarity
        assert score.similarity > 0.9

    def test_close_answer(self) -> None:
        """Should mark moderately similar as close."""
        mock_model = Mock()
        # Return embeddings that are moderately similar
        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.6, 0.8, 0.0],  # Moderate similarity
            ]
        )

        grader = SemanticGrader()
        grader._model = mock_model
        grader._model_available = True

        score = grader.grade_answer("photosynthesis", "plant process")

        assert 0.5 < score.similarity < 0.9

    def test_wrong_answer(self) -> None:
        """Should mark dissimilar as wrong."""
        mock_model = Mock()
        # Return orthogonal embeddings
        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],  # Orthogonal = 0 similarity
            ]
        )

        grader = SemanticGrader()
        grader._model = mock_model
        grader._model_available = True

        score = grader.grade_answer("Paris", "banana")

        assert score.similarity < 0.5
        assert score.grade == GradeLevel.WRONG


class TestSemanticGraderFallback:
    """Test fallback when model unavailable."""

    def test_fallback_exact_match(self) -> None:
        """Fallback should still detect exact matches."""
        grader = SemanticGrader()
        grader._model_available = False

        score = grader.grade_answer("Paris", "Paris")

        assert score.is_exact

    def test_fallback_string_similarity(self) -> None:
        """Fallback should use string similarity."""
        grader = SemanticGrader()
        grader._model_available = False

        # Similar strings should have some similarity
        score = grader.grade_answer("automobile", "automotive")

        # Should be similar due to shared characters
        assert score.similarity > 0


class TestSemanticGraderValidation:
    """Test input validation."""

    def test_empty_expected(self) -> None:
        """Should handle empty expected answer."""
        grader = SemanticGrader()
        score = grader.grade_answer("", "something")

        assert score.grade == GradeLevel.WRONG
        assert "Empty" in score.feedback

    def test_empty_actual(self) -> None:
        """Should handle empty actual answer."""
        grader = SemanticGrader()
        score = grader.grade_answer("Paris", "")

        assert score.grade == GradeLevel.WRONG
        assert "No answer" in score.feedback

    def test_whitespace_only_expected(self) -> None:
        """Should handle whitespace-only expected."""
        grader = SemanticGrader()
        score = grader.grade_answer("   ", "Paris")

        assert score.grade == GradeLevel.WRONG


class TestSemanticGraderMultiple:
    """Test multiple answer grading."""

    def test_grade_multiple(self) -> None:
        """Should grade multiple alternatives."""
        grader = SemanticGrader()
        grader._model_available = False

        scores = grader.grade_multiple(
            "Paris",
            ["Paris", "London", "Berlin"],
        )

        assert len(scores) == 3
        assert scores[0].is_exact  # Paris == Paris

    def test_find_best_match(self) -> None:
        """Should find best matching candidate."""
        grader = SemanticGrader()
        grader._model_available = False

        best, score = grader.find_best_match(
            "Paris",
            ["London", "Paris", "Berlin"],
        )

        assert best == "Paris"
        assert score.is_exact

    def test_find_best_match_empty(self) -> None:
        """Should handle empty candidates."""
        grader = SemanticGrader()

        best, score = grader.find_best_match("Paris", [])

        assert best == ""
        assert score.grade == GradeLevel.WRONG


class TestCosineSimilarity:
    """Test cosine similarity calculation."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        grader = SemanticGrader()

        vec = np.array([1.0, 2.0, 3.0])
        similarity = grader._cosine_similarity(vec, vec)

        assert similarity == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        grader = SemanticGrader()

        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])

        similarity = grader._cosine_similarity(vec1, vec2)

        assert similarity == pytest.approx(0.0)

    def test_zero_vector(self) -> None:
        """Zero vector should return 0.0 similarity."""
        grader = SemanticGrader()

        vec1 = np.array([1.0, 2.0, 3.0])
        vec2 = np.array([0.0, 0.0, 0.0])

        similarity = grader._cosine_similarity(vec1, vec2)

        assert similarity == 0.0


class TestConvenienceFunction:
    """Test grade_answer_semantic convenience function."""

    def test_grade_answer_semantic(self) -> None:
        """Should work as standalone function."""
        score = grade_answer_semantic("Paris", "Paris")

        assert score.is_exact

    def test_custom_threshold(self) -> None:
        """Should accept custom threshold."""
        score = grade_answer_semantic(
            "Paris",
            "Paris",
            accept_threshold=0.9,
        )

        assert score.is_exact


class TestFeedbackMessages:
    """Test feedback message generation."""

    def test_exact_feedback(self) -> None:
        """Exact match should have positive feedback."""
        grader = SemanticGrader()
        score = grader.grade_answer("Paris", "Paris")

        assert "Correct" in score.feedback

    def test_accepted_feedback_includes_synonym(self) -> None:
        """Accepted feedback should mention synonym."""
        mock_model = Mock()
        # Return embeddings with similarity ~0.85 (above accept threshold)
        mock_model.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.85, 0.53, 0.0],  # Cosine similarity ≈ 0.85
            ]
        )

        grader = SemanticGrader()
        grader._model = mock_model
        grader._model_available = True

        score = grader.grade_answer("automobile", "car")

        # Verify similarity is in acceptance range
        assert score.similarity >= 0.7  # Accept threshold
        assert score.similarity < 0.95  # Below exact threshold

        # Score should be accepted
        assert score.is_accepted or score.is_exact
        # Feedback should indicate acceptance
        if score.is_accepted and not score.is_exact:
            assert "Accepted" in score.feedback

    def test_wrong_feedback_shows_expected(self) -> None:
        """Wrong answer feedback should show expected."""
        grader = SemanticGrader()
        grader._model_available = False

        score = grader.grade_answer("Paris", "xyz123")

        assert "Paris" in score.feedback
