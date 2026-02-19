"""Semantic answer scoring for flexible quiz grading (NLP-002.1).

This module provides embedding-based similarity scoring to accept
"near-match" answers (synonyms, paraphrases) in quizzes.

NASA JPL Commandments compliance:
- Rule #1: No deep nesting, early returns
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.study.grading_semantic import SemanticGrader

    grader = SemanticGrader()
    result = grader.grade_answer(
        expected="photosynthesis",
        actual="plant energy conversion",
    )
    if result.is_accepted:
        print(f"Accepted with {result.similarity:.0%} similarity")
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Similarity thresholds
DEFAULT_ACCEPT_THRESHOLD: float = 0.7  # Accept as correct
DEFAULT_CLOSE_THRESHOLD: float = 0.5  # Close enough for partial credit
MIN_THRESHOLD: float = 0.0
MAX_THRESHOLD: float = 1.0


class GradeLevel(Enum):
    """Semantic grading levels."""

    EXACT = "exact"  # Exact or near-exact match
    ACCEPTED = "accepted"  # Semantically equivalent (synonym)
    CLOSE = "close"  # Close but not quite
    WRONG = "wrong"  # Incorrect answer


@dataclass
class SemanticScore:
    """Result of semantic similarity scoring.

    Attributes:
        expected: The expected answer
        actual: The actual answer provided
        similarity: Cosine similarity (0-1)
        grade: Grading level
        is_exact: Whether it's an exact match
        is_accepted: Whether answer is accepted
        is_close: Whether answer is close
        feedback: Feedback message for user
    """

    expected: str
    actual: str
    similarity: float
    grade: GradeLevel
    is_exact: bool = False
    is_accepted: bool = False
    is_close: bool = False
    feedback: str = ""

    @property
    def is_correct(self) -> bool:
        """Whether the answer counts as correct."""
        return self.is_exact or self.is_accepted


@dataclass
class GradingThresholds:
    """Configurable thresholds for semantic grading.

    Attributes:
        accept_threshold: Minimum similarity for acceptance
        close_threshold: Minimum similarity for "close"
        exact_threshold: Minimum for exact match detection
    """

    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD
    close_threshold: float = DEFAULT_CLOSE_THRESHOLD
    exact_threshold: float = 0.95

    def validate(self) -> List[str]:
        """Validate threshold values.

        Returns:
            List of validation error messages
        """
        errors: List[str] = []

        if not MIN_THRESHOLD <= self.accept_threshold <= MAX_THRESHOLD:
            errors.append(
                f"accept_threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}"
            )

        if not MIN_THRESHOLD <= self.close_threshold <= MAX_THRESHOLD:
            errors.append(
                f"close_threshold must be between {MIN_THRESHOLD} and {MAX_THRESHOLD}"
            )

        if self.close_threshold > self.accept_threshold:
            errors.append("close_threshold cannot exceed accept_threshold")

        return errors


class SemanticGrader:
    """Grades answers using semantic similarity.

    Uses embedding-based similarity to accept synonyms and
    paraphrases as correct answers.

    Args:
        model_name: Sentence transformer model name
        thresholds: Custom grading thresholds
        case_sensitive: Whether to be case-sensitive

    Example:
        grader = SemanticGrader()

        # Exact match
        result = grader.grade_answer("Paris", "Paris")
        assert result.is_exact

        # Synonym acceptance
        result = grader.grade_answer(
            expected="automobile",
            actual="car",
        )
        assert result.is_accepted
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        thresholds: Optional[GradingThresholds] = None,
        case_sensitive: bool = False,
    ) -> None:
        """Initialize the semantic grader."""
        self.model_name = model_name
        self.thresholds = thresholds or GradingThresholds()
        self.case_sensitive = case_sensitive

        # Validate thresholds
        errors = self.thresholds.validate()
        if errors:
            raise ValueError(f"Invalid thresholds: {'; '.join(errors)}")

        # Lazy-load model
        self._model = None
        self._model_available: Optional[bool] = None

    @property
    def model(self) -> Optional[object]:
        """Lazy-load the sentence transformer model."""
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return self._model
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to load model: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if semantic grading is available."""
        if self._model_available is not None:
            return self._model_available

        self._model_available = self.model is not None
        return self._model_available

    def grade_answer(
        self,
        expected: str,
        actual: str,
    ) -> SemanticScore:
        """Grade an answer using semantic similarity.

        Args:
            expected: The expected/correct answer
            actual: The actual answer provided

        Returns:
            SemanticScore with similarity and grade
        """
        if not expected or not expected.strip():
            return self._create_wrong_score(expected, actual, "Empty expected answer")

        if not actual or not actual.strip():
            return self._create_wrong_score(expected, actual, "No answer provided")

        # Normalize text
        expected_norm = self._normalize(expected)
        actual_norm = self._normalize(actual)

        # Check for exact match first
        if expected_norm == actual_norm:
            return SemanticScore(
                expected=expected,
                actual=actual,
                similarity=1.0,
                grade=GradeLevel.EXACT,
                is_exact=True,
                is_accepted=True,
                feedback="Correct!",
            )

        # Use semantic similarity
        similarity = self._compute_similarity(expected, actual)

        return self._score_from_similarity(expected, actual, similarity)

    def grade_multiple(
        self,
        expected: str,
        alternatives: List[str],
    ) -> List[SemanticScore]:
        """Grade multiple alternative answers.

        Args:
            expected: The expected/correct answer
            alternatives: List of alternative answers

        Returns:
            List of SemanticScore for each alternative
        """
        return [self.grade_answer(expected, alt) for alt in alternatives]

    def find_best_match(
        self,
        expected: str,
        candidates: List[str],
    ) -> Tuple[str, SemanticScore]:
        """Find the best matching candidate.

        Args:
            expected: The expected answer
            candidates: List of candidate answers

        Returns:
            Tuple of (best candidate, score)
        """
        if not candidates:
            return "", self._create_wrong_score(expected, "", "No candidates")

        scores = self.grade_multiple(expected, candidates)
        best_idx = max(range(len(scores)), key=lambda i: scores[i].similarity)

        return candidates[best_idx], scores[best_idx]

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        normalized = text.strip()

        if not self.case_sensitive:
            normalized = normalized.lower()

        return normalized

    def _compute_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute semantic similarity between texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Cosine similarity (0-1)
        """
        if not self.is_available:
            # Fallback to basic string similarity
            return self._string_similarity(text1, text2)

        try:
            # Encode both texts
            embeddings = self.model.encode(
                [text1, text2],
                convert_to_numpy=True,
            )

            # Compute cosine similarity
            similarity = self._cosine_similarity(
                embeddings[0],
                embeddings[1],
            )

            return float(similarity)

        except Exception as e:
            logger.warning(f"Embedding failed, using fallback: {e}")
            return self._string_similarity(text1, text2)

    def _cosine_similarity(
        self,
        vec1,
        vec2,
    ) -> float:
        """Compute cosine similarity between vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0-1)
        """
        import numpy as np

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _string_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Fallback string similarity using character overlap.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        t1 = self._normalize(text1)
        t2 = self._normalize(text2)

        if not t1 or not t2:
            return 0.0

        # Simple Jaccard-like character similarity
        set1 = set(t1)
        set2 = set(t2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _score_from_similarity(
        self,
        expected: str,
        actual: str,
        similarity: float,
    ) -> SemanticScore:
        """Create score based on similarity value.

        Args:
            expected: Expected answer
            actual: Actual answer
            similarity: Computed similarity

        Returns:
            SemanticScore
        """
        thresholds = self.thresholds

        # Determine grade level
        if similarity >= thresholds.exact_threshold:
            return SemanticScore(
                expected=expected,
                actual=actual,
                similarity=similarity,
                grade=GradeLevel.EXACT,
                is_exact=True,
                is_accepted=True,
                feedback="Correct!",
            )

        if similarity >= thresholds.accept_threshold:
            return SemanticScore(
                expected=expected,
                actual=actual,
                similarity=similarity,
                grade=GradeLevel.ACCEPTED,
                is_exact=False,
                is_accepted=True,
                is_close=True,
                feedback=f"Accepted! (synonym of '{expected}')",
            )

        if similarity >= thresholds.close_threshold:
            return SemanticScore(
                expected=expected,
                actual=actual,
                similarity=similarity,
                grade=GradeLevel.CLOSE,
                is_exact=False,
                is_accepted=False,
                is_close=True,
                feedback=f"Close! The answer was '{expected}'",
            )

        return SemanticScore(
            expected=expected,
            actual=actual,
            similarity=similarity,
            grade=GradeLevel.WRONG,
            is_exact=False,
            is_accepted=False,
            is_close=False,
            feedback=f"Incorrect. The answer was '{expected}'",
        )

    def _create_wrong_score(
        self,
        expected: str,
        actual: str,
        reason: str,
    ) -> SemanticScore:
        """Create a wrong score with reason.

        Args:
            expected: Expected answer
            actual: Actual answer
            reason: Why it's wrong

        Returns:
            SemanticScore marked as wrong
        """
        return SemanticScore(
            expected=expected,
            actual=actual,
            similarity=0.0,
            grade=GradeLevel.WRONG,
            is_exact=False,
            is_accepted=False,
            is_close=False,
            feedback=reason,
        )


def grade_answer_semantic(
    expected: str,
    actual: str,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
) -> SemanticScore:
    """Convenience function to grade an answer.

    Args:
        expected: Expected answer
        actual: Actual answer
        accept_threshold: Minimum similarity for acceptance

    Returns:
        SemanticScore
    """
    thresholds = GradingThresholds(accept_threshold=accept_threshold)
    grader = SemanticGrader(thresholds=thresholds)
    return grader.grade_answer(expected, actual)
