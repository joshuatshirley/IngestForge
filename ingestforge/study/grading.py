"""Grading engine for exam and quiz sessions.

QUIZ-001.2: Deferred Grading Engine for end-of-quiz grading.

Calculates scores, percentages, and letter grades."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum


class LetterGrade(Enum):
    """Letter grade enumeration."""

    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D_PLUS = "D+"
    D = "D"
    D_MINUS = "D-"
    F = "F"


# Grade thresholds (percentage -> letter grade)
GRADE_THRESHOLDS: List[tuple[float, LetterGrade]] = [
    (97, LetterGrade.A_PLUS),
    (93, LetterGrade.A),
    (90, LetterGrade.A_MINUS),
    (87, LetterGrade.B_PLUS),
    (83, LetterGrade.B),
    (80, LetterGrade.B_MINUS),
    (77, LetterGrade.C_PLUS),
    (73, LetterGrade.C),
    (70, LetterGrade.C_MINUS),
    (67, LetterGrade.D_PLUS),
    (63, LetterGrade.D),
    (60, LetterGrade.D_MINUS),
    (0, LetterGrade.F),
]


@dataclass
class QuestionResult:
    """Result for a single question.

    Attributes:
        question_id: Unique identifier
        question_text: The question asked
        correct_answer: The correct answer
        user_answer: What the user answered
        is_correct: Whether the answer was correct
        time_taken: Time taken in seconds
        points_earned: Points earned for this question
        points_possible: Maximum points possible
    """

    question_id: str
    question_text: str
    correct_answer: str
    user_answer: str
    is_correct: bool
    time_taken: float = 0.0
    points_earned: float = 0.0
    points_possible: float = 1.0


@dataclass
class GradeReport:
    """Complete grade report for an exam.

    Attributes:
        total_questions: Number of questions
        correct_count: Number correct
        incorrect_count: Number incorrect
        percentage: Score as percentage
        letter_grade: Letter grade
        points_earned: Total points earned
        points_possible: Total points possible
        results: Individual question results
        feedback: Personalized feedback message
    """

    total_questions: int
    correct_count: int
    incorrect_count: int
    percentage: float
    letter_grade: LetterGrade
    points_earned: float
    points_possible: float
    results: List[QuestionResult] = field(default_factory=list)
    feedback: str = ""


class GradingEngine:
    """Engine for grading quiz and exam sessions.

    Supports deferred grading where all questions are answered
    before any feedback is given.
    """

    def __init__(self, points_per_question: float = 1.0) -> None:
        """Initialize grading engine.

        Args:
            points_per_question: Points per correct answer
        """
        self.points_per_question = max(0.1, points_per_question)
        self._results: List[QuestionResult] = []

    def add_result(
        self,
        question_id: str,
        question_text: str,
        correct_answer: str,
        user_answer: str,
        time_taken: float = 0.0,
    ) -> QuestionResult:
        """Add a graded question result.

        Args:
            question_id: Question identifier
            question_text: The question text
            correct_answer: Expected answer
            user_answer: User's answer
            time_taken: Time taken in seconds

        Returns:
            QuestionResult for this question
        """
        # Case-insensitive comparison, strip whitespace
        is_correct = self._check_answer(correct_answer, user_answer)
        points = self.points_per_question if is_correct else 0.0

        result = QuestionResult(
            question_id=question_id,
            question_text=question_text,
            correct_answer=correct_answer,
            user_answer=user_answer,
            is_correct=is_correct,
            time_taken=time_taken,
            points_earned=points,
            points_possible=self.points_per_question,
        )

        self._results.append(result)
        return result

    def _check_answer(self, correct: str, user: str) -> bool:
        """Check if user answer matches correct answer.

        Args:
            correct: Correct answer
            user: User's answer

        Returns:
            True if answer is correct
        """
        # Normalize both answers
        correct_norm = correct.strip().lower()
        user_norm = user.strip().lower()

        # Exact match after normalization
        return correct_norm == user_norm

    def calculate_percentage(self) -> float:
        """Calculate score as percentage.

        Returns:
            Score percentage (0-100)
        """
        if not self._results:
            return 0.0

        correct = sum(1 for r in self._results if r.is_correct)
        return (correct / len(self._results)) * 100

    def get_letter_grade(self, percentage: Optional[float] = None) -> LetterGrade:
        """Get letter grade for score.

        Args:
            percentage: Score percentage (uses calculated if None)

        Returns:
            Letter grade
        """
        if percentage is None:
            percentage = self.calculate_percentage()

        for threshold, grade in GRADE_THRESHOLDS:
            if percentage >= threshold:
                return grade

        return LetterGrade.F

    def generate_feedback(self, percentage: float) -> str:
        """Generate personalized feedback based on score.

        Args:
            percentage: Score percentage

        Returns:
            Feedback message
        """
        if percentage >= 90:
            return "Excellent work! You've mastered this material."
        if percentage >= 80:
            return "Great job! You have a solid understanding."
        if percentage >= 70:
            return "Good effort! Review the missed questions."
        if percentage >= 60:
            return "Keep practicing. Focus on areas you struggled with."
        return "Consider reviewing this material more thoroughly."

    def get_report(self) -> GradeReport:
        """Generate complete grade report.

        Returns:
            GradeReport with all statistics
        """
        total = len(self._results)
        correct = sum(1 for r in self._results if r.is_correct)
        incorrect = total - correct

        points_earned = sum(r.points_earned for r in self._results)
        points_possible = sum(r.points_possible for r in self._results)

        percentage = self.calculate_percentage()
        letter_grade = self.get_letter_grade(percentage)
        feedback = self.generate_feedback(percentage)

        return GradeReport(
            total_questions=total,
            correct_count=correct,
            incorrect_count=incorrect,
            percentage=round(percentage, 1),
            letter_grade=letter_grade,
            points_earned=points_earned,
            points_possible=points_possible,
            results=self._results.copy(),
            feedback=feedback,
        )

    def reset(self) -> None:
        """Clear all results for a new exam."""
        self._results.clear()


def calculate_grade(correct: int, total: int) -> tuple[float, LetterGrade]:
    """Simple utility to calculate grade.

    Args:
        correct: Number of correct answers
        total: Total number of questions

    Returns:
        Tuple of (percentage, letter_grade)
    """
    if total <= 0:
        return 0.0, LetterGrade.F

    percentage = (correct / total) * 100

    for threshold, grade in GRADE_THRESHOLDS:
        if percentage >= threshold:
            return round(percentage, 1), grade

    return round(percentage, 1), LetterGrade.F


def format_grade_display(report: GradeReport) -> str:
    """Format grade report for display.

    Args:
        report: GradeReport to format

    Returns:
        Formatted string for display
    """
    lines = [
        "=" * 40,
        "           EXAM RESULTS",
        "=" * 40,
        "",
        f"  Score: {report.correct_count}/{report.total_questions} "
        f"({report.percentage:.1f}%)",
        f"  Grade: {report.letter_grade.value}",
        "",
        f"  Points: {report.points_earned:.1f}/{report.points_possible:.1f}",
        "",
        f"  {report.feedback}",
        "",
        "=" * 40,
    ]

    return "\n".join(lines)


def get_grade_color(grade: LetterGrade) -> str:
    """Get Rich color for letter grade.

    Args:
        grade: Letter grade

    Returns:
        Rich color name
    """
    if grade in (LetterGrade.A_PLUS, LetterGrade.A, LetterGrade.A_MINUS):
        return "green"
    if grade in (LetterGrade.B_PLUS, LetterGrade.B, LetterGrade.B_MINUS):
        return "cyan"
    if grade in (LetterGrade.C_PLUS, LetterGrade.C, LetterGrade.C_MINUS):
        return "yellow"
    if grade in (LetterGrade.D_PLUS, LetterGrade.D, LetterGrade.D_MINUS):
        return "orange3"
    return "red"
