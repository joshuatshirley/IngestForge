"""Tests for grading module (QUIZ-001.2).

Tests the grading engine functionality:
- Question result tracking
- Score calculation
- Letter grade assignment
- Grade report generation
- Feedback messages
"""


from ingestforge.study.grading import (
    GradingEngine,
    GradeReport,
    QuestionResult,
    LetterGrade,
    calculate_grade,
    format_grade_display,
    get_grade_color,
    GRADE_THRESHOLDS,
)


class TestLetterGrade:
    """Test LetterGrade enum."""

    def test_all_grades_defined(self) -> None:
        """All letter grades should be defined."""
        expected = [
            "A+",
            "A",
            "A-",
            "B+",
            "B",
            "B-",
            "C+",
            "C",
            "C-",
            "D+",
            "D",
            "D-",
            "F",
        ]
        actual = [g.value for g in LetterGrade]
        assert actual == expected


class TestGradeThresholds:
    """Test grade threshold definitions."""

    def test_thresholds_descending(self) -> None:
        """Thresholds should be in descending order."""
        percentages = [t[0] for t in GRADE_THRESHOLDS]
        assert percentages == sorted(percentages, reverse=True)

    def test_lowest_threshold_zero(self) -> None:
        """Lowest threshold should be 0 (F grade)."""
        last_threshold = GRADE_THRESHOLDS[-1]
        assert last_threshold[0] == 0
        assert last_threshold[1] == LetterGrade.F


class TestQuestionResult:
    """Test QuestionResult dataclass."""

    def test_result_creation(self) -> None:
        """QuestionResult should be creatable."""
        result = QuestionResult(
            question_id="q1",
            question_text="What is 2+2?",
            correct_answer="4",
            user_answer="4",
            is_correct=True,
        )
        assert result.question_id == "q1"
        assert result.is_correct is True

    def test_result_defaults(self) -> None:
        """QuestionResult should have sensible defaults."""
        result = QuestionResult(
            question_id="q1",
            question_text="Q?",
            correct_answer="A",
            user_answer="B",
            is_correct=False,
        )
        assert result.time_taken == 0.0
        assert result.points_earned == 0.0
        assert result.points_possible == 1.0


class TestGradingEngine:
    """Test GradingEngine class."""

    def test_engine_creation(self) -> None:
        """Engine should initialize properly."""
        engine = GradingEngine()
        assert engine.points_per_question == 1.0

    def test_custom_points(self) -> None:
        """Engine should accept custom points."""
        engine = GradingEngine(points_per_question=2.0)
        assert engine.points_per_question == 2.0

    def test_minimum_points(self) -> None:
        """Points should be at least 0.1."""
        engine = GradingEngine(points_per_question=0)
        assert engine.points_per_question == 0.1


class TestAddResult:
    """Test add_result method."""

    def test_add_correct_answer(self) -> None:
        """Adding correct answer should be marked correct."""
        engine = GradingEngine()
        result = engine.add_result(
            question_id="q1",
            question_text="What is 2+2?",
            correct_answer="4",
            user_answer="4",
        )
        assert result.is_correct is True
        assert result.points_earned == 1.0

    def test_add_incorrect_answer(self) -> None:
        """Adding incorrect answer should be marked wrong."""
        engine = GradingEngine()
        result = engine.add_result(
            question_id="q1",
            question_text="What is 2+2?",
            correct_answer="4",
            user_answer="5",
        )
        assert result.is_correct is False
        assert result.points_earned == 0.0

    def test_case_insensitive_matching(self) -> None:
        """Answer matching should be case insensitive."""
        engine = GradingEngine()
        result = engine.add_result(
            question_id="q1",
            question_text="Capital of France?",
            correct_answer="Paris",
            user_answer="PARIS",
        )
        assert result.is_correct is True

    def test_whitespace_trimming(self) -> None:
        """Answer matching should trim whitespace."""
        engine = GradingEngine()
        result = engine.add_result(
            question_id="q1",
            question_text="Capital of France?",
            correct_answer="Paris",
            user_answer="  Paris  ",
        )
        assert result.is_correct is True


class TestPercentageCalculation:
    """Test percentage calculation."""

    def test_all_correct(self) -> None:
        """100% for all correct answers."""
        engine = GradingEngine()
        engine.add_result("q1", "Q?", "A", "A")
        engine.add_result("q2", "Q?", "B", "B")
        assert engine.calculate_percentage() == 100.0

    def test_all_incorrect(self) -> None:
        """0% for all incorrect answers."""
        engine = GradingEngine()
        engine.add_result("q1", "Q?", "A", "X")
        engine.add_result("q2", "Q?", "B", "Y")
        assert engine.calculate_percentage() == 0.0

    def test_partial_correct(self) -> None:
        """Correct percentage for partial."""
        engine = GradingEngine()
        engine.add_result("q1", "Q?", "A", "A")  # correct
        engine.add_result("q2", "Q?", "B", "X")  # wrong
        assert engine.calculate_percentage() == 50.0

    def test_empty_results(self) -> None:
        """0% for no results."""
        engine = GradingEngine()
        assert engine.calculate_percentage() == 0.0


class TestLetterGradeAssignment:
    """Test letter grade assignment."""

    def test_a_plus_grade(self) -> None:
        """97%+ should be A+."""
        engine = GradingEngine()
        assert engine.get_letter_grade(97) == LetterGrade.A_PLUS
        assert engine.get_letter_grade(100) == LetterGrade.A_PLUS

    def test_a_grade(self) -> None:
        """93-96% should be A."""
        engine = GradingEngine()
        assert engine.get_letter_grade(93) == LetterGrade.A
        assert engine.get_letter_grade(96) == LetterGrade.A

    def test_f_grade(self) -> None:
        """Below 60% should be F."""
        engine = GradingEngine()
        assert engine.get_letter_grade(59) == LetterGrade.F
        assert engine.get_letter_grade(0) == LetterGrade.F


class TestFeedbackGeneration:
    """Test feedback message generation."""

    def test_excellent_feedback(self) -> None:
        """90%+ should get excellent feedback."""
        engine = GradingEngine()
        feedback = engine.generate_feedback(95)
        assert "Excellent" in feedback or "mastered" in feedback

    def test_struggling_feedback(self) -> None:
        """Below 60% should get study suggestion."""
        engine = GradingEngine()
        feedback = engine.generate_feedback(45)
        assert "review" in feedback.lower()


class TestGradeReport:
    """Test grade report generation."""

    def test_report_structure(self) -> None:
        """Report should have all fields."""
        engine = GradingEngine()
        engine.add_result("q1", "Q?", "A", "A")
        engine.add_result("q2", "Q?", "B", "X")

        report = engine.get_report()

        assert report.total_questions == 2
        assert report.correct_count == 1
        assert report.incorrect_count == 1
        assert report.percentage == 50.0
        assert report.letter_grade == LetterGrade.F
        assert len(report.results) == 2
        assert report.feedback != ""

    def test_report_points(self) -> None:
        """Report should track points."""
        engine = GradingEngine(points_per_question=2.0)
        engine.add_result("q1", "Q?", "A", "A")  # 2 points
        engine.add_result("q2", "Q?", "B", "X")  # 0 points

        report = engine.get_report()
        assert report.points_earned == 2.0
        assert report.points_possible == 4.0


class TestReset:
    """Test engine reset."""

    def test_reset_clears_results(self) -> None:
        """reset() should clear all results."""
        engine = GradingEngine()
        engine.add_result("q1", "Q?", "A", "A")
        engine.reset()

        assert engine.calculate_percentage() == 0.0
        report = engine.get_report()
        assert report.total_questions == 0


class TestCalculateGrade:
    """Test calculate_grade utility."""

    def test_perfect_score(self) -> None:
        """Perfect score should be A+."""
        pct, grade = calculate_grade(10, 10)
        assert pct == 100.0
        assert grade == LetterGrade.A_PLUS

    def test_zero_score(self) -> None:
        """Zero score should be F."""
        pct, grade = calculate_grade(0, 10)
        assert pct == 0.0
        assert grade == LetterGrade.F

    def test_zero_total(self) -> None:
        """Zero total should return 0% F."""
        pct, grade = calculate_grade(0, 0)
        assert pct == 0.0
        assert grade == LetterGrade.F


class TestFormatGradeDisplay:
    """Test format_grade_display utility."""

    def test_format_includes_score(self) -> None:
        """Display should include score."""
        report = GradeReport(
            total_questions=10,
            correct_count=8,
            incorrect_count=2,
            percentage=80.0,
            letter_grade=LetterGrade.B_MINUS,
            points_earned=8.0,
            points_possible=10.0,
            feedback="Good job!",
        )
        display = format_grade_display(report)
        assert "8/10" in display
        assert "80.0%" in display
        assert "B-" in display


class TestGetGradeColor:
    """Test get_grade_color utility."""

    def test_a_grades_green(self) -> None:
        """A grades should be green."""
        assert get_grade_color(LetterGrade.A) == "green"
        assert get_grade_color(LetterGrade.A_PLUS) == "green"
        assert get_grade_color(LetterGrade.A_MINUS) == "green"

    def test_b_grades_cyan(self) -> None:
        """B grades should be cyan."""
        assert get_grade_color(LetterGrade.B) == "cyan"

    def test_f_grade_red(self) -> None:
        """F grade should be red."""
        assert get_grade_color(LetterGrade.F) == "red"
