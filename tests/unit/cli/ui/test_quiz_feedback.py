"""Tests for quiz_feedback module (NLP-002.2).

Tests the quiz feedback UI:
- Similarity bar creation
- Grade colors and indicators
- Feedback panel generation
- Batch results display
"""

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.ui.quiz_feedback import (
    FeedbackStyle,
    get_grade_color,
    get_grade_indicator,
    create_similarity_bar,
    get_similarity_color,
    format_similarity_display,
    get_feedback_panel,
    show_answer_feedback,
    show_similarity_bar,
    get_comparison_table,
    format_score_summary,
    show_batch_results,
    GRADE_COLORS,
    GRADE_EMOJI,
)
from ingestforge.study.grading_semantic import (
    SemanticScore,
    GradeLevel,
)


class TestFeedbackStyle:
    """Test FeedbackStyle dataclass."""

    def test_default_style(self) -> None:
        """Default style should have sensible defaults."""
        style = FeedbackStyle()

        assert style.show_similarity_bar is True
        assert style.show_percentage is True
        assert style.show_expected is True
        assert style.use_emoji is False

    def test_custom_style(self) -> None:
        """Should accept custom values."""
        style = FeedbackStyle(
            show_similarity_bar=False,
            use_emoji=True,
            bar_width=10,
        )

        assert style.show_similarity_bar is False
        assert style.use_emoji is True
        assert style.bar_width == 10


class TestGradeColors:
    """Test grade color mapping."""

    def test_all_grades_have_colors(self) -> None:
        """All grade levels should have colors."""
        for grade in ["exact", "accepted", "close", "wrong"]:
            assert grade in GRADE_COLORS

    def test_get_grade_color(self) -> None:
        """Should return correct colors."""
        assert "green" in get_grade_color("exact")
        assert "green" in get_grade_color("accepted")
        assert "yellow" in get_grade_color("close")
        assert "red" in get_grade_color("wrong")

    def test_unknown_grade_default(self) -> None:
        """Unknown grade should return white."""
        color = get_grade_color("unknown")
        assert color == "white"


class TestGradeIndicators:
    """Test grade indicator generation."""

    def test_text_indicators(self) -> None:
        """Should return text indicators by default."""
        assert "[CORRECT]" in get_grade_indicator("exact")
        assert "[ACCEPTED]" in get_grade_indicator("accepted")
        assert "[CLOSE]" in get_grade_indicator("close")
        assert "[WRONG]" in get_grade_indicator("wrong")

    def test_emoji_indicators(self) -> None:
        """Should return emoji when requested."""
        indicator = get_grade_indicator("exact", use_emoji=True)
        assert indicator == GRADE_EMOJI["exact"]


class TestSimilarityBar:
    """Test similarity bar creation."""

    def test_full_bar(self) -> None:
        """Similarity 1.0 should be fully filled."""
        bar = create_similarity_bar(1.0, width=10)

        assert len(bar) == 10
        assert "░" not in bar

    def test_empty_bar(self) -> None:
        """Similarity 0.0 should be empty."""
        bar = create_similarity_bar(0.0, width=10)

        assert len(bar) == 10
        assert "█" not in bar

    def test_half_bar(self) -> None:
        """Similarity 0.5 should be half filled."""
        bar = create_similarity_bar(0.5, width=10)

        assert bar.count("█") == 5
        assert bar.count("░") == 5

    def test_custom_chars(self) -> None:
        """Should use custom characters."""
        bar = create_similarity_bar(
            0.5,
            width=10,
            filled_char="#",
            empty_char="-",
        )

        assert "#" in bar
        assert "-" in bar

    def test_clamps_similarity(self) -> None:
        """Should clamp out-of-range values."""
        bar_high = create_similarity_bar(1.5, width=10)
        bar_low = create_similarity_bar(-0.5, width=10)

        # Should clamp to valid range
        assert len(bar_high) == 10
        assert len(bar_low) == 10


class TestSimilarityColor:
    """Test similarity color selection."""

    def test_high_similarity_green(self) -> None:
        """High similarity should be green."""
        assert "green" in get_similarity_color(0.95)
        assert "green" in get_similarity_color(0.8)

    def test_medium_similarity_yellow(self) -> None:
        """Medium similarity should be yellow."""
        assert "yellow" in get_similarity_color(0.6)

    def test_low_similarity_red(self) -> None:
        """Low similarity should be red."""
        assert "red" in get_similarity_color(0.2)


class TestFormatSimilarityDisplay:
    """Test similarity display formatting."""

    def test_includes_bar(self) -> None:
        """Should include bar by default."""
        style = FeedbackStyle(show_similarity_bar=True)
        text = format_similarity_display(0.8, style)

        # Should have bar characters
        plain = text.plain
        assert "█" in plain or "░" in plain

    def test_includes_percentage(self) -> None:
        """Should include percentage by default."""
        style = FeedbackStyle(show_percentage=True)
        text = format_similarity_display(0.8, style)

        assert "80%" in text.plain

    def test_bar_only(self) -> None:
        """Should show only bar when percentage disabled."""
        style = FeedbackStyle(
            show_similarity_bar=True,
            show_percentage=False,
        )
        text = format_similarity_display(0.8, style)

        assert "80%" not in text.plain


class TestGetFeedbackPanel:
    """Test feedback panel generation."""

    def test_exact_match_panel(self) -> None:
        """Exact match should show correct panel."""
        score = SemanticScore(
            expected="Paris",
            actual="Paris",
            similarity=1.0,
            grade=GradeLevel.EXACT,
            is_exact=True,
            is_accepted=True,
        )

        panel = get_feedback_panel(score)

        assert isinstance(panel, Panel)
        assert "Correct" in panel.renderable.plain

    def test_accepted_panel_shows_both(self) -> None:
        """Accepted panel should show both answers."""
        score = SemanticScore(
            expected="automobile",
            actual="car",
            similarity=0.85,
            grade=GradeLevel.ACCEPTED,
            is_accepted=True,
        )

        panel = get_feedback_panel(score)

        plain = panel.renderable.plain
        assert "automobile" in plain
        assert "car" in plain

    def test_close_panel_shows_expected(self) -> None:
        """Close panel should show expected answer."""
        score = SemanticScore(
            expected="photosynthesis",
            actual="respiration",
            similarity=0.55,
            grade=GradeLevel.CLOSE,
            is_close=True,
        )

        style = FeedbackStyle(show_expected=True)
        panel = get_feedback_panel(score, style)

        assert "photosynthesis" in panel.renderable.plain

    def test_wrong_panel_shows_both(self) -> None:
        """Wrong panel should show both answers."""
        score = SemanticScore(
            expected="Paris",
            actual="London",
            similarity=0.3,
            grade=GradeLevel.WRONG,
        )

        panel = get_feedback_panel(score)

        plain = panel.renderable.plain
        assert "Paris" in plain
        assert "London" in plain

    def test_no_similarity_bar_for_exact(self) -> None:
        """Exact match should not show similarity bar."""
        score = SemanticScore(
            expected="Paris",
            actual="Paris",
            similarity=1.0,
            grade=GradeLevel.EXACT,
            is_exact=True,
        )

        panel = get_feedback_panel(score)

        # Should not have similarity bar for exact matches
        # (bar is only shown for non-exact)
        assert "░" not in panel.renderable.plain


class TestShowAnswerFeedback:
    """Test show_answer_feedback function."""

    def test_show_feedback_no_error(self) -> None:
        """Should display without error."""
        score = SemanticScore(
            expected="Paris",
            actual="Paris",
            similarity=1.0,
            grade=GradeLevel.EXACT,
            is_exact=True,
        )

        console = Console(force_terminal=True, width=80, record=True)
        show_answer_feedback(score, console)

        output = console.export_text()
        assert "Correct" in output


class TestShowSimilarityBar:
    """Test show_similarity_bar function."""

    def test_show_bar(self) -> None:
        """Should display similarity bar."""
        console = Console(force_terminal=True, width=80, record=True)
        show_similarity_bar(0.75, label="Match", console=console)

        output = console.export_text()
        assert "Match" in output
        assert "75%" in output


class TestGetComparisonTable:
    """Test comparison table generation."""

    def test_comparison_table(self) -> None:
        """Should create comparison table."""
        table = get_comparison_table(
            expected="Paris",
            actual="London",
            similarity=0.3,
        )

        assert isinstance(table, Table)


class TestFormatScoreSummary:
    """Test score summary formatting."""

    def test_exact_summary(self) -> None:
        """Exact match summary should mention exact."""
        score = SemanticScore(
            expected="Paris",
            actual="Paris",
            similarity=1.0,
            grade=GradeLevel.EXACT,
            is_exact=True,
        )

        summary = format_score_summary(score)

        assert "EXACT" in summary
        assert "Exact match" in summary

    def test_accepted_summary(self) -> None:
        """Accepted summary should mention synonym."""
        score = SemanticScore(
            expected="automobile",
            actual="car",
            similarity=0.85,
            grade=GradeLevel.ACCEPTED,
            is_accepted=True,
        )

        summary = format_score_summary(score)

        assert "ACCEPTED" in summary
        assert "85%" in summary

    def test_close_summary(self) -> None:
        """Close summary should show expected."""
        score = SemanticScore(
            expected="photosynthesis",
            actual="respiration",
            similarity=0.55,
            grade=GradeLevel.CLOSE,
            is_close=True,
        )

        summary = format_score_summary(score)

        assert "CLOSE" in summary
        assert "photosynthesis" in summary

    def test_wrong_summary(self) -> None:
        """Wrong summary should show expected."""
        score = SemanticScore(
            expected="Paris",
            actual="London",
            similarity=0.3,
            grade=GradeLevel.WRONG,
        )

        summary = format_score_summary(score)

        assert "WRONG" in summary
        assert "Paris" in summary


class TestShowBatchResults:
    """Test batch results display."""

    def test_batch_results(self) -> None:
        """Should display batch results table."""
        scores = [
            SemanticScore(
                expected="Paris",
                actual="Paris",
                similarity=1.0,
                grade=GradeLevel.EXACT,
                is_exact=True,
                is_accepted=True,
            ),
            SemanticScore(
                expected="London",
                actual="Berlin",
                similarity=0.3,
                grade=GradeLevel.WRONG,
            ),
        ]

        console = Console(force_terminal=True, width=80, record=True)
        show_batch_results(scores, console)

        output = console.export_text()
        assert "Paris" in output
        assert "London" in output
        assert "1/2" in output  # 1 correct out of 2
