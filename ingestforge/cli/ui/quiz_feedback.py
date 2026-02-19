"""Quiz feedback UI for semantic answer acceptance (NLP-002.2).

This module provides visual feedback for "close enough" answers,
displaying similarity scores and accepted synonyms.

NASA JPL Commandments compliance:
- Rule #1: No deep nesting, early returns
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.cli.ui.quiz_feedback import (
        show_answer_feedback,
        show_similarity_bar,
        get_feedback_panel,
    )

    show_answer_feedback(score, console)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from ingestforge.core.logging import get_logger

if TYPE_CHECKING:
    from ingestforge.study.grading_semantic import SemanticScore

logger = get_logger(__name__)


# Color scheme for grade levels
GRADE_COLORS = {
    "exact": "bold green",
    "accepted": "green",
    "close": "yellow",
    "wrong": "red",
}

# Emoji indicators (optional, disabled by default)
GRADE_EMOJI = {
    "exact": "✓",
    "accepted": "≈",
    "close": "~",
    "wrong": "✗",
}


@dataclass
class FeedbackStyle:
    """Styling options for feedback display.

    Attributes:
        show_similarity_bar: Show visual similarity bar
        show_percentage: Show numeric percentage
        show_expected: Show expected answer for wrong/close
        use_emoji: Use emoji indicators
        bar_width: Width of similarity bar
    """

    show_similarity_bar: bool = True
    show_percentage: bool = True
    show_expected: bool = True
    use_emoji: bool = False
    bar_width: int = 20


def get_grade_color(grade_value: str) -> str:
    """Get color for a grade level.

    Args:
        grade_value: Grade level value (exact, accepted, close, wrong)

    Returns:
        Rich color string
    """
    return GRADE_COLORS.get(grade_value, "white")


def get_grade_indicator(grade_value: str, use_emoji: bool = False) -> str:
    """Get visual indicator for a grade level.

    Args:
        grade_value: Grade level value
        use_emoji: Whether to use emoji

    Returns:
        Indicator string
    """
    if use_emoji:
        return GRADE_EMOJI.get(grade_value, "?")

    indicators = {
        "exact": "[CORRECT]",
        "accepted": "[ACCEPTED]",
        "close": "[CLOSE]",
        "wrong": "[WRONG]",
    }
    return indicators.get(grade_value, "[?]")


def create_similarity_bar(
    similarity: float,
    width: int = 20,
    filled_char: str = "█",
    empty_char: str = "░",
) -> str:
    """Create a text-based similarity bar.

    Args:
        similarity: Similarity value (0-1)
        width: Bar width in characters
        filled_char: Character for filled portion
        empty_char: Character for empty portion

    Returns:
        Bar string
    """
    # Clamp similarity to valid range
    similarity = max(0.0, min(1.0, similarity))

    filled = int(similarity * width)
    empty = width - filled

    return filled_char * filled + empty_char * empty


def get_similarity_color(similarity: float) -> str:
    """Get color based on similarity value.

    Args:
        similarity: Similarity value (0-1)

    Returns:
        Rich color string
    """
    if similarity >= 0.9:
        return "green"
    if similarity >= 0.7:
        return "green"
    if similarity >= 0.5:
        return "yellow"
    if similarity >= 0.3:
        return "orange1"
    return "red"


def format_similarity_display(
    similarity: float,
    style: Optional[FeedbackStyle] = None,
) -> Text:
    """Format similarity for display.

    Args:
        similarity: Similarity value (0-1)
        style: Display style options

    Returns:
        Rich Text object
    """
    style = style or FeedbackStyle()

    text = Text()
    color = get_similarity_color(similarity)

    if style.show_similarity_bar:
        bar = create_similarity_bar(similarity, style.bar_width)
        text.append(bar, style=color)
        text.append(" ")

    if style.show_percentage:
        percentage = f"{similarity * 100:.0f}%"
        text.append(percentage, style=f"bold {color}")

    return text


def get_feedback_panel(
    score: SemanticScore,
    style: Optional[FeedbackStyle] = None,
) -> Panel:
    """Create a Rich panel with answer feedback.

    Rule #1: Max 3 nesting levels via helper extraction.

    Args:
        score: SemanticScore from grading
        style: Display style options

    Returns:
        Rich Panel object
    """
    style = style or FeedbackStyle()

    # Build content
    content = Text()

    # Grade indicator and message
    grade_value = score.grade.value
    color = get_grade_color(grade_value)
    indicator = get_grade_indicator(grade_value, style.use_emoji)

    content.append(indicator, style=color)
    content.append(" ")

    # Add grade-specific message
    _append_feedback_message(content, score, style)

    # Add similarity bar for non-exact matches
    if not score.is_exact and (style.show_similarity_bar or style.show_percentage):
        content.append("\n\nSimilarity: ")
        content.append_text(format_similarity_display(score.similarity, style))

    # Determine panel border color
    border_style = color.replace("bold ", "")

    return Panel(
        content,
        title="Answer Feedback",
        border_style=border_style,
        padding=(0, 1),
    )


def _append_feedback_message(
    content: Text,
    score: SemanticScore,
    style: FeedbackStyle,
) -> None:
    """Append grade-specific feedback message to content.

    Rule #1: Extracted to reduce nesting in get_feedback_panel.

    Args:
        content: Text object to append to
        score: SemanticScore from grading
        style: Display style options
    """
    if score.is_exact:
        content.append("Correct!", style="bold green")
        return

    if score.is_accepted:
        content.append("Accepted as synonym!", style="green")
        content.append("\n\nYour answer: ", style="dim")
        content.append(score.actual, style="bold")
        content.append("\nExpected: ", style="dim")
        content.append(score.expected, style="italic")
        return

    # Close or wrong
    message = "Close, but not quite." if score.is_close else "Incorrect."
    message_style = "yellow" if score.is_close else "red"
    content.append(message, style=message_style)

    if not style.show_expected:
        return

    content.append("\n\nYour answer: ", style="dim")
    content.append(score.actual, style="bold red")
    content.append("\nCorrect answer: ", style="dim")
    content.append(score.expected, style="bold green")


def show_answer_feedback(
    score: SemanticScore,
    console: Optional[Console] = None,
    style: Optional[FeedbackStyle] = None,
) -> None:
    """Display answer feedback to console.

    Args:
        score: SemanticScore from grading
        console: Rich Console (creates default if not provided)
        style: Display style options
    """
    console = console or Console()
    panel = get_feedback_panel(score, style)
    console.print(panel)


def show_similarity_bar(
    similarity: float,
    label: str = "Similarity",
    console: Optional[Console] = None,
    width: int = 30,
) -> None:
    """Display a standalone similarity bar.

    Args:
        similarity: Similarity value (0-1)
        label: Label for the bar
        console: Rich Console
        width: Bar width
    """
    console = console or Console()
    color = get_similarity_color(similarity)

    bar = create_similarity_bar(similarity, width)
    percentage = f"{similarity * 100:.0f}%"

    text = Text()
    text.append(f"{label}: ", style="dim")
    text.append(bar, style=color)
    text.append(f" {percentage}", style=f"bold {color}")

    console.print(text)


def get_comparison_table(
    expected: str,
    actual: str,
    similarity: float,
) -> Table:
    """Create a comparison table showing expected vs actual.

    Args:
        expected: Expected answer
        actual: Actual answer
        similarity: Similarity score

    Returns:
        Rich Table
    """
    table = Table(show_header=True, header_style="bold")

    table.add_column("", style="dim", width=12)
    table.add_column("Answer", style="bold")

    table.add_row("Expected", expected, style="green")
    table.add_row("Your Answer", actual, style=get_similarity_color(similarity))

    return table


def format_score_summary(score: SemanticScore) -> str:
    """Format a single-line score summary.

    Args:
        score: SemanticScore

    Returns:
        Summary string
    """
    grade = score.grade.value.upper()
    similarity_pct = f"{score.similarity * 100:.0f}%"

    if score.is_exact:
        return f"{grade}: Exact match"
    if score.is_accepted:
        return f"{grade}: Synonym accepted ({similarity_pct} match)"
    if score.is_close:
        return f"{grade}: Close ({similarity_pct} match) - expected '{score.expected}'"

    return f"{grade}: Incorrect - expected '{score.expected}'"


def show_batch_results(
    scores: list[SemanticScore],
    console: Optional[Console] = None,
) -> None:
    """Display results for multiple answers.

    Args:
        scores: List of SemanticScore objects
        console: Rich Console
    """
    console = console or Console()

    # Create summary table
    table = Table(title="Answer Results", show_header=True)

    table.add_column("#", style="dim", width=4)
    table.add_column("Your Answer", style="bold")
    table.add_column("Expected", style="italic")
    table.add_column("Result", justify="center")
    table.add_column("Similarity", justify="right")

    for idx, score in enumerate(scores, 1):
        grade_color = get_grade_color(score.grade.value)
        indicator = get_grade_indicator(score.grade.value)

        similarity_str = f"{score.similarity * 100:.0f}%"
        similarity_color = get_similarity_color(score.similarity)

        table.add_row(
            str(idx),
            score.actual,
            score.expected,
            Text(indicator, style=grade_color),
            Text(similarity_str, style=similarity_color),
        )

    console.print(table)

    # Summary statistics
    correct = sum(1 for s in scores if s.is_correct)
    total = len(scores)
    accuracy = correct / total * 100 if total > 0 else 0

    console.print(f"\nResults: {correct}/{total} ({accuracy:.0f}%)")
