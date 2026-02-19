"""Rating prompt UI - Difficulty rating input with color-coded feedback.

SRS-002.1: Rating Input Handler with 1-4 scale, shortcuts, and color feedback.

Maps user-friendly 1-4 difficulty to SM-2 quality ratings:
- 1 = Again (forgot) → SM-2 quality 1
- 2 = Hard (barely recalled) → SM-2 quality 3
- 3 = Good (recalled with effort) → SM-2 quality 4
- 4 = Easy (instant recall) → SM-2 quality 5"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

logger = logging.getLogger(__name__)
# 10 attempts is reasonable limit for user input retries
MAX_RATING_ATTEMPTS = 10


@dataclass
class Rating:
    """Represents a difficulty rating with metadata.

    Attributes:
        value: User rating 1-4
        quality: Mapped SM-2 quality 0-5
        label: Display label
        color: Rich color for display
        description: Short description
    """

    value: int
    quality: int
    label: str
    color: str
    description: str


# Rating definitions with SM-2 quality mapping
RATINGS: Dict[int, Rating] = {
    1: Rating(
        value=1,
        quality=1,
        label="Again",
        color="red",
        description="Forgot - show again soon",
    ),
    2: Rating(
        value=2,
        quality=3,
        label="Hard",
        color="yellow",
        description="Barely recalled - needs work",
    ),
    3: Rating(
        value=3,
        quality=4,
        label="Good",
        color="green",
        description="Recalled with effort",
    ),
    4: Rating(
        value=4,
        quality=5,
        label="Easy",
        color="cyan",
        description="Instant recall",
    ),
}

# Keyboard shortcuts
SHORTCUTS: Dict[str, int] = {
    "1": 1,
    "a": 1,  # Again
    "2": 2,
    "h": 2,  # Hard
    "3": 3,
    "g": 3,  # Good
    "4": 4,
    "e": 4,  # Easy
}


def get_rating_prompt_panel(show_shortcuts: bool = True) -> Panel:
    """Create a panel showing rating options.

    Rule #1: Max 3 nesting levels via helper extraction.

    Args:
        show_shortcuts: Whether to show keyboard shortcuts

    Returns:
        Rich Panel with rating options
    """
    lines = []

    for value, rating in RATINGS.items():
        shortcut = _find_rating_shortcut(value) if show_shortcuts else ""
        line = _create_rating_line(value, rating, shortcut)
        lines.append(line)

    content = Text()
    for i, line in enumerate(lines):
        content.append_text(line)
        if i < len(lines) - 1:
            content.append("\n")

    return Panel(
        content,
        title="[bold]Rate Your Recall[/bold]",
        border_style="blue",
    )


def _find_rating_shortcut(rating_value: int) -> str:
    """Find letter shortcut for a rating value.

    Rule #1: Extracted to reduce nesting in get_rating_prompt_panel.

    Args:
        rating_value: Rating value (1-4)

    Returns:
        Formatted shortcut string or empty string
    """
    for key, val in SHORTCUTS.items():
        if val == rating_value and key.isalpha():
            return f" ({key})"
    return ""


def _create_rating_line(value: int, rating: Rating, shortcut: str) -> Text:
    """Create a formatted rating line.

    Rule #1: Extracted to reduce nesting in get_rating_prompt_panel.

    Args:
        value: Rating value
        rating: Rating object
        shortcut: Formatted shortcut string

    Returns:
        Rich Text object with formatted line
    """
    line = Text()
    line.append(f"  [{value}]", style=f"bold {rating.color}")
    line.append(f" {rating.label}", style=rating.color)
    line.append(shortcut, style="dim")
    line.append(f" - {rating.description}", style="dim")
    return line


def prompt_rating(console: Optional[Console] = None) -> Optional[Rating]:
    """Prompt user for difficulty rating.

    Supports both numeric (1-4) and letter shortcuts (a/h/g/e).

    Args:
        console: Rich Console instance (creates new if None)

    Returns:
        Rating object or None if cancelled or attempts exceeded
    """
    if console is None:
        console = Console()

    # Show rating options
    console.print()
    console.print(get_rating_prompt_panel())
    console.print()

    # Get input with retry (Rule #2: bounded loop)
    for attempt in range(MAX_RATING_ATTEMPTS):
        response = Prompt.ask(
            "[bold]Your rating[/bold]",
            choices=["1", "2", "3", "4", "a", "h", "g", "e", "q", "quit"],
            default="3",
        )

        # Handle quit
        if response.lower() in ("q", "quit"):
            return None

        # Parse rating
        rating_value = SHORTCUTS.get(response.lower())
        if rating_value is not None:
            return RATINGS[rating_value]

        # Shouldn't reach here due to choices, but handle gracefully
        console.print("[red]Invalid input. Try again.[/red]")
    logger.warning(
        f"Rating input attempts exceeded {MAX_RATING_ATTEMPTS}, using default rating"
    )
    console.print(
        f"[yellow]Max attempts ({MAX_RATING_ATTEMPTS}) reached. Using default rating: Good[/yellow]"
    )
    return RATINGS[3]  # Default to "Good" (middle rating)


def display_rating_feedback(
    rating: Rating,
    console: Optional[Console] = None,
) -> None:
    """Display color-coded feedback for a rating.

    Args:
        rating: The rating given
        console: Rich Console instance
    """
    if console is None:
        console = Console()

    console.print(f"[{rating.color}]✓ Rated: {rating.label}[/{rating.color}]")


def get_sm2_quality(rating_value: int) -> int:
    """Convert user rating (1-4) to SM-2 quality (0-5).

    Args:
        rating_value: User rating 1-4

    Returns:
        SM-2 quality rating 0-5
    """
    rating = RATINGS.get(rating_value)
    if rating is None:
        return 3  # Default to "passing" if invalid

    return rating.quality


def get_rating_stats_display(
    ratings: Dict[int, int],
    console: Optional[Console] = None,
) -> None:
    """Display session rating statistics.

    Args:
        ratings: Dictionary mapping rating values to counts
        console: Rich Console instance
    """
    if console is None:
        console = Console()

    if not ratings:
        console.print("[dim]No ratings recorded.[/dim]")
        return

    total = sum(ratings.values())

    console.print("\n[bold]Rating Breakdown:[/bold]")
    for value in range(1, 5):
        count = ratings.get(value, 0)
        rating = RATINGS[value]
        pct = (count / total * 100) if total > 0 else 0
        bar_length = int(pct / 5)  # Scale to 20 chars max
        bar = "█" * bar_length

        console.print(
            f"  [{rating.color}]{rating.label:6}[/{rating.color}] "
            f"{count:3d} ({pct:5.1f}%) [{rating.color}]{bar}[/{rating.color}]"
        )
