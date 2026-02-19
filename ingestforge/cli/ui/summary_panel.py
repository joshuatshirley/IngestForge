"""Study Session Summary Panel.

Displays post-session stats with motivational feedback.
Implements SRS-003 from the backlog.

Follows Commandments:
- #4 (Small Functions): Each function has a single clear purpose
- #6 (Smallest Scope): Stats encapsulated in dataclass
- #7 (Check Parameters): Validates quiet flag
- #9 (Type Hints): Full type annotations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@dataclass
class SessionStats:
    """Statistics for a study session."""

    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None
    cards_reviewed: int = 0
    correct_count: int = 0
    ratings: Dict[int, int] = field(default_factory=dict)
    topics: List[str] = field(default_factory=list)

    def record_rating(self, rating: int) -> None:
        """Record a rating (0-5 scale).

        Args:
            rating: Rating value from 0 (hardest) to 5 (easiest)
        """
        self.ratings[rating] = self.ratings.get(rating, 0) + 1
        self.cards_reviewed += 1
        if rating >= 3:
            self.correct_count += 1

    def finish(self) -> None:
        """Mark session as finished."""
        self.end_time = datetime.now()

    @property
    def duration(self) -> timedelta:
        """Get session duration."""
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def accuracy(self) -> float:
        """Calculate accuracy percentage."""
        if self.cards_reviewed == 0:
            return 0.0
        return (self.correct_count / self.cards_reviewed) * 100

    @property
    def avg_time_per_card(self) -> float:
        """Average seconds per card."""
        if self.cards_reviewed == 0:
            return 0.0
        return self.duration.total_seconds() / self.cards_reviewed


def get_motivational_message(stats: SessionStats) -> str:
    """Generate motivational message based on performance.

    Args:
        stats: Session statistics

    Returns:
        Motivational message string
    """
    if stats.accuracy >= 90:
        return "Outstanding! You're mastering this material!"
    elif stats.accuracy >= 70:
        return "Great job! Keep up the consistent practice!"
    elif stats.accuracy >= 50:
        return "Good effort! Review the challenging cards again."
    else:
        return "No worries! Review is the path to mastery."


def format_duration(td: timedelta) -> str:
    """Format timedelta as human-readable string.

    Args:
        td: Timedelta to format

    Returns:
        Formatted duration string (e.g., "5m 23s")
    """
    total_seconds = int(td.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    if minutes > 0:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def show_session_summary(stats: SessionStats, quiet: bool = False) -> None:
    """Display study session summary panel.

    Args:
        stats: Session statistics
        quiet: If True, suppress output

    Example:
        >>> stats = SessionStats()
        >>> stats.record_rating(4)
        >>> stats.record_rating(3)
        >>> show_session_summary(stats)
        ┌─────────────────────────┐
        │ Session Complete        │
        ├─────────────────────────┤
        │ Cards Reviewed      2   │
        │ Correct             2/2 │
        │ Accuracy            100%│
        │ Duration            1s  │
        │ ...                     │
        └─────────────────────────┘
    """
    if quiet:
        return

    stats.finish()

    # Build summary table
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Cards Reviewed", str(stats.cards_reviewed))
    table.add_row("Correct", f"{stats.correct_count}/{stats.cards_reviewed}")
    table.add_row("Accuracy", f"{stats.accuracy:.1f}%")
    table.add_row("Duration", format_duration(stats.duration))
    table.add_row("Avg Time/Card", f"{stats.avg_time_per_card:.1f}s")

    # Rating distribution
    if stats.ratings:
        ratings_str = " ".join(f"[{r}]:{c}" for r, c in sorted(stats.ratings.items()))
        table.add_row("Ratings", ratings_str)

    # Get motivational message
    message = get_motivational_message(stats)

    # Determine border color based on performance
    border_color = "green" if stats.accuracy >= 70 else "yellow"

    # Create panel
    console.print()
    console.print(
        Panel(
            table,
            title="[bold]Session Complete[/bold]",
            subtitle=message,
            border_style=border_color,
            padding=(1, 2),
        )
    )
