"""Study progress dashboard TUI.

UX-019.2: Rich-based visual dashboard for study progress.

Displays:
- Overview cards (due, reviewed, streak)
- Mastery distribution chart
- Topic breakdown table
- Performance trends"""

from __future__ import annotations

from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.text import Text

from ingestforge.study.stats import (
    StudyStats,
    MasteryLevel,
    get_study_stats,
)

# Colors for mastery levels
MASTERY_COLORS = {
    MasteryLevel.NEW: "white",
    MasteryLevel.LEARNING: "yellow",
    MasteryLevel.REVIEWING: "cyan",
    MasteryLevel.MATURE: "green",
}


def create_overview_panel(stats: StudyStats) -> Panel:
    """Create overview panel with key metrics.

    Args:
        stats: Study statistics

    Returns:
        Rich Panel with overview
    """
    lines = []

    # Due today
    due_color = "red" if stats.due_today > 0 else "green"
    lines.append(
        Text.assemble(
            ("📚 Due Today: ", "bold"),
            (str(stats.due_today), f"bold {due_color}"),
        )
    )

    # Reviewed today
    lines.append(
        Text.assemble(
            ("✅ Reviewed: ", "bold"),
            (str(stats.reviewed_today), "bold green"),
        )
    )

    # Time spent
    time_str = f"{stats.time_today_minutes:.0f} min"
    lines.append(
        Text.assemble(
            ("⏱️  Time: ", "bold"),
            (time_str, "cyan"),
        )
    )

    # Streak
    streak_color = "green" if stats.streak_days > 0 else "dim"
    streak_emoji = "🔥" if stats.streak_days >= 7 else "📅"
    lines.append(
        Text.assemble(
            (f"{streak_emoji} Streak: ", "bold"),
            (f"{stats.streak_days} days", streak_color),
        )
    )

    content = Text("\n").join(lines)

    return Panel(
        content,
        title="[bold cyan]Today's Progress[/bold cyan]",
        border_style="cyan",
    )


def create_accuracy_panel(stats: StudyStats) -> Panel:
    """Create accuracy panel.

    Args:
        stats: Study statistics

    Returns:
        Rich Panel with accuracy info
    """
    accuracy = stats.average_accuracy

    # Determine color based on accuracy
    if accuracy >= 80:
        color = "green"
        emoji = "🎯"
    elif accuracy >= 60:
        color = "yellow"
        emoji = "📊"
    else:
        color = "red"
        emoji = "📈"

    content = Text.assemble(
        (f"{emoji} ", ""),
        (f"{accuracy:.1f}%", f"bold {color}"),
        ("\n", ""),
        ("lifetime accuracy", "dim"),
    )

    return Panel(
        content,
        title="[bold]Accuracy[/bold]",
        border_style=color,
    )


def create_cards_panel(stats: StudyStats) -> Panel:
    """Create total cards panel.

    Args:
        stats: Study statistics

    Returns:
        Rich Panel with card count
    """
    content = Text.assemble(
        ("📇 ", ""),
        (str(stats.total_cards), "bold cyan"),
        ("\n", ""),
        ("total cards", "dim"),
    )

    return Panel(
        content,
        title="[bold]Library[/bold]",
        border_style="blue",
    )


def create_mastery_chart(stats: StudyStats) -> Panel:
    """Create mastery distribution chart.

    Args:
        stats: Study statistics

    Returns:
        Rich Panel with mastery chart
    """
    dist = stats.mastery_distribution
    total = sum(dist.values())

    if total == 0:
        content = Text("No cards yet", style="dim")
        return Panel(content, title="[bold]Mastery[/bold]")

    lines = []

    for level in MasteryLevel:
        count = dist.get(level, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar_width = int(pct / 5)  # Scale to 20 chars
        bar = "█" * bar_width
        color = MASTERY_COLORS[level]

        line = Text()
        line.append(f"{level.value:10}", style=color)
        line.append(f" {count:4d} ", style="dim")
        line.append(f"({pct:5.1f}%) ", style="dim")
        line.append(bar, style=color)
        lines.append(line)

    content = Text("\n").join(lines)

    return Panel(
        content,
        title="[bold]Mastery Distribution[/bold]",
        border_style="magenta",
    )


def create_topics_table(stats: StudyStats) -> Panel:
    """Create topics breakdown table.

    Args:
        stats: Study statistics

    Returns:
        Rich Panel with topics table
    """
    if not stats.topics:
        content = Text("No topics yet", style="dim")
        return Panel(content, title="[bold]Topics[/bold]")

    table = Table(show_header=True, header_style="bold")
    table.add_column("Topic", style="cyan", width=20)
    table.add_column("Cards", justify="right", width=8)
    table.add_column("Due", justify="right", width=6)
    table.add_column("Mastery", width=12)
    table.add_column("EF", justify="right", width=6)

    for topic in sorted(stats.topics, key=lambda t: t.due_cards, reverse=True):
        # Truncate long names
        name = topic.name[:18] + "…" if len(topic.name) > 18 else topic.name

        # Due count styling
        due_style = "red" if topic.due_cards > 0 else "green"

        # Mastery badge
        mastery_color = MASTERY_COLORS[topic.mastery_level]
        mastery_text = Text(topic.mastery_level.value, style=mastery_color)

        table.add_row(
            name,
            str(topic.total_cards),
            Text(str(topic.due_cards), style=due_style),
            mastery_text,
            f"{topic.average_ease:.2f}",
        )

    return Panel(
        table,
        title="[bold]Topics[/bold]",
        border_style="blue",
    )


def show_dashboard(
    stats: Optional[StudyStats] = None,
    console: Optional[Console] = None,
) -> None:
    """Display the study progress dashboard.

    Args:
        stats: Statistics to display (fetches if None)
        console: Console to use (creates if None)
    """
    if console is None:
        console = Console()

    if stats is None:
        stats = get_study_stats()

    console.print()
    console.print("[bold magenta]📊 Study Progress Dashboard[/bold magenta]")
    console.print()

    # Top row: Overview cards
    top_panels = [
        create_overview_panel(stats),
        create_accuracy_panel(stats),
        create_cards_panel(stats),
    ]
    console.print(Columns(top_panels, equal=True))

    console.print()

    # Middle row: Mastery distribution
    console.print(create_mastery_chart(stats))

    console.print()

    # Bottom row: Topics table
    console.print(create_topics_table(stats))

    console.print()


def get_dashboard_summary(stats: StudyStats) -> str:
    """Get a text summary of the dashboard.

    Args:
        stats: Study statistics

    Returns:
        Text summary string
    """
    lines = [
        "Study Progress Summary",
        "=" * 30,
        f"Due today: {stats.due_today}",
        f"Reviewed today: {stats.reviewed_today}",
        f"Total cards: {stats.total_cards}",
        f"Streak: {stats.streak_days} days",
        f"Accuracy: {stats.average_accuracy:.1f}%",
    ]

    if stats.topics:
        lines.append("")
        lines.append("Topics:")
        for topic in stats.topics[:5]:
            lines.append(f"  - {topic.name}: {topic.due_cards} due")

    return "\n".join(lines)
