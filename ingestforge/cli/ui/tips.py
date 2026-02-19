"""Contextual Tip Engine for post-action suggestions.

Shows relevant "What's next" tips based on the last command executed.
Implements UX-016 from the backlog.

Follows Commandments:
- #4 (Small Functions): Each function has a single clear purpose
- #6 (Smallest Scope): Tips data is module-level constant
- #7 (Check Parameters): Validates command input
"""

from __future__ import annotations

from typing import Dict, List

from rich.console import Console
from rich.panel import Panel

console = Console()


# Tips keyed by command name
COMMAND_TIPS: Dict[str, List[str]] = {
    "init": [
        "Run `ingestforge ingest <file>` to add your first document",
        "Use `ingestforge auth-wizard` to configure LLM API keys",
        "Try `ingestforge quickstart` for a guided walkthrough",
    ],
    "ingest": [
        "Run `ingestforge query 'your question'` to search your knowledge base",
        "Use `ingestforge status` to see what's been ingested",
        "Try `ingestforge export folder-export ./study` to generate study materials",
    ],
    "query": [
        "Use `ingestforge study flashcards 'topic'` to create flashcards",
        "Try `ingestforge study quiz 'topic'` for a practice quiz",
        "Export your findings with `ingestforge export markdown notes.md`",
    ],
    "status": [
        "Add more documents with `ingestforge ingest <file>`",
        "Query your knowledge base with `ingestforge query 'question'`",
    ],
    "export": [
        "Share your export or continue researching",
        "Use `ingestforge query` to explore related topics",
    ],
    "study": [
        "Review flashcards regularly for better retention",
        "Try `ingestforge study quiz` to test your knowledge",
    ],
    "auth-wizard": [
        "Your API keys are configured! Try `ingestforge quickstart`",
        "Run `ingestforge ingest <file>` to process documents with AI",
    ],
    "quickstart": [
        "Explore more with `ingestforge --help`",
        "Generate study materials with `ingestforge export folder-export`",
    ],
}

# Default tips for unknown commands
DEFAULT_TIPS = [
    "Use `ingestforge --help` to see all available commands",
    "Run `ingestforge doctor` to check your setup",
]


def get_tips_for_command(command: str) -> List[str]:
    """Get contextual tips for a completed command.

    Args:
        command: The command name that just completed

    Returns:
        List of tip strings

    Example:
        >>> tips = get_tips_for_command("ingest")
        >>> print(tips[0])
        Run `ingestforge query 'your question'` to search your knowledge base
    """
    # Handle subcommands (e.g., "export markdown" -> "export")
    base_command = command.split()[0] if command else ""
    return COMMAND_TIPS.get(base_command, DEFAULT_TIPS)


def show_tip(command: str, quiet: bool = False) -> None:
    """Display a contextual tip after command completion.

    Shows a single-line tip in dim styling. Designed for minimal
    visual impact while providing helpful guidance.

    Args:
        command: The command that just completed
        quiet: If True, suppress tip output

    Example:
        >>> show_tip("ingest")
        Tip: Run `ingestforge query 'your question'` to search...
    """
    if quiet:
        return

    tips = get_tips_for_command(command)
    if not tips:
        return

    # Show first tip (most relevant)
    tip_text = tips[0]

    console.print()
    console.print(f"[dim]Tip: {tip_text}[/dim]")


def show_tips_panel(command: str, quiet: bool = False, max_tips: int = 3) -> None:
    """Display tips in a panel format.

    Shows up to max_tips suggestions in a formatted panel.
    Useful for commands where multiple next steps are relevant.

    Args:
        command: The command that just completed
        quiet: If True, suppress output
        max_tips: Maximum number of tips to display (default: 3)

    Example:
        >>> show_tips_panel("init")
        ┌─────────────────────┐
        │ What's Next?        │
        ├─────────────────────┤
        │ • Run `ingestforge  │
        │   ingest <file>` to │
        │   add your first... │
        │ • Use `ingestforge  │
        │   auth-wizard` to...│
        │ • Try `ingestforge  │
        │   quickstart` for...│
        └─────────────────────┘
    """
    if quiet:
        return

    tips = get_tips_for_command(command)
    if not tips:
        return

    # Format tips as bullet points (limit to max_tips)
    display_tips = tips[:max_tips]
    tip_lines = "\n".join(f"• {tip}" for tip in display_tips)

    console.print()
    console.print(
        Panel(
            tip_lines,
            title="[bold]What's Next?[/bold]",
            border_style="dim",
            padding=(0, 1),
        )
    )


def show_tip_inline(tip_text: str, quiet: bool = False) -> None:
    """Display a custom tip message inline.

    For ad-hoc tips that aren't in the standard command mapping.

    Args:
        tip_text: Custom tip message to display
        quiet: If True, suppress output

    Example:
        >>> show_tip_inline("Configure your API key with `ingestforge auth-wizard`")
        Tip: Configure your API key with `ingestforge auth-wizard`
    """
    if quiet:
        return

    console.print()
    console.print(f"[dim]Tip: {tip_text}[/dim]")
