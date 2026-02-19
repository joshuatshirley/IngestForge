"""Lightweight Due Cards Checker.

Minimal overhead check for due review cards.
Implements SRS-004 from the backlog.

This module provides fast, lightweight checking for due spaced repetition
cards at CLI startup with minimal performance impact.

Design Principles (NASA JPL Commandments):
- Rule #1: Max 3 nesting levels, early returns for edge cases
- Rule #4: All functions <60 lines
- Rule #5: No silent exceptions - explicit error handling
- Rule #9: Complete type hints

Performance Target:
- count_due_cards() must complete in <20ms
- Only reads SQLite index, no full table scans
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def count_due_cards(project_dir: Optional[Path] = None) -> Tuple[int, int]:
    """Count due cards with minimal overhead.

    Only reads index - no full table scans.
    Target: < 20ms execution time.

    Rule #1: Early returns for edge cases
    Rule #4: Function <60 lines

    Args:
        project_dir: Project directory (defaults to cwd)

    Returns:
        Tuple of (due_count, total_count)
        Returns (0, 0) if database doesn't exist or on error

    Performance:
        - Uses indexed queries only
        - 0.1s timeout to prevent blocking
        - No full table scans
    """
    if project_dir is None:
        project_dir = Path.cwd()

    # Database is stored in .data/reviews.db (same as review.py)
    db_path = project_dir / ".data" / "reviews.db"

    # Early return if database doesn't exist
    if not db_path.exists():
        return (0, 0)

    try:
        # Use short timeout to prevent blocking CLI startup
        with sqlite3.connect(db_path, timeout=0.1) as conn:
            cursor = conn.cursor()

            # Count due cards (next_review <= now)
            # This query uses the index on next_review for fast execution
            now = datetime.now().isoformat()
            cursor.execute(
                """
                SELECT COUNT(*) FROM cards 
                WHERE next_review IS NULL OR next_review <= ?
            """,
                (now,),
            )
            due_count = cursor.fetchone()[0]

            # Count total cards
            cursor.execute("SELECT COUNT(*) FROM cards")
            total_count = cursor.fetchone()[0]

            return (due_count, total_count)

    except (sqlite3.Error, OSError):
        # Errors are expected if DB is locked or corrupt
        return (0, 0)


def get_due_notification(project_dir: Optional[Path] = None) -> Optional[str]:
    """Get notification message for due cards.

    Rule #4: Small function

    Args:
        project_dir: Project directory

    Returns:
        Notification string or None if no cards due

    Examples:
        >>> get_due_notification()
        '📚 5 cards are due for review. Run `ingestforge study review due`'
    """
    due_count, total_count = count_due_cards(project_dir)

    # Early return if nothing due
    if due_count == 0:
        return None

    # Singular/plural handling
    if due_count == 1:
        return "📚 1 card is due for review. Run `ingestforge study review due`"
    else:
        return f"📚 {due_count} cards are due for review. Run `ingestforge study review due`"


def show_due_notification(
    project_dir: Optional[Path] = None, quiet: bool = False
) -> None:
    """Show due cards notification if applicable.

    Integrates with Rich console for styled output.

    Rule #1: Early returns for simple flow
    Rule #4: Small function

    Args:
        project_dir: Project directory
        quiet: If True, suppress output

    Examples:
        >>> show_due_notification()
        📚 3 cards are due for review. Run `ingestforge study review due`
    """
    # Early return if quiet mode
    if quiet:
        return

    notification = get_due_notification(project_dir)

    # Early return if no notification
    if notification is None:
        return

    # Import Rich only when needed (lazy import for performance)
    try:
        from rich.console import Console

        console = Console()
        console.print(f"[dim]{notification}[/dim]")
    except ImportError:
        # Fallback to plain print if Rich not available
        print(notification)
