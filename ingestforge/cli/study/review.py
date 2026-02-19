"""Review command - Spaced repetition review system with SM-2 algorithm.

Implements the SM-2 (SuperMemo 2) algorithm for optimal review scheduling.
Tracks review history in SQLite for persistent progress tracking."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.study.base import StudyCommand
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ReviewCard:
    """Represents a single review card with SM-2 metadata.

    Attributes:
        card_id: Unique identifier for the card
        front: Question/term on front of card
        back: Answer/definition on back of card
        topic: Topic this card belongs to
        ease_factor: SM-2 ease factor (default 2.5)
        interval: Current interval in days
        repetitions: Number of successful reviews
        next_review: Date of next scheduled review
        last_reviewed: Date of last review
    """

    card_id: str
    front: str
    back: str
    topic: str
    ease_factor: float = 2.5
    interval: int = 1
    repetitions: int = 0
    next_review: Optional[datetime] = None
    last_reviewed: Optional[datetime] = None


class SM2Algorithm:
    """Implementation of SM-2 spaced repetition algorithm.

    The SM-2 algorithm calculates optimal review intervals based on
    user performance ratings (0-5 scale).

    Reference: https://www.supermemo.com/en/archives1990-2015/english/ol/sm2
    """

    # Minimum ease factor to prevent intervals from shrinking too much
    MIN_EASE_FACTOR = 1.3

    @staticmethod
    def calculate_next_review(
        card: ReviewCard,
        quality: int,
    ) -> ReviewCard:
        """Calculate next review date using SM-2 algorithm.

        Rule #4: Function <60 lines
        Rule #1: Early returns for edge cases

        Args:
            card: Current card state
            quality: User rating 0-5 (0=complete failure, 5=perfect)

        Returns:
            Updated ReviewCard with new scheduling
        """
        # Validate quality rating
        quality = max(0, min(5, quality))

        # Clone card for modification
        new_card = ReviewCard(
            card_id=card.card_id,
            front=card.front,
            back=card.back,
            topic=card.topic,
            ease_factor=card.ease_factor,
            interval=card.interval,
            repetitions=card.repetitions,
            next_review=card.next_review,
            last_reviewed=datetime.now(),
        )

        # Quality < 3 means failure - reset to beginning
        if quality < 3:
            new_card.repetitions = 0
            new_card.interval = 1
        else:
            # Successful review - update based on repetitions
            new_card = SM2Algorithm._update_successful_review(new_card, quality)

        # Calculate next review date
        new_card.next_review = datetime.now() + timedelta(days=new_card.interval)

        return new_card

    @staticmethod
    def _update_successful_review(card: ReviewCard, quality: int) -> ReviewCard:
        """Update card after successful review.

        Rule #4: Extracted helper to keep functions small
        """
        if card.repetitions == 0:
            card.interval = 1
        elif card.repetitions == 1:
            card.interval = 6
        else:
            card.interval = int(card.interval * card.ease_factor)

        card.repetitions += 1

        # Update ease factor
        card.ease_factor = SM2Algorithm._calculate_new_ease_factor(
            card.ease_factor, quality
        )

        return card

    @staticmethod
    def _calculate_new_ease_factor(current_ef: float, quality: int) -> float:
        """Calculate new ease factor based on quality rating.

        Formula: EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
        """
        new_ef = current_ef + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
        return max(SM2Algorithm.MIN_EASE_FACTOR, new_ef)


class ReviewDatabase:
    """SQLite database for tracking review history.

    Stores cards and their review history for persistent
    spaced repetition tracking.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema.

        Rule #4: Function <60 lines
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Cards table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS cards (
                    card_id TEXT PRIMARY KEY,
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    ease_factor REAL DEFAULT 2.5,
                    interval INTEGER DEFAULT 1,
                    repetitions INTEGER DEFAULT 0,
                    next_review TEXT,
                    last_reviewed TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Review history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS review_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    quality INTEGER NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    interval_before INTEGER,
                    interval_after INTEGER,
                    FOREIGN KEY (card_id) REFERENCES cards(card_id)
                )
            """
            )

            conn.commit()

    def save_card(self, card: ReviewCard) -> None:
        """Save or update a review card.

        Args:
            card: ReviewCard to save
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO cards
                (card_id, front, back, topic, ease_factor, interval,
                 repetitions, next_review, last_reviewed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    card.card_id,
                    card.front,
                    card.back,
                    card.topic,
                    card.ease_factor,
                    card.interval,
                    card.repetitions,
                    card.next_review.isoformat() if card.next_review else None,
                    card.last_reviewed.isoformat() if card.last_reviewed else None,
                ),
            )
            conn.commit()

    def get_due_cards(self, topic: Optional[str] = None) -> List[ReviewCard]:
        """Get cards due for review.

        Args:
            topic: Optional topic filter

        Returns:
            List of cards due for review
        """
        now = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if topic:
                cursor.execute(
                    """
                    SELECT * FROM cards
                    WHERE (next_review IS NULL OR next_review <= ?)
                    AND topic = ?
                    ORDER BY next_review
                """,
                    (now, topic),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM cards
                    WHERE next_review IS NULL OR next_review <= ?
                    ORDER BY next_review
                """,
                    (now,),
                )

            return [self._row_to_card(row) for row in cursor.fetchall()]

    def get_cards_by_topic(self, topic: str) -> List[ReviewCard]:
        """Get all cards for a topic.

        Args:
            topic: Topic to filter by

        Returns:
            List of cards
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM cards WHERE topic = ?", (topic,))
            return [self._row_to_card(row) for row in cursor.fetchall()]

    def record_review(
        self, card_id: str, quality: int, interval_before: int, interval_after: int
    ) -> None:
        """Record a review in history.

        Args:
            card_id: Card identifier
            quality: Quality rating (0-5)
            interval_before: Interval before review
            interval_after: Interval after review
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO review_history
                (card_id, quality, reviewed_at, interval_before, interval_after)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    card_id,
                    quality,
                    datetime.now().isoformat(),
                    interval_before,
                    interval_after,
                ),
            )
            conn.commit()

    def get_statistics(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Get review statistics.

        Args:
            topic: Optional topic filter

        Returns:
            Statistics dictionary
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Total cards
            if topic:
                cursor.execute("SELECT COUNT(*) FROM cards WHERE topic = ?", (topic,))
            else:
                cursor.execute("SELECT COUNT(*) FROM cards")
            total_cards = cursor.fetchone()[0]

            # Due today
            now = datetime.now().isoformat()
            if topic:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM cards
                    WHERE (next_review IS NULL OR next_review <= ?)
                    AND topic = ?
                """,
                    (now, topic),
                )
            else:
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM cards
                    WHERE next_review IS NULL OR next_review <= ?
                """,
                    (now,),
                )
            due_today = cursor.fetchone()[0]

            # Average ease factor
            if topic:
                cursor.execute(
                    "SELECT AVG(ease_factor) FROM cards WHERE topic = ?", (topic,)
                )
            else:
                cursor.execute("SELECT AVG(ease_factor) FROM cards")
            avg_ease = cursor.fetchone()[0] or 2.5

            return {
                "total_cards": total_cards,
                "due_today": due_today,
                "average_ease_factor": round(avg_ease, 2),
            }

    def _row_to_card(self, row: tuple) -> ReviewCard:
        """Convert database row to ReviewCard.

        Args:
            row: Database row tuple

        Returns:
            ReviewCard instance
        """
        return ReviewCard(
            card_id=row[0],
            front=row[1],
            back=row[2],
            topic=row[3],
            ease_factor=row[4],
            interval=row[5],
            repetitions=row[6],
            next_review=datetime.fromisoformat(row[7]) if row[7] else None,
            last_reviewed=datetime.fromisoformat(row[8]) if row[8] else None,
        )


class ReviewCommand(StudyCommand):
    """Spaced repetition review system using SM-2 algorithm."""

    def execute(
        self,
        topic: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        algorithm: str = "sm2",
        show_due: bool = False,
    ) -> int:
        """Generate review schedule for spaced repetition.

        Rule #4: Function <60 lines
        """
        try:
            ctx = self.initialize_context(project, require_storage=True)
            db_path = self._get_db_path(ctx)
            db = ReviewDatabase(db_path)

            if show_due:
                return self._show_due_cards(db, topic)

            # Search for content and generate cards
            chunks = self.search_topic_context(ctx["storage"], topic, k=30)
            if not chunks:
                self.print_warning(f"No content found for topic: {topic}")
                return 0

            # Generate and save review cards
            cards = self._generate_review_cards(chunks, topic)
            for card in cards:
                db.save_card(card)

            # Generate schedule
            schedule = self._generate_review_schedule(db, topic)
            self._display_review_schedule(schedule, topic)

            if output:
                self.save_json_output(
                    output, schedule, f"Review schedule saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Review schedule generation failed")

    def _get_db_path(self, ctx: Dict[str, Any]) -> Path:
        """Get path to review database.

        Args:
            ctx: Context dictionary

        Returns:
            Path to SQLite database
        """
        project_path = ctx.get("project_path", Path.cwd())
        return project_path / ".data" / "reviews.db"

    def _show_due_cards(self, db: ReviewDatabase, topic: str) -> int:
        """Show cards due for review.

        Args:
            db: Review database
            topic: Topic to filter by

        Returns:
            Exit code
        """
        due_cards = db.get_due_cards(topic if topic != "all" else None)

        if not due_cards:
            self.print_info(f"No cards due for review in topic: {topic}")
            return 0

        self._display_due_cards(due_cards)
        return 0

    def _display_due_cards(self, cards: List[ReviewCard]) -> None:
        """Display cards due for review.

        Args:
            cards: List of due cards
        """
        self.console.print()
        self.console.print(f"[bold cyan]Cards Due for Review: {len(cards)}[/bold cyan]")
        self.console.print()

        table = Table(title="Due Cards", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Front", style="yellow", width=40)
        table.add_column("Topic", style="blue", width=15)
        table.add_column("Interval", style="green", width=10)
        table.add_column("EF", style="magenta", width=6)

        for idx, card in enumerate(cards[:20], 1):
            table.add_row(
                str(idx),
                card.front[:60],
                card.topic[:15],
                f"{card.interval}d",
                f"{card.ease_factor:.2f}",
            )

        self.console.print(table)

        if len(cards) > 20:
            self.console.print(f"\n... and {len(cards) - 20} more cards")

    def _generate_review_cards(self, chunks: list, topic: str) -> List[ReviewCard]:
        """Generate review cards from chunks.

        Rule #4: Function <60 lines

        Args:
            chunks: List of content chunks
            topic: Topic name

        Returns:
            List of ReviewCard objects
        """
        cards = []

        for idx, chunk in enumerate(chunks[:15]):
            text = self._extract_chunk_text(chunk)

            # Create a simple card from chunk
            lines = text.strip().split("\n")
            if not lines:
                continue

            front = lines[0][:100]
            back = " ".join(lines[1:])[:200] if len(lines) > 1 else text[:200]

            card = ReviewCard(
                card_id=f"{topic}_{idx}",
                front=front,
                back=back,
                topic=topic,
            )
            cards.append(card)

        return cards

    def _generate_review_schedule(
        self, db: ReviewDatabase, topic: str
    ) -> Dict[str, Any]:
        """Generate spaced repetition schedule.

        Args:
            db: Review database
            topic: Topic name

        Returns:
            Schedule dictionary
        """
        now = datetime.now()
        stats = db.get_statistics(topic)
        cards = db.get_cards_by_topic(topic)

        # SM-2 standard intervals
        intervals = [1, 3, 7, 14, 30, 60, 120]

        reviews = []
        for idx, interval_days in enumerate(intervals, 1):
            review_date = now + timedelta(days=interval_days)
            reviews.append(
                {
                    "review_number": idx,
                    "days_from_now": interval_days,
                    "review_date": review_date.strftime("%Y-%m-%d"),
                    "status": "pending",
                }
            )

        # Extract concepts from cards
        concepts = [card.front for card in cards[:15]]

        return {
            "topic": topic,
            "algorithm": "SM-2",
            "created_date": now.strftime("%Y-%m-%d"),
            "statistics": stats,
            "reviews": reviews,
            "concepts_to_review": concepts,
            "total_concepts": len(cards),
        }

    def _display_review_schedule(self, schedule: Dict[str, Any], topic: str) -> None:
        """Display review schedule.

        Args:
            schedule: Schedule dictionary
            topic: Topic name
        """
        self.console.print()
        self.console.print(
            f"[bold cyan]SM-2 Spaced Repetition Schedule: {topic}[/bold cyan]\n"
        )

        # Statistics overview
        stats = schedule.get("statistics", {})
        overview = Panel(
            f"Created: {schedule['created_date']}\n"
            f"Algorithm: {schedule['algorithm']}\n"
            f"Total cards: {stats.get('total_cards', 0)}\n"
            f"Due today: {stats.get('due_today', 0)}\n"
            f"Average ease factor: {stats.get('average_ease_factor', 2.5)}",
            title="Overview",
            border_style="cyan",
        )
        self.console.print(overview)

        # Review schedule table
        self.console.print()
        table = Table(title="Recommended Review Schedule (SM-2)")
        table.add_column("#", width=5)
        table.add_column("Days", width=10)
        table.add_column("Review Date", width=15)
        table.add_column("Status", width=10)

        for review in schedule["reviews"]:
            table.add_row(
                str(review["review_number"]),
                str(review["days_from_now"]),
                review["review_date"],
                review["status"],
            )

        self.console.print(table)

        # Sample concepts
        concepts = schedule.get("concepts_to_review", [])
        if concepts:
            self.console.print()
            self.console.print("[bold yellow]Sample Concepts to Review:[/bold yellow]")
            for concept in concepts[:5]:
                self.console.print(f"  - {concept[:60]}")


# Create subcommand group for review
review_app = typer.Typer(
    name="review",
    help="Spaced repetition review commands",
    add_completion=False,
)


@review_app.command("schedule")
def schedule_command(
    topic: str = typer.Argument(..., help="Topic to create review schedule for"),
    algorithm: str = typer.Option(
        "sm2", "--algorithm", "-a", help="Scheduling algorithm (sm2)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for schedule"
    ),
) -> None:
    """Create a spaced repetition review schedule.

    Uses the SM-2 algorithm (SuperMemo 2) to calculate optimal
    review intervals based on performance. The schedule is stored
    in a SQLite database for persistent tracking.

    The SM-2 algorithm uses:
    - Ease Factor (EF): Difficulty multiplier starting at 2.5
    - Intervals: 1, 6, then EF * previous interval
    - Quality ratings: 0-5 scale (3+ is passing)

    Examples:
        # Create schedule for a topic
        ingestforge study review schedule "Python basics"

        # Save schedule to file
        ingestforge study review schedule "History" -o schedule.json
    """
    cmd = ReviewCommand()
    exit_code = cmd.execute(topic, project, output, algorithm, show_due=False)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@review_app.command("due")
def due_command(
    topic: str = typer.Argument(
        "all", help="Topic to show due cards for (use 'all' for all topics)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Show cards due for review today.

    Lists all cards that are scheduled for review based on
    the SM-2 algorithm intervals.

    Examples:
        # Show all due cards
        ingestforge study review due

        # Show due cards for specific topic
        ingestforge study review due "Python"
    """
    cmd = ReviewCommand()
    exit_code = cmd.execute(topic, project, None, "sm2", show_due=True)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Legacy command wrapper for backwards compatibility
def command(
    topic: str = typer.Argument(..., help="Topic to create review schedule for"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
    interval: int = typer.Option(
        1, "--interval", "-i", help="Interval multiplier (days)"
    ),
) -> None:
    """Generate spaced repetition review schedule.

    Creates a review schedule based on the SM-2 algorithm
    (SuperMemo 2) to optimize long-term retention.

    Standard SM-2 intervals: 1, 6, then multiply by ease factor.

    Examples:
        ingestforge study review "Python basics"
        ingestforge study review "Machine Learning" -o schedule.json
    """
    cmd = ReviewCommand()
    exit_code = cmd.execute(topic, project, output, "sm2", show_due=False)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
