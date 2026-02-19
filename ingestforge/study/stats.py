"""Study statistics aggregator.

UX-019.1: Stats Aggregator for study progress tracking.

Summarizes study metrics including:
- Cards due and reviewed
- Time spent studying
- Mastery levels by topic
- Performance trends"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List
from enum import Enum

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class MasteryLevel(Enum):
    """Mastery level classification."""

    NEW = "new"  # Never reviewed
    LEARNING = "learning"  # EF < 2.0 or reps < 3
    REVIEWING = "reviewing"  # Normal state
    MATURE = "mature"  # EF >= 2.5 and interval >= 21 days


@dataclass
class TopicStats:
    """Statistics for a single topic.

    Attributes:
        name: Topic name
        total_cards: Total cards in topic
        due_cards: Cards due for review
        mastery_level: Current mastery level
        average_ease: Average ease factor
        total_reviews: Total review count
        last_reviewed: Last review timestamp
    """

    name: str
    total_cards: int = 0
    due_cards: int = 0
    mastery_level: MasteryLevel = MasteryLevel.NEW
    average_ease: float = 2.5
    total_reviews: int = 0
    last_reviewed: Optional[datetime] = None


@dataclass
class StudyStats:
    """Aggregate study statistics.

    Attributes:
        total_cards: Total cards across all topics
        due_today: Cards due for review today
        reviewed_today: Cards reviewed today
        time_today_minutes: Time spent today (minutes)
        streak_days: Current streak in days
        average_accuracy: Average accuracy percentage
        topics: Per-topic statistics
        mastery_distribution: Count by mastery level
    """

    total_cards: int = 0
    due_today: int = 0
    reviewed_today: int = 0
    time_today_minutes: float = 0.0
    streak_days: int = 0
    average_accuracy: float = 0.0
    topics: List[TopicStats] = field(default_factory=list)
    mastery_distribution: Dict[MasteryLevel, int] = field(default_factory=dict)


class StatsAggregator:
    """Aggregates study statistics from the review database.

    Collects metrics for the study progress dashboard.
    """

    def __init__(self, db_path: Path) -> None:
        """Initialize stats aggregator.

        Args:
            db_path: Path to review database
        """
        self.db_path = db_path

    def get_stats(self) -> StudyStats:
        """Get comprehensive study statistics.

        Returns:
            StudyStats with all metrics
        """
        if not self.db_path.exists():
            return StudyStats()

        stats = StudyStats()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get card counts
            stats.total_cards = self._count_total_cards(cursor)
            stats.due_today = self._count_due_today(cursor)
            stats.reviewed_today = self._count_reviewed_today(cursor)
            stats.time_today_minutes = self._calculate_time_today(cursor)
            stats.streak_days = self._calculate_streak(cursor)
            stats.average_accuracy = self._calculate_accuracy(cursor)

            # Get per-topic stats
            stats.topics = self._get_topic_stats(cursor)

            # Get mastery distribution
            stats.mastery_distribution = self._get_mastery_distribution(cursor)

        return stats

    def _count_total_cards(self, cursor: sqlite3.Cursor) -> int:
        """Count total cards."""
        cursor.execute("SELECT COUNT(*) FROM cards")
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def _count_due_today(self, cursor: sqlite3.Cursor) -> int:
        """Count cards due today."""
        now = datetime.now().isoformat()
        cursor.execute(
            """
            SELECT COUNT(*) FROM cards
            WHERE next_review IS NULL OR next_review <= ?
        """,
            (now,),
        )
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def _count_reviewed_today(self, cursor: sqlite3.Cursor) -> int:
        """Count cards reviewed today."""
        today_start = (
            datetime.now()
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )

        cursor.execute(
            """
            SELECT COUNT(*) FROM review_history
            WHERE reviewed_at >= ?
        """,
            (today_start,),
        )
        result = cursor.fetchone()
        return int(result[0]) if result else 0

    def _calculate_time_today(self, cursor: sqlite3.Cursor) -> float:
        """Estimate time spent studying today (minutes)."""
        # Approximate: 10 seconds per review
        reviewed = self._count_reviewed_today(cursor)
        return (reviewed * 10) / 60.0

    def _calculate_streak(self, cursor: sqlite3.Cursor) -> int:
        """Calculate current review streak."""
        cursor.execute(
            """
            SELECT DISTINCT date(reviewed_at) as review_date
            FROM review_history
            ORDER BY review_date DESC
            LIMIT 30
        """
        )

        dates = [row[0] for row in cursor.fetchall()]
        if not dates:
            return 0

        streak = 0
        current_date = datetime.now().date()

        for date_str in dates:
            review_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            expected_date = current_date - timedelta(days=streak)

            if review_date == expected_date:
                streak += 1
            elif review_date == expected_date - timedelta(days=1):
                # Allow one day gap (yesterday counts)
                streak += 1
            else:
                break

        return streak

    def _calculate_accuracy(self, cursor: sqlite3.Cursor) -> float:
        """Calculate average accuracy from reviews."""
        cursor.execute(
            """
            SELECT AVG(CASE WHEN quality >= 3 THEN 1.0 ELSE 0.0 END)
            FROM review_history
        """
        )
        result = cursor.fetchone()
        if result and result[0] is not None:
            return round(float(result[0]) * 100, 1)
        return 0.0

    def _get_topic_stats(self, cursor: sqlite3.Cursor) -> List[TopicStats]:
        """Get statistics per topic."""
        cursor.execute(
            """
            SELECT DISTINCT topic FROM cards
        """
        )
        topics = [row[0] for row in cursor.fetchall()]

        stats = []
        now = datetime.now().isoformat()

        for topic in topics:
            # Card counts
            cursor.execute("SELECT COUNT(*) FROM cards WHERE topic = ?", (topic,))
            total = cursor.fetchone()[0]

            # Due cards
            cursor.execute(
                """
                SELECT COUNT(*) FROM cards
                WHERE topic = ? AND (next_review IS NULL OR next_review <= ?)
            """,
                (topic, now),
            )
            due = cursor.fetchone()[0]

            # Average ease
            cursor.execute(
                "SELECT AVG(ease_factor) FROM cards WHERE topic = ?", (topic,)
            )
            avg_ease = cursor.fetchone()[0] or 2.5

            # Total reviews
            cursor.execute(
                """
                SELECT COUNT(*) FROM review_history rh
                JOIN cards c ON rh.card_id = c.card_id
                WHERE c.topic = ?
            """,
                (topic,),
            )
            total_reviews = cursor.fetchone()[0]

            # Last reviewed
            cursor.execute(
                """
                SELECT MAX(rh.reviewed_at) FROM review_history rh
                JOIN cards c ON rh.card_id = c.card_id
                WHERE c.topic = ?
            """,
                (topic,),
            )
            last_reviewed_str = cursor.fetchone()[0]
            last_reviewed = None
            if last_reviewed_str:
                try:
                    last_reviewed = datetime.fromisoformat(last_reviewed_str)
                except ValueError as e:
                    logger.debug(f"Failed to parse last review date: {e}")

            # Determine mastery level
            mastery = self._determine_mastery(avg_ease, total_reviews, total)

            stats.append(
                TopicStats(
                    name=topic,
                    total_cards=total,
                    due_cards=due,
                    mastery_level=mastery,
                    average_ease=round(avg_ease, 2),
                    total_reviews=total_reviews,
                    last_reviewed=last_reviewed,
                )
            )

        return stats

    def _determine_mastery(
        self,
        avg_ease: float,
        total_reviews: int,
        total_cards: int,
    ) -> MasteryLevel:
        """Determine mastery level from metrics."""
        if total_reviews == 0:
            return MasteryLevel.NEW
        if avg_ease < 2.0 or total_reviews < total_cards * 3:
            return MasteryLevel.LEARNING
        if avg_ease >= 2.5:
            return MasteryLevel.MATURE
        return MasteryLevel.REVIEWING

    def _get_mastery_distribution(
        self, cursor: sqlite3.Cursor
    ) -> Dict[MasteryLevel, int]:
        """Get count of cards by mastery level."""
        distribution = {level: 0 for level in MasteryLevel}

        cursor.execute(
            """
            SELECT ease_factor, repetitions FROM cards
        """
        )

        for ease_factor, repetitions in cursor.fetchall():
            level = self._classify_mastery_level(ease_factor, repetitions)
            distribution[level] += 1

        return distribution

    def _classify_mastery_level(
        self, ease_factor: float, repetitions: int
    ) -> MasteryLevel:
        """Classify a card's mastery level.

        Args:
            ease_factor: Card ease factor
            repetitions: Number of repetitions

        Returns:
            MasteryLevel classification
        """
        if repetitions == 0:
            return MasteryLevel.NEW
        if ease_factor < 2.0 or repetitions < 3:
            return MasteryLevel.LEARNING
        if ease_factor >= 2.5 and repetitions >= 7:
            return MasteryLevel.MATURE
        return MasteryLevel.REVIEWING


def get_study_stats(project_dir: Optional[Path] = None) -> StudyStats:
    """Convenience function to get study statistics.

    Args:
        project_dir: Project directory (uses cwd if None)

    Returns:
        StudyStats with all metrics
    """
    if project_dir is None:
        project_dir = Path.cwd()

    db_path = project_dir / ".data" / "reviews.db"
    aggregator = StatsAggregator(db_path)
    return aggregator.get_stats()
