"""Tests for stats module (UX-019.1).

Tests the study statistics aggregator:
- Stats dataclasses
- Mastery level determination
- Database queries
- Aggregation logic
"""

import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta

from ingestforge.study.stats import (
    StudyStats,
    TopicStats,
    MasteryLevel,
    StatsAggregator,
    get_study_stats,
)


class TestMasteryLevel:
    """Test MasteryLevel enum."""

    def test_all_levels_defined(self) -> None:
        """All mastery levels should be defined."""
        levels = [l.value for l in MasteryLevel]
        assert "new" in levels
        assert "learning" in levels
        assert "reviewing" in levels
        assert "mature" in levels


class TestStudyStats:
    """Test StudyStats dataclass."""

    def test_default_stats(self) -> None:
        """Default stats should have zero values."""
        stats = StudyStats()
        assert stats.total_cards == 0
        assert stats.due_today == 0
        assert stats.reviewed_today == 0
        assert stats.streak_days == 0

    def test_stats_with_values(self) -> None:
        """Stats should accept values."""
        stats = StudyStats(
            total_cards=100,
            due_today=10,
            reviewed_today=5,
            streak_days=7,
        )
        assert stats.total_cards == 100
        assert stats.due_today == 10


class TestTopicStats:
    """Test TopicStats dataclass."""

    def test_default_topic_stats(self) -> None:
        """Topic stats should have defaults."""
        topic = TopicStats(name="Test")
        assert topic.name == "Test"
        assert topic.total_cards == 0
        assert topic.mastery_level == MasteryLevel.NEW


class TestStatsAggregator:
    """Test StatsAggregator class."""

    def setup_method(self) -> None:
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "reviews.db"
        self._init_db()

    def _init_db(self) -> None:
        """Initialize test database schema."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE cards (
                    card_id TEXT PRIMARY KEY,
                    front TEXT NOT NULL,
                    back TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    ease_factor REAL DEFAULT 2.5,
                    interval INTEGER DEFAULT 1,
                    repetitions INTEGER DEFAULT 0,
                    next_review TEXT,
                    last_reviewed TEXT
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE review_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    card_id TEXT NOT NULL,
                    quality INTEGER NOT NULL,
                    reviewed_at TEXT NOT NULL,
                    interval_before INTEGER,
                    interval_after INTEGER
                )
            """
            )

            conn.commit()

    def _add_card(
        self,
        card_id: str,
        topic: str = "Test",
        ease_factor: float = 2.5,
        repetitions: int = 0,
        next_review: str = None,
    ) -> None:
        """Add a test card."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO cards (card_id, front, back, topic,
                                   ease_factor, repetitions, next_review)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (card_id, "Q", "A", topic, ease_factor, repetitions, next_review),
            )
            conn.commit()

    def _add_review(
        self,
        card_id: str,
        quality: int = 4,
        reviewed_at: str = None,
    ) -> None:
        """Add a test review."""
        if reviewed_at is None:
            reviewed_at = datetime.now().isoformat()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO review_history (card_id, quality, reviewed_at)
                VALUES (?, ?, ?)
            """,
                (card_id, quality, reviewed_at),
            )
            conn.commit()

    def test_empty_database(self) -> None:
        """Empty database should return zero stats."""
        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        assert stats.total_cards == 0
        assert stats.due_today == 0

    def test_missing_database(self) -> None:
        """Missing database should return empty stats."""
        missing_path = Path(self.temp_dir) / "missing.db"
        aggregator = StatsAggregator(missing_path)
        stats = aggregator.get_stats()

        assert stats.total_cards == 0

    def test_count_cards(self) -> None:
        """Should count total cards."""
        self._add_card("card1")
        self._add_card("card2")
        self._add_card("card3")

        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        assert stats.total_cards == 3

    def test_count_due_today(self) -> None:
        """Should count cards due today."""
        # Due card (no next_review)
        self._add_card("card1")

        # Due card (past date)
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        self._add_card("card2", next_review=yesterday)

        # Not due (future date)
        tomorrow = (datetime.now() + timedelta(days=1)).isoformat()
        self._add_card("card3", next_review=tomorrow)

        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        assert stats.due_today == 2

    def test_count_reviewed_today(self) -> None:
        """Should count reviews from today."""
        self._add_card("card1")

        # Today's review
        self._add_review("card1")

        # Yesterday's review
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        self._add_review("card1", reviewed_at=yesterday)

        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        assert stats.reviewed_today == 1

    def test_topic_stats(self) -> None:
        """Should aggregate by topic."""
        self._add_card("card1", topic="Python")
        self._add_card("card2", topic="Python")
        self._add_card("card3", topic="JavaScript")

        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        assert len(stats.topics) == 2

        python_topic = next(t for t in stats.topics if t.name == "Python")
        assert python_topic.total_cards == 2

    def test_mastery_distribution(self) -> None:
        """Should calculate mastery distribution."""
        # New card (no reviews)
        self._add_card("card1", repetitions=0)

        # Learning card (low ease)
        self._add_card("card2", ease_factor=1.5, repetitions=2)

        # Mature card (high ease, many reps)
        self._add_card("card3", ease_factor=2.8, repetitions=10)

        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        assert stats.mastery_distribution[MasteryLevel.NEW] == 1
        assert stats.mastery_distribution[MasteryLevel.LEARNING] == 1
        assert stats.mastery_distribution[MasteryLevel.MATURE] == 1

    def test_accuracy_calculation(self) -> None:
        """Should calculate accuracy from reviews."""
        self._add_card("card1")

        # 3 successful reviews (quality >= 3)
        self._add_review("card1", quality=4)
        self._add_review("card1", quality=3)
        self._add_review("card1", quality=5)

        # 1 failed review
        self._add_review("card1", quality=1)

        aggregator = StatsAggregator(self.db_path)
        stats = aggregator.get_stats()

        # 3/4 = 75%
        assert stats.average_accuracy == 75.0


class TestGetStudyStats:
    """Test get_study_stats convenience function."""

    def test_returns_stats(self) -> None:
        """Should return StudyStats object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            stats = get_study_stats(Path(tmpdir))
            assert isinstance(stats, StudyStats)
