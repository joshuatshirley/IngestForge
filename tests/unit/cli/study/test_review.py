"""
Tests for Review CLI Command with SM-2 Algorithm.

Tests the spaced repetition review system with SM-2 algorithm
implementation and SQLite-based tracking.

Test Strategy
-------------
- Focus on SM-2 algorithm correctness
- Test database operations
- Mock storage for unit testing
- Keep tests simple (NASA JPL Rule #1)

Organization
------------
- TestSM2Algorithm: SM-2 calculation tests
- TestReviewCard: ReviewCard dataclass tests
- TestReviewDatabase: SQLite operations
- TestReviewCommand: CLI command tests
- TestIntegration: Full workflow tests
"""

from datetime import datetime, timedelta
from unittest.mock import Mock, patch

import pytest

from ingestforge.cli.study.review import (
    ReviewCard,
    SM2Algorithm,
    ReviewDatabase,
    ReviewCommand,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_test_card(
    card_id: str = "card_1",
    front: str = "What is Python?",
    back: str = "A programming language",
    topic: str = "Programming",
    ease_factor: float = 2.5,
    interval: int = 1,
    repetitions: int = 0,
) -> ReviewCard:
    """Create a test ReviewCard."""
    return ReviewCard(
        card_id=card_id,
        front=front,
        back=back,
        topic=topic,
        ease_factor=ease_factor,
        interval=interval,
        repetitions=repetitions,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestReviewCard:
    """Tests for ReviewCard dataclass."""

    def test_create_review_card(self):
        """Test creating ReviewCard instance."""
        card = make_test_card()

        assert card.card_id == "card_1"
        assert card.front == "What is Python?"
        assert card.back == "A programming language"
        assert card.topic == "Programming"

    def test_default_values(self):
        """Test ReviewCard default values."""
        card = ReviewCard(
            card_id="test",
            front="Q",
            back="A",
            topic="Test",
        )

        assert card.ease_factor == 2.5
        assert card.interval == 1
        assert card.repetitions == 0
        assert card.next_review is None
        assert card.last_reviewed is None


class TestSM2Algorithm:
    """Tests for SM-2 algorithm implementation."""

    def test_quality_below_3_resets(self):
        """Test that quality < 3 resets card."""
        card = make_test_card(repetitions=5, interval=30, ease_factor=2.5)

        # Quality 2 (failure) should reset
        result = SM2Algorithm.calculate_next_review(card, quality=2)

        assert result.repetitions == 0
        assert result.interval == 1

    def test_quality_3_or_above_passes(self):
        """Test that quality >= 3 is a pass."""
        card = make_test_card(repetitions=0, interval=1)

        result = SM2Algorithm.calculate_next_review(card, quality=4)

        assert result.repetitions == 1

    def test_first_success_interval_1(self):
        """Test first successful review sets interval to 1."""
        card = make_test_card(repetitions=0, interval=1)

        result = SM2Algorithm.calculate_next_review(card, quality=4)

        assert result.interval == 1
        assert result.repetitions == 1

    def test_second_success_interval_6(self):
        """Test second successful review sets interval to 6."""
        card = make_test_card(repetitions=1, interval=1)

        result = SM2Algorithm.calculate_next_review(card, quality=4)

        assert result.interval == 6
        assert result.repetitions == 2

    def test_third_success_uses_ease_factor(self):
        """Test third+ success multiplies by ease factor."""
        card = make_test_card(repetitions=2, interval=6, ease_factor=2.5)

        result = SM2Algorithm.calculate_next_review(card, quality=4)

        expected_interval = int(6 * 2.5)  # 15
        assert result.interval == expected_interval
        assert result.repetitions == 3

    def test_ease_factor_increases_on_perfect(self):
        """Test ease factor increases on quality 5."""
        card = make_test_card(ease_factor=2.5)

        result = SM2Algorithm.calculate_next_review(card, quality=5)

        assert result.ease_factor > 2.5

    def test_ease_factor_decreases_on_low_quality(self):
        """Test ease factor decreases on quality 3."""
        card = make_test_card(ease_factor=2.5)

        result = SM2Algorithm.calculate_next_review(card, quality=3)

        assert result.ease_factor < 2.5

    def test_ease_factor_minimum(self):
        """Test ease factor doesn't go below minimum."""
        card = make_test_card(ease_factor=1.3)

        # Multiple low quality reviews
        result = card
        for _ in range(5):
            result = SM2Algorithm.calculate_next_review(result, quality=3)

        assert result.ease_factor >= SM2Algorithm.MIN_EASE_FACTOR

    def test_quality_clamped(self):
        """Test quality is clamped to 0-5 range."""
        card = make_test_card()

        # Quality below 0
        result1 = SM2Algorithm.calculate_next_review(card, quality=-1)
        assert result1.repetitions == 0  # Treated as 0 (failure)

        # Quality above 5
        result2 = SM2Algorithm.calculate_next_review(card, quality=10)
        assert result2.repetitions == 1  # Treated as 5 (success)

    def test_next_review_date_set(self):
        """Test next review date is calculated."""
        card = make_test_card()

        result = SM2Algorithm.calculate_next_review(card, quality=4)

        assert result.next_review is not None
        assert result.next_review > datetime.now()

    def test_last_reviewed_set(self):
        """Test last reviewed date is set."""
        card = make_test_card()

        result = SM2Algorithm.calculate_next_review(card, quality=4)

        assert result.last_reviewed is not None


class TestReviewDatabase:
    """Tests for SQLite database operations."""

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create temporary database."""
        db_path = tmp_path / "test_reviews.db"
        return ReviewDatabase(db_path)

    def test_database_creation(self, tmp_path):
        """Test database is created."""
        db_path = tmp_path / "reviews.db"
        db = ReviewDatabase(db_path)

        assert db_path.exists()

    def test_save_and_get_card(self, temp_db):
        """Test saving and retrieving a card."""
        card = make_test_card()

        temp_db.save_card(card)

        cards = temp_db.get_cards_by_topic("Programming")

        assert len(cards) == 1
        assert cards[0].front == "What is Python?"

    def test_update_card(self, temp_db):
        """Test updating an existing card."""
        card = make_test_card()
        temp_db.save_card(card)

        # Update card
        card.ease_factor = 2.8
        card.interval = 10
        temp_db.save_card(card)

        cards = temp_db.get_cards_by_topic("Programming")

        assert len(cards) == 1
        assert cards[0].ease_factor == 2.8
        assert cards[0].interval == 10

    def test_get_due_cards(self, temp_db):
        """Test getting cards due for review."""
        # Card due now
        card1 = make_test_card(card_id="due")
        card1.next_review = datetime.now() - timedelta(hours=1)
        temp_db.save_card(card1)

        # Card not due yet
        card2 = make_test_card(card_id="not_due")
        card2.next_review = datetime.now() + timedelta(days=7)
        temp_db.save_card(card2)

        due_cards = temp_db.get_due_cards()

        assert len(due_cards) == 1
        assert due_cards[0].card_id == "due"

    def test_get_due_cards_by_topic(self, temp_db):
        """Test getting due cards filtered by topic."""
        # Due card for topic A
        card1 = make_test_card(card_id="a1", topic="TopicA")
        card1.next_review = datetime.now() - timedelta(hours=1)
        temp_db.save_card(card1)

        # Due card for topic B
        card2 = make_test_card(card_id="b1", topic="TopicB")
        card2.next_review = datetime.now() - timedelta(hours=1)
        temp_db.save_card(card2)

        due_a = temp_db.get_due_cards("TopicA")
        due_b = temp_db.get_due_cards("TopicB")

        assert len(due_a) == 1
        assert len(due_b) == 1
        assert due_a[0].card_id == "a1"

    def test_record_review(self, temp_db):
        """Test recording review history."""
        card = make_test_card()
        temp_db.save_card(card)

        # Record a review
        temp_db.record_review(
            card_id="card_1",
            quality=4,
            interval_before=1,
            interval_after=6,
        )

        # Verify no errors - history is recorded

    def test_get_statistics(self, temp_db):
        """Test getting review statistics."""
        # Add some cards
        for i in range(5):
            card = make_test_card(card_id=f"card_{i}", topic="Test")
            if i < 2:
                card.next_review = datetime.now() - timedelta(hours=1)  # Due
            else:
                card.next_review = datetime.now() + timedelta(days=7)  # Not due
            temp_db.save_card(card)

        stats = temp_db.get_statistics("Test")

        assert stats["total_cards"] == 5
        assert stats["due_today"] == 2
        assert "average_ease_factor" in stats


class TestReviewCommand:
    """Tests for ReviewCommand CLI."""

    def test_create_review_command(self):
        """Test creating ReviewCommand instance."""
        cmd = ReviewCommand()
        assert cmd is not None

    def test_inherits_from_study_command(self):
        """Test ReviewCommand inherits from StudyCommand."""
        from ingestforge.cli.study.base import StudyCommand

        cmd = ReviewCommand()
        assert isinstance(cmd, StudyCommand)

    def test_generate_review_cards(self):
        """Test generating review cards from chunks."""
        cmd = ReviewCommand()

        chunks = [
            Mock(text="Line one\nLine two\nLine three"),
            Mock(text="Another chunk content"),
        ]

        cards = cmd._generate_review_cards(chunks, "Test Topic")

        assert len(cards) > 0
        assert all(c.topic == "Test Topic" for c in cards)

    def test_generate_review_schedule(self, tmp_path):
        """Test generating review schedule."""
        cmd = ReviewCommand()
        db_path = tmp_path / "test.db"
        db = ReviewDatabase(db_path)

        # Add a test card
        card = make_test_card(topic="Test")
        db.save_card(card)

        schedule = cmd._generate_review_schedule(db, "Test")

        assert schedule["topic"] == "Test"
        assert schedule["algorithm"] == "SM-2"
        assert "reviews" in schedule
        assert "statistics" in schedule


class TestIntegration:
    """Integration tests for full review workflow."""

    def test_full_review_workflow(self, tmp_path):
        """Test complete review workflow."""
        cmd = ReviewCommand()

        chunks = [Mock(text="Concept one explanation")]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "search_topic_context") as mock_search:
                mock_init.return_value = {
                    "storage": Mock(),
                    "config": Mock(),
                    "project_path": tmp_path,
                }
                mock_search.return_value = chunks

                result = cmd.execute(
                    topic="Test Topic",
                    project=tmp_path,
                    algorithm="sm2",
                )

                assert result == 0

    def test_show_due_cards(self, tmp_path):
        """Test showing due cards."""
        cmd = ReviewCommand()

        # Create database with due card
        db_path = tmp_path / ".data" / "reviews.db"
        db = ReviewDatabase(db_path)

        card = make_test_card(topic="Test")
        card.next_review = datetime.now() - timedelta(hours=1)
        db.save_card(card)

        with patch.object(cmd, "initialize_context") as mock_init:
            mock_init.return_value = {
                "storage": Mock(),
                "config": Mock(),
                "project_path": tmp_path,
            }

            result = cmd.execute(
                topic="Test",
                project=tmp_path,
                show_due=True,
            )

            assert result == 0


class TestEaseFactorCalculation:
    """Tests for ease factor calculation formula."""

    def test_perfect_review_increases_ef(self):
        """Test perfect review (5) increases ease factor."""
        current_ef = 2.5
        new_ef = SM2Algorithm._calculate_new_ease_factor(current_ef, 5)

        assert new_ef > current_ef

    def test_good_review_slight_increase(self):
        """Test good review (4) maintains or slightly increases ease factor."""
        current_ef = 2.5
        new_ef = SM2Algorithm._calculate_new_ease_factor(current_ef, 4)

        # Quality 4 keeps EF roughly the same (formula: EF + 0)
        assert new_ef >= current_ef - 0.1  # Allow small variance
        assert new_ef < SM2Algorithm._calculate_new_ease_factor(current_ef, 5)

    def test_ok_review_decreases_ef(self):
        """Test OK review (3) decreases ease factor."""
        current_ef = 2.5
        new_ef = SM2Algorithm._calculate_new_ease_factor(current_ef, 3)

        assert new_ef < current_ef

    def test_minimum_ease_factor_enforced(self):
        """Test minimum ease factor is enforced."""
        current_ef = 1.5
        new_ef = SM2Algorithm._calculate_new_ease_factor(current_ef, 0)

        assert new_ef >= SM2Algorithm.MIN_EASE_FACTOR
