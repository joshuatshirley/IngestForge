"""Tests for session_tracker module (SRS-002.2).

Tests the session tracking and undo functionality:
- Recording reviews
- Undo stack
- Session statistics
- Rating counts
"""

from datetime import datetime
import time

from ingestforge.study.session_tracker import (
    SessionTracker,
    ReviewAction,
)


class TestReviewAction:
    """Test ReviewAction dataclass."""

    def test_action_creation(self) -> None:
        """ReviewAction should be creatable."""
        action = ReviewAction(
            card_id="card_1",
            rating=3,
            quality=4,
        )
        assert action.card_id == "card_1"
        assert action.rating == 3
        assert action.quality == 4

    def test_action_has_timestamp(self) -> None:
        """ReviewAction should have auto-timestamp."""
        action = ReviewAction(
            card_id="card_1",
            rating=3,
            quality=4,
        )
        assert isinstance(action.timestamp, datetime)

    def test_action_prev_state(self) -> None:
        """ReviewAction can store previous state."""
        prev_state = {"ease_factor": 2.5, "interval": 6}
        action = ReviewAction(
            card_id="card_1",
            rating=3,
            quality=4,
            prev_state=prev_state,
        )
        assert action.prev_state == prev_state


class TestSessionTracker:
    """Test SessionTracker class."""

    def test_tracker_creation(self) -> None:
        """SessionTracker should initialize properly."""
        tracker = SessionTracker()
        assert tracker.cards_reviewed == 0
        assert tracker.can_undo() is False

    def test_custom_max_undo(self) -> None:
        """SessionTracker should respect custom max_undo."""
        tracker = SessionTracker(max_undo=5)
        assert tracker.max_undo == 5


class TestRecordReview:
    """Test recording reviews."""

    def test_record_single_review(self) -> None:
        """Recording a review should increment count."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        assert tracker.cards_reviewed == 1

    def test_record_multiple_reviews(self) -> None:
        """Recording multiple reviews should track all."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        tracker.record_review("card_2", rating=4, quality=5)
        tracker.record_review("card_3", rating=2, quality=3)
        assert tracker.cards_reviewed == 3

    def test_record_returns_action(self) -> None:
        """record_review should return the action."""
        tracker = SessionTracker()
        action = tracker.record_review("card_1", rating=3, quality=4)
        assert isinstance(action, ReviewAction)
        assert action.card_id == "card_1"

    def test_record_with_prev_state(self) -> None:
        """Recording can include previous state."""
        tracker = SessionTracker()
        prev = {"interval": 6}
        action = tracker.record_review("card_1", rating=3, quality=4, prev_state=prev)
        assert action.prev_state == prev


class TestUndoStack:
    """Test undo functionality."""

    def test_can_undo_after_review(self) -> None:
        """can_undo should return True after review."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        assert tracker.can_undo() is True

    def test_undo_returns_action(self) -> None:
        """undo should return the undone action."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        action = tracker.undo()
        assert action is not None
        assert action.card_id == "card_1"

    def test_undo_decrements_count(self) -> None:
        """undo should decrement card count."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        tracker.record_review("card_2", rating=4, quality=5)
        assert tracker.cards_reviewed == 2
        tracker.undo()
        assert tracker.cards_reviewed == 1

    def test_undo_empty_stack(self) -> None:
        """undo on empty stack returns None."""
        tracker = SessionTracker()
        assert tracker.undo() is None

    def test_multiple_undos(self) -> None:
        """Multiple undos should work correctly."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        tracker.record_review("card_2", rating=4, quality=5)
        tracker.record_review("card_3", rating=2, quality=3)

        assert tracker.cards_reviewed == 3

        tracker.undo()
        assert tracker.cards_reviewed == 2

        tracker.undo()
        assert tracker.cards_reviewed == 1

        tracker.undo()
        assert tracker.cards_reviewed == 0

        assert tracker.can_undo() is False

    def test_max_undo_limit(self) -> None:
        """Stack should respect max_undo limit."""
        tracker = SessionTracker(max_undo=3)

        # Add 5 reviews
        for i in range(5):
            tracker.record_review(f"card_{i}", rating=3, quality=4)

        # Should only be able to undo 3
        assert len(tracker.actions) == 3

        # Total count should still be 5
        assert tracker.cards_reviewed == 5


class TestGetLastAction:
    """Test get_last_action method."""

    def test_get_last_action_empty(self) -> None:
        """get_last_action on empty stack returns None."""
        tracker = SessionTracker()
        assert tracker.get_last_action() is None

    def test_get_last_action(self) -> None:
        """get_last_action returns most recent."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        tracker.record_review("card_2", rating=4, quality=5)

        action = tracker.get_last_action()
        assert action is not None
        assert action.card_id == "card_2"

    def test_get_last_action_does_not_remove(self) -> None:
        """get_last_action should not modify stack."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)

        tracker.get_last_action()
        assert tracker.can_undo() is True


class TestRatingCounts:
    """Test rating count tracking."""

    def test_rating_counts(self) -> None:
        """rating_counts should track by rating value."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=1, quality=1)
        tracker.record_review("card_2", rating=3, quality=4)
        tracker.record_review("card_3", rating=3, quality=4)
        tracker.record_review("card_4", rating=4, quality=5)

        counts = tracker.rating_counts
        assert counts.get(1, 0) == 1
        assert counts.get(3, 0) == 2
        assert counts.get(4, 0) == 1

    def test_correct_count(self) -> None:
        """correct_count should count ratings >= 3."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=1, quality=1)  # fail
        tracker.record_review("card_2", rating=2, quality=3)  # fail
        tracker.record_review("card_3", rating=3, quality=4)  # correct
        tracker.record_review("card_4", rating=4, quality=5)  # correct

        assert tracker.correct_count == 2

    def test_accuracy(self) -> None:
        """accuracy should calculate correct percentage."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=1, quality=1)
        tracker.record_review("card_2", rating=3, quality=4)
        tracker.record_review("card_3", rating=4, quality=5)
        tracker.record_review("card_4", rating=4, quality=5)

        # 3 correct out of 4 = 75%
        assert tracker.accuracy == 75.0

    def test_accuracy_empty(self) -> None:
        """accuracy with no reviews should be 0."""
        tracker = SessionTracker()
        assert tracker.accuracy == 0.0


class TestSessionStats:
    """Test session statistics."""

    def test_get_session_stats(self) -> None:
        """get_session_stats should return comprehensive stats."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        tracker.record_review("card_2", rating=4, quality=5)

        stats = tracker.get_session_stats()

        assert "start_time" in stats
        assert "duration_seconds" in stats
        assert stats["cards_reviewed"] == 2
        assert stats["correct_count"] == 2
        assert stats["accuracy"] == 100.0
        assert "rating_breakdown" in stats
        assert stats["undo_available"] is True

    def test_session_duration(self) -> None:
        """session_duration_seconds should track time."""
        tracker = SessionTracker()
        time.sleep(0.1)  # Short delay
        duration = tracker.session_duration_seconds
        assert duration >= 0.1


class TestReset:
    """Test session reset."""

    def test_reset_clears_all(self) -> None:
        """reset should clear all state."""
        tracker = SessionTracker()
        tracker.record_review("card_1", rating=3, quality=4)
        tracker.record_review("card_2", rating=4, quality=5)

        tracker.reset()

        assert tracker.cards_reviewed == 0
        assert tracker.can_undo() is False
        assert len(tracker.actions) == 0
