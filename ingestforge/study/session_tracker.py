"""Session tracker with undo stack for review sessions.

SRS-002.2: Session Undo Stack - allows correcting recent ratings.

Tracks review session state with in-memory undo functionality.
Undo history is session-only and not persisted."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any


@dataclass
class ReviewAction:
    """Represents a single review action that can be undone.

    Attributes:
        card_id: ID of the reviewed card
        rating: Rating given (1-4 scale)
        quality: SM-2 quality (0-5 scale)
        timestamp: When the review occurred
        prev_state: Card state before this review (for undo)
    """

    card_id: str
    rating: int
    quality: int
    timestamp: datetime = field(default_factory=datetime.now)
    prev_state: Optional[Dict[str, Any]] = None


class SessionTracker:
    """Tracks a review session with undo capability.

    Maintains an in-memory stack of review actions that can be
    undone during the current session. State is not persisted
    across sessions.

    Attributes:
        start_time: When the session started
        actions: List of review actions (undo stack)
        max_undo: Maximum number of undoable actions
    """

    DEFAULT_MAX_UNDO = 50

    def __init__(self, max_undo: int = DEFAULT_MAX_UNDO) -> None:
        """Initialize session tracker.

        Args:
            max_undo: Maximum actions to keep in undo stack
        """
        self.start_time = datetime.now()
        self.actions: List[ReviewAction] = []
        self.max_undo = max_undo
        self._rating_counts: Dict[int, int] = {}

    def record_review(
        self,
        card_id: str,
        rating: int,
        quality: int,
        prev_state: Optional[Dict[str, Any]] = None,
    ) -> ReviewAction:
        """Record a review action.

        Args:
            card_id: ID of the reviewed card
            rating: User rating (1-4)
            quality: SM-2 quality (0-5)
            prev_state: Card state before review (for undo)

        Returns:
            The recorded ReviewAction
        """
        action = ReviewAction(
            card_id=card_id,
            rating=rating,
            quality=quality,
            prev_state=prev_state,
        )

        self.actions.append(action)

        # Update rating counts
        self._rating_counts[rating] = self._rating_counts.get(rating, 0) + 1

        # Trim undo stack if needed
        if len(self.actions) > self.max_undo:
            removed = self.actions.pop(0)
            # Don't decrement counts - we want total session counts

        return action

    def undo(self) -> Optional[ReviewAction]:
        """Undo the most recent review.

        Returns:
            The undone ReviewAction, or None if stack is empty
        """
        if not self.actions:
            return None

        action = self.actions.pop()

        # Decrement rating count
        if action.rating in self._rating_counts:
            self._rating_counts[action.rating] -= 1
            if self._rating_counts[action.rating] <= 0:
                del self._rating_counts[action.rating]

        return action

    def can_undo(self) -> bool:
        """Check if undo is available.

        Returns:
            True if there are actions to undo
        """
        return len(self.actions) > 0

    def get_last_action(self) -> Optional[ReviewAction]:
        """Get the most recent action without removing it.

        Returns:
            Most recent ReviewAction or None
        """
        if not self.actions:
            return None
        return self.actions[-1]

    @property
    def cards_reviewed(self) -> int:
        """Total cards reviewed this session."""
        return sum(self._rating_counts.values())

    @property
    def rating_counts(self) -> Dict[int, int]:
        """Get counts by rating value."""
        return self._rating_counts.copy()

    @property
    def correct_count(self) -> int:
        """Count of ratings >= 3 (Good or Easy)."""
        return sum(
            count for rating, count in self._rating_counts.items() if rating >= 3
        )

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as percentage of Good/Easy ratings."""
        total = self.cards_reviewed
        if total == 0:
            return 0.0
        return (self.correct_count / total) * 100

    @property
    def session_duration_seconds(self) -> float:
        """Get session duration in seconds."""
        return (datetime.now() - self.start_time).total_seconds()

    def get_session_stats(self) -> Dict[str, Any]:
        """Get comprehensive session statistics.

        Returns:
            Dictionary with session statistics
        """
        return {
            "start_time": self.start_time.isoformat(),
            "duration_seconds": self.session_duration_seconds,
            "cards_reviewed": self.cards_reviewed,
            "correct_count": self.correct_count,
            "accuracy": round(self.accuracy, 1),
            "rating_breakdown": self._rating_counts.copy(),
            "undo_available": self.can_undo(),
            "undo_stack_size": len(self.actions),
        }

    def reset(self) -> None:
        """Reset the session tracker for a new session."""
        self.start_time = datetime.now()
        self.actions.clear()
        self._rating_counts.clear()
