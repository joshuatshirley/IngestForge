"""SM-2 spaced repetition algorithm implementation.

TICKET-102: SM-2 Algorithm Implementation for spaced repetition scheduling.

Implements the SuperMemo SM-2 algorithm for calculating optimal review intervals
based on user performance ratings."""

from __future__ import annotations

from typing import Tuple

# SM-2 Algorithm Constants
MIN_EASE_FACTOR: float = 1.3
INITIAL_EASE_FACTOR: float = 2.5
INITIAL_INTERVAL: float = 1.0
FIRST_SUCCESS_INTERVAL: float = 1.0
SECOND_SUCCESS_INTERVAL: float = 6.0
FAIL_RESET_INTERVAL: float = 1.0


def get_initial_ease_factor() -> float:
    """Get the initial ease factor for new cards.

    Returns:
        Initial ease factor (typically 2.5 for SM-2 algorithm).
    """
    return INITIAL_EASE_FACTOR


def get_initial_interval() -> float:
    """Get the initial interval for new cards.

    Returns:
        Initial interval in days (typically 1.0 for SM-2 algorithm).
    """
    return INITIAL_INTERVAL


def calculate_next_interval(
    quality: int,
    prev_interval: float,
    ease_factor: float,
) -> Tuple[float, float]:
    """Calculate next review interval using SM-2 algorithm.

    The SM-2 algorithm determines optimal spacing for reviews based on
    how well the user recalled the information.

    Args:
        quality: Rating from 0-5 indicating recall quality.
            - 0: Complete blackout, no recall
            - 1: Incorrect, but correct answer remembered
            - 2: Incorrect, but correct answer seemed easy to recall
            - 3: Correct with serious difficulty
            - 4: Correct with some hesitation
            - 5: Perfect recall
        prev_interval: Previous interval in days (0 for first review).
        ease_factor: Current ease factor (minimum 1.3).

    Returns:
        Tuple of (new_interval, new_ease_factor):
            - new_interval: Days until next review
            - new_ease_factor: Updated ease factor for future reviews

    Raises:
        AssertionError: If quality is not in range [0, 5].

    Examples:
        >>> calculate_next_interval(5, 0, 2.5)  # First perfect review
        (1.0, 2.6)
        >>> calculate_next_interval(5, 1, 2.5)  # Second perfect review
        (6.0, 2.7)
        >>> calculate_next_interval(2, 10, 2.5)  # Failed review
        (1.0, 2.5)
    """
    assert 0 <= quality <= 5, f"Quality must be 0-5, got {quality}"

    # Ensure ease factor meets minimum
    ease_factor = max(MIN_EASE_FACTOR, ease_factor)

    # Failed review (quality < 3): reset interval, keep ease factor
    if quality < 3:
        return FAIL_RESET_INTERVAL, ease_factor

    # Successful review (quality >= 3): calculate new interval and ease
    new_interval = _calculate_success_interval(prev_interval, ease_factor)
    new_ease_factor = _calculate_new_ease_factor(quality, ease_factor)

    return new_interval, new_ease_factor


def _calculate_success_interval(prev_interval: float, ease_factor: float) -> float:
    """Calculate interval for successful review.

    Args:
        prev_interval: Previous interval in days.
        ease_factor: Current ease factor.

    Returns:
        New interval in days.
    """
    # First successful review: interval = 1 day
    if prev_interval < FIRST_SUCCESS_INTERVAL:
        return FIRST_SUCCESS_INTERVAL

    # Second successful review: interval = 6 days
    if prev_interval < SECOND_SUCCESS_INTERVAL:
        return SECOND_SUCCESS_INTERVAL

    # Subsequent reviews: interval = prev_interval * ease_factor
    return prev_interval * ease_factor


def _calculate_new_ease_factor(quality: int, ease_factor: float) -> float:
    """Calculate new ease factor based on quality.

    SM-2 ease factor formula:
    EF' = EF + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))

    Where:
        EF = current ease factor
        q = quality rating (0-5)
        EF' = new ease factor

    Args:
        quality: Rating from 3-5 (only called for successful reviews).
        ease_factor: Current ease factor.

    Returns:
        New ease factor (minimum 1.3).
    """
    # SM-2 ease factor adjustment formula
    # delta = (5 - q), which is 0 for perfect, positive for lower quality
    diff = 5 - quality
    adjustment = 0.1 - diff * (0.08 + diff * 0.02)
    new_ease_factor = ease_factor + adjustment

    return max(MIN_EASE_FACTOR, new_ease_factor)


def get_review_count(prev_interval: float) -> int:
    """Estimate review count from interval.

    Args:
        prev_interval: Previous interval in days.

    Returns:
        Estimated number of successful reviews.
    """
    if prev_interval < FIRST_SUCCESS_INTERVAL:
        return 0
    if prev_interval < SECOND_SUCCESS_INTERVAL:
        return 1
    return 2  # 2+ successful reviews
