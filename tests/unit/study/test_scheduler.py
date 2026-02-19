"""Tests for SM-2 scheduler module (TICKET-102).

Tests the SM-2 spaced repetition algorithm:
- First review intervals
- Second review intervals
- Subsequent review intervals
- Failed review resets
- Ease factor calculations
- Edge cases and boundary conditions

Test cases based on standard Anki/SuperMemo SM-2 behavior.
"""

import pytest

from ingestforge.study.scheduler import (
    calculate_next_interval,
    get_initial_ease_factor,
    get_initial_interval,
    MIN_EASE_FACTOR,
    FIRST_SUCCESS_INTERVAL,
    SECOND_SUCCESS_INTERVAL,
    FAIL_RESET_INTERVAL,
    get_review_count,
)


class TestConstants:
    """Test SM-2 algorithm constants."""

    def test_min_ease_factor(self) -> None:
        """Minimum ease factor should be 1.3."""
        assert MIN_EASE_FACTOR == 1.3

    def test_first_success_interval(self) -> None:
        """First success interval should be 1 day."""
        assert FIRST_SUCCESS_INTERVAL == 1.0

    def test_second_success_interval(self) -> None:
        """Second success interval should be 6 days."""
        assert SECOND_SUCCESS_INTERVAL == 6.0

    def test_fail_reset_interval(self) -> None:
        """Failed review reset should be 1 day."""
        assert FAIL_RESET_INTERVAL == 1.0


class TestHelperFunctions:
    """Test helper functions for initial values."""

    def test_get_initial_ease_factor(self) -> None:
        """Initial ease factor should be 2.5."""
        assert get_initial_ease_factor() == 2.5

    def test_get_initial_interval(self) -> None:
        """Initial interval should be 1.0 day."""
        assert get_initial_interval() == 1.0


class TestQualityValidation:
    """Test quality parameter validation."""

    def test_quality_below_range_raises(self) -> None:
        """Quality below 0 should raise AssertionError."""
        with pytest.raises(AssertionError, match="Quality must be 0-5"):
            calculate_next_interval(-1, 0, 2.5)

    def test_quality_above_range_raises(self) -> None:
        """Quality above 5 should raise AssertionError."""
        with pytest.raises(AssertionError, match="Quality must be 0-5"):
            calculate_next_interval(6, 0, 2.5)

    def test_quality_boundary_zero_valid(self) -> None:
        """Quality 0 should be valid."""
        interval, ef = calculate_next_interval(0, 0, 2.5)
        assert interval == FAIL_RESET_INTERVAL

    def test_quality_boundary_five_valid(self) -> None:
        """Quality 5 should be valid."""
        interval, ef = calculate_next_interval(5, 0, 2.5)
        assert interval == FIRST_SUCCESS_INTERVAL


class TestFailedReviews:
    """Test behavior when quality < 3 (failed review)."""

    @pytest.mark.parametrize("quality", [0, 1, 2])
    def test_failed_review_resets_interval(self, quality: int) -> None:
        """Failed reviews (q < 3) should reset interval to 1 day."""
        interval, ef = calculate_next_interval(quality, 30.0, 2.5)
        assert interval == FAIL_RESET_INTERVAL

    @pytest.mark.parametrize("quality", [0, 1, 2])
    def test_failed_review_keeps_ease_factor(self, quality: int) -> None:
        """Failed reviews should preserve ease factor."""
        original_ef = 2.5
        interval, ef = calculate_next_interval(quality, 30.0, original_ef)
        assert ef == original_ef

    def test_failed_from_long_interval(self) -> None:
        """Even cards with long intervals reset on failure."""
        interval, ef = calculate_next_interval(2, 180.0, 2.5)
        assert interval == FAIL_RESET_INTERVAL
        assert ef == 2.5


class TestFirstSuccessfulReview:
    """Test first successful review (prev_interval < 1)."""

    @pytest.mark.parametrize("quality", [3, 4, 5])
    def test_first_review_interval_is_one(self, quality: int) -> None:
        """First successful review should return 1 day interval."""
        interval, ef = calculate_next_interval(quality, 0, 2.5)
        assert interval == FIRST_SUCCESS_INTERVAL

    def test_first_review_from_zero_interval(self) -> None:
        """New card (interval=0) should get 1 day on success."""
        interval, ef = calculate_next_interval(4, 0, 2.5)
        assert interval == 1.0

    def test_first_review_from_partial_interval(self) -> None:
        """Cards with interval < 1 should get 1 day on success."""
        interval, ef = calculate_next_interval(4, 0.5, 2.5)
        assert interval == 1.0


class TestSecondSuccessfulReview:
    """Test second successful review (1 <= prev_interval < 6)."""

    @pytest.mark.parametrize("quality", [3, 4, 5])
    def test_second_review_interval_is_six(self, quality: int) -> None:
        """Second successful review should return 6 day interval."""
        interval, ef = calculate_next_interval(quality, 1.0, 2.5)
        assert interval == SECOND_SUCCESS_INTERVAL

    def test_second_review_from_interval_three(self) -> None:
        """Cards with interval 1-6 should get 6 days on success."""
        interval, ef = calculate_next_interval(4, 3.0, 2.5)
        assert interval == 6.0


class TestSubsequentReviews:
    """Test reviews after second (prev_interval >= 6)."""

    def test_third_review_multiplies_interval(self) -> None:
        """Third+ reviews multiply interval by ease factor."""
        interval, ef = calculate_next_interval(4, 6.0, 2.5)
        assert interval == 6.0 * 2.5
        assert interval == 15.0

    def test_subsequent_review_with_default_ease(self) -> None:
        """Standard progression with default 2.5 ease factor."""
        interval, ef = calculate_next_interval(4, 15.0, 2.5)
        assert interval == 15.0 * 2.5
        assert interval == 37.5

    def test_subsequent_review_with_high_ease(self) -> None:
        """High ease factor increases intervals faster."""
        interval, ef = calculate_next_interval(5, 10.0, 3.0)
        assert interval == 10.0 * 3.0
        assert interval == 30.0


class TestEaseFactorCalculation:
    """Test ease factor adjustments."""

    def test_perfect_quality_increases_ease(self) -> None:
        """Quality 5 (perfect) should increase ease factor."""
        _, ef = calculate_next_interval(5, 6.0, 2.5)
        assert ef > 2.5
        # EF' = 2.5 + (0.1 - 0 * (0.08 + 0 * 0.02)) = 2.5 + 0.1 = 2.6
        assert abs(ef - 2.6) < 0.001

    def test_good_quality_slightly_increases_ease(self) -> None:
        """Quality 4 should slightly increase ease factor."""
        _, ef = calculate_next_interval(4, 6.0, 2.5)
        # EF' = 2.5 + (0.1 - (-1) * (0.08 + (-1) * 0.02))
        # EF' = 2.5 + (0.1 + 0.06) = 2.5 + 0.0 = 2.5
        assert ef >= 2.5

    def test_difficult_quality_decreases_ease(self) -> None:
        """Quality 3 (difficult) should decrease ease factor."""
        _, ef = calculate_next_interval(3, 6.0, 2.5)
        # EF' = 2.5 + (0.1 - (-2) * (0.08 + (-2) * 0.02))
        # EF' = 2.5 + (0.1 - 2 * 0.04) = 2.5 + (0.1 - 0.08) = 2.5 - 0.14 = 2.36
        assert ef < 2.5

    def test_ease_factor_never_below_minimum(self) -> None:
        """Ease factor should never drop below 1.3."""
        # Start with low ease factor and give quality 3
        _, ef = calculate_next_interval(3, 6.0, 1.3)
        assert ef == MIN_EASE_FACTOR

    def test_ease_factor_minimum_enforced_on_input(self) -> None:
        """Ease factor below 1.3 on input should be raised to 1.3."""
        interval, ef = calculate_next_interval(0, 10.0, 1.0)
        # Failed review keeps ease factor, but at minimum
        assert ef == MIN_EASE_FACTOR


class TestStandardAnkiProgressions:
    """Test standard Anki-like card progressions."""

    def test_new_card_perfect_progression(self) -> None:
        """New card with perfect reviews progresses correctly."""
        # First review: perfect (5)
        interval, ef = calculate_next_interval(5, 0, 2.5)
        assert interval == 1.0
        assert abs(ef - 2.6) < 0.001

        # Second review: perfect (5)
        interval, ef = calculate_next_interval(5, 1.0, ef)
        assert interval == 6.0
        assert abs(ef - 2.7) < 0.001

        # Third review: perfect (5)
        interval, ef = calculate_next_interval(5, 6.0, ef)
        assert abs(interval - 16.2) < 0.1  # 6 * 2.7 = 16.2
        assert abs(ef - 2.8) < 0.001

    def test_card_with_struggle(self) -> None:
        """Card that user struggles with (quality 3)."""
        # First review: struggled (3)
        interval, ef = calculate_next_interval(3, 0, 2.5)
        assert interval == 1.0
        assert ef < 2.5

        # Second review: still struggling (3)
        interval, ef = calculate_next_interval(3, 1.0, ef)
        assert interval == 6.0

    def test_card_fails_after_progression(self) -> None:
        """Card that was learned but then forgotten."""
        # Progress to 30 day interval
        interval, ef = calculate_next_interval(4, 10.0, 2.5)
        assert interval > 20

        # Then fail
        interval, ef = calculate_next_interval(2, interval, ef)
        assert interval == 1.0  # Reset to start


class TestEaseFactorFormula:
    """Test precise ease factor formula values."""

    @pytest.mark.parametrize(
        "quality,expected_delta",
        [
            (5, 0.10),  # Perfect: +0.10
            (4, 0.00),  # Good: +0.00
            (3, -0.14),  # Difficult: -0.14
        ],
    )
    def test_ease_factor_deltas(self, quality: int, expected_delta: float) -> None:
        """Test ease factor change for each quality level."""
        base_ef = 2.5
        _, new_ef = calculate_next_interval(quality, 6.0, base_ef)

        actual_delta = new_ef - base_ef
        # Allow small floating point tolerance
        assert abs(actual_delta - expected_delta) < 0.001


class TestGetReviewCount:
    """Test review count estimation."""

    def test_zero_interval_is_zero_reviews(self) -> None:
        """New card has 0 reviews."""
        assert get_review_count(0) == 0

    def test_partial_interval_is_zero_reviews(self) -> None:
        """Interval < 1 has 0 successful reviews."""
        assert get_review_count(0.5) == 0

    def test_one_day_interval_is_one_review(self) -> None:
        """After first review, interval is 1 day."""
        assert get_review_count(1.0) == 1

    def test_three_day_interval_is_one_review(self) -> None:
        """Interval between 1-6 indicates 1 successful review."""
        assert get_review_count(3.0) == 1

    def test_six_day_interval_is_two_reviews(self) -> None:
        """After second review, interval is 6+ days."""
        assert get_review_count(6.0) == 2

    def test_long_interval_is_two_plus_reviews(self) -> None:
        """Long intervals indicate 2+ successful reviews."""
        assert get_review_count(100.0) == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_interval(self) -> None:
        """Very small intervals should work correctly."""
        interval, ef = calculate_next_interval(4, 0.001, 2.5)
        assert interval == 1.0

    def test_very_large_interval(self) -> None:
        """Very large intervals should multiply correctly."""
        interval, ef = calculate_next_interval(4, 365.0, 2.5)
        assert interval == 365.0 * 2.5

    def test_minimum_ease_factor_maintained(self) -> None:
        """Repeated struggles should maintain minimum ease."""
        ef = 2.5
        # Multiple quality-3 reviews should bottom out at 1.3
        for _ in range(20):
            _, ef = calculate_next_interval(3, 6.0, ef)

        assert ef == MIN_EASE_FACTOR

    def test_ease_factor_at_minimum_on_fail(self) -> None:
        """Failed review with minimum ease factor stays at minimum."""
        interval, ef = calculate_next_interval(0, 10.0, MIN_EASE_FACTOR)
        assert ef == MIN_EASE_FACTOR
        assert interval == FAIL_RESET_INTERVAL

    def test_all_quality_levels(self) -> None:
        """All valid quality levels should work."""
        for q in range(6):
            interval, ef = calculate_next_interval(q, 6.0, 2.5)
            assert interval > 0
            assert ef >= MIN_EASE_FACTOR
