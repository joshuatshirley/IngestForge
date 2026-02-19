"""Tests for rating_prompt module (SRS-002.1).

Tests the difficulty rating input handler with:
- Rating definitions and SM-2 quality mapping
- Keyboard shortcuts
- Rating panel display
- Feedback display
- Rating statistics
"""

from rich.console import Console
from rich.panel import Panel

from ingestforge.cli.ui.rating_prompt import (
    Rating,
    RATINGS,
    SHORTCUTS,
    get_rating_prompt_panel,
    display_rating_feedback,
    get_sm2_quality,
    get_rating_stats_display,
)


class TestRatingDefinitions:
    """Test rating definitions and structure."""

    def test_ratings_has_four_levels(self) -> None:
        """Rating system should have exactly 4 levels."""
        assert len(RATINGS) == 4
        assert set(RATINGS.keys()) == {1, 2, 3, 4}

    def test_rating_1_is_again(self) -> None:
        """Rating 1 should be 'Again' (fail)."""
        rating = RATINGS[1]
        assert rating.value == 1
        assert rating.label == "Again"
        assert rating.color == "red"

    def test_rating_2_is_hard(self) -> None:
        """Rating 2 should be 'Hard'."""
        rating = RATINGS[2]
        assert rating.value == 2
        assert rating.label == "Hard"
        assert rating.color == "yellow"

    def test_rating_3_is_good(self) -> None:
        """Rating 3 should be 'Good'."""
        rating = RATINGS[3]
        assert rating.value == 3
        assert rating.label == "Good"
        assert rating.color == "green"

    def test_rating_4_is_easy(self) -> None:
        """Rating 4 should be 'Easy'."""
        rating = RATINGS[4]
        assert rating.value == 4
        assert rating.label == "Easy"
        assert rating.color == "cyan"


class TestSM2QualityMapping:
    """Test SM-2 quality mapping."""

    def test_rating_1_maps_to_quality_1(self) -> None:
        """Rating 1 (Again) should map to SM-2 quality 1."""
        assert RATINGS[1].quality == 1

    def test_rating_2_maps_to_quality_3(self) -> None:
        """Rating 2 (Hard) should map to SM-2 quality 3."""
        assert RATINGS[2].quality == 3

    def test_rating_3_maps_to_quality_4(self) -> None:
        """Rating 3 (Good) should map to SM-2 quality 4."""
        assert RATINGS[3].quality == 4

    def test_rating_4_maps_to_quality_5(self) -> None:
        """Rating 4 (Easy) should map to SM-2 quality 5."""
        assert RATINGS[4].quality == 5

    def test_get_sm2_quality_valid(self) -> None:
        """get_sm2_quality should return correct mapping."""
        assert get_sm2_quality(1) == 1
        assert get_sm2_quality(2) == 3
        assert get_sm2_quality(3) == 4
        assert get_sm2_quality(4) == 5

    def test_get_sm2_quality_invalid(self) -> None:
        """Invalid rating should return default quality 3."""
        assert get_sm2_quality(0) == 3
        assert get_sm2_quality(5) == 3
        assert get_sm2_quality(99) == 3


class TestKeyboardShortcuts:
    """Test keyboard shortcuts."""

    def test_numeric_shortcuts(self) -> None:
        """Numeric keys should map to ratings."""
        assert SHORTCUTS["1"] == 1
        assert SHORTCUTS["2"] == 2
        assert SHORTCUTS["3"] == 3
        assert SHORTCUTS["4"] == 4

    def test_letter_shortcuts(self) -> None:
        """Letter shortcuts should map to ratings."""
        assert SHORTCUTS["a"] == 1  # Again
        assert SHORTCUTS["h"] == 2  # Hard
        assert SHORTCUTS["g"] == 3  # Good
        assert SHORTCUTS["e"] == 4  # Easy


class TestRatingPromptPanel:
    """Test rating prompt panel."""

    def test_panel_is_returned(self) -> None:
        """get_rating_prompt_panel should return a Panel."""
        panel = get_rating_prompt_panel()
        assert isinstance(panel, Panel)

    def test_panel_has_title(self) -> None:
        """Panel should have a title."""
        panel = get_rating_prompt_panel()
        assert panel.title is not None

    def test_panel_without_shortcuts(self) -> None:
        """Panel can hide shortcuts."""
        panel = get_rating_prompt_panel(show_shortcuts=False)
        assert isinstance(panel, Panel)


class TestRatingFeedback:
    """Test rating feedback display."""

    def test_display_feedback_does_not_raise(self) -> None:
        """display_rating_feedback should not raise."""
        console = Console(force_terminal=True, width=80, record=True)
        rating = RATINGS[3]
        # Should not raise
        display_rating_feedback(rating, console)

    def test_display_feedback_includes_label(self) -> None:
        """Feedback should include the rating label."""
        console = Console(force_terminal=True, width=80, record=True)
        rating = RATINGS[3]
        display_rating_feedback(rating, console)
        output = console.export_text()
        assert "Good" in output


class TestRatingStatsDisplay:
    """Test rating statistics display."""

    def test_empty_ratings(self) -> None:
        """Empty ratings should show message."""
        console = Console(force_terminal=True, width=80, record=True)
        get_rating_stats_display({}, console)
        output = console.export_text()
        assert "No ratings" in output

    def test_ratings_display(self) -> None:
        """Ratings should display breakdown."""
        console = Console(force_terminal=True, width=80, record=True)
        ratings = {1: 2, 2: 3, 3: 10, 4: 5}
        get_rating_stats_display(ratings, console)
        output = console.export_text()
        assert "Breakdown" in output


class TestRatingDataclass:
    """Test Rating dataclass."""

    def test_rating_attributes(self) -> None:
        """Rating should have all required attributes."""
        rating = Rating(
            value=3,
            quality=4,
            label="Good",
            color="green",
            description="Test description",
        )
        assert rating.value == 3
        assert rating.quality == 4
        assert rating.label == "Good"
        assert rating.color == "green"
        assert rating.description == "Test description"
