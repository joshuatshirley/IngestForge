"""Unit tests for Quick Bar functionality.

Tests the Quick Bar implementation in the interactive menu."""

from typing import Set
from unittest.mock import patch

import pytest

from ingestforge.cli.interactive.menu import (
    QUICK_BAR,
    QUICK_BAR_HANDLERS,
    QUICK_BAR_KEYS,
    display_quick_bar,
    handle_quick_bar,
)


class TestQuickBarConstants:
    """Tests for Quick Bar constant definitions."""

    def test_quick_bar_has_expected_items(self) -> None:
        """Verify Quick Bar has all expected items."""
        expected_keys: Set[str] = {"q", "i", "a", "s", "f", "e", "g", "?"}
        actual_keys: Set[str] = {item[0] for item in QUICK_BAR}
        assert actual_keys == expected_keys

    def test_quick_bar_keys_match_items(self) -> None:
        """Verify QUICK_BAR_KEYS matches QUICK_BAR items."""
        item_keys: Set[str] = {item[0] for item in QUICK_BAR}
        assert QUICK_BAR_KEYS == item_keys

    def test_quick_bar_item_structure(self) -> None:
        """Verify each Quick Bar item has (key, label, command)."""
        for item in QUICK_BAR:
            assert len(item) == 3
            key, label, command = item
            assert isinstance(key, str) and len(key) == 1
            assert isinstance(label, str) and len(label) > 0
            assert isinstance(command, str) and len(command) > 0

    def test_quick_bar_handlers_exist_for_all_keys(self) -> None:
        """Verify every Quick Bar key has a handler."""
        missing: Set[str] = QUICK_BAR_KEYS - set(QUICK_BAR_HANDLERS.keys())
        assert missing == set(), f"Missing handlers: {missing}"

    def test_no_extra_handlers(self) -> None:
        """Verify no extra handlers exist."""
        extra: Set[str] = set(QUICK_BAR_HANDLERS.keys()) - QUICK_BAR_KEYS
        assert extra == set(), f"Extra handlers: {extra}"


class TestHandleQuickBar:
    """Tests for handle_quick_bar function."""

    def test_handle_valid_key(self) -> None:
        """Verify valid keys are handled and return True."""
        # Mock the handler to prevent actual execution
        mock_handler = lambda: None
        with patch.dict(QUICK_BAR_HANDLERS, {"q": mock_handler}):
            result: bool = handle_quick_bar("q")
            assert result is True

    def test_handle_invalid_key(self) -> None:
        """Verify invalid keys return False."""
        result: bool = handle_quick_bar("z")
        assert result is False

    def test_handle_empty_key_raises(self) -> None:
        """Verify empty key raises AssertionError."""
        with pytest.raises(AssertionError):
            handle_quick_bar("")

    def test_handle_case_insensitive(self) -> None:
        """Verify handler lookup is case-insensitive."""
        # Uppercase Q should find lowercase handler
        with patch.dict(QUICK_BAR_HANDLERS, {"q": lambda: None}):
            result: bool = handle_quick_bar("Q")
            assert result is True


class TestDisplayQuickBar:
    """Tests for display_quick_bar function."""

    def test_display_does_not_raise(self) -> None:
        """Verify display_quick_bar runs without error."""
        # Capture output to prevent test noise
        with patch("ingestforge.cli.interactive.menu.console"):
            display_quick_bar()

    def test_display_shows_all_keys(self) -> None:
        """Verify all Quick Bar keys are shown in display."""
        from io import StringIO

        from rich.console import Console

        # Capture output
        output = StringIO()
        test_console = Console(file=output, force_terminal=True, width=100)

        with patch("ingestforge.cli.interactive.menu.console", test_console):
            display_quick_bar()

        rendered: str = output.getvalue()

        # Verify all keys appear in output
        for key, label, _ in QUICK_BAR:
            assert key.upper() in rendered, f"Missing key: {key}"
            assert label in rendered, f"Missing label: {label}"


class TestQuickBarIntegration:
    """Integration tests for Quick Bar in menu context."""

    def test_quick_bar_keys_do_not_conflict_with_menu(self) -> None:
        """Verify Quick Bar keys don't conflict with main menu keys unexpectedly."""
        from ingestforge.cli.interactive.menu import MAIN_MENU

        menu_keys: Set[str] = {item[0] for item in MAIN_MENU}
        # 'q' (quit) and 's' (status) are intentionally shared
        shared_expected: Set[str] = {"q", "s"}
        actual_shared: Set[str] = QUICK_BAR_KEYS & menu_keys

        assert (
            actual_shared == shared_expected
        ), f"Unexpected conflicts: {actual_shared - shared_expected}"

    def test_quick_bar_has_reasonable_size(self) -> None:
        """Verify Quick Bar isn't too large for display."""
        max_items: int = 10  # Upper bound for usability
        assert len(QUICK_BAR) <= max_items
