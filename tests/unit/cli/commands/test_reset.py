"""Tests for reset command (DEAD-D04).

NASA JPL Commandments compliance:
- Rule #1: Linear test structure
- Rule #4: Functions <60 lines
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from ingestforge.cli.commands.reset import ResetCommand


class TestResetCommand:
    """Tests for ResetCommand class."""

    def test_command_instantiation(self) -> None:
        """Command can be instantiated."""
        cmd = ResetCommand()
        assert cmd is not None

    def test_show_warning(self) -> None:
        """Warning is displayed with project info."""
        cmd = ResetCommand()

        # Mock config
        mock_config = MagicMock()
        mock_config.project.name = "test-project"
        mock_config.project.data_dir = ".data"

        # Should not raise
        cmd._show_warning(mock_config)

    def test_get_confirmation_positive(self) -> None:
        """Returns True when user types RESET."""
        cmd = ResetCommand()

        with patch("typer.prompt", return_value="RESET"):
            result = cmd._get_confirmation()
            assert result is True

    def test_get_confirmation_lowercase(self) -> None:
        """Returns True when user types reset (lowercase)."""
        cmd = ResetCommand()

        with patch("typer.prompt", return_value="reset"):
            result = cmd._get_confirmation()
            assert result is True

    def test_get_confirmation_negative(self) -> None:
        """Returns False when user types something else."""
        cmd = ResetCommand()

        with patch("typer.prompt", return_value="no"):
            result = cmd._get_confirmation()
            assert result is False

    def test_get_confirmation_empty(self) -> None:
        """Returns False when user presses enter."""
        cmd = ResetCommand()

        with patch("typer.prompt", return_value=""):
            result = cmd._get_confirmation()
            assert result is False

    @patch.object(ResetCommand, "initialize_context")
    @patch.object(ResetCommand, "_show_warning")
    @patch.object(ResetCommand, "_get_confirmation")
    def test_execute_cancelled(
        self,
        mock_confirm: MagicMock,
        mock_warning: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """Execute returns 0 when user cancels."""
        mock_init.return_value = {
            "config": MagicMock(),
            "storage": MagicMock(),
            "project_path": Path("/test"),
        }
        mock_confirm.return_value = False

        cmd = ResetCommand()
        result = cmd.execute()

        assert result == 0
        mock_warning.assert_called_once()
        mock_confirm.assert_called_once()

    @patch.object(ResetCommand, "initialize_context")
    @patch.object(ResetCommand, "_show_warning")
    @patch.object(ResetCommand, "_perform_reset")
    def test_execute_force_skips_confirmation(
        self,
        mock_reset: MagicMock,
        mock_warning: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """Execute with force=True skips confirmation."""
        mock_init.return_value = {
            "config": MagicMock(),
            "storage": MagicMock(),
            "project_path": Path("/test"),
        }

        cmd = ResetCommand()
        result = cmd.execute(force=True)

        assert result == 0
        mock_warning.assert_called_once()
        mock_reset.assert_called_once()

    @patch.object(ResetCommand, "initialize_context")
    @patch.object(ResetCommand, "_show_warning")
    @patch.object(ResetCommand, "_get_confirmation")
    @patch.object(ResetCommand, "_perform_reset")
    def test_execute_confirmed(
        self,
        mock_reset: MagicMock,
        mock_confirm: MagicMock,
        mock_warning: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """Execute performs reset when user confirms."""
        mock_init.return_value = {
            "config": MagicMock(),
            "storage": MagicMock(),
            "project_path": Path("/test"),
        }
        mock_confirm.return_value = True

        cmd = ResetCommand()
        result = cmd.execute()

        assert result == 0
        mock_reset.assert_called_once()

    @patch.object(ResetCommand, "initialize_context")
    @patch.object(ResetCommand, "handle_error")
    def test_execute_handles_error(
        self,
        mock_error: MagicMock,
        mock_init: MagicMock,
    ) -> None:
        """Execute handles exceptions gracefully."""
        mock_init.side_effect = RuntimeError("Test error")
        mock_error.return_value = 1

        cmd = ResetCommand()
        result = cmd.execute()

        assert result == 1
        mock_error.assert_called_once()
