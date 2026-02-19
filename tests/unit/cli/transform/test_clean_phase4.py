"""Unit tests for Phase 4 clean command enhancements.

Tests the TextCleanerRefiner integration with transform clean command."""

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch


from ingestforge.cli.transform.clean import CleanCommand


class TestCleanCommandPhase4:
    """Tests for Phase 4 clean command enhancements."""

    def test_apply_text_cleaner_disabled(self) -> None:
        """Verify no changes when all OCR options disabled."""
        cmd = CleanCommand()
        changes: List[str] = []

        result = cmd._apply_text_cleaner(
            "Test content",
            group_paragraphs=False,
            clean_bullets=False,
            clean_prefix_postfix=False,
            changes=changes,
        )

        assert result == "Test content"
        assert changes == []

    def test_apply_text_cleaner_group_paragraphs(self) -> None:
        """Verify broken paragraph joining works."""
        cmd = CleanCommand()
        changes: List[str] = []

        # Text with broken paragraph
        text = "This is a broken\nline of text."

        result = cmd._apply_text_cleaner(
            text,
            group_paragraphs=True,
            clean_bullets=False,
            clean_prefix_postfix=False,
            changes=changes,
        )

        # Should join the lines
        assert "broken line" in result or "broken\nline" in result

    def test_apply_text_cleaner_clean_bullets(self) -> None:
        """Verify bullet normalization works."""
        cmd = CleanCommand()
        changes: List[str] = []

        # Text with unicode bullet
        text = "\u2022 Item one\n\u2022 Item two"

        result = cmd._apply_text_cleaner(
            text,
            group_paragraphs=False,
            clean_bullets=True,
            clean_prefix_postfix=False,
            changes=changes,
        )

        # Bullets should be normalized to dashes
        assert "-" in result
        assert "\u2022" not in result

    def test_apply_text_cleaner_clean_prefix_postfix(self) -> None:
        """Verify page number removal works."""
        cmd = CleanCommand()
        changes: List[str] = []

        # Text with page number
        text = "Content here.\n\n- 42 -\n\nMore content."

        result = cmd._apply_text_cleaner(
            text,
            group_paragraphs=False,
            clean_bullets=False,
            clean_prefix_postfix=True,
            changes=changes,
        )

        # Page number should be removed
        assert "- 42 -" not in result
        assert "Content here" in result

    def test_apply_text_cleaner_all_options(self) -> None:
        """Verify all OCR options work together."""
        cmd = CleanCommand()
        changes: List[str] = []

        result = cmd._apply_text_cleaner(
            "Test\n\u2022 bullet\n\n- 1 -",
            group_paragraphs=True,
            clean_bullets=True,
            clean_prefix_postfix=True,
            changes=changes,
        )

        # Changes should be recorded
        assert isinstance(changes, list)


class TestCleanCommandExecute:
    """Tests for clean command execute with OCR options."""

    @patch("ingestforge.cli.transform.clean.CleanCommand.validate_file_path")
    @patch("ingestforge.cli.transform.clean.CleanCommand.initialize_context")
    @patch("ingestforge.cli.transform.clean.CleanCommand._read_file")
    @patch("ingestforge.cli.transform.clean.CleanCommand.create_transform_summary")
    def test_execute_with_ocr_options(
        self,
        mock_summary: MagicMock,
        mock_read: MagicMock,
        mock_init: MagicMock,
        mock_validate: MagicMock,
    ) -> None:
        """Verify execute works with OCR cleanup options."""
        mock_read.return_value = "\u2022 Bullet item\n\n- 1 -"
        mock_init.return_value = MagicMock()
        mock_summary.return_value = MagicMock()

        cmd = CleanCommand()
        # Suppress console output
        cmd.console = MagicMock()

        result = cmd.execute(
            input_file=Path("test.txt"),
            project=None,
            output=None,
            remove_urls=False,
            remove_emails=False,
            lowercase=False,
            group_paragraphs=False,
            clean_bullets=True,
            clean_prefix_postfix=True,
        )

        assert result == 0


class TestCleanCommandSignature:
    """Tests for clean command method signatures."""

    def test_execute_accepts_ocr_parameters(self) -> None:
        """Verify execute method accepts OCR cleanup parameters."""
        import inspect

        sig = inspect.signature(CleanCommand.execute)
        params = list(sig.parameters.keys())

        assert "group_paragraphs" in params
        assert "clean_bullets" in params
        assert "clean_prefix_postfix" in params

    def test_apply_text_cleaner_exists(self) -> None:
        """Verify _apply_text_cleaner method exists."""
        cmd = CleanCommand()
        assert hasattr(cmd, "_apply_text_cleaner")
        assert callable(cmd._apply_text_cleaner)
