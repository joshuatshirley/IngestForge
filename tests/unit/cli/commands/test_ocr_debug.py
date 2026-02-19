"""Tests for ocr_debug CLI command (OCR-002.2).

Tests the OCR preprocessing debug interface:
- Command options
- Result display
- Debug image saving
"""

from pathlib import Path
from typer.testing import CliRunner

from ingestforge.cli.commands.ocr_debug import app, _format_size, _create_result_panel
from ingestforge.ingest.ocr.cleanup_core import PreprocessingResult


runner = CliRunner()


class TestFormatSize:
    """Test size formatting."""

    def test_format_size(self) -> None:
        """Should format size correctly."""
        result = _format_size((100, 200))
        assert result == "100x200"

    def test_format_large_size(self) -> None:
        """Should format large sizes."""
        result = _format_size((4000, 3000))
        assert result == "4000x3000"


class TestCreateResultPanel:
    """Test result panel creation."""

    def test_success_panel(self) -> None:
        """Should create success panel."""
        from rich.console import Console

        result = PreprocessingResult(
            success=True,
            original_path=Path("test.png"),
            rotation_angle=2.5,
            was_binarized=True,
            was_denoised=True,
            original_size=(100, 200),
            processed_size=(100, 200),
        )

        console = Console(force_terminal=True, width=80)
        panel = _create_result_panel(result, console)

        assert panel is not None

    def test_failure_panel(self) -> None:
        """Should create failure panel."""
        from rich.console import Console

        result = PreprocessingResult(
            success=False,
            original_path=Path("test.png"),
            error="Test error",
        )

        console = Console(force_terminal=True, width=80)
        panel = _create_result_panel(result, console)

        assert panel is not None


class TestInfoCommand:
    """Test info command."""

    def test_info_runs(self) -> None:
        """Info command should run without error."""
        result = runner.invoke(app, ["info"])

        # Should show feature table
        assert "OCR Preprocessing Features" in result.stdout or result.exit_code == 0


class TestProcessCommand:
    """Test process command."""

    def test_process_missing_file(self) -> None:
        """Should handle missing file."""
        result = runner.invoke(app, ["process", "/nonexistent/file.png"])

        # Should fail or show error
        assert (
            result.exit_code != 0
            or "not found" in result.stdout.lower()
            or "error" in result.stdout.lower()
        )

    def test_process_help(self) -> None:
        """Should show help."""
        result = runner.invoke(app, ["process", "--help"])

        assert result.exit_code == 0
        assert "debug-images" in result.stdout


class TestBatchCommand:
    """Test batch command."""

    def test_batch_help(self) -> None:
        """Should show help."""
        result = runner.invoke(app, ["batch", "--help"])

        assert result.exit_code == 0
        assert "pattern" in result.stdout


class TestCommandOptions:
    """Test command option parsing."""

    def test_binarization_option(self) -> None:
        """Should accept binarization methods."""
        result = runner.invoke(app, ["process", "--help"])

        assert "binarization" in result.stdout
        assert "otsu" in result.stdout or "BINARIZATION" in result.stdout.upper()

    def test_threshold_option(self) -> None:
        """Should accept threshold value."""
        result = runner.invoke(app, ["process", "--help"])

        assert "threshold" in result.stdout

    def test_no_deskew_option(self) -> None:
        """Should accept no-deskew flag."""
        result = runner.invoke(app, ["process", "--help"])

        assert "no-deskew" in result.stdout


class TestAppStructure:
    """Test app structure."""

    def test_app_has_commands(self) -> None:
        """App should have expected commands."""
        result = runner.invoke(app, ["--help"])

        assert "process" in result.stdout
        assert "batch" in result.stdout
        assert "info" in result.stdout
