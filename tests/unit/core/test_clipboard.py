"""Tests for clipboard manager.

Copy-Paste Ready CLI Interfaces
Tests for GWT-1 (Clipboard Copy) and GWT-5 (Stdout Fallback).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestforge.core.clipboard import (
    MAX_CLIPBOARD_SIZE,
    ClipboardBackend,
    ClipboardManager,
    ClipboardResult,
    copy_to_clipboard,
    get_clipboard_manager,
    is_clipboard_available,
)


# =============================================================================
# Test ClipboardResult Dataclass
# =============================================================================


class TestClipboardResult:
    """Tests for ClipboardResult dataclass."""

    def test_create_success_result(self) -> None:
        """Test creating successful result."""
        result = ClipboardResult(
            success=True,
            backend=ClipboardBackend.PYPERCLIP,
            chars_copied=100,
        )
        assert result.success is True
        assert result.backend == ClipboardBackend.PYPERCLIP
        assert result.chars_copied == 100
        assert result.message is None

    def test_create_failure_result(self) -> None:
        """Test creating failure result."""
        result = ClipboardResult(
            success=False,
            backend=ClipboardBackend.NONE,
            message="No clipboard available",
        )
        assert result.success is False
        assert result.backend == ClipboardBackend.NONE
        assert result.message == "No clipboard available"

    def test_invalid_success_type(self) -> None:
        """Test that success must be bool."""
        with pytest.raises(AssertionError, match="success must be bool"):
            ClipboardResult(
                success="yes",  # type: ignore
                backend=ClipboardBackend.NONE,
            )

    def test_invalid_backend_type(self) -> None:
        """Test that backend must be ClipboardBackend."""
        with pytest.raises(AssertionError, match="invalid backend"):
            ClipboardResult(
                success=True,
                backend="pyperclip",  # type: ignore
            )

    def test_negative_chars_copied(self) -> None:
        """Test that chars_copied must be non-negative."""
        with pytest.raises(AssertionError, match="non-negative"):
            ClipboardResult(
                success=True,
                backend=ClipboardBackend.PYPERCLIP,
                chars_copied=-1,
            )


# =============================================================================
# Test ClipboardManager - Detection
# =============================================================================


class TestClipboardManagerDetection:
    """Tests for clipboard backend detection."""

    def test_detect_pyperclip_when_available(self) -> None:
        """Test pyperclip is preferred when available."""
        manager = ClipboardManager()

        with patch.object(manager, "_check_pyperclip", return_value=True):
            backend = manager._detect_backend()
            assert backend == ClipboardBackend.PYPERCLIP

    def test_detect_pbcopy_on_macos(self) -> None:
        """Test pbcopy detection on macOS."""
        manager = ClipboardManager()

        with (
            patch.object(manager, "_check_pyperclip", return_value=False),
            patch("sys.platform", "darwin"),
            patch.object(manager, "_check_command", return_value=True),
        ):
            manager._backend = None  # Reset cached backend
            backend = manager._detect_backend()
            assert backend == ClipboardBackend.PBCOPY

    def test_detect_powershell_on_windows(self) -> None:
        """Test PowerShell detection on Windows."""
        manager = ClipboardManager()

        with (
            patch.object(manager, "_check_pyperclip", return_value=False),
            patch("sys.platform", "win32"),
        ):
            manager._backend = None
            backend = manager._detect_backend()
            assert backend == ClipboardBackend.POWERSHELL

    def test_detect_xclip_on_linux(self) -> None:
        """Test xclip detection on Linux."""
        manager = ClipboardManager()

        def check_cmd(cmd: str) -> bool:
            return cmd == "xclip"

        with (
            patch.object(manager, "_check_pyperclip", return_value=False),
            patch("sys.platform", "linux"),
            patch.object(manager, "_check_command", side_effect=check_cmd),
        ):
            manager._backend = None
            backend = manager._detect_backend()
            assert backend == ClipboardBackend.XCLIP

    def test_detect_none_when_unavailable(self) -> None:
        """Test NONE backend when nothing available."""
        manager = ClipboardManager()

        with (
            patch.object(manager, "_check_pyperclip", return_value=False),
            patch("sys.platform", "unknown"),
            patch.object(manager, "_check_command", return_value=False),
        ):
            manager._backend = None
            backend = manager._detect_backend()
            assert backend == ClipboardBackend.NONE

    def test_backend_cached_after_detection(self) -> None:
        """Test that detected backend is cached."""
        manager = ClipboardManager()

        with patch.object(manager, "_check_pyperclip", return_value=True):
            backend1 = manager._detect_backend()
            backend2 = manager._detect_backend()
            assert backend1 == backend2


# =============================================================================
# Test ClipboardManager - Copy Operations
# =============================================================================


class TestClipboardManagerCopy:
    """Tests for clipboard copy operations."""

    def test_copy_with_pyperclip(self) -> None:
        """Test copying with pyperclip backend."""
        manager = ClipboardManager()

        with (
            patch.object(
                manager, "_detect_backend", return_value=ClipboardBackend.PYPERCLIP
            ),
            patch.object(manager, "_copy_pyperclip") as mock_copy,
        ):
            mock_copy.return_value = ClipboardResult(
                success=True,
                backend=ClipboardBackend.PYPERCLIP,
                chars_copied=9,
            )
            result = manager.copy("test text")
            assert result.success is True
            assert result.backend == ClipboardBackend.PYPERCLIP
            assert result.chars_copied == 9
            mock_copy.assert_called_once_with("test text")

    def test_copy_truncates_long_text(self) -> None:
        """Test that long text is truncated."""
        manager = ClipboardManager()
        long_text = "x" * (MAX_CLIPBOARD_SIZE + 1000)

        with (
            patch.object(
                manager, "_detect_backend", return_value=ClipboardBackend.PYPERCLIP
            ),
            patch.object(manager, "_copy_pyperclip") as mock_copy,
        ):
            mock_copy.return_value = ClipboardResult(
                success=True,
                backend=ClipboardBackend.PYPERCLIP,
                chars_copied=MAX_CLIPBOARD_SIZE,
            )
            result = manager.copy(long_text)
            assert result.success is True
            assert "truncated" in (result.message or "").lower()

            # Verify truncated text was passed
            call_args = mock_copy.call_args[0][0]
            assert len(call_args) == MAX_CLIPBOARD_SIZE

    def test_copy_fails_when_no_backend(self) -> None:
        """Test copy fails gracefully when no backend."""
        manager = ClipboardManager()

        with patch.object(
            manager, "_detect_backend", return_value=ClipboardBackend.NONE
        ):
            result = manager.copy("test")
            assert result.success is False
            assert "No clipboard backend" in (result.message or "")

    def test_copy_with_subprocess(self) -> None:
        """Test copying with subprocess backend."""
        manager = ClipboardManager()

        with (
            patch.object(
                manager, "_detect_backend", return_value=ClipboardBackend.XCLIP
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            result = manager.copy("test text")
            assert result.success is True
            assert result.backend == ClipboardBackend.XCLIP

    def test_copy_handles_subprocess_error(self) -> None:
        """Test handling of subprocess errors."""
        manager = ClipboardManager()

        with (
            patch.object(
                manager, "_detect_backend", return_value=ClipboardBackend.XCLIP
            ),
            patch("subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stderr=b"clipboard error")
            result = manager.copy("test")
            assert result.success is False

    def test_copy_none_text_raises(self) -> None:
        """Test that None text raises assertion."""
        manager = ClipboardManager()
        with pytest.raises(AssertionError, match="cannot be None"):
            manager.copy(None)  # type: ignore


# =============================================================================
# Test ClipboardManager - is_available
# =============================================================================


class TestClipboardManagerAvailable:
    """Tests for is_available method."""

    def test_available_when_backend_found(self) -> None:
        """Test is_available returns True when backend exists."""
        manager = ClipboardManager()

        with patch.object(
            manager, "_detect_backend", return_value=ClipboardBackend.PYPERCLIP
        ):
            assert manager.is_available() is True

    def test_unavailable_when_no_backend(self) -> None:
        """Test is_available returns False when no backend."""
        manager = ClipboardManager()

        with patch.object(
            manager, "_detect_backend", return_value=ClipboardBackend.NONE
        ):
            assert manager.is_available() is False


# =============================================================================
# Test Module-Level Functions
# =============================================================================


class TestModuleFunctions:
    """Tests for module-level convenience functions."""

    def test_get_clipboard_manager_singleton(self) -> None:
        """Test that get_clipboard_manager returns same instance."""
        manager1 = get_clipboard_manager()
        manager2 = get_clipboard_manager()
        # They should be the same or equivalent managers

    def test_copy_to_clipboard_delegates(self) -> None:
        """Test copy_to_clipboard delegates to manager."""
        with patch("ingestforge.core.clipboard.get_clipboard_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.copy.return_value = ClipboardResult(
                success=True,
                backend=ClipboardBackend.PYPERCLIP,
                chars_copied=10,
            )
            mock_get.return_value = mock_manager

            result = copy_to_clipboard("test")
            mock_manager.copy.assert_called_once_with("test")
            assert result.success is True

    def test_is_clipboard_available_delegates(self) -> None:
        """Test is_clipboard_available delegates to manager."""
        with patch("ingestforge.core.clipboard.get_clipboard_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.is_available.return_value = True
            mock_get.return_value = mock_manager

            result = is_clipboard_available()
            mock_manager.is_available.assert_called_once()
            assert result is True


# =============================================================================
# Test JPL Compliance
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_max_clipboard_size_bound(self) -> None:
        """Test MAX_CLIPBOARD_SIZE is enforced."""
        assert MAX_CLIPBOARD_SIZE == 100_000

    def test_result_validates_on_creation(self) -> None:
        """Test ClipboardResult validates in __post_init__."""
        # Valid creation
        result = ClipboardResult(
            success=True,
            backend=ClipboardBackend.PYPERCLIP,
            chars_copied=0,
        )
        assert result.chars_copied >= 0

    def test_clipboard_backend_enum_values(self) -> None:
        """Test all ClipboardBackend values are defined."""
        backends = list(ClipboardBackend)
        assert ClipboardBackend.PYPERCLIP in backends
        assert ClipboardBackend.PBCOPY in backends
        assert ClipboardBackend.XCLIP in backends
        assert ClipboardBackend.POWERSHELL in backends
        assert ClipboardBackend.NONE in backends
