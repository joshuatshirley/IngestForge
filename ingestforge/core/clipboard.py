"""Clipboard manager for cross-platform clipboard access.

Copy-Paste Ready CLI Interfaces
Epic: EP-08 (Structured Data Foundry)
Feature: FE-08-04 (Copy-Paste Ready CLI Interfaces)

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_CLIPBOARD_SIZE)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

# JPL Rule #2: Fixed upper bounds
MAX_CLIPBOARD_SIZE = 100_000  # 100KB max clipboard content
MAX_COMMAND_TIMEOUT = 5  # 5 second timeout for clipboard commands


class ClipboardBackend(Enum):
    """Available clipboard backends."""

    PYPERCLIP = "pyperclip"
    PBCOPY = "pbcopy"  # macOS
    XCLIP = "xclip"  # Linux X11
    XSEL = "xsel"  # Linux X11 alternative
    WSLCLIP = "clip.exe"  # WSL
    POWERSHELL = "powershell"  # Windows
    NONE = "none"


@dataclass
class ClipboardResult:
    """Result of a clipboard operation.

    Attributes:
        success: Whether the operation succeeded.
        backend: Backend used for the operation.
        message: Optional message (error or info).
        chars_copied: Number of characters copied (if applicable).
    """

    success: bool
    backend: ClipboardBackend
    message: Optional[str] = None
    chars_copied: int = 0

    def __post_init__(self) -> None:
        """Validate result fields."""
        # JPL Rule #5: Assert preconditions
        assert isinstance(self.success, bool), "success must be bool"
        assert isinstance(self.backend, ClipboardBackend), "invalid backend"
        assert self.chars_copied >= 0, "chars_copied must be non-negative"


class ClipboardManager:
    """Cross-platform clipboard manager.

    Provides a unified interface for clipboard operations across
    Windows, macOS, Linux, and WSL environments.

    JPL Compliance:
        - No recursion in any method
        - All loops bounded by MAX constants
        - Complete type hints
        - Precondition assertions
    """

    def __init__(self) -> None:
        """Initialize clipboard manager."""
        self._backend: Optional[ClipboardBackend] = None
        self._pyperclip_available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if clipboard is available on this platform.

        Returns:
            True if a clipboard backend is available.
        """
        backend = self._detect_backend()
        return backend != ClipboardBackend.NONE

    def copy(self, text: str) -> ClipboardResult:
        """Copy text to clipboard.

        Args:
            text: Text to copy. Will be truncated if exceeds MAX_CLIPBOARD_SIZE.

        Returns:
            ClipboardResult with operation status.
        """
        # JPL Rule #5: Assert preconditions
        assert text is not None, "text cannot be None"

        # JPL Rule #2: Enforce bounds
        if len(text) > MAX_CLIPBOARD_SIZE:
            text = text[:MAX_CLIPBOARD_SIZE]
            truncated = True
        else:
            truncated = False

        backend = self._detect_backend()

        if backend == ClipboardBackend.NONE:
            return ClipboardResult(
                success=False,
                backend=backend,
                message="No clipboard backend available",
            )

        # Try the detected backend
        result = self._copy_with_backend(text, backend)

        if result.success and truncated:
            result.message = f"Content truncated to {MAX_CLIPBOARD_SIZE} chars"

        return result

    def _detect_backend(self) -> ClipboardBackend:
        """Detect available clipboard backend.

        Returns:
            Best available ClipboardBackend for current platform.
        """
        if self._backend is not None:
            return self._backend

        # Try pyperclip first (most reliable if available)
        if self._check_pyperclip():
            self._backend = ClipboardBackend.PYPERCLIP
            return self._backend

        # Platform-specific fallbacks
        if sys.platform == "darwin":
            if self._check_command("pbcopy"):
                self._backend = ClipboardBackend.PBCOPY
                return self._backend

        elif sys.platform == "win32":
            self._backend = ClipboardBackend.POWERSHELL
            return self._backend

        elif sys.platform.startswith("linux"):
            # Check for WSL
            if self._check_command("clip.exe"):
                self._backend = ClipboardBackend.WSLCLIP
                return self._backend

            # X11 clipboard tools
            if self._check_command("xclip"):
                self._backend = ClipboardBackend.XCLIP
                return self._backend

            if self._check_command("xsel"):
                self._backend = ClipboardBackend.XSEL
                return self._backend

        self._backend = ClipboardBackend.NONE
        return self._backend

    def _check_pyperclip(self) -> bool:
        """Check if pyperclip is available.

        Returns:
            True if pyperclip can be imported and used.
        """
        if self._pyperclip_available is not None:
            return self._pyperclip_available

        try:
            import pyperclip

            # Test that it works
            pyperclip.determine_clipboard()
            self._pyperclip_available = True
        except Exception:
            self._pyperclip_available = False

        return self._pyperclip_available

    def _check_command(self, cmd: str) -> bool:
        """Check if a command is available.

        Args:
            cmd: Command name to check.

        Returns:
            True if command exists and is executable.
        """
        try:
            result = subprocess.run(
                ["which", cmd] if sys.platform != "win32" else ["where", cmd],
                capture_output=True,
                timeout=MAX_COMMAND_TIMEOUT,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _copy_with_backend(
        self, text: str, backend: ClipboardBackend
    ) -> ClipboardResult:
        """Copy text using specific backend.

        Args:
            text: Text to copy.
            backend: Backend to use.

        Returns:
            ClipboardResult with operation status.
        """
        try:
            if backend == ClipboardBackend.PYPERCLIP:
                return self._copy_pyperclip(text)

            elif backend == ClipboardBackend.PBCOPY:
                return self._copy_subprocess(text, ["pbcopy"], backend)

            elif backend == ClipboardBackend.XCLIP:
                return self._copy_subprocess(
                    text, ["xclip", "-selection", "clipboard"], backend
                )

            elif backend == ClipboardBackend.XSEL:
                return self._copy_subprocess(
                    text, ["xsel", "--clipboard", "--input"], backend
                )

            elif backend == ClipboardBackend.WSLCLIP:
                return self._copy_subprocess(text, ["clip.exe"], backend)

            elif backend == ClipboardBackend.POWERSHELL:
                return self._copy_powershell(text)

            else:
                return ClipboardResult(
                    success=False,
                    backend=backend,
                    message=f"Unsupported backend: {backend.value}",
                )

        except Exception as e:
            return ClipboardResult(
                success=False,
                backend=backend,
                message=str(e),
            )

    def _copy_pyperclip(self, text: str) -> ClipboardResult:
        """Copy using pyperclip.

        Args:
            text: Text to copy.

        Returns:
            ClipboardResult with operation status.
        """
        import pyperclip

        pyperclip.copy(text)

        return ClipboardResult(
            success=True,
            backend=ClipboardBackend.PYPERCLIP,
            chars_copied=len(text),
        )

    def _copy_subprocess(
        self, text: str, cmd: list[str], backend: ClipboardBackend
    ) -> ClipboardResult:
        """Copy using subprocess command.

        Args:
            text: Text to copy.
            cmd: Command and arguments.
            backend: Backend being used.

        Returns:
            ClipboardResult with operation status.
        """
        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=MAX_COMMAND_TIMEOUT,
        )

        if result.returncode == 0:
            return ClipboardResult(
                success=True,
                backend=backend,
                chars_copied=len(text),
            )

        return ClipboardResult(
            success=False,
            backend=backend,
            message=result.stderr.decode("utf-8", errors="replace"),
        )

    def _copy_powershell(self, text: str) -> ClipboardResult:
        """Copy using PowerShell on Windows.

        Args:
            text: Text to copy.

        Returns:
            ClipboardResult with operation status.
        """
        # Use PowerShell's Set-Clipboard
        cmd = [
            "powershell",
            "-NoProfile",
            "-Command",
            "Set-Clipboard -Value $input",
        ]

        result = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            capture_output=True,
            timeout=MAX_COMMAND_TIMEOUT,
        )

        if result.returncode == 0:
            return ClipboardResult(
                success=True,
                backend=ClipboardBackend.POWERSHELL,
                chars_copied=len(text),
            )

        return ClipboardResult(
            success=False,
            backend=ClipboardBackend.POWERSHELL,
            message=result.stderr.decode("utf-8", errors="replace"),
        )


# Module-level convenience functions
_manager: Optional[ClipboardManager] = None


def get_clipboard_manager() -> ClipboardManager:
    """Get singleton clipboard manager.

    Returns:
        Shared ClipboardManager instance.
    """
    global _manager
    if _manager is None:
        _manager = ClipboardManager()
    return _manager


def copy_to_clipboard(text: str) -> ClipboardResult:
    """Copy text to clipboard using default manager.

    Args:
        text: Text to copy.

    Returns:
        ClipboardResult with operation status.
    """
    return get_clipboard_manager().copy(text)


def is_clipboard_available() -> bool:
    """Check if clipboard is available.

    Returns:
        True if clipboard operations are supported.
    """
    return get_clipboard_manager().is_available()
