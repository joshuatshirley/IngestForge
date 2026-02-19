"""Console output helpers.

Provides consistent formatting for CLI output messages.

UX-004: Includes ErrorRenderer for helpful error messages.

Follows Commandment #4 (Small Functions) and #6 (Smallest Scope).
"""

from __future__ import annotations

import traceback
from typing import Optional, List

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Shared console instance
_console: Console | None = None

# Verbose mode flag (set by CLI --verbose flag)
_verbose_mode: bool = False


def get_console() -> Console:
    """Get shared console instance (lazy-loaded).

    Returns:
        Rich Console instance
    """
    global _console
    if _console is None:
        _console = Console()
    return _console


def set_verbose_mode(enabled: bool) -> None:
    """Enable or disable verbose mode for error display.

    When verbose mode is enabled, full tracebacks are shown.

    Args:
        enabled: True to show tracebacks, False to hide them
    """
    global _verbose_mode
    _verbose_mode = enabled


def is_verbose_mode() -> bool:
    """Check if verbose mode is enabled.

    Returns:
        True if tracebacks should be shown
    """
    return _verbose_mode


def tip(message: str) -> None:
    """Display a tip message to guide users.

    Tips are displayed in dim styling to differentiate from
    primary output while providing helpful guidance.

    Args:
        message: Tip text to display

    Example:
        tip("Use --recursive to process subdirectories")
        # Output: "  Tip: Use --recursive to process subdirectories"
    """
    get_console().print(f"  [dim]Tip: {message}[/dim]")


def hint(message: str) -> None:
    """Display a hint message (alias for tip).

    Args:
        message: Hint text to display
    """
    tip(message)


def note(message: str) -> None:
    """Display a note message.

    Notes are displayed in italic cyan for informational emphasis.

    Args:
        message: Note text to display
    """
    get_console().print(f"  [italic cyan]Note: {message}[/italic cyan]")


# ============================================================================
# UX-004: ErrorRenderer - Helpful Error Messages
# ============================================================================


class ErrorRenderer:
    """Renders helpful error messages with "Why" and "How to fix" sections.

    UX-004: Replaces generic tracebacks with user-friendly error panels.

    Example
    -------
        try:
            process_file(path)
        except Exception as e:
            ErrorRenderer.render(e)
            raise SystemExit(1)

        # Outputs:
        # +---------------------+
        # |  Error: IF-FILE-001 |
        # +---------------------+
        # | File not found      |
        # |                     |
        # | Why it happened:    |
        # | The file could not  |
        # | be found...         |
        # |                     |
        # | How to fix:         |
        # | - Check the path    |
        # | - Verify file exists|
        # +---------------------+
    """

    @staticmethod
    def render(
        exc: BaseException,
        context: str = "",
        show_traceback: Optional[bool] = None,
    ) -> None:
        """Render an exception as a helpful error panel.

        Args:
            exc: Exception to render
            context: Optional context message (e.g., "While processing file.pdf")
            show_traceback: Override for verbose mode (None = use global setting)
        """
        from ingestforge.core.exceptions import get_error_info, get_root_cause

        console = get_console()

        # Get error info from exception or lookup table
        error_info = get_error_info(exc)
        error_code = error_info.get("error_code", "IF-ERR-999")
        why = error_info.get("why_it_happened", "An unexpected error occurred")
        how_to_fix = error_info.get("how_to_fix", ["Check the error message"])

        # Get root cause for nested exceptions
        root_cause = get_root_cause(exc)
        root_message = str(root_cause) if root_cause != exc else None

        # Build the error panel content
        content = ErrorRenderer._build_error_content(
            message=str(exc),
            context=context,
            why=why,
            how_to_fix=how_to_fix,
            root_message=root_message,
        )

        # Create and display the panel
        panel = Panel(
            content,
            title=f"[bold red]Error: {error_code}[/bold red]",
            border_style="red",
            padding=(1, 2),
        )
        console.print(panel)

        # Show traceback if verbose
        should_show_traceback = (
            show_traceback if show_traceback is not None else is_verbose_mode()
        )
        if should_show_traceback:
            ErrorRenderer._render_traceback(exc)

    @staticmethod
    def _build_error_content(
        message: str,
        context: str,
        why: str,
        how_to_fix: List[str],
        root_message: Optional[str],
    ) -> Text:
        """Build the error panel content.

        Args:
            message: Main error message
            context: Optional context
            why: Why it happened explanation
            how_to_fix: List of fix suggestions
            root_message: Root cause message if different from main

        Returns:
            Rich Text object for panel content
        """
        text = Text()

        # Context line
        if context:
            text.append(f"{context}\n", style="dim")
            text.append("\n")

        # Main error message
        text.append(message, style="bold red")
        text.append("\n\n")

        # Root cause (if different)
        if root_message and root_message != message:
            text.append("Root cause: ", style="bold yellow")
            text.append(root_message, style="yellow")
            text.append("\n\n")

        # Why it happened
        text.append("Why it happened:\n", style="bold cyan")
        text.append(f"  {why}\n", style="cyan")
        text.append("\n")

        # How to fix
        text.append("How to fix:\n", style="bold green")
        for fix in how_to_fix:
            text.append(f"  - {fix}\n", style="green")

        return text

    @staticmethod
    def _render_traceback(exc: BaseException) -> None:
        """Render the full traceback in a collapsible style.

        Args:
            exc: Exception to show traceback for
        """
        console = get_console()
        console.print()
        console.print("[dim]--- Traceback (--verbose mode) ---[/dim]")

        # Format traceback
        tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
        tb_text = "".join(tb_lines)

        # Print in dim style to distinguish from main error
        console.print(f"[dim]{tb_text}[/dim]")

    @staticmethod
    def render_simple(message: str, error_code: str = "IF-ERR-000") -> None:
        """Render a simple error message without exception.

        For cases where you want to show a helpful error without raising.

        Args:
            message: Error message
            error_code: Error code for reference
        """
        console = get_console()
        panel = Panel(
            f"[bold red]{message}[/bold red]",
            title=f"[bold red]Error: {error_code}[/bold red]",
            border_style="red",
            padding=(0, 2),
        )
        console.print(panel)

    @staticmethod
    def render_warning(message: str, suggestion: str = "") -> None:
        """Render a warning panel (non-fatal).

        Args:
            message: Warning message
            suggestion: Optional suggestion for resolution
        """
        console = get_console()

        content = Text()
        content.append(message, style="bold yellow")
        if suggestion:
            content.append("\n\n")
            content.append("Suggestion: ", style="bold")
            content.append(suggestion, style="dim")

        panel = Panel(
            content,
            title="[bold yellow]Warning[/bold yellow]",
            border_style="yellow",
            padding=(0, 2),
        )
        console.print(panel)
