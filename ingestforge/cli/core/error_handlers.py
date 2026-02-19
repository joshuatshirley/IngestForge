"""Standard error handling for CLI commands.

This module provides centralized error handling with consistent formatting
across all CLI commands, following Commandment #7 (Check Parameters & Returns).

UX-004: Integrates with ErrorRenderer for helpful error messages.
"""

from __future__ import annotations

from typing import Optional, Dict, Type, Callable, Any
import typer
from rich.console import Console

from ingestforge.cli.console import ErrorRenderer, get_console

# Shared console instance (kept for backwards compatibility)
console = Console()


class CLIErrorHandler:
    """Centralized error handling with consistent formatting.

    UX-004: Now uses ErrorRenderer for helpful error messages with
    "Why it happened" and "How to fix" sections.

    This class provides:
    - Helpful error message formatting with root cause analysis
    - Type-specific error styling
    - Standard exit codes
    - Error context support
    - Verbose mode for full tracebacks

    Example:
        try:
            result = risky_operation()
        except Exception as e:
            CLIErrorHandler.exit_on_error(e, "Operation failed")
    """

    # Error type → (console style, display prefix) - kept for simple errors
    ERROR_STYLES: Dict[Type[Exception], tuple[str, str]] = {
        ValueError: ("red", "Validation error"),
        ImportError: ("red", "Missing dependency"),
        FileNotFoundError: ("yellow", "File not found"),
        PermissionError: ("red", "Permission denied"),
        KeyError: ("red", "Configuration error"),
        RuntimeError: ("red", "Runtime error"),
        OSError: ("red", "System error"),
        TypeError: ("red", "Type error"),
    }

    @staticmethod
    def handle_error(
        error: Exception,
        context: str = "",
        use_panel: bool = True,
    ) -> None:
        """Display error with helpful formatting.

        UX-004: Uses ErrorRenderer to show helpful error panels with
        "Why it happened" and "How to fix" sections.

        Args:
            error: Exception to display
            context: Optional context message (e.g., "While processing file X")
            use_panel: If True, use rich panel (default). If False, use simple format.

        Example:
            CLIErrorHandler.handle_error(
                ValueError("Invalid input"),
                "While validating project name"
            )
        """
        if use_panel:
            # UX-004: Use the new helpful error renderer
            ErrorRenderer.render(error, context=context)
        else:
            # Fallback to simple format for backwards compatibility
            error_type = type(error)
            style, prefix = CLIErrorHandler.ERROR_STYLES.get(
                error_type, ("red", "Error")
            )

            # Format main error message
            msg = f"[{style}]{prefix}:[/{style}] {error}"

            # Add context if provided
            if context:
                msg = f"[dim]{context}[/dim]\n{msg}"

            get_console().print(msg)

    @staticmethod
    def handle_error_simple(error: Exception, context: str = "") -> None:
        """Display error with simple formatting (no panel).

        Use this for errors that don't need the full panel treatment.

        Args:
            error: Exception to display
            context: Optional context message
        """
        CLIErrorHandler.handle_error(error, context, use_panel=False)

    @staticmethod
    def exit_on_error(
        error: Exception,
        context: str = "",
        exit_code: int = 1,
    ) -> None:
        """Handle error and exit CLI with appropriate exit code.

        UX-004: Shows helpful error panel before exiting.

        Args:
            error: Exception to handle
            context: Optional context message
            exit_code: Exit code (default 1)

        Raises:
            typer.Exit: Always raises to exit CLI

        Example:
            try:
                process_file(path)
            except Exception as e:
                CLIErrorHandler.exit_on_error(e, "File processing failed")
        """
        CLIErrorHandler.handle_error(error, context)
        raise typer.Exit(exit_code)

    @staticmethod
    def wrap_operation(
        operation: Callable[[], Any],
        operation_name: str = "Operation",
        on_error: Optional[Callable[[Exception], Any]] = None,
    ) -> Any:
        """Wrap operation with standard error handling.

        Args:
            operation: Callable to execute
            operation_name: Name for error messages
            on_error: Optional callback for custom error handling

        Returns:
            Operation result

        Raises:
            typer.Exit: On error (exits CLI)

        Example:
            result = CLIErrorHandler.wrap_operation(
                lambda: process_data(),
                "Data processing"
            )
        """
        try:
            return operation()
        except KeyboardInterrupt:
            CLIErrorHandler.handle_keyboard_interrupt()
        except Exception as e:
            if on_error:
                on_error(e)
            CLIErrorHandler.exit_on_error(e, f"{operation_name} failed")

    @staticmethod
    def handle_keyboard_interrupt() -> None:
        """Handle Ctrl+C gracefully."""
        get_console().print("\n[yellow]Interrupted by user[/yellow]")
        raise typer.Exit(130)  # Standard exit code for SIGINT

    @staticmethod
    def register_error_style(
        exception_type: Type[Exception], style: str, prefix: str
    ) -> None:
        """Register custom error style for specific exception type.

        Args:
            exception_type: Exception class
            style: Rich console style (e.g., "red", "yellow")
            prefix: Display prefix (e.g., "Custom Error")

        Example:
            CLIErrorHandler.register_error_style(
                MyCustomError, "magenta", "Custom Error"
            )
        """
        CLIErrorHandler.ERROR_STYLES[exception_type] = (style, prefix)


def cli_exception_handler(func: Callable) -> Callable:
    """Decorator to wrap CLI commands with exception handling.

    UX-004: Automatically catches exceptions and displays helpful error panels.

    Example:
        @cli_exception_handler
        def my_command():
            # Command logic
            pass
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except KeyboardInterrupt:
            CLIErrorHandler.handle_keyboard_interrupt()
        except SystemExit:
            # Let SystemExit through (from typer.Exit)
            raise
        except Exception as e:
            CLIErrorHandler.exit_on_error(e)

    return wrapper
