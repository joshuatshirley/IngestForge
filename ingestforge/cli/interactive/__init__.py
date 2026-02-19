"""Interactive commands package.

Provides interactive query tools:
- ask: Interactive REPL for conversational queries

Usage:
    from ingestforge.cli.interactive import get_interactive_app
    app = get_interactive_app()

NASA JPL Rule #9: Complete type hints on all functions.
"""

from __future__ import annotations

from typing import Any


def get_interactive_app() -> Any:
    """Lazy import to avoid circular dependency.

    Returns:
        typer.Typer: The interactive CLI application.
    """
    from ingestforge.cli.interactive.main import app

    return app


# For backward compatibility - lazy property
def __getattr__(name: str) -> Any:
    """Lazy attribute access for backward compatibility.

    Args:
        name: Attribute name to access.

    Returns:
        The requested attribute value.

    Raises:
        AttributeError: If attribute not found.
    """
    if name == "interactive_app":
        return get_interactive_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_interactive_app"]
