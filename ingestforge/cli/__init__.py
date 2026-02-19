"""IngestForge CLI - Command-line interface for RAG operations.

Main entry point is in main.py which registers all command groups.

Usage:
    python -m ingestforge          # Run CLI
    python -m ingestforge.cli      # Also works
"""


def __getattr__(name: str):
    """Lazy import to avoid RuntimeWarning when running as module."""
    if name == "app":
        from ingestforge.cli.main import app

        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["app"]
