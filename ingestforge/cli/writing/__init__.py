"""Writing support commands - Academic writing assistance.

This package provides CLI commands for writing assistance:
- draft: Generate drafts with citations
- outline: Create structured outlines
- paraphrase: Paraphrase and rewrite text
- cite: Citation management
- quote: Find quotable passages
- thesis: Evaluate thesis statements

Exports:
    writing_app: Main Typer application for writing commands
"""

from ingestforge.cli.writing.main import app as writing_app

__all__ = ["writing_app"]
