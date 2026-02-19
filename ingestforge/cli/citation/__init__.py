"""Citation commands package.

Provides citation extraction and formatting tools:
- extract: Extract citations from documents
- format: Format citations in various styles

Usage:
    from ingestforge.cli.citation import citation_app
"""

from __future__ import annotations

from ingestforge.cli.citation.main import app as citation_app

__all__ = ["citation_app"]
