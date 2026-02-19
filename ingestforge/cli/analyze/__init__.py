"""Content analysis commands package.

Provides content and usage analysis tools:
- topics: Analyze and extract topics from content
- similarity: Find similar documents and duplicates

Usage:
    from ingestforge.cli.analyze import analyze_app
"""

from __future__ import annotations

from ingestforge.cli.analyze.main import app as analyze_app

__all__ = ["analyze_app"]
