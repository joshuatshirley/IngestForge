"""Research commands package.

Provides research and verification tools:
- audit: Audit knowledge base quality and coverage
- verify: Verify source citations and references

Usage:
    from ingestforge.cli.research import research_app
"""

from __future__ import annotations

from ingestforge.cli.research.main import app as research_app

__all__ = ["research_app"]
