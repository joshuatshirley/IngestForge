"""Index command group - Index management operations.

Provides commands for managing search indexes:
- list: List all indexes
- info: Show index information
- rebuild: Rebuild indexes
- delete: Delete index

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations


from ingestforge.cli.index.main import app as index_app

__all__ = ["index_app"]
