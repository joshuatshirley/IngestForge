"""Storage CLI commands.

Provides storage management commands:
- stats: Show storage statistics
- migrate: Migrate between backends
- health: Check storage health
"""

from ingestforge.cli.storage.main import app as storage_app

__all__ = ["storage_app"]
