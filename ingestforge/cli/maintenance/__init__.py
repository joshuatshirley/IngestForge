"""Maintenance command group - System maintenance operations.

Provides commands for system maintenance and optimization:
- cleanup: Clean up temporary files and old data
- optimize: Optimize storage and indexes
- backup: Backup project data
- restore: Restore from backup

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations


from ingestforge.cli.maintenance.main import app as maintenance_app

__all__ = ["maintenance_app"]
