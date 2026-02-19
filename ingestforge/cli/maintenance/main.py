"""Maintenance subcommands.

Provides tools for system maintenance and optimization:
- cleanup: Clean up temporary files and old data
- optimize: Optimize storage and indexes
- backup: Backup project data
- restore: Restore from backup

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.maintenance import cleanup, optimize, backup, restore, orphans

# Create maintenance subcommand application
app = typer.Typer(
    name="maintenance",
    help="System maintenance tools",
    add_completion=False,
)

# Register maintenance commands
app.command("cleanup")(cleanup.command)
app.command("optimize")(optimize.command)
app.command("backup")(backup.command)
app.command("restore")(restore.command)
app.add_typer(orphans.app, name="orphans")


@app.callback()
def main() -> None:
    """System maintenance tools for IngestForge.

    Maintain and optimize your IngestForge projects:
    - Clean up temporary files and cache
    - Optimize storage and indexes
    - Create and restore backups
    - Free up disk space

    Features:
    - Safe cleanup operations
    - Storage optimization
    - Backup and restore
    - Space management
    - Performance tuning

    Use cases:
    - Regular maintenance
    - Pre-deployment cleanup
    - Disaster recovery
    - Storage optimization
    - Performance troubleshooting

    Examples:
        # Clean up temporary files
        ingestforge maintenance cleanup --force

        # Optimize storage and indexes
        ingestforge maintenance optimize

        # Create backup
        ingestforge maintenance backup backup.zip

        # Restore from backup
        ingestforge maintenance restore backup.zip

        # Full maintenance cycle
        ingestforge maintenance cleanup --force
        ingestforge maintenance optimize --compact
        ingestforge maintenance backup backup_$(date +%Y%m%d).zip

    For help on specific commands:
        ingestforge maintenance <command> --help
    """
    pass
