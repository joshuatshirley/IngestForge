"""Stats command - Show storage statistics.

Displays storage backend statistics including chunk count,
document count, and backend-specific metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ingestforge.cli.storage.base import StorageCommand


class StatsCommand(StorageCommand):
    """Display storage statistics."""

    def execute(
        self,
        project: Optional[Path] = None,
        format: str = "table",
    ) -> int:
        """Show storage statistics.

        Args:
            project: Project directory
            format: Output format (table/json)

        Returns:
            0 on success, 1 on error
        """
        try:
            config = self.get_config(project)
            storage = self.get_storage(config)

            stats = storage.get_statistics()
            stats["backend"] = config.storage.backend

            # Add health check if available
            if hasattr(storage, "health_check"):
                healthy, message = storage.health_check()
                stats["health"] = "OK" if healthy else "FAILED"
                stats["health_message"] = message

            if format == "json":
                self._display_json(stats)
            else:
                self._display_table(stats)

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to get storage stats")

    def _display_table(self, stats: dict) -> None:
        """Display stats as table."""
        self.console.print()
        table = self.create_stats_table(stats)
        self.console.print(table)

    def _display_json(self, stats: dict) -> None:
        """Display stats as JSON."""
        json_str = json.dumps(stats, indent=2, default=str)
        self.console.print(json_str)


def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
) -> None:
    """Show storage statistics.

    Displays information about the current storage backend including:
    - Total chunks stored
    - Total documents
    - Storage size (if available)
    - Backend type
    - Health status

    Examples:
        # Show storage stats
        ingestforge storage stats

        # Output as JSON
        ingestforge storage stats --format json

        # For specific project
        ingestforge storage stats -p /path/to/project
    """
    cmd = StatsCommand()
    exit_code = cmd.execute(project, format)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
