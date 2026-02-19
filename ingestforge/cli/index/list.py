"""List command - List all indexes.

Lists all available indexes with their information.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.index.base import IndexCommand


class ListCommand(IndexCommand):
    """List all indexes."""

    def execute(
        self,
        project: Optional[Path] = None,
        format: str = "table",
    ) -> int:
        """List indexes.

        Args:
            project: Project directory
            format: Output format (table/json)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # List indexes
            indexes = self.list_indexes(project_path)

            # Display based on format
            if format == "json":
                self._display_json(indexes)
            else:
                self._display_table(indexes)

            # Summary
            if indexes:
                total_size = sum(idx["size"] for idx in indexes)
                self.console.print()
                self.print_info(
                    f"Total: {len(indexes)} indexes, "
                    f"{self._format_size(total_size)} used"
                )
            else:
                self.console.print()
                self.print_info("No indexes found")

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to list indexes")

    def _display_table(self, indexes: list) -> None:
        """Display indexes as table.

        Args:
            indexes: List of index information
        """
        self.console.print()
        summary = self.create_index_summary(indexes)
        self.console.print(summary)

    def _display_json(self, indexes: list) -> None:
        """Display indexes as JSON.

        Args:
            indexes: List of index information
        """
        import json

        json_str = json.dumps(indexes, indent=2, ensure_ascii=False)
        self.console.print(json_str)


# Typer command wrapper
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
    """List all indexes.

    Shows all available indexes with their size, file count,
    and modification time.

    Formats:
    - table: Formatted table view (default)
    - json: JSON output

    Examples:
        # List indexes
        ingestforge index list

        # List as JSON
        ingestforge index list --format json

        # List for specific project
        ingestforge index list -p /path/to/project
    """
    cmd = ListCommand()
    exit_code = cmd.execute(project, format)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
