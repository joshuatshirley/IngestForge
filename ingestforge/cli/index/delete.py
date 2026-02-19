"""Delete command - Delete index.

Deletes an index and all its data.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.index.base import IndexCommand


class DeleteCommand(IndexCommand):
    """Delete index."""

    def execute(
        self,
        index_name: str,
        project: Optional[Path] = None,
        force: bool = False,
    ) -> int:
        """Delete index.

        Args:
            index_name: Index name
            project: Project directory
            force: Force deletion without confirmation

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate index name
            self.validate_index_name(index_name)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Check if index exists
            info = self.get_index_info(project_path, index_name)

            if not info:
                self.print_error(f"Index not found: {index_name}")
                return 1

            # Confirm deletion if not forced
            if not force:
                msg = (
                    f"Delete index '{index_name}'? "
                    f"This will permanently remove {info['size_formatted']} of data."
                )
                if not typer.confirm(msg):
                    self.print_info("Deletion cancelled")
                    return 0

            # Delete index
            self.print_info(f"Deleting index: {index_name}")
            success = self.delete_index(project_path, index_name)

            if success:
                self.print_success(f"Deleted index: {index_name}")
                return 0
            else:
                self.print_error(f"Failed to delete index: {index_name}")
                return 1

        except Exception as e:
            return self.handle_error(e, "Failed to delete index")


# Typer command wrapper
def command(
    index_name: str = typer.Argument(..., help="Index name"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force deletion without confirmation"
    ),
) -> None:
    """Delete index.

    Permanently deletes an index and all its data.

    WARNING: This action cannot be undone!

    Examples:
        # Delete index (with confirmation)
        ingestforge index delete documents

        # Force delete
        ingestforge index delete documents --force

        # Delete from specific project
        ingestforge index delete documents -p /path/to/project
    """
    cmd = DeleteCommand()
    exit_code = cmd.execute(index_name, project, force)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
