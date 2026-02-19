"""Info command - Show index information.

Shows detailed information about a specific index.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.index.base import IndexCommand


class InfoCommand(IndexCommand):
    """Show index information."""

    def execute(
        self,
        index_name: str,
        project: Optional[Path] = None,
    ) -> int:
        """Show index information.

        Args:
            index_name: Index name
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate index name
            self.validate_index_name(index_name)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Get index info
            info = self.get_index_info(project_path, index_name)

            if not info:
                self.print_error(f"Index not found: {index_name}")
                return 1

            # Display info
            self._display_info(info)

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to get index information")

    def _display_info(self, info: dict) -> None:
        """Display index information.

        Args:
            info: Index information dictionary
        """
        from rich.panel import Panel

        lines = [
            "[bold]Index Information[/bold]",
            "",
            f"[cyan]Name:[/cyan] {info['name']}",
            f"[cyan]Path:[/cyan] {info['path']}",
            "",
            f"[cyan]Size:[/cyan] {info['size_formatted']}",
            f"[cyan]Files:[/cyan] {info['files']}",
            "",
            f"[cyan]Modified:[/cyan] {info['modified_formatted']}",
        ]

        panel = Panel("\n".join(lines), border_style="cyan")
        self.console.print()
        self.console.print(panel)


# Typer command wrapper
def command(
    index_name: str = typer.Argument(..., help="Index name"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Show index information.

    Displays detailed information about a specific index
    including size, file count, and modification time.

    Examples:
        # Show index info
        ingestforge index info documents

        # Show info for specific project
        ingestforge index info documents -p /path/to/project
    """
    cmd = InfoCommand()
    exit_code = cmd.execute(index_name, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
