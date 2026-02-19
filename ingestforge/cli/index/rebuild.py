"""Rebuild command - Rebuild index.

Rebuilds an index from storage data.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.index.base import IndexCommand


class RebuildCommand(IndexCommand):
    """Rebuild index."""

    def execute(
        self,
        index_name: str,
        project: Optional[Path] = None,
        force: bool = False,
    ) -> int:
        """Rebuild index.

        Args:
            index_name: Index name
            project: Project directory
            force: Force rebuild without confirmation

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate index name
            self.validate_index_name(index_name)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)
            project_path = ctx["project_dir"]

            # Check if index exists
            existing_info = self.get_index_info(project_path, index_name)

            # Confirm rebuild if index exists and not forced
            if existing_info and not force:
                msg = f"Rebuild index '{index_name}'? This will recreate the index."
                if not typer.confirm(msg):
                    self.print_info("Rebuild cancelled")
                    return 0

            # Rebuild index
            self.print_info(f"Rebuilding index: {index_name}")
            results = self.rebuild_index(project_path, index_name)

            # Display results
            self._display_results(results)

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to rebuild index")

    def _display_results(self, results: dict) -> None:
        """Display rebuild results.

        Args:
            results: Rebuild results dictionary
        """
        from rich.panel import Panel

        lines = [
            "[bold]Index Rebuilt[/bold]",
            "",
            f"[cyan]Index:[/cyan] {results['index_name']}",
            f"[cyan]Status:[/cyan] {results['status']}",
            f"[cyan]Duration:[/cyan] {results['duration']:.2f}s",
            f"[cyan]Documents:[/cyan] {results['documents_indexed']}",
        ]

        panel = Panel("\n".join(lines), border_style="green")
        self.console.print()
        self.console.print(panel)


# Typer command wrapper
def command(
    index_name: str = typer.Argument(..., help="Index name"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force rebuild without confirmation"
    ),
) -> None:
    """Rebuild index.

    Rebuilds an index from storage data. This is useful
    after updating embeddings or changing index settings.

    WARNING: This will recreate the index!

    Examples:
        # Rebuild index (with confirmation)
        ingestforge index rebuild documents

        # Force rebuild
        ingestforge index rebuild documents --force

        # Rebuild for specific project
        ingestforge index rebuild documents -p /path/to/project
    """
    cmd = RebuildCommand()
    exit_code = cmd.execute(index_name, project, force)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
