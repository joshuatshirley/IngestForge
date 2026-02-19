"""
Lateral Connections Discovery Command.

Surfaces links between siloed domains.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.table import Table

from ingestforge.cli.analyze.base import AnalyzeCommand
from ingestforge.analysis.lateral_linker import LateralLinker


class ConnectionsCommand(AnalyzeCommand):
    """Discover lateral connections across domains."""

    def execute(self, project: Optional[Path] = None) -> int:
        try:
            ctx = self.initialize_context(project)
            # Retrieve all chunks (or a large sample for analysis)
            # In a real app, we'd use a specific retrieval or graph query
            all_chunks = ctx.storage.get_all_chunks(limit=1000)

            linker = LateralLinker()
            connections = linker.find_connections(all_chunks)

            if not connections:
                self.print_info("No lateral connections discovered yet.")
                return 0

            self._display_connections(connections)
            return 0
        except Exception as e:
            return self.handle_error(e, "Connection discovery failed")

    def _display_connections(self, connections):
        table = Table(title="Cross-Vertical Lateral Connections")
        table.add_column("Type", style="cyan")
        table.add_column("Anchor", style="bold yellow")
        table.add_column("Impacted Domains", style="green")
        table.add_column("Chunk Count", justify="right")

        for conn in connections:
            table.add_row(
                conn["type"],
                conn.get("entity") or conn.get("id"),
                ", ".join(conn["domains"]),
                str(len(conn["chunk_ids"])),
            )

        self.console.print(table)


def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project path"
    ),
):
    """Scan for lateral connections between different domain silos."""
    ConnectionsCommand().execute(project)
