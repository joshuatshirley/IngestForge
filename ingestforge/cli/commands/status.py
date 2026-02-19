"""Status command - Display project status and statistics.

This command shows:
- Project configuration
- Storage backend status
- Document count
- Chunk count
- Index status

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer
from rich.table import Table

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class StatusCommand(IngestForgeCommand):
    """Display project status and statistics.

    Shows project configuration and storage statistics.
    """

    def execute(self, project: Optional[Path] = None) -> int:
        """Display project status.

        Args:
            project: Project directory (default: current directory)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize with storage (Commandment #7: Check inputs)
            ctx = self.initialize_context(project, require_storage=True)

            # Display project info
            self._display_project_info(ctx)

            # Display storage stats with progress indicator
            stats = ProgressManager.run_with_spinner(
                lambda: self._get_storage_stats(ctx["storage"]),
                "Gathering statistics...",
            )

            self._display_storage_stats(stats)

            return 0

        except Exception as e:
            return self.handle_error(e, "Status check failed")

    def _display_project_info(self, ctx: dict) -> None:
        """Display project configuration info.

        Args:
            ctx: Context dict with 'config' and 'project_path'
        """
        config = ctx["config"]
        project_path = ctx["project_path"]

        # Create info table (Commandment #4: Small function)
        table = Table(title="Project Information", show_header=False)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Add project info rows
        table.add_row("Name", config.project.name)
        table.add_row("Version", config.project.version)
        table.add_row("Path", str(project_path))
        table.add_row("Storage", config.storage.backend)
        table.add_row("Mobile Mode", str(config.project.mobile_mode))

        self.console.print(table)

    def _get_storage_stats(self, storage: Any) -> dict[str, Any]:
        """Get storage statistics.

        Args:
            storage: ChunkRepository instance

        Returns:
            Dict with statistics
        """
        try:
            # Get chunk count
            total_chunks = storage.count() if hasattr(storage, "count") else 0

            # Count unique documents from chunks
            total_documents = self._count_unique_documents(storage)

            return {
                "total_chunks": total_chunks,
                "total_documents": total_documents,
                "backend": storage.__class__.__name__,
            }

        except Exception:
            # Storage backend may not implement all methods
            return {
                "total_chunks": "N/A",
                "total_documents": "N/A",
                "backend": storage.__class__.__name__,
            }

    def _extract_doc_ids_from_chunks(self, chunks: Any) -> set[str]:
        """Extract unique document IDs from chunks.

        JPL-003: Extracted to reduce nesting.

        Args:
            chunks: Iterator/list of chunk objects

        Returns:
            Set of unique document IDs
        """
        doc_ids = set()
        for chunk in chunks:
            doc_id = getattr(chunk, "document_id", None)
            if doc_id:
                doc_ids.add(doc_id)
        return doc_ids

    def _count_unique_documents(self, storage: Any) -> int:
        """Count unique documents in storage.

        Args:
            storage: ChunkRepository instance

        Returns:
            Number of unique documents
        """
        try:
            # Try get_all_chunks and count unique document_ids
            if hasattr(storage, "get_all_chunks"):
                chunks = storage.get_all_chunks()
                # JPL-003: Reduced nesting via helper function
                doc_ids = self._extract_doc_ids_from_chunks(chunks)
                return len(doc_ids)

            # Fallback: try get_statistics
            if hasattr(storage, "get_statistics"):
                stats = storage.get_statistics()
                if "total_documents" in stats:
                    return stats["total_documents"]

            return 0
        except Exception:
            return 0

    def _display_storage_stats(self, stats: dict) -> None:
        """Display storage statistics.

        Args:
            stats: Statistics dict from _get_storage_stats
        """
        table = Table(title="Storage Statistics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        # Add stats rows (Commandment #1: Simple flow)
        table.add_row("Backend", stats["backend"])
        table.add_row("Total Documents", str(stats["total_documents"]))
        table.add_row("Total Chunks", str(stats["total_chunks"]))

        self.console.print(table)


# Typer command wrapper (Commandment #9: Type safety)
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory (default: current: Any)"
    ),
) -> None:
    """Display project status and statistics.

    Shows project configuration, storage backend info, and document/chunk counts.

    Examples:
        # Show status for current directory
        ingestforge status

        # Show status for specific project
        ingestforge status --project /path/to/project
    """
    cmd = StatusCommand()
    exit_code = cmd.execute(project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
