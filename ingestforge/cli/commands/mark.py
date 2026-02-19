"""Mark command - Mark chunks as read or unread.

Allows tracking reading progress through a corpus by marking
individual chunks or entire documents as read/unread.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #9 (Type Safety).

Part of ORG-001: Read/Unread Tracking feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class MarkCommand(IngestForgeCommand):
    """Mark chunks as read or unread for progress tracking."""

    def execute(
        self,
        source_id: str,
        read: bool = True,
        unread: bool = False,
        document: bool = False,
        project: Optional[Path] = None,
    ) -> int:
        """Execute mark command.

        Args:
            source_id: Chunk ID or document ID to mark
            read: Mark as read (default)
            unread: Mark as unread
            document: If True, source_id is treated as document_id
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_non_empty_string(source_id, "source_id")

            # Determine status (unread flag takes precedence)
            status = not unread

            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)
            storage = ctx["storage"]

            # Mark document or single chunk
            if document:
                return self._mark_document(storage, source_id, status)
            return self._mark_chunk(storage, source_id, status)

        except Exception as e:
            return self.handle_error(e, "Mark command failed")

    def _mark_chunk(self, storage, chunk_id: str, status: bool) -> int:
        """Mark a single chunk as read/unread.

        Args:
            storage: ChunkRepository instance
            chunk_id: ID of chunk to mark
            status: True for read, False for unread

        Returns:
            0 on success, 1 on error
        """
        status_str = "read" if status else "unread"

        result = ProgressManager.run_with_spinner(
            lambda: storage.mark_read(chunk_id, status),
            f"Marking chunk as {status_str}...",
            f"Marked chunk as {status_str}",
        )

        if result:
            self.print_success(f"Chunk '{chunk_id}' marked as {status_str}")
            return 0

        self.print_error(f"Chunk '{chunk_id}' not found")
        return 1

    def _mark_document(self, storage, document_id: str, status: bool) -> int:
        """Mark all chunks in a document as read/unread.

        Args:
            storage: ChunkRepository instance
            document_id: ID of document to mark
            status: True for read, False for unread

        Returns:
            0 on success, 1 on error
        """
        status_str = "read" if status else "unread"

        # Get all chunks for the document
        chunks = storage.get_chunks_by_document(document_id)

        if not chunks:
            self.print_error(f"Document '{document_id}' not found")
            return 1

        # Mark each chunk
        success_count = 0
        for chunk in chunks:
            if storage.mark_read(chunk.chunk_id, status):
                success_count += 1

        self.print_success(
            f"Marked {success_count}/{len(chunks)} chunks "
            f"in document '{document_id}' as {status_str}"
        )
        return 0


# Typer command wrapper
def command(
    source_id: str = typer.Argument(..., help="Chunk ID or document ID to mark"),
    read: bool = typer.Option(False, "--read", "-r", help="Mark as read"),
    unread: bool = typer.Option(False, "--unread", "-u", help="Mark as unread"),
    document: bool = typer.Option(
        False,
        "--document",
        "-d",
        help="Treat source_id as document ID (mark all chunks)",
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Mark chunks as read or unread for progress tracking.

    Track your reading progress through a large corpus by marking
    chunks or entire documents as read or unread.

    By default, marks as read. Use --unread to mark as unread.

    Examples:
        # Mark a specific chunk as read
        ingestforge mark abc123def --read

        # Mark a chunk as unread
        ingestforge mark abc123def --unread

        # Mark all chunks in a document as read
        ingestforge mark my_document.pdf --document --read

        # Mark a document's chunks as unread
        ingestforge mark my_document.pdf --document --unread
    """
    # Default to read if neither flag specified
    if not read and not unread:
        read = True

    cmd = MarkCommand()
    exit_code = cmd.execute(source_id, read, unread, document, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
