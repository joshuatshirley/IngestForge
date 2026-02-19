"""Bookmark command - Save and manage bookmarked chunks.

Allows saving important passages with optional notes for quick retrieval.
Bookmarks are stored in `.data/bookmarks.json` with atomic writes.

Follows Commandments #1 (Early Returns), #4 (Small Functions),
#6 (Smallest Scope), #7 (Check Parameters), and #9 (Type Safety).

Part of ORG-003: Bookmarks feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from rich.table import Table

from ingestforge.cli.core import IngestForgeCommand, ProgressManager
from ingestforge.storage.bookmarks import BookmarkManager, get_bookmark_manager


class BookmarkCommand(IngestForgeCommand):
    """Add, update, or remove bookmarks on chunks."""

    def execute(
        self,
        chunk_id: str,
        note: Optional[str] = None,
        remove: bool = False,
        project: Optional[Path] = None,
    ) -> int:
        """Execute bookmark command.

        Args:
            chunk_id: Chunk ID to bookmark
            note: Optional note text
            remove: If True, remove the bookmark
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_non_empty_string(chunk_id, "chunk_id")

            # Get project path and initialize manager
            project_path = self.get_project_path(project)
            manager = get_bookmark_manager(project_path)

            if remove:
                return self._remove_bookmark(manager, chunk_id)
            return self._add_bookmark(manager, chunk_id, note, project_path)

        except Exception as e:
            return self.handle_error(e, "Bookmark command failed")

    def _add_bookmark(
        self,
        manager: BookmarkManager,
        chunk_id: str,
        note: Optional[str],
        project_path: Path,
    ) -> int:
        """Add or update a bookmark.

        Args:
            manager: BookmarkManager instance
            chunk_id: Chunk ID to bookmark
            note: Optional note text
            project_path: Project directory

        Returns:
            0 on success, 1 on error
        """
        # Try to get source file from storage for orphan detection
        source_file = ""
        try:
            ctx = self.initialize_context(project_path, require_storage=True)
            storage = ctx.get("storage")
            if storage:
                chunk = storage.get_chunk(chunk_id)
                if chunk and hasattr(chunk, "metadata"):
                    source_file = chunk.metadata.get("source_file", "")
        except Exception:
            # Continue without source file info
            pass

        # Check if this is an update
        existing = manager.get(chunk_id)
        action = "Updating" if existing else "Adding"

        bookmark = ProgressManager.run_with_spinner(
            lambda: manager.add(chunk_id, note or "", source_file),
            f"{action} bookmark...",
            "Bookmark saved",
        )

        if existing:
            self.print_success(f"Bookmark updated for chunk '{chunk_id}'")
            if note:
                self.print_info(f"Note: {note}")
        else:
            self.print_success(f"Bookmarked chunk '{chunk_id}'")
            if note:
                self.print_info(f"Note: {note}")

        return 0

    def _remove_bookmark(
        self,
        manager: BookmarkManager,
        chunk_id: str,
    ) -> int:
        """Remove a bookmark.

        Args:
            manager: BookmarkManager instance
            chunk_id: Chunk ID to unbookmark

        Returns:
            0 on success, 1 if not found
        """
        result = ProgressManager.run_with_spinner(
            lambda: manager.remove(chunk_id),
            "Removing bookmark...",
            "Bookmark removed",
        )

        if result:
            self.print_success(f"Bookmark removed for chunk '{chunk_id}'")
            return 0

        self.print_warning(f"No bookmark found for chunk '{chunk_id}'")
        return 1


class ListBookmarksCommand(IngestForgeCommand):
    """List all saved bookmarks."""

    def execute(
        self,
        orphaned_only: bool = False,
        check_orphans: bool = False,
        project: Optional[Path] = None,
    ) -> int:
        """Execute list bookmarks command.

        Args:
            orphaned_only: Only show orphaned bookmarks
            check_orphans: Check for and flag orphaned bookmarks
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            project_path = self.get_project_path(project)
            manager = get_bookmark_manager(project_path)

            # Check for orphans if requested
            if check_orphans:
                orphan_count = ProgressManager.run_with_spinner(
                    lambda: manager.check_orphans(),
                    "Checking for orphaned bookmarks...",
                    "Orphan check complete",
                )
                if orphan_count > 0:
                    self.print_warning(
                        f"Found {orphan_count} orphaned bookmark(s) "
                        "(source file deleted)"
                    )

            # Get bookmarks
            if orphaned_only:
                bookmarks = manager.get_orphaned()
                title = "Orphaned Bookmarks"
            else:
                bookmarks = manager.list_all()
                title = "Bookmarks"

            if not bookmarks:
                if orphaned_only:
                    self.print_info("No orphaned bookmarks found")
                else:
                    self.print_info("No bookmarks saved yet")
                return 0

            # Display as table
            self._display_bookmarks_table(bookmarks, title)
            return 0

        except Exception as e:
            return self.handle_error(e, "List bookmarks failed")

    def _display_bookmarks_table(
        self,
        bookmarks: list,
        title: str,
    ) -> None:
        """Display bookmarks in a formatted table.

        Args:
            bookmarks: List of Bookmark objects
            title: Table title
        """
        table = Table(title=f"{title} ({len(bookmarks)})")
        table.add_column("Chunk ID", style="cyan", no_wrap=True)
        table.add_column("Note", style="white", max_width=50)
        table.add_column("Created", style="dim")
        table.add_column("Status", style="yellow")

        for bookmark in bookmarks:
            # Format created timestamp
            created = bookmark.created_at[:10] if bookmark.created_at else "-"

            # Determine status
            if bookmark.is_orphaned:
                status = "[red]Orphaned[/red]"
            else:
                status = "[green]Active[/green]"

            # Truncate note if too long
            note = bookmark.note or "-"
            if len(note) > 47:
                note = note[:47] + "..."

            table.add_row(
                bookmark.chunk_id[:16] + "..."
                if len(bookmark.chunk_id) > 16
                else bookmark.chunk_id,
                note,
                created,
                status,
            )

        self.console.print()
        self.console.print(table)
        self.console.print()


# =============================================================================
# Typer Command Wrappers
# =============================================================================


def command(
    chunk_id: str = typer.Argument(..., help="Chunk ID to bookmark"),
    note: Optional[str] = typer.Option(
        None, "--note", "-n", help="Optional note to attach to the bookmark"
    ),
    remove: bool = typer.Option(
        False, "--remove", "-r", help="Remove the bookmark instead of adding"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Save a chunk as a bookmark with optional note.

    Bookmarks help you quickly find important passages later.
    Bookmarking the same chunk twice updates the note (no duplicates).

    Examples:
        # Bookmark a chunk
        ingestforge bookmark abc123def

        # Bookmark with a note
        ingestforge bookmark abc123def --note "Key quote for thesis"

        # Update an existing bookmark's note
        ingestforge bookmark abc123def --note "Updated note"

        # Remove a bookmark
        ingestforge bookmark abc123def --remove
    """
    cmd = BookmarkCommand()
    exit_code = cmd.execute(chunk_id, note, remove, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def list_bookmarks_command(
    orphaned_only: bool = typer.Option(
        False, "--orphaned", "-o", help="Show only orphaned bookmarks"
    ),
    check_orphans: bool = typer.Option(
        False,
        "--check",
        "-c",
        help="Check for orphaned bookmarks (source file deleted)",
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """List all saved bookmarks.

    Shows all bookmarked chunks with their notes and creation dates.
    Orphaned bookmarks (source file deleted) are flagged but not removed.

    Examples:
        # List all bookmarks
        ingestforge bookmarks list

        # Check for and show orphaned bookmarks
        ingestforge bookmarks list --check

        # Show only orphaned bookmarks
        ingestforge bookmarks list --orphaned
    """
    cmd = ListBookmarksCommand()
    exit_code = cmd.execute(orphaned_only, check_orphans, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
