"""Tag command - Add or remove tags from chunks.

Allows organizing chunks by theme or topic using tags.
Tags are sanitized to lowercase, alphanumeric only, max 32 characters.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #9 (Type Safety).

Part of ORG-002: Tagging System feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class TagCommand(IngestForgeCommand):
    """Add or remove tags from chunks for organization."""

    def execute(
        self,
        source_id: str,
        tag_name: str,
        remove: bool = False,
        document: bool = False,
        project: Optional[Path] = None,
    ) -> int:
        """Execute tag command.

        Args:
            source_id: Chunk ID or document ID to tag
            tag_name: Tag to add or remove
            remove: If True, remove the tag instead of adding
            document: If True, source_id is treated as document_id
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_non_empty_string(source_id, "source_id")
            self.validate_non_empty_string(tag_name, "tag_name")

            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)
            storage = ctx["storage"]

            # Tag document or single chunk
            if document:
                return self._tag_document(storage, source_id, tag_name, remove)
            return self._tag_chunk(storage, source_id, tag_name, remove)

        except Exception as e:
            return self.handle_error(e, "Tag command failed")

    def _tag_chunk(self, storage, chunk_id: str, tag_name: str, remove: bool) -> int:
        """Add or remove a tag from a single chunk.

        Args:
            storage: ChunkRepository instance
            chunk_id: ID of chunk to tag
            tag_name: Tag to add/remove
            remove: True to remove, False to add

        Returns:
            0 on success, 1 on error
        """
        action = "Removing" if remove else "Adding"
        action_past = "removed from" if remove else "added to"

        if remove:
            result = ProgressManager.run_with_spinner(
                lambda: storage.remove_tag(chunk_id, tag_name),
                f"{action} tag '{tag_name}'...",
                f"Tag {action_past} chunk",
            )
        else:
            result = ProgressManager.run_with_spinner(
                lambda: storage.add_tag(chunk_id, tag_name),
                f"{action} tag '{tag_name}'...",
                f"Tag {action_past} chunk",
            )

        if result:
            self.print_success(f"Tag '{tag_name}' {action_past} chunk '{chunk_id}'")
            return 0

        if remove:
            self.print_warning(
                f"Tag '{tag_name}' not found on chunk '{chunk_id}' "
                "(or chunk doesn't exist)"
            )
        else:
            self.print_warning(
                f"Tag '{tag_name}' already exists on chunk '{chunk_id}' "
                "(or chunk doesn't exist)"
            )
        return 1

    def _apply_tag_to_chunk(
        self, storage, chunk_id: str, tag_name: str, remove: bool
    ) -> bool:
        """Apply or remove tag from a single chunk.

        JPL-003: Extracted to reduce nesting.

        Args:
            storage: ChunkRepository instance
            chunk_id: Chunk ID to modify
            tag_name: Tag to add/remove
            remove: True to remove, False to add

        Returns:
            True if operation succeeded
        """
        try:
            if remove:
                return storage.remove_tag(chunk_id, tag_name)
            return storage.add_tag(chunk_id, tag_name)
        except ValueError:
            # Skip chunks that already have max tags or tag doesn't exist
            return False

    def _tag_document(
        self, storage, document_id: str, tag_name: str, remove: bool
    ) -> int:
        """Add or remove a tag from all chunks in a document.

        Args:
            storage: ChunkRepository instance
            document_id: ID of document to tag
            tag_name: Tag to add/remove
            remove: True to remove, False to add

        Returns:
            0 on success, 1 on error
        """
        action_past = "removed from" if remove else "added to"

        # Get all chunks for the document
        chunks = storage.get_chunks_by_document(document_id)

        # Early return: no chunks found (JPL-003: Guard clause)
        if not chunks:
            self.print_error(f"Document '{document_id}' not found")
            return 1

        # Tag each chunk (JPL-003: Reduced nesting via helper)
        success_count = 0
        for chunk in chunks:
            if self._apply_tag_to_chunk(storage, chunk.chunk_id, tag_name, remove):
                success_count += 1

        self.print_success(
            f"Tag '{tag_name}' {action_past} {success_count}/{len(chunks)} chunks "
            f"in document '{document_id}'"
        )
        return 0


class ListTagsCommand(IngestForgeCommand):
    """List all tags in the knowledge base."""

    def execute(self, project: Optional[Path] = None) -> int:
        """Execute list tags command.

        Args:
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)
            storage = ctx["storage"]

            # Get all tags
            tags = ProgressManager.run_with_spinner(
                lambda: storage.get_all_tags(),
                "Fetching tags...",
                "Tags retrieved",
            )

            if not tags:
                self.print_info("No tags found in the knowledge base")
                return 0

            self.console.print(f"\n[bold cyan]Tags ({len(tags)}):[/bold cyan]\n")
            for tag in tags:
                self.console.print(f"  - {tag}")
            self.console.print()

            return 0

        except Exception as e:
            return self.handle_error(e, "List tags failed")


# Typer command wrapper
def command(
    source_id: str = typer.Argument(..., help="Chunk ID or document ID to tag"),
    tag_name: str = typer.Argument(
        ..., help="Tag name (will be sanitized: lowercase, alphanumeric, max 32 chars)"
    ),
    remove: bool = typer.Option(
        False, "--remove", "-r", help="Remove the tag instead of adding"
    ),
    document: bool = typer.Option(
        False,
        "--document",
        "-d",
        help="Treat source_id as document ID (tag all chunks)",
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Add or remove tags from chunks for organization.

    Organize your chunks by theme or topic using tags. Tags are
    sanitized to lowercase, alphanumeric characters only, max 32 chars.
    Each chunk can have up to 50 tags.

    Examples:
        # Add a tag to a specific chunk
        ingestforge tag abc123def important

        # Add a tag with special characters (will be sanitized)
        ingestforge tag abc123def "Machine Learning"
        # Results in tag: machinelearning

        # Remove a tag from a chunk
        ingestforge tag abc123def important --remove

        # Tag all chunks in a document
        ingestforge tag my_document.pdf research --document

        # Remove a tag from all chunks in a document
        ingestforge tag my_document.pdf research --document --remove
    """
    cmd = TagCommand()
    exit_code = cmd.execute(source_id, tag_name, remove, document, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def list_tags_command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """List all tags in the knowledge base.

    Shows all unique tags that have been assigned to chunks.

    Examples:
        # List all tags
        ingestforge tags
    """
    cmd = ListTagsCommand()
    exit_code = cmd.execute(project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
