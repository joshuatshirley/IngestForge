"""Annotate command - Add personal notes to chunks.

Allows adding persistent annotations to chunks that remain even if
the source content is re-ingested (mapped via content hash).

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #9 (Type Safety).

Part of ORG-004: Annotations feature.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class AnnotateCommand(IngestForgeCommand):
    """Add annotations to chunks for personal note-taking."""

    def execute(
        self,
        chunk_id: str,
        note_text: str,
        project: Optional[Path] = None,
    ) -> int:
        """Execute annotate command.

        Args:
            chunk_id: ID of chunk to annotate
            note_text: Annotation text (up to 10,000 characters)
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_non_empty_string(chunk_id, "chunk_id")
            self.validate_non_empty_string(note_text, "note_text")

            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)
            storage = ctx["storage"]

            # Get the chunk to verify it exists and get content hash
            chunk = storage.get_chunk(chunk_id)
            if not chunk:
                self.print_error(f"Chunk '{chunk_id}' not found")
                return 1

            # Compute content hash for re-ingestion mapping
            from ingestforge.storage.annotations import (
                compute_content_hash,
                get_annotation_manager,
            )

            content_hash = compute_content_hash(chunk.content)

            # Get annotation manager
            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_annotation_manager(data_dir)

            # Create annotation
            annotation = ProgressManager.run_with_spinner(
                lambda: manager.add(chunk_id, note_text, content_hash),
                "Creating annotation...",
                "Annotation created",
            )

            self.print_success(
                f"Created annotation '{annotation.annotation_id}' "
                f"for chunk '{chunk_id}'"
            )

            # Show preview
            preview = annotation.preview(80)
            self.console.print(f"  [dim]{preview}[/dim]")

            return 0

        except ValueError as e:
            self.print_error(str(e))
            return 1
        except Exception as e:
            return self.handle_error(e, "Annotate command failed")


class ListAnnotationsCommand(IngestForgeCommand):
    """List all annotations in the knowledge base."""

    def execute(
        self,
        chunk_id: Optional[str] = None,
        limit: int = 50,
        project: Optional[Path] = None,
    ) -> int:
        """Execute list annotations command.

        Args:
            chunk_id: If provided, list only annotations for this chunk
            limit: Maximum annotations to display
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)

            from ingestforge.storage.annotations import get_annotation_manager

            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_annotation_manager(data_dir)

            # Get annotations
            if chunk_id:
                annotations = manager.get_for_chunk(chunk_id)
                title = f"Annotations for chunk '{chunk_id}'"
            else:
                annotations = manager.list_all(limit)
                title = "All Annotations"

            if not annotations:
                if chunk_id:
                    self.print_info(f"No annotations found for chunk '{chunk_id}'")
                else:
                    self.print_info("No annotations found in the knowledge base")
                return 0

            # Display header
            self.console.print(
                f"\n[bold cyan]{title} ({len(annotations)}):[/bold cyan]\n"
            )

            # Display each annotation
            for ann in annotations:
                self._display_annotation(ann)

            # Show statistics if listing all
            if not chunk_id:
                stats = manager.get_statistics()
                self.console.print(
                    f"\n[dim]Annotations: {stats['total_annotations']} | "
                    f"Chunks annotated: {stats['unique_chunks']} | "
                    f"Avg length: {stats['avg_length']} chars[/dim]"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "List annotations failed")

    def _display_annotation(self, annotation) -> None:
        """Display a single annotation.

        Args:
            annotation: Annotation to display
        """
        # Format timestamp
        created = annotation.created_at[:10] if annotation.created_at else "Unknown"

        # Show annotation
        self.console.print(
            f"[bold]{annotation.annotation_id}[/bold] " f"[dim]({created})[/dim]"
        )
        self.console.print(f"  Chunk: [cyan]{annotation.chunk_id}[/cyan]")

        # Show text preview (handle multi-line)
        preview = annotation.preview(200)
        for line in preview.split("\n")[:3]:
            self.console.print(f"  [dim]{line}[/dim]")

        self.console.print()


class DeleteAnnotationCommand(IngestForgeCommand):
    """Delete an annotation."""

    def execute(
        self,
        annotation_id: str,
        project: Optional[Path] = None,
    ) -> int:
        """Execute delete annotation command.

        Args:
            annotation_id: ID of annotation to delete
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_non_empty_string(annotation_id, "annotation_id")

            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)

            from ingestforge.storage.annotations import get_annotation_manager

            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_annotation_manager(data_dir)

            # Delete annotation
            result = ProgressManager.run_with_spinner(
                lambda: manager.delete(annotation_id),
                "Deleting annotation...",
                "Annotation deleted",
            )

            if result:
                self.print_success(f"Deleted annotation '{annotation_id}'")
                return 0

            self.print_error(f"Annotation '{annotation_id}' not found")
            return 1

        except Exception as e:
            return self.handle_error(e, "Delete annotation failed")


class UpdateAnnotationCommand(IngestForgeCommand):
    """Update an existing annotation."""

    def execute(
        self,
        annotation_id: str,
        note_text: str,
        project: Optional[Path] = None,
    ) -> int:
        """Execute update annotation command.

        Args:
            annotation_id: ID of annotation to update
            note_text: New annotation text
            project: Project directory

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_non_empty_string(annotation_id, "annotation_id")
            self.validate_non_empty_string(note_text, "note_text")

            # Initialize context with storage
            ctx = self.initialize_context(project, require_storage=True)

            from ingestforge.storage.annotations import get_annotation_manager

            data_dir = ctx.get("data_dir", Path.cwd() / ".data")
            manager = get_annotation_manager(data_dir)

            # Update annotation
            annotation = ProgressManager.run_with_spinner(
                lambda: manager.update(annotation_id, note_text),
                "Updating annotation...",
                "Annotation updated",
            )

            if annotation:
                self.print_success(f"Updated annotation '{annotation_id}'")
                preview = annotation.preview(80)
                self.console.print(f"  [dim]{preview}[/dim]")
                return 0

            self.print_error(f"Annotation '{annotation_id}' not found")
            return 1

        except ValueError as e:
            self.print_error(str(e))
            return 1
        except Exception as e:
            return self.handle_error(e, "Update annotation failed")


# =============================================================================
# Typer Command Wrappers
# =============================================================================


def command(
    chunk_id: str = typer.Argument(..., help="ID of the chunk to annotate"),
    note_text: str = typer.Argument(
        ..., help="Annotation text (up to 10,000 characters, multi-line supported)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Add a personal annotation to a chunk.

    Annotations are persistent notes attached to chunks that help you
    record your thoughts alongside source material. Annotations remain
    even if the source chunk is re-ingested (mapped via content hash).

    Examples:
        # Add a simple annotation
        ingestforge annotate abc123def "This is an important concept"

        # Add a multi-line annotation (using quotes)
        ingestforge annotate abc123def "Key points:
        - First point
        - Second point
        - Third point"

        # Annotate a chunk in a specific project
        ingestforge annotate abc123def "Note here" --project /path/to/project
    """
    cmd = AnnotateCommand()
    exit_code = cmd.execute(chunk_id, note_text, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def list_annotations_command(
    chunk_id: Optional[str] = typer.Option(
        None, "--chunk", "-c", help="Filter annotations to specific chunk"
    ),
    limit: int = typer.Option(
        50, "--limit", "-n", help="Maximum annotations to display"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """List all annotations in the knowledge base.

    Shows all annotations with their chunk IDs, creation dates,
    and text previews. Use --chunk to filter to a specific chunk.

    Examples:
        # List all annotations
        ingestforge annotations list

        # List annotations for a specific chunk
        ingestforge annotations list --chunk abc123def

        # Limit number of results
        ingestforge annotations list --limit 10
    """
    cmd = ListAnnotationsCommand()
    exit_code = cmd.execute(chunk_id, limit, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def delete_annotation_command(
    annotation_id: str = typer.Argument(
        ..., help="ID of the annotation to delete (e.g., ann_abc123)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Delete an annotation.

    Removes an annotation permanently. Use 'ingestforge annotations list'
    to find annotation IDs.

    Examples:
        # Delete an annotation
        ingestforge annotations delete ann_abc123def456
    """
    cmd = DeleteAnnotationCommand()
    exit_code = cmd.execute(annotation_id, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def update_annotation_command(
    annotation_id: str = typer.Argument(..., help="ID of the annotation to update"),
    note_text: str = typer.Argument(..., help="New annotation text"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
) -> None:
    """Update an existing annotation.

    Replaces the text of an existing annotation. Use 'ingestforge annotations list'
    to find annotation IDs.

    Examples:
        # Update an annotation
        ingestforge annotations update ann_abc123 "Updated note text"
    """
    cmd = UpdateAnnotationCommand()
    exit_code = cmd.execute(annotation_id, note_text, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# =============================================================================
# Typer Sub-App for 'annotations' command group
# =============================================================================

annotations_app = typer.Typer(
    name="annotations",
    help="Manage annotations attached to chunks (ORG-004)",
    add_completion=False,
)

annotations_app.command("list")(list_annotations_command)
annotations_app.command("delete")(delete_annotation_command)
annotations_app.command("update")(update_annotation_command)
