"""Preview command - Show chunk with surrounding context.

Displays a chunk along with surrounding text to provide context
for understanding where the chunk comes from in the source document.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any
import typer
from rich.panel import Panel

from ingestforge.cli.core import IngestForgeCommand
from ingestforge.cli.console import tip


class PreviewCommand(IngestForgeCommand):
    """Preview a chunk with surrounding context."""

    def execute(
        self,
        chunk_id: str,
        project: Optional[Path] = None,
        context_chars: int = 500,
    ) -> int:
        """Execute chunk preview.

        Args:
            chunk_id: ID of the chunk to preview
            project: Project directory
            context_chars: Number of characters to show before/after (default 500)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs
            self.validate_non_empty_string(chunk_id, "chunk_id")

            # Initialize with storage
            ctx = self.initialize_context(project, require_storage=True)
            storage = ctx["storage"]

            # Get the chunk
            chunk = storage.get_chunk(chunk_id)
            if chunk is None:
                self.print_error(f"Chunk not found: {chunk_id}")
                tip("Use 'ingestforge query' to find chunk IDs")
                return 1

            # Display chunk info header
            self._display_chunk_header(chunk)

            # Try to get surrounding context
            context_before, context_after = self._get_surrounding_context(
                storage, chunk, context_chars
            )

            # Display the preview
            self._display_preview(chunk, context_before, context_after)

            return 0

        except Exception as e:
            return self.handle_error(e, "Preview failed")

    def _display_chunk_header(self, chunk: Any) -> None:
        """Display chunk metadata header.

        Args:
            chunk: ChunkRecord to display info for
        """
        source = getattr(chunk, "source_file", "Unknown")
        doc_id = getattr(chunk, "document_id", "Unknown")
        chunk_type = getattr(chunk, "chunk_type", "content")
        section = getattr(chunk, "section_title", "") or "No section"
        word_count = getattr(chunk, "word_count", 0)

        self.console.print()
        self.console.print("[bold cyan]Chunk Preview[/bold cyan]")
        self.console.print(f"  [dim]ID:[/dim] {chunk.chunk_id}")
        self.console.print(f"  [dim]Source:[/dim] {source}")
        self.console.print(f"  [dim]Document:[/dim] {doc_id}")
        self.console.print(f"  [dim]Section:[/dim] {section}")
        self.console.print(f"  [dim]Type:[/dim] {chunk_type}")
        self.console.print(f"  [dim]Words:[/dim] {word_count}")
        self.console.print()

    def _get_surrounding_context(
        self,
        storage: Any,
        chunk: Any,
        context_chars: int,
    ) -> tuple[str, str]:
        """Get surrounding context from adjacent chunks.

        Args:
            storage: ChunkRepository instance
            chunk: Current chunk
            context_chars: Max chars for context

        Returns:
            Tuple of (context_before, context_after)
        """
        context_before = ""
        context_after = ""

        try:
            # Get all chunks from the same document
            doc_id = getattr(chunk, "document_id", None)
            if not doc_id:
                return "", ""

            doc_chunks = storage.get_chunks_by_document(doc_id)
            if not doc_chunks:
                return "", ""

            # Sort by chunk_index
            doc_chunks.sort(key=lambda c: getattr(c, "chunk_index", 0))

            # Find current chunk position
            current_idx = None
            for i, c in enumerate(doc_chunks):
                if c.chunk_id == chunk.chunk_id:
                    current_idx = i
                    break

            if current_idx is None:
                return "", ""

            # Get previous chunk(s) for context_before
            if current_idx > 0:
                prev_chunks = doc_chunks[max(0, current_idx - 2) : current_idx]
                context_parts = []
                for c in prev_chunks:
                    text = getattr(c, "content", "")
                    context_parts.append(text)
                context_before = "\n".join(context_parts)
                if len(context_before) > context_chars:
                    context_before = "..." + context_before[-context_chars:]

            # Get next chunk(s) for context_after
            if current_idx < len(doc_chunks) - 1:
                next_chunks = doc_chunks[current_idx + 1 : current_idx + 3]
                context_parts = []
                for c in next_chunks:
                    text = getattr(c, "content", "")
                    context_parts.append(text)
                context_after = "\n".join(context_parts)
                if len(context_after) > context_chars:
                    context_after = context_after[:context_chars] + "..."

        except Exception:
            # If we can't get context, that's okay - just show the chunk
            pass

        return context_before, context_after

    def _display_preview(
        self,
        chunk: Any,
        context_before: str,
        context_after: str,
    ) -> None:
        """Display the chunk with surrounding context.

        Args:
            chunk: ChunkRecord to display
            context_before: Text before the chunk
            context_after: Text after the chunk
        """
        content = getattr(chunk, "content", str(chunk))

        # Build preview text
        preview_parts = []

        if context_before:
            preview_parts.append("[dim]--- Context Before ---[/dim]")
            # Escape any Rich markup in the context
            safe_before = context_before.replace("[", r"\[").replace("]", r"\]")
            preview_parts.append(f"[dim]{safe_before}[/dim]")
            preview_parts.append("")

        preview_parts.append("[bold green]--- Chunk Content ---[/bold green]")
        # Keep chunk content readable - escape markup
        safe_content = content.replace("[", r"\[").replace("]", r"\]")
        preview_parts.append(safe_content)

        if context_after:
            preview_parts.append("")
            preview_parts.append("[dim]--- Context After ---[/dim]")
            safe_after = context_after.replace("[", r"\[").replace("]", r"\]")
            preview_parts.append(f"[dim]{safe_after}[/dim]")

        preview_text = "\n".join(preview_parts)

        # Display in a panel
        self.console.print(
            Panel(
                preview_text,
                title="Source Preview",
                border_style="cyan",
                expand=False,
            )
        )

        self.console.print()
        tip("Use 'ingestforge query' to find related chunks")


# Typer command wrapper
def command(
    chunk_id: str = typer.Argument(..., help="Chunk ID to preview"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    context: int = typer.Option(
        500,
        "--context",
        "-c",
        help="Characters of context to show before/after (default: 500)",
    ),
) -> None:
    """Preview a chunk with surrounding context.

    Shows a chunk along with text from adjacent chunks to provide
    context for understanding where the content comes from.

    The chunk ID can be found from query results or status output.

    Examples:
        # Preview a specific chunk
        ingestforge preview abc123

        # Preview with more context
        ingestforge preview abc123 --context 1000

        # Preview in specific project
        ingestforge preview abc123 --project /path/to/project
    """
    cmd = PreviewCommand()
    exit_code = cmd.execute(chunk_id, project, context)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
