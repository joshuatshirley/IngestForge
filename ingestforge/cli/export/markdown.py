"""Markdown export command - Export knowledge base to Markdown.

Exports chunks to well-formatted Markdown documents with streaming support
for large corpora and optional citation tracking for research notes.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
#1 (Simple Control Flow), and #3 (Avoid Memory Issues).

Features:
- Streaming export to handle large datasets efficiently (EXPORT-001.1)
- Citation section generation for bibliography tracking (EXPORT-001.2)
- Grouped or sequential output formats
- Query filtering and chunk limits
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Dict
import typer

from ingestforge.cli.export.base import ExportCommand


class MarkdownExportCommand(ExportCommand):
    """Export knowledge base to Markdown format."""

    def execute(
        self,
        output: Path,
        project: Optional[Path] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
        group_by_source: bool = True,
        include_citations: bool = False,
    ) -> int:
        """Export knowledge base to Markdown.

        Args:
            output: Output Markdown file
            project: Project directory
            query: Optional search query to filter chunks
            limit: Optional limit on number of chunks
            group_by_source: Group chunks by source document
            include_citations: Include citations section at end

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate parameters (Commandment #7)
            self.validate_output_path(output)

            if limit is not None and limit < 1:
                raise typer.BadParameter("Limit must be positive")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve chunks
            chunks = self.search_filtered_chunks(ctx["storage"], query, limit)

            if not chunks:
                self._handle_no_chunks(query)
                return 0

            # Use streaming export for large corpora (Commandment #3)
            self._stream_markdown_to_file(
                output, chunks, group_by_source, query, include_citations
            )

            # Display summary
            self._display_summary(len(chunks), output)

            return 0

        except Exception as e:
            return self.handle_error(e, "Markdown export failed")

    def _stream_markdown_to_file(
        self,
        output: Path,
        chunks: list,
        group_by_source: bool,
        query: Optional[str],
        include_citations: bool,
    ) -> None:
        """Stream Markdown content directly to file.

        Writes incrementally to avoid memory issues with large corpora.
        Follows Commandment #3 (Avoid Memory Issues).

        Args:
            output: Output file path
            chunks: List of chunks to export
            group_by_source: Whether to group by source
            query: Optional search query
            include_citations: Whether to include citations section
        """
        sources_seen: Dict[str, int] = {}

        with open(output, "w", encoding="utf-8") as f:
            # Write header
            f.write(self._generate_header(query, len(chunks)))

            # Stream content chunks
            if group_by_source:
                self._stream_grouped_content(f, chunks, sources_seen)
            else:
                self._stream_sequential_content(f, chunks, sources_seen)

            # Write citations section if requested
            if include_citations and sources_seen:
                f.write(self._generate_citations_section(sources_seen))

        self.print_success(f"Exported to: {output}")

    def _handle_no_chunks(self, query: Optional[str]) -> None:
        """Handle case where no chunks found.

        Args:
            query: Optional search query
        """
        if query:
            self.print_warning(f"No chunks found matching: '{query}'")
        else:
            self.print_warning("Knowledge base is empty")

        self.print_info("Try ingesting some documents first")

    def _generate_markdown(
        self, chunks: list, group_by_source: bool, query: Optional[str]
    ) -> str:
        """Generate Markdown content from chunks.

        Args:
            chunks: List of chunks to export
            group_by_source: Whether to group by source
            query: Optional search query

        Returns:
            Formatted Markdown string
        """
        lines = []

        # Header
        lines.append(self._generate_header(query, len(chunks)))

        # Content
        if group_by_source:
            lines.append(self._generate_grouped_content(chunks))
        else:
            lines.append(self._generate_sequential_content(chunks))

        return "\n".join(lines)

    def _generate_header(self, query: Optional[str], count: int) -> str:
        """Generate Markdown header.

        Args:
            query: Optional search query
            count: Number of chunks

        Returns:
            Header string
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_parts = [
            "# Knowledge Base Export\n",
            f"\n**Generated:** {timestamp}\n",
            f"**Chunks:** {count}\n",
        ]

        if query:
            header_parts.append(f"**Filter:** {query}\n")

        header_parts.append("\n---\n\n")

        return "".join(header_parts)

    def _generate_grouped_content(self, chunks: list) -> str:
        """Generate content grouped by source.

        Args:
            chunks: List of chunks

        Returns:
            Formatted content string
        """
        grouped = self.group_chunks_by_source(chunks)

        content_parts = []

        for source, source_chunks in sorted(grouped.items()):
            # Source header
            content_parts.append(f"## {source}\n\n")
            content_parts.append(f"*{len(source_chunks)} chunks*\n\n")

            # Chunks from this source
            for idx, chunk in enumerate(source_chunks, 1):
                content_parts.append(self._format_chunk(chunk, idx))

            content_parts.append("\n---\n\n")

        return "".join(content_parts)

    def _generate_sequential_content(self, chunks: list) -> str:
        """Generate sequential content.

        Args:
            chunks: List of chunks

        Returns:
            Formatted content string
        """
        content_parts = []

        for idx, chunk in enumerate(chunks, 1):
            content_parts.append(self._format_chunk(chunk, idx))

        return "".join(content_parts)

    def _format_chunk(self, chunk: Any, idx: int) -> str:
        """Format a single chunk as Markdown.

        Args:
            chunk: Chunk to format
            idx: Chunk index

        Returns:
            Formatted chunk string
        """
        text = self.extract_chunk_text(chunk)
        metadata = self.extract_chunk_metadata(chunk)

        parts = [f"### Chunk {idx}\n\n"]

        # Add metadata if available
        if metadata:
            parts.append(self._format_metadata(metadata))

        # Add text content
        parts.append(f"{text}\n\n")

        return "".join(parts)

    def _format_metadata(self, metadata: dict) -> str:
        """Format metadata as Markdown.

        Args:
            metadata: Metadata dictionary

        Returns:
            Formatted metadata string
        """
        parts = ["**Metadata:**\n"]

        for key, value in sorted(metadata.items()):
            if key in ["source", "page", "chapter", "section"]:
                parts.append(f"- {key}: {value}\n")

        parts.append("\n")

        return "".join(parts)

    def _stream_grouped_content(
        self, f: Any, chunks: list, sources_seen: Dict[str, int]
    ) -> None:
        """Stream grouped content to file.

        Args:
            f: File handle to write to
            chunks: List of chunks
            sources_seen: Dictionary to track sources (modified in place)
        """
        grouped = self.group_chunks_by_source(chunks)

        for source, source_chunks in sorted(grouped.items()):
            # Track source for citations
            sources_seen[source] = len(source_chunks)

            # Write source header
            f.write(f"## {source}\n\n")
            f.write(f"*{len(source_chunks)} chunks*\n\n")

            # Write chunks from this source
            for idx, chunk in enumerate(source_chunks, 1):
                f.write(self._format_chunk(chunk, idx))

            f.write("\n---\n\n")

    def _stream_sequential_content(
        self, f: Any, chunks: list, sources_seen: Dict[str, int]
    ) -> None:
        """Stream sequential content to file.

        Args:
            f: File handle to write to
            chunks: List of chunks
            sources_seen: Dictionary to track sources (modified in place)
        """
        for idx, chunk in enumerate(chunks, 1):
            # Track source for citations
            metadata = self.extract_chunk_metadata(chunk)
            source = metadata.get("source", "unknown")
            sources_seen[source] = sources_seen.get(source, 0) + 1

            # Write chunk
            f.write(self._format_chunk(chunk, idx))

    def _generate_citations_section(self, sources: Dict[str, int]) -> str:
        """Generate citations section from sources.

        Args:
            sources: Dictionary mapping source names to chunk counts

        Returns:
            Formatted citations section
        """
        lines = ["\n---\n\n## Sources\n\n"]

        # Sort sources alphabetically for consistent output
        for idx, (source, count) in enumerate(sorted(sources.items()), 1):
            chunk_word = "chunk" if count == 1 else "chunks"
            lines.append(f"{idx}. **{source}** - {count} {chunk_word}\n")

        lines.append("\n")

        return "".join(lines)

    def _display_summary(self, chunk_count: int, output: Path) -> None:
        """Display export summary.

        Args:
            chunk_count: Number of chunks exported
            output: Output file path
        """
        self.console.print()
        self.print_info(f"Exported {chunk_count} chunks")
        self.print_info(f"Output file: {output}")


# Typer command wrapper
def command(
    output: Path = typer.Argument(..., help="Output Markdown file"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    query: Optional[str] = typer.Option(
        None, "--query", "-q", help="Filter chunks by search query"
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Limit number of chunks"
    ),
    no_grouping: bool = typer.Option(
        False, "--no-grouping", help="Don't group chunks by source"
    ),
    include_citations: bool = typer.Option(
        False, "--include-citations", "-c", help="Include citations section at end"
    ),
) -> None:
    """Export knowledge base to Markdown format.

    Creates a well-formatted Markdown document containing
    your knowledge base chunks.

    Features:
    - Groups chunks by source document
    - Includes metadata
    - Clean, readable formatting
    - Optional filtering and limits
    - Citation tracking for research notes

    Examples:
        # Export entire knowledge base
        ingestforge export markdown output.md

        # Export filtered chunks
        ingestforge export markdown ml_docs.md --query "machine learning"

        # Export with citations
        ingestforge export markdown notes.md --include-citations

        # Limit number of chunks
        ingestforge export markdown sample.md --limit 100

        # Export without grouping
        ingestforge export markdown sequential.md --no-grouping

        # Specific project
        ingestforge export markdown docs.md -p /path/to/project
    """
    cmd = MarkdownExportCommand()
    group_by_source = not no_grouping
    exit_code = cmd.execute(
        output, project, query, limit, group_by_source, include_citations
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
