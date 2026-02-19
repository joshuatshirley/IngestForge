"""Outline export command - Generate structured research documents.

Creates thesis-style documents with evidence mapped to outline points.
Uses LLM to match source chunks to outline structure.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
#1 (Simple Control Flow), and #3 (Avoid Memory Issues).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.export.base import ExportCommand
from ingestforge.cli.export.formatters import get_formatter


class OutlineExportCommand(ExportCommand):
    """Export knowledge base with thesis-style structure."""

    def execute(
        self,
        outline_file: Path,
        output: Path,
        project: Optional[Path] = None,
        output_format: str = "markdown",
        min_relevance: float = 0.5,
        include_citations: bool = True,
    ) -> int:
        """Export knowledge base as structured outline document.

        Args:
            outline_file: Path to outline file (markdown format)
            output: Output file path
            project: Project directory
            output_format: Output format (markdown, docx)
            min_relevance: Minimum relevance score for evidence (0.0-1.0)
            include_citations: Include citations section

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate parameters (Commandment #7)
            self._validate_parameters(outline_file, output, min_relevance)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Load outline
            outline_text = self._load_outline(outline_file)

            # Parse outline and map evidence
            mapped_outline = self._map_evidence(
                ctx, outline_text, outline_file.stem, min_relevance
            )

            # Format and save
            self._save_document(
                mapped_outline, output, output_format, include_citations
            )

            # Display summary
            self._display_summary(mapped_outline, output)

            return 0

        except Exception as e:
            return self.handle_error(e, "Outline export failed")

    def _validate_parameters(
        self, outline_file: Path, output: Path, min_relevance: float
    ) -> None:
        """Validate input parameters.

        Args:
            outline_file: Outline file path
            output: Output path
            min_relevance: Relevance threshold

        Raises:
            typer.BadParameter: If validation fails
        """
        if not outline_file.exists():
            raise typer.BadParameter(f"Outline file not found: {outline_file}")

        if min_relevance < 0.0 or min_relevance > 1.0:
            raise typer.BadParameter("min_relevance must be between 0.0 and 1.0")

        self.validate_output_path(output)

    def _load_outline(self, outline_file: Path) -> str:
        """Load outline from file.

        Args:
            outline_file: Path to outline file

        Returns:
            Outline text content
        """
        self.print_info(f"Loading outline from: {outline_file}")
        return outline_file.read_text(encoding="utf-8")

    def _map_evidence(
        self,
        ctx: dict,
        outline_text: str,
        title: str,
        min_relevance: float,
    ) -> "MappedOutline":
        """Parse outline and map chunks to points.

        Args:
            ctx: Execution context with storage
            outline_text: Outline content
            title: Document title
            min_relevance: Minimum relevance threshold

        Returns:
            MappedOutline with evidence
        """
        from ingestforge.core.export.outline_mapper import OutlineMapper

        self.print_info("Parsing outline structure...")
        mapper = OutlineMapper(ctx["config"])
        outline = mapper.parse_outline(outline_text, title)

        self.print_info(f"Found {len(outline.points)} outline points")

        # Get all chunks
        self.print_info("Loading chunks from storage...")
        storage = ctx["storage"]
        chunks = storage.get_all_chunks()

        if not chunks:
            self.print_warning("No chunks in storage")
            return outline

        self.print_info(f"Mapping {len(chunks)} chunks to outline points...")
        self.print_info("(This may take a moment with LLM scoring)")

        mapped = mapper.map_chunks_to_outline(outline, chunks, min_relevance)

        evidence_count = mapped.get_total_evidence_count()
        self.print_success(f"Found {evidence_count} evidence matches")

        return mapped

    def _save_document(
        self,
        outline: "MappedOutline",
        output: Path,
        output_format: str,
        include_citations: bool,
    ) -> None:
        """Save formatted document.

        Args:
            outline: Mapped outline
            output: Output path
            output_format: Format type
            include_citations: Include citations
        """

        formatter = get_formatter(output_format, include_citations)

        self.print_info(f"Generating {output_format} document...")
        formatter.save(outline, output)

        self.print_success(f"Saved to: {output}")

    def _display_summary(self, outline: "MappedOutline", output: Path) -> None:
        """Display export summary.

        Args:
            outline: Mapped outline
            output: Output file path
        """

        self.console.print()
        self.print_info(f"Title: {outline.title}")
        self.print_info(f"Points: {len(outline.points)}")
        self.print_info(f"Evidence matches: {outline.get_total_evidence_count()}")
        self.print_info(f"Output: {output}")


# Typer command wrapper
def command(
    outline: Path = typer.Argument(..., help="Path to outline file (markdown format)"),
    output: Path = typer.Argument(..., help="Output file path (.md or .docx)"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    format_type: str = typer.Option(
        "markdown", "--format", "-f", help="Output format (markdown, docx)"
    ),
    min_relevance: float = typer.Option(
        0.5, "--min-relevance", "-r", help="Minimum relevance score (0.0-1.0)"
    ),
    no_citations: bool = typer.Option(
        False, "--no-citations", help="Exclude citations section"
    ),
) -> None:
    """Export knowledge base as structured outline document.

    Takes an outline file (markdown format) and matches evidence
    from your knowledge base to each outline point using AI.

    The outline should use markdown headers to indicate structure:

        # Main Thesis

        ## First Supporting Point

        ## Second Supporting Point

        ### Detail under second point

    Examples:
        # Export with markdown outline
        ingestforge export outline thesis.md output.md

        # Export as Word document
        ingestforge export outline outline.md paper.docx --format docx

        # Higher relevance threshold
        ingestforge export outline outline.md notes.md --min-relevance 0.7

        # Exclude citations
        ingestforge export outline outline.md draft.md --no-citations
    """
    cmd = OutlineExportCommand()
    include_citations = not no_citations
    exit_code = cmd.execute(
        outline, output, project, format_type, min_relevance, include_citations
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
