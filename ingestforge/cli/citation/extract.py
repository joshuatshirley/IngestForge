"""Extract command - Extract citations from knowledge base.

Extracts citations and references from ingested documents.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer
from rich.table import Table

from ingestforge.cli.citation.base import CitationCommand


class ExtractCommand(CitationCommand):
    """Extract citations from knowledge base."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> int:
        """Extract citations from knowledge base.

        Args:
            project: Project directory
            output: Output file for citations
            source: Filter by specific source

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context (Commandment #7: Check parameters)
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve chunks
            chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not chunks:
                self._handle_no_chunks()
                return 0

            # Extract citations
            citations = self.extract_citations_from_chunks(chunks)

            # Filter by source if specified
            if source:
                citations = self._filter_by_source(citations, source)

            if not citations:
                self._handle_no_citations(source)
                return 0

            # Display results
            self._display_citations(citations)

            # Save to file if requested
            if output:
                self._save_citations(output, citations)

            return 0

        except Exception as e:
            return self.handle_error(e, "Citation extraction failed")

    def _handle_no_chunks(self) -> None:
        """Handle case where no chunks found."""
        self.print_warning("Knowledge base is empty")
        self.print_info("Try ingesting some documents first")

    def _filter_by_source(self, citations: list, source: str) -> list[Any]:
        """Filter citations by source.

        Args:
            citations: List of citations
            source: Source filter

        Returns:
            Filtered list
        """
        return [c for c in citations if source.lower() in c["source"].lower()]

    def _handle_no_citations(self, source: Optional[str]) -> None:
        """Handle case where no citations found.

        Args:
            source: Optional source filter
        """
        if source:
            self.print_warning(f"No citations found for source: '{source}'")
        else:
            self.print_warning("No citations found in knowledge base")

        self.print_info(
            "Citations may not be present in metadata. "
            "Try re-ingesting with citation extraction enabled."
        )

    def _display_citations(self, citations: list) -> None:
        """Display citations table.

        Args:
            citations: List of citations
        """
        self.console.print()

        table = Table(title=f"Extracted Citations ({len(citations)})")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Citation", style="green", width=60)
        table.add_column("Source", style="yellow", width=30)
        table.add_column("Page", style="magenta", width=8)

        for idx, citation in enumerate(citations, 1):
            page = str(citation.get("page", "")) if citation.get("page") else "-"

            table.add_row(
                str(idx),
                citation["text"][:100],  # Truncate long citations
                citation["source"][:30],
                page,
            )

        self.console.print(table)

    def _format_citation_line(self, idx: int, citation: dict) -> str:
        """
        Format a single citation line.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            idx: Citation index (1-based)
            citation: Citation dictionary

        Returns:
            Formatted citation string
        """
        assert idx > 0, "Index must be positive"
        assert citation is not None, "Citation cannot be None"
        assert "text" in citation, "Citation must have 'text' field"
        text = f"{idx}. {citation['text']}"
        page = citation.get("page")
        if not page:
            return text + "\n"

        # Add page number
        return text + f" (p. {page})\n"

    def _format_source_section(self, source: str, source_citations: list[Any]) -> str:
        """
        Format a source section with all its citations.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            source: Source name
            source_citations: List of citations for this source

        Returns:
            Formatted section string
        """
        assert source is not None, "Source cannot be None"
        assert source_citations is not None, "Citations cannot be None"
        assert isinstance(source_citations, list), "Citations must be list"
        lines = [f"## {source}\n\n"]
        for idx, citation in enumerate(source_citations, 1):
            citation_line = self._format_citation_line(idx, citation)
            lines.append(citation_line)

        lines.append("\n")
        return "".join(lines)

    def _generate_header(self, citation_count: int) -> str:
        """
        Generate file header with metadata.

        Rule #1: Extracted helper
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            citation_count: Total number of citations

        Returns:
            Header string
        """
        from datetime import datetime

        assert citation_count >= 0, "Count must be non-negative"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return (
            "# Extracted Citations\n\n"
            f"**Generated:** {timestamp}\n"
            f"**Total Citations:** {citation_count}\n\n"
            "---\n\n"
        )

    def _save_citations(self, output: Path, citations: list) -> None:
        """
        Save citations to file.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            output: Output file path
            citations: List of citations
        """
        assert output is not None, "Output path cannot be None"
        assert citations is not None, "Citations cannot be None"
        assert isinstance(citations, list), "Citations must be list"

        try:
            header = self._generate_header(len(citations))
            grouped = self.group_citations_by_source(citations)

            # Build content sections
            sections = [header]
            for source, source_citations in sorted(grouped.items()):
                section = self._format_source_section(source, source_citations)
                sections.append(section)

            # Write to file
            content = "".join(sections)
            output.write_text(content, encoding="utf-8")

            self.print_success(f"Citations saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save citations: {e}")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for citations"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter by specific source"
    ),
) -> None:
    """Extract citations from knowledge base.

    Extracts all citations and references found in document
    metadata for bibliography generation.

    Examples:
        # Extract all citations
        ingestforge citation extract

        # Filter by source
        ingestforge citation extract --source "research_paper.pdf"

        # Save to file
        ingestforge citation extract --output citations.md

        # Specific project
        ingestforge citation extract -p /path/to/project
    """
    cmd = ExtractCommand()
    exit_code = cmd.execute(project, output, source)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
