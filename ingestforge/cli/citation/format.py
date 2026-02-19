"""Format command - Format citations in various styles.

Formats extracted citations in standard academic styles.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer
from rich.panel import Panel

from ingestforge.cli.citation.base import CitationCommand


class FormatCommand(CitationCommand):
    """Format citations in various styles."""

    def execute(
        self,
        style: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        source: Optional[str] = None,
    ) -> int:
        """Format citations in specified style.

        Args:
            style: Citation style (apa/mla/chicago/bibtex)
            project: Project directory
            output: Output file for formatted citations
            source: Filter by specific source

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_style(style)

            # Initialize context
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

            # Format citations
            formatted = self._format_all_citations(citations, style)

            # Display results
            self._display_formatted_citations(formatted, style)

            # Save to file if requested
            if output:
                self._save_formatted_citations(output, formatted, style)

            return 0

        except Exception as e:
            return self.handle_error(e, "Citation formatting failed")

    def validate_style(self, style: str) -> None:
        """Validate citation style.

        Args:
            style: Style string to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        valid_styles = ["apa", "mla", "chicago", "bibtex"]

        if style.lower() not in valid_styles:
            raise typer.BadParameter(
                f"Invalid style '{style}'. "
                f"Must be one of: {', '.join(valid_styles)}"
            )

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

    def _format_all_citations(self, citations: list, style: str) -> list[Any]:
        """Format all citations.

        Args:
            citations: List of citations
            style: Citation style

        Returns:
            List of formatted citation strings
        """
        formatted = []

        for citation in citations:
            formatted_text = self.format_citation(citation, style)
            formatted.append(
                {
                    "original": citation,
                    "formatted": formatted_text,
                }
            )

        return formatted

    def _display_formatted_citations(self, formatted: list, style: str) -> None:
        """Display formatted citations.

        Args:
            formatted: List of formatted citations
            style: Citation style
        """
        self.console.print()

        # Build content using shared grouping logic
        lines = [f"# Bibliography ({style.upper()} Style)\n\n"]
        by_source = self._group_citations_by_source(formatted)

        for source, items in sorted(by_source.items()):
            lines.append(f"## {source}\n\n")

            for idx, item in enumerate(items, 1):
                citation_line = self._format_citation_item(item, style, idx)
                lines.append(citation_line)

            lines.append("\n")

        content = "".join(lines)

        panel = Panel(
            content,
            title=f"[bold green]Formatted Citations ({style.upper()})[/bold green]",
            border_style="green",
        )

        self.console.print(panel)

    def _save_formatted_citations(
        self, output: Path, formatted: list, style: str
    ) -> None:
        """Save formatted citations to file.

        Args:
            output: Output file path
            formatted: List of formatted citations
            style: Citation style
        """
        try:
            header = self._build_citation_header(formatted, style)
            by_source = self._group_citations_by_source(formatted)

            lines = [header]

            for source, items in sorted(by_source.items()):
                lines.append(f"## {source}\n\n")

                for idx, item in enumerate(items, 1):
                    citation_line = self._format_citation_item(item, style, idx)
                    lines.append(citation_line)

                lines.append("\n")

            output.write_text("".join(lines), encoding="utf-8")
            self.print_success(f"Formatted citations saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save citations: {e}")

    def _group_citations_by_source(self, formatted: list) -> dict[str, Any]:
        """Group formatted citations by source.

        Args:
            formatted: List of formatted citation dicts

        Returns:
            Dictionary mapping source to list of citations
        """
        by_source = {}

        for item in formatted:
            source = item["original"]["source"]
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)

        return by_source

    def _format_citation_item(self, item: dict, style: str, idx: int) -> str:
        """Format a single citation item.

        Args:
            item: Citation item dictionary
            style: Citation style
            idx: Citation index (1-based)

        Returns:
            Formatted citation line with newline
        """
        if style.lower() == "bibtex":
            return f"{item['formatted']}\n\n"

        return f"{idx}. {item['formatted']}\n"

    def _build_citation_header(self, formatted: list, style: str) -> str:
        """Build header section for saved citations.

        Args:
            formatted: List of formatted citations
            style: Citation style

        Returns:
            Header string with metadata
        """
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return (
            f"# Bibliography ({style.upper()} Style)\n\n"
            f"**Generated:** {timestamp}\n"
            f"**Total Citations:** {len(formatted)}\n"
            f"**Style:** {style.upper()}\n\n"
            "---\n\n"
        )


# Typer command wrapper
def command(
    style: str = typer.Argument(..., help="Citation style (apa/mla/chicago/bibtex)"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for formatted citations"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter by specific source"
    ),
) -> None:
    """Format citations in standard academic styles.

    Formats extracted citations according to academic citation
    standards for use in papers and documents.

    Supported styles:
    - apa: APA (American Psychological Association)
    - mla: MLA (Modern Language Association)
    - chicago: Chicago Manual of Style
    - bibtex: BibTeX format for LaTeX

    Examples:
        # Format in APA style
        ingestforge citation format apa

        # Format in MLA style
        ingestforge citation format mla --output bibliography.md

        # Format specific source
        ingestforge citation format chicago --source "paper.pdf"

        # Generate BibTeX for LaTeX
        ingestforge citation format bibtex -o references.bib

        # Specific project
        ingestforge citation format apa -p /path/to/project
    """
    cmd = FormatCommand()
    exit_code = cmd.execute(style, project, output, source)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
