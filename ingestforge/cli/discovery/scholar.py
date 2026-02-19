"""Semantic Scholar search command - Search papers with citation data.

This command provides access to the Semantic Scholar API for searching
academic papers with citation counts and traversing citation networks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.discovery.base import DiscoveryCommand
from ingestforge.discovery.semantic_scholar import (
    SemanticScholarClient,
    ScholarPaper,
    export_bibtex,
)


class ScholarSearchCommand(DiscoveryCommand):
    """Search Semantic Scholar for papers with citation data."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = SemanticScholarClient(api_key=api_key)

    def execute(
        self,
        query: str,
        max_results: int = 10,
        fields_of_study: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute Semantic Scholar search.

        Rule #4: Function <60 lines.
        Rule #7: All parameters validated.
        """
        try:
            self.print_info(f"Searching Semantic Scholar for: {query}")
            papers = self._client.search(
                query,
                limit=max_results,
                fields_of_study=fields_of_study,
                year_range=year_range,
            )

            if not papers:
                self.print_warning("No results found")
                return 0

            self._output_results(papers, query, output_format, output_file)
            return 0

        except Exception as e:
            return self.handle_error(e, "Semantic Scholar search failed")

    def _output_results(
        self,
        papers: List[ScholarPaper],
        query: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output results in requested format.

        Rule #1: Early return for each format.
        """
        if output_format == "bibtex":
            bibtex = export_bibtex(papers)
            self._write_or_print(bibtex, output_file)
            return

        if output_format == "json":
            json_data = self._papers_to_json(papers, query)
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Default: table format
        self._display_table(papers, query)
        if output_file:
            json_data = self._papers_to_json(papers, query)
            output_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")

    def _display_table(self, papers: List[ScholarPaper], query: str) -> None:
        """Display results as rich table.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Semantic Scholar: {query}[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

        table = Table(title=f"Found {len(papers)} papers", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Title", width=40)
        table.add_column("Authors", width=20)
        table.add_column("Year", width=6)
        table.add_column("Citations", width=10, justify="right")
        table.add_column("Fields", width=15)

        for i, paper in enumerate(papers, 1):
            authors_str = self._format_authors(paper.authors)
            fields = (
                ", ".join(paper.fields_of_study[:2]) if paper.fields_of_study else "N/A"
            )

            table.add_row(
                str(i),
                paper.title[:80] + ("..." if len(paper.title) > 80 else ""),
                authors_str,
                str(paper.year) if paper.year else "N/A",
                f"{paper.citation_count:,}",
                fields[:20],
            )

        self.console.print(table)
        self.console.print()
        self.print_info("Use --format bibtex for BibTeX export")

    def _format_authors(self, authors: list, max_names: int = 2) -> str:
        """Format author list with truncation."""
        if not authors:
            return "Unknown"
        names = [a.name for a in authors[:max_names]]
        result = ", ".join(names)
        if len(authors) > max_names:
            result += " et al."
        return result

    def _papers_to_json(self, papers: List[ScholarPaper], query: str) -> dict:
        """Convert papers to JSON-serializable dict."""
        return {
            "query": query,
            "count": len(papers),
            "papers": [
                {
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": [
                        {"name": a.name, "author_id": a.author_id} for a in p.authors
                    ],
                    "abstract": p.abstract,
                    "year": p.year,
                    "citation_count": p.citation_count,
                    "reference_count": p.reference_count,
                    "url": p.url,
                    "venue": p.venue,
                    "fields_of_study": p.fields_of_study,
                    "is_open_access": p.is_open_access,
                    "open_access_pdf": p.open_access_pdf_url,
                    "doi": p.doi,
                    "arxiv_id": p.arxiv_id,
                }
                for p in papers
            ],
        }

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


class ScholarCitationsCommand(DiscoveryCommand):
    """Get citations for a specific paper."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = SemanticScholarClient(api_key=api_key)

    def execute(
        self,
        paper_id: str,
        limit: int = 50,
        depth: int = 1,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute citation lookup.

        Rule #4: Function <60 lines.
        """
        try:
            # Get source paper info
            self.print_info(f"Fetching paper: {paper_id}")
            source_paper = self._client.get_paper(paper_id)

            if not source_paper:
                self.print_error(f"Paper not found: {paper_id}")
                return 1

            self._display_source_paper(source_paper)

            # Get citations
            self.print_info(f"Fetching citations (depth={depth})...")
            result = self._client.get_citations(paper_id, limit=limit, depth=depth)

            if not result.papers:
                self.print_warning("No citations found")
                return 0

            self._output_results(
                result.papers, source_paper.title, output_format, output_file
            )
            return 0

        except Exception as e:
            return self.handle_error(e, "Citation lookup failed")

    def _display_source_paper(self, paper: ScholarPaper) -> None:
        """Display source paper info."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{paper.title}[/bold]\n"
                f"[dim]{paper.citation_count:,} citations | {paper.reference_count} references[/dim]",
                title="Source Paper",
                border_style="green",
            )
        )

    def _output_results(
        self,
        papers: List[ScholarPaper],
        source_title: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output citation results."""
        if output_format == "bibtex":
            bibtex = export_bibtex(papers)
            self._write_or_print(bibtex, output_file)
            return

        if output_format == "json":
            json_data = {
                "source_title": source_title,
                "citing_papers": len(papers),
                "papers": [
                    {
                        "paper_id": p.paper_id,
                        "title": p.title,
                        "year": p.year,
                        "citations": p.citation_count,
                    }
                    for p in papers
                ],
            }
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Table format
        self._display_table(papers, source_title)

    def _display_table(self, papers: List[ScholarPaper], source_title: str) -> None:
        """Display citations as table."""
        self.console.print()
        table = Table(title=f"Papers citing: {source_title[:50]}...", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Title", width=50)
        table.add_column("Year", width=6)
        table.add_column("Citations", width=10, justify="right")

        for i, paper in enumerate(papers, 1):
            table.add_row(
                str(i),
                paper.title[:100],
                str(paper.year) if paper.year else "N/A",
                f"{paper.citation_count:,}",
            )

        self.console.print(table)
        self.console.print()

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


class ScholarReferencesCommand(DiscoveryCommand):
    """Get references for a specific paper."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = SemanticScholarClient(api_key=api_key)

    def execute(
        self,
        paper_id: str,
        limit: int = 50,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute references lookup.

        Rule #4: Function <60 lines.
        """
        try:
            self.print_info(f"Fetching references for: {paper_id}")
            papers = self._client.get_references(paper_id, limit=limit)

            if not papers:
                self.print_warning("No references found")
                return 0

            self._output_results(papers, paper_id, output_format, output_file)
            return 0

        except Exception as e:
            return self.handle_error(e, "References lookup failed")

    def _output_results(
        self,
        papers: List[ScholarPaper],
        paper_id: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output reference results."""
        if output_format == "bibtex":
            bibtex = export_bibtex(papers)
            if output_file:
                output_file.write_text(bibtex, encoding="utf-8")
                self.print_success(f"Saved to: {output_file}")
            else:
                self.console.print(bibtex)
            return

        # Table format
        self.console.print()
        table = Table(title=f"References ({len(papers)} papers)", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Title", width=50)
        table.add_column("Year", width=6)
        table.add_column("Citations", width=10, justify="right")

        for i, paper in enumerate(papers, 1):
            table.add_row(
                str(i),
                paper.title[:100],
                str(paper.year) if paper.year else "N/A",
                f"{paper.citation_count:,}",
            )

        self.console.print(table)


# Typer command wrappers


def command(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(
        10,
        "--max-results",
        "-n",
        help="Maximum results (1-100)",
    ),
    fields: Optional[str] = typer.Option(
        None,
        "--fields",
        "-F",
        help="Filter by fields of study (comma-separated)",
    ),
    year_from: Optional[int] = typer.Option(
        None,
        "--year-from",
        help="Filter publications from this year",
    ),
    year_to: Optional[int] = typer.Option(
        None,
        "--year-to",
        help="Filter publications up to this year",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, bibtex",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save output to file",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="SEMANTIC_SCHOLAR_API_KEY",
        help="Semantic Scholar API key (for higher rate limits)",
    ),
) -> None:
    """Search Semantic Scholar for academic papers.

    Searches the Semantic Scholar database for papers with
    citation counts and metadata.

    Examples:
        # Search for papers
        ingestforge discovery scholar "neural networks"

        # Filter by field and year
        ingestforge discovery scholar "transformers" -F "Computer Science" --year-from 2020

        # Export as BibTeX
        ingestforge discovery scholar "attention" -f bibtex -o refs.bib

    Note:
        - Free tier: 100 requests per 5 minutes
        - Use --api-key for higher limits
    """
    if max_results > 100:
        typer.echo("Warning: Limiting results to 100")
        max_results = 100

    fields_list = [f.strip() for f in fields.split(",")] if fields else None
    year_range = None
    if year_from or year_to:
        year_range = (year_from or 1900, year_to or 2100)

    cmd = ScholarSearchCommand(api_key=api_key)
    exit_code = cmd.execute(
        query,
        max_results,
        fields_list,
        year_range,
        output_format,
        output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def citations_command(
    paper_id: str = typer.Argument(
        ..., help="Paper ID (Semantic Scholar ID, DOI, or arXiv ID)"
    ),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum citations to fetch",
    ),
    depth: int = typer.Option(
        1,
        "--depth",
        "-d",
        help="Citation depth (1-3)",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, bibtex",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save output to file",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="SEMANTIC_SCHOLAR_API_KEY",
        help="Semantic Scholar API key",
    ),
) -> None:
    """Get citations for a paper.

    Fetches papers that cite the given paper. Supports multiple
    depth levels for citation network traversal.

    Paper ID formats:
        - Semantic Scholar ID: 649def34f8be52c8b66281af98ae884c09aef38b
        - DOI: DOI:10.1234/example
        - arXiv: ARXIV:2301.12345

    Examples:
        # Get citations for a paper
        ingestforge discovery scholar-citations "DOI:10.1038/nature12373"

        # Traverse 2 levels deep
        ingestforge discovery scholar-citations paper_id -d 2

        # Export as BibTeX
        ingestforge discovery scholar-citations paper_id -f bibtex -o citations.bib
    """
    if depth > 3:
        typer.echo("Warning: Limiting depth to 3")
        depth = 3

    cmd = ScholarCitationsCommand(api_key=api_key)
    exit_code = cmd.execute(paper_id, limit, depth, output_format, output_file)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def references_command(
    paper_id: str = typer.Argument(..., help="Paper ID"),
    limit: int = typer.Option(
        50,
        "--limit",
        "-n",
        help="Maximum references to fetch",
    ),
    output_format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table, json, bibtex",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save output to file",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="SEMANTIC_SCHOLAR_API_KEY",
        help="Semantic Scholar API key",
    ),
) -> None:
    """Get references cited by a paper.

    Fetches papers that are referenced by the given paper.

    Examples:
        # Get references
        ingestforge discovery scholar-references paper_id

        # Export as BibTeX
        ingestforge discovery scholar-references paper_id -f bibtex -o refs.bib
    """
    cmd = ScholarReferencesCommand(api_key=api_key)
    exit_code = cmd.execute(paper_id, limit, output_format, output_file)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)
