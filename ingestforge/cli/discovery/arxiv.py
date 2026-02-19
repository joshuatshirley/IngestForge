"""arXiv search command - Search and download papers from arXiv.

This command provides direct access to the arXiv API for searching
and downloading academic papers with full metadata and BibTeX export."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.discovery.base import DiscoveryCommand
from ingestforge.discovery.arxiv_client import (
    ArxivSearcher,
    Paper,
    SortOrder,
    export_bibtex,
)


class ArxivSearchCommand(DiscoveryCommand):
    """Search arXiv and download papers using the enhanced client."""

    def __init__(self) -> None:
        """Initialize command."""
        super().__init__()
        self._searcher = ArxivSearcher()

    def execute(
        self,
        query: str,
        max_results: int = 10,
        sort: str = "relevance",
        download: bool = False,
        output_dir: Optional[Path] = None,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute arXiv search.

        Rule #4: Function <60 lines.
        Rule #7: All parameters validated.
        """
        try:
            sort_order = self._parse_sort_order(sort)

            if download and not output_dir:
                output_dir = Path("./arxiv_downloads")

            # Search arXiv
            self.print_info(f"Searching arXiv for: {query}")
            papers = self._searcher.search(
                query,
                limit=max_results,
                sort=sort_order,
            )

            if not papers:
                self.print_warning("No results found")
                return 0

            # Display/export results
            self._output_results(papers, query, output_format, output_file)

            # Download PDFs if requested
            if download and output_dir:
                self._download_papers(papers, output_dir)

            return 0

        except Exception as e:
            return self.handle_error(e, "arXiv search failed")

    def _parse_sort_order(self, sort: str) -> SortOrder:
        """Parse sort option to SortOrder enum."""
        sort_map = {
            "relevance": SortOrder.RELEVANCE,
            "date": SortOrder.SUBMITTED,
            "updated": SortOrder.LAST_UPDATED,
        }
        return sort_map.get(sort.lower(), SortOrder.RELEVANCE)

    def _output_results(
        self,
        papers: List[Paper],
        query: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output results in requested format.

        Rule #1: Early return for each format.
        Rule #4: Function <60 lines.
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

    def _display_table(self, papers: List[Paper], query: str) -> None:
        """Display results as rich table.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]arXiv Search: {query}[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

        table = Table(title=f"Found {len(papers)} papers", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("arXiv ID", width=14)
        table.add_column("Title", width=40)
        table.add_column("Authors", width=20)
        table.add_column("Year", width=6)
        table.add_column("Category", width=12)

        for i, paper in enumerate(papers, 1):
            authors_str = self._format_authors(paper.authors, max_names=2)
            table.add_row(
                str(i),
                paper.arxiv_id,
                paper.title[:80] + ("..." if len(paper.title) > 80 else ""),
                authors_str,
                str(paper.published.year),
                paper.primary_category or "N/A",
            )

        self.console.print(table)
        self.console.print()
        self.print_info("Use --download to fetch PDFs, --format bibtex for BibTeX")

    def _format_authors(self, authors: List[str], max_names: int = 2) -> str:
        """Format author list with truncation."""
        if not authors:
            return "Unknown"
        result = ", ".join(authors[:max_names])
        if len(authors) > max_names:
            result += " et al."
        return result

    def _papers_to_json(self, papers: List[Paper], query: str) -> dict:
        """Convert papers to JSON-serializable dict."""
        return {
            "query": query,
            "count": len(papers),
            "papers": [
                {
                    "arxiv_id": p.arxiv_id,
                    "title": p.title,
                    "authors": p.authors,
                    "abstract": p.abstract,
                    "published": p.published.isoformat(),
                    "updated": p.updated.isoformat(),
                    "pdf_url": p.pdf_url,
                    "abs_url": p.abs_url,
                    "categories": p.categories,
                    "primary_category": p.primary_category,
                    "doi": p.doi,
                    "journal_ref": p.journal_ref,
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

    def _download_papers(self, papers: List[Paper], output_dir: Path) -> None:
        """Download PDFs for papers.

        Rule #4: Function <60 lines.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.console.print()
        self.print_info(f"Downloading PDFs to: {output_dir}")

        downloaded = 0
        failed = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Downloading...", total=len(papers))

            for paper in papers:
                progress.update(task, description=f"[cyan]{paper.arxiv_id}[/]")
                result = self._searcher.download_pdf(paper.arxiv_id, output_dir)

                if result.success:
                    downloaded += 1
                else:
                    failed += 1

                progress.advance(task)

        self.console.print()
        self.print_success(f"Downloaded {downloaded} papers")
        if failed > 0:
            self.print_warning(f"Failed: {failed} papers")


class ArxivDownloadCommand(DiscoveryCommand):
    """Download a specific arXiv paper by ID."""

    def __init__(self) -> None:
        """Initialize command."""
        super().__init__()
        self._searcher = ArxivSearcher()

    def execute(
        self,
        arxiv_id: str,
        output_dir: Path,
        show_metadata: bool = False,
    ) -> int:
        """Execute download command.

        Rule #4: Function <60 lines.
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Get paper metadata
            if show_metadata:
                self.print_info(f"Fetching metadata for: {arxiv_id}")
                paper = self._searcher.get_paper(arxiv_id)
                if paper:
                    self._display_paper_details(paper)
                else:
                    self.print_warning("Paper not found")

            # Download PDF
            self.print_info(f"Downloading PDF for: {arxiv_id}")
            result = self._searcher.download_pdf(arxiv_id, output_dir)

            if result.success:
                self.print_success(f"Downloaded: {result.filename}")
                self.print_info(f"Size: {result.size_bytes:,} bytes")
                return 0
            else:
                self.print_error(f"Download failed: {result.error}")
                return 1

        except Exception as e:
            return self.handle_error(e, "Download failed")

    def _display_paper_details(self, paper: Paper) -> None:
        """Display detailed paper metadata."""
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{paper.title}[/bold]",
                title=f"arXiv:{paper.arxiv_id}",
                border_style="cyan",
            )
        )

        self.console.print(f"[bold]Authors:[/] {', '.join(paper.authors)}")
        self.console.print(
            f"[bold]Published:[/] {paper.published.strftime('%Y-%m-%d')}"
        )
        self.console.print(f"[bold]Categories:[/] {', '.join(paper.categories)}")

        if paper.abstract:
            self.console.print()
            self.console.print("[bold]Abstract:[/]")
            self.console.print(
                paper.abstract[:500] + "..."
                if len(paper.abstract) > 500
                else paper.abstract
            )

        self.console.print()


# Typer command wrappers


def command(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(
        10,
        "--max-results",
        "-n",
        help="Maximum results (1-100)",
    ),
    sort: str = typer.Option(
        "relevance",
        "--sort",
        "-s",
        help="Sort order: relevance, date, updated",
    ),
    download: bool = typer.Option(
        False,
        "--download",
        "-d",
        help="Download PDFs",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory for PDF downloads",
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
        "-O",
        help="Save output to file",
    ),
) -> None:
    """Search arXiv for academic papers.

    Searches the arXiv preprint repository for academic papers
    matching your query. Supports multiple output formats and PDF download.

    Examples:
        # Search for papers
        ingestforge discovery arxiv "transformer architecture"

        # Sort by date, get 20 results
        ingestforge discovery arxiv "quantum computing" -n 20 -s date

        # Export as BibTeX
        ingestforge discovery arxiv "machine learning" -f bibtex -O refs.bib

        # Download PDFs
        ingestforge discovery arxiv "climate change" -d -o ./papers

    Note:
        - Respects arXiv API rate limits (1 request/3 seconds)
        - Max 100 results per search
    """
    if max_results > 100:
        typer.echo("Warning: Limiting results to 100 (arXiv API limit)")
        max_results = 100

    if sort not in ["relevance", "date", "updated"]:
        typer.echo(f"Warning: Unknown sort order '{sort}', using relevance")
        sort = "relevance"

    if output_format not in ["table", "json", "bibtex"]:
        typer.echo(f"Warning: Unknown format '{output_format}', using table")
        output_format = "table"

    cmd = ArxivSearchCommand()
    exit_code = cmd.execute(
        query,
        max_results,
        sort,
        download,
        output_dir,
        output_format,
        output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def download_command(
    arxiv_id: str = typer.Argument(..., help="arXiv paper ID (e.g., 2301.12345)"),
    output_dir: Path = typer.Option(
        Path("./arxiv_downloads"),
        "--output",
        "-o",
        help="Output directory for PDF",
    ),
    metadata: bool = typer.Option(
        False,
        "--metadata",
        "-m",
        help="Show paper metadata",
    ),
) -> None:
    """Download a specific arXiv paper by ID.

    Downloads the PDF for a specific arXiv paper ID.

    Examples:
        # Download a paper
        ingestforge discovery arxiv-download 2301.12345

        # Download with metadata
        ingestforge discovery arxiv-download 2301.12345 -m

        # Download to specific directory
        ingestforge discovery arxiv-download 2301.12345 -o ./papers

    Note:
        - Supports old and new arXiv ID formats
        - Respects rate limits
    """
    cmd = ArxivDownloadCommand()
    exit_code = cmd.execute(arxiv_id, output_dir, metadata)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)
