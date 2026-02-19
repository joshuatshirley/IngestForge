"""arXiv Discovery CLI - Search and download papers using arxiv library.

This command uses the ArxivDiscovery wrapper (RES-002) which leverages
the official `arxiv` Python library for cleaner API access."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.discovery.base import DiscoveryCommand


class ArxivDiscoveryCommand(DiscoveryCommand):
    """Search arXiv using the arxiv library wrapper."""

    def __init__(self) -> None:
        """Initialize command."""
        super().__init__()
        self._discovery = None

    def _get_discovery(self):
        """Lazy-load ArxivDiscovery to handle missing dependency.

        Rule #4: Function <60 lines.
        """
        if self._discovery is not None:
            return self._discovery

        try:
            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            self._discovery = ArxivDiscovery()
            return self._discovery
        except ImportError as e:
            self.print_error(
                "arxiv library not installed. "
                "Install with: pip install ingestforge[research]"
            )
            raise typer.Exit(code=1) from e

    def execute(
        self,
        query: str,
        max_results: int = 5,
        download: bool = False,
        output_dir: Optional[Path] = None,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute arXiv discovery search.

        Rule #4: Function <60 lines.
        Rule #7: All parameters validated.
        """
        try:
            discovery = self._get_discovery()

            if download and not output_dir:
                output_dir = Path("./arxiv_downloads")

            # Search arXiv
            self.print_info(f"Searching arXiv for: {query}")
            papers = discovery.search(query, max_results=max_results)

            if not papers:
                self.print_warning("No results found")
                return 0

            # Display/export results
            self._output_results(papers, query, output_format, output_file)

            # Download PDFs if requested
            if download and output_dir:
                self._download_papers(papers, output_dir, discovery)

            return 0

        except typer.Exit:
            raise
        except Exception as e:
            return self.handle_error(e, "arXiv discovery failed")

    def _output_results(
        self,
        papers: List,
        query: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output results in requested format.

        Rule #1: Early return for each format.
        Rule #4: Function <60 lines.
        """
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

    def _display_table(self, papers: List, query: str) -> None:
        """Display results as rich table.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]arXiv Discovery: {query}[/bold cyan]",
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
            title = paper.title[:80] + ("..." if len(paper.title) > 80 else "")
            year = str(paper.published_date.year)
            category = paper.primary_category or "N/A"

            table.add_row(
                str(i),
                paper.arxiv_id,
                title,
                authors_str,
                year,
                category,
            )

        self.console.print(table)
        self.console.print()
        self.print_info("Use --download to fetch PDFs, --format json for JSON output")

    def _format_authors(self, authors: List[str], max_names: int = 2) -> str:
        """Format author list with truncation."""
        if not authors:
            return "Unknown"
        result = ", ".join(authors[:max_names])
        if len(authors) > max_names:
            result += " et al."
        return result

    def _papers_to_json(self, papers: List, query: str) -> dict:
        """Convert papers to JSON-serializable dict."""
        return {
            "query": query,
            "count": len(papers),
            "papers": [paper.to_dict() for paper in papers],
        }

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)

    def _download_papers(self, papers: List, output_dir: Path, discovery) -> None:
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
                result = discovery.download_pdf(paper, output_dir)

                if result:
                    downloaded += 1
                else:
                    failed += 1

                progress.advance(task)

        self.console.print()
        self.print_success(f"Downloaded {downloaded} papers")
        if failed > 0:
            self.print_warning(f"Failed: {failed} papers")


# Typer command wrapper


def command(
    query: str = typer.Argument(..., help="Search query (keywords, title, authors)"),
    max_results: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Maximum results (1-300, default 5)",
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
        help="Output format: table, json",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-O",
        help="Save output to file",
    ),
) -> None:
    """Search arXiv using the arxiv library (RES-002).

    This command uses the ArxivDiscovery wrapper which leverages
    the official arxiv Python library for improved reliability.

    Examples:
        # Search for papers
        ingestforge discover arxiv-lib "transformer architecture"

        # Limit results
        ingestforge discover arxiv-lib "quantum computing" -n 10

        # Export as JSON
        ingestforge discover arxiv-lib "machine learning" -f json -O results.json

        # Download PDFs
        ingestforge discover arxiv-lib "climate change" -d -o ./papers

    Requirements:
        Install with: pip install ingestforge[research]
    """
    if max_results > 300:
        typer.echo("Warning: Limiting results to 300 (max allowed)")
        max_results = 300

    if max_results < 1:
        typer.echo("Warning: Setting minimum results to 1")
        max_results = 1

    if output_format not in ["table", "json"]:
        typer.echo(f"Warning: Unknown format '{output_format}', using table")
        output_format = "table"

    cmd = ArxivDiscoveryCommand()
    exit_code = cmd.execute(
        query,
        max_results,
        download,
        output_dir,
        output_format,
        output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)
