"""CrossRef search command - DOI lookup and publication search.

This command provides access to the CrossRef API for DOI-based
publication lookup and full-text search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.discovery.base import DiscoveryCommand
from ingestforge.discovery.crossref import (
    CrossRefClient,
    Publication,
    PublicationType,
    export_bibtex,
)


class CrossRefLookupCommand(DiscoveryCommand):
    """Lookup publication by DOI."""

    def __init__(self, mailto: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = CrossRefClient(mailto=mailto)

    def execute(
        self,
        doi: str,
        output_format: str = "detailed",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute DOI lookup.

        Rule #4: Function <60 lines.
        """
        try:
            self.print_info(f"Looking up DOI: {doi}")
            publication = self._client.lookup_doi(doi)

            if not publication:
                self.print_error(f"DOI not found: {doi}")
                return 1

            self._output_result(publication, output_format, output_file)
            return 0

        except Exception as e:
            return self.handle_error(e, "DOI lookup failed")

    def _output_result(
        self,
        pub: Publication,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output publication in requested format.

        Rule #1: Early return for each format.
        """
        if output_format == "bibtex":
            bibtex = pub.to_bibtex()
            self._write_or_print(bibtex, output_file)
            return

        if output_format == "json":
            json_data = self._publication_to_json(pub)
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Detailed format
        self._display_detailed(pub)
        if output_file:
            json_data = self._publication_to_json(pub)
            output_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")

    def _display_detailed(self, pub: Publication) -> None:
        """Display detailed publication info.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{pub.title}[/bold]",
                title=f"DOI: {pub.doi}",
                border_style="cyan",
            )
        )

        # Authors
        authors = ", ".join(a.full_name for a in pub.authors)
        self.console.print(f"[bold]Authors:[/] {authors}")

        # Publication info
        if pub.container_title:
            self.console.print(f"[bold]Journal:[/] {pub.container_title}")
        if pub.publisher:
            self.console.print(f"[bold]Publisher:[/] {pub.publisher}")
        if pub.year:
            self.console.print(f"[bold]Year:[/] {pub.year}")
        if pub.volume or pub.issue or pub.pages:
            vol_info = []
            if pub.volume:
                vol_info.append(f"Vol. {pub.volume}")
            if pub.issue:
                vol_info.append(f"Issue {pub.issue}")
            if pub.pages:
                vol_info.append(f"pp. {pub.pages}")
            self.console.print(f"[bold]Volume:[/] {', '.join(vol_info)}")

        # Metrics
        self.console.print(f"[bold]Type:[/] {pub.publication_type}")
        self.console.print(f"[bold]References:[/] {pub.reference_count}")
        self.console.print(f"[bold]Cited by:[/] {pub.is_referenced_by_count:,}")

        # Abstract
        if pub.abstract:
            self.console.print()
            self.console.print("[bold]Abstract:[/]")
            abstract = (
                pub.abstract[:500] + "..." if len(pub.abstract) > 500 else pub.abstract
            )
            self.console.print(abstract)

        # URL
        self.console.print()
        self.console.print(f"[link={pub.url}]{pub.url}[/link]")
        self.console.print()

    def _publication_to_json(self, pub: Publication) -> dict:
        """Convert publication to JSON-serializable dict."""
        return {
            "doi": pub.doi,
            "title": pub.title,
            "authors": [
                {"given": a.given, "family": a.family, "orcid": a.orcid}
                for a in pub.authors
            ],
            "container_title": pub.container_title,
            "publisher": pub.publisher,
            "year": pub.year,
            "publication_date": pub.published_date.isoformat()
            if pub.published_date
            else None,
            "publication_type": pub.publication_type,
            "url": pub.url,
            "abstract": pub.abstract,
            "volume": pub.volume,
            "issue": pub.issue,
            "pages": pub.pages,
            "reference_count": pub.reference_count,
            "cited_by_count": pub.is_referenced_by_count,
            "subject": pub.subject,
            "issn": pub.issn,
            "isbn": pub.isbn,
            "license": pub.license_url,
        }

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


class CrossRefSearchCommand(DiscoveryCommand):
    """Search CrossRef for publications."""

    def __init__(self, mailto: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = CrossRefClient(mailto=mailto)

    def execute(
        self,
        query: str,
        limit: int = 20,
        pub_type: Optional[str] = None,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        sort: str = "relevance",
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute CrossRef search.

        Rule #4: Function <60 lines.
        """
        try:
            # Parse publication type
            publication_type = self._parse_pub_type(pub_type)

            self.print_info(f"Searching CrossRef for: {query}")
            publications = self._client.search(
                query,
                limit=limit,
                publication_type=publication_type,
                from_year=from_year,
                to_year=to_year,
                sort=sort,
            )

            if not publications:
                self.print_warning("No results found")
                return 0

            self._output_results(publications, query, output_format, output_file)
            return 0

        except Exception as e:
            return self.handle_error(e, "CrossRef search failed")

    def _parse_pub_type(self, pub_type: Optional[str]) -> Optional[PublicationType]:
        """Parse publication type string to enum."""
        if not pub_type:
            return None

        type_map = {
            "article": PublicationType.JOURNAL_ARTICLE,
            "journal-article": PublicationType.JOURNAL_ARTICLE,
            "proceedings": PublicationType.PROCEEDINGS_ARTICLE,
            "proceedings-article": PublicationType.PROCEEDINGS_ARTICLE,
            "chapter": PublicationType.BOOK_CHAPTER,
            "book-chapter": PublicationType.BOOK_CHAPTER,
            "book": PublicationType.BOOK,
            "preprint": PublicationType.POSTED_CONTENT,
            "posted-content": PublicationType.POSTED_CONTENT,
            "dissertation": PublicationType.DISSERTATION,
            "report": PublicationType.REPORT,
            "dataset": PublicationType.DATASET,
        }
        return type_map.get(pub_type.lower())

    def _output_results(
        self,
        publications: List[Publication],
        query: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output results in requested format."""
        if output_format == "bibtex":
            bibtex = export_bibtex(publications)
            self._write_or_print(bibtex, output_file)
            return

        if output_format == "json":
            json_data = {
                "query": query,
                "count": len(publications),
                "publications": [self._pub_to_json(p) for p in publications],
            }
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Table format
        self._display_table(publications, query)
        if output_file:
            json_data = {
                "query": query,
                "count": len(publications),
                "publications": [self._pub_to_json(p) for p in publications],
            }
            output_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")

    def _display_table(self, publications: List[Publication], query: str) -> None:
        """Display results as table.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]CrossRef Search: {query}[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

        table = Table(title=f"Found {len(publications)} publications", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Title", width=40)
        table.add_column("Authors", width=20)
        table.add_column("Year", width=6)
        table.add_column("Type", width=12)
        table.add_column("Cited", width=8, justify="right")

        for i, pub in enumerate(publications, 1):
            authors_str = self._format_authors(pub.authors)
            table.add_row(
                str(i),
                pub.title[:80] + ("..." if len(pub.title) > 80 else ""),
                authors_str,
                str(pub.year) if pub.year else "N/A",
                pub.publication_type[:15],
                f"{pub.is_referenced_by_count:,}",
            )

        self.console.print(table)
        self.console.print()
        self.print_info("Use --format bibtex for BibTeX export")

    def _format_authors(self, authors: list, max_names: int = 2) -> str:
        """Format author list with truncation."""
        if not authors:
            return "Unknown"
        names = [a.full_name for a in authors[:max_names]]
        result = ", ".join(names)
        if len(authors) > max_names:
            result += " et al."
        return result

    def _pub_to_json(self, pub: Publication) -> dict:
        """Convert publication to JSON dict."""
        return {
            "doi": pub.doi,
            "title": pub.title,
            "year": pub.year,
            "type": pub.publication_type,
            "cited_by": pub.is_referenced_by_count,
        }

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


# Typer command wrappers


def lookup_command(
    doi: str = typer.Argument(..., help="DOI to lookup (e.g., 10.1038/nature12373)"),
    output_format: str = typer.Option(
        "detailed",
        "--format",
        "-f",
        help="Output format: detailed, json, bibtex",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save output to file",
    ),
    mailto: Optional[str] = typer.Option(
        None,
        "--mailto",
        envvar="CROSSREF_MAILTO",
        help="Email for CrossRef polite pool (recommended)",
    ),
) -> None:
    """Lookup publication by DOI.

    Retrieves complete metadata for a publication using its DOI.

    DOI formats:
        - 10.1038/nature12373
        - https://doi.org/10.1038/nature12373
        - doi:10.1038/nature12373

    Examples:
        # Lookup a DOI
        ingestforge discovery crossref 10.1038/nature12373

        # Export as BibTeX
        ingestforge discovery crossref 10.1038/nature12373 -f bibtex

        # Save to file
        ingestforge discovery crossref 10.1038/nature12373 -f json -o paper.json
    """
    cmd = CrossRefLookupCommand(mailto=mailto)
    exit_code = cmd.execute(doi, output_format, output_file)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def search_command(
    query: str = typer.Argument(..., help="Search query"),
    limit: int = typer.Option(
        20,
        "--limit",
        "-n",
        help="Maximum results (1-100)",
    ),
    pub_type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Publication type: article, proceedings, chapter, book, preprint",
    ),
    from_year: Optional[int] = typer.Option(
        None,
        "--from-year",
        help="Filter publications from this year",
    ),
    to_year: Optional[int] = typer.Option(
        None,
        "--to-year",
        help="Filter publications up to this year",
    ),
    sort: str = typer.Option(
        "relevance",
        "--sort",
        "-s",
        help="Sort by: relevance, published, cited",
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
    mailto: Optional[str] = typer.Option(
        None,
        "--mailto",
        envvar="CROSSREF_MAILTO",
        help="Email for CrossRef polite pool",
    ),
) -> None:
    """Search CrossRef for publications.

    Full-text search across the CrossRef database of scholarly publications.

    Examples:
        # Search for publications
        ingestforge discovery crossref-search "neural networks"

        # Filter by type and year
        ingestforge discovery crossref-search "climate" -t article --from-year 2020

        # Sort by citations
        ingestforge discovery crossref-search "machine learning" -s cited

        # Export as BibTeX
        ingestforge discovery crossref-search "transformers" -f bibtex -o refs.bib
    """
    if limit > 100:
        typer.echo("Warning: Limiting results to 100")
        limit = 100

    cmd = CrossRefSearchCommand(mailto=mailto)
    exit_code = cmd.execute(
        query,
        limit,
        pub_type,
        from_year,
        to_year,
        sort,
        output_format,
        output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)
