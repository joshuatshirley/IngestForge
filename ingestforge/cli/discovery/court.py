"""Court discovery command - CourtListener case law search.

This command provides access to the CourtListener API for court
opinion search and case law discovery."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.discovery.base import DiscoveryCommand
from ingestforge.discovery.courtlistener_wrapper import (
    CourtCase,
    CourtListenerDiscovery,
    FEDERAL_COURTS,
)


class CourtSearchCommand(DiscoveryCommand):
    """Search CourtListener for court opinions."""

    def __init__(self, api_token: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = CourtListenerDiscovery(api_token=api_token)

    def execute(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        limit: int = 5,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute court opinion search.

        Rule #4: Function <60 lines.
        """
        try:
            self.print_info(f"Searching CourtListener for: {query}")
            cases = self._client.search(
                query,
                jurisdiction=jurisdiction,
                max_results=limit,
                from_date=from_date,
                to_date=to_date,
            )

            if not cases:
                self.print_warning("No results found")
                return 0

            self._output_results(cases, query, output_format, output_file)
            return 0

        except Exception as e:
            return self.handle_error(e, "Court search failed")

    def _output_results(
        self,
        cases: List[CourtCase],
        query: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output results in requested format.

        Rule #1: Early return for each format.
        """
        if output_format == "json":
            json_data = {
                "query": query,
                "count": len(cases),
                "cases": [c.to_dict() for c in cases],
            }
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Table format
        self._display_table(cases, query)
        if output_file:
            json_data = {
                "query": query,
                "count": len(cases),
                "cases": [c.to_dict() for c in cases],
            }
            output_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")

    def _display_table(self, cases: List[CourtCase], query: str) -> None:
        """Display results as table.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Court Search: {query}[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

        table = Table(title=f"Found {len(cases)} cases", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Case Name", width=35)
        table.add_column("Court", width=15)
        table.add_column("Year", width=6)
        table.add_column("Docket", width=15)
        table.add_column("Status", width=12)

        for i, case in enumerate(cases, 1):
            case_name = case.case_name
            if len(case_name) > 35:
                case_name = case_name[:32] + "..."

            table.add_row(
                str(i),
                case_name,
                case.court[:15] if case.court else "N/A",
                str(case.year) if case.year else "N/A",
                case.docket_number[:15] if case.docket_number else "N/A",
                case.precedential_status[:12] if case.precedential_status else "N/A",
            )

        self.console.print(table)
        self.console.print()
        self.print_info("Use --format json for detailed output")

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


class CourtDetailCommand(DiscoveryCommand):
    """Get detailed case information by cluster ID."""

    def __init__(self, api_token: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = CourtListenerDiscovery(api_token=api_token)

    def execute(
        self,
        cluster_id: str,
        output_format: str = "detailed",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute case detail lookup.

        Rule #4: Function <60 lines.
        """
        try:
            self.print_info(f"Looking up case: {cluster_id}")
            case = self._client.get_case(cluster_id)

            if not case:
                self.print_error(f"Case not found: {cluster_id}")
                return 1

            self._output_result(case, output_format, output_file)
            return 0

        except Exception as e:
            return self.handle_error(e, "Case lookup failed")

    def _output_result(
        self,
        case: CourtCase,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output case in requested format."""
        if output_format == "json":
            self._write_or_print(json.dumps(case.to_dict(), indent=2), output_file)
            return

        # Detailed format
        self._display_detailed(case)
        if output_file:
            output_file.write_text(
                json.dumps(case.to_dict(), indent=2), encoding="utf-8"
            )
            self.print_success(f"Saved to: {output_file}")

    def _display_detailed(self, case: CourtCase) -> None:
        """Display detailed case info.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]{case.case_name}[/bold]",
                title=f"Docket: {case.docket_number}",
                border_style="cyan",
            )
        )

        self.console.print(f"[bold]Court:[/] {case.court}")
        self.console.print(f"[bold]Jurisdiction:[/] {case.jurisdiction}")

        if case.judges:
            self.console.print(f"[bold]Judges:[/] {', '.join(case.judges)}")

        if case.date_filed:
            self.console.print(
                f"[bold]Date Filed:[/] {case.date_filed.strftime('%Y-%m-%d')}"
            )

        if case.date_decided:
            self.console.print(
                f"[bold]Date Decided:[/] {case.date_decided.strftime('%Y-%m-%d')}"
            )

        if case.citations:
            self.console.print(f"[bold]Citations:[/] {', '.join(case.citations)}")

        self.console.print(f"[bold]Status:[/] {case.precedential_status}")

        if case.opinion_url:
            self.console.print()
            self.console.print(f"[link={case.opinion_url}]{case.opinion_url}[/link]")

        self.console.print()

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


class CourtDownloadCommand(DiscoveryCommand):
    """Download opinion text for a case."""

    def __init__(self, api_token: Optional[str] = None) -> None:
        """Initialize command."""
        super().__init__()
        self._client = CourtListenerDiscovery(api_token=api_token)

    def execute(
        self,
        cluster_id: str,
        output_dir: Path,
    ) -> int:
        """Execute opinion download.

        Rule #4: Function <60 lines.
        """
        try:
            self.print_info(f"Fetching case: {cluster_id}")
            case = self._client.get_case(cluster_id)

            if not case:
                self.print_error(f"Case not found: {cluster_id}")
                return 1

            self.print_info(f"Downloading opinion for: {case.case_name}")
            result = self._client.download_opinion(case, output_dir)

            if result:
                self.print_success(f"Downloaded to: {result}")
                return 0
            else:
                self.print_error("Failed to download opinion")
                return 1

        except Exception as e:
            return self.handle_error(e, "Download failed")


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse date string to datetime."""
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return None


# Typer command wrappers


def search_command(
    query: str = typer.Argument(..., help="Search query for court opinions"),
    jurisdiction: Optional[str] = typer.Option(
        None,
        "--jurisdiction",
        "-j",
        help="Court jurisdiction filter (e.g., ca9, scotus, calctapp)",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Maximum results (1-100)",
    ),
    from_date: Optional[str] = typer.Option(
        None,
        "--from-date",
        help="Filter opinions from date (YYYY-MM-DD)",
    ),
    to_date: Optional[str] = typer.Option(
        None,
        "--to-date",
        help="Filter opinions until date (YYYY-MM-DD)",
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
        "-o",
        help="Save output to file",
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--token",
        envvar="COURTLISTENER_TOKEN",
        help="CourtListener API token (optional)",
    ),
) -> None:
    """Search CourtListener for court opinions.

    Searches the Free Law Project's CourtListener database for court
    opinions matching your query.

    Supported jurisdictions:
        - Federal: scotus, ca1-ca11, cadc, cafc
        - State courts: See CourtListener documentation

    Examples:
        # Search for qualified immunity cases
        ingestforge discovery court "qualified immunity"

        # Search 9th Circuit cases
        ingestforge discovery court "fourth amendment" -j ca9

        # Search with date range
        ingestforge discovery court "section 1983" --from-date 2020-01-01

        # Export as JSON
        ingestforge discovery court "miranda" -f json -o results.json
    """
    if limit > 100:
        typer.echo("Warning: Limiting results to 100")
        limit = 100

    cmd = CourtSearchCommand(api_token=api_token)
    exit_code = cmd.execute(
        query,
        jurisdiction=jurisdiction,
        limit=limit,
        from_date=_parse_date(from_date),
        to_date=_parse_date(to_date),
        output_format=output_format,
        output_file=output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def detail_command(
    cluster_id: str = typer.Argument(..., help="CourtListener cluster ID"),
    output_format: str = typer.Option(
        "detailed",
        "--format",
        "-f",
        help="Output format: detailed, json",
    ),
    output_file: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save output to file",
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--token",
        envvar="COURTLISTENER_TOKEN",
        help="CourtListener API token",
    ),
) -> None:
    """Get detailed case information by cluster ID.

    Retrieves complete metadata for a case from CourtListener.

    Examples:
        # Get case details
        ingestforge discovery court-detail 12345678

        # Export as JSON
        ingestforge discovery court-detail 12345678 -f json -o case.json
    """
    cmd = CourtDetailCommand(api_token=api_token)
    exit_code = cmd.execute(cluster_id, output_format, output_file)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def download_command(
    cluster_id: str = typer.Argument(..., help="CourtListener cluster ID"),
    output_dir: Path = typer.Option(
        Path("./opinions"),
        "--output-dir",
        "-o",
        help="Directory to save the opinion",
    ),
    api_token: Optional[str] = typer.Option(
        None,
        "--token",
        envvar="COURTLISTENER_TOKEN",
        help="CourtListener API token",
    ),
) -> None:
    """Download opinion text for a case.

    Downloads the full opinion text for a case by its cluster ID.

    Examples:
        # Download an opinion
        ingestforge discovery court-download 12345678

        # Download to specific directory
        ingestforge discovery court-download 12345678 -o ./legal/opinions
    """
    cmd = CourtDownloadCommand(api_token=api_token)
    exit_code = cmd.execute(cluster_id, output_dir)

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def list_courts_command() -> None:
    """List supported federal court jurisdiction codes.

    Displays the available federal court codes that can be used
    with the --jurisdiction flag.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    console.print()
    console.print(
        Panel(
            "[bold cyan]Federal Court Codes[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    table = Table(show_lines=False)
    table.add_column("Code", style="cyan")
    table.add_column("Court Name")

    for code, name in sorted(FEDERAL_COURTS.items()):
        table.add_row(code, name)

    console.print(table)
    console.print()
    console.print(
        "[dim]For state courts, see: https://www.courtlistener.com/api/jurisdictions/[/dim]"
    )
    console.print()
