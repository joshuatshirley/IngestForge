"""CVE Discovery CLI - Search vulnerabilities using NVD API.

This command uses the NVDDiscovery wrapper (CYBER-002) to search
the NIST National Vulnerability Database for CVE information."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.discovery.base import DiscoveryCommand


class CVEDiscoveryCommand(DiscoveryCommand):
    """Search CVE vulnerabilities using NVD API."""

    def __init__(self) -> None:
        """Initialize command."""
        super().__init__()
        self._discovery = None

    def _get_discovery(self):
        """Lazy-load NVDDiscovery to handle missing dependency.

        Rule #4: Function <60 lines.
        """
        if self._discovery is not None:
            return self._discovery

        try:
            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            # Check for API key in environment
            api_key = os.environ.get("NVD_API_KEY")
            self._discovery = NVDDiscovery(api_key=api_key)
            return self._discovery
        except ImportError as e:
            self.print_error(
                "requests library not installed. " "Install with: pip install requests"
            )
            raise typer.Exit(code=1) from e

    def execute(
        self,
        product: str,
        severity: Optional[str] = None,
        max_results: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Execute CVE discovery search.

        Rule #4: Function <60 lines.
        Rule #7: All parameters validated.
        """
        try:
            discovery = self._get_discovery()

            # Search NVD
            self.print_info(f"Searching NVD for: {product}")
            if severity:
                self.print_info(f"Filtering by severity: {severity}")

            cves = discovery.search(
                product=product,
                severity=severity,
                max_results=max_results,
                start_date=start_date,
                end_date=end_date,
            )

            if not cves:
                self.print_warning("No vulnerabilities found")
                return 0

            # Display/export results
            self._output_results(cves, product, output_format, output_file)

            return 0

        except typer.Exit:
            raise
        except Exception as e:
            return self.handle_error(e, "CVE discovery failed")

    def execute_get(
        self,
        cve_id: str,
        output_format: str = "table",
        output_file: Optional[Path] = None,
    ) -> int:
        """Get specific CVE by ID.

        Rule #4: Function <60 lines.
        """
        try:
            discovery = self._get_discovery()

            self.print_info(f"Fetching CVE: {cve_id}")
            cve = discovery.get_cve(cve_id)

            if not cve:
                self.print_warning(f"CVE not found: {cve_id}")
                return 1

            # Display result
            self._output_single_cve(cve, output_format, output_file)

            return 0

        except typer.Exit:
            raise
        except Exception as e:
            return self.handle_error(e, "CVE lookup failed")

    def _output_results(
        self,
        cves: List,
        query: str,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output results in requested format.

        Rule #1: Early return for each format.
        Rule #4: Function <60 lines.
        """
        if output_format == "json":
            json_data = self._cves_to_json(cves, query)
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Default: table format
        self._display_table(cves, query)
        if output_file:
            json_data = self._cves_to_json(cves, query)
            output_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")

    def _output_single_cve(
        self,
        cve,
        output_format: str,
        output_file: Optional[Path],
    ) -> None:
        """Output single CVE details.

        Rule #4: Function <60 lines.
        """
        if output_format == "json":
            json_data = cve.to_dict()
            self._write_or_print(json.dumps(json_data, indent=2), output_file)
            return

        # Default: detailed view
        self._display_cve_detail(cve)
        if output_file:
            json_data = cve.to_dict()
            output_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")

    def _display_table(self, cves: List, query: str) -> None:
        """Display results as rich table.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]CVE Search: {query}[/bold cyan]",
                border_style="cyan",
            )
        )
        self.console.print()

        table = Table(title=f"Found {len(cves)} vulnerabilities", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("CVE ID", width=16)
        table.add_column("Severity", width=10)
        table.add_column("Score", width=6)
        table.add_column("Description", width=50)
        table.add_column("Published", width=12)

        for i, cve in enumerate(cves, 1):
            severity_str = self._format_severity(cve.severity)
            score_str = f"{cve.cvss_score:.1f}" if cve.cvss_score else "N/A"
            desc = cve.description[:80] + ("..." if len(cve.description) > 80 else "")
            pub_date = (
                cve.published_date.strftime("%Y-%m-%d") if cve.published_date else "N/A"
            )

            table.add_row(
                str(i),
                cve.cve_id,
                severity_str,
                score_str,
                desc,
                pub_date,
            )

        self.console.print(table)
        self.console.print()
        self.print_info("Use --format json for full details, --output to save results")

    def _display_cve_detail(self, cve) -> None:
        """Display detailed CVE information.

        Rule #4: Function <60 lines.
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold red]{cve.cve_id}[/bold red]",
                border_style="red",
            )
        )
        self.console.print()

        # Basic info
        severity_str = self._format_severity(cve.severity)
        score_str = f"{cve.cvss_score:.1f}" if cve.cvss_score else "N/A"

        self.console.print(f"[bold]Severity:[/bold] {severity_str}")
        self.console.print(f"[bold]CVSS Score:[/bold] {score_str}")

        if cve.cvss_vector:
            self.console.print(f"[bold]CVSS Vector:[/bold] {cve.cvss_vector}")

        self.console.print()
        self.console.print("[bold]Description:[/bold]")
        self.console.print(cve.description)
        self.console.print()

        # Dates
        if cve.published_date:
            self.console.print(
                f"[bold]Published:[/bold] {cve.published_date.strftime('%Y-%m-%d')}"
            )
        if cve.last_modified:
            self.console.print(
                f"[bold]Last Modified:[/bold] {cve.last_modified.strftime('%Y-%m-%d')}"
            )

        # CWE IDs
        if cve.cwe_ids:
            self.console.print(f"[bold]CWE IDs:[/bold] {', '.join(cve.cwe_ids)}")

        # Affected products (limited)
        if cve.affected_products:
            self.console.print()
            self.console.print("[bold]Affected Products:[/bold]")
            for product in cve.affected_products[:5]:
                self.console.print(f"  - {product}")
            if len(cve.affected_products) > 5:
                self.console.print(f"  ... and {len(cve.affected_products) - 5} more")

        # References (limited)
        if cve.references:
            self.console.print()
            self.console.print("[bold]References:[/bold]")
            for ref in cve.references[:5]:
                self.console.print(f"  - {ref}")
            if len(cve.references) > 5:
                self.console.print(f"  ... and {len(cve.references) - 5} more")

        self.console.print()

    def _format_severity(self, severity: Optional[str]) -> str:
        """Format severity with color coding."""
        if not severity:
            return "[dim]Unknown[/dim]"

        color_map = {
            "CRITICAL": "bold red",
            "HIGH": "red",
            "MEDIUM": "yellow",
            "LOW": "green",
        }
        color = color_map.get(severity, "white")
        return f"[{color}]{severity}[/{color}]"

    def _cves_to_json(self, cves: List, query: str) -> dict:
        """Convert CVEs to JSON-serializable dict."""
        return {
            "query": query,
            "count": len(cves),
            "vulnerabilities": [cve.to_dict() for cve in cves],
        }

    def _write_or_print(self, content: str, output_file: Optional[Path]) -> None:
        """Write to file or print to console."""
        if output_file:
            output_file.write_text(content, encoding="utf-8")
            self.print_success(f"Saved to: {output_file}")
        else:
            self.console.print(content)


# Typer command wrappers


def command(
    product: str = typer.Argument(..., help="Software/product name to search for"),
    severity: Optional[str] = typer.Option(
        None,
        "--severity",
        "-s",
        help="Filter by severity: LOW, MEDIUM, HIGH, CRITICAL",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Maximum results (1-100, default 10)",
    ),
    start_date: Optional[str] = typer.Option(
        None,
        "--start-date",
        help="Filter CVEs published after this date (YYYY-MM-DD)",
    ),
    end_date: Optional[str] = typer.Option(
        None,
        "--end-date",
        help="Filter CVEs published before this date (YYYY-MM-DD)",
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
) -> None:
    """Search CVE vulnerabilities by software/product name (CYBER-002).

    Searches the NIST National Vulnerability Database (NVD) for
    vulnerabilities affecting the specified software or product.

    Examples:
        # Search for Apache Tomcat vulnerabilities
        ingestforge discover cve "apache tomcat"

        # Filter by severity
        ingestforge discover cve "log4j" --severity CRITICAL

        # Limit results
        ingestforge discover cve "openssl" -n 20

        # Filter by date range
        ingestforge discover cve "python" --start-date 2024-01-01

        # Export as JSON
        ingestforge discover cve "nginx" -f json -o vulnerabilities.json

    Note:
        - NVD API has rate limits (~10 requests/minute without API key)
        - Set NVD_API_KEY environment variable for higher limits
        - Get an API key at: https://nvd.nist.gov/developers/request-an-api-key
    """
    # Validate and clamp limit
    if limit > 100:
        typer.echo("Warning: Limiting results to 100")
        limit = 100
    if limit < 1:
        limit = 1

    # Validate severity
    valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    if severity and severity.upper() not in valid_severities:
        typer.echo(
            f"Warning: Invalid severity '{severity}'. Valid: {', '.join(valid_severities)}"
        )
        severity = None
    elif severity:
        severity = severity.upper()

    # Validate output format
    if output_format not in ["table", "json"]:
        typer.echo(f"Warning: Unknown format '{output_format}', using table")
        output_format = "table"

    # Parse dates
    parsed_start_date = _parse_date_option(start_date, "start-date")
    parsed_end_date = _parse_date_option(end_date, "end-date")

    cmd = CVEDiscoveryCommand()
    exit_code = cmd.execute(
        product=product,
        severity=severity,
        max_results=limit,
        start_date=parsed_start_date,
        end_date=parsed_end_date,
        output_format=output_format,
        output_file=output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def get_command(
    cve_id: str = typer.Argument(..., help="CVE identifier (e.g., CVE-2024-12345)"),
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
) -> None:
    """Get details for a specific CVE by ID.

    Fetches detailed information about a specific CVE from
    the NIST National Vulnerability Database.

    Examples:
        # Get CVE details
        ingestforge discover cve-get CVE-2024-12345

        # Export as JSON
        ingestforge discover cve-get CVE-2021-44228 -f json -o log4shell.json
    """
    # Validate output format
    if output_format not in ["table", "json"]:
        typer.echo(f"Warning: Unknown format '{output_format}', using table")
        output_format = "table"

    cmd = CVEDiscoveryCommand()
    exit_code = cmd.execute_get(
        cve_id=cve_id,
        output_format=output_format,
        output_file=output_file,
    )

    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def _parse_date_option(date_str: Optional[str], param_name: str) -> Optional[datetime]:
    """Parse date string from command line option.

    Rule #4: Small helper function.
    """
    if not date_str:
        return None

    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        typer.echo(
            f"Warning: Invalid date format for --{param_name}: {date_str}. Use YYYY-MM-DD."
        )
        return None
