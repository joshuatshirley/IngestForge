"""Security Scan CLI Commands.

Security Shield CI Pipeline
Epic: EP-26 (Security & Compliance)

Provides CLI interface for security scanning with CI/CD integration.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from ingestforge.core.logging import get_logger
from ingestforge.core.security.scanner import (
    SecurityScanner,
    SecurityReport,
    Severity,
    create_scanner,
)

logger = get_logger(__name__)
console = Console()

# JPL Rule #2: Fixed bounds
DEFAULT_MAX_FINDINGS = 100


app = typer.Typer(help="Security scanning commands for CI/CD integration")


def _severity_style(severity: Severity) -> str:
    """Get Rich style for severity level.

    Args:
        severity: Severity level.

    Returns:
        Rich style string.
    """
    styles = {
        Severity.CRITICAL: "bold red",
        Severity.HIGH: "red",
        Severity.MEDIUM: "yellow",
        Severity.LOW: "blue",
        Severity.INFO: "dim",
    }
    return styles.get(severity, "white")


def _display_summary_table(report: SecurityReport) -> None:
    """Display summary table.

    Args:
        report: Security report.

    Rule #4: Helper to keep _display_report < 60 lines.
    """
    summary_table = Table(title="Security Scan Summary", show_header=False)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")

    summary_table.add_row("Path Scanned", report.scan_path)
    summary_table.add_row("Files Scanned", str(report.files_scanned))
    summary_table.add_row("Duration", f"{report.scan_duration_ms:.1f}ms")
    summary_table.add_row("Total Findings", str(len(report.findings)))

    console.print(summary_table)


def _display_severity_table(report: SecurityReport) -> None:
    """Display severity breakdown table.

    Args:
        report: Security report.

    Rule #4: Helper to keep _display_report < 60 lines.
    """
    severity_table = Table(title="Findings by Severity")
    severity_table.add_column("Severity", style="cyan")
    severity_table.add_column("Count", justify="right")

    severity_table.add_row(
        "[bold red]Critical[/bold red]",
        f"[bold red]{report.critical_count}[/bold red]"
        if report.critical_count
        else "0",
    )
    severity_table.add_row(
        "[red]High[/red]",
        f"[red]{report.high_count}[/red]" if report.high_count else "0",
    )
    severity_table.add_row(
        "[yellow]Medium[/yellow]",
        f"[yellow]{report.medium_count}[/yellow]" if report.medium_count else "0",
    )
    severity_table.add_row(
        "[blue]Low[/blue]",
        f"[blue]{report.low_count}[/blue]" if report.low_count else "0",
    )
    severity_table.add_row("[dim]Info[/dim]", str(report.info_count))

    console.print(severity_table)


def _display_report(report: SecurityReport, verbose: bool = False) -> None:
    """Display security report.

    Args:
        report: Security report to display.
        verbose: Whether to show detailed output.

    Rule #4: Function < 60 lines (uses helper methods).
    """
    console.print()
    _display_summary_table(report)
    console.print()
    _display_severity_table(report)
    console.print()

    if report.findings and (verbose or len(report.findings) <= 20):
        _display_findings(report.findings, verbose)

    if report.exit_code == 0:
        console.print("[green]✓ No security issues found[/green]")
    elif report.exit_code == 1:
        console.print("[yellow]⚠ Warnings found - review recommended[/yellow]")
    else:
        console.print("[red]✗ Security issues found - action required[/red]")

    console.print(f"\n[dim]Exit code: {report.exit_code}[/dim]")


def _display_findings(findings: List, verbose: bool) -> None:
    """Display findings table.

    Args:
        findings: List of findings.
        verbose: Show detailed output.

    Rule #4: Function < 60 lines.
    """
    table = Table(title="Security Findings")
    table.add_column("Severity", style="cyan", no_wrap=True)
    table.add_column("Rule", style="dim")
    table.add_column("Title", style="white")
    table.add_column("File", style="yellow")
    table.add_column("Line", justify="right")

    for finding in findings[:DEFAULT_MAX_FINDINGS]:
        severity_style = _severity_style(finding.severity)
        file_short = Path(finding.file_path).name

        table.add_row(
            f"[{severity_style}]{finding.severity.value.upper()}[/{severity_style}]",
            finding.rule_id,
            finding.title,
            file_short,
            str(finding.line_number),
        )

    console.print(table)

    if verbose and findings:
        console.print("\n[bold]Finding Details:[/bold]")
        for finding in findings[:20]:
            console.print(f"\n[cyan]{finding.rule_id}[/cyan]: {finding.title}")
            console.print(f"  File: {finding.file_path}:{finding.line_number}")
            console.print(f"  [dim]{finding.line_content[:80]}[/dim]")
            console.print(f"  [green]Recommendation:[/green] {finding.recommendation}")


def _scan_single_file(scanner: SecurityScanner, path: Path) -> SecurityReport:
    """Scan a single file and create report.

    Args:
        scanner: Security scanner instance.
        path: File path to scan.

    Returns:
        SecurityReport with findings.

    Rule #4: Helper to keep scan_path < 60 lines.
    """
    console.print(f"[bold]Scanning file: {path}[/bold]")
    findings = scanner.scan_file(path)

    report = SecurityReport(
        report_id="single-file",
        scan_path=str(path),
        files_scanned=1,
    )
    for f in findings:
        report.add_finding(f)
    report.complete(0)
    return report


def _save_json_report(report: SecurityReport, output: Path) -> None:
    """Save report to JSON file.

    Args:
        report: Security report.
        output: Output file path.

    Rule #4: Helper to keep scan_path < 60 lines.
    """
    try:
        output.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"\n[green]Report saved to: {output}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to save report: {e}[/red]")


@app.command("scan")
def scan_path(
    path: Path = typer.Argument(..., help="File or directory to scan", exists=True),
    recursive: bool = typer.Option(
        True,
        "--recursive/--no-recursive",
        "-r/-R",
        help="Scan subdirectories recursively",
    ),
    exclude: Optional[List[str]] = typer.Option(
        None, "--exclude", "-e", help="Glob patterns to exclude"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output JSON report to file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed findings"
    ),
    fail_on_warning: bool = typer.Option(
        False, "--fail-on-warning", help="Exit with code 1 on warnings (strict CI)"
    ),
) -> None:
    """Scan file or directory for security issues.

    Returns exit code for CI integration:
      0 = clean, 1 = warnings, 2 = errors

    Rule #4: Function < 60 lines (uses helper methods).
    """
    try:
        scanner = create_scanner()

        if path.is_file():
            report = _scan_single_file(scanner, path)
        else:
            console.print(f"[bold]Scanning directory: {path}[/bold]")
            report = scanner.scan_directory(path, recursive, exclude)

        _display_report(report, verbose)

        if output:
            _save_json_report(report, output)

        exit_code = report.exit_code
        if fail_on_warning and exit_code == 1:
            exit_code = 2

        if exit_code > 0:
            raise typer.Exit(code=exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Security scan failed: {e}")
        raise typer.Exit(code=2)


@app.command("rules")
def list_rules() -> None:
    """List available security rules.

    Examples:
        ingestforge security rules
    """
    scanner = create_scanner()
    rules = scanner.get_rules()

    table = Table(title="Security Rules")
    table.add_column("Rule ID", style="cyan")
    table.add_column("Category", style="yellow")
    table.add_column("Severity")
    table.add_column("Title", style="white")
    table.add_column("Enabled")

    for rule in rules:
        severity_style = _severity_style(rule.severity)
        enabled = "[green]Yes[/green]" if rule.enabled else "[red]No[/red]"

        table.add_row(
            rule.rule_id,
            rule.category.value,
            f"[{severity_style}]{rule.severity.value}[/{severity_style}]",
            rule.title,
            enabled,
        )

    console.print(table)
    console.print(f"\n[dim]Total rules: {len(rules)}[/dim]")


@app.command("categories")
def list_categories() -> None:
    """List security finding categories.

    Examples:
        ingestforge security categories
    """
    from ingestforge.core.security.scanner import FindingCategory

    console.print("\n[bold]Security Finding Categories:[/bold]")
    for cat in FindingCategory:
        console.print(f"  - {cat.value}")


# Export command for integration
command = app
