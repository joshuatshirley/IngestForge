"""Audit CLI Commands.

Append-Only Audit Log
LLM Model Parity Auditor
BUG002: Dependency Integrity Audit
Epic: EP-19 (Governance & Compliance), EP-26 (Security & Compliance)

Provides CLI interface for viewing and managing the audit log,
auditing LLM provider configurations, and dependency integrity.

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
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ingestforge.core.audit import (
    AuditOperation,
    AppendOnlyAuditLog,
    create_audit_log,
    MAX_QUERY_RESULTS,
    # Model Auditor ()
    ModelCapability,
    AuditStatus,
    create_model_auditor,
    # Dependency Auditor (BUG002)
    create_dependency_auditor,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
console = Console()

# JPL Rule #2: Fixed bounds
DEFAULT_LIST_LIMIT = 50
MAX_LIST_LIMIT = MAX_QUERY_RESULTS


app = typer.Typer(help="Audit log management commands")


def _get_default_log_path() -> Path:
    """Get default audit log path.

    Returns:
        Path to default audit log file.
    """
    return Path.home() / ".ingestforge" / "audit.jsonl"


def _get_audit_log(log_path: Optional[Path] = None) -> AppendOnlyAuditLog:
    """Get audit log instance.

    Args:
        log_path: Optional custom path.

    Returns:
        AppendOnlyAuditLog instance.
    """
    path = log_path or _get_default_log_path()
    return create_audit_log(log_path=path)


def _resolve_operation(op_str: str) -> Optional[AuditOperation]:
    """Resolve operation string to enum.

    Args:
        op_str: Operation string.

    Returns:
        AuditOperation or None if not found.
    """
    try:
        return AuditOperation(op_str)
    except ValueError:
        return None


@app.command("log")
def log_entry(
    operation: str = typer.Argument(
        ..., help="Operation type (e.g., 'custom', 'create')"
    ),
    message: str = typer.Option("", "--message", "-m", help="Log message"),
    source: str = typer.Option("cli", "--source", "-s", help="Operation source"),
    target: str = typer.Option("", "--target", "-t", help="Operation target"),
    log_path: Optional[Path] = typer.Option(
        None, "--log-path", "-l", help="Path to audit log file"
    ),
) -> None:
    """Log a custom audit entry.

    Examples:
        ingestforge audit log custom -m "Manual configuration change"
        ingestforge audit log create -t "document.pdf" -m "Added new document"
    """
    try:
        audit_log = _get_audit_log(log_path)

        # Resolve operation
        op = _resolve_operation(operation)
        if op is None:
            console.print(
                f"[yellow]Unknown operation '{operation}', using 'custom'[/yellow]"
            )
            op = AuditOperation.CUSTOM

        # Log the entry
        entry = audit_log.log(
            operation=op,
            source=source,
            target=target,
            message=message,
            metadata={"manual_entry": True},
        )

        console.print(f"[green]Logged:[/green] {op.value}")
        console.print(f"[dim]Entry ID: {entry.entry_id}[/dim]")
        console.print(f"[dim]Hash: {entry.entry_hash[:16]}...[/dim]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to log entry: {e}")
        raise typer.Exit(code=1)


def _display_entries_table(result) -> None:
    """Display audit entries in a table.

    Args:
        result: AuditQueryResult with entries.

    Rule #4: Helper to keep list_entries < 60 lines.
    """
    table = Table(title="Audit Log Entries")
    table.add_column("Timestamp", style="cyan", no_wrap=True)
    table.add_column("Operation", style="green")
    table.add_column("Source", style="yellow")
    table.add_column("Target", style="magenta")
    table.add_column("Message", style="dim")

    for entry in result.entries:
        ts_short = entry.timestamp[:19].replace("T", " ")
        msg_short = (
            entry.message[:40] + "..." if len(entry.message) > 40 else entry.message
        )

        table.add_row(
            ts_short,
            entry.operation.value,
            entry.source[:20] if entry.source else "-",
            entry.target[:20] if entry.target else "-",
            msg_short,
        )

    console.print(table)


def _display_query_summary(result) -> None:
    """Display query result summary.

    Args:
        result: AuditQueryResult.

    Rule #4: Helper to keep list_entries < 60 lines.
    """
    console.print(
        f"\n[dim]Showing {len(result.entries)} of {result.total_matches} entries[/dim]"
    )

    if result.truncated:
        console.print("[dim]Results truncated. Use --limit to see more.[/dim]")

    console.print(f"[dim]Query time: {result.query_time_ms:.1f}ms[/dim]")


@app.command("list")
def list_entries(
    operation: Optional[str] = typer.Option(
        None, "--operation", "-o", help="Filter by operation type"
    ),
    source: Optional[str] = typer.Option(
        None, "--source", "-s", help="Filter by source (substring)"
    ),
    target: Optional[str] = typer.Option(
        None, "--target", "-t", help="Filter by target (substring)"
    ),
    start_time: Optional[str] = typer.Option(
        None, "--start", help="Filter after ISO timestamp"
    ),
    end_time: Optional[str] = typer.Option(
        None, "--end", help="Filter before ISO timestamp"
    ),
    limit: int = typer.Option(
        DEFAULT_LIST_LIMIT, "--limit", "-l", help="Maximum entries to show"
    ),
    log_path: Optional[Path] = typer.Option(
        None, "--log-path", help="Path to audit log file"
    ),
) -> None:
    """List audit log entries.

    Rule #4: Function < 60 lines (uses helper methods).
    """
    try:
        audit_log = _get_audit_log(log_path)

        op_filter = _resolve_operation(operation) if operation else None
        if operation and op_filter is None:
            console.print(f"[yellow]Unknown operation '{operation}'[/yellow]")

        result = audit_log.query(
            operation=op_filter,
            source=source,
            target=target,
            start_time=start_time,
            end_time=end_time,
            limit=min(limit, MAX_LIST_LIMIT),
        )

        if not result.entries:
            console.print("[yellow]No audit entries found[/yellow]")
            return

        _display_entries_table(result)
        _display_query_summary(result)

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to list entries: {e}")
        raise typer.Exit(code=1)


@app.command("export")
def export_log(
    output: Path = typer.Argument(..., help="Output file path (JSONL format)"),
    log_path: Optional[Path] = typer.Option(
        None, "--log-path", help="Path to audit log file"
    ),
) -> None:
    """Export audit log to JSONL file.

    Examples:
        ingestforge audit export backup.jsonl
        ingestforge audit export /path/to/export.jsonl
    """
    try:
        audit_log = _get_audit_log(log_path)
        count = audit_log.export_jsonl(output)

        if count > 0:
            console.print(f"[green]Exported {count} entries to {output}[/green]")
        else:
            console.print("[yellow]No entries to export[/yellow]")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to export audit log: {e}")
        raise typer.Exit(code=1)


@app.command("stats")
def show_stats(
    log_path: Optional[Path] = typer.Option(
        None, "--log-path", help="Path to audit log file"
    ),
) -> None:
    """Show audit log statistics.

    Examples:
        ingestforge audit stats
    """
    try:
        audit_log = _get_audit_log(log_path)
        stats = audit_log.get_statistics()

        console.print("\n[bold]Audit Log Statistics[/bold]")
        console.print(f"  Total entries: {stats['total_entries']}")
        console.print(f"  Max entries: {stats['max_entries']}")
        console.print(f"  Entries remaining: {stats['entries_remaining']}")
        console.print(f"  Log file: {stats['log_path'] or 'In-memory only'}")

        integrity = (
            "[green]Valid[/green]" if stats["integrity_valid"] else "[red]BROKEN[/red]"
        )
        console.print(f"  Hash chain integrity: {integrity}")

        if stats["operation_counts"]:
            console.print("\n[bold]Operations by Type:[/bold]")
            for op, count in sorted(stats["operation_counts"].items()):
                console.print(f"  - {op}: {count}")

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to get stats: {e}")
        raise typer.Exit(code=1)


@app.command("verify")
def verify_integrity(
    log_path: Optional[Path] = typer.Option(
        None, "--log-path", help="Path to audit log file"
    ),
) -> None:
    """Verify audit log hash chain integrity.

    Examples:
        ingestforge audit verify
    """
    try:
        audit_log = _get_audit_log(log_path)
        stats = audit_log.get_statistics()

        console.print(f"\n[bold]Verifying {stats['total_entries']} entries...[/bold]")

        if audit_log.verify_integrity():
            console.print("[green]Hash chain integrity: VALID[/green]")
            console.print("All entries have valid hash links.")
        else:
            console.print("[red]Hash chain integrity: BROKEN[/red]")
            console.print("The audit log may have been tampered with.")
            raise typer.Exit(code=1)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to verify integrity: {e}")
        raise typer.Exit(code=1)


@app.command("operations")
def list_operations() -> None:
    """List available audit operation types.

    Examples:
        ingestforge audit operations
    """
    console.print("\n[bold]Available Audit Operations:[/bold]")
    for op in AuditOperation:
        console.print(f"  - {op.value}")


# ============================================================================
# Model Auditor Commands ()
# ============================================================================


@app.command("models")
def audit_models(
    provider: Optional[str] = typer.Option(
        None, "--provider", "-p", help="Audit specific provider only"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed capability info"
    ),
) -> None:
    """Audit LLM provider configurations and feature parity.

    Rule #4: Function < 60 lines.

    Examples:
        ingestforge audit models
        ingestforge audit models --provider openai
        ingestforge audit models --json
    """
    try:
        auditor = create_model_auditor()

        if provider:
            report = auditor.audit_providers([provider])
        else:
            report = auditor.audit_all()

        if json_output:
            _output_models_json(report)
        else:
            _display_models_report(report, verbose)

        raise typer.Exit(code=report.exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to audit models: {e}")
        raise typer.Exit(code=2)


def _output_models_json(report) -> None:
    """Output model audit report as JSON.

    Args:
        report: AuditReport to output.

    Rule #4: Helper function.
    """
    console.print(json.dumps(report.to_dict(), indent=2))


def _display_models_report(report, verbose: bool) -> None:
    """Display model audit report as table.

    Args:
        report: AuditReport to display.
        verbose: Show detailed capability info.

    Rule #4: Helper function.
    """
    table = Table(title="LLM Provider Audit")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Configured", style="yellow")
    table.add_column("Capabilities", style="magenta")
    table.add_column("Missing", style="red")

    for provider in report.providers:
        status_style = _get_status_style(provider.status)
        caps = ", ".join(
            c.value for c in sorted(provider.capabilities, key=lambda x: x.value)
        )
        missing = ", ".join(
            c.value
            for c in sorted(provider.missing_capabilities, key=lambda x: x.value)
        )

        table.add_row(
            provider.provider_name,
            f"[{status_style}]{provider.status.value}[/{status_style}]",
            "Yes" if provider.config_present else "No",
            caps[:30] + "..." if len(caps) > 30 else caps or "-",
            missing[:25] + "..." if len(missing) > 25 else missing or "-",
        )

    console.print(table)
    _display_models_summary(report)

    if verbose:
        _display_capability_details(report)


def _get_status_style(status: AuditStatus) -> str:
    """Get Rich style for audit status.

    Args:
        status: AuditStatus to style.

    Returns:
        Style string.
    """
    styles = {
        AuditStatus.AVAILABLE: "green",
        AuditStatus.UNAVAILABLE: "yellow",
        AuditStatus.ERROR: "red",
        AuditStatus.NOT_CONFIGURED: "dim",
    }
    return styles.get(status, "white")


def _display_models_summary(report) -> None:
    """Display model audit summary.

    Args:
        report: AuditReport to summarize.

    Rule #4: Helper function.
    """
    console.print(
        f"\n[dim]Providers: {report.available_providers}/{report.total_providers} available[/dim]"
    )
    console.print(f"[dim]Audit time: {report.audit_duration_ms:.1f}ms[/dim]")

    summary = report.to_dict().get("summary", {})
    if summary.get("full_parity"):
        console.print(
            f"[green]Full parity:[/green] {', '.join(summary['full_parity'])}"
        )
    if summary.get("partial_parity"):
        console.print(
            f"[yellow]Partial parity:[/yellow] {', '.join(summary['partial_parity'])}"
        )


def _display_capability_details(report) -> None:
    """Display detailed capability information.

    Args:
        report: AuditReport with providers.

    Rule #4: Helper function.
    """
    console.print("\n[bold]Capability Details:[/bold]")
    for provider in report.providers:
        if provider.status == AuditStatus.AVAILABLE:
            console.print(f"\n  [cyan]{provider.provider_name}[/cyan]")
            for cap in sorted(provider.capabilities, key=lambda x: x.value):
                console.print(f"    [green]✓[/green] {cap.value}")
            for cap in sorted(provider.missing_capabilities, key=lambda x: x.value):
                console.print(f"    [red]✗[/red] {cap.value}")


@app.command("providers")
def list_providers() -> None:
    """List known LLM providers.

    Examples:
        ingestforge audit providers
    """
    auditor = create_model_auditor()
    providers = auditor.get_known_providers()

    console.print("\n[bold]Known LLM Providers:[/bold]")
    for provider in providers:
        console.print(f"  - {provider}")


@app.command("capabilities")
def list_capabilities() -> None:
    """List all model capabilities checked during audit.

    Examples:
        ingestforge audit capabilities
    """
    console.print("\n[bold]Model Capabilities:[/bold]")
    for cap in ModelCapability:
        console.print(f"  - {cap.value}")


# ============================================================================
# Dependency Auditor Commands (BUG002)
# ============================================================================


@app.command("deps")
def audit_dependencies(
    project_root: Optional[Path] = typer.Option(
        None, "--root", "-r", help="Project root directory"
    ),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed information"
    ),
) -> None:
    """Audit dependency declarations against actual imports.

    Rule #4: Function < 60 lines.

    Examples:
        ingestforge audit deps
        ingestforge audit deps --json
        ingestforge audit deps --verbose
    """
    try:
        auditor = create_dependency_auditor(project_root=project_root)
        report = auditor.audit()

        if json_output:
            console.print(json.dumps(report.to_dict(), indent=2))
        else:
            _display_deps_report(report, verbose)

        raise typer.Exit(code=report.exit_code)

    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        logger.error(f"Failed to audit dependencies: {e}")
        raise typer.Exit(code=2)


def _display_deps_report(report, verbose: bool) -> None:
    """Display dependency audit report.

    Args:
        report: DependencyReport to display.
        verbose: Show detailed information.

    Rule #4: Helper function.
    """
    console.print("\n[bold]Dependency Audit Report[/bold]")
    console.print(f"[dim]Files scanned: {report.files_scanned}[/dim]")
    console.print(f"[dim]Declared packages: {len(report.declared_packages)}[/dim]")
    console.print(f"[dim]Imported packages: {len(report.imported_packages)}[/dim]")

    if report.missing:
        _display_missing_deps(report.missing, verbose)

    if report.unused:
        _display_unused_deps(report.unused, verbose)

    if not report.missing and not report.unused:
        console.print("\n[green]✓ All dependencies are properly declared[/green]")

    console.print(f"\n[dim]Audit time: {report.audit_duration_ms:.1f}ms[/dim]")


def _display_missing_deps(missing: list, verbose: bool) -> None:
    """Display missing dependencies.

    Args:
        missing: List of missing dependency issues.
        verbose: Show file details.

    Rule #4: Helper function.
    """
    console.print(f"\n[red]Missing Dependencies ({len(missing)}):[/red]")
    table = Table()
    table.add_column("Package", style="red")
    table.add_column("Files", style="dim")

    for issue in missing[:20]:  # Limit display
        files_str = ", ".join(issue.files[:3])
        if len(issue.files) > 3:
            files_str += f" (+{len(issue.files) - 3} more)"
        table.add_row(
            issue.package_name, files_str if verbose else f"{len(issue.files)} files"
        )

    console.print(table)


def _display_unused_deps(unused: list, verbose: bool) -> None:
    """Display unused dependencies.

    Args:
        unused: List of unused dependency issues.
        verbose: Show details.

    Rule #4: Helper function.
    """
    console.print(f"\n[yellow]Unused Dependencies ({len(unused)}):[/yellow]")
    for issue in unused[:20]:  # Limit display
        console.print(f"  - {issue.package_name}")


# Export command for integration
command = app
