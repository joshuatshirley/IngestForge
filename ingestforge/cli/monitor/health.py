"""Health command - System health check.

Performs comprehensive system health checks.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.monitor.base import MonitorCommand


class HealthCommand(MonitorCommand):
    """Perform system health check."""

    def execute(
        self,
        project: Optional[Path] = None,
        detailed: bool = False,
    ) -> int:
        """Run health check.

        Args:
            project: Project directory
            detailed: Show detailed information

        Returns:
            0 if healthy, 1 if issues found
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Run health checks
            checks = self.run_health_checks(project_path)

            # Display results
            self._display_health_results(checks, detailed)

            # Determine overall health
            has_failures = any(c["status"] == "fail" for c in checks)

            return 1 if has_failures else 0

        except Exception as e:
            return self.handle_error(e, "Health check failed")

    def _display_health_results(self, checks: list, detailed: bool) -> None:
        """Display health check results.

        Args:
            checks: List of check results
            detailed: Show detailed info
        """
        from rich.table import Table

        self.console.print()
        table = Table(title="System Health Check")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Message", style="dim")

        for check in checks:
            status = check["status"]
            status_display = self._format_status(status)

            table.add_row(
                check["name"],
                status_display,
                check["message"],
            )

        self.console.print(table)

        # Overall summary
        passed = sum(1 for c in checks if c["status"] == "pass")
        failed = sum(1 for c in checks if c["status"] == "fail")
        warnings = sum(1 for c in checks if c["status"] == "warn")

        self.console.print()
        self.console.print("[bold]Summary:[/bold]")
        self.console.print(f"  Passed: [green]{passed}[/green]")
        if warnings > 0:
            self.console.print(f"  Warnings: [yellow]{warnings}[/yellow]")
        if failed > 0:
            self.console.print(f"  Failed: [red]{failed}[/red]")

        # Overall status
        self.console.print()
        if failed > 0:
            self.print_error("System health check failed")
        elif warnings > 0:
            self.print_warning("System health check passed with warnings")
        else:
            self.print_success("System health check passed")

    def _format_status(self, status: str) -> str:
        """Format status for display.

        Args:
            status: Status string

        Returns:
            Formatted status
        """
        if status == "pass":
            return "[green]✓ Pass[/green]"
        elif status == "warn":
            return "[yellow]! Warn[/yellow]"
        else:
            return "[red]✗ Fail[/red]"


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show detailed information"
    ),
) -> None:
    """Run system health check.

    Performs comprehensive health checks on the IngestForge
    system including initialization, configuration, storage,
    and dependencies.

    Health checks:
    - Project initialization
    - Configuration status
    - Storage availability
    - Dependencies

    Exit codes:
    - 0: All checks passed
    - 1: One or more checks failed

    Examples:
        # Run health check
        ingestforge monitor health

        # Detailed health check
        ingestforge monitor health --detailed

        # Check specific project
        ingestforge monitor health -p /path/to/project
    """
    cmd = HealthCommand()
    exit_code = cmd.execute(project, detailed)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
