"""Metrics command - Show system metrics.

Displays system metrics and statistics.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer

from ingestforge.cli.monitor.base import MonitorCommand


class MetricsCommand(MonitorCommand):
    """Show system metrics."""

    def execute(
        self,
        project: Optional[Path] = None,
        format: str = "summary",
    ) -> int:
        """Show system metrics.

        Args:
            project: Project directory
            format: Output format (summary/detailed/json)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Collect metrics
            metrics = self._collect_metrics(project_path)

            # Display based on format
            if format == "json":
                self._display_json(metrics)
            elif format == "detailed":
                self._display_detailed(metrics)
            else:
                self._display_summary(metrics)

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to collect metrics")

    def _collect_metrics(self, project: Path) -> dict[str, Any]:
        """Collect system metrics.

        Args:
            project: Project directory

        Returns:
            Metrics dictionary
        """
        # System status
        status = self.get_system_status(project)

        # Storage metrics
        storage = self.get_storage_metrics(project)

        # Memory metrics
        memory = self.get_memory_metrics()

        # Dependencies
        deps = self.check_dependencies()

        return {
            "status": status,
            "storage": storage,
            "memory": memory,
            "dependencies": deps,
        }

    def _display_summary(self, metrics: dict) -> None:
        """Display metrics summary.

        Args:
            metrics: Metrics dictionary
        """
        from rich.table import Table

        self.console.print()

        # Status
        status = metrics["status"]
        self.console.print(f"[bold]Status:[/bold] {status['status']}")
        self.console.print()

        # Storage metrics
        storage = metrics["storage"]
        if storage["exists"]:
            table = Table(title="Storage Metrics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="bold")

            table.add_row("Size", storage["size_formatted"])
            table.add_row("Files", str(storage["files"]))

            self.console.print(table)
            self.console.print()

        # Dependencies
        deps = metrics["dependencies"]
        table = Table(title="Dependencies")
        table.add_column("Package", style="cyan")
        table.add_column("Status", style="bold")

        for dep, available in deps.items():
            status_text = "[green]✓[/green]" if available else "[red]✗[/red]"
            table.add_row(dep, status_text)

        self.console.print(table)

    def _display_detailed(self, metrics: dict) -> None:
        """Display detailed metrics.

        Args:
            metrics: Metrics dictionary
        """
        self._display_summary(metrics)

        # Additional details
        self.console.print()
        self.console.print("[bold]System Information:[/bold]")

        memory = metrics["memory"]
        self.console.print(f"  Python: {memory['python_version']}")
        self.console.print(f"  Platform: {memory['platform']}")

    def _display_json(self, metrics: dict) -> None:
        """Display metrics as JSON.

        Args:
            metrics: Metrics dictionary
        """
        import json

        json_str = json.dumps(metrics, indent=2, ensure_ascii=False)
        self.console.print(json_str)


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    format: str = typer.Option(
        "summary",
        "--format",
        "-f",
        help="Output format (summary/detailed/json)",
    ),
) -> None:
    """Show system metrics.

    Displays system metrics including storage usage,
    dependencies, and system information.

    Metrics:
    - System status
    - Storage size and file count
    - Dependencies availability
    - System information

    Formats:
    - summary: Key metrics overview (default)
    - detailed: Full metrics with system info
    - json: Raw JSON output

    Examples:
        # Show metrics summary
        ingestforge monitor metrics

        # Show detailed metrics
        ingestforge monitor metrics --format detailed

        # Show as JSON
        ingestforge monitor metrics --format json

        # Check specific project
        ingestforge monitor metrics -p /path/to/project
    """
    cmd = MetricsCommand()
    exit_code = cmd.execute(project, format)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
