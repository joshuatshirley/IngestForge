"""Logs command - View and analyze logs.

Views and analyzes application logs.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.monitor.base import MonitorCommand


class LogsCommand(MonitorCommand):
    """View and analyze logs."""

    def execute(
        self,
        project: Optional[Path] = None,
        lines: int = 100,
        follow: bool = False,
        analyze: bool = False,
    ) -> int:
        """View and analyze logs.

        Args:
            project: Project directory
            lines: Number of lines to show
            follow: Follow log (live tail)
            analyze: Analyze log patterns

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Get log files
            log_files = self.get_log_files(project_path)

            if not log_files:
                self.print_warning("No log files found")
                return 0

            # Use most recent log file
            log_file = log_files[0]

            self.print_info(f"Reading: {log_file.name}")

            # Read and display logs
            if analyze:
                self._analyze_logs(log_file, lines)
            else:
                self._display_logs(log_file, lines)

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to read logs")

    def _display_logs(self, log_file: Path, lines: int) -> None:
        """Display log lines.

        Args:
            log_file: Log file path
            lines: Number of lines
        """
        log_lines = self.read_log_file(log_file, lines)

        if not log_lines:
            self.print_warning("Log file is empty")
            return

        self.console.print()
        self.console.print(f"[bold]Last {len(log_lines)} lines:[/bold]")
        self.console.print()

        for line in log_lines:
            color = self._get_line_color(line)
            self.console.print(f"[{color}]{line}[/{color}]")

    def _get_line_color(self, line: str) -> str:
        """Get color tag for log line based on content.

        Args:
            line: Log line text

        Returns:
            Color tag for rich console
        """
        line_lower = line.lower()

        if "error" in line_lower:
            return "red"
        if "warn" in line_lower:
            return "yellow"
        if "info" in line_lower:
            return "cyan"

        return "white"

    def _analyze_logs(self, log_file: Path, lines: int) -> None:
        """Analyze log patterns.

        Args:
            log_file: Log file path
            lines: Number of lines
        """
        log_lines = self.read_log_file(log_file, lines)

        if not log_lines:
            self.print_warning("Log file is empty")
            return

        # Analyze patterns
        analysis = self.analyze_logs(log_lines)

        # Display analysis
        from rich.table import Table

        self.console.print()
        table = Table(title="Log Analysis")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="bold")

        table.add_row("Total Lines", str(analysis["total_lines"]))
        table.add_row("Errors", f"[red]{analysis['errors']}[/red]")
        table.add_row("Warnings", f"[yellow]{analysis['warnings']}[/yellow]")
        table.add_row("Info", f"[cyan]{analysis['info']}[/cyan]")

        self.console.print(table)

        # Show recent errors
        if analysis["errors"] > 0:
            self._show_recent_errors(log_lines)

    def _show_recent_errors(self, log_lines: list) -> None:
        """Show recent error lines.

        Args:
            log_lines: Log lines
        """
        self.console.print()
        self.console.print("[bold red]Recent Errors:[/bold red]")
        self.console.print()

        error_lines = [line for line in log_lines if "error" in line.lower()]

        # Show last 5 errors
        for line in error_lines[-5:]:
            self.console.print(f"[red]{line}[/red]")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    lines: int = typer.Option(100, "--lines", "-n", help="Number of lines to show"),
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow log (live tail)"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Analyze log patterns"),
) -> None:
    """View and analyze logs.

    Views application logs with optional pattern analysis.
    Shows the most recent log file by default.

    Features:
    - View recent log lines
    - Color-coded output (errors/warnings/info)
    - Pattern analysis
    - Error highlighting

    Examples:
        # View last 100 lines
        ingestforge monitor logs

        # View last 500 lines
        ingestforge monitor logs --lines 500

        # Analyze log patterns
        ingestforge monitor logs --analyze

        # View logs from specific project
        ingestforge monitor logs -p /path/to/project
    """
    cmd = LogsCommand()
    exit_code = cmd.execute(project, lines, follow, analyze)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
