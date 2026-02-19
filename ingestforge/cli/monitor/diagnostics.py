"""Diagnostics command - Run system diagnostics.

Runs comprehensive system diagnostics.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer

from ingestforge.cli.monitor.base import MonitorCommand


class DiagnosticsCommand(MonitorCommand):
    """Run system diagnostics."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Run diagnostics.

        Args:
            project: Project directory
            output: Output file for report

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)
            project_path = ctx["project_dir"]

            # Run diagnostics
            self.print_info("Running diagnostics...")
            results = self._run_diagnostics(project_path)

            # Display results
            self._display_diagnostics(results)

            # Save report if requested
            if output:
                self._save_report(output, results)

            return 0

        except Exception as e:
            return self.handle_error(e, "Diagnostics failed")

    def _run_diagnostics(self, project: Path) -> Dict[str, Any]:
        """Run diagnostic tests.

        Args:
            project: Project directory

        Returns:
            Diagnostic results
        """
        results = {
            "health": self.run_health_checks(project),
            "storage": self.get_storage_metrics(project),
            "memory": self.get_memory_metrics(),
            "dependencies": self.check_dependencies(),
        }

        # Check logs
        log_files = self.get_log_files(project)
        if log_files:
            log_lines = self.read_log_file(log_files[0], 1000)
            results["log_analysis"] = self.analyze_logs(log_lines)
        else:
            results["log_analysis"] = {
                "total_lines": 0,
                "errors": 0,
                "warnings": 0,
                "info": 0,
            }

        return results

    def _display_diagnostics(self, results: Dict[str, Any]) -> None:
        """Display diagnostic results.

        Args:
            results: Diagnostic results
        """

        # Health checks
        self.console.print()
        self.console.print("[bold]Health Checks:[/bold]")
        for check in results["health"]:
            status = self._format_status(check["status"])
            self.console.print(f"  {status} {check['name']}: {check['message']}")

        # Storage
        storage = results["storage"]
        if storage["exists"]:
            self.console.print()
            self.console.print("[bold]Storage:[/bold]")
            self.console.print(f"  Size: {storage['size_formatted']}")
            self.console.print(f"  Files: {storage['files']}")

        # Log analysis
        log_analysis = results["log_analysis"]
        if log_analysis["total_lines"] > 0:
            self.console.print()
            self.console.print("[bold]Log Analysis (last 1000 lines):[/bold]")
            self.console.print(f"  Total: {log_analysis['total_lines']}")
            self.console.print(f"  Errors: [red]{log_analysis['errors']}[/red]")
            self.console.print(
                f"  Warnings: [yellow]{log_analysis['warnings']}[/yellow]"
            )
            self.console.print(f"  Info: [cyan]{log_analysis['info']}[/cyan]")

        # Dependencies
        deps = results["dependencies"]
        self.console.print()
        self.console.print("[bold]Dependencies:[/bold]")
        for dep, available in deps.items():
            status = "[green]✓[/green]" if available else "[red]✗[/red]"
            self.console.print(f"  {status} {dep}")

        # Summary
        self.console.print()
        health_passed = all(c["status"] != "fail" for c in results["health"])
        if health_passed and log_analysis.get("errors", 0) == 0:
            self.print_success("Diagnostics: System healthy")
        else:
            self.print_warning("Diagnostics: Issues detected")

    def _format_status(self, status: str) -> str:
        """Format status for display.

        Args:
            status: Status string

        Returns:
            Formatted status
        """
        if status == "pass":
            return "[green]✓[/green]"
        elif status == "warn":
            return "[yellow]![/yellow]"
        else:
            return "[red]✗[/red]"

    def _build_report_sections(self, results: Dict[str, Any]) -> List[str]:
        """Build all report sections.

        Rule #4: No large functions - Extracted from _save_report
        """
        lines = [
            "# System Diagnostics Report",
            "",
            f"**Date:** {self._get_timestamp()}",
            "",
            "## Health Checks",
            "",
        ]

        # Health checks
        for check in results["health"]:
            status_text = "✓" if check["status"] == "pass" else "✗"
            lines.append(f"- {status_text} **{check['name']}:** {check['message']}")

        # Storage
        storage = results["storage"]
        if storage["exists"]:
            lines.extend(
                [
                    "",
                    "## Storage",
                    "",
                    f"- **Size:** {storage['size_formatted']}",
                    f"- **Files:** {storage['files']}",
                ]
            )

        # Log analysis
        log_analysis = results["log_analysis"]
        if log_analysis["total_lines"] > 0:
            lines.extend(
                [
                    "",
                    "## Log Analysis",
                    "",
                    f"- **Total Lines:** {log_analysis['total_lines']}",
                    f"- **Errors:** {log_analysis['errors']}",
                    f"- **Warnings:** {log_analysis['warnings']}",
                    f"- **Info:** {log_analysis['info']}",
                ]
            )

        # Dependencies
        lines.extend(
            [
                "",
                "## Dependencies",
                "",
            ]
        )

        for dep, available in results["dependencies"].items():
            status_text = "✓" if available else "✗"
            lines.append(f"- {status_text} {dep}")

        return lines

    def _save_report(self, output: Path, results: Dict[str, Any]) -> None:
        """Save diagnostic report.

        Rule #4: Function <60 lines (refactored to 17 lines)

        Args:
            output: Output file path
            results: Diagnostic results
        """
        # Build report sections
        lines = self._build_report_sections(results)
        report_text = "\n".join(lines)

        # Write to file
        try:
            output.write_text(report_text, encoding="utf-8")
            self.print_success(f"Report saved: {output}")
        except Exception as e:
            self.print_error(f"Failed to save report: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp string.

        Returns:
            Formatted timestamp
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
) -> None:
    """Run comprehensive system diagnostics.

    Runs a full diagnostic suite including:
    - Health checks
    - Storage metrics
    - Log analysis
    - Dependencies check
    - System information

    Useful for troubleshooting issues and verifying
    system configuration.

    Examples:
        # Run diagnostics
        ingestforge monitor diagnostics

        # Run diagnostics with report
        ingestforge monitor diagnostics -o diagnostics.md

        # Check specific project
        ingestforge monitor diagnostics -p /path/to/project
    """
    cmd = DiagnosticsCommand()
    exit_code = cmd.execute(project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
