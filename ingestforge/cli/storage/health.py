"""Health command - Check storage backend health.

Performs health checks on the storage backend including
connection tests and basic operations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import typer
from rich.table import Table

from ingestforge.cli.storage.base import StorageCommand


class HealthCommand(StorageCommand):
    """Check storage backend health."""

    def execute(
        self,
        project: Optional[Path] = None,
        format: str = "table",
    ) -> int:
        """Check storage health.

        Args:
            project: Project directory
            format: Output format (table/json)

        Returns:
            0 if healthy, 1 if unhealthy
        """
        try:
            config = self.get_config(project)
            storage = self.get_storage(config)

            health_data = self._collect_health_data(storage, config)

            if format == "json":
                self._display_json(health_data)
            else:
                self._display_table(health_data)

            return 0 if health_data["overall_status"] == "healthy" else 1

        except Exception as e:
            return self.handle_error(e, "Health check failed")

    def _collect_health_data(self, storage: Any, config: Any) -> Dict[str, Any]:
        """Collect health check data.

        Args:
            storage: Storage backend
            config: Configuration

        Returns:
            Health data dictionary
        """
        health_data = {
            "backend": config.storage.backend,
            "overall_status": "healthy",
            "checks": {},
        }

        # Connection check
        health_data["checks"]["connection"] = self._check_connection(storage)

        # Count check
        health_data["checks"]["count"] = self._check_count(storage)

        # Backend-specific checks
        if hasattr(storage, "health_check"):
            healthy, message = storage.health_check()
            health_data["checks"]["backend"] = {
                "status": "pass" if healthy else "fail",
                "message": message,
            }

        # Determine overall status
        for check in health_data["checks"].values():
            if check.get("status") == "fail":
                health_data["overall_status"] = "unhealthy"
                break

        return health_data

    def _check_connection(self, storage: Any) -> Dict[str, str]:
        """Check storage connection.

        Args:
            storage: Storage backend

        Returns:
            Check result
        """
        try:
            # Try to get count - this requires working connection
            storage.count()
            return {"status": "pass", "message": "Connected"}
        except Exception as e:
            return {"status": "fail", "message": str(e)}

    def _check_count(self, storage: Any) -> Dict[str, Any]:
        """Check chunk count.

        Args:
            storage: Storage backend

        Returns:
            Check result
        """
        try:
            count = storage.count()
            return {
                "status": "pass",
                "message": f"{count} chunks",
                "count": count,
            }
        except Exception as e:
            return {"status": "fail", "message": str(e)}

    def _display_table(self, health_data: Dict[str, Any]) -> None:
        """Display health as table.

        Args:
            health_data: Health data
        """
        self.console.print()

        # Overall status
        status = health_data["overall_status"]
        if status == "healthy":
            self.print_success(f"Storage is healthy ({health_data['backend']})")
        else:
            self.print_error(f"Storage is unhealthy ({health_data['backend']})")

        # Checks table
        table = Table(title="Health Checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Message")

        for name, result in health_data["checks"].items():
            status_style = "green" if result["status"] == "pass" else "red"
            status_icon = (
                "[green]PASS[/green]"
                if result["status"] == "pass"
                else "[red]FAIL[/red]"
            )
            table.add_row(
                name.title(),
                status_icon,
                result.get("message", ""),
            )

        self.console.print(table)

    def _display_json(self, health_data: Dict[str, Any]) -> None:
        """Display health as JSON.

        Args:
            health_data: Health data
        """
        json_str = json.dumps(health_data, indent=2)
        self.console.print(json_str)


def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    format: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format (table/json)",
    ),
) -> None:
    """Check storage backend health.

    Performs health checks on the storage backend:
    - Connection test
    - Count verification
    - Backend-specific checks

    Returns exit code 0 if healthy, 1 if unhealthy.

    Examples:
        # Check storage health
        ingestforge storage health

        # Output as JSON
        ingestforge storage health --format json

        # For specific project
        ingestforge storage health -p /path/to/project

        # Use in scripts
        if ingestforge storage health --format json | jq '.overall_status == "healthy"'; then
            echo "Storage OK"
        fi
    """
    cmd = HealthCommand()
    exit_code = cmd.execute(project, format)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
