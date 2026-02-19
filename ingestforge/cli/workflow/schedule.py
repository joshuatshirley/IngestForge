"""Schedule command - Manage scheduled workflows.

Create, list, run, and delete scheduled workflow tasks.
Uses SQLite for persistent schedule storage."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.workflow.base import WorkflowCommand
from ingestforge.core.security.command import (
    CommandInjectionError,
    safe_run,
)


class ScheduleAction(str, Enum):
    """Schedule command actions."""

    CREATE = "create"
    LIST = "list"
    RUN = "run"
    DELETE = "delete"
    ENABLE = "enable"
    DISABLE = "disable"


@dataclass
class Schedule:
    """Scheduled workflow definition."""

    name: str
    cron_expression: str
    command: str
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    created_at: Optional[datetime] = None
    id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "cron_expression": self.cron_expression,
            "command": self.command,
            "enabled": self.enabled,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "run_count": self.run_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class RunResult:
    """Result of running a schedule."""

    schedule_name: str
    success: bool
    output: str
    duration: float
    started_at: datetime
    error: Optional[str] = None


class CronParser:
    """Simple cron expression parser.

    Supports standard 5-field cron format:
    minute hour day month weekday
    """

    def __init__(self, expression: str) -> None:
        """Initialize parser with expression.

        Args:
            expression: Cron expression string
        """
        self.expression = expression
        self.parts = self._parse(expression)

    def _parse(self, expression: str) -> List[str]:
        """Parse cron expression into parts.

        Args:
            expression: Cron expression

        Returns:
            List of cron fields
        """
        parts = expression.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: expected 5 fields, got {len(parts)}"
            )

        return parts

    def get_next_run(self, from_time: Optional[datetime] = None) -> datetime:
        """Calculate next run time.

        Simple implementation - for production, use croniter library.

        Args:
            from_time: Starting time (defaults to now)

        Returns:
            Next run datetime
        """
        if from_time is None:
            from_time = datetime.now()

        # Parse minute and hour fields
        minute_spec = self.parts[0]
        hour_spec = self.parts[1]

        # Handle simple cases
        next_time = self._calculate_next(from_time, minute_spec, hour_spec)

        return next_time

    def _calculate_next(
        self, from_time: datetime, minute_spec: str, hour_spec: str
    ) -> datetime:
        """Calculate next run based on minute/hour spec.

        Args:
            from_time: Starting time
            minute_spec: Minute field
            hour_spec: Hour field

        Returns:
            Next run time
        """
        # Handle wildcards and simple values
        target_minute = self._parse_field(minute_spec, 0, 59)
        target_hour = self._parse_field(hour_spec, 0, 23)

        # Start with current time
        next_time = from_time.replace(second=0, microsecond=0)

        # Adjust to target time
        next_time = next_time.replace(minute=target_minute, hour=target_hour)

        # If target time is in the past, move to next day
        if next_time <= from_time:
            next_time = next_time + timedelta(days=1)

        return next_time

    def _parse_field(self, field: str, min_val: int, max_val: int) -> int:
        """Parse a cron field value.

        Args:
            field: Field string
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Parsed value
        """
        if field == "*":
            return min_val

        if field.startswith("*/"):
            # Interval notation
            return min_val

        try:
            value = int(field)
            return max(min_val, min(value, max_val))
        except ValueError:
            return min_val


class WorkflowScheduler:
    """Manage scheduled workflows with SQLite persistence."""

    def __init__(self, db_path: Path) -> None:
        """Initialize scheduler.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self) -> None:
        """Ensure database and tables exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schedules (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    cron_expression TEXT NOT NULL,
                    command TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    last_run TEXT,
                    next_run TEXT,
                    run_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    schedule_name TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    duration REAL,
                    success INTEGER,
                    output TEXT,
                    error TEXT
                )
            """
            )
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        """Get database connection.

        Returns:
            SQLite connection
        """
        return sqlite3.connect(str(self.db_path))

    def create_schedule(self, name: str, cron: str, command: str) -> Schedule:
        """Create a new schedule.

        Args:
            name: Schedule name
            cron: Cron expression
            command: Command to execute

        Returns:
            Created schedule
        """
        # Validate cron expression
        parser = CronParser(cron)
        next_run = parser.get_next_run()

        schedule = Schedule(
            name=name,
            cron_expression=cron,
            command=command,
            enabled=True,
            next_run=next_run,
            created_at=datetime.now(),
        )

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO schedules (name, cron_expression, command, next_run, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    name,
                    cron,
                    command,
                    next_run.isoformat(),
                    schedule.created_at.isoformat(),
                ),
            )
            schedule.id = cursor.lastrowid
            conn.commit()

        return schedule

    def list_schedules(self) -> List[Schedule]:
        """List all schedules.

        Returns:
            List of schedules
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, name, cron_expression, command, enabled, "
                "last_run, next_run, run_count, created_at FROM schedules"
            )
            rows = cursor.fetchall()

        return [self._row_to_schedule(row) for row in rows]

    def _row_to_schedule(self, row: tuple) -> Schedule:
        """Convert database row to Schedule.

        Args:
            row: Database row

        Returns:
            Schedule object
        """
        return Schedule(
            id=row[0],
            name=row[1],
            cron_expression=row[2],
            command=row[3],
            enabled=bool(row[4]),
            last_run=self._parse_datetime(row[5]),
            next_run=self._parse_datetime(row[6]),
            run_count=row[7] or 0,
            created_at=self._parse_datetime(row[8]),
        )

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse datetime string.

        Args:
            value: Datetime string or None

        Returns:
            Parsed datetime or None
        """
        if not value:
            return None

        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def get_schedule(self, name: str) -> Optional[Schedule]:
        """Get schedule by name.

        Args:
            name: Schedule name

        Returns:
            Schedule or None
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, name, cron_expression, command, enabled, "
                "last_run, next_run, run_count, created_at FROM schedules WHERE name = ?",
                (name,),
            )
            row = cursor.fetchone()

        if row:
            return self._row_to_schedule(row)
        return None

    def run_schedule(self, name: str) -> RunResult:
        """Run a schedule immediately.

        Args:
            name: Schedule name

        Returns:
            Run result

        Security:
            Uses safe_run to execute commands without shell=True,
            preventing shell injection attacks.
        """
        import subprocess
        import time

        schedule = self.get_schedule(name)
        if not schedule:
            return RunResult(
                schedule_name=name,
                success=False,
                output="",
                duration=0,
                started_at=datetime.now(),
                error=f"Schedule not found: {name}",
            )

        start_time = time.time()
        started_at = datetime.now()

        try:
            # Execute the command safely (no shell=True)
            result = safe_run(
                schedule.command,
                timeout=3600,  # 1 hour timeout
            )

            duration = time.time() - start_time
            success = result.returncode == 0

            run_result = RunResult(
                schedule_name=name,
                success=success,
                output=result.stdout,
                duration=duration,
                started_at=started_at,
                error=result.stderr if not success else None,
            )

        except CommandInjectionError as e:
            # Command validation failed - potential injection attempt
            duration = time.time() - start_time
            run_result = RunResult(
                schedule_name=name,
                success=False,
                output="",
                duration=duration,
                started_at=started_at,
                error=f"Command validation failed: {e}",
            )

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            run_result = RunResult(
                schedule_name=name,
                success=False,
                output="",
                duration=duration,
                started_at=started_at,
                error="Command timed out after 1 hour",
            )

        except Exception as e:
            duration = time.time() - start_time
            run_result = RunResult(
                schedule_name=name,
                success=False,
                output="",
                duration=duration,
                started_at=started_at,
                error=str(e),
            )

        # Update schedule and log run
        self._update_after_run(schedule, run_result)

        return run_result

    def _update_after_run(self, schedule: Schedule, result: RunResult) -> None:
        """Update schedule after running.

        Args:
            schedule: The schedule that was run
            result: Run result
        """
        # Calculate next run
        parser = CronParser(schedule.cron_expression)
        next_run = parser.get_next_run()

        with self._connect() as conn:
            # Update schedule
            conn.execute(
                """
                UPDATE schedules
                SET last_run = ?, next_run = ?, run_count = run_count + 1
                WHERE name = ?
                """,
                (result.started_at.isoformat(), next_run.isoformat(), schedule.name),
            )

            # Log run history
            conn.execute(
                """
                INSERT INTO run_history (schedule_name, started_at, duration, success, output, error)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    schedule.name,
                    result.started_at.isoformat(),
                    result.duration,
                    1 if result.success else 0,
                    result.output[:10000] if result.output else "",
                    result.error,
                ),
            )

            conn.commit()

    def delete_schedule(self, name: str) -> bool:
        """Delete a schedule.

        Args:
            name: Schedule name

        Returns:
            True if deleted
        """
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM schedules WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0

    def enable_schedule(self, name: str, enabled: bool = True) -> bool:
        """Enable or disable a schedule.

        Args:
            name: Schedule name
            enabled: Whether to enable

        Returns:
            True if updated
        """
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE schedules SET enabled = ? WHERE name = ?",
                (1 if enabled else 0, name),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_next_run(self, schedule: Schedule) -> datetime:
        """Get next run time for a schedule.

        Args:
            schedule: Schedule object

        Returns:
            Next run datetime
        """
        parser = CronParser(schedule.cron_expression)
        return parser.get_next_run()


class ScheduleCommand(WorkflowCommand):
    """Manage scheduled workflows."""

    def _get_db_path(self, project: Optional[Path]) -> Path:
        """Get database path.

        Args:
            project: Project directory

        Returns:
            Database file path
        """
        if project:
            return project / ".ingestforge" / "schedules.db"
        return Path.cwd() / ".ingestforge" / "schedules.db"

    def execute_create(
        self,
        name: str,
        cron: str,
        command: str,
        project: Optional[Path] = None,
    ) -> int:
        """Create a new schedule.

        Args:
            name: Schedule name
            cron: Cron expression
            command: Command to execute
            project: Project directory

        Returns:
            Exit code
        """
        try:
            scheduler = WorkflowScheduler(self._get_db_path(project))
            schedule = scheduler.create_schedule(name, cron, command)

            self.console.print()
            self.console.print(
                Panel(
                    f"[bold green]Schedule created:[/bold green] {name}\n\n"
                    f"Cron: {cron}\n"
                    f"Command: {command}\n"
                    f"Next run: {schedule.next_run}",
                    title="Schedule Created",
                    border_style="green",
                )
            )

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to create schedule")

    def execute_list(self, project: Optional[Path] = None) -> int:
        """List all schedules.

        Args:
            project: Project directory

        Returns:
            Exit code
        """
        try:
            scheduler = WorkflowScheduler(self._get_db_path(project))
            schedules = scheduler.list_schedules()

            if not schedules:
                self.print_info("No schedules found")
                return 0

            self.console.print()
            table = Table(title="Scheduled Workflows")
            table.add_column("Name", style="cyan")
            table.add_column("Cron", style="yellow")
            table.add_column("Command", style="dim", max_width=40)
            table.add_column("Enabled", style="green")
            table.add_column("Runs", style="blue")
            table.add_column("Next Run", style="magenta")

            for schedule in schedules:
                enabled = "[green]Yes[/green]" if schedule.enabled else "[red]No[/red]"
                next_run = (
                    schedule.next_run.strftime("%Y-%m-%d %H:%M")
                    if schedule.next_run
                    else "N/A"
                )
                command_display = (
                    schedule.command[:37] + "..."
                    if len(schedule.command) > 40
                    else schedule.command
                )

                table.add_row(
                    schedule.name,
                    schedule.cron_expression,
                    command_display,
                    enabled,
                    str(schedule.run_count),
                    next_run,
                )

            self.console.print(table)
            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to list schedules")

    def execute_run(self, name: str, project: Optional[Path] = None) -> int:
        """Run a schedule immediately.

        Args:
            name: Schedule name
            project: Project directory

        Returns:
            Exit code
        """
        try:
            scheduler = WorkflowScheduler(self._get_db_path(project))
            self.print_info(f"Running schedule: {name}")

            result = scheduler.run_schedule(name)

            if result.success:
                self.print_success(f"Schedule completed in {result.duration:.2f}s")
                if result.output:
                    self.console.print("\n[dim]Output:[/dim]")
                    self.console.print(result.output[:1000])
            else:
                self.print_error(f"Schedule failed: {result.error}")

            return 0 if result.success else 1

        except Exception as e:
            return self.handle_error(e, "Failed to run schedule")

    def execute_delete(self, name: str, project: Optional[Path] = None) -> int:
        """Delete a schedule.

        Args:
            name: Schedule name
            project: Project directory

        Returns:
            Exit code
        """
        try:
            scheduler = WorkflowScheduler(self._get_db_path(project))
            deleted = scheduler.delete_schedule(name)

            if deleted:
                self.print_success(f"Deleted schedule: {name}")
            else:
                self.print_warning(f"Schedule not found: {name}")

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to delete schedule")

    def execute(
        self,
        action: str,
        name: Optional[str] = None,
        cron: Optional[str] = None,
        cmd: Optional[str] = None,
        project: Optional[Path] = None,
    ) -> int:
        """Execute schedule command.

        Args:
            action: Action to perform
            name: Schedule name
            cron: Cron expression
            cmd: Command to execute
            project: Project directory

        Returns:
            Exit code
        """
        handlers = {
            "create": lambda: self.execute_create(
                name or "", cron or "", cmd or "", project
            ),
            "list": lambda: self.execute_list(project),
            "run": lambda: self.execute_run(name or "", project),
            "delete": lambda: self.execute_delete(name or "", project),
        }

        handler = handlers.get(action.lower())
        if handler is None:
            self.print_error(f"Unknown action: {action}")
            return 1

        return handler()


# Typer app for subcommands
app = typer.Typer(
    name="schedule",
    help="Manage scheduled workflows",
    add_completion=False,
)


@app.command("create")
def schedule_create(
    name: str = typer.Argument(..., help="Schedule name"),
    cron: str = typer.Option(..., "--cron", "-c", help="Cron expression"),
    cmd: str = typer.Option(..., "--command", help="Command to execute"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """Create a new scheduled workflow.

    Creates a schedule that runs the specified command at the given cron interval.

    Cron format: minute hour day month weekday
    - minute: 0-59
    - hour: 0-23
    - day: 1-31
    - month: 1-12
    - weekday: 0-6 (Sunday=0)

    Examples:
        # Run at 2 AM daily
        ingestforge workflow schedule create nightly-ingest --cron "0 2 * * *" --command "ingestforge ingest ./docs"

        # Run every hour
        ingestforge workflow schedule create hourly-sync --cron "0 * * * *" --command "ingestforge sync"
    """
    schedule_cmd = ScheduleCommand()
    exit_code = schedule_cmd.execute("create", name, cron, cmd, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("list")
def schedule_list(
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """List all scheduled workflows.

    Shows all configured schedules with their cron expressions,
    commands, and execution status.

    Examples:
        ingestforge workflow schedule list
    """
    schedule_cmd = ScheduleCommand()
    exit_code = schedule_cmd.execute("list", project=project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("run")
def schedule_run(
    name: str = typer.Argument(..., help="Schedule name to run"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """Run a scheduled workflow immediately.

    Executes the specified schedule's command immediately,
    updating its last run time and run count.

    Examples:
        ingestforge workflow schedule run nightly-ingest
    """
    schedule_cmd = ScheduleCommand()
    exit_code = schedule_cmd.execute("run", name, project=project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("delete")
def schedule_delete(
    name: str = typer.Argument(..., help="Schedule name to delete"),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """Delete a scheduled workflow.

    Removes the schedule from the database. This does not
    remove any external cron jobs or task scheduler entries.

    Examples:
        ingestforge workflow schedule delete nightly-ingest
    """
    schedule_cmd = ScheduleCommand()
    exit_code = schedule_cmd.execute("delete", name, project=project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Legacy command function for backward compatibility
def command(
    workflow: str = typer.Argument(
        ..., help="Workflow to schedule (ingest, analyze, backup, cleanup, full)"
    ),
    cron: Optional[str] = typer.Option(
        None, "--cron", "-c", help="Cron expression (e.g., '0 * * * *')"
    ),
    interval: Optional[int] = typer.Option(
        None, "--interval", "-i", help="Interval in minutes"
    ),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output configuration file"),
) -> None:
    """Schedule automated workflow execution.

    Creates a schedule configuration for automated workflow execution.
    Use cron expressions or simple intervals for scheduling.

    Examples:
        # Run every hour
        ingestforge workflow schedule ingest --cron "0 * * * *"

        # Run every 30 minutes
        ingestforge workflow schedule analyze --interval 30

        # Daily backup at 2 AM
        ingestforge workflow schedule backup --cron "0 2 * * *"

        # Save to custom location
        ingestforge workflow schedule full --interval 60 -o schedule.json
    """
    # Convert interval to cron if provided
    if interval and not cron:
        cron = f"*/{interval} * * * *"

    if not cron:
        raise typer.BadParameter("Must specify either --cron or --interval")

    # Create default command based on workflow
    cmd = f"ingestforge workflow pipeline {workflow}"

    schedule_cmd = ScheduleCommand()
    exit_code = schedule_cmd.execute("create", workflow, cron, cmd, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
