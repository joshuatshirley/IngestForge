"""Tests for workflow schedule command.

Tests WorkflowScheduler, CronParser, and schedule management.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ingestforge.cli.workflow.schedule import (
    Schedule,
    CronParser,
    WorkflowScheduler,
    ScheduleCommand,
)


class TestScheduleDataclass:
    """Tests for Schedule dataclass."""

    def test_schedule_creation(self) -> None:
        """Test schedule creation with defaults."""
        schedule = Schedule(
            name="test",
            cron_expression="0 2 * * *",
            command="ingestforge ingest ./docs",
        )

        assert schedule.name == "test"
        assert schedule.cron_expression == "0 2 * * *"
        assert schedule.command == "ingestforge ingest ./docs"
        assert schedule.enabled is True
        assert schedule.run_count == 0

    def test_schedule_to_dict(self) -> None:
        """Test schedule serialization."""
        schedule = Schedule(
            name="test",
            cron_expression="0 2 * * *",
            command="test cmd",
            enabled=True,
            run_count=5,
        )

        data = schedule.to_dict()

        assert data["name"] == "test"
        assert data["cron_expression"] == "0 2 * * *"
        assert data["command"] == "test cmd"
        assert data["enabled"] is True
        assert data["run_count"] == 5

    def test_schedule_with_dates(self) -> None:
        """Test schedule with datetime fields."""
        now = datetime.now()
        schedule = Schedule(
            name="test",
            cron_expression="0 * * * *",
            command="cmd",
            last_run=now,
            next_run=now + timedelta(hours=1),
        )

        data = schedule.to_dict()

        assert data["last_run"] == now.isoformat()
        assert data["next_run"] is not None


class TestCronParser:
    """Tests for CronParser class."""

    def test_parse_valid_cron(self) -> None:
        """Test parsing valid cron expression."""
        parser = CronParser("0 2 * * *")

        assert len(parser.parts) == 5
        assert parser.parts[0] == "0"
        assert parser.parts[1] == "2"

    def test_parse_invalid_cron_too_few(self) -> None:
        """Test parsing invalid cron with too few fields."""
        with pytest.raises(ValueError, match="expected 5 fields"):
            CronParser("0 2 * *")

    def test_parse_invalid_cron_too_many(self) -> None:
        """Test parsing invalid cron with too many fields."""
        with pytest.raises(ValueError, match="expected 5 fields"):
            CronParser("0 2 * * * *")

    def test_get_next_run_hourly(self) -> None:
        """Test next run for hourly schedule."""
        parser = CronParser("0 * * * *")
        from_time = datetime(2024, 1, 15, 10, 30, 0)

        next_run = parser.get_next_run(from_time)

        # Should be next hour at minute 0
        assert next_run > from_time

    def test_get_next_run_specific_time(self) -> None:
        """Test next run for specific time."""
        parser = CronParser("30 14 * * *")
        from_time = datetime(2024, 1, 15, 10, 0, 0)

        next_run = parser.get_next_run(from_time)

        assert next_run.minute == 30
        assert next_run.hour == 14

    def test_get_next_run_already_past(self) -> None:
        """Test next run when target time is past."""
        parser = CronParser("0 8 * * *")
        from_time = datetime(2024, 1, 15, 12, 0, 0)  # Noon

        next_run = parser.get_next_run(from_time)

        # Should be next day at 8 AM
        assert next_run > from_time
        assert next_run.hour == 8

    def test_parse_field_wildcard(self) -> None:
        """Test parsing wildcard field."""
        parser = CronParser("* * * * *")

        assert parser._parse_field("*", 0, 59) == 0

    def test_parse_field_numeric(self) -> None:
        """Test parsing numeric field."""
        parser = CronParser("30 * * * *")

        assert parser._parse_field("30", 0, 59) == 30

    def test_parse_field_out_of_range(self) -> None:
        """Test parsing out of range value."""
        parser = CronParser("0 * * * *")

        assert parser._parse_field("100", 0, 59) == 59  # Clamped to max


class TestWorkflowScheduler:
    """Tests for WorkflowScheduler class."""

    @pytest.fixture
    def db_path(self, tmp_path: Path) -> Path:
        """Create temporary database path."""
        return tmp_path / "test_schedules.db"

    def test_scheduler_initialization(self, db_path: Path) -> None:
        """Test scheduler creates database."""
        scheduler = WorkflowScheduler(db_path)

        assert db_path.exists()

    def test_create_schedule(self, db_path: Path) -> None:
        """Test creating a schedule."""
        scheduler = WorkflowScheduler(db_path)

        schedule = scheduler.create_schedule(
            name="test-schedule",
            cron="0 2 * * *",
            command="ingestforge ingest ./docs",
        )

        assert schedule.name == "test-schedule"
        assert schedule.cron_expression == "0 2 * * *"
        assert schedule.next_run is not None
        assert schedule.id is not None

    def test_create_schedule_duplicate_name(self, db_path: Path) -> None:
        """Test creating schedule with duplicate name fails."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "cmd")

        with pytest.raises(Exception):
            scheduler.create_schedule("test", "0 * * * *", "cmd2")

    def test_list_schedules_empty(self, db_path: Path) -> None:
        """Test listing schedules when empty."""
        scheduler = WorkflowScheduler(db_path)

        schedules = scheduler.list_schedules()

        assert schedules == []

    def test_list_schedules(self, db_path: Path) -> None:
        """Test listing schedules."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("sched1", "0 * * * *", "cmd1")
        scheduler.create_schedule("sched2", "0 2 * * *", "cmd2")

        schedules = scheduler.list_schedules()

        assert len(schedules) == 2
        names = {s.name for s in schedules}
        assert "sched1" in names
        assert "sched2" in names

    def test_get_schedule(self, db_path: Path) -> None:
        """Test getting schedule by name."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "cmd")

        schedule = scheduler.get_schedule("test")

        assert schedule is not None
        assert schedule.name == "test"

    def test_get_schedule_not_found(self, db_path: Path) -> None:
        """Test getting non-existent schedule."""
        scheduler = WorkflowScheduler(db_path)

        schedule = scheduler.get_schedule("nonexistent")

        assert schedule is None

    def test_delete_schedule(self, db_path: Path) -> None:
        """Test deleting a schedule."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "cmd")

        deleted = scheduler.delete_schedule("test")

        assert deleted is True
        assert scheduler.get_schedule("test") is None

    def test_delete_schedule_not_found(self, db_path: Path) -> None:
        """Test deleting non-existent schedule."""
        scheduler = WorkflowScheduler(db_path)

        deleted = scheduler.delete_schedule("nonexistent")

        assert deleted is False

    def test_enable_disable_schedule(self, db_path: Path) -> None:
        """Test enabling and disabling schedule."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "cmd")

        # Disable
        scheduler.enable_schedule("test", enabled=False)
        schedule = scheduler.get_schedule("test")
        assert schedule.enabled is False

        # Enable
        scheduler.enable_schedule("test", enabled=True)
        schedule = scheduler.get_schedule("test")
        assert schedule.enabled is True

    def test_run_schedule_not_found(self, db_path: Path) -> None:
        """Test running non-existent schedule."""
        scheduler = WorkflowScheduler(db_path)

        result = scheduler.run_schedule("nonexistent")

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_run_schedule_simple_command(self, db_path: Path) -> None:
        """Test running a simple schedule."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "echo hello")

        result = scheduler.run_schedule("test")

        assert result.success is True
        assert "hello" in result.output

    def test_run_schedule_updates_count(self, db_path: Path) -> None:
        """Test that running schedule updates run count."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "echo test")

        scheduler.run_schedule("test")

        schedule = scheduler.get_schedule("test")
        assert schedule.run_count == 1
        assert schedule.last_run is not None

    def test_run_schedule_failing_command(self, db_path: Path) -> None:
        """Test running a failing command."""
        scheduler = WorkflowScheduler(db_path)
        scheduler.create_schedule("test", "0 * * * *", "exit 1")

        result = scheduler.run_schedule("test")

        assert result.success is False


class TestScheduleCommand:
    """Tests for ScheduleCommand class."""

    @pytest.fixture
    def temp_project(self, tmp_path: Path) -> Path:
        """Create temporary project directory."""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".ingestforge").mkdir()
        return project

    def test_get_db_path_with_project(self, temp_project: Path) -> None:
        """Test database path with project."""
        cmd = ScheduleCommand()

        db_path = cmd._get_db_path(temp_project)

        assert ".ingestforge" in str(db_path)
        assert "schedules.db" in str(db_path)

    def test_get_db_path_without_project(self) -> None:
        """Test database path without project."""
        cmd = ScheduleCommand()

        db_path = cmd._get_db_path(None)

        assert ".ingestforge" in str(db_path)

    def test_execute_create(self, temp_project: Path) -> None:
        """Test executing create action."""
        cmd = ScheduleCommand()

        exit_code = cmd.execute_create(
            name="test",
            cron="0 2 * * *",
            command="echo test",
            project=temp_project,
        )

        assert exit_code == 0

    def test_execute_list_empty(self, temp_project: Path) -> None:
        """Test executing list action when empty."""
        cmd = ScheduleCommand()

        exit_code = cmd.execute_list(project=temp_project)

        assert exit_code == 0

    def test_execute_list_with_schedules(self, temp_project: Path) -> None:
        """Test executing list action with schedules."""
        cmd = ScheduleCommand()
        cmd.execute_create("test1", "0 * * * *", "cmd1", temp_project)
        cmd.execute_create("test2", "0 2 * * *", "cmd2", temp_project)

        exit_code = cmd.execute_list(project=temp_project)

        assert exit_code == 0

    def test_execute_run(self, temp_project: Path) -> None:
        """Test executing run action."""
        cmd = ScheduleCommand()
        cmd.execute_create("test", "0 * * * *", "echo hello", temp_project)

        exit_code = cmd.execute_run("test", project=temp_project)

        assert exit_code == 0

    def test_execute_delete(self, temp_project: Path) -> None:
        """Test executing delete action."""
        cmd = ScheduleCommand()
        cmd.execute_create("test", "0 * * * *", "cmd", temp_project)

        exit_code = cmd.execute_delete("test", project=temp_project)

        assert exit_code == 0

    def test_execute_unknown_action(self, temp_project: Path) -> None:
        """Test executing unknown action."""
        cmd = ScheduleCommand()

        exit_code = cmd.execute("unknown", project=temp_project)

        assert exit_code == 1
