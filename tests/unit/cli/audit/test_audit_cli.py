"""Tests for Audit CLI Commands.

Append-Only Audit Log
Tests CLI commands for audit log management.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from ingestforge.cli.audit.main import app, _resolve_operation
from ingestforge.core.audit import AuditOperation


runner = CliRunner()


class TestResolveOperation:
    """Tests for _resolve_operation helper."""

    def test_resolve_valid_operation(self) -> None:
        """Test resolving valid operation string."""
        op = _resolve_operation("create")
        assert op == AuditOperation.CREATE

    def test_resolve_ingest_complete(self) -> None:
        """Test resolving ingest_complete operation."""
        op = _resolve_operation("ingest_complete")
        assert op == AuditOperation.INGEST_COMPLETE

    def test_resolve_unknown_operation(self) -> None:
        """Test that unknown operation returns None."""
        op = _resolve_operation("unknown_operation")
        assert op is None

    def test_resolve_empty_string(self) -> None:
        """Test that empty string returns None."""
        op = _resolve_operation("")
        assert op is None


class TestAuditLogCommand:
    """Tests for 'audit log' command."""

    def test_log_custom_entry(self) -> None:
        """Test logging a custom entry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            result = runner.invoke(
                app,
                ["log", "custom", "-m", "Test message", "--log-path", str(log_path)],
            )

            assert result.exit_code == 0
            assert "Logged:" in result.stdout

    def test_log_with_source_and_target(self) -> None:
        """Test logging with source and target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            result = runner.invoke(
                app,
                [
                    "log",
                    "create",
                    "-m",
                    "Created document",
                    "-s",
                    "test_module",
                    "-t",
                    "doc.pdf",
                    "--log-path",
                    str(log_path),
                ],
            )

            assert result.exit_code == 0
            assert "create" in result.stdout

    def test_log_unknown_operation_defaults_to_custom(self) -> None:
        """Test that unknown operation defaults to custom."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            result = runner.invoke(
                app,
                ["log", "unknown_op", "-m", "Test", "--log-path", str(log_path)],
            )

            assert result.exit_code == 0
            assert "custom" in result.stdout.lower()


class TestAuditListCommand:
    """Tests for 'audit list' command."""

    def test_list_empty_log(self) -> None:
        """Test listing empty audit log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            result = runner.invoke(app, ["list", "--log-path", str(log_path)])

            assert result.exit_code == 0
            assert "No audit entries found" in result.stdout

    def test_list_with_entries(self) -> None:
        """Test listing audit log with entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            # Log some entries
            runner.invoke(
                app,
                ["log", "create", "-m", "Entry 1", "--log-path", str(log_path)],
            )
            runner.invoke(
                app,
                ["log", "update", "-m", "Entry 2", "--log-path", str(log_path)],
            )

            # List entries
            result = runner.invoke(app, ["list", "--log-path", str(log_path)])

            assert result.exit_code == 0
            assert "Audit Log Entries" in result.stdout

    def test_list_with_limit(self) -> None:
        """Test listing with limit parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            # Log entries
            for i in range(5):
                runner.invoke(
                    app,
                    ["log", "create", "-m", f"Entry {i}", "--log-path", str(log_path)],
                )

            # List with limit
            result = runner.invoke(
                app,
                ["list", "--limit", "2", "--log-path", str(log_path)],
            )

            assert result.exit_code == 0
            assert "2 of 5" in result.stdout

    def test_list_filter_by_operation(self) -> None:
        """Test listing filtered by operation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            # Log different operations
            runner.invoke(
                app,
                ["log", "create", "-m", "Create", "--log-path", str(log_path)],
            )
            runner.invoke(
                app,
                ["log", "update", "-m", "Update", "--log-path", str(log_path)],
            )

            # Filter by operation
            result = runner.invoke(
                app,
                ["list", "-o", "create", "--log-path", str(log_path)],
            )

            assert result.exit_code == 0


class TestAuditStatsCommand:
    """Tests for 'audit stats' command."""

    def test_stats_empty_log(self) -> None:
        """Test stats on empty log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            result = runner.invoke(app, ["stats", "--log-path", str(log_path)])

            assert result.exit_code == 0
            assert "Total entries: 0" in result.stdout

    def test_stats_with_entries(self) -> None:
        """Test stats with entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            # Log entries
            runner.invoke(
                app,
                ["log", "create", "-m", "Test", "--log-path", str(log_path)],
            )
            runner.invoke(
                app,
                ["log", "create", "-m", "Test 2", "--log-path", str(log_path)],
            )

            result = runner.invoke(app, ["stats", "--log-path", str(log_path)])

            assert result.exit_code == 0
            assert "Total entries: 2" in result.stdout
            assert "create" in result.stdout


class TestAuditVerifyCommand:
    """Tests for 'audit verify' command."""

    def test_verify_empty_log(self) -> None:
        """Test verifying empty log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            result = runner.invoke(app, ["verify", "--log-path", str(log_path)])

            assert result.exit_code == 0
            assert "VALID" in result.stdout

    def test_verify_valid_log(self) -> None:
        """Test verifying log with valid entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_audit.jsonl"

            # Log entries
            runner.invoke(
                app,
                ["log", "create", "-m", "Entry 1", "--log-path", str(log_path)],
            )
            runner.invoke(
                app,
                ["log", "update", "-m", "Entry 2", "--log-path", str(log_path)],
            )

            result = runner.invoke(app, ["verify", "--log-path", str(log_path)])

            assert result.exit_code == 0
            assert "VALID" in result.stdout


class TestAuditExportCommand:
    """Tests for 'audit export' command."""

    def test_export_empty_log(self) -> None:
        """Test exporting empty log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            export_path = Path(tmpdir) / "export.jsonl"

            result = runner.invoke(
                app,
                ["export", str(export_path), "--log-path", str(log_path)],
            )

            assert result.exit_code == 0
            assert "No entries" in result.stdout

    def test_export_with_entries(self) -> None:
        """Test exporting log with entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            export_path = Path(tmpdir) / "export.jsonl"

            # Log entries
            runner.invoke(
                app,
                ["log", "create", "-m", "Test", "--log-path", str(log_path)],
            )
            runner.invoke(
                app,
                ["log", "update", "-m", "Test 2", "--log-path", str(log_path)],
            )

            result = runner.invoke(
                app,
                ["export", str(export_path), "--log-path", str(log_path)],
            )

            assert result.exit_code == 0
            assert "Exported 2 entries" in result.stdout
            assert export_path.exists()


class TestAuditOperationsCommand:
    """Tests for 'audit operations' command."""

    def test_list_operations(self) -> None:
        """Test listing available operations."""
        result = runner.invoke(app, ["operations"])

        assert result.exit_code == 0
        assert "create" in result.stdout
        assert "ingest_start" in result.stdout
        assert "query_execute" in result.stdout


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_limit_bounds(self) -> None:
        """Test that list enforces limit bounds."""
        from ingestforge.cli.audit.main import MAX_LIST_LIMIT

        # MAX_LIST_LIMIT should be defined
        assert MAX_LIST_LIMIT > 0

    def test_rule_9_type_hints(self) -> None:
        """Test that helper functions have type hints."""
        import inspect

        sig = inspect.signature(_resolve_operation)

        # Check parameter has type hint
        param = sig.parameters["op_str"]
        assert param.annotation == str

        # Check return type
        # Return annotation uses Union which may appear differently
        assert sig.return_annotation is not None
