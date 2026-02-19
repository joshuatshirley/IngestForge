"""Tests for Append-Only Audit Log.

Append-Only Audit Log
Tests audit log creation, logging, querying, and hash chain integrity.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from ingestforge.core.audit.log import (
    AuditLogEntry,
    AuditOperation,
    AuditQueryResult,
    AppendOnlyAuditLog,
    create_audit_log,
    MAX_AUDIT_ENTRIES,
    MAX_QUERY_RESULTS,
    MAX_SOURCE_LENGTH,
    MAX_TARGET_LENGTH,
    MAX_MESSAGE_LENGTH,
)


class TestAuditLogEntry:
    """Tests for AuditLogEntry dataclass."""

    def test_create_entry(self) -> None:
        """Test creating a valid entry."""
        entry = AuditLogEntry(
            entry_id="test-123",
            timestamp="2026-02-17T12:00:00Z",
            operation=AuditOperation.CREATE,
            source="test_module",
            target="document.pdf",
            message="Created document",
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )

        assert entry.entry_id == "test-123"
        assert entry.operation == AuditOperation.CREATE
        assert entry.source == "test_module"

    def test_entry_immutable(self) -> None:
        """Test that entry is frozen/immutable."""
        entry = AuditLogEntry(
            entry_id="test-123",
            timestamp="2026-02-17T12:00:00Z",
            operation=AuditOperation.CREATE,
            source="test",
            target="test",
            message="test",
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )

        with pytest.raises(AttributeError):
            entry.source = "modified"  # type: ignore

    def test_entry_to_dict(self) -> None:
        """Test converting entry to dictionary."""
        entry = AuditLogEntry(
            entry_id="test-123",
            timestamp="2026-02-17T12:00:00Z",
            operation=AuditOperation.INGEST_COMPLETE,
            source="pipeline",
            target="file.txt",
            message="Ingestion complete",
            previous_hash="0" * 64,
            entry_hash="a" * 64,
            metadata={"chunks": 10},
        )

        data = entry.to_dict()

        assert data["entry_id"] == "test-123"
        assert data["operation"] == "ingest_complete"
        assert data["metadata"]["chunks"] == 10

    def test_entry_from_dict(self) -> None:
        """Test creating entry from dictionary."""
        data = {
            "entry_id": "test-456",
            "timestamp": "2026-02-17T13:00:00Z",
            "operation": "query_execute",
            "source": "user",
            "target": "corpus",
            "message": "Ran query",
            "previous_hash": "0" * 64,
            "entry_hash": "b" * 64,
            "metadata": {"query": "test"},
        }

        entry = AuditLogEntry.from_dict(data)

        assert entry.entry_id == "test-456"
        assert entry.operation == AuditOperation.QUERY_EXECUTE
        assert entry.metadata["query"] == "test"

    def test_entry_from_dict_unknown_operation(self) -> None:
        """Test that unknown operation defaults to CUSTOM."""
        data = {
            "operation": "unknown_op",
            "source": "test",
            "target": "test",
            "message": "test",
            "previous_hash": "0" * 64,
            "entry_hash": "a" * 64,
        }

        entry = AuditLogEntry.from_dict(data)

        assert entry.operation == AuditOperation.CUSTOM

    def test_entry_to_json_roundtrip(self) -> None:
        """Test JSON serialization roundtrip."""
        original = AuditLogEntry(
            entry_id="round-trip",
            timestamp="2026-02-17T14:00:00Z",
            operation=AuditOperation.CONFIG_CHANGE,
            source="admin",
            target="settings",
            message="Changed config",
            previous_hash="c" * 64,
            entry_hash="d" * 64,
            metadata={"key": "value"},
        )

        json_str = original.to_json()
        restored = AuditLogEntry.from_json(json_str)

        assert restored.entry_id == original.entry_id
        assert restored.operation == original.operation
        assert restored.metadata == original.metadata

    def test_entry_truncates_long_source(self) -> None:
        """Test that long source is truncated."""
        long_source = "x" * (MAX_SOURCE_LENGTH + 50)

        entry = AuditLogEntry(
            entry_id="test",
            timestamp="2026-02-17T12:00:00Z",
            operation=AuditOperation.CUSTOM,
            source=long_source,
            target="",
            message="",
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )

        assert len(entry.source) == MAX_SOURCE_LENGTH

    def test_entry_truncates_long_message(self) -> None:
        """Test that long message is truncated."""
        long_message = "y" * (MAX_MESSAGE_LENGTH + 100)

        entry = AuditLogEntry(
            entry_id="test",
            timestamp="2026-02-17T12:00:00Z",
            operation=AuditOperation.CUSTOM,
            source="",
            target="",
            message=long_message,
            previous_hash="0" * 64,
            entry_hash="a" * 64,
        )

        assert len(entry.message) == MAX_MESSAGE_LENGTH

    def test_invalid_empty_entry_id(self) -> None:
        """Test that empty entry_id fails validation."""
        with pytest.raises(AssertionError):
            AuditLogEntry(
                entry_id="",
                timestamp="2026-02-17T12:00:00Z",
                operation=AuditOperation.CUSTOM,
                source="",
                target="",
                message="",
                previous_hash="0" * 64,
                entry_hash="a" * 64,
            )


class TestAppendOnlyAuditLog:
    """Tests for AppendOnlyAuditLog class."""

    def test_create_audit_log(self) -> None:
        """Test creating an audit log."""
        log = AppendOnlyAuditLog()

        assert log is not None
        assert len(log.get_entries()) == 0

    def test_log_entry(self) -> None:
        """Test logging an entry."""
        log = AppendOnlyAuditLog()

        entry = log.log(
            operation=AuditOperation.CREATE,
            source="test",
            target="document",
            message="Created document",
        )

        assert entry.operation == AuditOperation.CREATE
        assert entry.source == "test"
        assert entry.entry_id != ""
        assert entry.entry_hash != ""

    def test_log_multiple_entries(self) -> None:
        """Test logging multiple entries."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.INGEST_START, source="pipeline")
        log.log(AuditOperation.PIPELINE_STAGE, source="pipeline", message="Stage 1")
        log.log(AuditOperation.INGEST_COMPLETE, source="pipeline")

        entries = log.get_entries()

        assert len(entries) == 3

    def test_hash_chain_maintained(self) -> None:
        """Test that hash chain links entries."""
        log = AppendOnlyAuditLog()

        entry1 = log.log(AuditOperation.CREATE, message="First")
        entry2 = log.log(AuditOperation.UPDATE, message="Second")
        entry3 = log.log(AuditOperation.DELETE, message="Third")

        # Entry 1 should reference genesis
        assert entry1.previous_hash == "0" * 64

        # Entry 2 should reference entry 1
        assert entry2.previous_hash == entry1.entry_hash

        # Entry 3 should reference entry 2
        assert entry3.previous_hash == entry2.entry_hash

    def test_verify_integrity_valid(self) -> None:
        """Test integrity verification passes for valid log."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.CREATE)
        log.log(AuditOperation.UPDATE)
        log.log(AuditOperation.DELETE)

        assert log.verify_integrity() is True

    def test_verify_integrity_empty_log(self) -> None:
        """Test integrity verification passes for empty log."""
        log = AppendOnlyAuditLog()

        assert log.verify_integrity() is True

    def test_query_all_entries(self) -> None:
        """Test querying without filters."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.CREATE, source="user1")
        log.log(AuditOperation.UPDATE, source="user2")
        log.log(AuditOperation.DELETE, source="user1")

        result = log.query()

        assert result.total_matches == 3
        assert len(result.entries) == 3

    def test_query_by_operation(self) -> None:
        """Test querying by operation type."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.INGEST_START)
        log.log(AuditOperation.INGEST_COMPLETE)
        log.log(AuditOperation.INGEST_START)
        log.log(AuditOperation.QUERY_EXECUTE)

        result = log.query(operation=AuditOperation.INGEST_START)

        assert result.total_matches == 2
        for entry in result.entries:
            assert entry.operation == AuditOperation.INGEST_START

    def test_query_by_source(self) -> None:
        """Test querying by source substring."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.CREATE, source="pipeline_ingest")
        log.log(AuditOperation.CREATE, source="pipeline_export")
        log.log(AuditOperation.CREATE, source="cli_user")

        result = log.query(source="pipeline")

        assert result.total_matches == 2

    def test_query_by_target(self) -> None:
        """Test querying by target substring."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.CREATE, target="document.pdf")
        log.log(AuditOperation.CREATE, target="image.png")
        log.log(AuditOperation.CREATE, target="document.txt")

        result = log.query(target="document")

        assert result.total_matches == 2

    def test_query_with_limit(self) -> None:
        """Test query respects limit."""
        log = AppendOnlyAuditLog()

        for i in range(10):
            log.log(AuditOperation.CREATE, message=f"Entry {i}")

        result = log.query(limit=5)

        assert len(result.entries) == 5
        assert result.total_matches == 10
        assert result.truncated is True

    def test_query_enforces_max_limit(self) -> None:
        """Test that query enforces MAX_QUERY_RESULTS."""
        log = AppendOnlyAuditLog()

        # Request more than MAX_QUERY_RESULTS
        result = log.query(limit=MAX_QUERY_RESULTS + 1000)

        # Should be capped
        assert len(result.entries) <= MAX_QUERY_RESULTS

    def test_get_statistics(self) -> None:
        """Test getting log statistics."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.CREATE)
        log.log(AuditOperation.CREATE)
        log.log(AuditOperation.UPDATE)

        stats = log.get_statistics()

        assert stats["total_entries"] == 3
        assert stats["operation_counts"]["create"] == 2
        assert stats["operation_counts"]["update"] == 1
        assert stats["integrity_valid"] is True

    def test_clear_entries(self) -> None:
        """Test clearing all entries."""
        log = AppendOnlyAuditLog()

        log.log(AuditOperation.CREATE)
        log.log(AuditOperation.UPDATE)

        count = log.clear()

        assert count == 2
        assert len(log.get_entries()) == 0

    def test_log_with_metadata(self) -> None:
        """Test logging with metadata."""
        log = AppendOnlyAuditLog()

        entry = log.log(
            operation=AuditOperation.INGEST_COMPLETE,
            metadata={"chunks": 42, "duration_ms": 1500},
        )

        assert entry.metadata["chunks"] == 42
        assert entry.metadata["duration_ms"] == 1500


class TestAuditLogPersistence:
    """Tests for audit log file persistence."""

    def test_persist_to_jsonl(self) -> None:
        """Test persisting entries to JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            log = AppendOnlyAuditLog(log_path=log_path)

            log.log(AuditOperation.CREATE, message="Entry 1")
            log.log(AuditOperation.UPDATE, message="Entry 2")

            # Verify file was created with entries
            assert log_path.exists()

            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2

    def test_load_from_jsonl(self) -> None:
        """Test loading entries from existing JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"

            # Create log and add entries
            log1 = AppendOnlyAuditLog(log_path=log_path)
            log1.log(AuditOperation.CREATE, message="Persisted 1")
            log1.log(AuditOperation.UPDATE, message="Persisted 2")

            # Create new log instance that loads from file
            log2 = AppendOnlyAuditLog(log_path=log_path)

            entries = log2.get_entries()
            assert len(entries) == 2

    def test_export_jsonl(self) -> None:
        """Test exporting to a new JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "export.jsonl"
            log = AppendOnlyAuditLog()

            log.log(AuditOperation.CREATE)
            log.log(AuditOperation.DELETE)

            count = log.export_jsonl(export_path)

            assert count == 2
            assert export_path.exists()

            # Verify content
            lines = export_path.read_text().strip().split("\n")
            assert len(lines) == 2

            # Verify JSON parses
            for line in lines:
                data = json.loads(line)
                assert "entry_id" in data


class TestBoundsEnforcement:
    """Tests for JPL Rule #2 bounds enforcement."""

    def test_max_entries_enforced(self) -> None:
        """Test that max entries limit is enforced."""
        log = AppendOnlyAuditLog(max_entries=5)

        for i in range(10):
            log.log(AuditOperation.CREATE, message=f"Entry {i}")

        entries = log.get_entries()

        # Should only keep last 5
        assert len(entries) == 5

    def test_max_entries_capped_at_constant(self) -> None:
        """Test that max_entries is capped at MAX_AUDIT_ENTRIES."""
        log = AppendOnlyAuditLog(max_entries=MAX_AUDIT_ENTRIES * 2)

        # Internal max should be capped
        assert log._max_entries == MAX_AUDIT_ENTRIES


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_audit_log_factory(self) -> None:
        """Test create_audit_log factory function."""
        log = create_audit_log()

        assert isinstance(log, AppendOnlyAuditLog)

    def test_create_audit_log_with_path(self) -> None:
        """Test create_audit_log with custom path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "custom.jsonl"
            log = create_audit_log(log_path=log_path)

            log.log(AuditOperation.CREATE)

            assert log_path.exists()


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_bounds_constants_exist(self) -> None:
        """Test that bound constants are defined."""
        assert MAX_AUDIT_ENTRIES > 0
        assert MAX_QUERY_RESULTS > 0
        assert MAX_SOURCE_LENGTH > 0
        assert MAX_TARGET_LENGTH > 0
        assert MAX_MESSAGE_LENGTH > 0

    def test_rule_5_invalid_max_entries(self) -> None:
        """Test that invalid max_entries fails assertion."""
        with pytest.raises(AssertionError):
            AppendOnlyAuditLog(max_entries=0)

    def test_rule_7_log_returns_entry(self) -> None:
        """Test that log() always returns AuditLogEntry."""
        log = AppendOnlyAuditLog()

        result = log.log(AuditOperation.CREATE)

        assert isinstance(result, AuditLogEntry)

    def test_rule_7_query_returns_result(self) -> None:
        """Test that query() always returns AuditQueryResult."""
        log = AppendOnlyAuditLog()

        result = log.query()

        assert isinstance(result, AuditQueryResult)

    def test_rule_7_verify_integrity_returns_bool(self) -> None:
        """Test that verify_integrity() returns bool."""
        log = AppendOnlyAuditLog()

        result = log.verify_integrity()

        assert isinstance(result, bool)

    def test_rule_9_type_hints(self) -> None:
        """Test that key methods have type hints."""
        import inspect

        log = AppendOnlyAuditLog()

        # Check log method
        sig = inspect.signature(log.log)
        assert sig.return_annotation == AuditLogEntry

        # Check query method
        sig = inspect.signature(log.query)
        assert sig.return_annotation == AuditQueryResult

        # Check verify_integrity method
        sig = inspect.signature(log.verify_integrity)
        assert sig.return_annotation == bool
