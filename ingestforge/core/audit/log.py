"""Append-Only Audit Log Implementation.

Append-Only Audit Log
Epic: EP-19 (Governance & Compliance)

Provides an immutable, append-only audit log system for recording
all significant operations with hash chain integrity.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_AUDIT_ENTRIES, MAX_QUERY_RESULTS)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_AUDIT_ENTRIES = 100_000  # Maximum entries in memory
MAX_QUERY_RESULTS = 1000  # Maximum query results returned
MAX_METADATA_SIZE = 10_000  # Maximum metadata JSON size (chars)
MAX_SOURCE_LENGTH = 256  # Maximum source string length
MAX_TARGET_LENGTH = 256  # Maximum target string length
MAX_MESSAGE_LENGTH = 1000  # Maximum message string length


class AuditOperation(Enum):
    """Audit operation types."""

    # Ingestion operations
    INGEST_START = "ingest_start"
    INGEST_COMPLETE = "ingest_complete"
    INGEST_FAILED = "ingest_failed"

    # Query operations
    QUERY_EXECUTE = "query_execute"
    QUERY_COMPLETE = "query_complete"

    # Data operations
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"

    # Pipeline operations
    PIPELINE_START = "pipeline_start"
    PIPELINE_STAGE = "pipeline_stage"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_FAILED = "pipeline_failed"

    # Configuration operations
    CONFIG_CHANGE = "config_change"

    # System operations
    STARTUP = "startup"
    SHUTDOWN = "shutdown"
    ERROR = "error"

    # Custom operation
    CUSTOM = "custom"


@dataclass(frozen=True)
class AuditLogEntry:
    """Immutable audit log entry.

    Rule #9: Complete type hints.
    """

    entry_id: str
    timestamp: str
    operation: AuditOperation
    source: str
    target: str
    message: str
    previous_hash: str
    entry_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and enforce bounds.

        Rule #2: Enforce maximum lengths.
        Rule #5: Assert preconditions.
        """
        assert self.entry_id, "entry_id cannot be empty"
        assert self.timestamp, "timestamp cannot be empty"
        assert self.operation is not None, "operation cannot be None"

        # Truncate long strings (frozen dataclass requires object.__setattr__)
        if len(self.source) > MAX_SOURCE_LENGTH:
            object.__setattr__(self, "source", self.source[:MAX_SOURCE_LENGTH])
        if len(self.target) > MAX_TARGET_LENGTH:
            object.__setattr__(self, "target", self.target[:MAX_TARGET_LENGTH])
        if len(self.message) > MAX_MESSAGE_LENGTH:
            object.__setattr__(self, "message", self.message[:MAX_MESSAGE_LENGTH])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "operation": self.operation.value,
            "source": self.source,
            "target": self.target,
            "message": self.message,
            "previous_hash": self.previous_hash,
            "entry_hash": self.entry_hash,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string.

        Returns:
            JSON string.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuditLogEntry":
        """Create from dictionary.

        Args:
            data: Dictionary with entry data.

        Returns:
            AuditLogEntry instance.
        """
        operation_str = data.get("operation", "custom")
        try:
            operation = AuditOperation(operation_str)
        except ValueError:
            operation = AuditOperation.CUSTOM

        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            operation=operation,
            source=data.get("source", ""),
            target=data.get("target", ""),
            message=data.get("message", ""),
            previous_hash=data.get("previous_hash", ""),
            entry_hash=data.get("entry_hash", ""),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AuditLogEntry":
        """Create from JSON string.

        Args:
            json_str: JSON string.

        Returns:
            AuditLogEntry instance.
        """
        return cls.from_dict(json.loads(json_str))


@dataclass
class AuditQueryResult:
    """Result of audit log query.

    Rule #9: Complete type hints.
    """

    entries: List[AuditLogEntry] = field(default_factory=list)
    total_matches: int = 0
    truncated: bool = False
    query_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "entries": [e.to_dict() for e in self.entries],
            "total_matches": self.total_matches,
            "truncated": self.truncated,
            "query_time_ms": self.query_time_ms,
        }


class AppendOnlyAuditLog:
    """Append-only audit log with hash chain integrity.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        max_entries: int = MAX_AUDIT_ENTRIES,
    ) -> None:
        """Initialize audit log.

        Args:
            log_path: Optional path to persist log as JSONL file.
            max_entries: Maximum entries to keep in memory.

        Rule #5: Assert preconditions.
        """
        assert max_entries > 0, "max_entries must be positive"

        self._entries: List[AuditLogEntry] = []
        self._log_path = log_path
        self._max_entries = min(max_entries, MAX_AUDIT_ENTRIES)
        self._lock = threading.Lock()
        self._last_hash = "0" * 64  # Genesis hash

        # Load existing entries if log file exists
        if log_path and log_path.exists():
            self._load_from_file()

    def _prepare_metadata(self, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Prepare and truncate metadata if needed.

        Args:
            metadata: Raw metadata dictionary.

        Returns:
            Prepared metadata, truncated if over size limit.

        Rule #2: Enforce MAX_METADATA_SIZE.
        Rule #4: Helper to keep log() < 60 lines.
        """
        meta = metadata or {}
        meta_json = json.dumps(meta)
        if len(meta_json) > MAX_METADATA_SIZE:
            return {"_truncated": True, "_original_size": len(meta_json)}
        return meta

    def _create_entry(
        self,
        operation: AuditOperation,
        source: str,
        target: str,
        message: str,
        metadata: Dict[str, Any],
    ) -> AuditLogEntry:
        """Create a new audit log entry with hash chain.

        Args:
            operation: Type of operation.
            source: Source of operation.
            target: Target of operation.
            message: Human-readable message.
            metadata: Entry metadata.

        Returns:
            Created AuditLogEntry.

        Rule #4: Helper to keep log() < 60 lines.
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        previous_hash = self._last_hash

        entry_hash = self._compute_hash(
            entry_id,
            timestamp,
            operation.value,
            source,
            target,
            message,
            previous_hash,
            metadata,
        )

        return AuditLogEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            operation=operation,
            source=source,
            target=target,
            message=message,
            previous_hash=previous_hash,
            entry_hash=entry_hash,
            metadata=metadata,
        )

    def log(
        self,
        operation: AuditOperation,
        source: str = "",
        target: str = "",
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditLogEntry:
        """Log an audit entry.

        Args:
            operation: Type of operation being logged.
            source: Source of the operation (e.g., user, module).
            target: Target of the operation (e.g., file, entity).
            message: Human-readable message.
            metadata: Additional metadata.

        Returns:
            Created AuditLogEntry.

        Rule #4: Function < 60 lines (uses helper methods).
        Rule #7: Return explicit result.
        """
        with self._lock:
            meta = self._prepare_metadata(metadata)
            entry = self._create_entry(operation, source, target, message, meta)

            self._entries.append(entry)
            self._last_hash = entry.entry_hash

            # Enforce bounds (remove oldest entries)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries :]

            # Persist if path configured
            if self._log_path:
                self._append_to_file(entry)

            return entry

    def _compute_hash(
        self,
        entry_id: str,
        timestamp: str,
        operation: str,
        source: str,
        target: str,
        message: str,
        previous_hash: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Compute SHA-256 hash for entry.

        Args:
            entry_id: Entry identifier.
            timestamp: Timestamp string.
            operation: Operation value.
            source: Source string.
            target: Target string.
            message: Message string.
            previous_hash: Hash of previous entry.
            metadata: Entry metadata.

        Returns:
            Hex-encoded SHA-256 hash.

        Rule #4: Function < 60 lines.
        """
        content = f"{entry_id}|{timestamp}|{operation}|{source}|{target}|{message}|{previous_hash}|{json.dumps(metadata, sort_keys=True)}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def query(
        self,
        operation: Optional[AuditOperation] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = MAX_QUERY_RESULTS,
    ) -> AuditQueryResult:
        """Query audit log entries.

        Args:
            operation: Filter by operation type.
            source: Filter by source (substring match).
            target: Filter by target (substring match).
            start_time: Filter entries after this ISO timestamp.
            end_time: Filter entries before this ISO timestamp.
            limit: Maximum entries to return.

        Returns:
            AuditQueryResult with matching entries.

        Rule #2: Enforce MAX_QUERY_RESULTS.
        Rule #4: Function < 60 lines.
        """
        import time

        start = time.perf_counter()

        limit = min(limit, MAX_QUERY_RESULTS)
        matches: List[AuditLogEntry] = []

        with self._lock:
            for entry in self._entries:
                # Apply filters
                if operation and entry.operation != operation:
                    continue
                if source and source.lower() not in entry.source.lower():
                    continue
                if target and target.lower() not in entry.target.lower():
                    continue
                if start_time and entry.timestamp < start_time:
                    continue
                if end_time and entry.timestamp > end_time:
                    continue

                matches.append(entry)

        total_matches = len(matches)
        truncated = total_matches > limit

        elapsed_ms = (time.perf_counter() - start) * 1000

        return AuditQueryResult(
            entries=matches[:limit],
            total_matches=total_matches,
            truncated=truncated,
            query_time_ms=elapsed_ms,
        )

    def verify_integrity(self) -> bool:
        """Verify hash chain integrity.

        Returns:
            True if all hashes are valid, False if chain is broken.

        Rule #7: Return explicit result.
        """
        with self._lock:
            if not self._entries:
                return True

            expected_prev = "0" * 64  # Genesis

            for entry in self._entries:
                # Check previous hash link
                if entry.previous_hash != expected_prev:
                    logger.warning(
                        f"Hash chain broken at entry {entry.entry_id}: "
                        f"expected prev={expected_prev[:16]}..., "
                        f"got prev={entry.previous_hash[:16]}..."
                    )
                    return False

                # Recompute and verify entry hash
                computed = self._compute_hash(
                    entry.entry_id,
                    entry.timestamp,
                    entry.operation.value,
                    entry.source,
                    entry.target,
                    entry.message,
                    entry.previous_hash,
                    entry.metadata,
                )

                if entry.entry_hash != computed:
                    logger.warning(
                        f"Entry hash mismatch at {entry.entry_id}: "
                        f"stored={entry.entry_hash[:16]}..., "
                        f"computed={computed[:16]}..."
                    )
                    return False

                expected_prev = entry.entry_hash

            return True

    def get_entries(self, limit: int = MAX_QUERY_RESULTS) -> List[AuditLogEntry]:
        """Get recent entries.

        Args:
            limit: Maximum entries to return.

        Returns:
            List of recent entries (newest first).

        Rule #2: Enforce limit bounds.
        """
        limit = min(limit, MAX_QUERY_RESULTS)
        with self._lock:
            return list(reversed(self._entries[-limit:]))

    def get_statistics(self) -> Dict[str, Any]:
        """Get audit log statistics.

        Returns:
            Dictionary with statistics.
        """
        with self._lock:
            op_counts: Dict[str, int] = {}
            for entry in self._entries:
                op_name = entry.operation.value
                op_counts[op_name] = op_counts.get(op_name, 0) + 1

            return {
                "total_entries": len(self._entries),
                "max_entries": self._max_entries,
                "entries_remaining": self._max_entries - len(self._entries),
                "operation_counts": op_counts,
                "integrity_valid": self.verify_integrity(),
                "log_path": str(self._log_path) if self._log_path else None,
            }

    def _append_to_file(self, entry: AuditLogEntry) -> None:
        """Append entry to JSONL file.

        Args:
            entry: Entry to append.
        """
        if not self._log_path:
            return

        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(entry.to_json() + "\n")
        except Exception as e:
            logger.error(f"Failed to append to audit log: {e}")

    def _load_from_file(self) -> None:
        """Load entries from JSONL file.

        Rule #4: Function < 60 lines.
        """
        if not self._log_path or not self._log_path.exists():
            return

        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    entry = AuditLogEntry.from_json(line)
                    self._entries.append(entry)
                    self._last_hash = entry.entry_hash

                    # Enforce bounds
                    if len(self._entries) >= self._max_entries:
                        break

            logger.info(
                f"Loaded {len(self._entries)} audit entries from {self._log_path}"
            )
        except Exception as e:
            logger.error(f"Failed to load audit log: {e}")

    def export_jsonl(self, output_path: Path) -> int:
        """Export entries to JSONL file.

        Args:
            output_path: Path to write JSONL file.

        Returns:
            Number of entries exported.

        Rule #7: Return explicit count.
        """
        with self._lock:
            try:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    for entry in self._entries:
                        f.write(entry.to_json() + "\n")
                return len(self._entries)
            except Exception as e:
                logger.error(f"Failed to export audit log: {e}")
                return 0

    def clear(self) -> int:
        """Clear all entries from memory (file persists).

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._entries)
            self._entries = []
            self._last_hash = "0" * 64
            return count


# Singleton instance for convenience
_default_log: Optional[AppendOnlyAuditLog] = None
_default_lock = threading.Lock()


def create_audit_log(
    log_path: Optional[Path] = None,
    max_entries: int = MAX_AUDIT_ENTRIES,
) -> AppendOnlyAuditLog:
    """Factory function to create an audit log.

    Args:
        log_path: Optional path to persist log.
        max_entries: Maximum entries in memory.

    Returns:
        Configured AppendOnlyAuditLog instance.
    """
    return AppendOnlyAuditLog(log_path=log_path, max_entries=max_entries)


def get_default_audit_log(
    log_path: Optional[Path] = None,
) -> AppendOnlyAuditLog:
    """Get or create the default audit log instance.

    Args:
        log_path: Optional path for persistence.

    Returns:
        Default AppendOnlyAuditLog instance.
    """
    global _default_log

    with _default_lock:
        if _default_log is None:
            _default_log = create_audit_log(log_path=log_path)
        return _default_log


def log_operation(
    operation: AuditOperation,
    source: str = "",
    target: str = "",
    message: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> AuditLogEntry:
    """Convenience function to log to default audit log.

    Args:
        operation: Type of operation.
        source: Source of operation.
        target: Target of operation.
        message: Human-readable message.
        metadata: Additional metadata.

    Returns:
        Created AuditLogEntry.
    """
    return get_default_audit_log().log(
        operation=operation,
        source=source,
        target=target,
        message=message,
        metadata=metadata,
    )
