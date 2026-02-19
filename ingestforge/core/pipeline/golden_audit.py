"""
Golden Dataset Hash Audit System.

Audit - Golden Dataset Hash Audit
Provides hash-based regression detection for pipeline outputs
against known golden datasets.

Follows NASA JPL Power of Ten rules.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_GOLDEN_ENTRIES = 1000
MAX_HASH_LENGTH = 64  # SHA-256 hex
MAX_REPORT_SIZE = 100_000  # 100KB
MAX_DESCRIPTION_LENGTH = 500


@dataclass(frozen=True)
class GoldenEntry:
    """
    Single golden dataset entry representing an input/expected hash pair.

    Rule #9: Complete type hints.
    """

    entry_id: str
    input_path: str
    expected_hash: str
    processor_id: str
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate entry constraints."""
        if len(self.expected_hash) > MAX_HASH_LENGTH:
            object.__setattr__(
                self, "expected_hash", self.expected_hash[:MAX_HASH_LENGTH]
            )
        if len(self.description) > MAX_DESCRIPTION_LENGTH:
            object.__setattr__(
                self, "description", self.description[:MAX_DESCRIPTION_LENGTH]
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "input_path": self.input_path,
            "expected_hash": self.expected_hash,
            "processor_id": self.processor_id,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenEntry":
        """Create from dictionary."""
        return cls(
            entry_id=data.get("entry_id", str(uuid.uuid4())),
            input_path=data["input_path"],
            expected_hash=data["expected_hash"],
            processor_id=data["processor_id"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


@dataclass
class GoldenDataset:
    """
    Collection of golden entries with metadata.

    Rule #9: Complete type hints.
    """

    dataset_id: str
    name: str
    version: str = "1.0.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    entries: List[GoldenEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entry(self, entry: GoldenEntry) -> bool:
        """
        Add entry to dataset.

        Rule #2: Enforce maximum entries.
        Rule #7: Return explicit success/failure.
        """
        if len(self.entries) >= MAX_GOLDEN_ENTRIES:
            logger.warning(f"Dataset at max capacity: {MAX_GOLDEN_ENTRIES}")
            return False
        self.entries.append(entry)
        return True

    def remove_entry(self, entry_id: str) -> bool:
        """Remove entry by ID."""
        original_len = len(self.entries)
        self.entries = [e for e in self.entries if e.entry_id != entry_id]
        return len(self.entries) < original_len

    def get_entry(self, entry_id: str) -> Optional[GoldenEntry]:
        """Get entry by ID."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "entries": [e.to_dict() for e in self.entries],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GoldenDataset":
        """Create from dictionary."""
        entries = [GoldenEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            dataset_id=data.get("dataset_id", str(uuid.uuid4())),
            name=data["name"],
            version=data.get("version", "1.0.0"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            entries=entries[:MAX_GOLDEN_ENTRIES],
            metadata=data.get("metadata", {}),
        )

    def save(self, path: Path) -> bool:
        """
        Save dataset to JSON file.

        Rule #7: Return explicit success/failure.
        """
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save dataset: {e}")
            return False

    @classmethod
    def load(cls, path: Path) -> Optional["GoldenDataset"]:
        """
        Load dataset from JSON file.

        Rule #7: Return None on failure.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return None


@dataclass(frozen=True)
class AuditResult:
    """
    Result of auditing a single golden entry.

    Rule #9: Complete type hints.
    """

    entry_id: str
    input_path: str
    expected_hash: str
    actual_hash: str
    passed: bool
    error_message: Optional[str] = None
    audited_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def hash_diff(self) -> Optional[str]:
        """Get hash difference description if failed."""
        if self.passed:
            return None
        return f"Expected: {self.expected_hash[:16]}... Got: {self.actual_hash[:16]}..."

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "entry_id": self.entry_id,
            "input_path": self.input_path,
            "expected_hash": self.expected_hash,
            "actual_hash": self.actual_hash,
            "passed": self.passed,
            "error_message": self.error_message,
            "hash_diff": self.hash_diff,
            "audited_at": self.audited_at,
        }


@dataclass
class AuditReport:
    """
    Aggregated audit results with summary.

    Rule #9: Complete type hints.
    """

    report_id: str
    dataset_name: str
    results: List[AuditResult] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_entries(self) -> int:
        """Total number of audited entries."""
        return len(self.results)

    @property
    def passed_count(self) -> int:
        """Number of passed entries."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of failed entries."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        """Pass rate as percentage."""
        if self.total_entries == 0:
            return 0.0
        return (self.passed_count / self.total_entries) * 100

    @property
    def all_passed(self) -> bool:
        """Check if all entries passed."""
        return self.failed_count == 0 and self.total_entries > 0

    def add_result(self, result: AuditResult) -> None:
        """Add audit result."""
        self.results.append(result)

    def complete(self) -> None:
        """Mark report as completed."""
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "report_id": self.report_id,
            "dataset_name": self.dataset_name,
            "summary": {
                "total_entries": self.total_entries,
                "passed_count": self.passed_count,
                "failed_count": self.failed_count,
                "pass_rate": f"{self.pass_rate:.2f}%",
                "all_passed": self.all_passed,
            },
            "results": [r.to_dict() for r in self.results],
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        result = json.dumps(self.to_dict(), indent=2)
        if len(result) > MAX_REPORT_SIZE:
            result = result[:MAX_REPORT_SIZE]
        return result

    def to_markdown(self) -> str:
        """
        Generate markdown report.

        Rule #4: Function < 60 lines.
        """
        lines = [
            f"# Audit Report: {self.dataset_name}",
            "",
            f"**Report ID**: {self.report_id}",
            f"**Started**: {self.started_at}",
            f"**Completed**: {self.completed_at or 'In Progress'}",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Entries | {self.total_entries} |",
            f"| Passed | {self.passed_count} |",
            f"| Failed | {self.failed_count} |",
            f"| Pass Rate | {self.pass_rate:.2f}% |",
            f"| Status | {'✅ ALL PASSED' if self.all_passed else '❌ REGRESSIONS DETECTED'} |",
            "",
        ]

        if self.failed_count > 0:
            lines.extend(
                [
                    "## Failures",
                    "",
                    "| Entry | Input | Expected | Actual |",
                    "|-------|-------|----------|--------|",
                ]
            )
            for r in self.results:
                if not r.passed:
                    lines.append(
                        f"| {r.entry_id[:8]}... | {Path(r.input_path).name} | "
                        f"{r.expected_hash[:12]}... | {r.actual_hash[:12]}... |"
                    )
            lines.append("")

        result = "\n".join(lines)
        if len(result) > MAX_REPORT_SIZE:
            result = result[:MAX_REPORT_SIZE]
        return result


class HashAuditor:
    """
    Executes audits against golden datasets.

    Rule #9: Complete type hints.
    """

    def __init__(self, compute_hash_func: Optional[Any] = None):
        """
        Initialize auditor.

        Args:
            compute_hash_func: Optional custom hash function.
                              Signature: (input_path: Path) -> str
        """
        self._compute_hash = compute_hash_func or self._default_hash

    def _default_hash(self, input_path: Path) -> str:
        """
        Compute SHA-256 hash of file content.

        Rule #4: Function < 60 lines.
        """
        hasher = hashlib.sha256()
        with open(input_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def audit_entry(self, entry: GoldenEntry) -> AuditResult:
        """
        Audit a single golden entry.

        Rule #7: Always return explicit result.
        """
        input_path = Path(entry.input_path)

        if not input_path.exists():
            return AuditResult(
                entry_id=entry.entry_id,
                input_path=entry.input_path,
                expected_hash=entry.expected_hash,
                actual_hash="",
                passed=False,
                error_message=f"Input file not found: {entry.input_path}",
            )

        try:
            actual_hash = self._compute_hash(input_path)
            passed = actual_hash == entry.expected_hash

            return AuditResult(
                entry_id=entry.entry_id,
                input_path=entry.input_path,
                expected_hash=entry.expected_hash,
                actual_hash=actual_hash,
                passed=passed,
            )
        except Exception as e:
            logger.error(f"Audit failed for {entry.input_path}: {e}")
            return AuditResult(
                entry_id=entry.entry_id,
                input_path=entry.input_path,
                expected_hash=entry.expected_hash,
                actual_hash="",
                passed=False,
                error_message=str(e),
            )

    def audit_dataset(self, dataset: GoldenDataset) -> AuditReport:
        """
        Audit all entries in a golden dataset.

        Rule #4: Function < 60 lines.
        Rule #7: Return explicit report.
        """
        report = AuditReport(report_id=str(uuid.uuid4()), dataset_name=dataset.name)

        for entry in dataset.entries:
            result = self.audit_entry(entry)
            report.add_result(result)

        report.complete()
        return report


class BaselineManager:
    """
    Manages golden dataset baseline updates.

    Rule #9: Complete type hints.
    """

    def __init__(self, auditor: Optional[HashAuditor] = None):
        """Initialize baseline manager."""
        self._auditor = auditor or HashAuditor()

    def create_baseline(
        self,
        name: str,
        input_paths: List[Path],
        processor_id: str = "default",
        descriptions: Optional[List[str]] = None,
    ) -> GoldenDataset:
        """
        Create a new golden dataset baseline from input files.

        Rule #4: Function < 60 lines.
        Rule #7: Return explicit dataset.
        """
        dataset = GoldenDataset(dataset_id=str(uuid.uuid4()), name=name)

        descriptions = descriptions or [""] * len(input_paths)

        for i, input_path in enumerate(input_paths[:MAX_GOLDEN_ENTRIES]):
            if not input_path.exists():
                logger.warning(f"Skipping non-existent file: {input_path}")
                continue

            try:
                hash_value = self._auditor._compute_hash(input_path)
                entry = GoldenEntry(
                    entry_id=str(uuid.uuid4()),
                    input_path=str(input_path),
                    expected_hash=hash_value,
                    processor_id=processor_id,
                    description=descriptions[i] if i < len(descriptions) else "",
                )
                dataset.add_entry(entry)
            except Exception as e:
                logger.error(f"Failed to hash {input_path}: {e}")

        return dataset

    def update_baseline(
        self, dataset: GoldenDataset, entry_ids: Optional[List[str]] = None
    ) -> int:
        """
        Update baseline hashes for specified entries (or all).

        Rule #7: Return count of updated entries.
        """
        updated = 0
        target_ids = entry_ids or [e.entry_id for e in dataset.entries]

        new_entries = []
        for entry in dataset.entries:
            if entry.entry_id in target_ids:
                input_path = Path(entry.input_path)
                if input_path.exists():
                    try:
                        new_hash = self._auditor._compute_hash(input_path)
                        new_entry = GoldenEntry(
                            entry_id=entry.entry_id,
                            input_path=entry.input_path,
                            expected_hash=new_hash,
                            processor_id=entry.processor_id,
                            description=entry.description,
                            created_at=datetime.now(timezone.utc).isoformat(),
                        )
                        new_entries.append(new_entry)
                        updated += 1
                        continue
                    except Exception as e:
                        logger.error(f"Failed to update {entry.input_path}: {e}")
            new_entries.append(entry)

        dataset.entries = new_entries
        return updated

    def add_to_baseline(
        self,
        dataset: GoldenDataset,
        input_path: Path,
        processor_id: str = "default",
        description: str = "",
    ) -> Optional[GoldenEntry]:
        """
        Add a new entry to existing baseline.

        Rule #7: Return entry or None on failure.
        """
        if len(dataset.entries) >= MAX_GOLDEN_ENTRIES:
            logger.warning("Dataset at max capacity")
            return None

        if not input_path.exists():
            logger.warning(f"File not found: {input_path}")
            return None

        try:
            hash_value = self._auditor._compute_hash(input_path)
            entry = GoldenEntry(
                entry_id=str(uuid.uuid4()),
                input_path=str(input_path),
                expected_hash=hash_value,
                processor_id=processor_id,
                description=description,
            )
            dataset.add_entry(entry)
            return entry
        except Exception as e:
            logger.error(f"Failed to add entry: {e}")
            return None


def create_auditor(compute_hash_func: Optional[Any] = None) -> HashAuditor:
    """
    Factory function to create a HashAuditor.

    Args:
        compute_hash_func: Optional custom hash function.

    Returns:
        Configured HashAuditor instance.
    """
    return HashAuditor(compute_hash_func)


def create_baseline_manager(auditor: Optional[HashAuditor] = None) -> BaselineManager:
    """
    Factory function to create a BaselineManager.

    Args:
        auditor: Optional HashAuditor to use.

    Returns:
        Configured BaselineManager instance.
    """
    return BaselineManager(auditor)
