"""
Incremental Pipeline Runner.

Skip unchanged files via hash check.
Tracks file hashes and skips processing for unchanged content.

Follows NASA JPL Power of Ten:
- Rule #1: No recursion
- Rule #2: Fixed bounds on all data structures
- Rule #4: Functions under 60 lines
- Rule #5: Assertions at entry points
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.ingest.content_hash_verifier.hasher import ContentHasher
from ingestforge.ingest.content_hash_verifier.models import HashAlgorithm

# JPL Rule #2: Fixed upper bounds
MAX_TRACKED_FILES = 100000
MAX_BATCH_SIZE = 1000
MAX_PATH_LENGTH = 4096
HASH_MANIFEST_VERSION = "1.0"

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class FileHashRecord:
    """
    Record of a file's hash for incremental tracking.

    Rule #9: Complete type hints.
    """

    path: str
    hash_value: str
    algorithm: str
    size_bytes: int
    last_processed: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.path, "path must be non-empty"
        assert self.hash_value, "hash_value must be non-empty"
        assert self.size_bytes >= 0, "size_bytes cannot be negative"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "hash_value": self.hash_value,
            "algorithm": self.algorithm,
            "size_bytes": self.size_bytes,
            "last_processed": self.last_processed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileHashRecord":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            hash_value=data["hash_value"],
            algorithm=data.get("algorithm", "sha256"),
            size_bytes=data.get("size_bytes", 0),
            last_processed=data.get("last_processed", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class IncrementalCheckResult:
    """
    Result of checking if a file needs processing.

    Rule #9: Complete type hints.
    """

    path: str
    needs_processing: bool
    reason: str  # "new", "modified", "unchanged", "error"
    current_hash: Optional[str] = None
    stored_hash: Optional[str] = None

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.path, "path must be non-empty"
        assert self.reason in (
            "new",
            "modified",
            "unchanged",
            "error",
        ), f"invalid reason: {self.reason}"


@dataclass
class IncrementalRunReport:
    """
    Report from incremental run.

    Rule #9: Complete type hints.
    """

    total_files: int
    processed_files: int
    skipped_files: int
    new_files: int
    modified_files: int
    error_files: int
    time_ms: float

    def __post_init__(self) -> None:
        """JPL Rule #5: Assert preconditions."""
        assert self.total_files >= 0, "total_files cannot be negative"
        assert self.processed_files >= 0, "processed_files cannot be negative"
        assert self.skipped_files >= 0, "skipped_files cannot be negative"

    @property
    def skip_rate(self) -> float:
        """Calculate skip rate percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.skipped_files / self.total_files) * 100


# =============================================================================
# Hash Manifest Store
# =============================================================================


class HashManifest:
    """
    Persistent store for file hash records.

    Rule #2: Bounded by MAX_TRACKED_FILES.
    Rule #9: Complete type hints.
    """

    def __init__(self, manifest_path: Path) -> None:
        """
        Initialize hash manifest.

        Args:
            manifest_path: Path to the manifest JSON file.

        Rule #5: Assert preconditions.
        """
        assert manifest_path is not None, "manifest_path cannot be None"

        self._path = manifest_path
        self._records: Dict[str, FileHashRecord] = {}
        self._dirty = False

    @property
    def record_count(self) -> int:
        """Return number of tracked files."""
        return len(self._records)

    def load(self) -> bool:
        """
        Load manifest from disk.

        Rule #4: Under 60 lines.
        Rule #7: Check return values.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not self._path.exists():
            logger.debug(f"No manifest at {self._path}, starting fresh")
            return True

        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)

            version = data.get("version", "0.0")
            if version != HASH_MANIFEST_VERSION:
                logger.warning(f"Manifest version mismatch: {version}")

            records = data.get("records", {})
            for path, record_data in list(records.items())[:MAX_TRACKED_FILES]:
                self._records[path] = FileHashRecord.from_dict(record_data)

            logger.info(f"Loaded {self.record_count} records from manifest")
            return True

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse manifest: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return False

    def save(self) -> bool:
        """
        Save manifest to disk.

        Rule #4: Under 60 lines.
        Rule #7: Check return values.

        Returns:
            True if saved successfully, False otherwise.
        """
        if not self._dirty:
            return True

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": HASH_MANIFEST_VERSION,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "record_count": self.record_count,
                "records": {
                    path: record.to_dict() for path, record in self._records.items()
                },
            }

            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            self._dirty = False
            logger.info(f"Saved {self.record_count} records to manifest")
            return True

        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")
            return False

    def get(self, path: str) -> Optional[FileHashRecord]:
        """
        Get hash record for a file.

        Args:
            path: File path to look up.

        Returns:
            FileHashRecord if found, None otherwise.
        """
        return self._records.get(path)

    def set(self, record: FileHashRecord) -> bool:
        """
        Store or update a hash record.

        Rule #2: Enforces MAX_TRACKED_FILES.

        Args:
            record: The hash record to store.

        Returns:
            True if stored, False if limit reached.
        """
        if record.path not in self._records:
            if self.record_count >= MAX_TRACKED_FILES:
                self._evict_oldest()

        self._records[record.path] = record
        self._dirty = True
        return True

    def remove(self, path: str) -> bool:
        """
        Remove a hash record.

        Args:
            path: File path to remove.

        Returns:
            True if removed, False if not found.
        """
        if path in self._records:
            del self._records[path]
            self._dirty = True
            return True
        return False

    def _evict_oldest(self) -> None:
        """
        Evict oldest records to make room.

        Rule #4: Under 60 lines.
        """
        if not self._records:
            return

        # Sort by last_processed and remove oldest 10%
        sorted_paths = sorted(
            self._records.keys(),
            key=lambda p: self._records[p].last_processed,
        )

        to_remove = max(1, int(MAX_TRACKED_FILES * 0.1))
        for path in sorted_paths[:to_remove]:
            del self._records[path]

        logger.info(f"Evicted {to_remove} old records from manifest")


# =============================================================================
# Incremental Runner
# =============================================================================


class IncrementalRunner:
    """
    Pipeline runner that skips unchanged files.

    Skip unchanged files via hash check.

    Rule #2: Fixed bounds on batch sizes.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        manifest_path: Path,
        algorithm: HashAlgorithm = HashAlgorithm.SHA256,
    ) -> None:
        """
        Initialize incremental runner.

        Args:
            manifest_path: Path to hash manifest file.
            algorithm: Hash algorithm to use.

        Rule #5: Assert preconditions.
        """
        assert manifest_path is not None, "manifest_path cannot be None"

        self._manifest = HashManifest(manifest_path)
        self._hasher = ContentHasher(algorithms=[algorithm])
        self._algorithm = algorithm

    def initialize(self) -> bool:
        """
        Load the hash manifest.

        Returns:
            True if initialized successfully.
        """
        return self._manifest.load()

    def finalize(self) -> bool:
        """
        Save the hash manifest.

        Returns:
            True if saved successfully.
        """
        return self._manifest.save()

    def check_file(self, file_path: Path) -> IncrementalCheckResult:
        """
        Check if a file needs processing.

        Rule #4: Under 60 lines.

        Args:
            file_path: Path to the file to check.

        Returns:
            IncrementalCheckResult indicating if processing is needed.
        """
        path_str = str(file_path.resolve())

        if len(path_str) > MAX_PATH_LENGTH:
            return IncrementalCheckResult(
                path=path_str[:100],
                needs_processing=False,
                reason="error",
            )

        if not file_path.exists():
            return IncrementalCheckResult(
                path=path_str,
                needs_processing=False,
                reason="error",
            )

        # Compute current hash
        try:
            multi_hash = self._hasher.hash_file(file_path)
            current_hash = multi_hash.hashes[self._algorithm].hash_value
        except Exception as e:
            logger.warning(f"Failed to hash {file_path}: {e}")
            return IncrementalCheckResult(
                path=path_str,
                needs_processing=True,
                reason="error",
            )

        # Check stored hash
        stored_record = self._manifest.get(path_str)

        if stored_record is None:
            return IncrementalCheckResult(
                path=path_str,
                needs_processing=True,
                reason="new",
                current_hash=current_hash,
            )

        if stored_record.hash_value != current_hash:
            return IncrementalCheckResult(
                path=path_str,
                needs_processing=True,
                reason="modified",
                current_hash=current_hash,
                stored_hash=stored_record.hash_value,
            )

        return IncrementalCheckResult(
            path=path_str,
            needs_processing=False,
            reason="unchanged",
            current_hash=current_hash,
            stored_hash=stored_record.hash_value,
        )

    def mark_processed(self, file_path: Path, metadata: Optional[Dict] = None) -> bool:
        """
        Mark a file as processed and update its hash.

        Rule #4: Under 60 lines.

        Args:
            file_path: Path to the processed file.
            metadata: Optional metadata to store.

        Returns:
            True if marked successfully.
        """
        path_str = str(file_path.resolve())

        if not file_path.exists():
            return False

        try:
            multi_hash = self._hasher.hash_file(file_path)
            hash_value = multi_hash.hashes[self._algorithm].hash_value
            size_bytes = multi_hash.content_size

            record = FileHashRecord(
                path=path_str,
                hash_value=hash_value,
                algorithm=self._algorithm.value,
                size_bytes=size_bytes,
                last_processed=datetime.now(timezone.utc).isoformat(),
                metadata=metadata or {},
            )

            return self._manifest.set(record)

        except Exception as e:
            logger.error(f"Failed to mark {file_path} as processed: {e}")
            return False

    def filter_files(
        self,
        file_paths: List[Path],
    ) -> Tuple[List[Path], IncrementalRunReport]:
        """
        Filter files to only those needing processing.

        Rule #2: Bounded by MAX_BATCH_SIZE per iteration.
        Rule #4: Under 60 lines.

        Args:
            file_paths: List of file paths to check.

        Returns:
            Tuple of (files to process, report).
        """
        start_time = time.perf_counter()

        to_process: List[Path] = []
        new_count = 0
        modified_count = 0
        skipped_count = 0
        error_count = 0

        # Process in bounded batches
        total = len(file_paths)
        for i in range(0, total, MAX_BATCH_SIZE):
            batch = file_paths[i : i + MAX_BATCH_SIZE]

            for file_path in batch:
                result = self.check_file(file_path)

                if result.needs_processing:
                    to_process.append(file_path)
                    if result.reason == "new":
                        new_count += 1
                    elif result.reason == "modified":
                        modified_count += 1
                    elif result.reason == "error":
                        error_count += 1
                else:
                    skipped_count += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        report = IncrementalRunReport(
            total_files=total,
            processed_files=len(to_process),
            skipped_files=skipped_count,
            new_files=new_count,
            modified_files=modified_count,
            error_files=error_count,
            time_ms=elapsed_ms,
        )

        logger.info(
            f"Incremental filter: {len(to_process)}/{total} files need processing "
            f"(skipped {skipped_count}, new {new_count}, modified {modified_count})"
        )

        return to_process, report

    def get_stats(self) -> Dict[str, Any]:
        """
        Get runner statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "tracked_files": self._manifest.record_count,
            "max_tracked_files": MAX_TRACKED_FILES,
            "algorithm": self._algorithm.value,
        }


# =============================================================================
# Convenience Functions
# =============================================================================


def create_incremental_runner(
    manifest_path: Optional[Path] = None,
    algorithm: HashAlgorithm = HashAlgorithm.SHA256,
) -> IncrementalRunner:
    """
    Create an incremental runner.

    Args:
        manifest_path: Path to manifest file. Defaults to .ingestforge/hash_manifest.json
        algorithm: Hash algorithm to use.

    Returns:
        Configured IncrementalRunner.
    """
    if manifest_path is None:
        manifest_path = Path.cwd() / ".ingestforge" / "hash_manifest.json"

    return IncrementalRunner(manifest_path, algorithm)


def filter_unchanged_files(
    file_paths: List[Path],
    manifest_path: Optional[Path] = None,
) -> List[Path]:
    """
    Convenience function to filter out unchanged files.

    Args:
        file_paths: Files to filter.
        manifest_path: Path to manifest.

    Returns:
        List of files that need processing.
    """
    runner = create_incremental_runner(manifest_path)
    runner.initialize()

    to_process, _ = runner.filter_files(file_paths)

    runner.finalize()

    return to_process
