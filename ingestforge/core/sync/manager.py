"""
Sync manager for incremental file synchronization.

Provides SyncManager for intelligent sync of files to IngestForge.
"""

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.core.sync.models import FileState, SyncReport
from ingestforge.core.sync.store import SyncStateStore
from ingestforge.storage.base import ChunkRepository

logger = get_logger(__name__)


class SyncManager:
    """
    Manages incremental synchronization of files to IngestForge.

    Tracks file content hashes to detect:
    - New files (not in state store)
    - Changed files (hash mismatch)
    - Deleted files (in state store but not on disk)
    """

    def __init__(
        self,
        config: Config,
        repository: ChunkRepository,
        state_path: Optional[Path] = None,
    ):
        """
        Initialize sync manager.

        Args:
            config: IngestForge configuration
            repository: Chunk repository for storage operations
            state_path: Path to sync state file (default: .data/sync_state.json)
        """
        self.config = config
        self.repository = repository
        self.state_path = state_path or config.data_path / "sync_state.json"
        self.state_store = SyncStateStore(self.state_path)

        # Pipeline is lazy-loaded
        self._pipeline = None

    @property
    def pipeline(self) -> Any:
        """Lazy-load pipeline."""
        if self._pipeline is None:
            from ingestforge.core.pipeline import Pipeline

            self._pipeline = Pipeline(self.config, self.repository)
        return self._pipeline

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file content."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate document ID from file path."""
        path_str = str(file_path.resolve())
        return hashlib.sha256(path_str.encode()).hexdigest()[:16]

    def _collect_files(
        self,
        directory: Path,
        patterns: List[str],
        recursive: bool = True,
    ) -> List[Path]:
        """Collect files matching patterns."""

        files = []
        if recursive:
            for pattern in patterns:
                files.extend(directory.rglob(pattern))
        else:
            for pattern in patterns:
                files.extend(directory.glob(pattern))

        # Filter to only files
        return [f for f in files if f.is_file()]

    def analyze_changes(
        self,
        directory: Path,
        patterns: List[str],
        recursive: bool = True,
    ) -> Tuple[List[Path], List[Path], List[str]]:
        """
        Analyze directory for changes without making modifications.

        Returns:
            Tuple of (new_files, changed_files, deleted_paths)
        """
        # Collect current files
        current_files = self._collect_files(directory, patterns, recursive)
        current_paths = {str(f.resolve()) for f in current_files}

        # Get tracked paths for this directory pattern
        tracked_paths = self.state_store.get_all_paths()

        new_files = []
        changed_files = []
        deleted_paths = []

        # Check each current file
        for file_path in current_files:
            path_str = str(file_path.resolve())
            state = self.state_store.get(path_str)

            if state is None:
                # New file
                new_files.append(file_path)
            else:
                # Check if content changed
                current_hash = self._compute_hash(file_path)
                if current_hash != state.content_hash:
                    changed_files.append(file_path)

        # Check for deleted files (only within this directory)
        dir_prefix = str(directory.resolve())
        for tracked_path in tracked_paths:
            if (
                tracked_path.startswith(dir_prefix)
                and tracked_path not in current_paths
            ):
                deleted_paths.append(tracked_path)

        return new_files, changed_files, deleted_paths

    def _create_file_state(self, file_path: Path, chunk_count: int) -> FileState:
        """
        Create FileState from file path and processing result.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            file_path: File path
            chunk_count: Number of chunks created

        Returns:
            FileState object
        """
        path_str = str(file_path.resolve())
        return FileState(
            path=path_str,
            content_hash=self._compute_hash(file_path),
            size=file_path.stat().st_size,
            modified_time=file_path.stat().st_mtime,
            document_id=self._generate_document_id(file_path),
            chunk_count=chunk_count,
            last_synced=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

    def _process_deleted_files(
        self,
        deleted_paths: List[str],
        report: SyncReport,
        report_progress: Optional[Callable],
    ) -> None:
        """
        Process deleted files.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            deleted_paths: Paths to deleted files
            report: Sync report to update (mutated)
            report_progress: Progress callback function
        """
        for path_str in deleted_paths:
            try:
                state = self.state_store.get(path_str)
                if not state:
                    continue

                # Delete chunks from repository
                deleted_count = self.repository.delete_document(state.document_id)
                logger.info(
                    f"Removed {deleted_count} chunks for deleted file: {path_str}"
                )

                # Remove from state store
                self.state_store.remove(path_str)

                report.removed += 1
                report.removed_files.append(path_str)
                report_progress("remove", f"Removed: {Path(path_str).name}")

            except Exception as e:
                logger.error(f"Failed to remove deleted file {path_str}: {e}")
                report.errors += 1
                report.error_files.append((path_str, str(e)))

    def _process_changed_file(
        self, file_path: Path, report: SyncReport, report_progress: Optional[Callable]
    ) -> None:
        """
        Process single changed file.

        Rule #1: Early return for errors
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: File to process
            report: Sync report to update (mutated)
            report_progress: Progress callback function
        """
        path_str = str(file_path.resolve())

        try:
            # Delete old chunks if state exists
            state = self.state_store.get(path_str)
            if state:
                deleted_count = self.repository.delete_document(state.document_id)
                logger.debug(
                    f"Removed {deleted_count} old chunks for update: {file_path.name}"
                )

            # Re-ingest
            result = self.pipeline.process_file(file_path)
            if not result.success:
                raise Exception(result.error or "Processing failed")

            # Update state
            new_state = self._create_file_state(file_path, result.chunks_created)
            self.state_store.set(new_state)

            report.updated += 1
            report.updated_files.append(path_str)
            report_progress("update", f"Updated: {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to update file {file_path}: {e}")
            report.errors += 1
            report.error_files.append((path_str, str(e)))

    def _process_new_file(
        self, file_path: Path, report: SyncReport, report_progress: Optional[Callable]
    ) -> None:
        """
        Process single new file.

        Rule #1: Early return for errors
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: File to process
            report: Sync report to update (mutated)
            report_progress: Progress callback function
        """
        path_str = str(file_path.resolve())

        try:
            result = self.pipeline.process_file(file_path)
            if not result.success:
                raise Exception(result.error or "Processing failed")

            # Store state
            new_state = self._create_file_state(file_path, result.chunks_created)
            self.state_store.set(new_state)

            report.added += 1
            report.added_files.append(path_str)
            report_progress("add", f"Added: {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to add file {file_path}: {e}")
            report.errors += 1
            report.error_files.append((path_str, str(e)))

    def sync_directory(
        self,
        directory: Path,
        patterns: List[str],
        recursive: bool = True,
        remove_deleted: bool = True,
        progress_callback: Any = None,
    ) -> SyncReport:
        """
        Synchronize a directory with the IngestForge database.

        Rule #4: Reduced from 64 → 49 lines (shortened docstring)
        """
        report = SyncReport()
        report.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time = datetime.now()

        # Analyze changes
        new_files, changed_files, deleted_paths = self.analyze_changes(
            directory, patterns, recursive
        )

        total_operations = len(new_files) + len(changed_files) + len(deleted_paths)
        current_op = 0

        # Helper to report progress
        def report_progress(stage: str, message: str) -> None:
            nonlocal current_op
            current_op += 1
            if progress_callback:
                progress_callback(stage, current_op, total_operations, message)

        # Process changes using helpers
        if remove_deleted:
            self._process_deleted_files(deleted_paths, report, report_progress)

        for file_path in changed_files:
            self._process_changed_file(file_path, report, report_progress)

        for file_path in new_files:
            self._process_new_file(file_path, report, report_progress)

        # Calculate skipped files
        total_files = len(self._collect_files(directory, patterns, recursive))
        report.skipped = total_files - report.added - report.updated

        # Finalize report
        report.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report.duration_seconds = (datetime.now() - start_time).total_seconds()

        return report

    def sync_file(self, file_path: Path, force: bool = False) -> SyncReport:
        """
        Sync a single file.

        Args:
            file_path: File to sync
            force: Force re-ingestion even if unchanged

        Returns:
            SyncReport
        """
        report = SyncReport()
        report.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        start_time = datetime.now()

        path_str = str(file_path.resolve())

        if not file_path.exists():
            return self._handle_file_deletion(path_str, report)

        state = self.state_store.get(path_str)
        current_hash = self._compute_hash(file_path)

        if state and not force and state.content_hash == current_hash:
            report.skipped = 1
            return report

        try:
            self._ingest_and_update_state(
                file_path, path_str, state, current_hash, report
            )
        except Exception as e:
            report.errors = 1
            report.error_files.append((path_str, str(e)))

        self._finalize_sync_report(report, start_time)
        return report

    def _handle_file_deletion(self, path_str: str, report: SyncReport) -> SyncReport:
        """Handle deletion of a tracked file."""
        state = self.state_store.get(path_str)
        if state:
            self.repository.delete_document(state.document_id)
            self.state_store.remove(path_str)
            report.removed = 1
            report.removed_files.append(path_str)
        return report

    def _ingest_and_update_state(
        self,
        file_path: Path,
        path_str: str,
        state: Any,
        current_hash: str,
        report: SyncReport,
    ):
        """Ingest file and update state tracking."""
        if state:
            self.repository.delete_document(state.document_id)

        result = self.pipeline.process_file(file_path)

        if not result.success:
            raise Exception(result.error or "Processing failed")

        new_state = FileState(
            path=path_str,
            content_hash=current_hash,
            size=file_path.stat().st_size,
            modified_time=file_path.stat().st_mtime,
            document_id=self._generate_document_id(file_path),
            chunk_count=result.chunks_created,
            last_synced=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self.state_store.set(new_state)

        if state:
            report.updated = 1
            report.updated_files.append(path_str)
        else:
            report.added = 1
            report.added_files.append(path_str)

    def _finalize_sync_report(self, report: SyncReport, start_time: Any) -> None:
        """Finalize sync report with completion timestamp."""
        report.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        report.duration_seconds = (datetime.now() - start_time).total_seconds()

    def get_stats(self) -> Dict[str, Any]:
        """Get sync statistics."""
        all_paths = self.state_store.get_all_paths()

        total_chunks = 0
        total_size = 0
        by_extension: Dict[str, int] = {}

        for path_str in all_paths:
            state = self.state_store.get(path_str)
            if state:
                total_chunks += state.chunk_count
                total_size += state.size
                ext = Path(path_str).suffix.lower()
                by_extension[ext] = by_extension.get(ext, 0) + 1

        return {
            "tracked_files": len(all_paths),
            "total_chunks": total_chunks,
            "total_size_bytes": total_size,
            "by_extension": by_extension,
            "state_file": str(self.state_path),
        }

    def reset(self) -> None:
        """Reset sync state (forces full re-sync on next run)."""
        self.state_store.clear()
        logger.info("Sync state reset - next sync will re-process all files")
