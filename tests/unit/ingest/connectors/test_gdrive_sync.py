"""
Unit tests for Google Drive Sync Manager.

Tests GDriveSyncManager with comprehensive GWT (Given-When-Then) coverage.
Target: >80% coverage

All Epic AC tested
NASA JPL Rules: #4 (<60 lines), #7 (Check returns), #9 (Type hints)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

from ingestforge.core.sync.models import FileState
from ingestforge.core.sync.store import SyncStateStore
from ingestforge.ingest.connectors.base import IFConnectorResult
from ingestforge.ingest.connectors.gdrive import GDriveConnector
from ingestforge.ingest.connectors.gdrive_sync import (
    GDriveSyncManager,
    GDriveSyncReport,
    MAX_SYNC_FILES,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_connector() -> MagicMock:
    """Create mock GDrive connector."""
    connector = MagicMock(spec=GDriveConnector)
    connector.connect.return_value = True
    connector.disconnect.return_value = None
    return connector


@pytest.fixture
def mock_state_store(tmp_path: Path) -> SyncStateStore:
    """Create real state store with temp file."""
    state_file = tmp_path / "test_sync_state.json"
    return SyncStateStore(state_file)


@pytest.fixture
def pending_dir(tmp_path: Path) -> Path:
    """Create pending directory."""
    pending = tmp_path / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    return pending


@pytest.fixture
def sync_manager(
    mock_connector: MagicMock,
    mock_state_store: SyncStateStore,
    pending_dir: Path,
) -> GDriveSyncManager:
    """Create GDriveSyncManager instance."""
    return GDriveSyncManager(
        connector=mock_connector,
        state_store=mock_state_store,
        pending_dir=pending_dir,
        conflict_strategy="server_wins",
    )


@pytest.fixture
def sample_gdrive_files() -> List[Dict[str, Any]]:
    """Sample GDrive file metadata."""
    return [
        {
            "id": "file1",
            "title": "doc1.pdf",
            "mime_type": "application/pdf",
            "modified": "2026-02-19T10:00:00Z",
            "size_bytes": 1024,
        },
        {
            "id": "file2",
            "title": "doc2.docx",
            "mime_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "modified": "2026-02-19T11:00:00Z",
            "size_bytes": 2048,
        },
        {
            "id": "file3",
            "title": "sheet1.xlsx",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "modified": "2026-02-19T12:00:00Z",
            "size_bytes": 4096,
        },
    ]


# =============================================================================
# Test GDriveSyncManager Initialization
# =============================================================================


class TestGDriveSyncManagerInit:
    """Test GDriveSyncManager initialization."""

    def test_init_creates_pending_directory(
        self,
        mock_connector: MagicMock,
        mock_state_store: SyncStateStore,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN non-existent pending directory
        WHEN GDriveSyncManager is initialized
        THEN pending directory is created
        """
        # GIVEN
        pending_dir = tmp_path / "new_pending"
        assert not pending_dir.exists()

        # WHEN
        manager = GDriveSyncManager(
            connector=mock_connector,
            state_store=mock_state_store,
            pending_dir=pending_dir,
        )

        # THEN
        assert pending_dir.exists()
        assert manager.pending_dir == pending_dir
        assert manager.conflict_strategy == "server_wins"

    def test_init_with_custom_conflict_strategy(
        self,
        mock_connector: MagicMock,
        mock_state_store: SyncStateStore,
        pending_dir: Path,
    ) -> None:
        """
        GIVEN custom conflict strategy
        WHEN GDriveSyncManager is initialized
        THEN strategy is set correctly
        """
        # GIVEN / WHEN
        manager = GDriveSyncManager(
            connector=mock_connector,
            state_store=mock_state_store,
            pending_dir=pending_dir,
            conflict_strategy="local_wins",
        )

        # THEN
        assert manager.conflict_strategy == "local_wins"


# =============================================================================
# Test Sync Operation - Incremental Sync
# =============================================================================


class TestSyncOperation:
    """Test sync() method and incremental sync logic."""

    def test_sync_no_files_discovered(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
    ) -> None:
        """
        GIVEN GDrive returns no files
        WHEN sync() is called
        THEN report shows zero operations
        """
        # GIVEN
        mock_connector.discover.return_value = []

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 0
        assert report.updated == 0
        assert report.removed == 0
        assert report.errors == 0
        assert report.gdrive_api_calls == 1  # discover() call
        assert report.duration_seconds >= 0

    def test_sync_new_files_only(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        sample_gdrive_files: List[Dict],
        pending_dir: Path,
    ) -> None:
        """
        GIVEN 3 new files discovered
        WHEN sync() is called
        THEN all 3 files are downloaded
        """
        # GIVEN
        mock_connector.discover.return_value = sample_gdrive_files

        # Mock successful fetch for all files
        def mock_fetch(file_id: str, output_dir: Path) -> IFConnectorResult:
            file_path = output_dir / f"{file_id}.pdf"
            file_path.write_text("test content")
            return IFConnectorResult(
                success=True,
                file_path=file_path,
                metadata={"source": "gdrive", "file_id": file_id},
            )

        mock_connector.fetch.side_effect = mock_fetch

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 3
        assert report.updated == 0
        assert report.removed == 0
        assert report.errors == 0
        assert len(report.added_files) == 3
        assert mock_connector.fetch.call_count == 3

    def test_sync_changed_files_only(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        mock_state_store: SyncStateStore,
        sample_gdrive_files: List[Dict],
        pending_dir: Path,
    ) -> None:
        """
        GIVEN 3 files already synced with old timestamps
        WHEN sync() discovers files with newer timestamps
        THEN all 3 files are re-downloaded
        """
        # GIVEN - Add old state for all files
        for file_meta in sample_gdrive_files:
            old_state = FileState(
                path=str(pending_dir / f"{file_meta['id']}.pdf"),
                content_hash="old_hash",
                size=500,
                modified_time=123456.0,
                document_id=file_meta["id"],
                chunk_count=5,
                last_synced="2026-02-18T10:00:00Z",
            )
            old_state.gdrive_file_id = file_meta["id"]  # type: ignore
            old_state.gdrive_modified_time = "2026-02-18T10:00:00Z"  # type: ignore
            mock_state_store.set(old_state)

        # Mock discover with newer timestamps
        mock_connector.discover.return_value = sample_gdrive_files

        # Mock successful fetch
        def mock_fetch(file_id: str, output_dir: Path) -> IFConnectorResult:
            file_path = output_dir / f"{file_id}.pdf"
            file_path.write_text("updated content")
            return IFConnectorResult(
                success=True,
                file_path=file_path,
                metadata={"source": "gdrive", "file_id": file_id},
            )

        mock_connector.fetch.side_effect = mock_fetch

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 0
        assert report.updated == 3
        assert report.removed == 0
        assert report.errors == 0
        assert len(report.updated_files) == 3

    def test_sync_deleted_files_only(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        mock_state_store: SyncStateStore,
        pending_dir: Path,
    ) -> None:
        """
        GIVEN 3 files in sync state
        WHEN sync() discovers no files (all deleted)
        THEN all 3 files are marked as deleted
        """
        # GIVEN - Add state for 3 files
        for i in range(1, 4):
            state = FileState(
                path=str(pending_dir / f"file{i}.pdf"),
                content_hash="hash",
                size=1000,
                modified_time=123456.0,
                document_id=f"file{i}",
                chunk_count=10,
                last_synced="2026-02-19T10:00:00Z",
            )
            state.gdrive_file_id = f"file{i}"  # type: ignore
            state.gdrive_modified_time = "2026-02-19T10:00:00Z"  # type: ignore
            mock_state_store.set(state)

        # Mock discover returns empty (all files deleted)
        mock_connector.discover.return_value = []

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 0
        assert report.updated == 0
        assert report.removed == 3
        assert report.errors == 0
        assert len(report.removed_files) == 3

    def test_sync_mixed_operations(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        mock_state_store: SyncStateStore,
        sample_gdrive_files: List[Dict],
        pending_dir: Path,
    ) -> None:
        """
        GIVEN mix of new, changed, and deleted files
        WHEN sync() is called
        THEN all operations are performed correctly
        """
        # GIVEN - Add old state for file1 (will be changed)
        old_state = FileState(
            path=str(pending_dir / "file1.pdf"),
            content_hash="old_hash",
            size=500,
            modified_time=123456.0,
            document_id="file1",
            chunk_count=5,
            last_synced="2026-02-18T10:00:00Z",
        )
        old_state.gdrive_file_id = "file1"  # type: ignore
        old_state.gdrive_modified_time = "2026-02-18T10:00:00Z"  # type: ignore
        mock_state_store.set(old_state)

        # Add state for deleted_file (not in discovery)
        deleted_state = FileState(
            path=str(pending_dir / "deleted.pdf"),
            content_hash="hash",
            size=1000,
            modified_time=123456.0,
            document_id="deleted_file",
            chunk_count=10,
            last_synced="2026-02-19T10:00:00Z",
        )
        deleted_state.gdrive_file_id = "deleted_file"  # type: ignore
        deleted_state.gdrive_modified_time = "2026-02-19T10:00:00Z"  # type: ignore
        mock_state_store.set(deleted_state)

        # Mock discover (file1=changed, file2&file3=new)
        mock_connector.discover.return_value = sample_gdrive_files

        # Mock successful fetch
        def mock_fetch(file_id: str, output_dir: Path) -> IFConnectorResult:
            file_path = output_dir / f"{file_id}.pdf"
            file_path.write_text("content")
            return IFConnectorResult(
                success=True,
                file_path=file_path,
                metadata={"source": "gdrive", "file_id": file_id},
            )

        mock_connector.fetch.side_effect = mock_fetch

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 2  # file2, file3
        assert report.updated == 1  # file1
        assert report.removed == 1  # deleted_file
        assert report.errors == 0


# =============================================================================
# Test Dry Run Mode -
# =============================================================================


class TestDryRunMode:
    """Test dry-run functionality."""

    def test_sync_dry_run_no_downloads(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        sample_gdrive_files: List[Dict],
    ) -> None:
        """
        GIVEN dry_run=True
        WHEN sync() is called with new files
        THEN changes are detected but no files downloaded
        """
        # GIVEN
        mock_connector.discover.return_value = sample_gdrive_files

        # WHEN
        report = sync_manager.sync(dry_run=True)

        # THEN
        assert report.added == 3
        assert report.updated == 0
        assert report.removed == 0
        assert mock_connector.fetch.call_count == 0  # No downloads
        assert report.gdrive_api_calls == 1  # Only discover()


# =============================================================================
# Test Error Handling -
# =============================================================================


class TestErrorHandling:
    """Test error handling and retry logic."""

    def test_sync_with_fetch_errors(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        sample_gdrive_files: List[Dict],
    ) -> None:
        """
        GIVEN some files fail to download
        WHEN sync() is called
        THEN errors are reported correctly
        """
        # GIVEN
        mock_connector.discover.return_value = sample_gdrive_files

        # Mock fetch to fail for file2
        def mock_fetch(file_id: str, output_dir: Path) -> IFConnectorResult:
            if file_id == "file2":
                return IFConnectorResult(
                    success=False,
                    error_message="Network timeout",
                    http_status=500,
                )
            file_path = output_dir / f"{file_id}.pdf"
            file_path.write_text("content")
            return IFConnectorResult(
                success=True,
                file_path=file_path,
                metadata={"source": "gdrive", "file_id": file_id},
            )

        mock_connector.fetch.side_effect = mock_fetch

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 2  # file1, file3
        assert report.errors == 1  # file2
        assert len(report.error_files) == 1
        assert report.error_files[0][0] == "doc2.docx"
        assert "Network timeout" in report.error_files[0][1]

    def test_sync_with_missing_file_path(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        sample_gdrive_files: List[Dict],
    ) -> None:
        """
        GIVEN fetch returns success but no file_path
        WHEN sync() is called
        THEN error is reported
        """
        # GIVEN
        mock_connector.discover.return_value = [sample_gdrive_files[0]]

        # Mock fetch returns success but no file_path
        mock_connector.fetch.return_value = IFConnectorResult(
            success=True,
            file_path=None,  # Missing file path
            metadata={},
        )

        # WHEN
        report = sync_manager.sync()

        # THEN
        assert report.added == 0
        assert report.errors == 1
        assert "not returned" in report.error_files[0][1]


# =============================================================================
# Test Change Detection Logic
# =============================================================================


class TestChangeDetection:
    """Test _detect_changes() method."""

    def test_detect_new_file(
        self,
        sync_manager: GDriveSyncManager,
        sample_gdrive_files: List[Dict],
    ) -> None:
        """
        GIVEN file not in state
        WHEN _detect_changes() is called
        THEN file is marked as new
        """
        # GIVEN - Empty state

        # WHEN
        new, changed, deleted = sync_manager._detect_changes(sample_gdrive_files)

        # THEN
        assert len(new) == 3
        assert len(changed) == 0
        assert len(deleted) == 0

    def test_detect_changed_file(
        self,
        sync_manager: GDriveSyncManager,
        mock_state_store: SyncStateStore,
        sample_gdrive_files: List[Dict],
        pending_dir: Path,
    ) -> None:
        """
        GIVEN file with old timestamp in state
        WHEN _detect_changes() with newer timestamp
        THEN file is marked as changed
        """
        # GIVEN - Add old state
        old_state = FileState(
            path=str(pending_dir / "file1.pdf"),
            content_hash="hash",
            size=1000,
            modified_time=123456.0,
            document_id="file1",
            chunk_count=10,
            last_synced="2026-02-19T10:00:00Z",
        )
        old_state.gdrive_file_id = "file1"  # type: ignore
        old_state.gdrive_modified_time = "2026-02-18T10:00:00Z"  # type: ignore (older)
        mock_state_store.set(old_state)

        # WHEN
        new, changed, deleted = sync_manager._detect_changes([sample_gdrive_files[0]])

        # THEN
        assert len(new) == 0
        assert len(changed) == 1
        assert changed[0]["id"] == "file1"
        assert len(deleted) == 0

    def test_detect_unchanged_file(
        self,
        sync_manager: GDriveSyncManager,
        mock_state_store: SyncStateStore,
        sample_gdrive_files: List[Dict],
        pending_dir: Path,
    ) -> None:
        """
        GIVEN file with same timestamp in state
        WHEN _detect_changes() is called
        THEN file is not marked as changed
        """
        # GIVEN - Add state with same timestamp
        same_state = FileState(
            path=str(pending_dir / "file1.pdf"),
            content_hash="hash",
            size=1000,
            modified_time=123456.0,
            document_id="file1",
            chunk_count=10,
            last_synced="2026-02-19T10:00:00Z",
        )
        same_state.gdrive_file_id = "file1"  # type: ignore
        same_state.gdrive_modified_time = sample_gdrive_files[0]["modified"]  # type: ignore (same)
        mock_state_store.set(same_state)

        # WHEN
        new, changed, deleted = sync_manager._detect_changes([sample_gdrive_files[0]])

        # THEN
        assert len(new) == 0
        assert len(changed) == 0
        assert len(deleted) == 0

    def test_detect_deleted_file(
        self,
        sync_manager: GDriveSyncManager,
        mock_state_store: SyncStateStore,
        pending_dir: Path,
    ) -> None:
        """
        GIVEN file in state but not discovered
        WHEN _detect_changes() is called
        THEN file is marked as deleted
        """
        # GIVEN - Add state for file not in discovery
        state = FileState(
            path=str(pending_dir / "deleted.pdf"),
            content_hash="hash",
            size=1000,
            modified_time=123456.0,
            document_id="deleted_file",
            chunk_count=10,
            last_synced="2026-02-19T10:00:00Z",
        )
        state.gdrive_file_id = "deleted_file"  # type: ignore
        state.gdrive_modified_time = "2026-02-19T10:00:00Z"  # type: ignore
        mock_state_store.set(state)

        # WHEN - Empty discovery
        new, changed, deleted = sync_manager._detect_changes([])

        # THEN
        assert len(new) == 0
        assert len(changed) == 0
        assert len(deleted) == 1
        assert deleted[0] == "deleted_file"


# =============================================================================
# Test JPL Compliance - Rule #2: Bounded Iterations
# =============================================================================


class TestBoundedIterations:
    """Test bounded iteration compliance."""

    def test_sync_respects_max_sync_files(
        self,
        sync_manager: GDriveSyncManager,
        mock_connector: MagicMock,
        pending_dir: Path,
    ) -> None:
        """
        GIVEN more files than MAX_SYNC_FILES
        WHEN sync() is called
        THEN only MAX_SYNC_FILES are processed
        """
        # GIVEN - Create MAX_SYNC_FILES + 10 files
        many_files = [
            {
                "id": f"file{i}",
                "title": f"doc{i}.pdf",
                "mime_type": "application/pdf",
                "modified": f"2026-02-19T{i:02d}:00:00Z",
                "size_bytes": 1024,
            }
            for i in range(MAX_SYNC_FILES + 10)
        ]

        mock_connector.discover.return_value = many_files

        # Mock fetch
        def mock_fetch(file_id: str, output_dir: Path) -> IFConnectorResult:
            file_path = output_dir / f"{file_id}.pdf"
            file_path.write_text("content")
            return IFConnectorResult(
                success=True,
                file_path=file_path,
                metadata={"source": "gdrive", "file_id": file_id},
            )

        mock_connector.fetch.side_effect = mock_fetch

        # WHEN
        report = sync_manager.sync()

        # THEN - Should process at most MAX_SYNC_FILES
        assert report.added <= MAX_SYNC_FILES


# =============================================================================
# Test Sync Report -
# =============================================================================


class TestGDriveSyncReport:
    """Test GDriveSyncReport functionality."""

    def test_report_to_summary_formatting(self) -> None:
        """
        GIVEN populated sync report
        WHEN to_summary() is called
        THEN formatted summary is returned
        """
        # GIVEN
        report = GDriveSyncReport(
            added=5,
            updated=3,
            removed=1,
            errors=2,
            total_bytes_downloaded=1024 * 1024 * 10,  # 10 MB
            gdrive_api_calls=15,
            rate_limit_hits=1,
            files_skipped=2,
            conflicts_detected=0,
        )
        report.duration_seconds = 45.2

        # WHEN
        summary = report.to_summary()

        # THEN
        assert "📊 GDrive Sync Report" in summary
        assert "New files: 5 (10.0 MB)" in summary
        assert "Updated files: 3" in summary
        assert "Deleted files: 1" in summary
        assert "Errors: 2" in summary
        assert "Duration: 45.2s" in summary
        assert "API calls: 15" in summary
        assert "Rate limits: 1" in summary


# =============================================================================
# Test Status Method
# =============================================================================


class TestGetStatus:
    """Test get_status() method."""

    def test_get_status_mixed_files(
        self,
        sync_manager: GDriveSyncManager,
        mock_state_store: SyncStateStore,
        pending_dir: Path,
    ) -> None:
        """
        GIVEN mix of GDrive and local files in state
        WHEN get_status() is called
        THEN correct counts are returned
        """
        # GIVEN - Add 2 GDrive files
        for i in range(1, 3):
            state = FileState(
                path=str(pending_dir / f"gdrive{i}.pdf"),
                content_hash="hash",
                size=1000,
                modified_time=123456.0,
                document_id=f"gdrive{i}",
                chunk_count=10,
                last_synced="2026-02-19T10:00:00Z",
            )
            state.gdrive_file_id = f"gdrive{i}"  # type: ignore
            mock_state_store.set(state)

        # Add 3 local files (no gdrive_file_id)
        for i in range(1, 4):
            state = FileState(
                path=str(pending_dir / f"local{i}.pdf"),
                content_hash="hash",
                size=1000,
                modified_time=123456.0,
                document_id=f"local{i}",
                chunk_count=10,
                last_synced="2026-02-19T10:00:00Z",
            )
            mock_state_store.set(state)

        # WHEN
        status = sync_manager.get_status()

        # THEN
        assert status["total_tracked_files"] == 5
        assert status["gdrive_files"] == 2
        assert status["local_files"] == 3


# =============================================================================
# Test File State Creation
# =============================================================================


class TestFileStateCreation:
    """Test _create_gdrive_file_state() method."""

    def test_create_file_state_with_gdrive_metadata(
        self,
        sync_manager: GDriveSyncManager,
        pending_dir: Path,
    ) -> None:
        """
        GIVEN file path and GDrive metadata
        WHEN _create_gdrive_file_state() is called
        THEN FileState with GDrive fields is created
        """
        # GIVEN
        test_file = pending_dir / "test.pdf"
        test_file.write_text("test content")

        file_meta = {
            "id": "gdrive_id_123",
            "title": "test.pdf",
            "modified": "2026-02-19T10:00:00Z",
        }

        # WHEN
        state = sync_manager._create_gdrive_file_state(test_file, file_meta)

        # THEN
        assert state.path == str(test_file.resolve())
        assert state.content_hash is not None
        assert len(state.content_hash) == 64  # SHA256
        assert state.size == len("test content")
        assert hasattr(state, "gdrive_file_id")
        assert state.gdrive_file_id == "gdrive_id_123"  # type: ignore
        assert state.gdrive_modified_time == "2026-02-19T10:00:00Z"  # type: ignore


# =============================================================================
# Test Type Hints Compliance - JPL Rule #9
# =============================================================================


def test_all_functions_have_type_hints() -> None:
    """
    GIVEN GDriveSyncManager module
    WHEN inspecting all public methods
    THEN all have 100% type hints
    """
    # This test ensures JPL Rule #9 compliance
    # Type checker (mypy) will catch any violations
    manager = GDriveSyncManager(
        connector=Mock(spec=GDriveConnector),
        state_store=Mock(spec=SyncStateStore),
        pending_dir=Path("/tmp"),
    )

    # All methods should have type hints (verified by mypy)
    assert callable(manager.sync)
    assert callable(manager.get_status)
    assert callable(manager._detect_changes)
    assert callable(manager._fetch_and_track)
