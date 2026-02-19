"""
Data models for sync operations.

Provides FileState and SyncReport for tracking file synchronization.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class FileState:
    """
    Tracked state for a single file.

    Epic GDrive sync state tracking
    ---------------------------------------
    Extended with GDrive-specific metadata for incremental sync:

    Core Fields (Original):
    - path: Local file path
    - content_hash: SHA256 for integrity verification
    - size: File size in bytes
    - modified_time: Local filesystem mtime
    - document_id: IngestForge document identifier
    - chunk_count: Number of chunks created
    - last_synced: ISO timestamp of last sync

    GDrive Extension Fields ():
    - gdrive_file_id: GDrive file ID for remote tracking
    - gdrive_modified_time: GDrive modifiedTime for change detection

    Backward Compatibility
    ----------------------
    - Uses Optional[str] with None defaults
    - Existing sync_state.json files load without errors
    - Fields auto-populate on next GDrive sync
    - No migration required

    Change Detection Logic
    ----------------------
    Local files: Compare content_hash (existing behavior)
    GDrive files: Compare gdrive_modified_time (new behavior)

    This dual approach allows:
    - Local file sync (original SyncManager)
    - Remote GDrive sync (new GDriveSyncManager)
    - Both using same FileState model

    Implementation: (2026-02-18)
    File: ingestforge/core/sync/models.py:12-31
    Tests: test_create_file_state_with_gdrive_metadata
    Usage: GDriveSyncManager._create_gdrive_file_state()
    """

    path: str
    content_hash: str
    size: int
    modified_time: float
    document_id: str
    chunk_count: int = 0
    last_synced: str = ""

    # Epic GDrive-specific fields ()
    gdrive_file_id: Optional[str] = None  # Remote file ID for tracking
    gdrive_modified_time: Optional[str] = None  # Remote timestamp for change detection

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileState":
        return cls(**data)


@dataclass
class SyncReport:
    """Report from a sync operation."""

    added: int = 0
    updated: int = 0
    removed: int = 0
    skipped: int = 0
    errors: int = 0

    # Detailed lists
    added_files: List[str] = field(default_factory=list)
    updated_files: List[str] = field(default_factory=list)
    removed_files: List[str] = field(default_factory=list)
    error_files: List[Tuple[str, str]] = field(default_factory=list)  # (path, error)

    # Timing
    started_at: str = ""
    completed_at: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "added": self.added,
            "updated": self.updated,
            "removed": self.removed,
            "skipped": self.skipped,
            "errors": self.errors,
            "added_files": self.added_files,
            "updated_files": self.updated_files,
            "removed_files": self.removed_files,
            "error_files": self.error_files,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_seconds": self.duration_seconds,
        }
