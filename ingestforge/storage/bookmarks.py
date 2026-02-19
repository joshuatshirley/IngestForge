"""Bookmark Storage - Save and retrieve bookmarked chunks.

Provides localized storage for bookmarks in `.data/bookmarks.json`.
Supports atomic writes to prevent data corruption.
Part of ORG-003: Bookmarks feature.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Bookmark Dataclass
# =============================================================================


@dataclass
class Bookmark:
    """A saved bookmark linking to a chunk.

    Attributes:
        chunk_id: ID of the bookmarked chunk
        note: Optional user note
        created_at: ISO timestamp when created
        updated_at: ISO timestamp when last updated
        source_file: Original source file path (for orphan detection)
        is_orphaned: True if source file was deleted
    """

    chunk_id: str
    note: str = ""
    created_at: str = ""
    updated_at: str = ""
    source_file: str = ""
    is_orphaned: bool = False

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        now = datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON storage."""
        return {
            "chunk_id": self.chunk_id,
            "note": self.note,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_file": self.source_file,
            "is_orphaned": self.is_orphaned,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Bookmark":
        """Create from dictionary."""
        return cls(
            chunk_id=data.get("chunk_id", ""),
            note=data.get("note", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            source_file=data.get("source_file", ""),
            is_orphaned=data.get("is_orphaned", False),
        )


# =============================================================================
# BookmarkManager Class
# =============================================================================


class BookmarkManager:
    """Manages bookmark CRUD operations with atomic JSON persistence.

    Stores bookmarks in `.data/bookmarks.json` relative to the project root.
    Uses atomic writes (temp file + rename) to prevent data corruption.

    Performance target: Load 500 bookmarks in <200ms.
    """

    DEFAULT_PATH = ".data/bookmarks.json"

    def __init__(self, project_path: Optional[Path] = None) -> None:
        """Initialize bookmark manager.

        Args:
            project_path: Project root directory. Defaults to current directory.
        """
        self.project_path = project_path or Path.cwd()
        self.bookmarks_path = self.project_path / self.DEFAULT_PATH
        self._cache: Optional[Dict[str, Bookmark]] = None
        self._cache_time: float = 0.0

    def _ensure_directory(self) -> None:
        """Ensure the .data directory exists."""
        self.bookmarks_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_bookmarks(self) -> Dict[str, Bookmark]:
        """Load bookmarks from JSON file.

        Rule #1: Early return if file doesn't exist.

        Returns:
            Dictionary mapping chunk_id to Bookmark
        """
        if not self.bookmarks_path.exists():
            return {}

        try:
            with open(self.bookmarks_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            bookmarks = {}
            for item in data.get("bookmarks", []):
                bookmark = Bookmark.from_dict(item)
                bookmarks[bookmark.chunk_id] = bookmark

            return bookmarks

        except (json.JSONDecodeError, IOError):
            return {}

    def _save_bookmarks(self, bookmarks: Dict[str, Bookmark]) -> None:
        """Save bookmarks to JSON file using atomic write.

        Uses temp file + rename pattern for atomic writes.

        Args:
            bookmarks: Dictionary of bookmarks to save
        """
        self._ensure_directory()

        data = {
            "version": 1,
            "updated_at": datetime.now().isoformat(),
            "bookmarks": [b.to_dict() for b in bookmarks.values()],
        }

        # Atomic write: write to temp file, then rename
        dir_path = self.bookmarks_path.parent
        fd, temp_path = tempfile.mkstemp(
            suffix=".json",
            prefix="bookmarks_",
            dir=str(dir_path),
        )

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Atomic rename (works on POSIX, Windows needs replace)
            temp_file = Path(temp_path)
            temp_file.replace(self.bookmarks_path)

        except Exception as e:
            logger.error(f"Failed to save bookmarks: {e}")
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError as cleanup_err:
                logger.debug(f"Failed to clean up temp file {temp_path}: {cleanup_err}")
            raise

        # Invalidate cache
        self._cache = None

    def _get_bookmarks(self) -> Dict[str, Bookmark]:
        """Get bookmarks with caching.

        Cache is valid for 1 second to handle rapid access patterns.

        Returns:
            Dictionary of bookmarks
        """
        now = time.time()

        if self._cache is not None and (now - self._cache_time) < 1.0:
            return self._cache

        self._cache = self._load_bookmarks()
        self._cache_time = now
        return self._cache

    def add(
        self,
        chunk_id: str,
        note: str = "",
        source_file: str = "",
    ) -> Bookmark:
        """Add or update a bookmark.

        If bookmark exists, updates the note. No duplicates.

        Args:
            chunk_id: Chunk to bookmark
            note: Optional note text
            source_file: Source file path for orphan detection

        Returns:
            Created or updated Bookmark
        """
        bookmarks = self._get_bookmarks()

        if chunk_id in bookmarks:
            # Update existing bookmark
            bookmark = bookmarks[chunk_id]
            bookmark.note = note
            bookmark.updated_at = datetime.now().isoformat()
            if source_file:
                bookmark.source_file = source_file
        else:
            # Create new bookmark
            bookmark = Bookmark(
                chunk_id=chunk_id,
                note=note,
                source_file=source_file,
            )
            bookmarks[chunk_id] = bookmark

        self._save_bookmarks(bookmarks)
        return bookmark

    def get(self, chunk_id: str) -> Optional[Bookmark]:
        """Get a bookmark by chunk ID.

        Args:
            chunk_id: Chunk ID to look up

        Returns:
            Bookmark or None if not found
        """
        bookmarks = self._get_bookmarks()
        return bookmarks.get(chunk_id)

    def remove(self, chunk_id: str) -> bool:
        """Remove a bookmark.

        Args:
            chunk_id: Chunk ID to remove

        Returns:
            True if removed, False if not found
        """
        bookmarks = self._get_bookmarks()

        if chunk_id not in bookmarks:
            return False

        del bookmarks[chunk_id]
        self._save_bookmarks(bookmarks)
        return True

    def list_all(self) -> List[Bookmark]:
        """List all bookmarks.

        Returns:
            List of all bookmarks, sorted by created_at descending
        """
        bookmarks = self._get_bookmarks()
        result = list(bookmarks.values())
        result.sort(key=lambda b: b.created_at, reverse=True)
        return result

    def count(self) -> int:
        """Get total number of bookmarks."""
        return len(self._get_bookmarks())

    def check_orphans(self) -> int:
        """Check for orphaned bookmarks (source file deleted).

        Updates is_orphaned flag for bookmarks whose source files
        no longer exist.

        Returns:
            Number of newly orphaned bookmarks
        """
        bookmarks = self._get_bookmarks()
        orphaned_count = 0

        for bookmark in bookmarks.values():
            # Skip if no source file or already orphaned
            if not bookmark.source_file or bookmark.is_orphaned:
                continue

            # Check if source file exists
            source_path = Path(bookmark.source_file)
            if not source_path.is_absolute():
                source_path = self.project_path / source_path

            if not source_path.exists():
                bookmark.is_orphaned = True
                bookmark.updated_at = datetime.now().isoformat()
                orphaned_count += 1

        if orphaned_count > 0:
            self._save_bookmarks(bookmarks)

        return orphaned_count

    def get_orphaned(self) -> List[Bookmark]:
        """Get all orphaned bookmarks.

        Returns:
            List of orphaned bookmarks
        """
        bookmarks = self._get_bookmarks()
        return [b for b in bookmarks.values() if b.is_orphaned]

    def clear_orphaned(self) -> int:
        """Remove all orphaned bookmarks.

        Returns:
            Number of bookmarks removed
        """
        bookmarks = self._get_bookmarks()
        initial_count = len(bookmarks)

        bookmarks = {k: v for k, v in bookmarks.items() if not v.is_orphaned}

        removed_count = initial_count - len(bookmarks)
        if removed_count > 0:
            self._save_bookmarks(bookmarks)

        return removed_count


# =============================================================================
# Factory Function
# =============================================================================


def get_bookmark_manager(
    project_path: Optional[Path] = None,
) -> BookmarkManager:
    """Get bookmark manager instance.

    Args:
        project_path: Project directory (defaults to current directory)

    Returns:
        BookmarkManager instance
    """
    return BookmarkManager(project_path=project_path)
