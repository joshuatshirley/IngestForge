"""
Tests for Bookmark Storage.

This module tests the BookmarkManager CRUD operations and atomic writes.

Test Strategy
-------------
- Focus on CRUD operations (add, get, remove, list)
- Test atomic write behavior
- Test orphan detection
- Verify performance target (<200ms for 500 bookmarks)
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestBookmarkDataclass: Bookmark model tests
- TestBookmarkManagerInit: Initialization tests
- TestBookmarkManagerCRUD: CRUD operation tests
- TestBookmarkOrphanDetection: Orphan handling tests
- TestBookmarkPerformance: Performance tests

Part of ORG-003: Bookmarks feature.
"""

import json
import time
from pathlib import Path


from ingestforge.storage.bookmarks import (
    Bookmark,
    BookmarkManager,
    get_bookmark_manager,
)


# ============================================================================
# Test Helpers
# ============================================================================


def create_test_bookmark(
    chunk_id: str = "test_chunk_001",
    note: str = "",
    source_file: str = "",
) -> Bookmark:
    """Create a test Bookmark."""
    return Bookmark(
        chunk_id=chunk_id,
        note=note,
        source_file=source_file,
    )


def create_source_file(project_path: Path, filename: str) -> Path:
    """Create a test source file."""
    source_path = project_path / "documents" / filename
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text("Test content")
    return source_path


# ============================================================================
# TestBookmarkDataclass
# ============================================================================


class TestBookmarkDataclass:
    """Tests for the Bookmark dataclass.

    Rule #4: Focused test class - tests data model only
    """

    def test_create_with_defaults(self):
        """Test creating bookmark with default values."""
        bookmark = Bookmark(chunk_id="chunk_123")

        assert bookmark.chunk_id == "chunk_123"
        assert bookmark.note == ""
        assert bookmark.is_orphaned is False
        assert bookmark.created_at != ""  # Auto-set
        assert bookmark.updated_at != ""  # Auto-set

    def test_create_with_all_fields(self):
        """Test creating bookmark with all fields."""
        bookmark = Bookmark(
            chunk_id="chunk_456",
            note="Important quote",
            source_file="docs/chapter1.md",
            is_orphaned=False,
        )

        assert bookmark.chunk_id == "chunk_456"
        assert bookmark.note == "Important quote"
        assert bookmark.source_file == "docs/chapter1.md"
        assert bookmark.is_orphaned is False

    def test_to_dict(self):
        """Test converting bookmark to dictionary."""
        bookmark = Bookmark(
            chunk_id="chunk_789",
            note="Test note",
        )

        data = bookmark.to_dict()

        assert data["chunk_id"] == "chunk_789"
        assert data["note"] == "Test note"
        assert "created_at" in data
        assert "updated_at" in data
        assert data["is_orphaned"] is False

    def test_from_dict(self):
        """Test creating bookmark from dictionary."""
        data = {
            "chunk_id": "chunk_abc",
            "note": "Restored note",
            "created_at": "2026-02-14T10:00:00",
            "updated_at": "2026-02-14T10:00:00",
            "source_file": "test.md",
            "is_orphaned": True,
        }

        bookmark = Bookmark.from_dict(data)

        assert bookmark.chunk_id == "chunk_abc"
        assert bookmark.note == "Restored note"
        assert bookmark.created_at == "2026-02-14T10:00:00"
        assert bookmark.is_orphaned is True

    def test_from_dict_with_missing_fields(self):
        """Test creating bookmark from partial dictionary."""
        data = {"chunk_id": "chunk_def"}

        bookmark = Bookmark.from_dict(data)

        assert bookmark.chunk_id == "chunk_def"
        assert bookmark.note == ""
        assert bookmark.is_orphaned is False


# ============================================================================
# TestBookmarkManagerInit
# ============================================================================


class TestBookmarkManagerInit:
    """Tests for BookmarkManager initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_with_defaults(self, tmp_path):
        """Test creating manager with default settings."""
        manager = BookmarkManager(project_path=tmp_path)

        assert manager.project_path == tmp_path
        assert manager.bookmarks_path == tmp_path / ".data" / "bookmarks.json"

    def test_factory_function(self, tmp_path):
        """Test get_bookmark_manager factory."""
        manager = get_bookmark_manager(tmp_path)

        assert isinstance(manager, BookmarkManager)
        assert manager.project_path == tmp_path

    def test_creates_data_directory(self, tmp_path):
        """Test that .data directory is created on first save."""
        manager = BookmarkManager(project_path=tmp_path)

        # Directory doesn't exist yet
        assert not (tmp_path / ".data").exists()

        # Add a bookmark
        manager.add("chunk_001")

        # Directory should now exist
        assert (tmp_path / ".data").exists()
        assert manager.bookmarks_path.exists()


# ============================================================================
# TestBookmarkManagerCRUD
# ============================================================================


class TestBookmarkManagerCRUD:
    """Tests for BookmarkManager CRUD operations.

    Rule #4: Focused test class - tests CRUD only
    """

    def test_add_bookmark(self, tmp_path):
        """Test adding a new bookmark."""
        manager = BookmarkManager(project_path=tmp_path)

        bookmark = manager.add("chunk_001", note="Important")

        assert bookmark.chunk_id == "chunk_001"
        assert bookmark.note == "Important"

    def test_add_bookmark_without_note(self, tmp_path):
        """Test adding a bookmark without a note."""
        manager = BookmarkManager(project_path=tmp_path)

        bookmark = manager.add("chunk_002")

        assert bookmark.chunk_id == "chunk_002"
        assert bookmark.note == ""

    def test_add_duplicate_updates_note(self, tmp_path):
        """Test that adding same chunk updates the note (no duplicates)."""
        manager = BookmarkManager(project_path=tmp_path)

        # Add initial bookmark
        manager.add("chunk_003", note="First note")

        # Add same chunk with different note
        bookmark = manager.add("chunk_003", note="Updated note")

        assert bookmark.note == "Updated note"
        assert manager.count() == 1  # No duplicate

    def test_get_existing_bookmark(self, tmp_path):
        """Test getting an existing bookmark."""
        manager = BookmarkManager(project_path=tmp_path)
        manager.add("chunk_004", note="Test")

        bookmark = manager.get("chunk_004")

        assert bookmark is not None
        assert bookmark.chunk_id == "chunk_004"
        assert bookmark.note == "Test"

    def test_get_nonexistent_bookmark(self, tmp_path):
        """Test getting a non-existent bookmark returns None."""
        manager = BookmarkManager(project_path=tmp_path)

        bookmark = manager.get("nonexistent")

        assert bookmark is None

    def test_remove_existing_bookmark(self, tmp_path):
        """Test removing an existing bookmark."""
        manager = BookmarkManager(project_path=tmp_path)
        manager.add("chunk_005")

        result = manager.remove("chunk_005")

        assert result is True
        assert manager.get("chunk_005") is None
        assert manager.count() == 0

    def test_remove_nonexistent_bookmark(self, tmp_path):
        """Test removing a non-existent bookmark returns False."""
        manager = BookmarkManager(project_path=tmp_path)

        result = manager.remove("nonexistent")

        assert result is False

    def test_list_all_bookmarks(self, tmp_path):
        """Test listing all bookmarks."""
        manager = BookmarkManager(project_path=tmp_path)
        manager.add("chunk_006")
        manager.add("chunk_007", note="Second")
        manager.add("chunk_008", note="Third")

        bookmarks = manager.list_all()

        assert len(bookmarks) == 3
        chunk_ids = [b.chunk_id for b in bookmarks]
        assert "chunk_006" in chunk_ids
        assert "chunk_007" in chunk_ids
        assert "chunk_008" in chunk_ids

    def test_list_all_empty(self, tmp_path):
        """Test listing bookmarks when none exist."""
        manager = BookmarkManager(project_path=tmp_path)

        bookmarks = manager.list_all()

        assert len(bookmarks) == 0

    def test_count(self, tmp_path):
        """Test counting bookmarks."""
        manager = BookmarkManager(project_path=tmp_path)

        assert manager.count() == 0

        manager.add("chunk_009")
        assert manager.count() == 1

        manager.add("chunk_010")
        assert manager.count() == 2

        manager.remove("chunk_009")
        assert manager.count() == 1


# ============================================================================
# TestBookmarkPersistence
# ============================================================================


class TestBookmarkPersistence:
    """Tests for bookmark persistence and atomic writes.

    Rule #4: Focused test class - tests persistence only
    """

    def test_bookmarks_persist_across_instances(self, tmp_path):
        """Test that bookmarks persist when creating new manager instance."""
        # Add bookmarks with first instance
        manager1 = BookmarkManager(project_path=tmp_path)
        manager1.add("chunk_011", note="Persistent")

        # Create new instance and verify bookmarks exist
        manager2 = BookmarkManager(project_path=tmp_path)
        bookmark = manager2.get("chunk_011")

        assert bookmark is not None
        assert bookmark.note == "Persistent"

    def test_json_file_format(self, tmp_path):
        """Test that JSON file has correct structure."""
        manager = BookmarkManager(project_path=tmp_path)
        manager.add("chunk_012", note="Format test")

        # Read raw JSON
        with open(manager.bookmarks_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "version" in data
        assert "updated_at" in data
        assert "bookmarks" in data
        assert len(data["bookmarks"]) == 1
        assert data["bookmarks"][0]["chunk_id"] == "chunk_012"

    def test_handles_corrupted_json(self, tmp_path):
        """Test that corrupted JSON doesn't crash the manager."""
        # Create corrupted file
        data_dir = tmp_path / ".data"
        data_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = data_dir / "bookmarks.json"
        corrupted_path.write_text("{ corrupted json")

        # Manager should handle gracefully
        manager = BookmarkManager(project_path=tmp_path)
        bookmarks = manager.list_all()

        assert len(bookmarks) == 0


# ============================================================================
# TestBookmarkOrphanDetection
# ============================================================================


class TestBookmarkOrphanDetection:
    """Tests for orphan detection.

    Rule #4: Focused test class - tests orphan handling only
    """

    def test_check_orphans_with_missing_file(self, tmp_path):
        """Test detecting orphaned bookmarks when source file is deleted."""
        manager = BookmarkManager(project_path=tmp_path)

        # Create source file
        source_file = create_source_file(tmp_path, "chapter.md")

        # Add bookmark with source file
        manager.add(
            "chunk_013",
            source_file=str(source_file.relative_to(tmp_path)),
        )

        # Delete the source file
        source_file.unlink()

        # Check for orphans
        orphan_count = manager.check_orphans()

        assert orphan_count == 1
        bookmark = manager.get("chunk_013")
        assert bookmark.is_orphaned is True

    def test_check_orphans_with_existing_file(self, tmp_path):
        """Test that existing files don't trigger orphan detection."""
        manager = BookmarkManager(project_path=tmp_path)

        # Create source file
        source_file = create_source_file(tmp_path, "exists.md")

        # Add bookmark with source file
        manager.add(
            "chunk_014",
            source_file=str(source_file.relative_to(tmp_path)),
        )

        # Check for orphans (file still exists)
        orphan_count = manager.check_orphans()

        assert orphan_count == 0
        bookmark = manager.get("chunk_014")
        assert bookmark.is_orphaned is False

    def test_get_orphaned_bookmarks(self, tmp_path):
        """Test getting list of orphaned bookmarks."""
        manager = BookmarkManager(project_path=tmp_path)

        # Add some bookmarks
        manager.add("chunk_015")
        manager.add("chunk_016")

        # Manually mark one as orphaned
        bookmarks = manager._get_bookmarks()
        bookmarks["chunk_015"].is_orphaned = True
        manager._save_bookmarks(bookmarks)

        # Get orphaned
        orphaned = manager.get_orphaned()

        assert len(orphaned) == 1
        assert orphaned[0].chunk_id == "chunk_015"

    def test_clear_orphaned_bookmarks(self, tmp_path):
        """Test removing all orphaned bookmarks."""
        manager = BookmarkManager(project_path=tmp_path)

        # Add bookmarks and mark some as orphaned
        manager.add("chunk_017")
        manager.add("chunk_018")
        manager.add("chunk_019")

        bookmarks = manager._get_bookmarks()
        bookmarks["chunk_017"].is_orphaned = True
        bookmarks["chunk_018"].is_orphaned = True
        manager._save_bookmarks(bookmarks)

        # Clear orphaned
        removed = manager.clear_orphaned()

        assert removed == 2
        assert manager.count() == 1
        assert manager.get("chunk_019") is not None


# ============================================================================
# TestBookmarkPerformance
# ============================================================================


class TestBookmarkPerformance:
    """Tests for bookmark performance requirements.

    Rule #4: Focused test class - tests performance only
    AC: Loading 500 bookmarks should complete in <200ms
    """

    def test_load_500_bookmarks_under_200ms(self, tmp_path):
        """Test that loading 500 bookmarks completes in under 200ms."""
        manager = BookmarkManager(project_path=tmp_path)

        # Add 500 bookmarks
        for i in range(500):
            manager.add(f"chunk_{i:04d}", note=f"Note for chunk {i}")

        # Clear cache
        manager._cache = None

        # Time loading
        start = time.time()
        bookmarks = manager.list_all()
        elapsed_ms = (time.time() - start) * 1000

        assert len(bookmarks) == 500
        assert elapsed_ms < 200, f"Loading took {elapsed_ms:.1f}ms (limit: 200ms)"

    def test_caching_improves_performance(self, tmp_path):
        """Test that caching improves subsequent access."""
        manager = BookmarkManager(project_path=tmp_path)

        # Add some bookmarks
        for i in range(100):
            manager.add(f"chunk_{i:04d}")

        # Clear cache
        manager._cache = None

        # First access (uncached)
        start1 = time.time()
        manager.list_all()
        time1 = time.time() - start1

        # Second access (cached)
        start2 = time.time()
        manager.list_all()
        time2 = time.time() - start2

        # Cached access should be faster
        assert time2 < time1
