"""
Tests for Annotation Storage.

This module tests the AnnotationManager CRUD operations, content hash mapping,
and HTML sanitization.

Test Strategy
-------------
- Focus on CRUD operations (add, get, update, delete, list)
- Test content hash mapping for re-ingestion persistence
- Test HTML sanitization for safe export
- Verify multi-line annotation support up to 10,000 characters
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestAnnotationDataclass: Annotation model tests
- TestAnnotationManagerInit: Initialization tests
- TestAnnotationManagerCRUD: CRUD operation tests
- TestAnnotationContentHash: Content hash mapping tests
- TestAnnotationSanitization: HTML sanitization tests
- TestAnnotationValidation: Input validation tests

Part of ORG-004: Annotations feature.
"""

import json

import pytest

from ingestforge.storage.annotations import (
    Annotation,
    AnnotationManager,
    get_annotation_manager,
    compute_content_hash,
    sanitize_annotation_text,
    MAX_ANNOTATION_LENGTH,
)


# ============================================================================
# Test Helpers
# ============================================================================


def create_test_annotation(
    chunk_id: str = "test_chunk_001",
    content_hash: str = "abc123def456",
    text: str = "Test annotation",
) -> Annotation:
    """Create a test Annotation."""
    return Annotation(
        annotation_id=Annotation.generate_id(),
        chunk_id=chunk_id,
        content_hash=content_hash,
        text=text,
    )


# ============================================================================
# TestAnnotationDataclass
# ============================================================================


class TestAnnotationDataclass:
    """Tests for the Annotation dataclass.

    Rule #4: Focused test class - tests data model only
    """

    def test_create_with_required_fields(self):
        """Test creating annotation with required fields."""
        annotation = Annotation(
            annotation_id="ann_001",
            chunk_id="chunk_123",
            content_hash="abcdef123456",
            text="My annotation",
        )

        assert annotation.annotation_id == "ann_001"
        assert annotation.chunk_id == "chunk_123"
        assert annotation.content_hash == "abcdef123456"
        assert annotation.text == "My annotation"
        assert annotation.created_at != ""  # Auto-set
        assert annotation.updated_at != ""  # Auto-set

    def test_generate_id_format(self):
        """Test that generated IDs have correct format."""
        id1 = Annotation.generate_id()
        id2 = Annotation.generate_id()

        assert id1.startswith("ann_")
        assert len(id1) == 16  # "ann_" + 12 hex chars
        assert id1 != id2  # Should be unique

    def test_to_dict(self):
        """Test converting annotation to dictionary."""
        annotation = Annotation(
            annotation_id="ann_002",
            chunk_id="chunk_456",
            content_hash="hash789",
            text="Test note",
        )

        data = annotation.to_dict()

        assert data["annotation_id"] == "ann_002"
        assert data["chunk_id"] == "chunk_456"
        assert data["content_hash"] == "hash789"
        assert data["text"] == "Test note"
        assert "created_at" in data
        assert "updated_at" in data

    def test_from_dict(self):
        """Test creating annotation from dictionary."""
        data = {
            "annotation_id": "ann_003",
            "chunk_id": "chunk_abc",
            "content_hash": "hash_abc",
            "text": "Restored note",
            "created_at": "2026-02-14T10:00:00",
            "updated_at": "2026-02-14T11:00:00",
        }

        annotation = Annotation.from_dict(data)

        assert annotation.annotation_id == "ann_003"
        assert annotation.chunk_id == "chunk_abc"
        assert annotation.content_hash == "hash_abc"
        assert annotation.text == "Restored note"
        assert annotation.created_at == "2026-02-14T10:00:00"

    def test_from_dict_with_missing_fields(self):
        """Test creating annotation from partial dictionary."""
        data = {"annotation_id": "ann_004", "chunk_id": "chunk_def"}

        annotation = Annotation.from_dict(data)

        assert annotation.annotation_id == "ann_004"
        assert annotation.chunk_id == "chunk_def"
        assert annotation.content_hash == ""
        assert annotation.text == ""

    def test_preview_short_text(self):
        """Test preview returns full text when short."""
        annotation = Annotation(
            annotation_id="ann_005",
            chunk_id="chunk_001",
            content_hash="hash",
            text="Short note",
        )

        preview = annotation.preview(100)

        assert preview == "Short note"

    def test_preview_truncates_long_text(self):
        """Test preview truncates long text with ellipsis."""
        long_text = "A" * 200
        annotation = Annotation(
            annotation_id="ann_006",
            chunk_id="chunk_001",
            content_hash="hash",
            text=long_text,
        )

        preview = annotation.preview(50)

        assert len(preview) == 50
        assert preview.endswith("...")


# ============================================================================
# TestAnnotationSanitization
# ============================================================================


class TestAnnotationSanitization:
    """Tests for HTML sanitization.

    Rule #7: Check parameters, sanitize for HTML export
    """

    def test_sanitize_for_html_escapes_tags(self):
        """Test that HTML tags are escaped."""
        annotation = Annotation(
            annotation_id="ann_007",
            chunk_id="chunk_001",
            content_hash="hash",
            text="<script>alert('xss')</script>",
        )

        sanitized = annotation.sanitize_for_html()

        assert "<script>" not in sanitized
        assert "&lt;script&gt;" in sanitized

    def test_sanitize_for_html_escapes_quotes(self):
        """Test that quotes are escaped."""
        annotation = Annotation(
            annotation_id="ann_008",
            chunk_id="chunk_001",
            content_hash="hash",
            text='He said "hello"',
        )

        sanitized = annotation.sanitize_for_html()

        assert "&quot;" in sanitized

    def test_sanitize_for_html_escapes_ampersand(self):
        """Test that ampersands are escaped."""
        annotation = Annotation(
            annotation_id="ann_009",
            chunk_id="chunk_001",
            content_hash="hash",
            text="A & B",
        )

        sanitized = annotation.sanitize_for_html()

        assert "&amp;" in sanitized

    def test_sanitize_annotation_text_trims_whitespace(self):
        """Test that text is trimmed."""
        result = sanitize_annotation_text("  Hello World  ")

        assert result == "Hello World"

    def test_sanitize_annotation_text_normalizes_spaces(self):
        """Test that multiple spaces are collapsed."""
        result = sanitize_annotation_text("Hello    World")

        assert result == "Hello World"

    def test_sanitize_annotation_text_preserves_newlines(self):
        """Test that intentional line breaks are preserved."""
        result = sanitize_annotation_text("Line 1\nLine 2\nLine 3")

        assert "Line 1\nLine 2\nLine 3" == result

    def test_sanitize_annotation_text_empty_raises(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_annotation_text("")

    def test_sanitize_annotation_text_whitespace_only_raises(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            sanitize_annotation_text("   \n\t  ")

    def test_sanitize_annotation_text_too_long_raises(self):
        """Test that text exceeding max length raises ValueError."""
        long_text = "A" * (MAX_ANNOTATION_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum length"):
            sanitize_annotation_text(long_text)


# ============================================================================
# TestAnnotationManagerInit
# ============================================================================


class TestAnnotationManagerInit:
    """Tests for AnnotationManager initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_with_defaults(self, tmp_path):
        """Test creating manager with default settings."""
        manager = AnnotationManager(data_dir=tmp_path)

        assert manager.data_dir == tmp_path
        assert manager.file_path == tmp_path / "annotations.json"

    def test_factory_function(self, tmp_path):
        """Test get_annotation_manager factory."""
        manager = get_annotation_manager(tmp_path)

        assert isinstance(manager, AnnotationManager)
        assert manager.data_dir == tmp_path

    def test_creates_directory_on_save(self, tmp_path):
        """Test that directory is created on first save."""
        data_dir = tmp_path / "new_data"
        manager = AnnotationManager(data_dir=data_dir)

        # Add an annotation
        manager.add("chunk_001", "Test note", "hash123")

        # Directory should now exist
        assert data_dir.exists()
        assert manager.file_path.exists()


# ============================================================================
# TestAnnotationManagerCRUD
# ============================================================================


class TestAnnotationManagerCRUD:
    """Tests for AnnotationManager CRUD operations.

    Rule #4: Focused test class - tests CRUD only
    """

    def test_add_annotation(self, tmp_path):
        """Test adding a new annotation."""
        manager = AnnotationManager(data_dir=tmp_path)

        annotation = manager.add("chunk_001", "My note", "hash123")

        assert annotation.chunk_id == "chunk_001"
        assert annotation.text == "My note"
        assert annotation.content_hash == "hash123"
        assert annotation.annotation_id.startswith("ann_")

    def test_add_multi_line_annotation(self, tmp_path):
        """Test adding a multi-line annotation."""
        manager = AnnotationManager(data_dir=tmp_path)

        text = """Line 1: Introduction
Line 2: Main point
Line 3: Conclusion"""

        annotation = manager.add("chunk_002", text, "hash456")

        assert "Line 1" in annotation.text
        assert "Line 2" in annotation.text
        assert "Line 3" in annotation.text

    def test_get_existing_annotation(self, tmp_path):
        """Test getting an existing annotation by ID."""
        manager = AnnotationManager(data_dir=tmp_path)
        created = manager.add("chunk_003", "Test", "hash789")

        retrieved = manager.get(created.annotation_id)

        assert retrieved is not None
        assert retrieved.annotation_id == created.annotation_id
        assert retrieved.text == "Test"

    def test_get_nonexistent_annotation(self, tmp_path):
        """Test getting a non-existent annotation returns None."""
        manager = AnnotationManager(data_dir=tmp_path)

        annotation = manager.get("nonexistent_id")

        assert annotation is None

    def test_get_for_chunk(self, tmp_path):
        """Test getting all annotations for a chunk."""
        manager = AnnotationManager(data_dir=tmp_path)
        manager.add("chunk_004", "Note 1", "hash1")
        manager.add("chunk_004", "Note 2", "hash1")
        manager.add("chunk_005", "Different chunk", "hash2")

        annotations = manager.get_for_chunk("chunk_004")

        assert len(annotations) == 2
        assert all(a.chunk_id == "chunk_004" for a in annotations)

    def test_get_for_chunk_empty(self, tmp_path):
        """Test getting annotations for chunk with none."""
        manager = AnnotationManager(data_dir=tmp_path)

        annotations = manager.get_for_chunk("nonexistent")

        assert len(annotations) == 0

    def test_update_annotation(self, tmp_path):
        """Test updating an annotation."""
        manager = AnnotationManager(data_dir=tmp_path)
        created = manager.add("chunk_006", "Original", "hash")

        updated = manager.update(created.annotation_id, "Updated text")

        assert updated is not None
        assert updated.text == "Updated text"
        assert updated.updated_at != created.created_at

    def test_update_nonexistent_annotation(self, tmp_path):
        """Test updating non-existent annotation returns None."""
        manager = AnnotationManager(data_dir=tmp_path)

        result = manager.update("nonexistent", "New text")

        assert result is None

    def test_delete_annotation(self, tmp_path):
        """Test deleting an annotation."""
        manager = AnnotationManager(data_dir=tmp_path)
        created = manager.add("chunk_007", "To delete", "hash")

        result = manager.delete(created.annotation_id)

        assert result is True
        assert manager.get(created.annotation_id) is None

    def test_delete_nonexistent_annotation(self, tmp_path):
        """Test deleting non-existent annotation returns False."""
        manager = AnnotationManager(data_dir=tmp_path)

        result = manager.delete("nonexistent")

        assert result is False

    def test_delete_for_chunk(self, tmp_path):
        """Test deleting all annotations for a chunk."""
        manager = AnnotationManager(data_dir=tmp_path)
        manager.add("chunk_008", "Note 1", "hash")
        manager.add("chunk_008", "Note 2", "hash")
        manager.add("chunk_009", "Keep this", "hash")

        deleted = manager.delete_for_chunk("chunk_008")

        assert deleted == 2
        assert len(manager.get_for_chunk("chunk_008")) == 0
        assert len(manager.get_for_chunk("chunk_009")) == 1

    def test_list_all(self, tmp_path):
        """Test listing all annotations."""
        manager = AnnotationManager(data_dir=tmp_path)
        manager.add("chunk_010", "Note 1", "hash1")
        manager.add("chunk_011", "Note 2", "hash2")
        manager.add("chunk_012", "Note 3", "hash3")

        annotations = manager.list_all()

        assert len(annotations) == 3

    def test_list_all_with_limit(self, tmp_path):
        """Test listing annotations with limit."""
        manager = AnnotationManager(data_dir=tmp_path)
        for i in range(10):
            manager.add(f"chunk_{i:02d}", f"Note {i}", f"hash{i}")

        annotations = manager.list_all(limit=5)

        assert len(annotations) == 5

    def test_count(self, tmp_path):
        """Test counting annotations."""
        manager = AnnotationManager(data_dir=tmp_path)

        assert manager.count() == 0

        manager.add("chunk_013", "Note 1", "hash")
        assert manager.count() == 1

        manager.add("chunk_014", "Note 2", "hash")
        assert manager.count() == 2


# ============================================================================
# TestAnnotationContentHash
# ============================================================================


class TestAnnotationContentHash:
    """Tests for content hash mapping.

    Content hashes allow annotations to persist even when
    source chunks are re-ingested with new IDs.
    """

    def test_compute_content_hash(self):
        """Test computing content hash."""
        content = "This is test content"
        hash1 = compute_content_hash(content)
        hash2 = compute_content_hash(content)

        # Same content should produce same hash
        assert hash1 == hash2

        # Different content should produce different hash
        hash3 = compute_content_hash("Different content")
        assert hash1 != hash3

    def test_content_hash_is_sha256(self):
        """Test that content hash is valid SHA256."""
        content = "Test content"
        hash_value = compute_content_hash(content)

        # SHA256 produces 64 hex characters
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_get_by_content_hash(self, tmp_path):
        """Test finding annotations by content hash."""
        manager = AnnotationManager(data_dir=tmp_path)

        # Add annotations with same content hash (same content)
        hash1 = "same_hash_123"
        manager.add("old_chunk_id", "Note for content", hash1)

        # Find by content hash
        annotations = manager.get_by_content_hash(hash1)

        assert len(annotations) == 1
        assert annotations[0].content_hash == hash1

    def test_get_by_content_hash_multiple(self, tmp_path):
        """Test finding multiple annotations by content hash."""
        manager = AnnotationManager(data_dir=tmp_path)

        hash1 = "content_hash_456"
        manager.add("chunk_A", "Note 1", hash1)
        manager.add("chunk_A", "Note 2", hash1)

        annotations = manager.get_by_content_hash(hash1)

        assert len(annotations) == 2

    def test_remap_chunk_id(self, tmp_path):
        """Test remapping annotations from old chunk ID to new."""
        manager = AnnotationManager(data_dir=tmp_path)
        manager.add("old_chunk", "Note 1", "hash")
        manager.add("old_chunk", "Note 2", "hash")
        manager.add("other_chunk", "Keep original", "hash2")

        # Remap after re-ingestion
        count = manager.remap_chunk_id("old_chunk", "new_chunk")

        assert count == 2

        # Verify remapping
        assert len(manager.get_for_chunk("old_chunk")) == 0
        assert len(manager.get_for_chunk("new_chunk")) == 2
        assert len(manager.get_for_chunk("other_chunk")) == 1


# ============================================================================
# TestAnnotationPersistence
# ============================================================================


class TestAnnotationPersistence:
    """Tests for annotation persistence and atomic writes.

    Rule #4: Focused test class - tests persistence only
    """

    def test_annotations_persist_across_instances(self, tmp_path):
        """Test that annotations persist when creating new manager instance."""
        # Add annotations with first instance
        manager1 = AnnotationManager(data_dir=tmp_path)
        created = manager1.add("chunk_015", "Persistent note", "hash")

        # Create new instance and verify annotations exist
        manager2 = AnnotationManager(data_dir=tmp_path)
        annotation = manager2.get(created.annotation_id)

        assert annotation is not None
        assert annotation.text == "Persistent note"

    def test_json_file_format(self, tmp_path):
        """Test that JSON file has correct structure."""
        manager = AnnotationManager(data_dir=tmp_path)
        manager.add("chunk_016", "Format test", "hash")

        # Read raw JSON
        with open(manager.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "version" in data
        assert "annotations" in data
        assert len(data["annotations"]) == 1
        assert data["annotations"][0]["chunk_id"] == "chunk_016"

    def test_handles_corrupted_json(self, tmp_path):
        """Test that corrupted JSON doesn't crash the manager."""
        # Create corrupted file
        data_dir = tmp_path
        data_dir.mkdir(parents=True, exist_ok=True)
        corrupted_path = data_dir / "annotations.json"
        corrupted_path.write_text("{ corrupted json")

        # Manager should handle gracefully
        manager = AnnotationManager(data_dir=data_dir)
        annotations = manager.list_all()

        assert len(annotations) == 0


# ============================================================================
# TestAnnotationStatistics
# ============================================================================


class TestAnnotationStatistics:
    """Tests for annotation statistics."""

    def test_get_statistics_empty(self, tmp_path):
        """Test statistics with no annotations."""
        manager = AnnotationManager(data_dir=tmp_path)

        stats = manager.get_statistics()

        assert stats["total_annotations"] == 0
        assert stats["unique_chunks"] == 0
        assert stats["avg_length"] == 0
        assert stats["total_characters"] == 0

    def test_get_statistics_with_data(self, tmp_path):
        """Test statistics with annotations."""
        manager = AnnotationManager(data_dir=tmp_path)
        manager.add("chunk_017", "Short", "hash1")  # 5 chars
        manager.add("chunk_018", "Medium note", "hash2")  # 11 chars
        manager.add("chunk_017", "Another", "hash1")  # 7 chars

        stats = manager.get_statistics()

        assert stats["total_annotations"] == 3
        assert stats["unique_chunks"] == 2
        assert stats["total_characters"] == 23
        assert 7 < stats["avg_length"] < 8  # 23/3 = 7.67


# ============================================================================
# TestAnnotationValidation
# ============================================================================


class TestAnnotationValidation:
    """Tests for input validation.

    Rule #7: Check parameters
    """

    def test_add_empty_text_raises(self, tmp_path):
        """Test that empty text raises ValueError."""
        manager = AnnotationManager(data_dir=tmp_path)

        with pytest.raises(ValueError, match="cannot be empty"):
            manager.add("chunk_019", "", "hash")

    def test_add_too_long_text_raises(self, tmp_path):
        """Test that text exceeding max length raises ValueError."""
        manager = AnnotationManager(data_dir=tmp_path)
        long_text = "A" * (MAX_ANNOTATION_LENGTH + 1)

        with pytest.raises(ValueError, match="exceeds maximum length"):
            manager.add("chunk_020", long_text, "hash")

    def test_add_max_length_text_succeeds(self, tmp_path):
        """Test that text at max length succeeds."""
        manager = AnnotationManager(data_dir=tmp_path)
        max_text = "A" * MAX_ANNOTATION_LENGTH

        annotation = manager.add("chunk_021", max_text, "hash")

        assert len(annotation.text) == MAX_ANNOTATION_LENGTH

    def test_update_empty_text_raises(self, tmp_path):
        """Test that updating with empty text raises ValueError."""
        manager = AnnotationManager(data_dir=tmp_path)
        created = manager.add("chunk_022", "Original", "hash")

        with pytest.raises(ValueError, match="cannot be empty"):
            manager.update(created.annotation_id, "")
