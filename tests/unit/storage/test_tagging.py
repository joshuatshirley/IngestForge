"""
Tests for Tagging System (ORG-002).

This module tests tag sanitization, CRUD operations, and filtering.

Test Strategy
-------------
- Test tag sanitization utility functions
- Test add_tag/remove_tag operations for both backends
- Test get_chunks_by_tag filtering
- Test get_all_tags retrieval
- Test max tags limit (50 per chunk)
- Test search with tag_filter

Organization
------------
- TestTagSanitization: Tag sanitization utility tests
- TestJSONLTagging: JSONL backend tagging tests
- TestChromaDBTagging: ChromaDB backend tagging tests
"""

from unittest.mock import Mock, patch

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import (
    sanitize_tag,
    validate_tag,
    MAX_TAG_LENGTH,
    MAX_TAGS_PER_CHUNK,
)
from ingestforge.storage.jsonl import JSONLRepository


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(
    chunk_id: str,
    content: str,
    document_id: str = "test_doc",
    tags: list = None,
) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
        source_file="test.txt",
        tags=tags or [],
    )


# ============================================================================
# Test Tag Sanitization
# ============================================================================


class TestTagSanitization:
    """Tests for tag sanitization utility functions.

    Rule #7: Input sanitization is critical for tag system
    """

    def test_sanitize_tag_lowercase(self):
        """Test that tags are converted to lowercase."""
        assert sanitize_tag("IMPORTANT") == "important"
        assert sanitize_tag("MyTag") == "mytag"
        assert sanitize_tag("CamelCase") == "camelcase"

    def test_sanitize_tag_removes_special_chars(self):
        """Test that special characters are removed."""
        assert sanitize_tag("my-tag") == "mytag"
        assert sanitize_tag("my_tag") == "mytag"
        assert sanitize_tag("my.tag") == "mytag"
        assert sanitize_tag("my tag") == "mytag"
        assert sanitize_tag("tag!@#$%") == "tag"

    def test_sanitize_tag_preserves_alphanumeric(self):
        """Test that alphanumeric characters are preserved."""
        assert sanitize_tag("tag123") == "tag123"
        assert sanitize_tag("123tag") == "123tag"
        assert sanitize_tag("abc123xyz") == "abc123xyz"

    def test_sanitize_tag_truncates_to_max_length(self):
        """Test that tags are truncated to MAX_TAG_LENGTH."""
        long_tag = "a" * 100
        result = sanitize_tag(long_tag)
        assert len(result) == MAX_TAG_LENGTH
        assert result == "a" * MAX_TAG_LENGTH

    def test_sanitize_tag_strips_whitespace(self):
        """Test that whitespace is stripped."""
        assert sanitize_tag("  important  ") == "important"
        assert sanitize_tag("\ttag\n") == "tag"

    def test_sanitize_tag_empty_raises_error(self):
        """Test that empty tags raise ValueError."""
        with pytest.raises(ValueError):
            sanitize_tag("")
        with pytest.raises(ValueError):
            sanitize_tag("   ")
        with pytest.raises(ValueError):
            sanitize_tag(None)

    def test_sanitize_tag_only_special_chars_raises_error(self):
        """Test that tags with only special characters raise ValueError."""
        with pytest.raises(ValueError):
            sanitize_tag("!@#$%^&*()")
        with pytest.raises(ValueError):
            sanitize_tag("---")

    def test_validate_tag_valid(self):
        """Test validate_tag with valid tags."""
        is_valid, error = validate_tag("important")
        assert is_valid is True
        assert error == ""

        is_valid, error = validate_tag("tag123")
        assert is_valid is True

    def test_validate_tag_empty(self):
        """Test validate_tag with empty tag."""
        is_valid, error = validate_tag("")
        assert is_valid is False
        assert "empty" in error.lower()

    def test_validate_tag_too_long(self):
        """Test validate_tag with tag exceeding max length."""
        long_tag = "a" * (MAX_TAG_LENGTH + 1)
        is_valid, error = validate_tag(long_tag)
        assert is_valid is False
        assert "length" in error.lower()

    def test_validate_tag_invalid_chars(self):
        """Test validate_tag with invalid characters."""
        is_valid, error = validate_tag("my-tag")
        assert is_valid is False
        assert "lowercase" in error.lower() or "alphanumeric" in error.lower()


# ============================================================================
# Test JSONL Tagging
# ============================================================================


class TestJSONLTagging:
    """Tests for JSONL backend tagging operations.

    Rule #4: Focused test class - tests JSONL tagging
    """

    def test_add_tag_success(self, tmp_path):
        """Test adding a tag to a chunk."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)

        result = repo.add_tag("chunk_1", "important")

        assert result is True
        updated_chunk = repo.get_chunk("chunk_1")
        assert "important" in updated_chunk.tags

    def test_add_tag_sanitizes_input(self, tmp_path):
        """Test that tags are sanitized before adding."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)

        result = repo.add_tag("chunk_1", "IMPORTANT Tag!")

        assert result is True
        updated_chunk = repo.get_chunk("chunk_1")
        assert "importanttag" in updated_chunk.tags

    def test_add_tag_chunk_not_found(self, tmp_path):
        """Test adding tag to non-existent chunk returns False."""
        repo = JSONLRepository(tmp_path)

        result = repo.add_tag("nonexistent", "tag")

        assert result is False

    def test_add_tag_already_exists(self, tmp_path):
        """Test adding duplicate tag returns False."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)
        repo.add_tag("chunk_1", "important")

        result = repo.add_tag("chunk_1", "important")

        assert result is False

    def test_add_tag_max_tags_limit(self, tmp_path):
        """Test that adding more than MAX_TAGS_PER_CHUNK raises error."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)

        # Add MAX_TAGS_PER_CHUNK tags
        for i in range(MAX_TAGS_PER_CHUNK):
            repo.add_tag("chunk_1", f"tag{i}")

        # Adding one more should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            repo.add_tag("chunk_1", "onemore")
        assert "50" in str(exc_info.value) or "maximum" in str(exc_info.value).lower()

    def test_remove_tag_success(self, tmp_path):
        """Test removing a tag from a chunk."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)
        repo.add_tag("chunk_1", "important")

        result = repo.remove_tag("chunk_1", "important")

        assert result is True
        updated_chunk = repo.get_chunk("chunk_1")
        assert "important" not in updated_chunk.tags

    def test_remove_tag_chunk_not_found(self, tmp_path):
        """Test removing tag from non-existent chunk returns False."""
        repo = JSONLRepository(tmp_path)

        result = repo.remove_tag("nonexistent", "tag")

        assert result is False

    def test_remove_tag_not_exists(self, tmp_path):
        """Test removing non-existent tag returns False."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)

        result = repo.remove_tag("chunk_1", "nonexistent")

        assert result is False

    def test_get_chunks_by_tag(self, tmp_path):
        """Test getting chunks filtered by tag."""
        repo = JSONLRepository(tmp_path)

        # Add chunks with different tags
        chunk1 = make_chunk("chunk_1", "Content 1")
        chunk2 = make_chunk("chunk_2", "Content 2")
        chunk3 = make_chunk("chunk_3", "Content 3")
        repo.add_chunk(chunk1)
        repo.add_chunk(chunk2)
        repo.add_chunk(chunk3)

        repo.add_tag("chunk_1", "important")
        repo.add_tag("chunk_2", "important")
        repo.add_tag("chunk_3", "other")

        results = repo.get_chunks_by_tag("important")

        assert len(results) == 2
        chunk_ids = [c.chunk_id for c in results]
        assert "chunk_1" in chunk_ids
        assert "chunk_2" in chunk_ids
        assert "chunk_3" not in chunk_ids

    def test_get_chunks_by_tag_empty_result(self, tmp_path):
        """Test getting chunks by non-existent tag returns empty list."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)

        results = repo.get_chunks_by_tag("nonexistent")

        assert results == []

    def test_get_chunks_by_tag_with_library_filter(self, tmp_path):
        """Test getting chunks by tag with library filter."""
        repo = JSONLRepository(tmp_path)

        chunk1 = make_chunk("chunk_1", "Content 1")
        chunk1.library = "lib1"
        chunk2 = make_chunk("chunk_2", "Content 2")
        chunk2.library = "lib2"

        repo.add_chunk(chunk1)
        repo.add_chunk(chunk2)
        repo.add_tag("chunk_1", "important")
        repo.add_tag("chunk_2", "important")

        results = repo.get_chunks_by_tag("important", library_filter="lib1")

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"

    def test_get_all_tags(self, tmp_path):
        """Test getting all unique tags."""
        repo = JSONLRepository(tmp_path)

        chunk1 = make_chunk("chunk_1", "Content 1")
        chunk2 = make_chunk("chunk_2", "Content 2")
        repo.add_chunk(chunk1)
        repo.add_chunk(chunk2)

        repo.add_tag("chunk_1", "important")
        repo.add_tag("chunk_1", "research")
        repo.add_tag("chunk_2", "important")
        repo.add_tag("chunk_2", "review")

        tags = repo.get_all_tags()

        assert len(tags) == 3
        assert "important" in tags
        assert "research" in tags
        assert "review" in tags
        # Should be sorted
        assert tags == sorted(tags)

    def test_get_all_tags_empty(self, tmp_path):
        """Test getting all tags when none exist."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)

        tags = repo.get_all_tags()

        assert tags == []

    def test_search_with_tag_filter(self, tmp_path):
        """Test search with tag filter."""
        repo = JSONLRepository(tmp_path)

        chunk1 = make_chunk("chunk_1", "Machine learning is great")
        chunk2 = make_chunk("chunk_2", "Machine learning is awesome")
        chunk3 = make_chunk("chunk_3", "Machine learning is cool")
        repo.add_chunk(chunk1)
        repo.add_chunk(chunk2)
        repo.add_chunk(chunk3)

        repo.add_tag("chunk_1", "important")
        repo.add_tag("chunk_2", "important")

        # Search with tag filter
        results = repo.search("machine learning", tag_filter="important")

        assert len(results) == 2
        chunk_ids = [r.chunk_id for r in results]
        assert "chunk_1" in chunk_ids
        assert "chunk_2" in chunk_ids
        assert "chunk_3" not in chunk_ids

    def test_tags_persist_after_reload(self, tmp_path):
        """Test that tags are persisted and survive reload."""
        repo = JSONLRepository(tmp_path)
        chunk = make_chunk("chunk_1", "Test content")
        repo.add_chunk(chunk)
        repo.add_tag("chunk_1", "important")

        # Create new repository instance (simulates reload)
        repo2 = JSONLRepository(tmp_path)

        loaded_chunk = repo2.get_chunk("chunk_1")
        assert "important" in loaded_chunk.tags


# ============================================================================
# Test ChromaDB Tagging
# ============================================================================


class TestChromaDBTagging:
    """Tests for ChromaDB backend tagging operations.

    Rule #4: Focused test class - tests ChromaDB tagging
    Uses mocking to avoid external ChromaDB dependency.
    """

    @patch("chromadb.PersistentClient")
    def test_add_tag_success(self, mock_client_cls, tmp_path):
        """Test adding a tag to a chunk in ChromaDB."""
        import json
        from ingestforge.storage.chromadb import ChromaDBRepository

        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        # Mock get() to return existing chunk
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"document_id": "doc_1", "tags_json": "[]"}],
        }

        repo = ChromaDBRepository(tmp_path / "chroma")

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            result = repo.add_tag("chunk_1", "important")

        assert result is True
        # Verify update was called with new tags
        mock_collection.update.assert_called_once()
        call_kwargs = mock_collection.update.call_args[1]
        assert "chunk_1" in call_kwargs["ids"]
        metadata = call_kwargs["metadatas"][0]
        assert json.loads(metadata["tags_json"]) == ["important"]

    @patch("chromadb.PersistentClient")
    def test_remove_tag_success(self, mock_client_cls, tmp_path):
        """Test removing a tag from a chunk in ChromaDB."""
        import json
        from ingestforge.storage.chromadb import ChromaDBRepository

        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        # Mock get() to return chunk with existing tags
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [
                {
                    "document_id": "doc_1",
                    "tags_json": json.dumps(["important", "research"]),
                }
            ],
        }

        repo = ChromaDBRepository(tmp_path / "chroma")

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            result = repo.remove_tag("chunk_1", "important")

        assert result is True
        # Verify update was called with tag removed
        mock_collection.update.assert_called_once()
        call_kwargs = mock_collection.update.call_args[1]
        metadata = call_kwargs["metadatas"][0]
        assert json.loads(metadata["tags_json"]) == ["research"]

    @patch("chromadb.PersistentClient")
    def test_get_all_tags(self, mock_client_cls, tmp_path):
        """Test getting all tags from ChromaDB."""
        import json
        from ingestforge.storage.chromadb import ChromaDBRepository

        # Setup mocks
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        # Mock get() to return chunks with various tags
        mock_collection.get.return_value = {
            "ids": ["chunk_1", "chunk_2"],
            "metadatas": [
                {"tags_json": json.dumps(["important", "research"])},
                {"tags_json": json.dumps(["important", "review"])},
            ],
        }

        repo = ChromaDBRepository(tmp_path / "chroma")

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            tags = repo.get_all_tags()

        assert len(tags) == 3
        assert "important" in tags
        assert "research" in tags
        assert "review" in tags


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Tag Sanitization: 11 tests (lowercase, special chars, length, validation)
    - JSONL Tagging: 13 tests (add/remove/get/search/persistence)
    - ChromaDB Tagging: 3 tests (add/remove/get_all with mocking)

    Total: 27 tests

Design Decisions:
    1. Focus on tag sanitization as Rule #7 compliance
    2. Test both success and error cases
    3. Test max tags limit (50 per chunk)
    4. Test persistence across repository reloads
    5. Test search with tag filtering
    6. Mock ChromaDB to avoid external dependencies
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - Tag sanitization (lowercase, alphanumeric, truncation)
    - Invalid tag handling (empty, special chars only)
    - Adding tags to chunks
    - Removing tags from chunks
    - Max tags per chunk limit (50)
    - Getting chunks by tag
    - Library filtering with tag filtering
    - Getting all unique tags
    - Search with tag filter
    - Tag persistence across reloads
"""
