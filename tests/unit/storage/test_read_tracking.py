"""
Tests for Read/Unread Tracking (ORG-001).

This module tests the is_read field and mark_read() functionality
across storage backends (JSONL and ChromaDB).

Test Strategy
-------------
- Test mark_read for single chunks
- Test get_unread_chunks filtering
- Test idempotent marking (re-marking is safe)
- Test error handling for non-existent chunks
- Performance: metadata update should complete in <100ms

Organization
------------
- TestJSONLReadTracking: JSONL backend tests
- TestChromaDBReadTracking: ChromaDB backend tests
- TestMarkCommand: CLI command tests
"""

from unittest.mock import Mock, patch
import time

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.jsonl import JSONLRepository
from ingestforge.storage.chromadb import ChromaDBRepository


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(
    chunk_id: str,
    document_id: str = "test_doc",
    content: str = "",
    library: str = "default",
    is_read: bool = False,
) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"Content for {chunk_id}",
        library=library,
        word_count=len((content or "").split()),
        char_count=len(content or ""),
        is_read=is_read,
    )


# ============================================================================
# JSONL Backend Tests
# ============================================================================


class TestJSONLReadTracking:
    """Tests for read/unread tracking in JSONL backend.

    Rule #4: Focused test class
    """

    def test_mark_chunk_as_read(self, tmp_path):
        """Test marking a chunk as read."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        # Add an unread chunk
        chunk = make_chunk("chunk_1", is_read=False)
        repo.add_chunk(chunk)

        # Mark as read
        result = repo.mark_read("chunk_1", True)

        assert result is True
        updated_chunk = repo.get_chunk("chunk_1")
        assert updated_chunk.is_read is True

    def test_mark_chunk_as_unread(self, tmp_path):
        """Test marking a chunk as unread."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        # Add a read chunk
        chunk = make_chunk("chunk_1", is_read=True)
        repo.add_chunk(chunk)

        # Mark as unread
        result = repo.mark_read("chunk_1", False)

        assert result is True
        updated_chunk = repo.get_chunk("chunk_1")
        assert updated_chunk.is_read is False

    def test_mark_read_idempotent(self, tmp_path):
        """Test that marking a read chunk as read again is idempotent."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        chunk = make_chunk("chunk_1", is_read=True)
        repo.add_chunk(chunk)

        # Mark as read again (should succeed without error)
        result = repo.mark_read("chunk_1", True)

        assert result is True
        updated_chunk = repo.get_chunk("chunk_1")
        assert updated_chunk.is_read is True

    def test_mark_read_nonexistent_chunk(self, tmp_path):
        """Test marking non-existent chunk returns False."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        result = repo.mark_read("nonexistent", True)

        assert result is False

    def test_mark_read_validates_chunk_id(self, tmp_path):
        """Test that empty chunk_id raises ValueError."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            repo.mark_read("", True)

        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            repo.mark_read(None, True)

    def test_mark_read_validates_status_type(self, tmp_path):
        """Test that non-bool status raises ValueError."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)
        repo.add_chunk(make_chunk("chunk_1"))

        with pytest.raises(ValueError, match="status must be bool"):
            repo.mark_read("chunk_1", "true")

    def test_get_unread_chunks(self, tmp_path):
        """Test getting all unread chunks."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        # Add mix of read and unread chunks
        repo.add_chunk(make_chunk("chunk_1", is_read=False))
        repo.add_chunk(make_chunk("chunk_2", is_read=True))
        repo.add_chunk(make_chunk("chunk_3", is_read=False))

        unread = repo.get_unread_chunks()

        assert len(unread) == 2
        chunk_ids = {c.chunk_id for c in unread}
        assert chunk_ids == {"chunk_1", "chunk_3"}

    def test_get_unread_chunks_with_library_filter(self, tmp_path):
        """Test getting unread chunks filtered by library."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", library="lib1", is_read=False))
        repo.add_chunk(make_chunk("chunk_2", library="lib2", is_read=False))
        repo.add_chunk(make_chunk("chunk_3", library="lib1", is_read=True))

        unread = repo.get_unread_chunks(library_filter="lib1")

        assert len(unread) == 1
        assert unread[0].chunk_id == "chunk_1"

    def test_mark_read_performance(self, tmp_path):
        """Test that mark_read completes in <100ms."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        # Add a chunk
        repo.add_chunk(make_chunk("chunk_1"))

        # Time the mark_read operation
        start = time.perf_counter()
        repo.mark_read("chunk_1", True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100, f"mark_read took {elapsed_ms:.2f}ms, expected <100ms"

    def test_is_read_persists_to_file(self, tmp_path):
        """Test that is_read status persists when reloading."""
        data_path = tmp_path / "data"

        # Create repo and add chunk
        repo1 = JSONLRepository(data_path)
        repo1.add_chunk(make_chunk("chunk_1", is_read=False))
        repo1.mark_read("chunk_1", True)

        # Create new repo (reloads from file)
        repo2 = JSONLRepository(data_path)
        chunk = repo2.get_chunk("chunk_1")

        assert chunk.is_read is True


# ============================================================================
# ChromaDB Backend Tests
# ============================================================================


class TestChromaDBReadTracking:
    """Tests for read/unread tracking in ChromaDB backend.

    Rule #4: Focused test class
    """

    @patch("chromadb.PersistentClient")
    def test_mark_chunk_as_read(self, mock_client_cls, tmp_path):
        """Test marking a chunk as read in ChromaDB."""
        persist_dir = tmp_path / "chroma"
        mock_client = Mock()
        mock_collection = Mock()

        # Setup mock to return chunk on get
        mock_collection.get.return_value = {
            "ids": ["chunk_1"],
            "metadatas": [{"document_id": "doc_1", "is_read": False}],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        repo = ChromaDBRepository(persist_dir)

        # Mark as read
        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            result = repo.mark_read("chunk_1", True)

        assert result is True
        mock_collection.update.assert_called_once()
        update_call = mock_collection.update.call_args
        assert update_call[1]["metadatas"][0]["is_read"] is True

    @patch("chromadb.PersistentClient")
    def test_mark_read_nonexistent_chunk(self, mock_client_cls, tmp_path):
        """Test marking non-existent chunk returns False."""
        persist_dir = tmp_path / "chroma"
        mock_client = Mock()
        mock_collection = Mock()

        # Setup mock to return empty result
        mock_collection.get.return_value = {
            "ids": [],
            "metadatas": [],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        repo = ChromaDBRepository(persist_dir)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            result = repo.mark_read("nonexistent", True)

        assert result is False

    @patch("chromadb.PersistentClient")
    def test_mark_read_validates_chunk_id(self, mock_client_cls, tmp_path):
        """Test that empty chunk_id raises ValueError."""
        persist_dir = tmp_path / "chroma"
        mock_client_cls.return_value = Mock()

        repo = ChromaDBRepository(persist_dir)

        with pytest.raises(ValueError, match="chunk_id cannot be empty"):
            repo.mark_read("", True)

    @patch("chromadb.PersistentClient")
    def test_is_read_stored_in_metadata(self, mock_client_cls, tmp_path):
        """Test that is_read is stored in chunk metadata."""
        persist_dir = tmp_path / "chroma"
        mock_client = Mock()
        mock_collection = Mock()
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        repo = ChromaDBRepository(persist_dir)
        chunk = make_chunk("chunk_1", is_read=True)

        # Add chunk
        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            repo.add_chunk(chunk)

        # Verify is_read was included in metadata
        add_call = mock_collection.add.call_args
        metadata = add_call[1]["metadatas"][0]
        assert metadata["is_read"] is True

    @patch("chromadb.PersistentClient")
    def test_get_unread_chunks(self, mock_client_cls, tmp_path):
        """Test getting all unread chunks from ChromaDB."""
        persist_dir = tmp_path / "chroma"
        mock_client = Mock()
        mock_collection = Mock()

        # Setup mock to return unread chunks
        mock_collection.get.return_value = {
            "ids": ["chunk_1", "chunk_3"],
            "documents": ["Content 1", "Content 3"],
            "metadatas": [
                {"document_id": "doc_1", "is_read": False, "source_file": "test.txt"},
                {"document_id": "doc_1", "is_read": False, "source_file": "test.txt"},
            ],
        }
        mock_client.get_or_create_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        repo = ChromaDBRepository(persist_dir)

        with patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            unread = repo.get_unread_chunks()

        assert len(unread) == 2
        # Verify the where clause was correct
        get_call = mock_collection.get.call_args
        where_clause = get_call[1]["where"]
        assert where_clause == {"is_read": {"$ne": True}}


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - JSONL Read Tracking: 10 tests
        - mark_chunk_as_read
        - mark_chunk_as_unread
        - mark_read_idempotent
        - mark_read_nonexistent_chunk
        - mark_read_validates_chunk_id
        - mark_read_validates_status_type
        - get_unread_chunks
        - get_unread_chunks_with_library_filter
        - mark_read_performance (<100ms)
        - is_read_persists_to_file

    - ChromaDB Read Tracking: 5 tests
        - mark_chunk_as_read
        - mark_read_nonexistent_chunk
        - mark_read_validates_chunk_id
        - is_read_stored_in_metadata
        - get_unread_chunks

    Total: 15 tests

Design Decisions:
    1. Test both success and error cases
    2. Test parameter validation (Rule #7)
    3. Test idempotent behavior
    4. Test performance requirement (<100ms)
    5. Test persistence across sessions
    6. Test library filtering
    7. Mock ChromaDB to avoid external dependencies

Behaviors Tested:
    - Marking chunks as read/unread
    - Retrieving only unread chunks
    - Parameter validation (empty chunk_id, non-bool status)
    - Idempotent operations (re-marking is safe)
    - Performance requirements
    - Persistence across repository instances
    - Library filtering for unread chunks
"""
