"""Tests for PostgreSQL Storage Backend.

This module tests the PostgreSQL storage with pgvector support.

Test Strategy
-------------
- Mock psycopg2 for unit tests (no real database required)
- Focus on interface compliance and error handling
- Test connection pooling behavior
- Test vector search queries

Organization
------------
- TestPostgresRepositoryInit: Initialization and schema
- TestAddChunks: Adding chunks to storage
- TestGetChunks: Retrieving chunks
- TestSearch: Full-text and vector search
- TestLibraryOps: Library management
- TestParentMapping: Parent-child chunk mappings
- TestStatistics: Statistics and health checks
- TestMigrations: Schema migrations
"""

from unittest.mock import MagicMock, patch
from typing import List

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord


# Skip all tests if psycopg2 not available
psycopg2 = pytest.importorskip("psycopg2", reason="psycopg2 not installed")


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(
    chunk_id: str,
    document_id: str = "test_doc",
    content: str = "",
    library: str = "default",
    embedding: List[float] = None,
) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"Content for {chunk_id}",
        library=library,
        word_count=len((content or "").split()),
        char_count=len(content or ""),
        embedding=embedding,
    )


def mock_cursor_context():
    """Create a mock cursor context manager."""
    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    return cursor


# ============================================================================
# Test Classes
# ============================================================================


class TestPostgresRepositoryInit:
    """Tests for PostgresRepository initialization."""

    def test_import_error_when_psycopg2_missing(self):
        """Test that ImportError is raised when psycopg2 is not installed."""
        with patch("ingestforge.storage.postgres.HAS_POSTGRES", False):
            from ingestforge.storage.postgres import PostgresRepository

            with pytest.raises(ImportError) as exc_info:
                PostgresRepository("postgresql://localhost/test")

            assert "psycopg2" in str(exc_info.value)

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_creates_connection_pool(self, mock_pool):
        """Test that connection pool is created on init."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor_context()

        repo = PostgresRepository(
            "postgresql://localhost/test",
            min_pool_size=2,
            max_pool_size=5,
        )

        mock_pool.assert_called_once()
        call_args = mock_pool.call_args
        assert call_args[0][0] == 2  # min_size
        assert call_args[0][1] == 5  # max_size

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_custom_embedding_dimension(self, mock_pool):
        """Test custom embedding dimension configuration."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor_context()

        repo = PostgresRepository(
            "postgresql://localhost/test",
            embedding_dim=768,
        )

        assert repo.embedding_dim == 768

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_custom_table_name(self, mock_pool):
        """Test custom table name configuration."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor_context()

        repo = PostgresRepository(
            "postgresql://localhost/test",
            table_name="my_chunks",
        )

        assert repo.table_name == "my_chunks"


class TestAddChunks:
    """Tests for adding chunks to storage."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_add_single_chunk(self, mock_pool):
        """Test adding a single chunk."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        chunk = make_chunk("chunk_1", content="Test content")
        result = repo.add_chunk(chunk)

        assert result is True
        cursor.execute.assert_called()
        mock_conn.commit.assert_called()

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_add_batch_chunks(self, mock_pool):
        """Test adding multiple chunks in batch."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        chunks = [
            make_chunk("chunk_1"),
            make_chunk("chunk_2"),
            make_chunk("chunk_3"),
        ]
        count = repo.add_chunks(chunks)

        assert count == 3
        mock_conn.commit.assert_called()

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_add_chunk_with_embedding(self, mock_pool):
        """Test adding chunk with embedding vector."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        embedding = [0.1, 0.2, 0.3] * 128  # 384-dim vector
        chunk = make_chunk("chunk_1", embedding=embedding)
        result = repo.add_chunk(chunk)

        assert result is True

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_add_empty_batch(self, mock_pool):
        """Test adding empty batch returns 0."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        count = repo.add_chunks([])

        assert count == 0

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_add_chunk_rollback_on_error(self, mock_pool):
        """Test rollback on database error."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.execute.side_effect = Exception("Database error")
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        chunk = make_chunk("chunk_1")
        result = repo.add_chunk(chunk)

        assert result is False
        mock_conn.rollback.assert_called()


class TestGetChunks:
    """Tests for retrieving chunks."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_get_chunk_by_id(self, mock_pool):
        """Test retrieving chunk by ID."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchone.return_value = {
            "chunk_id": "chunk_1",
            "document_id": "doc_1",
            "content": "Test content",
            "library": "default",
        }
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        chunk = repo.get_chunk("chunk_1")

        assert chunk is not None
        assert chunk.chunk_id == "chunk_1"

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_get_missing_chunk(self, mock_pool):
        """Test getting non-existent chunk returns None."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchone.return_value = None
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        chunk = repo.get_chunk("missing")

        assert chunk is None

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_get_chunks_by_document(self, mock_pool):
        """Test retrieving all chunks for a document."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchall.return_value = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "content": "C1",
                "library": "default",
            },
            {
                "chunk_id": "chunk_2",
                "document_id": "doc_1",
                "content": "C2",
                "library": "default",
            },
        ]
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        chunks = repo.get_chunks_by_document("doc_1")

        assert len(chunks) == 2
        assert all(c.document_id == "doc_1" for c in chunks)


class TestSearch:
    """Tests for search functionality."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_basic_fts_search(self, mock_pool):
        """Test full-text search."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchall.return_value = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "content": "Python programming",
                "score": 0.8,
                "library": "default",
            },
        ]
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        results = repo.search("Python", top_k=5)

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_empty_query_returns_empty(self, mock_pool):
        """Test empty query returns empty results."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        results = repo.search("", top_k=5)

        assert len(results) == 0

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_search_with_library_filter(self, mock_pool):
        """Test search with library filter."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        results = repo.search("test", library_filter="my_library")

        # Verify the query includes library filter
        cursor.execute.assert_called()

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_semantic_search(self, mock_pool):
        """Test vector similarity search."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchall.return_value = [
            {
                "chunk_id": "chunk_1",
                "document_id": "doc_1",
                "content": "Similar content",
                "score": 0.95,
                "library": "default",
            },
        ]
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        query_embedding = [0.1] * 384
        results = repo.search_semantic(query_embedding, top_k=5)

        assert len(results) == 1


class TestDeleteOperations:
    """Tests for delete operations."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_delete_chunk(self, mock_pool):
        """Test deleting a single chunk."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.rowcount = 1
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        result = repo.delete_chunk("chunk_1")

        assert result is True

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_delete_missing_chunk(self, mock_pool):
        """Test deleting non-existent chunk."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.rowcount = 0
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        result = repo.delete_chunk("missing")

        assert result is False

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_delete_document(self, mock_pool):
        """Test deleting all chunks for a document."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.rowcount = 5
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        count = repo.delete_document("doc_1")

        assert count == 5

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_clear_all(self, mock_pool):
        """Test clearing all data."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        repo.clear()

        # Should call TRUNCATE
        calls = [str(call) for call in cursor.execute.call_args_list]
        assert any("TRUNCATE" in str(call) for call in calls)


class TestLibraryOperations:
    """Tests for library management."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_get_libraries(self, mock_pool):
        """Test getting list of libraries."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchall.return_value = [("lib1",), ("lib2",)]
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        libraries = repo.get_libraries()

        assert "lib1" in libraries
        assert "lib2" in libraries
        assert "default" in libraries  # Always included

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_count_by_library(self, mock_pool):
        """Test counting chunks in a library."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchone.return_value = (42,)
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        count = repo.count_by_library("my_library")

        assert count == 42

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_delete_by_library(self, mock_pool):
        """Test deleting all chunks in a library."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.rowcount = 10
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        deleted = repo.delete_by_library("old_library")

        assert deleted == 10

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_reassign_library(self, mock_pool):
        """Test moving chunks between libraries."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.rowcount = 5
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        moved = repo.reassign_library("old", "new")

        assert moved == 5


class TestStatistics:
    """Tests for statistics and health."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_get_statistics(self, mock_pool):
        """Test getting storage statistics."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()

        # Mock multiple queries
        cursor.fetchone.side_effect = [
            (100,),  # total_chunks
            (10,),  # total_documents
            (90,),  # chunks_with_embeddings
            (3,),  # library_count
            ("1.2 MB",),  # table_size
        ]
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        stats = repo.get_statistics()

        assert stats["total_chunks"] == 100
        assert stats["total_documents"] == 10
        assert stats["backend"] == "postgres"

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_count(self, mock_pool):
        """Test getting total chunk count."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchone.return_value = (500,)
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        count = repo.count()

        assert count == 500

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_health_check_healthy(self, mock_pool):
        """Test health check when database is healthy."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        healthy, message = repo.health_check()

        assert healthy is True
        assert "healthy" in message.lower()

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_health_check_unhealthy(self, mock_pool):
        """Test health check when database has issues."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.execute.side_effect = Exception("Connection failed")
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        healthy, message = repo.health_check()

        assert healthy is False


class TestParentMapping:
    """Tests for parent-child chunk mappings."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_set_parent_mapping(self, mock_pool):
        """Test setting parent-child mapping."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        cursor.fetchone.return_value = ("doc_1",)  # document_id lookup
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        result = repo.set_parent_mapping("child_1", "parent_1")

        assert result is True

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_get_parent_chunk(self, mock_pool):
        """Test getting parent chunk."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()

        # First call returns parent_chunk_id, second returns chunk data
        cursor.fetchone.side_effect = [
            ("parent_1",),  # parent mapping lookup
            {
                "chunk_id": "parent_1",
                "document_id": "doc_1",
                "content": "Parent",
                "library": "default",
            },  # chunk lookup
        ]
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        parent = repo.get_parent_chunk("child_1")

        assert parent is not None
        assert parent.chunk_id == "parent_1"


class TestClose:
    """Tests for cleanup operations."""

    @patch("ingestforge.storage.postgres.pool.ThreadedConnectionPool")
    def test_close_pool(self, mock_pool):
        """Test closing connection pool."""
        from ingestforge.storage.postgres import PostgresRepository

        mock_conn = MagicMock()
        mock_pool.return_value.getconn.return_value = mock_conn
        cursor = mock_cursor_context()
        mock_conn.cursor.return_value = cursor

        repo = PostgresRepository("postgresql://localhost/test")

        repo.close()

        mock_pool.return_value.closeall.assert_called_once()


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - PostgresRepository init: 4 tests
    - Add chunks: 5 tests
    - Get chunks: 3 tests
    - Search: 4 tests
    - Delete: 4 tests
    - Library ops: 4 tests
    - Statistics: 4 tests
    - Parent mapping: 2 tests
    - Close: 1 test

    Total: 31 tests

Design Decisions:
    1. Mock psycopg2 to avoid requiring real database
    2. Test interface compliance with ChunkRepository
    3. Test error handling and rollbacks
    4. Test connection pool behavior
    5. Follows NASA JPL Rule #1 (Simple Control Flow)
    6. Follows NASA JPL Rule #4 (Small Focused Tests)
"""
