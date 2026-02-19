"""
Tests for JSONL Storage Backend.

This module tests file-based JSONL storage with in-memory indexing.

Test Strategy
-------------
- Focus on file I/O, parsing, and indexing
- Test BM25-style search
- Test library operations
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Use tmp_path fixture for isolated file operations

Organization
------------
- TestJSONLRepositoryInit: Initialization
- TestFileLoading: File reading and parsing
- TestAddGet: Adding and retrieving chunks
- TestDocumentOps: Document-level operations
- TestSearch: BM25 search functionality
- TestDelete: Deletion operations
- TestStatistics: Statistics and metadata
- TestLibraryOps: Library management
"""

import json
from pathlib import Path
from typing import List


from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.jsonl import JSONLRepository


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(
    chunk_id: str,
    document_id: str = "test_doc",
    content: str = "",
    library: str = "default",
) -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content or f"Content for {chunk_id}",
        library=library,
        word_count=len((content or "").split()),
        char_count=len(content or ""),
    )


def write_jsonl(data_path: Path, chunks: List[ChunkRecord]) -> None:
    """Write chunks to JSONL file in correct directory structure."""
    chunks_file = data_path / "chunks" / "chunks.jsonl"
    chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            data = chunk.to_dict()
            f.write(json.dumps(data) + "\n")


# ============================================================================
# Test Classes
# ============================================================================


class TestJSONLRepositoryInit:
    """Tests for JSONLRepository initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_with_defaults(self, tmp_path):
        """Test creating repository with default data path."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        assert repo.data_path == data_path
        assert repo.chunks_file == data_path / "chunks" / "chunks.jsonl"
        assert repo.count() == 0

    def test_create_with_custom_path(self, tmp_path):
        """Test creating repository with custom path."""
        custom_path = tmp_path / "custom"
        repo = JSONLRepository(custom_path)

        assert repo.data_path == custom_path
        assert repo.chunks_file == custom_path / "chunks" / "chunks.jsonl"


class TestFileLoading:
    """Tests for file loading and parsing.

    Rule #4: Focused test class - tests _load() and parsing
    """

    def test_load_from_existing_file(self, tmp_path):
        """Test loading chunks from existing JSONL file."""
        data_path = tmp_path / "data"
        chunks = [
            make_chunk("chunk_1", content="First chunk"),
            make_chunk("chunk_2", content="Second chunk"),
        ]
        write_jsonl(data_path, chunks)

        repo = JSONLRepository(data_path)

        assert repo.count() == 2
        assert repo.get_chunk("chunk_1") is not None
        assert repo.get_chunk("chunk_2") is not None

    def test_load_handles_invalid_json(self, tmp_path):
        """Test loading gracefully skips invalid JSON lines."""
        data_path = tmp_path / "data"
        chunks_file = data_path / "chunks" / "chunks.jsonl"
        chunks_file.parent.mkdir(parents=True)
        with open(chunks_file, "w") as f:
            # Valid chunk
            chunk1 = make_chunk("chunk_1")
            f.write(json.dumps(chunk1.to_dict()) + "\n")
            # Invalid JSON
            f.write('{"invalid": json}\n')
            # Valid chunk
            chunk2 = make_chunk("chunk_2")
            f.write(json.dumps(chunk2.to_dict()) + "\n")

        repo = JSONLRepository(data_path)

        # Should load 2 valid chunks, skip 1 invalid
        assert repo.count() == 2

    def test_load_missing_file(self, tmp_path):
        """Test loading from non-existent file creates empty repository."""
        data_path = tmp_path / "missing"
        repo = JSONLRepository(data_path)

        assert repo.count() == 0


class TestAddGet:
    """Tests for adding and retrieving chunks.

    Rule #4: Focused test class - tests add/get operations
    """

    def test_add_single_chunk(self, tmp_path):
        """Test adding a single chunk."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        chunk = make_chunk("chunk_1", content="Test content")
        result = repo.add_chunk(chunk)

        assert result is True
        assert repo.count() == 1
        assert repo.get_chunk("chunk_1") == chunk

    def test_add_batch_chunks(self, tmp_path):
        """Test adding multiple chunks in batch."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        chunks = [
            make_chunk("chunk_1"),
            make_chunk("chunk_2"),
            make_chunk("chunk_3"),
        ]
        count = repo.add_chunks(chunks)

        assert count == 3
        assert repo.count() == 3

    def test_get_chunk_by_id(self, tmp_path):
        """Test retrieving chunk by ID."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)
        chunk = make_chunk("chunk_1", content="Specific content")
        repo.add_chunk(chunk)

        retrieved = repo.get_chunk("chunk_1")

        assert retrieved is not None
        assert retrieved.chunk_id == "chunk_1"
        assert retrieved.content == "Specific content"

    def test_get_missing_chunk(self, tmp_path):
        """Test getting non-existent chunk returns None."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        result = repo.get_chunk("missing_id")

        assert result is None


class TestDocumentOps:
    """Tests for document-level operations.

    Rule #4: Focused test class - tests document operations
    """

    def test_get_chunks_by_document(self, tmp_path):
        """Test retrieving all chunks for a document."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        # Add chunks from two documents
        repo.add_chunk(make_chunk("chunk_1", document_id="doc_1"))
        repo.add_chunk(make_chunk("chunk_2", document_id="doc_1"))
        repo.add_chunk(make_chunk("chunk_3", document_id="doc_2"))

        doc1_chunks = repo.get_chunks_by_document("doc_1")

        assert len(doc1_chunks) == 2
        assert all(c.document_id == "doc_1" for c in doc1_chunks)

    def test_delete_document(self, tmp_path):
        """Test deleting all chunks for a document."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", document_id="doc_1"))
        repo.add_chunk(make_chunk("chunk_2", document_id="doc_1"))
        repo.add_chunk(make_chunk("chunk_3", document_id="doc_2"))

        deleted_count = repo.delete_document("doc_1")

        assert deleted_count == 2
        assert repo.count() == 1
        assert repo.get_chunk("chunk_3") is not None


class TestSearch:
    """Tests for BM25 search functionality.

    Rule #4: Focused test class - tests search()
    """

    def test_basic_search(self, tmp_path):
        """Test basic keyword search."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", content="Python programming language"))
        repo.add_chunk(make_chunk("chunk_2", content="Java development tools"))
        repo.add_chunk(make_chunk("chunk_3", content="Python data science"))

        results = repo.search("Python", top_k=5)

        # Should return chunks with "Python" ranked higher
        assert len(results) > 0
        assert results[0].chunk_id in ["chunk_1", "chunk_3"]

    def test_search_with_document_filter(self, tmp_path):
        """Test search with document filter."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(
            make_chunk("chunk_1", document_id="doc_1", content="Python code")
        )
        repo.add_chunk(
            make_chunk("chunk_2", document_id="doc_2", content="Python script")
        )

        results = repo.search("Python", top_k=5, document_filter="doc_1")

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"

    def test_search_with_library_filter(self, tmp_path):
        """Test search with library filter."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", library="lib1", content="Test content"))
        repo.add_chunk(make_chunk("chunk_2", library="lib2", content="Test content"))

        results = repo.search("Test", top_k=5, library_filter="lib1")

        assert len(results) == 1
        assert results[0].chunk_id == "chunk_1"

    def test_search_empty_query(self, tmp_path):
        """Test search with empty query returns empty list."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1"))

        results = repo.search("", top_k=5)

        assert len(results) == 0


class TestDelete:
    """Tests for deletion operations.

    Rule #4: Focused test class - tests delete operations
    """

    def test_delete_single_chunk(self, tmp_path):
        """Test deleting a single chunk."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1"))
        repo.add_chunk(make_chunk("chunk_2"))

        result = repo.delete_chunk("chunk_1")

        assert result is True
        assert repo.count() == 1
        assert repo.get_chunk("chunk_1") is None

    def test_delete_missing_chunk(self, tmp_path):
        """Test deleting non-existent chunk returns False."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        result = repo.delete_chunk("missing_id")

        assert result is False

    def test_clear_all_data(self, tmp_path):
        """Test clearing all chunks and deleting file."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1"))
        repo.add_chunk(make_chunk("chunk_2"))

        repo.clear()

        assert repo.count() == 0
        assert not repo.chunks_file.exists()


class TestStatistics:
    """Tests for statistics and metadata.

    Rule #4: Focused test class - tests get_statistics()
    """

    def test_get_statistics(self, tmp_path):
        """Test retrieving storage statistics."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", content="Python programming"))
        repo.add_chunk(make_chunk("chunk_2", content="Java development"))

        stats = repo.get_statistics()

        assert stats["total_chunks"] == 2
        assert stats["total_documents"] == 1  # Both chunks from "test_doc"
        assert stats["index_terms"] > 0
        assert stats["storage_file"] == str(repo.chunks_file)
        assert stats["file_size_bytes"] > 0


class TestLibraryOps:
    """Tests for library management operations.

    Rule #4: Focused test class - tests library operations
    """

    def test_get_libraries(self, tmp_path):
        """Test getting list of unique library names."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", library="lib1"))
        repo.add_chunk(make_chunk("chunk_2", library="lib2"))

        libraries = repo.get_libraries()

        assert "default" in libraries  # Always includes default
        assert "lib1" in libraries
        assert "lib2" in libraries

    def test_count_by_library(self, tmp_path):
        """Test counting chunks in a specific library."""
        data_path = tmp_path / "data"
        repo = JSONLRepository(data_path)

        repo.add_chunk(make_chunk("chunk_1", library="lib1"))
        repo.add_chunk(make_chunk("chunk_2", library="lib1"))
        repo.add_chunk(make_chunk("chunk_3", library="lib2"))

        count = repo.count_by_library("lib1")

        assert count == 2


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - JSONLRepository init: 2 tests (defaults, custom path)
    - File loading: 3 tests (load file, invalid JSON, missing file)
    - Add/Get: 4 tests (add single, add batch, get by ID, get missing)
    - Document ops: 2 tests (get by document, delete document)
    - Search: 4 tests (basic, document filter, library filter, empty query)
    - Delete: 3 tests (delete chunk, delete missing, clear all)
    - Statistics: 1 test (get statistics)
    - Library ops: 2 tests (get libraries, count by library)

    Total: 21 tests

Design Decisions:
    1. Use tmp_path fixture for isolated file operations
    2. Helper functions for creating test chunks and writing JSONL
    3. Focus on file I/O, parsing, and BM25 search
    4. Test error handling (invalid JSON, missing files)
    5. Test filtering (document, library)
    6. Simple, clear tests that verify storage works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - Repository initialization with default/custom paths
    - Loading chunks from existing JSONL file
    - Parsing valid/invalid JSON lines
    - Handling missing files gracefully
    - Adding single and batch chunks
    - Retrieving chunks by ID
    - Getting all chunks for a document
    - Deleting chunks and documents
    - BM25-style keyword search
    - Search filters (document, library)
    - Empty query handling
    - Clearing all data and deleting file
    - Storage statistics
    - Library management (get, count)

Justification:
    - JSONL storage is critical for file-based persistence
    - BM25 search provides keyword-based retrieval
    - Library operations enable multi-tenant usage
    - Simple tests verify storage system works correctly
"""
