"""Tests for ChunkRecord author fields (TICKET-301).

Verifies:
1. ChunkRecord includes author_id and author_name fields
2. Author fields are preserved in to_dict() serialization
3. Author fields are restored in from_dict() deserialization
4. SearchResult includes author fields when created from ChunkRecord
"""


from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.storage.base import SearchResult


class TestChunkRecordAuthorFields:
    """Test ChunkRecord author_id and author_name fields."""

    def test_create_with_author_fields(self) -> None:
        """Create ChunkRecord with author fields."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="john.doe@example.com",
            author_name="John Doe",
        )
        assert chunk.author_id == "john.doe@example.com"
        assert chunk.author_name == "John Doe"

    def test_default_author_fields_none(self) -> None:
        """Author fields default to None."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        assert chunk.author_id is None
        assert chunk.author_name is None

    def test_to_dict_includes_author_fields(self) -> None:
        """to_dict() includes author fields."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="user123",
            author_name="Jane Smith",
        )
        data = chunk.to_dict()
        assert data["author_id"] == "user123"
        assert data["author_name"] == "Jane Smith"

    def test_to_dict_with_none_author(self) -> None:
        """to_dict() includes None author fields."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        data = chunk.to_dict()
        assert "author_id" in data
        assert "author_name" in data
        assert data["author_id"] is None
        assert data["author_name"] is None

    def test_from_dict_restores_author_fields(self) -> None:
        """from_dict() restores author fields."""
        data = {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "content": "Test content",
            "author_id": "user123",
            "author_name": "Jane Smith",
        }
        chunk = ChunkRecord.from_dict(data)
        assert chunk.author_id == "user123"
        assert chunk.author_name == "Jane Smith"

    def test_from_dict_without_author_fields(self) -> None:
        """from_dict() handles missing author fields gracefully."""
        data = {
            "chunk_id": "chunk-001",
            "document_id": "doc-001",
            "content": "Test content",
        }
        chunk = ChunkRecord.from_dict(data)
        assert chunk.author_id is None
        assert chunk.author_name is None

    def test_roundtrip_serialization(self) -> None:
        """Roundtrip through to_dict/from_dict preserves author."""
        original = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="john.doe@example.com",
            author_name="John Doe",
        )
        restored = ChunkRecord.from_dict(original.to_dict())
        assert restored.author_id == original.author_id
        assert restored.author_name == original.author_name


class TestSearchResultAuthorFields:
    """Test SearchResult author fields from ChunkRecord."""

    def test_from_chunk_includes_author(self) -> None:
        """SearchResult.from_chunk() copies author fields."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            source_file="test.txt",
            author_id="user123",
            author_name="Jane Smith",
        )
        result = SearchResult.from_chunk(chunk, score=0.9)
        assert result.author_id == "user123"
        assert result.author_name == "Jane Smith"

    def test_from_chunk_without_author(self) -> None:
        """SearchResult.from_chunk() handles missing author fields."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            source_file="test.txt",
        )
        result = SearchResult.from_chunk(chunk, score=0.9)
        assert result.author_id is None
        assert result.author_name is None

    def test_to_dict_includes_author(self) -> None:
        """SearchResult.to_dict() includes author fields."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            source_file="test.txt",
            author_id="user123",
            author_name="Jane Smith",
        )
        result = SearchResult.from_chunk(chunk, score=0.9)
        data = result.to_dict()
        assert data["author_id"] == "user123"
        assert data["author_name"] == "Jane Smith"
