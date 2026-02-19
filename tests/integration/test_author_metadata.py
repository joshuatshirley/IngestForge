"""Integration test for author metadata (TICKET-301).

Tests the full workflow:
1. Create chunks with author metadata
2. Store chunks
3. Retrieve chunks with author info preserved
4. Display author attribution in query results
"""


from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.provenance import set_author, get_author_info
from ingestforge.storage.base import SearchResult


class TestAuthorMetadataWorkflow:
    """Test complete author metadata workflow."""

    def test_chunk_creation_with_author(self) -> None:
        """Create chunk with author metadata."""
        chunk = ChunkRecord(
            chunk_id="test-chunk-001",
            document_id="test-doc-001",
            content="This is test content contributed by a user.",
            source_file="test.txt",
        )

        # Set author using helper
        set_author(chunk, "test.user@example.com", "Test User")

        assert chunk.author_id == "test.user@example.com"
        assert chunk.author_name == "Test User"

    def test_author_preserved_in_search_result(self) -> None:
        """Author metadata is preserved when converting to SearchResult."""
        chunk = ChunkRecord(
            chunk_id="test-chunk-001",
            document_id="test-doc-001",
            content="This is test content.",
            source_file="test.txt",
            author_id="jane.doe@example.com",
            author_name="Jane Doe",
        )

        # Convert to SearchResult
        result = SearchResult.from_chunk(chunk, score=0.95)

        assert result.author_id == "jane.doe@example.com"
        assert result.author_name == "Jane Doe"

    def test_author_preserved_in_dict_serialization(self) -> None:
        """Author metadata is preserved in dictionary serialization."""
        chunk = ChunkRecord(
            chunk_id="test-chunk-001",
            document_id="test-doc-001",
            content="This is test content.",
            source_file="test.txt",
            author_id="john.smith@example.com",
            author_name="John Smith",
        )

        # Convert to dict (for JSONL export)
        chunk_dict = chunk.to_dict()

        assert chunk_dict["author_id"] == "john.smith@example.com"
        assert chunk_dict["author_name"] == "John Smith"

    def test_author_roundtrip_through_dict(self) -> None:
        """Author metadata survives roundtrip through dict serialization."""
        original = ChunkRecord(
            chunk_id="test-chunk-001",
            document_id="test-doc-001",
            content="This is test content.",
            source_file="test.txt",
            author_id="alice@example.com",
            author_name="Alice Wonder",
        )

        # Roundtrip through dict
        chunk_dict = original.to_dict()
        restored = ChunkRecord.from_dict(chunk_dict)

        assert restored.author_id == original.author_id
        assert restored.author_name == original.author_name

    def test_get_author_info_returns_contributor_identity(self) -> None:
        """get_author_info() returns properly formatted ContributorIdentity."""
        chunk = ChunkRecord(
            chunk_id="test-chunk-001",
            document_id="test-doc-001",
            content="This is test content.",
            source_file="test.txt",
            author_id="bob@example.com",
            author_name="Bob Builder",
        )

        author_info = get_author_info(chunk)

        assert author_info.is_populated()
        assert author_info.format_attribution() == "Contributed by: Bob Builder"

    def test_multiple_chunks_different_authors(self) -> None:
        """Multiple chunks can have different authors."""
        chunks = [
            ChunkRecord(
                chunk_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Content {i}",
                source_file="test.txt",
                author_id=f"user{i}@example.com",
                author_name=f"User {i}",
            )
            for i in range(3)
        ]

        # Verify each has unique author
        for i, chunk in enumerate(chunks):
            assert chunk.author_id == f"user{i}@example.com"
            assert chunk.author_name == f"User {i}"

            author_info = get_author_info(chunk)
            assert author_info.format_attribution() == f"Contributed by: User {i}"

    def test_chunks_without_author_work_normally(self) -> None:
        """Chunks without author metadata work normally (backward compatibility)."""
        chunk = ChunkRecord(
            chunk_id="test-chunk-001",
            document_id="test-doc-001",
            content="This is test content.",
            source_file="test.txt",
        )

        # Should have None for author fields
        assert chunk.author_id is None
        assert chunk.author_name is None

        # get_author_info should work but return empty
        author_info = get_author_info(chunk)
        assert not author_info.is_populated()
        assert author_info.format_attribution() == ""

        # Should convert to SearchResult without errors
        result = SearchResult.from_chunk(chunk, score=0.8)
        assert result.author_id is None
        assert result.author_name is None
