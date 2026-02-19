"""Tests for author helper functions (TICKET-301).

Verifies:
1. set_author() sets author_id and author_name on chunks
2. set_author() validates parameters
3. get_author_info() extracts ContributorIdentity from chunks
4. get_author_info() validates parameters
5. Helper functions work with both populated and empty author fields
"""

import pytest

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.provenance import (
    set_author,
    get_author_info,
    ContributorIdentity,
)


class TestSetAuthor:
    """Test set_author() helper function."""

    def test_set_both_fields(self) -> None:
        """Set both author_id and author_name."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        result = set_author(chunk, "user123", "Jane Smith")

        assert chunk.author_id == "user123"
        assert chunk.author_name == "Jane Smith"
        assert result is chunk  # Returns same chunk for chaining

    def test_set_only_id(self) -> None:
        """Set only author_id."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        set_author(chunk, "user123", None)

        assert chunk.author_id == "user123"
        assert chunk.author_name is None

    def test_set_only_name(self) -> None:
        """Set only author_name."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        set_author(chunk, None, "Anonymous Contributor")

        assert chunk.author_id is None
        assert chunk.author_name == "Anonymous Contributor"

    def test_set_both_none(self) -> None:
        """Set both fields to None (clear author)."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="old_user",
            author_name="Old Name",
        )
        set_author(chunk, None, None)

        assert chunk.author_id is None
        assert chunk.author_name is None

    def test_method_chaining(self) -> None:
        """set_author() returns chunk for method chaining."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        result = set_author(chunk, "user123", "Jane Smith")

        assert result is chunk
        assert isinstance(result, ChunkRecord)

    def test_validates_chunk_not_none(self) -> None:
        """Raises ValueError if chunk is None."""
        with pytest.raises(ValueError, match="chunk cannot be None"):
            set_author(None, "user123", "Jane Smith")  # type: ignore


class TestGetAuthorInfo:
    """Test get_author_info() helper function."""

    def test_get_author_with_both_fields(self) -> None:
        """Extract author info when both fields are set."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="user123",
            author_name="Jane Smith",
        )
        author_info = get_author_info(chunk)

        assert isinstance(author_info, ContributorIdentity)
        assert author_info.author_id == "user123"
        assert author_info.author_name == "Jane Smith"

    def test_get_author_with_only_id(self) -> None:
        """Extract author info when only author_id is set."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="user123",
        )
        author_info = get_author_info(chunk)

        assert author_info.author_id == "user123"
        assert author_info.author_name is None

    def test_get_author_with_only_name(self) -> None:
        """Extract author info when only author_name is set."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_name="Jane Smith",
        )
        author_info = get_author_info(chunk)

        assert author_info.author_id is None
        assert author_info.author_name == "Jane Smith"

    def test_get_author_with_no_fields(self) -> None:
        """Extract author info when neither field is set."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )
        author_info = get_author_info(chunk)

        assert author_info.author_id is None
        assert author_info.author_name is None
        assert not author_info.is_populated()

    def test_validates_chunk_not_none(self) -> None:
        """Raises ValueError if chunk is None."""
        with pytest.raises(ValueError, match="chunk cannot be None"):
            get_author_info(None)  # type: ignore


class TestHelperFunctionsIntegration:
    """Test integration between set_author() and get_author_info()."""

    def test_roundtrip(self) -> None:
        """Set author, then get it back."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )

        # Set author
        set_author(chunk, "john.doe@example.com", "John Doe")

        # Get it back
        author_info = get_author_info(chunk)

        assert author_info.author_id == "john.doe@example.com"
        assert author_info.author_name == "John Doe"
        assert author_info.format_attribution() == "Contributed by: John Doe"

    def test_chained_operations(self) -> None:
        """Use set_author() in chained operations."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )

        # Chain set_author with other operations
        result_chunk = set_author(chunk, "user123", "Jane Smith")

        # Verify chaining worked
        assert result_chunk is chunk
        author_info = get_author_info(result_chunk)
        assert author_info.is_populated()
        assert author_info.author_name == "Jane Smith"

    def test_format_attribution_after_set(self) -> None:
        """Format attribution after setting author."""
        chunk = ChunkRecord(
            chunk_id="chunk-001", document_id="doc-001", content="Test content"
        )

        set_author(chunk, "user123", "Jane Smith")
        author_info = get_author_info(chunk)

        assert author_info.format_attribution() == "Contributed by: Jane Smith"

    def test_clear_author_then_get(self) -> None:
        """Clear author info then retrieve it."""
        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            author_id="old_user",
            author_name="Old Name",
        )

        # Clear author
        set_author(chunk, None, None)

        # Verify it's cleared
        author_info = get_author_info(chunk)
        assert not author_info.is_populated()
        assert author_info.format_attribution() == ""
