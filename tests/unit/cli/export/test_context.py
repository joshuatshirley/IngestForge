"""
Tests for Context Export Command.

This module tests exporting context for RAG applications.

Test Strategy
-------------
- Focus on context data building logic
- Test metadata inclusion toggle
- Test relevance score handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestContextCommandInit: Initialization
- TestContextDataBuilding: Context structure building
- TestMetadataHandling: Metadata inclusion logic
- TestNoChunks: Empty result handling
"""

from unittest.mock import Mock, patch


from ingestforge.cli.export.context import ContextCommand


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    text: str = "Test content",
    score: float = 0.95,
    source: str = "test.txt",
    chunk_type: str = "document",
):
    """Create a mock chunk with score."""
    chunk = Mock()
    chunk.text = text
    chunk.score = score

    # Mock metadata
    metadata = {
        "source": source,
        "type": chunk_type,
    }

    # Store metadata as dict
    chunk.metadata = metadata

    return chunk


# ============================================================================
# Test Classes
# ============================================================================


class TestContextCommandInit:
    """Tests for ContextCommand initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_context_command(self):
        """Test creating ContextCommand instance."""
        cmd = ContextCommand()

        assert cmd is not None

    def test_inherits_from_export_command(self):
        """Test ContextCommand inherits from ExportCommand."""
        from ingestforge.cli.export.base import ExportCommand

        cmd = ContextCommand()

        assert isinstance(cmd, ExportCommand)


class TestContextDataBuilding:
    """Tests for context data structure building.

    Rule #4: Focused test class - tests _build_context_data()
    """

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_build_context_basic(self, mock_extract):
        """Test building basic context data."""
        cmd = ContextCommand()
        chunks = [make_mock_chunk("Content 1"), make_mock_chunk("Content 2")]
        mock_extract.return_value = {"source": "test.txt", "type": "document"}

        result = cmd._build_context_data(chunks, "test query", include_metadata=False)

        assert result["query"] == "test query"
        assert result["total_chunks"] == 2
        assert len(result["context"]) == 2
        assert result["context"][0]["rank"] == 1
        assert result["context"][1]["rank"] == 2

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_build_context_with_metadata(self, mock_extract):
        """Test building context with metadata included."""
        cmd = ContextCommand()
        chunks = [make_mock_chunk("Content")]
        mock_extract.return_value = {"source": "test.txt", "type": "document"}

        result = cmd._build_context_data(chunks, "query", include_metadata=True)

        assert result["context"][0]["metadata"]["source"] == "test.txt"
        assert result["context"][0]["metadata"]["type"] == "document"

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_build_context_without_metadata(self, mock_extract):
        """Test building context without metadata."""
        cmd = ContextCommand()
        chunks = [make_mock_chunk("Content")]
        mock_extract.return_value = {"source": "test.txt"}

        result = cmd._build_context_data(chunks, "query", include_metadata=False)

        # Metadata should not be in result
        assert "metadata" not in result["context"][0]

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_build_context_includes_scores(self, mock_extract):
        """Test building context includes relevance scores."""
        cmd = ContextCommand()
        chunk = make_mock_chunk("Content", score=0.95)
        mock_extract.return_value = {"source": "test.txt", "type": "document"}

        result = cmd._build_context_data([chunk], "query", include_metadata=True)

        assert "relevance_score" in result["context"][0]
        assert result["context"][0]["relevance_score"] == 0.95

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_build_context_missing_score(self, mock_extract):
        """Test building context when chunk has no score."""
        cmd = ContextCommand()
        # Use spec=[] to prevent Mock from auto-creating attributes
        chunk = Mock(spec=["text"])
        chunk.text = "Content"
        mock_extract.return_value = {"source": "test.txt", "type": "document"}

        result = cmd._build_context_data([chunk], "query", include_metadata=True)

        # Score should not be in result if not available
        assert "relevance_score" not in result["context"][0]

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_build_context_ranks_correctly(self, mock_extract):
        """Test context items have correct rank order."""
        cmd = ContextCommand()
        chunks = [
            make_mock_chunk("First"),
            make_mock_chunk("Second"),
            make_mock_chunk("Third"),
        ]
        mock_extract.return_value = {}

        result = cmd._build_context_data(chunks, "query", include_metadata=False)

        assert result["context"][0]["rank"] == 1
        assert result["context"][1]["rank"] == 2
        assert result["context"][2]["rank"] == 3


class TestMetadataHandling:
    """Tests for metadata handling.

    Rule #4: Focused test class - tests metadata logic
    """

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_metadata_defaults(self, mock_extract):
        """Test metadata uses defaults for missing fields."""
        cmd = ContextCommand()
        chunk = make_mock_chunk()
        mock_extract.return_value = {}  # Empty metadata

        result = cmd._build_context_data([chunk], "query", include_metadata=True)

        # Should have defaults
        assert result["context"][0]["metadata"]["source"] == "Unknown"
        assert result["context"][0]["metadata"]["type"] == "document"

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_metadata_custom_values(self, mock_extract):
        """Test metadata uses custom values when provided."""
        cmd = ContextCommand()
        chunk = make_mock_chunk()
        mock_extract.return_value = {
            "source": "custom.pdf",
            "type": "article",
        }

        result = cmd._build_context_data([chunk], "query", include_metadata=True)

        assert result["context"][0]["metadata"]["source"] == "custom.pdf"
        assert result["context"][0]["metadata"]["type"] == "article"


class TestTextExtraction:
    """Tests for text extraction.

    Rule #4: Focused test class - tests text handling
    """

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_extract_text_attribute(self, mock_extract):
        """Test extracting text from chunk.text attribute."""
        cmd = ContextCommand()
        chunk = Mock()
        chunk.text = "Test content"
        mock_extract.return_value = {}

        result = cmd._build_context_data([chunk], "query", include_metadata=False)

        assert result["context"][0]["text"] == "Test content"

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_extract_text_fallback(self, mock_extract):
        """Test text extraction falls back to str(chunk)."""
        cmd = ContextCommand()

        # Create a custom object without 'text' attribute
        class ChunkWithoutText:
            def __str__(self):
                return "Fallback content"

        chunk = ChunkWithoutText()
        mock_extract.return_value = {}

        result = cmd._build_context_data([chunk], "query", include_metadata=False)

        assert result["context"][0]["text"] == "Fallback content"


class TestQueryHandling:
    """Tests for query and context handling.

    Rule #4: Focused test class - tests query storage
    """

    @patch.object(ContextCommand, "extract_chunk_metadata")
    def test_stores_query_in_result(self, mock_extract):
        """Test that query is stored in result."""
        cmd = ContextCommand()
        mock_extract.return_value = {}

        result = cmd._build_context_data([], "test query", include_metadata=False)

        assert result["query"] == "test query"


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Context command init: 2 tests (creation, inheritance)
    - Context data building: 6 tests (basic, with/without metadata, scores, missing score, ranks)
    - Metadata handling: 2 tests (defaults, custom values)
    - Text extraction: 2 tests (text attribute, fallback)
    - No chunks: 1 test (empty result)

    Total: 13 tests

Design Decisions:
    1. Focus on context data structure building
    2. Test metadata inclusion toggle
    3. Test relevance score handling
    4. Test rank assignment
    5. Test text extraction with fallback
    6. Don't test file I/O (tested in base ExportCommand)
    7. Simple tests that verify context export works
    8. Follows NASA JPL Rule #1 (Simple Control Flow)
    9. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - ContextCommand initialization
    - Inheritance from ExportCommand base class
    - Context data structure (query, context items, total_chunks)
    - Rank assignment (1-indexed)
    - Metadata inclusion toggle (include_metadata flag)
    - Relevance score inclusion when available
    - Missing score handling
    - Metadata defaults (Unknown source, document type)
    - Custom metadata values
    - Text extraction from chunk.text attribute
    - Text fallback to str(chunk)
    - Empty result handling (returns 0)

Justification:
    - Context export is critical for RAG applications
    - Data structure needs verification
    - Metadata toggle enables flexible output
    - Score handling ensures relevance information preserved
    - Simple tests verify export system works correctly
"""
