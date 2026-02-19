"""
Tests for Markdown Export Command.

This module tests exporting knowledge base to Markdown format.

Test Strategy
-------------
- Focus on markdown generation and formatting logic
- Mock storage and chunk retrieval
- Test grouping strategies (by source vs sequential)
- Test header and metadata formatting
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestMarkdownExportInit: Initialization
- TestValidation: Parameter validation
- TestHeaderGeneration: Header formatting
- TestChunkFormatting: Individual chunk formatting
- TestContentGeneration: Grouped vs sequential content
- TestNoChunks: Empty result handling
"""

from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime


from ingestforge.cli.export.markdown import MarkdownExportCommand


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "Test content",
    source_file: str = "test.txt",
    **metadata,
):
    """Create a mock chunk object."""
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.source_file = source_file

    # Add any additional metadata
    for key, value in metadata.items():
        setattr(chunk, key, value)

    return chunk


def make_mock_context(has_storage: bool = True):
    """Create a mock context dictionary."""
    ctx = {}

    if has_storage:
        ctx["storage"] = Mock()
        ctx["config"] = Mock()

    return ctx


# ============================================================================
# Test Classes
# ============================================================================


class TestMarkdownExportInit:
    """Tests for MarkdownExportCommand initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_markdown_export_command(self):
        """Test creating MarkdownExportCommand instance."""
        cmd = MarkdownExportCommand()

        assert cmd is not None

    def test_inherits_from_export_command(self):
        """Test MarkdownExportCommand inherits from ExportCommand."""
        from ingestforge.cli.export.base import ExportCommand

        cmd = MarkdownExportCommand()

        assert isinstance(cmd, ExportCommand)


class TestValidation:
    """Tests for parameter validation.

    Rule #4: Focused test class - tests validation logic
    """

    def test_validate_limit_positive(self):
        """Test limit must be positive."""
        cmd = MarkdownExportCommand()
        output = Path("output.md")

        # Execute returns 1 (error) for invalid limit
        result = cmd.execute(output, limit=0)
        assert result == 1

        result = cmd.execute(output, limit=-1)
        assert result == 1

    @patch.object(MarkdownExportCommand, "validate_output_path")
    @patch.object(MarkdownExportCommand, "initialize_context")
    @patch.object(MarkdownExportCommand, "search_filtered_chunks")
    @patch.object(MarkdownExportCommand, "save_export_file")
    def test_validate_limit_accepts_valid(
        self, mock_save, mock_search, mock_init, mock_validate
    ):
        """Test validation accepts valid limit values."""
        cmd = MarkdownExportCommand()
        output = Path("output.md")

        # Mock return values
        mock_init.return_value = make_mock_context()
        mock_search.return_value = [make_mock_chunk()]

        # Should not raise for valid values
        result = cmd.execute(output, limit=1)
        assert result == 0

        result = cmd.execute(output, limit=100)
        assert result == 0


class TestHeaderGeneration:
    """Tests for Markdown header generation.

    Rule #4: Focused test class - tests _generate_header()
    """

    def test_generate_header_basic(self):
        """Test generating basic header without query."""
        cmd = MarkdownExportCommand()

        header = cmd._generate_header(None, 10)

        assert "# Knowledge Base Export" in header
        assert "**Chunks:** 10" in header
        assert "**Generated:**" in header
        assert "---" in header

    def test_generate_header_with_query(self):
        """Test generating header with query filter."""
        cmd = MarkdownExportCommand()

        header = cmd._generate_header("Python", 5)

        assert "**Filter:** Python" in header
        assert "**Chunks:** 5" in header

    def test_generate_header_includes_timestamp(self):
        """Test header includes timestamp."""
        cmd = MarkdownExportCommand()

        header = cmd._generate_header(None, 1)

        # Should contain a timestamp (year at minimum)
        current_year = str(datetime.now().year)
        assert current_year in header


class TestChunkFormatting:
    """Tests for individual chunk formatting.

    Rule #4: Focused test class - tests _format_chunk()
    """

    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_format_chunk_basic(self, mock_metadata, mock_text):
        """Test formatting chunk without metadata."""
        cmd = MarkdownExportCommand()
        chunk = make_mock_chunk()
        mock_text.return_value = "Test content"
        mock_metadata.return_value = {}

        result = cmd._format_chunk(chunk, 1)

        assert "### Chunk 1" in result
        assert "Test content" in result

    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_format_chunk_with_metadata(self, mock_metadata, mock_text):
        """Test formatting chunk with metadata."""
        cmd = MarkdownExportCommand()
        chunk = make_mock_chunk()
        mock_text.return_value = "Test content"
        mock_metadata.return_value = {"source": "test.txt", "page": 1}

        result = cmd._format_chunk(chunk, 2)

        assert "### Chunk 2" in result
        assert "**Metadata:**" in result
        assert "source: test.txt" in result
        assert "page: 1" in result

    def test_format_metadata(self):
        """Test metadata formatting."""
        cmd = MarkdownExportCommand()
        metadata = {
            "source": "test.pdf",
            "page": 10,
            "chapter": "Introduction",
            "other_key": "ignored",
        }

        result = cmd._format_metadata(metadata)

        assert "**Metadata:**" in result
        assert "source: test.pdf" in result
        assert "page: 10" in result
        assert "chapter: Introduction" in result
        # Other keys not in whitelist should not appear
        assert "other_key" not in result

    def test_format_metadata_empty(self):
        """Test formatting empty metadata."""
        cmd = MarkdownExportCommand()

        result = cmd._format_metadata({})

        assert "**Metadata:**" in result


class TestContentGeneration:
    """Tests for content generation strategies.

    Rule #4: Focused test class - tests content generation
    """

    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_generate_sequential_content(self, mock_metadata, mock_text):
        """Test generating sequential (non-grouped) content."""
        cmd = MarkdownExportCommand()
        chunks = [
            make_mock_chunk("chunk_1", "Content 1"),
            make_mock_chunk("chunk_2", "Content 2"),
        ]
        mock_text.side_effect = ["Content 1", "Content 2"]
        mock_metadata.return_value = {}

        result = cmd._generate_sequential_content(chunks)

        assert "### Chunk 1" in result
        assert "### Chunk 2" in result
        assert "Content 1" in result
        assert "Content 2" in result

    @patch.object(MarkdownExportCommand, "group_chunks_by_source")
    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_generate_grouped_content(self, mock_metadata, mock_text, mock_group):
        """Test generating grouped content."""
        cmd = MarkdownExportCommand()
        chunks = [
            make_mock_chunk("chunk_1", "Content 1", "source1.txt"),
            make_mock_chunk("chunk_2", "Content 2", "source1.txt"),
        ]

        # Mock grouping
        mock_group.return_value = {"source1.txt": chunks}
        mock_text.side_effect = ["Content 1", "Content 2"]
        mock_metadata.return_value = {}

        result = cmd._generate_grouped_content(chunks)

        assert "## source1.txt" in result
        assert "*2 chunks*" in result
        assert "### Chunk 1" in result
        assert "### Chunk 2" in result

    def test_generate_markdown_with_grouping(self):
        """Test full markdown generation with grouping."""
        cmd = MarkdownExportCommand()
        chunks = [make_mock_chunk()]

        with patch.object(cmd, "_generate_header", return_value="# Header\n"):
            with patch.object(
                cmd, "_generate_grouped_content", return_value="Content\n"
            ):
                result = cmd._generate_markdown(
                    chunks, group_by_source=True, query=None
                )

        assert "# Header" in result
        assert "Content" in result

    def test_generate_markdown_without_grouping(self):
        """Test full markdown generation without grouping."""
        cmd = MarkdownExportCommand()
        chunks = [make_mock_chunk()]

        with patch.object(cmd, "_generate_header", return_value="# Header\n"):
            with patch.object(
                cmd, "_generate_sequential_content", return_value="Content\n"
            ):
                result = cmd._generate_markdown(
                    chunks, group_by_source=False, query=None
                )

        assert "# Header" in result
        assert "Content" in result


class TestNoChunks:
    """Tests for handling no chunks found.

    Rule #4: Focused test class - tests _handle_no_chunks()
    """

    def test_handle_no_chunks_with_query(self):
        """Test handling no chunks with query filter."""
        cmd = MarkdownExportCommand()

        # Should not raise
        cmd._handle_no_chunks("Python")

    def test_handle_no_chunks_without_query(self):
        """Test handling no chunks without query."""
        cmd = MarkdownExportCommand()

        # Should not raise
        cmd._handle_no_chunks(None)


class TestStreamingExport:
    """Tests for streaming export functionality.

    Rule #4: Focused test class - tests streaming methods
    Rule #3: Tests for memory efficiency
    """

    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_stream_sequential_content(self, mock_metadata, mock_text):
        """Test streaming sequential content to file."""
        cmd = MarkdownExportCommand()
        chunks = [
            make_mock_chunk("chunk_1", "Content 1", "source1.txt"),
            make_mock_chunk("chunk_2", "Content 2", "source2.txt"),
        ]

        # Mock return values - called twice per chunk (once for format, once for source tracking)
        def get_metadata(chunk):
            if hasattr(chunk, "source_file"):
                return {"source": chunk.source_file}
            return {}

        mock_text.side_effect = ["Content 1", "Content 2"]
        mock_metadata.side_effect = get_metadata

        # Create a mock file object
        mock_file = MagicMock()
        sources_seen = {}

        cmd._stream_sequential_content(mock_file, chunks, sources_seen)

        # Verify file was written to
        assert mock_file.write.called
        # Verify sources were tracked
        assert "source1.txt" in sources_seen
        assert "source2.txt" in sources_seen
        assert sources_seen["source1.txt"] == 1
        assert sources_seen["source2.txt"] == 1

    @patch.object(MarkdownExportCommand, "group_chunks_by_source")
    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_stream_grouped_content(self, mock_metadata, mock_text, mock_group):
        """Test streaming grouped content to file."""
        cmd = MarkdownExportCommand()
        chunks = [
            make_mock_chunk("chunk_1", "Content 1", "source1.txt"),
            make_mock_chunk("chunk_2", "Content 2", "source1.txt"),
        ]

        # Mock grouping
        mock_group.return_value = {"source1.txt": chunks}
        mock_text.side_effect = ["Content 1", "Content 2"]
        mock_metadata.return_value = {}

        # Create a mock file object
        mock_file = MagicMock()
        sources_seen = {}

        cmd._stream_grouped_content(mock_file, chunks, sources_seen)

        # Verify file was written to
        assert mock_file.write.called
        # Verify source was tracked
        assert "source1.txt" in sources_seen
        assert sources_seen["source1.txt"] == 2

    @patch("builtins.open", create=True)
    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_stream_markdown_to_file(self, mock_metadata, mock_text, mock_open):
        """Test streaming markdown to file."""
        cmd = MarkdownExportCommand()
        chunks = [make_mock_chunk("chunk_1", "Content 1", "source1.txt")]
        mock_text.return_value = "Content 1"
        mock_metadata.return_value = {"source": "source1.txt"}

        # Mock file handle
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        output = Path("test_output.md")

        cmd._stream_markdown_to_file(
            output, chunks, group_by_source=False, query=None, include_citations=False
        )

        # Verify file was opened
        mock_open.assert_called_once_with(output, "w", encoding="utf-8")
        # Verify content was written
        assert mock_file.write.called


class TestCitations:
    """Tests for citation generation functionality.

    Rule #4: Focused test class - tests citation methods
    """

    def test_generate_citations_section_single_source(self):
        """Test generating citations with single source."""
        cmd = MarkdownExportCommand()
        sources = {"test.pdf": 5}

        result = cmd._generate_citations_section(sources)

        assert "## Sources" in result
        assert "1. **test.pdf** - 5 chunks" in result

    def test_generate_citations_section_multiple_sources(self):
        """Test generating citations with multiple sources."""
        cmd = MarkdownExportCommand()
        sources = {
            "doc1.pdf": 10,
            "doc2.txt": 5,
            "doc3.md": 1,
        }

        result = cmd._generate_citations_section(sources)

        assert "## Sources" in result
        # Should be sorted alphabetically
        assert "1. **doc1.pdf** - 10 chunks" in result
        assert "2. **doc2.txt** - 5 chunks" in result
        assert "3. **doc3.md** - 1 chunk" in result  # Singular "chunk"

    def test_generate_citations_section_singular_plural(self):
        """Test citations use correct singular/plural for chunks."""
        cmd = MarkdownExportCommand()

        # Single chunk
        sources_single = {"test.pdf": 1}
        result_single = cmd._generate_citations_section(sources_single)
        assert "1 chunk" in result_single
        assert "chunks" not in result_single or "1 chunks" not in result_single

        # Multiple chunks
        sources_multiple = {"test.pdf": 5}
        result_multiple = cmd._generate_citations_section(sources_multiple)
        assert "5 chunks" in result_multiple

    @patch("builtins.open", create=True)
    @patch.object(MarkdownExportCommand, "extract_chunk_text")
    @patch.object(MarkdownExportCommand, "extract_chunk_metadata")
    def test_stream_markdown_with_citations(self, mock_metadata, mock_text, mock_open):
        """Test streaming markdown with citations enabled."""
        cmd = MarkdownExportCommand()
        chunks = [
            make_mock_chunk("chunk_1", "Content 1", "source1.txt"),
            make_mock_chunk("chunk_2", "Content 2", "source2.txt"),
        ]

        # Mock return values
        def get_metadata(chunk):
            if hasattr(chunk, "source_file"):
                return {"source": chunk.source_file}
            return {}

        mock_text.side_effect = ["Content 1", "Content 2"]
        mock_metadata.side_effect = get_metadata

        # Mock file handle
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file

        output = Path("test_output.md")

        cmd._stream_markdown_to_file(
            output, chunks, group_by_source=False, query=None, include_citations=True
        )

        # Verify file was opened
        mock_open.assert_called_once_with(output, "w", encoding="utf-8")
        # Verify content was written including citations
        assert mock_file.write.called
        # Check that citations were written (last write call should include "## Sources")
        write_calls = [str(call) for call in mock_file.write.call_args_list]
        assert any("## Sources" in str(call) for call in write_calls)


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Markdown export init: 2 tests (creation, inheritance)
    - Validation: 2 tests (limit validation positive/negative)
    - Header generation: 3 tests (basic, with query, timestamp)
    - Chunk formatting: 4 tests (basic, with metadata, metadata formatting, empty metadata)
    - Content generation: 4 tests (sequential, grouped, full with/without grouping)
    - No chunks: 2 tests (with/without query)
    - Streaming export: 3 tests (sequential streaming, grouped streaming, file streaming)
    - Citations: 4 tests (single source, multiple sources, singular/plural, with citations)

    Total: 24 tests

Design Decisions:
    1. Focus on markdown generation and formatting logic
    2. Mock storage and chunk retrieval
    3. Test grouping strategies (by source vs sequential)
    4. Test header and metadata formatting
    5. Don't test file I/O (tested in base ExportCommand)
    6. Simple tests that verify markdown export works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)
    9. Test streaming functionality for memory efficiency (Rule #3)
    10. Test citation generation for research notes

Behaviors Tested:
    - MarkdownExportCommand initialization
    - Inheritance from ExportCommand base class
    - Limit parameter validation (must be positive)
    - Header generation with/without query filter
    - Header includes timestamp
    - Chunk formatting with/without metadata
    - Metadata formatting (whitelisted keys only)
    - Sequential content generation
    - Grouped content generation (by source)
    - Full markdown generation (grouped vs sequential)
    - No chunks handling (with/without query)
    - Streaming export to file (sequential and grouped)
    - Source tracking for citations
    - Citation section generation
    - Singular/plural handling in citations
    - Export with citations enabled

Justification:
    - Markdown export is critical for document generation
    - Formatting logic needs verification
    - Grouping strategies enable different output styles
    - Metadata handling ensures proper information display
    - Streaming export prevents memory issues with large corpora (EXPORT-001.1)
    - Citation tracking enables research note generation (EXPORT-001.2)
    - Simple tests verify export system works correctly
"""
