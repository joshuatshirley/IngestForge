"""Tests for LLM formatter.

Copy-Paste Ready CLI Interfaces
Tests for GWT-2 (LLM Template Formats) and GWT-4 (Provenance Embedding).
"""

from __future__ import annotations

import pytest

from ingestforge.core.llm_formatter import (
    MAX_CHUNKS,
    MAX_CONTEXT_LENGTH,
    ChatGPTFormatter,
    ClaudeFormatter,
    ContextChunk,
    FormattedOutput,
    FormatterContext,
    LLMFormat,
    MarkdownFormatter,
    PlainTextFormatter,
    format_context,
    get_formatter,
)


# =============================================================================
# Test ContextChunk Dataclass
# =============================================================================


class TestContextChunk:
    """Tests for ContextChunk dataclass."""

    def test_create_basic_chunk(self) -> None:
        """Test creating basic chunk."""
        chunk = ContextChunk(text="Test content", source="test.pdf")
        assert chunk.text == "Test content"
        assert chunk.source == "test.pdf"
        assert chunk.rank == 1
        assert chunk.score == 1.0

    def test_create_full_chunk(self) -> None:
        """Test creating chunk with all fields."""
        chunk = ContextChunk(
            text="Content",
            source="document.pdf",
            rank=3,
            score=0.85,
            page=42,
            section="Introduction",
        )
        assert chunk.rank == 3
        assert chunk.score == 0.85
        assert chunk.page == 42
        assert chunk.section == "Introduction"

    def test_none_text_raises(self) -> None:
        """Test that None text raises assertion."""
        with pytest.raises(AssertionError, match="cannot be None"):
            ContextChunk(text=None, source="test.pdf")  # type: ignore

    def test_invalid_rank_raises(self) -> None:
        """Test that rank < 1 raises assertion."""
        with pytest.raises(AssertionError, match="rank must be >= 1"):
            ContextChunk(text="test", source="test.pdf", rank=0)

    def test_invalid_score_raises(self) -> None:
        """Test that score outside 0-1 raises assertion."""
        with pytest.raises(AssertionError, match="score must be between"):
            ContextChunk(text="test", source="test.pdf", score=1.5)


# =============================================================================
# Test FormatterContext Dataclass
# =============================================================================


class TestFormatterContext:
    """Tests for FormatterContext dataclass."""

    def test_create_basic_context(self) -> None:
        """Test creating basic context."""
        context = FormatterContext(query="test query")
        assert context.query == "test query"
        assert context.chunks == []
        assert context.include_citations is True
        assert context.max_chunks == MAX_CHUNKS

    def test_create_with_chunks(self) -> None:
        """Test creating context with chunks."""
        chunks = [
            ContextChunk(text="chunk 1", source="a.pdf"),
            ContextChunk(text="chunk 2", source="b.pdf", rank=2),
        ]
        context = FormatterContext(query="test", chunks=chunks)
        assert len(context.chunks) == 2

    def test_max_chunks_bounded(self) -> None:
        """Test that max_chunks is bounded by MAX_CHUNKS."""
        context = FormatterContext(query="test", max_chunks=1000)
        assert context.max_chunks <= MAX_CHUNKS

    def test_none_query_raises(self) -> None:
        """Test that None query raises assertion."""
        with pytest.raises(AssertionError, match="cannot be None"):
            FormatterContext(query=None)  # type: ignore

    def test_invalid_max_chunks_raises(self) -> None:
        """Test that max_chunks <= 0 raises assertion."""
        with pytest.raises(AssertionError, match="must be positive"):
            FormatterContext(query="test", max_chunks=0)


# =============================================================================
# Test FormattedOutput Dataclass
# =============================================================================


class TestFormattedOutput:
    """Tests for FormattedOutput dataclass."""

    def test_create_output(self) -> None:
        """Test creating formatted output."""
        output = FormattedOutput(
            content="test content",
            format=LLMFormat.MARKDOWN,
            chunk_count=5,
        )
        assert output.content == "test content"
        assert output.format == LLMFormat.MARKDOWN
        assert output.chunk_count == 5
        assert output.char_count == len("test content")

    def test_char_count_auto_calculated(self) -> None:
        """Test char_count is auto-calculated."""
        output = FormattedOutput(
            content="12345",
            format=LLMFormat.PLAIN,
        )
        assert output.char_count == 5

    def test_truncated_flag(self) -> None:
        """Test truncated flag."""
        output = FormattedOutput(
            content="truncated...",
            format=LLMFormat.PLAIN,
            truncated=True,
        )
        assert output.truncated is True


# =============================================================================
# Test ChatGPTFormatter
# =============================================================================


class TestChatGPTFormatter:
    """Tests for ChatGPT formatter."""

    def test_format_type(self) -> None:
        """Test format type is CHATGPT."""
        formatter = ChatGPTFormatter()
        assert formatter.format_type == LLMFormat.CHATGPT

    def test_format_basic_context(self) -> None:
        """Test formatting basic context."""
        chunks = [ContextChunk(text="Test content", source="doc.pdf")]
        context = FormatterContext(query="test query", chunks=chunks)

        formatter = ChatGPTFormatter()
        output = formatter.format(context)

        assert "## Context for Your Query" in output.content
        assert "test query" in output.content
        assert "Test content" in output.content
        assert output.chunk_count == 1

    def test_format_with_citations(self) -> None:
        """Test formatting includes citations."""
        chunks = [ContextChunk(text="Content", source="paper.pdf", page=10)]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = ChatGPTFormatter()
        output = formatter.format(context)

        assert "paper.pdf" in output.content
        assert "p. 10" in output.content
        assert "### Sources" in output.content

    def test_format_without_citations(self) -> None:
        """Test formatting without citations."""
        chunks = [ContextChunk(text="Content", source="doc.pdf")]
        context = FormatterContext(query="test", chunks=chunks, include_citations=False)

        formatter = ChatGPTFormatter()
        output = formatter.format(context)

        assert "### Sources" not in output.content

    def test_format_multiple_chunks(self) -> None:
        """Test formatting multiple chunks."""
        chunks = [
            ContextChunk(text="First", source="a.pdf", rank=1),
            ContextChunk(text="Second", source="b.pdf", rank=2),
            ContextChunk(text="Third", source="c.pdf", rank=3),
        ]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = ChatGPTFormatter()
        output = formatter.format(context)

        assert output.chunk_count == 3
        assert "First" in output.content
        assert "Second" in output.content
        assert "Third" in output.content

    def test_format_with_scores(self) -> None:
        """Test formatting with relevance scores."""
        chunks = [ContextChunk(text="Content", source="doc.pdf", score=0.95)]
        context = FormatterContext(query="test", chunks=chunks, include_scores=True)

        formatter = ChatGPTFormatter()
        output = formatter.format(context)

        assert "0.95" in output.content

    def test_format_none_context_raises(self) -> None:
        """Test that None context raises assertion."""
        formatter = ChatGPTFormatter()
        with pytest.raises(AssertionError, match="cannot be None"):
            formatter.format(None)  # type: ignore


# =============================================================================
# Test ClaudeFormatter
# =============================================================================


class TestClaudeFormatter:
    """Tests for Claude formatter."""

    def test_format_type(self) -> None:
        """Test format type is CLAUDE."""
        formatter = ClaudeFormatter()
        assert formatter.format_type == LLMFormat.CLAUDE

    def test_format_uses_xml_tags(self) -> None:
        """Test formatting uses XML-style tags."""
        chunks = [ContextChunk(text="Test content", source="doc.pdf")]
        context = FormatterContext(query="test query", chunks=chunks)

        formatter = ClaudeFormatter()
        output = formatter.format(context)

        assert "<context>" in output.content
        assert "</context>" in output.content
        assert "<query>" in output.content
        assert "</query>" in output.content
        assert "<source" in output.content

    def test_format_includes_references(self) -> None:
        """Test formatting includes references section."""
        chunks = [ContextChunk(text="Content", source="paper.pdf")]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = ClaudeFormatter()
        output = formatter.format(context)

        assert "<references>" in output.content
        assert "</references>" in output.content

    def test_format_without_citations(self) -> None:
        """Test formatting without citations."""
        chunks = [ContextChunk(text="Content", source="doc.pdf")]
        context = FormatterContext(query="test", chunks=chunks, include_citations=False)

        formatter = ClaudeFormatter()
        output = formatter.format(context)

        assert "<references>" not in output.content


# =============================================================================
# Test PlainTextFormatter
# =============================================================================


class TestPlainTextFormatter:
    """Tests for plain text formatter."""

    def test_format_type(self) -> None:
        """Test format type is PLAIN."""
        formatter = PlainTextFormatter()
        assert formatter.format_type == LLMFormat.PLAIN

    def test_format_basic_structure(self) -> None:
        """Test basic plain text structure."""
        chunks = [ContextChunk(text="Test content", source="doc.pdf")]
        context = FormatterContext(query="test query", chunks=chunks)

        formatter = PlainTextFormatter()
        output = formatter.format(context)

        assert "Query: test query" in output.content
        assert "Test content" in output.content
        assert "=" * 40 in output.content

    def test_format_separators(self) -> None:
        """Test chunk separators."""
        chunks = [
            ContextChunk(text="First", source="a.pdf"),
            ContextChunk(text="Second", source="b.pdf", rank=2),
        ]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = PlainTextFormatter()
        output = formatter.format(context)

        assert "-" * 40 in output.content


# =============================================================================
# Test MarkdownFormatter
# =============================================================================


class TestMarkdownFormatter:
    """Tests for markdown formatter."""

    def test_format_type(self) -> None:
        """Test format type is MARKDOWN."""
        formatter = MarkdownFormatter()
        assert formatter.format_type == LLMFormat.MARKDOWN

    def test_format_uses_headers(self) -> None:
        """Test formatting uses markdown headers."""
        chunks = [ContextChunk(text="Content", source="doc.pdf")]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = MarkdownFormatter()
        output = formatter.format(context)

        assert "# Retrieved Context" in output.content
        assert "##" in output.content

    def test_format_uses_blockquote(self) -> None:
        """Test formatting uses blockquote for query."""
        chunks = [ContextChunk(text="Content", source="doc.pdf")]
        context = FormatterContext(query="my query", chunks=chunks)

        formatter = MarkdownFormatter()
        output = formatter.format(context)

        assert "> **Query:**" in output.content

    def test_format_references_list(self) -> None:
        """Test references as bulleted list."""
        chunks = [
            ContextChunk(text="A", source="a.pdf"),
            ContextChunk(text="B", source="b.pdf", rank=2),
        ]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = MarkdownFormatter()
        output = formatter.format(context)

        assert "## References" in output.content
        assert "- [1]" in output.content
        assert "- [2]" in output.content


# =============================================================================
# Test get_formatter Function
# =============================================================================


class TestGetFormatter:
    """Tests for get_formatter function."""

    def test_get_chatgpt_formatter(self) -> None:
        """Test getting ChatGPT formatter."""
        formatter = get_formatter(LLMFormat.CHATGPT)
        assert isinstance(formatter, ChatGPTFormatter)

    def test_get_claude_formatter(self) -> None:
        """Test getting Claude formatter."""
        formatter = get_formatter(LLMFormat.CLAUDE)
        assert isinstance(formatter, ClaudeFormatter)

    def test_get_plain_formatter(self) -> None:
        """Test getting plain text formatter."""
        formatter = get_formatter(LLMFormat.PLAIN)
        assert isinstance(formatter, PlainTextFormatter)

    def test_get_markdown_formatter(self) -> None:
        """Test getting markdown formatter."""
        formatter = get_formatter(LLMFormat.MARKDOWN)
        assert isinstance(formatter, MarkdownFormatter)


# =============================================================================
# Test format_context Function
# =============================================================================


class TestFormatContext:
    """Tests for format_context convenience function."""

    def test_format_context_basic(self) -> None:
        """Test basic format_context usage."""
        chunks = [{"text": "test content", "source": "doc.pdf"}]
        output = format_context("my query", chunks)

        assert "my query" in output.content
        assert "test content" in output.content

    def test_format_context_different_formats(self) -> None:
        """Test format_context with different formats."""
        chunks = [{"text": "content", "source": "doc.pdf"}]

        md_output = format_context("q", chunks, LLMFormat.MARKDOWN)
        assert md_output.format == LLMFormat.MARKDOWN

        claude_output = format_context("q", chunks, LLMFormat.CLAUDE)
        assert claude_output.format == LLMFormat.CLAUDE

    def test_format_context_without_citations(self) -> None:
        """Test format_context without citations."""
        chunks = [{"text": "content", "source": "doc.pdf"}]
        output = format_context("q", chunks, include_citations=False)

        assert "References" not in output.content


# =============================================================================
# Test Truncation and Bounds
# =============================================================================


class TestTruncationAndBounds:
    """Tests for content truncation and bounds."""

    def test_max_chunks_enforced(self) -> None:
        """Test MAX_CHUNKS limit is enforced."""
        chunks = [
            ContextChunk(text=f"Chunk {i}", source=f"doc{i}.pdf", rank=i)
            for i in range(1, MAX_CHUNKS + 10)
        ]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = MarkdownFormatter()
        output = formatter.format(context)

        assert output.chunk_count <= MAX_CHUNKS

    def test_content_truncation(self) -> None:
        """Test content is truncated when too long."""
        # Create very long content
        long_text = "x" * (MAX_CONTEXT_LENGTH + 1000)
        chunks = [ContextChunk(text=long_text, source="doc.pdf")]
        context = FormatterContext(query="test", chunks=chunks)

        formatter = PlainTextFormatter()
        output = formatter.format(context)

        assert output.truncated is True
        assert len(output.content) <= MAX_CONTEXT_LENGTH + 10  # +10 for "..."

    def test_citation_truncation(self) -> None:
        """Test long citations are truncated."""
        long_source = "x" * 300
        chunk = ContextChunk(text="content", source=long_source)
        context = FormatterContext(query="test", chunks=[chunk])

        formatter = ChatGPTFormatter()
        citation = formatter._format_citation(chunk)

        assert len(citation) <= 200  # MAX_CITATION_LENGTH


# =============================================================================
# Test JPL Compliance
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_constants_defined(self) -> None:
        """Test all required constants are defined."""
        assert MAX_CHUNKS == 50
        assert MAX_CONTEXT_LENGTH == 50_000

    def test_dataclass_validation(self) -> None:
        """Test dataclasses validate in __post_init__."""
        # All dataclasses should validate their fields
        with pytest.raises(AssertionError):
            ContextChunk(text="test", source="test", score=2.0)

        with pytest.raises(AssertionError):
            FormatterContext(query="test", max_chunks=0)

    def test_format_enum_values(self) -> None:
        """Test all LLMFormat values exist."""
        formats = list(LLMFormat)
        assert LLMFormat.CHATGPT in formats
        assert LLMFormat.CLAUDE in formats
        assert LLMFormat.PLAIN in formats
        assert LLMFormat.MARKDOWN in formats
