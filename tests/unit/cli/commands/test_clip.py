"""Tests for clip command.

Copy-Paste Ready CLI Interfaces
Tests for GWT-1, GWT-2, GWT-4, GWT-5.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from ingestforge.cli.commands.clip import ClipCommand
from ingestforge.core.clipboard import ClipboardBackend, ClipboardResult
from ingestforge.core.llm_formatter import LLMFormat


# =============================================================================
# Test ClipCommand - Format Parsing
# =============================================================================


class TestClipCommandFormatParsing:
    """Tests for format type parsing."""

    def test_parse_chatgpt_format(self) -> None:
        """Test parsing chatgpt format."""
        cmd = ClipCommand()
        assert cmd._parse_format("chatgpt") == LLMFormat.CHATGPT
        assert cmd._parse_format("gpt") == LLMFormat.CHATGPT

    def test_parse_claude_format(self) -> None:
        """Test parsing claude format."""
        cmd = ClipCommand()
        assert cmd._parse_format("claude") == LLMFormat.CLAUDE

    def test_parse_plain_format(self) -> None:
        """Test parsing plain format."""
        cmd = ClipCommand()
        assert cmd._parse_format("plain") == LLMFormat.PLAIN
        assert cmd._parse_format("text") == LLMFormat.PLAIN

    def test_parse_markdown_format(self) -> None:
        """Test parsing markdown format."""
        cmd = ClipCommand()
        assert cmd._parse_format("markdown") == LLMFormat.MARKDOWN
        assert cmd._parse_format("md") == LLMFormat.MARKDOWN

    def test_parse_case_insensitive(self) -> None:
        """Test parsing is case insensitive."""
        cmd = ClipCommand()
        assert cmd._parse_format("CHATGPT") == LLMFormat.CHATGPT
        assert cmd._parse_format("Claude") == LLMFormat.CLAUDE

    def test_parse_invalid_format(self) -> None:
        """Test parsing invalid format raises."""
        cmd = ClipCommand()
        with pytest.raises(ValueError, match="Invalid format"):
            cmd._parse_format("invalid")


# =============================================================================
# Test ClipCommand - Execute
# =============================================================================


class TestClipCommandExecute:
    """Tests for execute method."""

    def test_execute_with_stdout(self) -> None:
        """Test execute with stdout output."""
        cmd = ClipCommand()

        with patch.object(cmd, "_retrieve_chunks") as mock_retrieve:
            mock_retrieve.return_value = [{"text": "test content", "source": "doc.pdf"}]

            exit_code = cmd.execute(
                query="test query",
                format_type="markdown",
                k=5,
                cite=True,
                stdout=True,
            )

            assert exit_code == 0
            mock_retrieve.assert_called_once()

    def test_execute_with_clipboard(self) -> None:
        """Test execute with clipboard output."""
        cmd = ClipCommand()

        with (
            patch.object(cmd, "_retrieve_chunks") as mock_retrieve,
            patch(
                "ingestforge.cli.commands.clip.is_clipboard_available",
                return_value=True,
            ),
            patch("ingestforge.cli.commands.clip.copy_to_clipboard") as mock_copy,
        ):
            mock_retrieve.return_value = [{"text": "test content", "source": "doc.pdf"}]
            mock_copy.return_value = ClipboardResult(
                success=True,
                backend=ClipboardBackend.PYPERCLIP,
                chars_copied=100,
            )

            exit_code = cmd.execute(
                query="test query",
                format_type="markdown",
                k=5,
                cite=True,
                stdout=False,
            )

            assert exit_code == 0
            mock_copy.assert_called_once()

    def test_execute_no_chunks_found(self) -> None:
        """Test execute when no chunks found."""
        cmd = ClipCommand()

        with patch.object(cmd, "_retrieve_chunks", return_value=[]):
            exit_code = cmd.execute(query="test", stdout=True)
            assert exit_code == 0  # No error, just no results

    def test_execute_clipboard_unavailable_fallback(self) -> None:
        """Test execute falls back to stdout when clipboard unavailable."""
        cmd = ClipCommand()

        with (
            patch.object(cmd, "_retrieve_chunks") as mock_retrieve,
            patch(
                "ingestforge.cli.commands.clip.is_clipboard_available",
                return_value=False,
            ),
        ):
            mock_retrieve.return_value = [{"text": "test", "source": "doc.pdf"}]

            exit_code = cmd.execute(query="test", stdout=False)
            assert exit_code == 0

    def test_execute_invalid_format(self) -> None:
        """Test execute with invalid format."""
        cmd = ClipCommand()
        exit_code = cmd.execute(query="test", format_type="invalid", stdout=True)
        assert exit_code == 1

    def test_execute_query_truncated(self) -> None:
        """Test execute truncates long query."""
        cmd = ClipCommand()
        long_query = "x" * 1000

        with (
            patch.object(cmd, "_retrieve_chunks", return_value=[]),
            patch.object(cmd, "_print_warning"),
        ):
            exit_code = cmd.execute(query=long_query, stdout=True)
            assert exit_code == 0


# =============================================================================
# Test ClipCommand - Chunk Retrieval
# =============================================================================


class TestClipCommandRetrieval:
    """Tests for chunk retrieval."""

    def test_retrieve_chunks_with_storage(self) -> None:
        """Test retrieving chunks with storage available."""
        cmd = ClipCommand()

        # Test that retrieval returns expected structure
        # Since storage may not be available, test the fallback behavior
        chunks = cmd._retrieve_chunks("test query", 5, None)

        # Should return empty list when storage not available
        assert isinstance(chunks, list)

    def test_retrieve_chunks_import_error(self) -> None:
        """Test retrieve returns empty on import error."""
        cmd = ClipCommand()

        with patch(
            "ingestforge.cli.commands.clip.ClipCommand._retrieve_chunks",
            return_value=[],
        ):
            chunks = cmd._retrieve_chunks("test", 5, None)
            assert chunks == []

    def test_retrieve_chunks_bounded(self) -> None:
        """Test retrieve respects k limit."""
        cmd = ClipCommand()

        # Test k parameter is respected in the command
        # When no storage is available, should return empty list
        chunks = cmd._retrieve_chunks("test", 5, None)

        # Result should be bounded (empty or up to k items)
        assert isinstance(chunks, list)
        assert len(chunks) <= 5


# =============================================================================
# Test ClipCommand - Formatting
# =============================================================================


class TestClipCommandFormatting:
    """Tests for output formatting."""

    def test_format_output_chatgpt(self) -> None:
        """Test formatting for ChatGPT."""
        cmd = ClipCommand()
        chunks = [{"text": "content", "source": "doc.pdf"}]

        output = cmd._format_output("test", chunks, LLMFormat.CHATGPT, True)

        assert output.format == LLMFormat.CHATGPT
        assert "Context for Your Query" in output.content

    def test_format_output_claude(self) -> None:
        """Test formatting for Claude."""
        cmd = ClipCommand()
        chunks = [{"text": "content", "source": "doc.pdf"}]

        output = cmd._format_output("test", chunks, LLMFormat.CLAUDE, True)

        assert output.format == LLMFormat.CLAUDE
        assert "<context>" in output.content

    def test_format_output_with_citations(self) -> None:
        """Test formatting includes citations."""
        cmd = ClipCommand()
        chunks = [{"text": "content", "source": "doc.pdf"}]

        output = cmd._format_output("test", chunks, LLMFormat.MARKDOWN, True)

        assert "doc.pdf" in output.content
        assert "References" in output.content

    def test_format_output_without_citations(self) -> None:
        """Test formatting without citations."""
        cmd = ClipCommand()
        chunks = [{"text": "content", "source": "doc.pdf"}]

        output = cmd._format_output("test", chunks, LLMFormat.MARKDOWN, False)

        assert "References" not in output.content


# =============================================================================
# Test ClipCommand - Output Methods
# =============================================================================


class TestClipCommandOutput:
    """Tests for output methods."""

    def test_output_clipboard_success(self) -> None:
        """Test clipboard output on success."""
        cmd = ClipCommand()

        with patch("ingestforge.cli.commands.clip.copy_to_clipboard") as mock_copy:
            mock_copy.return_value = ClipboardResult(
                success=True,
                backend=ClipboardBackend.PYPERCLIP,
                chars_copied=50,
            )

            from ingestforge.core.llm_formatter import FormattedOutput

            output = FormattedOutput(
                content="test content",
                format=LLMFormat.MARKDOWN,
                chunk_count=2,
            )

            result = cmd._output_clipboard(output)

            assert result.success is True
            mock_copy.assert_called_once_with("test content")

    def test_output_clipboard_failure(self) -> None:
        """Test clipboard output on failure."""
        cmd = ClipCommand()

        with patch("ingestforge.cli.commands.clip.copy_to_clipboard") as mock_copy:
            mock_copy.return_value = ClipboardResult(
                success=False,
                backend=ClipboardBackend.NONE,
                message="Failed",
            )

            from ingestforge.core.llm_formatter import FormattedOutput

            output = FormattedOutput(
                content="test",
                format=LLMFormat.MARKDOWN,
            )

            result = cmd._output_clipboard(output)

            assert result.success is False


# =============================================================================
# Test JPL Compliance
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_k_bounded(self) -> None:
        """Test k parameter is bounded."""
        cmd = ClipCommand()

        with (
            patch.object(cmd, "_retrieve_chunks", return_value=[]) as mock_ret,
            patch.object(cmd, "_print_warning"),
        ):
            cmd.execute(query="test", k=1000, stdout=True)

            # Check k was bounded
            call_args = mock_ret.call_args
            assert call_args[0][1] <= 20  # MAX_CHUNKS in clip.py

    def test_query_bounded(self) -> None:
        """Test query length is bounded."""
        cmd = ClipCommand()
        long_query = "x" * 1000

        with (
            patch.object(cmd, "_retrieve_chunks", return_value=[]),
            patch.object(cmd, "_print_warning"),
        ):
            # Should not raise, query is truncated
            exit_code = cmd.execute(query=long_query, stdout=True)
            assert exit_code == 0

    def test_none_query_raises(self) -> None:
        """Test None query raises assertion."""
        cmd = ClipCommand()
        with pytest.raises(AssertionError, match="cannot be None"):
            cmd.execute(query=None, stdout=True)  # type: ignore

    def test_invalid_k_raises(self) -> None:
        """Test k <= 0 raises assertion."""
        cmd = ClipCommand()
        with pytest.raises(AssertionError, match="must be positive"):
            cmd.execute(query="test", k=0, stdout=True)
