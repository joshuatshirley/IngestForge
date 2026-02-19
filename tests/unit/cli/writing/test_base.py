"""
Tests for Writing Command Base Class.

This module tests the base class for writing assistance commands,
focusing on exception handling and logging.

Test Strategy
-------------
- Test proper exception handling (NASA JPL Rule #7)
- Verify logging behavior
- Test JSON parsing edge cases
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestLLMClient: LLM client initialization
- TestJSONParsing: JSON response parsing
- TestContextFormatting: Context formatting
"""

from unittest.mock import Mock, patch

from ingestforge.cli.writing.base import WritingCommand


# ============================================================================
# Test Helpers
# ============================================================================


class ConcreteWritingCommand(WritingCommand):
    """Concrete implementation for testing."""

    def execute(self, *args, **kwargs):
        """Dummy execute method."""
        return 0


def make_mock_context():
    """Create mock context dictionary."""
    return {"config": Mock(), "storage": Mock()}


# ============================================================================
# Test Classes
# ============================================================================


class TestLLMClient:
    """Tests for LLM client initialization.

    Rule #4: Focused test class
    Rule #7: Tests proper exception handling
    """

    @patch("ingestforge.llm.factory.get_best_available_client")
    def test_get_llm_client_success(self, mock_get_client):
        """Test successful LLM client retrieval."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        cmd = ConcreteWritingCommand()
        ctx = make_mock_context()

        result = cmd.get_llm_client(ctx)

        assert result == mock_client
        mock_get_client.assert_called_once_with(ctx["config"])

    @patch("ingestforge.llm.factory.get_best_available_client")
    def test_get_llm_client_import_error(self, mock_get_client, caplog):
        """Test LLM client handles ImportError with logging."""
        mock_get_client.side_effect = ImportError("Module not found")

        cmd = ConcreteWritingCommand()
        ctx = make_mock_context()

        result = cmd.get_llm_client(ctx)

        assert result is None
        assert "Failed to get LLM client" in caplog.text

    @patch("ingestforge.llm.factory.get_best_available_client")
    def test_get_llm_client_key_error(self, mock_get_client, caplog):
        """Test LLM client handles KeyError with logging."""
        mock_get_client.side_effect = KeyError("config")

        cmd = ConcreteWritingCommand()
        ctx = make_mock_context()

        result = cmd.get_llm_client(ctx)

        assert result is None
        assert "Failed to get LLM client" in caplog.text

    @patch("ingestforge.llm.factory.get_best_available_client")
    def test_get_llm_client_attribute_error(self, mock_get_client, caplog):
        """Test LLM client handles AttributeError with logging."""
        mock_get_client.side_effect = AttributeError("No attribute")

        cmd = ConcreteWritingCommand()
        ctx = make_mock_context()

        result = cmd.get_llm_client(ctx)

        assert result is None
        assert "Failed to get LLM client" in caplog.text

    @patch("ingestforge.llm.factory.get_best_available_client")
    def test_get_llm_client_unexpected_error(self, mock_get_client, caplog):
        """Test LLM client handles unexpected errors with logging."""
        mock_get_client.side_effect = RuntimeError("Unexpected error")

        cmd = ConcreteWritingCommand()
        ctx = make_mock_context()

        result = cmd.get_llm_client(ctx)

        assert result is None
        assert "Unexpected error getting LLM client" in caplog.text


class TestJSONParsing:
    """Tests for JSON response parsing.

    Rule #4: Focused test class
    Rule #7: Tests defensive exception handling
    """

    def test_parse_json_valid(self):
        """Test parsing valid JSON."""
        cmd = ConcreteWritingCommand()
        response = '{"key": "value", "number": 42}'

        result = cmd.parse_json(response)

        assert result == {"key": "value", "number": 42}

    def test_parse_json_invalid_returns_none(self, caplog):
        """Test invalid JSON returns None with logging."""
        cmd = ConcreteWritingCommand()
        response = "Not JSON at all"

        result = cmd.parse_json(response)

        assert result is None
        assert "Initial JSON parse failed" in caplog.text

    def test_parse_json_extracts_from_text(self, caplog):
        """Test extracting JSON from surrounding text."""
        cmd = ConcreteWritingCommand()
        response = 'Here is some text {"key": "value"} and more text'

        result = cmd.parse_json(response)

        assert result == {"key": "value"}

    def test_parse_json_extraction_failure(self, caplog):
        """Test extraction failure when embedded JSON is invalid."""
        cmd = ConcreteWritingCommand()
        response = "Text {invalid json} more text"

        result = cmd.parse_json(response)

        assert result is None
        assert "JSON extraction failed" in caplog.text

    def test_parse_json_no_json_in_text(self, caplog):
        """Test no JSON found in text."""
        cmd = ConcreteWritingCommand()
        response = "Plain text with no JSON"

        result = cmd.parse_json(response)

        assert result is None


class TestContextFormatting:
    """Tests for context formatting.

    Rule #4: Focused test class
    """

    def test_format_context_empty(self):
        """Test formatting empty chunk list."""
        cmd = ConcreteWritingCommand()

        result = cmd.format_context([])

        assert result == ""

    def test_format_context_single_chunk(self):
        """Test formatting single chunk."""
        cmd = ConcreteWritingCommand()
        chunk = Mock()
        chunk.text = "Test content"

        result = cmd.format_context([chunk])

        assert result == "Test content"

    def test_format_context_multiple_chunks(self):
        """Test formatting multiple chunks."""
        cmd = ConcreteWritingCommand()
        chunk1 = Mock()
        chunk1.text = "First chunk"
        chunk2 = Mock()
        chunk2.text = "Second chunk"

        result = cmd.format_context([chunk1, chunk2])

        assert "First chunk" in result
        assert "Second chunk" in result
        assert "\n\n" in result  # Chunks separated

    def test_format_context_respects_max_length(self):
        """Test context formatting respects max length."""
        cmd = ConcreteWritingCommand()
        chunk1 = Mock()
        chunk1.text = "a" * 100
        chunk2 = Mock()
        chunk2.text = "b" * 100

        result = cmd.format_context([chunk1, chunk2], max_length=150)

        # Should only include first chunk due to length limit
        assert "a" in result
        assert "b" not in result

    def test_format_context_handles_string_chunks(self):
        """Test formatting chunks without text attribute."""
        cmd = ConcreteWritingCommand()

        # Create a simple object without text attribute
        class SimpleChunk:
            def __str__(self):
                return "String representation"

        chunk = SimpleChunk()
        result = cmd.format_context([chunk])

        assert "String representation" in result


class TestSearchContext:
    """Tests for context searching.

    Rule #4: Focused test class
    """

    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_search_context_calls_storage(self, mock_spinner):
        """Test search context calls storage search."""
        cmd = ConcreteWritingCommand()
        storage = Mock()
        storage.search = Mock(return_value=["chunk1", "chunk2"])
        mock_spinner.side_effect = lambda fn, *args: fn()

        result = cmd.search_context(storage, "test query", k=30)

        storage.search.assert_called_once_with("test query", k=30)


class TestLLMGeneration:
    """Tests for LLM generation.

    Rule #4: Focused test class
    """

    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_generate_with_llm(self, mock_spinner):
        """Test LLM generation."""
        cmd = ConcreteWritingCommand()
        llm_client = Mock()
        llm_client.generate = Mock(return_value="Generated text")
        mock_spinner.side_effect = lambda fn, *args: fn()

        result = cmd.generate_with_llm(llm_client, "prompt", "description")

        assert result == "Generated text"
        llm_client.generate.assert_called_once_with("prompt")
