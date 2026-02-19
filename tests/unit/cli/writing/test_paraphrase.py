"""
Tests for Paraphrase Module.

This module tests the paraphrase and rewrite functionality including:
- Paraphraser class methods
- ParaphraseResult and RewriteResult dataclasses
- Style and tone validation
- Batch processing

Test Strategy
-------------
- Test each public method of Paraphraser
- Test dataclass operations
- Test validation methods
- Test batch processing
- Follow NASA JPL Rule #1: Simple test structure

Organization
------------
- TestParaphraseResult: ParaphraseResult dataclass tests
- TestRewriteResult: RewriteResult dataclass tests
- TestBatchResult: BatchResult dataclass tests
- TestParaphraser: Paraphraser class tests
- TestParaphraseCommand: CLI command tests
- TestRewriteCommand: CLI rewrite command tests
"""

from unittest.mock import Mock, patch
import json

from ingestforge.cli.writing.paraphrase import (
    ParaphraseResult,
    RewriteResult,
    BatchResult,
    Paraphraser,
    ParaphraseCommand,
    RewriteCommand,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_llm_client(response: str = "Paraphrased text"):
    """Create mock LLM client."""
    client = Mock()
    client.generate.return_value = response
    return client


# ============================================================================
# TestParaphraseResult: ParaphraseResult dataclass tests
# ============================================================================


class TestParaphraseResult:
    """Tests for ParaphraseResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = ParaphraseResult(
            original="Original text",
            paraphrased="Rephrased text",
        )
        assert result.original == "Original text"
        assert result.paraphrased == "Rephrased text"
        assert result.style == "formal"

    def test_result_word_count_change(self):
        """Test automatic word count change calculation."""
        result = ParaphraseResult(
            original="One two three",
            paraphrased="One two three four five",
        )
        assert result.word_count_change == 2

    def test_result_to_dict(self):
        """Test dictionary conversion."""
        result = ParaphraseResult(
            original="A",
            paraphrased="B",
            style="casual",
        )
        data = result.to_dict()

        assert data["original"] == "A"
        assert data["paraphrased"] == "B"
        assert data["style"] == "casual"


# ============================================================================
# TestRewriteResult: RewriteResult dataclass tests
# ============================================================================


class TestRewriteResult:
    """Tests for RewriteResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        result = RewriteResult(
            original="Original",
            rewritten="Rewritten",
        )
        assert result.original == "Original"
        assert result.rewritten == "Rewritten"
        assert result.tone == "professional"

    def test_result_with_improvements(self):
        """Test result with improvements list."""
        result = RewriteResult(
            original="Text",
            rewritten="Better text",
            improvements=["Clearer language", "Better flow"],
        )
        assert len(result.improvements) == 2

    def test_result_to_dict(self):
        """Test dictionary conversion."""
        result = RewriteResult(
            original="A",
            rewritten="B",
            tone="formal",
            reading_level="high_school",
        )
        data = result.to_dict()

        assert data["tone"] == "formal"
        assert data["reading_level"] == "high_school"


# ============================================================================
# TestBatchResult: BatchResult dataclass tests
# ============================================================================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_batch_creation(self):
        """Test basic batch result creation."""
        batch = BatchResult()
        assert batch.total_processed == 0
        assert batch.total_failed == 0

    def test_add_result(self):
        """Test adding successful result."""
        batch = BatchResult()
        result = ParaphraseResult("A", "B")

        batch.add_result(result)

        assert batch.total_processed == 1
        assert len(batch.results) == 1

    def test_add_error(self):
        """Test adding error."""
        batch = BatchResult()

        batch.add_error("Test error")

        assert batch.total_failed == 1
        assert "Test error" in batch.errors


# ============================================================================
# TestParaphraser: Paraphraser class tests
# ============================================================================


class TestParaphraser:
    """Tests for Paraphraser class."""

    def test_initialization(self):
        """Test paraphraser initialization."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)
        assert paraphraser.llm_client == client

    def test_paraphrase_basic(self):
        """Test basic paraphrasing."""
        client = make_mock_llm_client("Rephrased text")
        paraphraser = Paraphraser(client)

        result = paraphraser.paraphrase("Original text")

        assert result.paraphrased == "Rephrased text"
        assert result.style == "formal"

    def test_paraphrase_empty_text(self):
        """Test paraphrasing empty text."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        result = paraphraser.paraphrase("")

        assert result.original == ""
        assert result.paraphrased == ""
        client.generate.assert_not_called()

    def test_paraphrase_all_styles(self):
        """Test paraphrasing with all styles."""
        client = make_mock_llm_client("Styled text")
        paraphraser = Paraphraser(client)

        for style in ["formal", "casual", "academic", "professional"]:
            result = paraphraser.paraphrase("Text", style)
            assert result.style == style

    def test_paraphrase_invalid_style(self):
        """Test paraphrasing with invalid style defaults to formal."""
        client = make_mock_llm_client("Text")
        paraphraser = Paraphraser(client)

        result = paraphraser.paraphrase("Text", "invalid_style")

        assert result.style == "formal"

    def test_paraphrase_multiple(self):
        """Test generating multiple paraphrases."""
        response = json.dumps(
            {
                "paraphrases": [
                    {
                        "version": "Version 1",
                        "style": "formal",
                        "maintains_meaning": True,
                    },
                    {
                        "version": "Version 2",
                        "style": "casual",
                        "maintains_meaning": True,
                    },
                ]
            }
        )
        client = make_mock_llm_client(response)
        paraphraser = Paraphraser(client)

        results = paraphraser.paraphrase_multiple("Text", count=2)

        assert len(results) == 2

    def test_paraphrase_multiple_empty_text(self):
        """Test multiple paraphrases with empty text."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        results = paraphraser.paraphrase_multiple("")

        assert results == []

    def test_rewrite_basic(self):
        """Test basic rewriting."""
        response = json.dumps(
            {"rewritten": "Rewritten text", "improvements": ["Better clarity"]}
        )
        client = make_mock_llm_client(response)
        paraphraser = Paraphraser(client)

        result = paraphraser.rewrite("Original text")

        assert (
            "Rewritten" in result.rewritten or "rewritten" in result.rewritten.lower()
        )

    def test_rewrite_empty_text(self):
        """Test rewriting empty text."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        result = paraphraser.rewrite("")

        assert result.original == ""
        assert result.rewritten == ""

    def test_rewrite_all_tones(self):
        """Test rewriting with all tones."""
        client = make_mock_llm_client('{"rewritten": "Text", "improvements": []}')
        paraphraser = Paraphraser(client)

        tones = [
            "formal",
            "casual",
            "professional",
            "academic",
            "enthusiastic",
            "neutral",
        ]
        for tone in tones:
            result = paraphraser.rewrite("Text", tone)
            assert result.tone == tone

    def test_rewrite_invalid_tone(self):
        """Test rewriting with invalid tone defaults to professional."""
        client = make_mock_llm_client('{"rewritten": "Text", "improvements": []}')
        paraphraser = Paraphraser(client)

        result = paraphraser.rewrite("Text", "invalid_tone")

        assert result.tone == "professional"

    def test_rewrite_preserve_meaning(self):
        """Test rewriting with preserve meaning flag."""
        client = make_mock_llm_client('{"rewritten": "Text", "improvements": []}')
        paraphraser = Paraphraser(client)

        result = paraphraser.rewrite("Text", preserve_meaning=True)

        # Check that prompt includes preservation instruction
        call_args = client.generate.call_args[0][0]
        assert "meaning" in call_args.lower() or "preserve" in call_args.lower()

    def test_simplify_basic(self):
        """Test basic simplification."""
        client = make_mock_llm_client("Simplified text")
        paraphraser = Paraphraser(client)

        result = paraphraser.simplify("Complex text")

        assert result.rewritten == "Simplified text"

    def test_simplify_empty_text(self):
        """Test simplification of empty text."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        result = paraphraser.simplify("")

        assert result.original == ""
        assert result.rewritten == ""

    def test_simplify_all_levels(self):
        """Test simplification at all reading levels."""
        client = make_mock_llm_client("Simple text")
        paraphraser = Paraphraser(client)

        levels = ["elementary", "high_school", "college", "professional"]
        for level in levels:
            result = paraphraser.simplify("Text", level)
            assert result.reading_level == level

    def test_simplify_invalid_level(self):
        """Test simplification with invalid level defaults to high_school."""
        client = make_mock_llm_client("Text")
        paraphraser = Paraphraser(client)

        result = paraphraser.simplify("Text", "invalid_level")

        assert result.reading_level == "high_school"

    def test_batch_process(self):
        """Test batch processing."""
        client = make_mock_llm_client("Processed text")
        paraphraser = Paraphraser(client)

        texts = ["Text 1", "Text 2", "Text 3"]
        result = paraphraser.batch_process(texts)

        assert result.total_processed == 3
        assert result.total_failed == 0

    def test_batch_process_with_errors(self):
        """Test batch processing with errors."""
        client = Mock()
        client.generate.side_effect = [
            "Processed",
            Exception("Error"),
            "Processed",
        ]
        paraphraser = Paraphraser(client)

        texts = ["Text 1", "Text 2", "Text 3"]
        result = paraphraser.batch_process(texts)

        assert result.total_processed == 2
        assert result.total_failed == 1

    def test_process_file(self, tmp_path):
        """Test file processing."""
        client = make_mock_llm_client("Processed")
        paraphraser = Paraphraser(client)

        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3")

        result = paraphraser.process_file(test_file)

        assert result.total_processed == 3

    def test_process_file_not_found(self, tmp_path):
        """Test file processing with non-existent file."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        result = paraphraser.process_file(tmp_path / "nonexistent.txt")

        assert result.total_failed == 1
        assert "not found" in result.errors[0].lower()

    def test_validate_style(self):
        """Test style validation."""
        paraphraser = Paraphraser(make_mock_llm_client())

        assert paraphraser._validate_style("FORMAL") == "formal"
        assert paraphraser._validate_style("CaSuAl") == "casual"
        assert paraphraser._validate_style("unknown") == "formal"

    def test_validate_tone(self):
        """Test tone validation."""
        paraphraser = Paraphraser(make_mock_llm_client())

        assert paraphraser._validate_tone("PROFESSIONAL") == "professional"
        assert paraphraser._validate_tone("EnThUsIaStIc") == "enthusiastic"
        assert paraphraser._validate_tone("invalid") == "professional"

    def test_validate_reading_level(self):
        """Test reading level validation."""
        paraphraser = Paraphraser(make_mock_llm_client())

        assert paraphraser._validate_reading_level("ELEMENTARY") == "elementary"
        assert paraphraser._validate_reading_level("high school") == "high_school"
        assert paraphraser._validate_reading_level("invalid") == "high_school"

    def test_parse_json_response_valid(self):
        """Test parsing valid JSON."""
        paraphraser = Paraphraser(make_mock_llm_client())

        data = paraphraser._parse_json_response('{"key": "value"}')

        assert data["key"] == "value"

    def test_parse_json_response_invalid(self):
        """Test parsing invalid JSON returns empty dict."""
        paraphraser = Paraphraser(make_mock_llm_client())

        data = paraphraser._parse_json_response("not json")

        assert data == {}

    def test_parse_json_response_embedded(self):
        """Test parsing embedded JSON."""
        paraphraser = Paraphraser(make_mock_llm_client())

        data = paraphraser._parse_json_response('Here is {"key": "value"} text')

        assert data["key"] == "value"


# ============================================================================
# TestParaphraseCommand: CLI command tests
# ============================================================================


class TestParaphraseCommand:
    """Tests for ParaphraseCommand CLI class."""

    @patch.object(ParaphraseCommand, "initialize_context")
    @patch.object(ParaphraseCommand, "get_llm_client")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_success(self, mock_spinner, mock_llm, mock_ctx):
        """Test successful execution."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        response = json.dumps(
            {
                "paraphrases": [
                    {"version": "V1", "style": "formal", "maintains_meaning": True}
                ]
            }
        )
        mock_llm.return_value = make_mock_llm_client(response)
        mock_spinner.side_effect = lambda fn, *args: fn()

        cmd = ParaphraseCommand(console=Mock())
        result = cmd.execute("Test text")

        assert result == 0

    @patch.object(ParaphraseCommand, "initialize_context")
    @patch.object(ParaphraseCommand, "get_llm_client")
    def test_execute_no_llm(self, mock_llm, mock_ctx):
        """Test execution without LLM client."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = None

        cmd = ParaphraseCommand(console=Mock())
        result = cmd.execute("Text")

        assert result == 1

    @patch.object(ParaphraseCommand, "initialize_context")
    @patch.object(ParaphraseCommand, "get_llm_client")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_with_output(self, mock_spinner, mock_llm, mock_ctx, tmp_path):
        """Test execution with output file."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        response = json.dumps({"paraphrases": []})
        mock_llm.return_value = make_mock_llm_client(response)
        mock_spinner.side_effect = lambda fn, *args: fn()

        output = tmp_path / "output.json"
        cmd = ParaphraseCommand(console=Mock())
        result = cmd.execute("Text", output=output)

        assert result == 0
        assert output.exists()

    @patch.object(ParaphraseCommand, "initialize_context")
    def test_execute_handles_exception(self, mock_ctx):
        """Test exception handling."""
        mock_ctx.side_effect = Exception("Error")

        cmd = ParaphraseCommand(console=Mock())
        result = cmd.execute("Text")

        assert result == 1


# ============================================================================
# TestRewriteCommand: CLI rewrite command tests
# ============================================================================


class TestRewriteCommand:
    """Tests for RewriteCommand CLI class."""

    @patch.object(RewriteCommand, "initialize_context")
    @patch.object(RewriteCommand, "get_llm_client")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_success(self, mock_spinner, mock_llm, mock_ctx, tmp_path):
        """Test successful execution."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client(
            '{"rewritten": "Text", "improvements": []}'
        )
        mock_spinner.side_effect = lambda fn, *args: fn()

        # Create test file
        input_file = tmp_path / "input.txt"
        input_file.write_text("Original text")

        cmd = RewriteCommand(console=Mock())
        result = cmd.execute(input_file)

        assert result == 0

    def test_execute_file_not_found(self, tmp_path):
        """Test execution with non-existent file."""
        cmd = RewriteCommand(console=Mock())
        result = cmd.execute(tmp_path / "nonexistent.txt")

        assert result == 1

    @patch.object(RewriteCommand, "initialize_context")
    @patch.object(RewriteCommand, "get_llm_client")
    def test_execute_no_llm(self, mock_llm, mock_ctx, tmp_path):
        """Test execution without LLM client."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = None

        input_file = tmp_path / "input.txt"
        input_file.write_text("Text")

        cmd = RewriteCommand(console=Mock())
        result = cmd.execute(input_file)

        assert result == 1

    @patch.object(RewriteCommand, "initialize_context")
    @patch.object(RewriteCommand, "get_llm_client")
    @patch("ingestforge.cli.core.ProgressManager.run_with_spinner")
    def test_execute_with_output(self, mock_spinner, mock_llm, mock_ctx, tmp_path):
        """Test execution with output file."""
        mock_ctx.return_value = {"storage": Mock(), "config": Mock()}
        mock_llm.return_value = make_mock_llm_client(
            '{"rewritten": "Rewritten", "improvements": []}'
        )
        mock_spinner.side_effect = lambda fn, *args: fn()

        input_file = tmp_path / "input.txt"
        input_file.write_text("Original")
        output_file = tmp_path / "output.txt"

        cmd = RewriteCommand(console=Mock())
        result = cmd.execute(input_file, output=output_file)

        assert result == 0
        assert output_file.exists()


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestParaphraseEdgeCases:
    """Edge case tests for paraphrase functionality."""

    def test_whitespace_only_text(self):
        """Test paraphrasing whitespace-only text."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        result = paraphraser.paraphrase("   \n\t  ")

        assert result.paraphrased == "   \n\t  "

    def test_very_long_text(self):
        """Test paraphrasing very long text."""
        client = make_mock_llm_client("Paraphrased long text")
        paraphraser = Paraphraser(client)

        long_text = "Word " * 1000
        result = paraphraser.paraphrase(long_text)

        assert result.paraphrased == "Paraphrased long text"

    def test_special_characters(self):
        """Test paraphrasing text with special characters."""
        client = make_mock_llm_client("Cleaned text")
        paraphraser = Paraphraser(client)

        result = paraphraser.paraphrase('Text with <special> & "chars"')

        assert result.paraphrased == "Cleaned text"

    def test_unicode_text(self):
        """Test paraphrasing unicode text."""
        client = make_mock_llm_client("Translated text")
        paraphraser = Paraphraser(client)

        result = paraphraser.paraphrase("Unicode: 日本語 中文 한국어")

        assert result.original == "Unicode: 日本語 中文 한국어"

    def test_batch_empty_list(self):
        """Test batch processing empty list."""
        client = make_mock_llm_client()
        paraphraser = Paraphraser(client)

        result = paraphraser.batch_process([])

        assert result.total_processed == 0
        assert result.total_failed == 0

    def test_json_response_with_plain_text_fallback(self):
        """Test rewrite handles plain text response."""
        client = make_mock_llm_client("Plain rewritten text")
        paraphraser = Paraphraser(client)

        result = paraphraser.rewrite("Original")

        # Should use plain text as fallback
        assert len(result.rewritten) > 0
