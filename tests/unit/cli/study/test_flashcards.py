"""
Tests for Flashcards CLI Command.

Tests the flashcard generation with multiple format support
(Anki, Quizlet, Markdown, JSON).

Test Strategy
-------------
- Focus on format conversion and output
- Mock LLM and storage for unit testing
- Test each format independently
- Test validation logic
- Keep tests simple (NASA JPL Rule #1)

Organization
------------
- TestFlashcardsInit: Initialization
- TestValidation: Input validation
- TestFormatConversion: Format-specific output
- TestAnkiFormat: Anki CSV generation
- TestQuizletFormat: Quizlet tab-separated output
- TestMarkdownFormat: Markdown document generation
- TestSourceCitations: Source citation handling
"""

from unittest.mock import Mock, patch
import json

import pytest
import typer

from ingestforge.cli.study.flashcards import (
    FlashcardsCommand,
    VALID_FORMATS,
    VALID_CARD_TYPES,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "Test content about machine learning",
    source_file: str = "test.txt",
    **metadata,
):
    """Create a mock chunk object."""
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.text = content
    chunk.source_file = source_file
    chunk.metadata = {"source": source_file, **metadata}
    return chunk


def make_mock_context(has_storage: bool = True):
    """Create a mock context dictionary."""
    ctx = {}
    if has_storage:
        ctx["storage"] = Mock()
        ctx["config"] = Mock()
    return ctx


def make_mock_llm_client():
    """Create a mock LLM client."""
    llm = Mock()
    llm.generate = Mock(
        return_value=json.dumps(
            {
                "topic": "Test Topic",
                "card_type": "definition",
                "cards": [
                    {
                        "front": "What is machine learning?",
                        "back": "A subset of AI that learns from data",
                        "hint": "Think about patterns",
                        "tags": ["ML", "AI"],
                        "source": "test.txt",
                    },
                    {
                        "front": "What is deep learning?",
                        "back": "Neural networks with multiple layers",
                        "hint": "Think about neurons",
                        "tags": ["DL", "AI"],
                        "source": "test.txt",
                    },
                ],
            }
        )
    )
    return llm


# ============================================================================
# Test Classes
# ============================================================================


class TestFlashcardsInit:
    """Tests for FlashcardsCommand initialization."""

    def test_create_flashcards_command(self):
        """Test creating FlashcardsCommand instance."""
        cmd = FlashcardsCommand()
        assert cmd is not None

    def test_inherits_from_study_command(self):
        """Test FlashcardsCommand inherits from StudyCommand."""
        from ingestforge.cli.study.base import StudyCommand

        cmd = FlashcardsCommand()
        assert isinstance(cmd, StudyCommand)


class TestValidation:
    """Tests for input validation."""

    def test_valid_card_types(self):
        """Test all valid card types are accepted."""
        cmd = FlashcardsCommand()

        for card_type in VALID_CARD_TYPES:
            # Should not raise
            cmd._validate_card_type(card_type)

    def test_invalid_card_type_raises(self):
        """Test invalid card type raises error."""
        cmd = FlashcardsCommand()

        with pytest.raises(typer.BadParameter):
            cmd._validate_card_type("invalid_type")

    def test_valid_formats(self):
        """Test all valid formats are accepted."""
        cmd = FlashcardsCommand()

        for fmt in VALID_FORMATS:
            # Should not raise
            cmd._validate_format(fmt)

    def test_invalid_format_raises(self):
        """Test invalid format raises error."""
        cmd = FlashcardsCommand()

        with pytest.raises(typer.BadParameter):
            cmd._validate_format("invalid_format")


class TestAnkiFormat:
    """Tests for Anki CSV format generation."""

    def test_anki_csv_creation(self, tmp_path):
        """Test Anki CSV file is created."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.csv"

        flashcard_data = {
            "topic": "Test",
            "card_type": "definition",
            "cards": [
                {"front": "Q1", "back": "A1", "tags": ["tag1"], "source": "doc.txt"},
                {"front": "Q2", "back": "A2", "tags": ["tag2"], "source": "doc.txt"},
            ],
        }

        cmd._save_anki_format(output, flashcard_data)

        assert output.exists()

    def test_anki_csv_format(self, tmp_path):
        """Test Anki CSV format is correct."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.csv"

        flashcard_data = {
            "topic": "Test",
            "card_type": "definition",
            "cards": [
                {
                    "front": "What is Python?",
                    "back": "A programming language",
                    "tags": ["python"],
                    "source": "doc.txt",
                    "hint": "Snake",
                },
            ],
        }

        cmd._save_anki_format(output, flashcard_data)

        content = output.read_text()
        lines = content.strip().split("\n")

        # Check header
        assert "Front" in lines[0]
        assert "Back" in lines[0]

        # Check data row contains content
        assert "Python" in content
        assert "programming language" in content

    def test_anki_includes_hints_in_back(self, tmp_path):
        """Test that hints are included in the back of the card."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.csv"

        flashcard_data = {
            "topic": "Test",
            "cards": [
                {
                    "front": "Question",
                    "back": "Answer",
                    "hint": "This is a hint",
                    "tags": [],
                    "source": "",
                },
            ],
        }

        cmd._save_anki_format(output, flashcard_data)

        content = output.read_text()
        assert "Hint:" in content


class TestQuizletFormat:
    """Tests for Quizlet tab-separated format."""

    def test_quizlet_format_creation(self, tmp_path):
        """Test Quizlet format file is created."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.txt"

        flashcard_data = {
            "topic": "Test",
            "cards": [
                {"front": "Term1", "back": "Definition1", "tags": []},
                {"front": "Term2", "back": "Definition2", "tags": []},
            ],
        }

        cmd._save_quizlet_format(output, flashcard_data)

        assert output.exists()

    def test_quizlet_tab_separated(self, tmp_path):
        """Test Quizlet format uses tabs."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.txt"

        flashcard_data = {
            "topic": "Test",
            "cards": [
                {"front": "Term", "back": "Definition", "tags": []},
            ],
        }

        cmd._save_quizlet_format(output, flashcard_data)

        content = output.read_text()
        assert "\t" in content
        assert "Term\tDefinition" in content

    def test_quizlet_removes_newlines(self, tmp_path):
        """Test that newlines are removed from cards."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.txt"

        flashcard_data = {
            "topic": "Test",
            "cards": [
                {"front": "Line1\nLine2", "back": "Answer\nMultiline", "tags": []},
            ],
        }

        cmd._save_quizlet_format(output, flashcard_data)

        content = output.read_text()
        # Should have newlines removed within fields
        assert "Line1 Line2" in content


class TestMarkdownFormat:
    """Tests for Markdown format generation."""

    def test_markdown_format_creation(self, tmp_path):
        """Test Markdown file is created."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.md"

        flashcard_data = {
            "topic": "Test Topic",
            "card_type": "definition",
            "cards": [
                {"front": "Q1", "back": "A1", "tags": ["tag1"], "source": "doc.txt"},
            ],
        }

        cmd._save_markdown_format(output, flashcard_data)

        assert output.exists()

    def test_markdown_structure(self, tmp_path):
        """Test Markdown has proper structure."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.md"

        flashcard_data = {
            "topic": "Machine Learning",
            "card_type": "concept",
            "cards": [
                {
                    "front": "Neural Network",
                    "back": "A computing system",
                    "tags": ["AI"],
                    "hint": "Think biology",
                },
            ],
        }

        cmd._save_markdown_format(output, flashcard_data)

        content = output.read_text()

        # Check structure elements
        assert "# Machine Learning - Flashcards" in content
        assert "## Card 1" in content
        assert "**Q:**" in content
        assert "**A:**" in content

    def test_markdown_includes_sources(self, tmp_path):
        """Test Markdown includes sources section."""
        cmd = FlashcardsCommand()
        output = tmp_path / "flashcards.md"

        flashcard_data = {
            "topic": "Test",
            "cards": [{"front": "Q", "back": "A", "tags": []}],
            "sources": [
                {"source": "document1.pdf", "page": "5"},
                {"source": "document2.pdf", "page": ""},
            ],
        }

        cmd._save_markdown_format(output, flashcard_data)

        content = output.read_text()
        assert "## Sources" in content
        assert "document1.pdf" in content


class TestSourceCitations:
    """Tests for source citation handling."""

    def test_extract_sources_from_chunks(self):
        """Test source extraction from chunks."""
        cmd = FlashcardsCommand()

        chunks = [
            make_mock_chunk(chunk_id="1", source_file="doc1.pdf", page="5"),
            make_mock_chunk(chunk_id="2", source_file="doc1.pdf", page="10"),
            make_mock_chunk(chunk_id="3", source_file="doc2.pdf"),
        ]

        sources = cmd._extract_sources(chunks)

        # Should dedupe by source
        source_names = [s["source"] for s in sources]
        assert "doc1.pdf" in source_names
        assert "doc2.pdf" in source_names
        assert len(sources) == 2


class TestFlashcardGeneration:
    """Tests for flashcard generation with LLM."""

    def test_generate_flashcards_with_llm(self, tmp_path):
        """Test flashcard generation using mocked LLM."""
        cmd = FlashcardsCommand()

        chunks = [make_mock_chunk()]
        llm_client = make_mock_llm_client()

        with patch.object(cmd, "format_context_for_prompt", return_value="context"):
            with patch.object(cmd, "generate_with_llm") as mock_generate:
                mock_generate.return_value = json.dumps(
                    {
                        "topic": "Test",
                        "cards": [{"front": "Q", "back": "A", "tags": []}],
                    }
                )

                result = cmd._generate_flashcards(
                    llm_client, "Test", chunks, 10, "definition", False
                )

                assert result is not None
                assert "cards" in result

    def test_fallback_on_invalid_json(self):
        """Test fallback when LLM returns invalid JSON."""
        cmd = FlashcardsCommand()

        chunks = [make_mock_chunk()]
        llm_client = Mock()
        llm_client.generate = Mock(return_value="This is not valid JSON")

        with patch.object(cmd, "format_context_for_prompt", return_value="context"):
            with patch.object(cmd, "generate_with_llm", return_value="invalid"):
                with patch.object(cmd, "parse_json_response", return_value=None):
                    result = cmd._generate_flashcards(
                        llm_client, "Test", chunks, 3, "definition", False
                    )

                    # Should return fallback structure
                    assert result is not None
                    assert "cards" in result
                    assert len(result["cards"]) <= 3


class TestIntegration:
    """Integration tests for full flashcard workflow."""

    def test_full_flashcard_generation(self, tmp_path):
        """Test complete flashcard generation workflow."""
        cmd = FlashcardsCommand()
        output = tmp_path / "cards.csv"

        chunks = [make_mock_chunk(), make_mock_chunk(chunk_id="2")]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "search_topic_context") as mock_search:
                    with patch.object(cmd, "generate_with_llm") as mock_gen:
                        mock_init.return_value = make_mock_context()
                        mock_llm.return_value = make_mock_llm_client()
                        mock_search.return_value = chunks
                        mock_gen.return_value = json.dumps(
                            {
                                "topic": "Test",
                                "cards": [
                                    {"front": "Q1", "back": "A1", "tags": []},
                                    {"front": "Q2", "back": "A2", "tags": []},
                                ],
                            }
                        )

                        result = cmd.execute(
                            topic="Machine Learning",
                            count=10,
                            output=output,
                            output_format="anki",
                        )

                        assert result == 0
                        assert output.exists()

    def test_handles_no_llm_client(self):
        """Test graceful handling when no LLM available."""
        cmd = FlashcardsCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client", return_value=None):
                mock_init.return_value = make_mock_context()

                result = cmd.execute("Test Topic", count=5)

                assert result == 1  # Should fail


class TestCardTypeInstructions:
    """Tests for card type specific instructions."""

    def test_definition_type_instructions(self):
        """Test definition card type instructions."""
        cmd = FlashcardsCommand()
        instructions = cmd._get_card_type_instructions("definition")

        assert "term" in instructions.lower()
        assert "definition" in instructions.lower()

    def test_concept_type_instructions(self):
        """Test concept card type instructions."""
        cmd = FlashcardsCommand()
        instructions = cmd._get_card_type_instructions("concept")

        assert "concept" in instructions.lower()
        assert "explanation" in instructions.lower()

    def test_fact_type_instructions(self):
        """Test fact card type instructions."""
        cmd = FlashcardsCommand()
        instructions = cmd._get_card_type_instructions("fact")

        assert "question" in instructions.lower()
        assert "answer" in instructions.lower()

    def test_process_type_instructions(self):
        """Test process card type instructions."""
        cmd = FlashcardsCommand()
        instructions = cmd._get_card_type_instructions("process")

        assert "process" in instructions.lower()
        assert "step" in instructions.lower()
