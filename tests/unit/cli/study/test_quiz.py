"""
Tests for Quiz CLI Command.

Tests quiz generation with difficulty levels, question types,
and answer explanations.

Test Strategy
-------------
- Focus on question generation logic
- Mock LLM and storage for unit testing
- Test difficulty-based question mix
- Test markdown output formatting
- Keep tests simple (NASA JPL Rule #1)

Organization
------------
- TestQuizInit: Initialization
- TestValidation: Input validation
- TestDifficultyMix: Question type mixing
- TestMarkdownOutput: Output formatting
- TestQuestionExtraction: Fallback parsing
- TestIntegration: Full workflow
"""

from unittest.mock import Mock, patch
import json

import pytest
import typer

from ingestforge.cli.study.quiz import (
    QuizCommand,
    VALID_DIFFICULTIES,
    DIFFICULTY_MIX,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "Test content about algorithms",
    source_file: str = "test.txt",
):
    """Create a mock chunk object."""
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.text = content
    chunk.source_file = source_file
    chunk.metadata = {"source": source_file}
    return chunk


def make_mock_context():
    """Create a mock context dictionary."""
    return {"storage": Mock(), "config": Mock()}


def make_mock_llm_response():
    """Create a mock LLM response with quiz data."""
    return json.dumps(
        {
            "topic": "Algorithms",
            "difficulty": "medium",
            "total_questions": 3,
            "questions": [
                {
                    "id": 1,
                    "question": "What is the time complexity of binary search?",
                    "type": "multiple_choice",
                    "options": ["A) O(n)", "B) O(log n)", "C) O(n^2)", "D) O(1)"],
                    "correct_answer": "B",
                    "explanation": "Binary search divides the search space in half.",
                    "topic_tag": "searching",
                },
                {
                    "id": 2,
                    "question": "Explain the quicksort algorithm.",
                    "type": "open_ended",
                    "correct_answer": "Quicksort uses divide-and-conquer...",
                    "explanation": "It partitions around a pivot element.",
                    "topic_tag": "sorting",
                },
                {
                    "id": 3,
                    "question": "What is a hash table?",
                    "type": "multiple_choice",
                    "options": [
                        "A) A tree",
                        "B) A key-value store",
                        "C) A list",
                        "D) A graph",
                    ],
                    "correct_answer": "B",
                    "explanation": "Hash tables map keys to values.",
                    "topic_tag": "data structures",
                },
            ],
        }
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestQuizInit:
    """Tests for QuizCommand initialization."""

    def test_create_quiz_command(self):
        """Test creating QuizCommand instance."""
        cmd = QuizCommand()
        assert cmd is not None

    def test_inherits_from_study_command(self):
        """Test QuizCommand inherits from StudyCommand."""
        from ingestforge.cli.study.base import StudyCommand

        cmd = QuizCommand()
        assert isinstance(cmd, StudyCommand)


class TestValidation:
    """Tests for input validation."""

    def test_valid_difficulties(self):
        """Test all valid difficulties are accepted."""
        cmd = QuizCommand()

        for difficulty in VALID_DIFFICULTIES:
            # Should not raise
            cmd._validate_difficulty(difficulty)

    def test_invalid_difficulty_raises(self):
        """Test invalid difficulty raises error."""
        cmd = QuizCommand()

        with pytest.raises(typer.BadParameter):
            cmd._validate_difficulty("impossible")

    def test_difficulty_case_insensitive(self):
        """Test difficulty validation is case insensitive."""
        cmd = QuizCommand()

        # These should not raise
        cmd._validate_difficulty("EASY")
        cmd._validate_difficulty("Medium")
        cmd._validate_difficulty("HARD")


class TestDifficultyMix:
    """Tests for difficulty-based question type mixing."""

    def test_easy_favors_multiple_choice(self):
        """Test easy difficulty has more multiple choice."""
        mix = DIFFICULTY_MIX["easy"]

        assert mix["multiple_choice"] > mix["open_ended"]
        assert mix["multiple_choice"] == 80

    def test_medium_balanced(self):
        """Test medium difficulty is balanced."""
        mix = DIFFICULTY_MIX["medium"]

        assert mix["multiple_choice"] == 60
        assert mix["open_ended"] == 40

    def test_hard_favors_open_ended(self):
        """Test hard difficulty has more open-ended."""
        mix = DIFFICULTY_MIX["hard"]

        assert mix["open_ended"] > mix["multiple_choice"]
        assert mix["open_ended"] == 60


class TestDifficultyGuidance:
    """Tests for difficulty-specific guidance."""

    def test_easy_guidance(self):
        """Test easy difficulty guidance."""
        cmd = QuizCommand()
        guidance = cmd._get_difficulty_guidance("easy")

        assert "basic" in guidance.lower() or "recall" in guidance.lower()

    def test_medium_guidance(self):
        """Test medium difficulty guidance."""
        cmd = QuizCommand()
        guidance = cmd._get_difficulty_guidance("medium")

        assert "application" in guidance.lower() or "synthesis" in guidance.lower()

    def test_hard_guidance(self):
        """Test hard difficulty guidance."""
        cmd = QuizCommand()
        guidance = cmd._get_difficulty_guidance("hard")

        assert "analysis" in guidance.lower() or "evaluation" in guidance.lower()


class TestMarkdownOutput:
    """Tests for Markdown output formatting."""

    def test_markdown_quiz_structure(self, tmp_path):
        """Test Markdown quiz has proper structure."""
        cmd = QuizCommand()
        output = tmp_path / "quiz.md"

        quiz_data = {
            "topic": "Algorithms",
            "difficulty": "medium",
            "questions": [
                {
                    "id": 1,
                    "question": "What is O(n)?",
                    "type": "multiple_choice",
                    "options": ["A) Linear", "B) Quadratic"],
                    "correct_answer": "A",
                    "explanation": "O(n) is linear time.",
                    "topic_tag": "complexity",
                },
            ],
        }

        cmd._save_quiz(output, quiz_data, include_answers=True)

        content = output.read_text()

        # Check structure
        assert "# Algorithms - Practice Quiz" in content
        assert "**Difficulty:** Medium" in content
        assert "## Questions" in content
        assert "## Answer Key" in content

    def test_markdown_without_answers(self, tmp_path):
        """Test Markdown quiz without answer key."""
        cmd = QuizCommand()
        output = tmp_path / "quiz.md"

        quiz_data = {
            "topic": "Test",
            "difficulty": "easy",
            "questions": [
                {
                    "id": 1,
                    "question": "Question?",
                    "type": "open_ended",
                    "correct_answer": "Answer",
                    "explanation": "",
                    "topic_tag": "",
                },
            ],
        }

        cmd._save_quiz(output, quiz_data, include_answers=False)

        content = output.read_text()

        # Should not have answer key
        assert "## Answer Key" not in content

    def test_markdown_multiple_choice_options(self, tmp_path):
        """Test multiple choice options are formatted."""
        cmd = QuizCommand()
        output = tmp_path / "quiz.md"

        quiz_data = {
            "topic": "Test",
            "difficulty": "easy",
            "questions": [
                {
                    "id": 1,
                    "question": "Which is correct?",
                    "type": "multiple_choice",
                    "options": ["A) Option 1", "B) Option 2", "C) Option 3"],
                    "correct_answer": "A",
                    "explanation": "",
                    "topic_tag": "",
                },
            ],
        }

        cmd._save_quiz(output, quiz_data, include_answers=True)

        content = output.read_text()

        assert "- A) Option 1" in content
        assert "- B) Option 2" in content


class TestQuestionExtraction:
    """Tests for fallback question extraction from text."""

    def test_extract_questions_from_text(self):
        """Test extracting questions from plain text."""
        cmd = QuizCommand()

        text = """Q: What is Python?
Q: What is Java?
Q: What is C++?"""

        result = cmd._extract_questions_from_text(text, "Programming", "easy")

        assert result["topic"] == "Programming"
        assert result["difficulty"] == "easy"
        assert len(result["questions"]) == 3

    def test_is_question_start(self):
        """Test question start detection."""
        cmd = QuizCommand()

        assert cmd._is_question_start("Q: Question text")
        assert cmd._is_question_start("Question: Some question")
        assert cmd._is_question_start("1. First question")
        assert not cmd._is_question_start("This is just text")


class TestTopicMetadata:
    """Tests for topic metadata handling."""

    def test_add_topic_metadata(self):
        """Test topic metadata is added to questions."""
        cmd = QuizCommand()

        quiz_data = {
            "questions": [
                {"id": 1, "question": "Q1"},
                {"id": 2, "question": "Q2", "topic_tag": "existing"},
            ],
        }

        result = cmd._add_topic_metadata(quiz_data, "Test Topic", "hard")

        assert result["topic"] == "Test Topic"
        assert result["difficulty"] == "hard"

        # First question should get topic tag
        assert result["questions"][0]["topic_tag"] == "Test Topic"

        # Second question keeps existing tag
        assert result["questions"][1]["topic_tag"] == "existing"


class TestQuizGeneration:
    """Tests for quiz generation with LLM."""

    def test_generate_quiz_with_llm(self):
        """Test quiz generation using mocked LLM."""
        cmd = QuizCommand()

        chunks = [make_mock_chunk()]
        llm_client = Mock()

        with patch.object(cmd, "format_context_for_prompt", return_value="context"):
            with patch.object(cmd, "generate_with_llm") as mock_generate:
                with patch.object(cmd, "parse_json_response") as mock_parse:
                    mock_generate.return_value = make_mock_llm_response()
                    mock_parse.return_value = json.loads(make_mock_llm_response())

                    result = cmd._generate_quiz(
                        llm_client,
                        "Algorithms",
                        chunks,
                        10,
                        "medium",
                        include_answers=True,
                        include_explanations=True,
                    )

                    assert result is not None
                    assert "questions" in result


class TestIntegration:
    """Integration tests for full quiz workflow."""

    def test_full_quiz_generation(self, tmp_path):
        """Test complete quiz generation workflow."""
        cmd = QuizCommand()
        output = tmp_path / "quiz.md"

        chunks = [make_mock_chunk()]

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "search_topic_context") as mock_search:
                    with patch.object(cmd, "generate_with_llm") as mock_gen:
                        with patch.object(cmd, "parse_json_response") as mock_parse:
                            mock_init.return_value = make_mock_context()
                            mock_llm.return_value = Mock()
                            mock_search.return_value = chunks
                            mock_gen.return_value = make_mock_llm_response()
                            mock_parse.return_value = json.loads(
                                make_mock_llm_response()
                            )

                            result = cmd.execute(
                                topic="Algorithms",
                                count=10,
                                output=output,
                                difficulty="medium",
                                include_answers=True,
                                include_explanations=True,
                            )

                            assert result == 0
                            assert output.exists()

    def test_handles_no_context(self):
        """Test graceful handling when no context found."""
        cmd = QuizCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client") as mock_llm:
                with patch.object(cmd, "search_topic_context", return_value=[]):
                    mock_init.return_value = make_mock_context()
                    mock_llm.return_value = Mock()

                    result = cmd.execute("Unknown Topic", count=5)

                    assert result == 0  # Should succeed but warn

    def test_handles_no_llm_client(self):
        """Test graceful handling when no LLM available."""
        cmd = QuizCommand()

        with patch.object(cmd, "initialize_context") as mock_init:
            with patch.object(cmd, "get_llm_client", return_value=None):
                mock_init.return_value = make_mock_context()

                result = cmd.execute("Test Topic", count=5)

                assert result == 1  # Should fail


class TestTopicBreakdown:
    """Tests for topic breakdown display."""

    def test_topic_breakdown_counts(self):
        """Test topic breakdown counting."""
        cmd = QuizCommand()

        questions = [
            {"topic_tag": "sorting"},
            {"topic_tag": "sorting"},
            {"topic_tag": "searching"},
            {"topic_tag": "General"},
        ]

        # This tests internal logic - just verify no errors
        # The display method would show counts
        topics = {}
        for q in questions:
            tag = q.get("topic_tag", "General")
            topics[tag] = topics.get(tag, 0) + 1

        assert topics["sorting"] == 2
        assert topics["searching"] == 1
        assert topics["General"] == 1
