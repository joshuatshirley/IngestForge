"""Tests for explanations module (QUIZ-002.1).

Tests the LLM-powered explanation generator:
- Explanation dataclass
- ExplanationRequest validation
- ExplanationGenerator with mocked LLM
- Fallback behavior when LLM unavailable
"""

from unittest.mock import Mock

from ingestforge.study.explanations import (
    Explanation,
    ExplanationRequest,
    ExplanationGenerator,
    explain_answer,
    MAX_EXPLANATION_TOKENS,
)


class TestExplanation:
    """Test Explanation dataclass."""

    def test_basic_explanation(self) -> None:
        """Should create basic explanation."""
        exp = Explanation(
            text="Paris is the capital because...",
            question="What is the capital of France?",
            correct_answer="Paris",
        )

        assert exp.text == "Paris is the capital because..."
        assert exp.question == "What is the capital of France?"
        assert exp.correct_answer == "Paris"
        assert exp.user_answer is None
        assert exp.source == "llm"

    def test_was_wrong_false(self) -> None:
        """was_wrong should be False when no user answer."""
        exp = Explanation(
            text="Explanation",
            question="Q",
            correct_answer="A",
        )

        assert not exp.was_wrong

    def test_was_wrong_correct_answer(self) -> None:
        """was_wrong should be False when user was correct."""
        exp = Explanation(
            text="Explanation",
            question="Q",
            correct_answer="A",
            user_answer="A",
        )

        assert not exp.was_wrong

    def test_was_wrong_true(self) -> None:
        """was_wrong should be True when user was incorrect."""
        exp = Explanation(
            text="Explanation",
            question="Q",
            correct_answer="A",
            user_answer="B",
        )

        assert exp.was_wrong

    def test_is_fallback_false(self) -> None:
        """is_fallback should be False for LLM explanations."""
        exp = Explanation(
            text="Explanation",
            question="Q",
            correct_answer="A",
            source="llm",
        )

        assert not exp.is_fallback

    def test_is_fallback_true(self) -> None:
        """is_fallback should be True for fallback explanations."""
        exp = Explanation(
            text="Explanation",
            question="Q",
            correct_answer="A",
            source="fallback",
        )

        assert exp.is_fallback


class TestExplanationRequest:
    """Test ExplanationRequest dataclass."""

    def test_valid_request(self) -> None:
        """Valid request should have no errors."""
        request = ExplanationRequest(
            question="What is 2+2?",
            correct_answer="4",
        )

        errors = request.validate()
        assert len(errors) == 0

    def test_empty_question(self) -> None:
        """Empty question should be invalid."""
        request = ExplanationRequest(
            question="",
            correct_answer="4",
        )

        errors = request.validate()
        assert len(errors) == 1
        assert "question" in errors[0].lower()

    def test_whitespace_question(self) -> None:
        """Whitespace-only question should be invalid."""
        request = ExplanationRequest(
            question="   ",
            correct_answer="4",
        )

        errors = request.validate()
        assert len(errors) == 1

    def test_empty_answer(self) -> None:
        """Empty answer should be invalid."""
        request = ExplanationRequest(
            question="What is 2+2?",
            correct_answer="",
        )

        errors = request.validate()
        assert len(errors) == 1
        assert "answer" in errors[0].lower()

    def test_multiple_errors(self) -> None:
        """Multiple validation errors should all be reported."""
        request = ExplanationRequest(
            question="",
            correct_answer="",
        )

        errors = request.validate()
        assert len(errors) == 2

    def test_optional_fields(self) -> None:
        """Optional fields should not affect validation."""
        request = ExplanationRequest(
            question="Q",
            correct_answer="A",
            user_answer="B",
            context="Some context",
        )

        errors = request.validate()
        assert len(errors) == 0


class TestExplanationGeneratorInit:
    """Test ExplanationGenerator initialization."""

    def test_init_no_args(self) -> None:
        """Should initialize without arguments."""
        generator = ExplanationGenerator()

        assert generator._config is None
        assert generator._llm_client is None

    def test_init_with_config(self) -> None:
        """Should accept config."""
        mock_config = Mock()
        generator = ExplanationGenerator(config=mock_config)

        assert generator._config is mock_config

    def test_init_with_llm_client(self) -> None:
        """Should accept pre-created LLM client."""
        mock_client = Mock()
        generator = ExplanationGenerator(llm_client=mock_client)

        assert generator.llm_client is mock_client


class TestExplanationGeneratorAvailability:
    """Test availability checking."""

    def test_unavailable_no_config(self) -> None:
        """Should be unavailable without config."""
        generator = ExplanationGenerator()

        assert not generator.is_available

    def test_available_with_client(self) -> None:
        """Should be available with working client."""
        mock_client = Mock()
        mock_client.is_available.return_value = True

        generator = ExplanationGenerator(llm_client=mock_client)

        assert generator.is_available

    def test_unavailable_client_not_ready(self) -> None:
        """Should be unavailable when client not ready."""
        mock_client = Mock()
        mock_client.is_available.return_value = False

        generator = ExplanationGenerator(llm_client=mock_client)

        assert not generator.is_available

    def test_availability_cached(self) -> None:
        """Availability should be cached."""
        mock_client = Mock()
        mock_client.is_available.return_value = True

        generator = ExplanationGenerator(llm_client=mock_client)

        # Call twice
        _ = generator.is_available
        _ = generator.is_available

        # is_available should only be called once
        assert mock_client.is_available.call_count == 1


class TestExplanationGeneratorExplain:
    """Test explanation generation."""

    def test_explain_with_llm(self) -> None:
        """Should generate explanation with LLM."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "This is the explanation."

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_answer(
            question="What is the capital of France?",
            correct_answer="Paris",
        )

        assert explanation.text == "This is the explanation."
        assert explanation.source == "llm"
        assert not explanation.is_fallback

    def test_explain_wrong_answer(self) -> None:
        """Should include wrong answer context in prompt."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Lyon is not the capital..."

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_answer(
            question="What is the capital of France?",
            correct_answer="Paris",
            user_answer="Lyon",
        )

        assert explanation.text == "Lyon is not the capital..."
        assert explanation.was_wrong
        assert explanation.user_answer == "Lyon"

        # Check that prompt included user's wrong answer
        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "Lyon" in prompt

    def test_explain_with_context(self) -> None:
        """Should include source context in prompt."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Based on the source..."

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_answer(
            question="What is X?",
            correct_answer="Y",
            context="From the textbook: X is defined as Y",
        )

        # Check that context was included
        call_args = mock_client.generate.call_args
        prompt = call_args[0][0]
        assert "From the textbook" in prompt

    def test_explain_fallback_no_llm(self) -> None:
        """Should return fallback when LLM unavailable."""
        generator = ExplanationGenerator()
        explanation = generator.explain_answer(
            question="What is 2+2?",
            correct_answer="4",
        )

        assert explanation.is_fallback
        assert "4" in explanation.text

    def test_explain_fallback_wrong_answer(self) -> None:
        """Fallback should include both answers for wrong answer."""
        generator = ExplanationGenerator()
        explanation = generator.explain_answer(
            question="What is 2+2?",
            correct_answer="4",
            user_answer="5",
        )

        assert explanation.is_fallback
        assert "4" in explanation.text
        assert "5" in explanation.text

    def test_explain_invalid_request(self) -> None:
        """Should return fallback for invalid request."""
        mock_client = Mock()
        mock_client.is_available.return_value = True

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_answer(
            question="",  # Invalid
            correct_answer="A",
        )

        assert explanation.is_fallback
        mock_client.generate.assert_not_called()

    def test_explain_llm_error(self) -> None:
        """Should return fallback on LLM error."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = RuntimeError("API Error")

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_answer(
            question="What is X?",
            correct_answer="Y",
        )

        assert explanation.is_fallback

    def test_generation_config_tokens(self) -> None:
        """Should use appropriate max tokens."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Explanation"

        generator = ExplanationGenerator(llm_client=mock_client)
        generator.explain_answer(
            question="Q",
            correct_answer="A",
        )

        # Check generation config
        call_args = mock_client.generate.call_args
        gen_config = call_args[0][1]
        assert gen_config.max_tokens == MAX_EXPLANATION_TOKENS

    def test_generation_config_temperature(self) -> None:
        """Should use low temperature for factual explanations."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Explanation"

        generator = ExplanationGenerator(llm_client=mock_client)
        generator.explain_answer(
            question="Q",
            correct_answer="A",
        )

        call_args = mock_client.generate.call_args
        gen_config = call_args[0][1]
        assert gen_config.temperature <= 0.5


class TestExplanationGeneratorConcept:
    """Test concept explanation."""

    def test_explain_concept_with_llm(self) -> None:
        """Should explain concept using LLM."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "This tests knowledge of geography."

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_concept(
            question="What is the capital of France?",
            answer="Paris",
        )

        assert explanation.text == "This tests knowledge of geography."
        assert not explanation.is_fallback

    def test_explain_concept_fallback(self) -> None:
        """Should return fallback when LLM unavailable."""
        generator = ExplanationGenerator()
        explanation = generator.explain_concept(
            question="What is the capital of France?",
            answer="Paris",
        )

        assert explanation.is_fallback
        assert "Paris" in explanation.text

    def test_explain_concept_error(self) -> None:
        """Should return fallback on error."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.side_effect = RuntimeError("Error")

        generator = ExplanationGenerator(llm_client=mock_client)
        explanation = generator.explain_concept(
            question="Q",
            answer="A",
        )

        assert explanation.is_fallback


class TestExplainAnswerFunction:
    """Test convenience function."""

    def test_explain_answer_function(self) -> None:
        """Should work as standalone function."""
        # Without config, should return fallback
        text = explain_answer(
            question="What is 2+2?",
            correct_answer="4",
        )

        assert isinstance(text, str)
        assert "4" in text

    def test_explain_answer_with_wrong_answer(self) -> None:
        """Should include wrong answer in text."""
        text = explain_answer(
            question="What is 2+2?",
            correct_answer="4",
            user_answer="5",
        )

        assert "4" in text
        assert "5" in text


class TestPromptBuilding:
    """Test prompt construction."""

    def test_prompt_includes_question(self) -> None:
        """Prompt should include the question."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Explanation"

        generator = ExplanationGenerator(llm_client=mock_client)
        generator.explain_answer(
            question="What is photosynthesis?",
            correct_answer="Plant energy conversion",
        )

        prompt = mock_client.generate.call_args[0][0]
        assert "What is photosynthesis?" in prompt

    def test_prompt_includes_correct_answer(self) -> None:
        """Prompt should include correct answer."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Explanation"

        generator = ExplanationGenerator(llm_client=mock_client)
        generator.explain_answer(
            question="Q",
            correct_answer="Plant energy conversion",
        )

        prompt = mock_client.generate.call_args[0][0]
        assert "Plant energy conversion" in prompt

    def test_prompt_asks_for_brief(self) -> None:
        """Prompt should request brief explanation."""
        mock_client = Mock()
        mock_client.is_available.return_value = True
        mock_client.generate.return_value = "Explanation"

        generator = ExplanationGenerator(llm_client=mock_client)
        generator.explain_answer(
            question="Q",
            correct_answer="A",
        )

        prompt = mock_client.generate.call_args[0][0]
        # Should ask for brief/concise response
        assert "brief" in prompt.lower() or "concise" in prompt.lower()
