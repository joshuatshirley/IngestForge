"""
Unit tests for the Few-Shot Prompt Tuner.

Tests for IFPromptTuner middleware.

Follows NASA JPL Power of Ten rules.
"""

import pytest
from typing import Dict, List, Any, Tuple

from ingestforge.core.pipeline.prompt_tuner import (
    IFPromptTuner,
    FewShotExample,
    _estimate_tokens,
    _get_context_limit,
    _sanitize_text,
    _truncate_text,
    MAX_EXAMPLES,
    MAX_EXAMPLE_TEXT_LENGTH,
    CHARS_PER_TOKEN,
    MODEL_CONTEXT_LIMITS,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_estimate_tokens_basic(self) -> None:
        """Should estimate tokens from text length."""
        text = "a" * 100
        tokens = _estimate_tokens(text)
        assert tokens == 100 // CHARS_PER_TOKEN

    def test_estimate_tokens_empty(self) -> None:
        """Should handle empty text."""
        tokens = _estimate_tokens("")
        assert tokens == 0

    def test_get_context_limit_known_model(self) -> None:
        """Should return known limit for supported models."""
        limit = _get_context_limit("gpt-4o-mini")
        assert limit == 128000

    def test_get_context_limit_partial_match(self) -> None:
        """Should match partial model names."""
        limit = _get_context_limit("gpt-4o-mini-2024")
        assert limit == 128000

    def test_get_context_limit_unknown_model(self) -> None:
        """Should return default for unknown models."""
        limit = _get_context_limit("unknown-model")
        assert limit == MODEL_CONTEXT_LIMITS["_default"]

    def test_sanitize_text_email(self) -> None:
        """Should redact email addresses."""
        text = "Contact john@example.com for info"
        result = _sanitize_text(text)
        assert "[EMAIL]" in result
        assert "john@example.com" not in result

    def test_sanitize_text_phone(self) -> None:
        """Should redact phone numbers."""
        text = "Call 555-123-4567 for support"
        result = _sanitize_text(text)
        assert "[PHONE]" in result
        assert "555-123-4567" not in result

    def test_sanitize_text_ssn(self) -> None:
        """Should redact SSN patterns."""
        text = "SSN: 123-45-6789"
        result = _sanitize_text(text)
        assert "[SSN]" in result
        assert "123-45-6789" not in result

    def test_sanitize_text_credit_card(self) -> None:
        """Should redact credit card patterns."""
        text = "Card: 4111-1111-1111-1111"
        result = _sanitize_text(text)
        assert "[CREDIT_CARD]" in result
        assert "4111-1111-1111-1111" not in result

    def test_sanitize_text_preserves_other(self) -> None:
        """Should preserve non-PII text."""
        text = "The quick brown fox jumps"
        result = _sanitize_text(text)
        assert result == text

    def test_truncate_text_short(self) -> None:
        """Should not truncate short text."""
        text = "Hello"
        result = _truncate_text(text, 100)
        assert result == text

    def test_truncate_text_long(self) -> None:
        """Should truncate long text with ellipsis."""
        text = "a" * 100
        result = _truncate_text(text, 50)
        assert len(result) == 50
        assert result.endswith("...")


class TestFewShotExample:
    """Tests for FewShotExample dataclass."""

    def test_token_estimate(self) -> None:
        """Should estimate total tokens."""
        example = FewShotExample(
            chunk_text="a" * 100,
            entities_json="b" * 100,
        )
        tokens = example.token_estimate()
        assert tokens == 200 // CHARS_PER_TOKEN

    def test_sanitized_flag(self) -> None:
        """Should track sanitization status."""
        example = FewShotExample(
            chunk_text="text",
            entities_json="{}",
            sanitized=True,
        )
        assert example.sanitized is True


class TestIFPromptTuner:
    """Tests for IFPromptTuner middleware."""

    @pytest.fixture
    def tuner(self) -> IFPromptTuner:
        """Create a basic tuner instance."""
        return IFPromptTuner(model="gpt-4o-mini", cloud_provider=True)

    @pytest.fixture
    def sample_examples(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Sample examples for testing."""
        return [
            (
                "John Smith works at ACME Corp.",
                {"entities": [{"type": "PERSON", "text": "John Smith"}]},
            ),
            (
                "The company is located in New York.",
                {"entities": [{"type": "LOC", "text": "New York"}]},
            ),
            (
                "Contact support@example.com for help.",
                {"entities": [{"type": "EMAIL", "text": "support@example.com"}]},
            ),
        ]

    def test_init_default(self) -> None:
        """Should initialize with defaults."""
        tuner = IFPromptTuner()
        assert tuner.model == "gpt-4o-mini"
        assert tuner.cloud_provider is True
        assert tuner.max_examples == 3

    def test_init_custom(self) -> None:
        """Should accept custom parameters."""
        tuner = IFPromptTuner(
            model="gpt-4o",
            cloud_provider=False,
            max_examples=5,
        )
        assert tuner.model == "gpt-4o"
        assert tuner.cloud_provider is False
        assert tuner.max_examples == 5

    def test_init_max_examples_limit(self) -> None:
        """Should enforce max_examples upper bound."""
        with pytest.raises(AssertionError):
            IFPromptTuner(max_examples=MAX_EXAMPLES + 1)

    def test_prepare_examples_basic(
        self,
        tuner: IFPromptTuner,
        sample_examples: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Should prepare examples from raw input."""
        prepared = tuner.prepare_examples(sample_examples)
        assert len(prepared) == 3
        assert all(isinstance(e, FewShotExample) for e in prepared)

    def test_prepare_examples_sanitizes(
        self,
        sample_examples: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Should sanitize PII when cloud_provider=True."""
        tuner = IFPromptTuner(cloud_provider=True)
        prepared = tuner.prepare_examples(sample_examples)

        # The email example should be sanitized
        email_example = prepared[2]
        assert "[EMAIL]" in email_example.chunk_text
        assert email_example.sanitized is True

    def test_prepare_examples_no_sanitize(
        self,
        sample_examples: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Should not sanitize when cloud_provider=False."""
        tuner = IFPromptTuner(cloud_provider=False)
        prepared = tuner.prepare_examples(sample_examples)

        # The email should be preserved
        email_example = prepared[2]
        assert "support@example.com" in email_example.chunk_text
        assert email_example.sanitized is False

    def test_build_few_shot_block_empty(self, tuner: IFPromptTuner) -> None:
        """Should return empty string for no examples."""
        block = tuner.build_few_shot_block([])
        assert block == ""

    def test_build_few_shot_block_structure(
        self,
        tuner: IFPromptTuner,
    ) -> None:
        """Should build structured markdown block."""
        examples = [
            FewShotExample(
                chunk_text="Test text",
                entities_json='{"entities": []}',
            )
        ]
        block = tuner.build_few_shot_block(examples)

        assert "## Few-Shot Examples" in block
        assert "### Example 1" in block
        assert "**Source Text:**" in block
        assert "Test text" in block
        assert "**Extracted Entities:**" in block

    def test_inject_examples_basic(
        self,
        tuner: IFPromptTuner,
        sample_examples: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Should inject examples into base prompt."""
        base_prompt = "You are an entity extraction system."
        enhanced = tuner.inject_examples(base_prompt, sample_examples)

        assert base_prompt in enhanced
        assert "## Few-Shot Examples" in enhanced
        assert "### Example 1" in enhanced

    def test_inject_examples_empty(self, tuner: IFPromptTuner) -> None:
        """Should return base prompt when no examples."""
        base_prompt = "You are an entity extraction system."
        enhanced = tuner.inject_examples(base_prompt, [])

        assert enhanced == base_prompt

    def test_inject_examples_respects_limit(
        self,
        sample_examples: List[Tuple[str, Dict[str, Any]]],
    ) -> None:
        """Should respect max_examples limit."""
        tuner = IFPromptTuner(max_examples=2)
        base_prompt = "Extract entities."
        enhanced = tuner.inject_examples(base_prompt, sample_examples)

        # Should only have 2 examples
        assert enhanced.count("### Example") == 2

    def test_inject_examples_context_limit(self) -> None:
        """Should enforce context limit."""
        # Create tuner with very small limit
        tuner = IFPromptTuner(max_prompt_tokens=500)

        # Create large base prompt
        base_prompt = "x" * 2000  # Exceeds limit

        # Should return base prompt (no room for examples)
        result = tuner.inject_examples(base_prompt, [("text", {"entities": []})])
        assert result == base_prompt

    def test_get_context_info(self, tuner: IFPromptTuner) -> None:
        """Should return context information."""
        info = tuner.get_context_info()

        assert "model" in info
        assert "context_limit" in info
        assert "max_prompt_tokens" in info
        assert "cloud_provider" in info
        assert info["model"] == "gpt-4o-mini"


class TestJPLRuleCompliance:
    """Tests verifying JPL rule compliance."""

    def test_rule2_max_examples_constant(self) -> None:
        """JPL Rule #2: Should have fixed upper bound."""
        assert MAX_EXAMPLES == 5
        assert MAX_EXAMPLE_TEXT_LENGTH == 2000

    def test_rule5_assert_context_limits(self) -> None:
        """JPL Rule #5: Should assert context limits."""
        tuner = IFPromptTuner(max_prompt_tokens=100)

        # Base prompt that's already near limit
        base_prompt = "x" * 350  # ~87 tokens

        # Large example that would exceed limit
        big_example = ("y" * 500, {"entities": []})

        # Should not raise (examples should be excluded)
        result = tuner.inject_examples(base_prompt, [big_example])
        assert result == base_prompt  # Examples excluded due to budget

    def test_rule5_empty_examples_assertion(self) -> None:
        """JPL Rule #5: Should assert on None input."""
        tuner = IFPromptTuner()
        with pytest.raises(AssertionError):
            tuner.prepare_examples(None)

    def test_rule5_null_base_prompt(self) -> None:
        """JPL Rule #5: Should assert on None base prompt."""
        tuner = IFPromptTuner()
        with pytest.raises(AssertionError):
            tuner.inject_examples(None, [])


class TestPrivacySanitization:
    """Tests for privacy/PII sanitization (AC)."""

    def test_sanitize_multiple_pii(self) -> None:
        """Should sanitize multiple PII types in one text."""
        text = "Email john@test.com, call 555-123-4567, SSN 123-45-6789"
        result = _sanitize_text(text)

        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[SSN]" in result
        assert "john@test.com" not in result
        assert "555-123-4567" not in result
        assert "123-45-6789" not in result

    def test_cloud_provider_sanitization(self) -> None:
        """Cloud provider mode should sanitize all examples."""
        tuner = IFPromptTuner(cloud_provider=True)
        examples = [
            ("Contact john@example.com", {"entities": []}),
        ]

        prepared = tuner.prepare_examples(examples)
        assert all(e.sanitized for e in prepared)

    def test_local_mode_no_sanitization(self) -> None:
        """Local mode should not sanitize."""
        tuner = IFPromptTuner(cloud_provider=False)
        examples = [
            ("Contact john@example.com", {"entities": []}),
        ]

        prepared = tuner.prepare_examples(examples)
        assert all(not e.sanitized for e in prepared)
        assert "john@example.com" in prepared[0].chunk_text


class TestIntegration:
    """Integration tests with realistic examples."""

    def test_full_workflow(self) -> None:
        """Test complete prompt tuning workflow."""
        # Create tuner
        tuner = IFPromptTuner(model="gpt-4o-mini", cloud_provider=True, max_examples=3)

        # Simulate examples from IFExampleRegistry.find_similar()
        raw_examples = [
            (
                "Dr. Jane Smith presented at the conference.",
                {
                    "entities": [
                        {"type": "PERSON", "text": "Dr. Jane Smith", "confidence": 0.95}
                    ]
                },
            ),
            (
                "Microsoft announced a new product on January 15, 2024.",
                {
                    "entities": [
                        {"type": "ORG", "text": "Microsoft", "confidence": 0.92},
                        {
                            "type": "DATE",
                            "text": "January 15, 2024",
                            "confidence": 0.88,
                        },
                    ]
                },
            ),
        ]

        # Base extraction prompt
        base_prompt = """You are an expert entity extraction system.
Extract all named entities from the provided text.
For each entity, identify the type, text, and confidence."""

        # Inject examples
        enhanced = tuner.inject_examples(base_prompt, raw_examples)

        # Verify structure
        assert base_prompt in enhanced
        assert "## Few-Shot Examples" in enhanced
        assert "Dr. Jane Smith" in enhanced
        assert "Microsoft" in enhanced

    def test_token_budget_management(self) -> None:
        """Test that token budget is respected."""
        tuner = IFPromptTuner(max_prompt_tokens=2000)

        base_prompt = "Extract entities from text."  # Small prompt

        # Create examples of varying sizes
        examples = [
            ("Short text 1", {"entities": []}),
            ("Short text 2", {"entities": []}),
            ("x" * 3000, {"entities": []}),  # Large - might not fit
        ]

        # Use smaller reserved tokens so there's room for examples
        enhanced = tuner.inject_examples(base_prompt, examples, reserved_tokens=100)

        # Should include at least some examples
        assert "### Example 1" in enhanced

        # Total should be within budget (minus reserved)
        total_tokens = _estimate_tokens(enhanced) + 100
        assert total_tokens <= 2000
