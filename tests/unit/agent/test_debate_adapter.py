"""Tests for debate adapter module.

Tests LLM adapter functions for fact-checker debate orchestration."""

from __future__ import annotations

import pytest

from ingestforge.agent.debate_adapter import (
    create_proponent_function,
    create_critic_function,
    _parse_debate_response,
    _extract_position,
    _extract_evidence,
    _extract_confidence,
    _make_fallback_argument,
)
from ingestforge.agent.fact_checker import (
    DebateArgument,
    DebateRole,
)
from tests.fixtures.agents import MockLLM


class TestProponentFunction:
    """Test proponent debate function creation."""

    def test_create_proponent_function_success(self) -> None:
        """Test creating proponent function with valid LLM."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Position: The claim is well-supported.\n"
                "Evidence: Multiple studies confirm this.\n"
                "Confidence: 0.85"
            ]
        )

        proponent_fn = create_proponent_function(llm)
        arg = proponent_fn("Defend: test claim", [])

        assert arg.role == DebateRole.PROPONENT
        assert "well-supported" in arg.position
        assert len(arg.evidence) > 0
        assert arg.evidence[0].supports_claim is True
        assert arg.evidence[0].confidence == 0.85

    def test_create_proponent_function_none_client(self) -> None:
        """Test creating proponent with None client raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_proponent_function(None)

    def test_proponent_handles_llm_error(self) -> None:
        """Test proponent function handles LLM errors gracefully."""
        llm = MockLLM()
        llm.set_availability(False)
        llm.set_responses(["This won't be used"])

        proponent_fn = create_proponent_function(llm)

        # Mock generate to raise exception
        def raise_error(prompt, config=None):
            raise Exception("LLM service unavailable")

        llm.generate = raise_error

        arg = proponent_fn("Defend: test", [])

        assert arg.role == DebateRole.PROPONENT
        assert "error" in arg.position.lower()
        assert len(arg.evidence) > 0
        assert arg.evidence[0].confidence == 0.0

    def test_proponent_includes_history_context(self) -> None:
        """Test proponent function includes debate history."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Position: Addressing the critique.\n"
                "Evidence: New evidence supports the claim.\n"
                "Confidence: 0.75"
            ]
        )

        proponent_fn = create_proponent_function(llm)

        # Create history
        history = [
            DebateArgument(
                role=DebateRole.CRITIC,
                position="This is weak evidence.",
                evidence=[],
            )
        ]

        arg = proponent_fn("Defend: test", history)

        # Check that LLM was called with history context
        last_prompt = llm.get_last_prompt()
        assert "Previous arguments" in last_prompt or "critique" in last_prompt.lower()


class TestCriticFunction:
    """Test critic debate function creation."""

    def test_create_critic_function_success(self) -> None:
        """Test creating critic function with valid LLM."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Position: The claim lacks evidence.\n"
                "Evidence: Key studies show contrary results.\n"
                "Confidence: 0.70"
            ]
        )

        critic_fn = create_critic_function(llm)
        arg = critic_fn("Critique: test claim", [])

        assert arg.role == DebateRole.CRITIC
        assert "lacks evidence" in arg.position
        assert len(arg.evidence) > 0
        assert arg.evidence[0].supports_claim is False
        assert arg.evidence[0].confidence == 0.70

    def test_create_critic_function_none_client(self) -> None:
        """Test creating critic with None client raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            create_critic_function(None)

    def test_critic_handles_llm_error(self) -> None:
        """Test critic function handles LLM errors gracefully."""
        llm = MockLLM()
        llm.set_responses(["Not used"])

        critic_fn = create_critic_function(llm)

        # Mock generate to raise exception
        def raise_error(prompt, config=None):
            raise Exception("Connection timeout")

        llm.generate = raise_error

        arg = critic_fn("Critique: test", [])

        assert arg.role == DebateRole.CRITIC
        assert "error" in arg.position.lower()
        assert arg.evidence[0].confidence == 0.0


class TestResponseParsing:
    """Test parsing of LLM debate responses."""

    def test_parse_complete_response(self) -> None:
        """Test parsing complete well-formatted response."""
        response = """Position: This is my position on the claim.
Evidence: Here is supporting evidence and reasoning.
Confidence: 0.80"""

        arg = _parse_debate_response(response, DebateRole.PROPONENT)

        assert arg.role == DebateRole.PROPONENT
        assert arg.position == "This is my position on the claim."
        assert len(arg.evidence) == 1
        assert "supporting evidence" in arg.evidence[0].content
        assert arg.evidence[0].confidence == 0.80

    def test_parse_missing_position(self) -> None:
        """Test parsing response with missing position."""
        response = """Evidence: Some evidence.
Confidence: 0.5"""

        arg = _parse_debate_response(response, DebateRole.CRITIC)

        assert "Unable to parse" in arg.position
        assert len(arg.evidence) == 1

    def test_parse_missing_evidence(self) -> None:
        """Test parsing response with missing evidence."""
        response = """Position: Valid position.
Confidence: 0.6"""

        arg = _parse_debate_response(response, DebateRole.PROPONENT)

        assert arg.position == "Valid position."
        # No evidence should be added if missing
        assert len(arg.evidence) == 0

    def test_parse_missing_confidence(self) -> None:
        """Test parsing response with missing confidence uses default."""
        response = """Position: Test position.
Evidence: Test evidence.
Confidence: """

        arg = _parse_debate_response(response, DebateRole.CRITIC)

        assert len(arg.evidence) == 1
        # Should use default confidence (0.6)
        assert arg.evidence[0].confidence == 0.6

    def test_parse_empty_response(self) -> None:
        """Test parsing empty response returns fallback."""
        arg = _parse_debate_response("", DebateRole.PROPONENT)

        assert "No response" in arg.position
        assert len(arg.evidence) > 0
        assert arg.evidence[0].confidence == 0.0

    def test_parse_multiline_position(self) -> None:
        """Test parsing position with multiple lines."""
        response = """Position: This is a long position
that spans multiple lines
and contains detailed reasoning.
Evidence: Supporting evidence here.
Confidence: 0.75"""

        arg = _parse_debate_response(response, DebateRole.PROPONENT)

        assert "long position" in arg.position
        assert "multiple lines" in arg.position
        assert "detailed reasoning" in arg.position


class TestExtraction:
    """Test individual extraction functions."""

    def test_extract_position_success(self) -> None:
        """Test extracting position from text."""
        text = "Position: This is the position.\nEvidence: Some evidence."
        position = _extract_position(text)
        assert position == "This is the position."

    def test_extract_position_case_insensitive(self) -> None:
        """Test position extraction is case insensitive."""
        text = "POSITION: Upper case position\nEvidence: test"
        position = _extract_position(text)
        assert position == "Upper case position"

    def test_extract_position_missing(self) -> None:
        """Test extracting missing position returns default."""
        text = "Evidence: No position here"
        position = _extract_position(text)
        assert "Unable to parse" in position

    def test_extract_evidence_success(self) -> None:
        """Test extracting evidence from text."""
        text = "Position: Test\nEvidence: This is the evidence.\nConfidence: 0.5"
        evidence = _extract_evidence(text)
        assert evidence == "This is the evidence."

    def test_extract_evidence_missing(self) -> None:
        """Test extracting missing evidence returns empty string."""
        text = "Position: Test position only"
        evidence = _extract_evidence(text)
        assert evidence == ""

    def test_extract_confidence_decimal(self) -> None:
        """Test extracting decimal confidence value."""
        text = "Confidence: 0.85"
        confidence = _extract_confidence(text)
        assert confidence == 0.85

    def test_extract_confidence_integer(self) -> None:
        """Test extracting integer confidence values."""
        assert _extract_confidence("Confidence: 1") == 1.0
        assert _extract_confidence("Confidence: 0") == 0.0

    def test_extract_confidence_clamping(self) -> None:
        """Test confidence values are clamped to valid range."""
        # Values should be clamped to 0.0-1.0
        # Note: Negative values don't match the regex, so default is returned
        assert _extract_confidence("Confidence: 1.5") == 1.0
        assert _extract_confidence("Confidence: -0.2") == 0.6  # Returns default

    def test_extract_confidence_missing(self) -> None:
        """Test extracting missing confidence returns default."""
        text = "Position: Test\nEvidence: Test"
        confidence = _extract_confidence(text)
        assert confidence == 0.6  # Default value

    def test_extract_confidence_invalid_format(self) -> None:
        """Test invalid confidence format returns default."""
        text = "Confidence: invalid"
        confidence = _extract_confidence(text)
        assert confidence == 0.6


class TestFallbackArgument:
    """Test fallback argument creation."""

    def test_make_fallback_proponent(self) -> None:
        """Test creating fallback argument for proponent."""
        arg = _make_fallback_argument(DebateRole.PROPONENT, "Test error")

        assert arg.role == DebateRole.PROPONENT
        assert arg.position == "Test error"
        assert len(arg.evidence) == 1
        assert arg.evidence[0].content == "Test error"
        assert arg.evidence[0].source == "error_fallback"
        assert arg.evidence[0].confidence == 0.0

    def test_make_fallback_critic(self) -> None:
        """Test creating fallback argument for critic."""
        arg = _make_fallback_argument(DebateRole.CRITIC, "Connection failed")

        assert arg.role == DebateRole.CRITIC
        assert arg.position == "Connection failed"
        assert arg.evidence[0].supports_claim is False


class TestIntegration:
    """Integration tests for debate adapter with orchestrator."""

    def test_full_debate_cycle(self) -> None:
        """Test full debate cycle with mock LLM."""
        llm = MockLLM()
        llm.set_responses(
            [
                # Round 1: Proponent
                "Position: The claim is supported by evidence.\n"
                "Evidence: Research shows positive results.\n"
                "Confidence: 0.80",
                # Round 1: Critic
                "Position: The evidence is insufficient.\n"
                "Evidence: Sample sizes are too small.\n"
                "Confidence: 0.75",
                # Round 2: Proponent
                "Position: Additional studies confirm the findings.\n"
                "Evidence: Meta-analysis supports the claim.\n"
                "Confidence: 0.85",
                # Round 2: Critic
                "Position: Publication bias may skew results.\n"
                "Evidence: Negative studies are underreported.\n"
                "Confidence: 0.70",
            ]
        )

        proponent_fn = create_proponent_function(llm)
        critic_fn = create_critic_function(llm)

        # Simulate debate rounds
        history = []

        # Round 1
        prop_arg = proponent_fn("Defend: test claim", history)
        history.append(prop_arg)
        assert prop_arg.role == DebateRole.PROPONENT

        critic_arg = critic_fn("Critique: test claim", history)
        history.append(critic_arg)
        assert critic_arg.role == DebateRole.CRITIC

        # Round 2
        prop_arg2 = proponent_fn("Defend: test claim", history)
        history.append(prop_arg2)

        critic_arg2 = critic_fn("Critique: test claim", history)
        history.append(critic_arg2)

        # Verify all arguments were created
        assert len(history) == 4
        assert all(isinstance(arg, DebateArgument) for arg in history)

        # Verify roles alternate
        assert history[0].role == DebateRole.PROPONENT
        assert history[1].role == DebateRole.CRITIC
        assert history[2].role == DebateRole.PROPONENT
        assert history[3].role == DebateRole.CRITIC
