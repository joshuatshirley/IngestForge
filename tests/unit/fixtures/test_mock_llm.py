"""Tests for MockLLM fixture.

Comprehensive tests for the mock LLM implementation used in agent testing."""

from __future__ import annotations


import pytest

from ingestforge.llm.base import GenerationConfig
from tests.fixtures.agents import MockLLM

# Basic functionality tests


class TestMockLLMBasics:
    """Tests for basic MockLLM functionality."""

    def test_initialization(self) -> None:
        """Test MockLLM initializes with correct defaults."""
        llm = MockLLM()

        assert llm.call_count == 0
        assert llm.call_history == []
        assert llm.response_history == []
        assert llm.model_name == "mock-llm"
        assert llm.is_available() is True

    def test_model_name_property(self) -> None:
        """Test model_name property."""
        llm = MockLLM()

        assert llm.model_name == "mock-llm"

    def test_set_model_name(self) -> None:
        """Test setting custom model name."""
        llm = MockLLM()
        llm.set_model_name("test-model-v2")

        assert llm.model_name == "test-model-v2"

    def test_set_model_name_validates_type(self) -> None:
        """Test set_model_name validates string type."""
        llm = MockLLM()

        with pytest.raises(ValueError, match="Model name must be a string"):
            llm.set_model_name(123)  # type: ignore

    def test_availability_default(self) -> None:
        """Test is_available returns True by default."""
        llm = MockLLM()

        assert llm.is_available() is True

    def test_set_availability(self) -> None:
        """Test setting availability status."""
        llm = MockLLM()

        llm.set_availability(False)
        assert llm.is_available() is False

        llm.set_availability(True)
        assert llm.is_available() is True


# Response configuration tests


class TestSetResponses:
    """Tests for set_responses() method."""

    def test_set_responses_basic(self) -> None:
        """Test setting basic response sequence."""
        llm = MockLLM()
        responses = ["response 1", "response 2", "response 3"]

        llm.set_responses(responses)

        # Verify configuration without modifying state
        assert llm.get_remaining_count() == 3
        assert llm.has_responses_remaining() is True

    def test_set_responses_validates_empty(self) -> None:
        """Test set_responses rejects empty list."""
        llm = MockLLM()

        with pytest.raises(ValueError, match="Response sequence cannot be empty"):
            llm.set_responses([])

    def test_set_responses_validates_list_type(self) -> None:
        """Test set_responses validates list type."""
        llm = MockLLM()

        with pytest.raises(ValueError, match="Sequence must be a list"):
            llm.set_responses("not a list")  # type: ignore

    def test_set_responses_validates_string_elements(self) -> None:
        """Test set_responses validates all elements are strings."""
        llm = MockLLM()

        with pytest.raises(ValueError, match="Response at index 1 must be a string"):
            llm.set_responses(["valid", 123, "valid"])  # type: ignore

    def test_set_responses_makes_copy(self) -> None:
        """Test set_responses makes a copy of the list."""
        llm = MockLLM()
        original = ["response 1"]

        llm.set_responses(original)
        original.append("response 2")

        # Only one response should be configured
        assert llm.get_remaining_count() == 1

    def test_set_responses_resets_index(self) -> None:
        """Test set_responses resets call index."""
        llm = MockLLM()

        llm.set_responses(["first", "second"])
        llm.generate("prompt 1")  # Advance index

        llm.set_responses(["new first", "new second"])

        # Index should be reset
        assert llm.generate("prompt 2") == "new first"


# Generate method tests


class TestGenerate:
    """Tests for generate() method."""

    def test_generate_returns_responses_in_order(self) -> None:
        """Test generate returns responses in configured order."""
        llm = MockLLM()
        llm.set_responses(["first", "second", "third"])

        assert llm.generate("prompt 1") == "first"
        assert llm.generate("prompt 2") == "second"
        assert llm.generate("prompt 3") == "third"

    def test_generate_validates_prompt_type(self) -> None:
        """Test generate validates prompt is a string."""
        llm = MockLLM()
        llm.set_responses(["response"])

        with pytest.raises(ValueError, match="Prompt must be a string"):
            llm.generate(123)  # type: ignore

    def test_generate_raises_on_exhausted_responses(self) -> None:
        """Test generate raises AssertionError when responses exhausted (Rule #5)."""
        llm = MockLLM()
        llm.set_responses(["only response"])

        llm.generate("first call")

        # Second call should raise AssertionError
        with pytest.raises(AssertionError, match="MockLLM called 2 times"):
            llm.generate("second call")

    def test_generate_with_config_parameter(self) -> None:
        """Test generate accepts GenerationConfig (for interface compatibility)."""
        llm = MockLLM()
        llm.set_responses(["response"])
        config = GenerationConfig(temperature=0.7, max_tokens=100)

        # Should not raise, config is ignored
        result = llm.generate("prompt", config=config)

        assert result == "response"

    def test_generate_with_kwargs(self) -> None:
        """Test generate accepts extra kwargs (for interface compatibility)."""
        llm = MockLLM()
        llm.set_responses(["response"])

        # Should not raise, kwargs are ignored
        result = llm.generate("prompt", extra_param="value", another=123)

        assert result == "response"


# Generate with context tests


class TestGenerateWithContext:
    """Tests for generate_with_context() method."""

    def test_generate_with_context_returns_response(self) -> None:
        """Test generate_with_context returns next response."""
        llm = MockLLM()
        llm.set_responses(["context response"])

        result = llm.generate_with_context(
            system_prompt="You are helpful",
            user_prompt="What is 2+2?",
        )

        assert result == "context response"

    def test_generate_with_context_with_context_param(self) -> None:
        """Test generate_with_context with context parameter."""
        llm = MockLLM()
        llm.set_responses(["response"])

        result = llm.generate_with_context(
            system_prompt="System",
            user_prompt="User",
            context="Additional context",
        )

        assert result == "response"

    def test_generate_with_context_with_config(self) -> None:
        """Test generate_with_context with GenerationConfig."""
        llm = MockLLM()
        llm.set_responses(["response"])
        config = GenerationConfig(temperature=0.5)

        result = llm.generate_with_context(
            system_prompt="System",
            user_prompt="User",
            config=config,
        )

        assert result == "response"

    def test_generate_with_context_tracks_history(self) -> None:
        """Test generate_with_context adds to call history."""
        llm = MockLLM()
        llm.set_responses(["response"])

        llm.generate_with_context(
            system_prompt="Be helpful",
            user_prompt="Help me",
        )

        history = llm.call_history
        assert len(history) == 1
        assert "System: Be helpful" in history[0]
        assert "User: Help me" in history[0]


# Call tracking tests


class TestCallTracking:
    """Tests for call history tracking."""

    def test_call_count_increments(self) -> None:
        """Test call_count increments with each call."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2", "r3"])

        assert llm.call_count == 0

        llm.generate("p1")
        assert llm.call_count == 1

        llm.generate("p2")
        assert llm.call_count == 2

        llm.generate("p3")
        assert llm.call_count == 3

    def test_call_history_records_prompts(self) -> None:
        """Test call_history records all prompts."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        llm.generate("first prompt")
        llm.generate("second prompt")

        history = llm.call_history
        assert len(history) == 2
        assert history[0] == "first prompt"
        assert history[1] == "second prompt"

    def test_call_history_returns_copy(self) -> None:
        """Test call_history returns a copy (Rule #6 - immutability)."""
        llm = MockLLM()
        llm.set_responses(["response"])

        llm.generate("prompt")
        history = llm.call_history

        # Modify the returned list
        history.append("should not affect internal state")

        # Get fresh copy
        new_history = llm.call_history
        assert len(new_history) == 1

    def test_response_history_records_responses(self) -> None:
        """Test response_history records all responses."""
        llm = MockLLM()
        llm.set_responses(["first response", "second response"])

        llm.generate("p1")
        llm.generate("p2")

        history = llm.response_history
        assert len(history) == 2
        assert history[0] == "first response"
        assert history[1] == "second response"

    def test_response_history_returns_copy(self) -> None:
        """Test response_history returns a copy."""
        llm = MockLLM()
        llm.set_responses(["response"])

        llm.generate("prompt")
        history = llm.response_history

        # Modify the returned list
        history.append("should not affect internal state")

        # Get fresh copy
        new_history = llm.response_history
        assert len(new_history) == 1

    def test_get_last_prompt(self) -> None:
        """Test get_last_prompt returns most recent prompt."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2", "r3"])

        llm.generate("first")
        assert llm.get_last_prompt() == "first"

        llm.generate("second")
        assert llm.get_last_prompt() == "second"

        llm.generate("third")
        assert llm.get_last_prompt() == "third"

    def test_get_last_prompt_empty_when_no_calls(self) -> None:
        """Test get_last_prompt returns empty string when no calls made."""
        llm = MockLLM()

        assert llm.get_last_prompt() == ""


# Remaining responses tests


class TestRemainingResponses:
    """Tests for tracking remaining responses."""

    def test_get_remaining_count_initial(self) -> None:
        """Test get_remaining_count with fresh configuration."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2", "r3"])

        assert llm.get_remaining_count() == 3

    def test_get_remaining_count_decrements(self) -> None:
        """Test get_remaining_count decrements with use."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2", "r3"])

        llm.generate("p1")
        assert llm.get_remaining_count() == 2

        llm.generate("p2")
        assert llm.get_remaining_count() == 1

        llm.generate("p3")
        assert llm.get_remaining_count() == 0

    def test_has_responses_remaining_true(self) -> None:
        """Test has_responses_remaining when responses available."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        assert llm.has_responses_remaining() is True

        llm.generate("p1")
        assert llm.has_responses_remaining() is True

    def test_has_responses_remaining_false(self) -> None:
        """Test has_responses_remaining when exhausted."""
        llm = MockLLM()
        llm.set_responses(["r1"])

        llm.generate("p1")

        assert llm.has_responses_remaining() is False


# Reset functionality tests


class TestReset:
    """Tests for reset() and clear_responses() methods."""

    def test_reset_clears_history(self) -> None:
        """Test reset clears call and response history."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        llm.generate("p1")
        llm.reset()

        assert llm.call_count == 0
        assert llm.call_history == []
        assert llm.response_history == []

    def test_reset_preserves_responses(self) -> None:
        """Test reset preserves configured responses."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        llm.generate("p1")
        llm.reset()

        # Should still have responses available
        assert llm.get_remaining_count() == 2
        assert llm.generate("new prompt") == "r1"

    def test_reset_resets_index(self) -> None:
        """Test reset resets index to start of sequence."""
        llm = MockLLM()
        llm.set_responses(["first", "second"])

        llm.generate("p1")
        assert llm.generate("p2") == "second"

        llm.reset()

        # Should start from beginning
        assert llm.generate("p3") == "first"

    def test_clear_responses_removes_all(self) -> None:
        """Test clear_responses removes all responses."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        llm.generate("p1")
        llm.clear_responses()

        assert llm.get_remaining_count() == 0
        assert llm.has_responses_remaining() is False

    def test_clear_responses_resets_state(self) -> None:
        """Test clear_responses also resets history."""
        llm = MockLLM()
        llm.set_responses(["r1"])

        llm.generate("p1")
        llm.clear_responses()

        assert llm.call_count == 0
        assert llm.call_history == []
        assert llm.response_history == []


# Integration tests with agent-like usage


class TestAgentIntegration:
    """Tests simulating real agent usage patterns."""

    def test_multi_turn_conversation(self) -> None:
        """Test simulating multi-turn agent conversation."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Thought: I should search\nAction: search",
                "Thought: I found info\nAction: analyze",
                "Thought: Analysis complete\nFinal Answer: result",
            ]
        )

        # Agent makes multiple calls
        turn1 = llm.generate("What is the task?")
        turn2 = llm.generate("Continue with info")
        turn3 = llm.generate("Finalize")

        assert "search" in turn1
        assert "analyze" in turn2
        assert "result" in turn3

        # Verify all turns tracked
        assert llm.call_count == 3

    def test_agent_with_error_recovery(self) -> None:
        """Test agent pattern with error and recovery."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Try action A",
                "Error occurred, try B",
                "Success with B",
            ]
        )

        responses = []
        for i in range(3):
            responses.append(llm.generate(f"iteration {i}"))

        assert len(responses) == 3
        assert "Error occurred" in responses[1]

    def test_agent_exhausts_responses_correctly(self) -> None:
        """Test that agent properly detects when responses exhausted."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        # Agent can check before making call
        iterations = 0
        while llm.has_responses_remaining():
            llm.generate(f"iteration {iterations}")
            iterations += 1

        assert iterations == 2

    def test_react_style_prompting(self) -> None:
        """Test ReAct-style thought-action-observation loop."""
        llm = MockLLM()

        # ReAct agent responses
        llm.set_responses(
            [
                'Thought: Need to search\nAction: search\nInput: {"query": "test"}',
                "Thought: Found answer\nFinal Answer: The result is 42",
            ]
        )

        # First iteration
        response1 = llm.generate_with_context(
            system_prompt="You are a helpful assistant",
            user_prompt="Solve: What is the answer?",
        )
        assert "Action: search" in response1

        # Second iteration
        response2 = llm.generate_with_context(
            system_prompt="You are a helpful assistant",
            user_prompt="Continue with observation: Found data",
        )
        assert "Final Answer" in response2

        # Verify history
        assert llm.call_count == 2


# Edge case tests


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_response(self) -> None:
        """Test with single response in sequence."""
        llm = MockLLM()
        llm.set_responses(["only one"])

        assert llm.generate("prompt") == "only one"
        assert llm.get_remaining_count() == 0

    def test_empty_string_response(self) -> None:
        """Test with empty string as response."""
        llm = MockLLM()
        llm.set_responses([""])

        assert llm.generate("prompt") == ""

    def test_very_long_response(self) -> None:
        """Test with very long response."""
        llm = MockLLM()
        long_response = "x" * 10000
        llm.set_responses([long_response])

        result = llm.generate("prompt")

        assert len(result) == 10000

    def test_special_characters_in_response(self) -> None:
        """Test with special characters in responses."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Response with\nnewlines\nand\ttabs",
                "Unicode: 你好 🚀 Ω",
                "Quotes: \"double\" and 'single'",
            ]
        )

        r1 = llm.generate("p1")
        r2 = llm.generate("p2")
        r3 = llm.generate("p3")

        assert "\n" in r1
        assert "你好" in r2
        assert '"double"' in r3

    def test_multiple_resets(self) -> None:
        """Test multiple reset calls."""
        llm = MockLLM()
        llm.set_responses(["r1", "r2"])

        llm.generate("p1")
        llm.reset()
        llm.reset()  # Second reset
        llm.reset()  # Third reset

        # Should still work
        assert llm.generate("new") == "r1"

    def test_set_responses_multiple_times(self) -> None:
        """Test reconfiguring responses multiple times."""
        llm = MockLLM()

        llm.set_responses(["first set"])
        assert llm.generate("p1") == "first set"

        llm.set_responses(["second", "set"])
        assert llm.generate("p2") == "second"
        assert llm.generate("p3") == "set"

        llm.set_responses(["third"])
        assert llm.generate("p4") == "third"


# Error message quality tests


class TestErrorMessages:
    """Tests for error message quality."""

    def test_exhausted_error_message_helpful(self) -> None:
        """Test that exhaustion error message is helpful."""
        llm = MockLLM()
        llm.set_responses(["r1"])

        llm.generate("p1")

        with pytest.raises(AssertionError) as exc_info:
            llm.generate("p2")

        error_msg = str(exc_info.value)
        assert "called 2 times" in error_msg
        assert "1 responses configured" in error_msg
        assert "set_responses()" in error_msg

    def test_empty_sequence_error_message(self) -> None:
        """Test empty sequence error message is clear."""
        llm = MockLLM()

        with pytest.raises(ValueError) as exc_info:
            llm.set_responses([])

        assert "cannot be empty" in str(exc_info.value)

    def test_invalid_type_error_messages(self) -> None:
        """Test type validation error messages are clear."""
        llm = MockLLM()

        with pytest.raises(ValueError) as exc_info:
            llm.set_responses(["valid", 123])  # type: ignore

        error_msg = str(exc_info.value)
        assert "index 1" in error_msg
        assert "must be a string" in error_msg
