"""Integration test demonstrating MockLLM usage with ReAct engine.

This test shows how to use the MockLLM fixture to test agent behavior
without requiring an actual LLM API."""

from __future__ import annotations

from typing import Any, Optional

import pytest

from ingestforge.agent.react_engine import (
    AgentState,
    ReActEngine,
    ReActStep,
    SimpleTool,
)
from tests.fixtures.agents import MockLLM


def create_think_fn_with_llm(llm: MockLLM):
    """Create a thinking function that uses the mock LLM.

    Args:
        llm: MockLLM instance to use for generating thoughts

    Returns:
        Thinking function that delegates to the LLM
    """

    def think_fn(
        task: str,
        history: list[ReActStep],
        tools: list[str],
    ) -> tuple[str, Optional[str], dict[str, Any]]:
        """Think function using mock LLM."""
        # Build prompt from task and history
        prompt = f"Task: {task}\n"
        if history:
            prompt += f"Previous steps: {len(history)}\n"
        prompt += f"Available tools: {', '.join(tools)}"

        # Get response from mock LLM
        response = llm.generate(prompt)

        # Parse response (simplified for demo)
        if "Final Answer:" in response:
            # Agent is done
            answer = response.split("Final Answer:")[1].strip()
            return (answer, None, {})

        if "Action:" in response:
            # Extract action
            action_part = response.split("Action:")[1].split("\n")[0].strip()
            thought = response.split("Action:")[0].strip()
            return (thought, action_part, {"query": "test"})

        # Default: continue thinking
        return (response, None, {})

    return think_fn


class TestMockLLMIntegration:
    """Integration tests for MockLLM with ReAct engine."""

    def test_agent_completes_with_mock_llm(self) -> None:
        """Test agent successfully completes task using MockLLM."""
        # Setup mock LLM with planned responses
        llm = MockLLM()
        llm.set_responses(
            [
                "I need to search for information\nAction: search",
                "I have found the answer\nFinal Answer: The result is 42",
            ]
        )

        # Create ReAct engine with LLM-based thinking
        think_fn = create_think_fn_with_llm(llm)
        engine = ReActEngine(think_fn=think_fn)

        # Register a simple search tool
        search_tool = SimpleTool(
            name="search",
            description="Search for information",
            _fn=lambda query: "Found: information about the topic",
        )
        engine.register_tool(search_tool)

        # Run agent
        result = engine.run("What is the answer?")

        # Verify success
        assert result.success is True
        assert result.state == AgentState.COMPLETE
        assert "42" in result.final_answer

        # Verify LLM was called correctly
        assert llm.call_count == 2

    def test_agent_tracks_llm_history(self) -> None:
        """Test that agent interactions are tracked in LLM history."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Starting work\nAction: analyze",
                "Work complete\nFinal Answer: Done",
            ]
        )

        think_fn = create_think_fn_with_llm(llm)
        engine = ReActEngine(think_fn=think_fn)

        analyze_tool = SimpleTool(
            name="analyze",
            description="Analyze data",
            _fn=lambda query: "Analysis results",
        )
        engine.register_tool(analyze_tool)

        engine.run("Analyze the data")

        # Verify history tracking
        history = llm.call_history
        assert len(history) == 2
        assert "Task: Analyze the data" in history[0]

    def test_agent_error_when_responses_exhausted(self) -> None:
        """Test agent behavior when mock LLM runs out of responses."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Step 1\nAction: tool1",
                # Only 2 responses, but agent might need more
            ]
        )

        think_fn = create_think_fn_with_llm(llm)
        engine = ReActEngine(think_fn=think_fn, max_iterations=3)

        tool = SimpleTool(
            name="tool1",
            description="Tool",
            _fn=lambda query: "result",
        )
        engine.register_tool(tool)

        # Should raise AssertionError when LLM exhausted
        with pytest.raises(AssertionError, match="MockLLM called"):
            engine.run("Multi-step task")

    def test_reset_llm_between_runs(self) -> None:
        """Test reusing MockLLM for multiple agent runs."""
        llm = MockLLM()
        llm.set_responses(
            [
                "First run\nFinal Answer: First result",
            ]
        )

        think_fn = create_think_fn_with_llm(llm)
        engine = ReActEngine(think_fn=think_fn)

        # First run
        result1 = engine.run("Task 1")
        assert result1.success is True
        assert llm.call_count == 1

        # Reset LLM for second run (set_responses resets index but not history)
        llm.set_responses(
            [
                "Second run\nFinal Answer: Second result",
            ]
        )

        # Second run
        result2 = engine.run("Task 2")
        assert result2.success is True

        # Verify both runs used LLM (call_count accumulates)
        assert llm.call_count == 2

    def test_mock_llm_verifies_exact_call_count(self) -> None:
        """Test MockLLM enforces exact response count (Rule #5)."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Only response\nFinal Answer: done",
            ]
        )

        think_fn = create_think_fn_with_llm(llm)
        engine = ReActEngine(think_fn=think_fn)

        result = engine.run("Simple task")

        # Verify success
        assert result.success is True

        # Verify all responses consumed
        assert llm.get_remaining_count() == 0
        assert llm.has_responses_remaining() is False

    def test_check_remaining_responses_before_run(self) -> None:
        """Test checking response availability before running agent."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Response 1\nAction: tool",
                "Response 2\nFinal Answer: done",
            ]
        )

        # Verify responses available
        assert llm.has_responses_remaining() is True
        assert llm.get_remaining_count() == 2

        think_fn = create_think_fn_with_llm(llm)
        engine = ReActEngine(think_fn=think_fn)

        tool = SimpleTool(
            name="tool",
            description="Tool",
            _fn=lambda query: "result",
        )
        engine.register_tool(tool)

        engine.run("Task")

        # Verify all consumed
        assert llm.has_responses_remaining() is False
        assert llm.get_remaining_count() == 0


class TestMockLLMErrorScenarios:
    """Tests for error scenarios with MockLLM."""

    def test_helpful_error_on_exhaustion(self) -> None:
        """Test error message quality when responses exhausted."""
        llm = MockLLM()
        llm.set_responses(["Only one"])

        llm.generate("First call")

        with pytest.raises(AssertionError) as exc_info:
            llm.generate("Second call")

        error_msg = str(exc_info.value)
        assert "called 2 times" in error_msg
        assert "1 responses configured" in error_msg

    def test_validation_catches_empty_responses(self) -> None:
        """Test validation prevents empty response list."""
        llm = MockLLM()

        with pytest.raises(ValueError, match="cannot be empty"):
            llm.set_responses([])

    def test_validation_catches_non_string_responses(self) -> None:
        """Test validation prevents non-string responses."""
        llm = MockLLM()

        with pytest.raises(ValueError, match="must be a string"):
            llm.set_responses(["valid", 123, "valid"])  # type: ignore


class TestMockLLMReusability:
    """Tests for MockLLM reusability patterns."""

    def test_reset_preserves_responses(self) -> None:
        """Test reset() allows reusing same responses."""
        llm = MockLLM()
        llm.set_responses(["response 1", "response 2"])

        # First use
        r1 = llm.generate("prompt 1")
        assert r1 == "response 1"

        # Reset and reuse
        llm.reset()

        r2 = llm.generate("prompt 2")
        assert r2 == "response 1"  # Back to start

    def test_multiple_agents_share_llm(self) -> None:
        """Test multiple agents can share one MockLLM (with reset)."""
        llm = MockLLM()

        # First agent
        llm.set_responses(["Agent 1\nFinal Answer: A1"])
        think_fn1 = create_think_fn_with_llm(llm)
        engine1 = ReActEngine(think_fn=think_fn1)
        result1 = engine1.run("Task 1")

        assert result1.success is True

        # Second agent (reset LLM)
        llm.set_responses(["Agent 2\nFinal Answer: A2"])
        think_fn2 = create_think_fn_with_llm(llm)
        engine2 = ReActEngine(think_fn=think_fn2)
        result2 = engine2.run("Task 2")

        assert result2.success is True

        # Verify independent results
        assert "A1" in result1.final_answer
        assert "A2" in result2.final_answer
