"""Tests for LLM Adapter.

Tests the adapter that connects LLMClient to ReActEngine's ThinkFunction."""

from __future__ import annotations


import pytest

from ingestforge.agent.llm_adapter import (
    LLMThinkAdapter,
    create_llm_think_adapter,
    MAX_PROMPT_LENGTH,
    MAX_HISTORY_STEPS,
)
from ingestforge.agent.react_engine import ReActStep
from ingestforge.llm.base import GenerationConfig
from tests.fixtures.agents import MockLLM


class TestLLMThinkAdapterCreation:
    """Tests for adapter instantiation and factory."""

    def test_create_with_llm_client(self) -> None:
        """Test creating adapter with LLM client."""
        llm = MockLLM()
        adapter = LLMThinkAdapter(llm_client=llm)

        assert adapter is not None
        assert adapter._llm is llm

    def test_create_with_config(self) -> None:
        """Test creating adapter with custom config."""
        llm = MockLLM()
        config = GenerationConfig(max_tokens=500, temperature=0.5)
        adapter = LLMThinkAdapter(llm_client=llm, config=config)

        assert adapter._config.max_tokens == 500
        assert adapter._config.temperature == 0.5

    def test_create_with_none_llm_raises(self) -> None:
        """Test creating with None LLM raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            LLMThinkAdapter(llm_client=None)  # type: ignore

    def test_factory_function(self) -> None:
        """Test factory function creates adapter."""
        llm = MockLLM()
        adapter = create_llm_think_adapter(llm)

        assert isinstance(adapter, LLMThinkAdapter)

    def test_default_config_values(self) -> None:
        """Test default config has reasonable values."""
        llm = MockLLM()
        adapter = LLMThinkAdapter(llm_client=llm)

        assert adapter._config.max_tokens == 1000
        assert adapter._config.temperature == 0.7
        assert "Observation:" in adapter._config.stop_sequences


class TestReActFormatParsing:
    """Tests for parsing ReAct format responses."""

    def test_parse_complete_response(self) -> None:
        """Test parsing well-formed ReAct response."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Thought: I should search for information\n"
                "Action: search_docs\n"
                'Action Input: {"query": "solar panels"}'
            ]
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Find info",
            history=[],
            tools=["search_docs"],
        )

        assert "search for information" in thought
        assert action == "search_docs"
        assert action_input["query"] == "solar panels"

    def test_parse_finish_action(self) -> None:
        """Test parsing FINISH action returns None."""
        llm = MockLLM()
        llm.set_responses(
            ["Thought: I have the answer now\n" "Action: FINISH\n" "Action Input: {}"]
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Complete task",
            history=[],
            tools=[],
        )

        assert "have the answer" in thought
        assert action is None
        assert action_input == {}

    def test_parse_missing_thought(self) -> None:
        """Test parsing response without thought."""
        llm = MockLLM()
        llm.set_responses(["Action: search\n" 'Action Input: {"q": "test"}'])

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=["search"],
        )

        assert thought == "Unable to parse thought"
        assert action == "search"

    def test_parse_missing_action(self) -> None:
        """Test parsing response without action."""
        llm = MockLLM()
        llm.set_responses(["Thought: Just thinking here\n"])

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=[],
        )

        assert "Just thinking" in thought
        assert action is None

    def test_parse_missing_action_input(self) -> None:
        """Test parsing response without action input."""
        llm = MockLLM()
        llm.set_responses(["Thought: Need to search\n" "Action: search\n"])

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=["search"],
        )

        assert "Need to search" in thought
        assert action == "search"
        assert action_input == {}

    def test_parse_invalid_json_action_input(self) -> None:
        """Test parsing response with invalid JSON in action input."""
        llm = MockLLM()
        llm.set_responses(
            ["Thought: Searching\n" "Action: search\n" "Action Input: {invalid json}"]
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=["search"],
        )

        assert action == "search"
        assert action_input == {}

    def test_parse_empty_response(self) -> None:
        """Test parsing empty LLM response."""
        llm = MockLLM()
        llm.set_responses([""])

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=[],
        )

        assert "No response from LLM" in thought
        assert action is None


class TestPromptBuilding:
    """Tests for building ReAct prompts."""

    def test_prompt_includes_task(self) -> None:
        """Test prompt includes the task description."""
        llm = MockLLM()
        llm.set_responses(["Thought: ok\nAction: FINISH\nAction Input: {}"])

        adapter = LLMThinkAdapter(llm_client=llm)
        adapter(task="Find solar panel info", history=[], tools=[])

        prompt = llm.call_history[0]
        assert "Find solar panel info" in prompt

    def test_prompt_includes_tools(self) -> None:
        """Test prompt includes available tools."""
        llm = MockLLM()
        llm.set_responses(["Thought: ok\nAction: FINISH\nAction Input: {}"])

        adapter = LLMThinkAdapter(llm_client=llm)
        adapter(
            task="Test",
            history=[],
            tools=["search_docs", "analyze_data"],
        )

        prompt = llm.call_history[0]
        assert "search_docs" in prompt
        assert "analyze_data" in prompt

    def test_prompt_includes_history(self) -> None:
        """Test prompt includes conversation history."""
        llm = MockLLM()
        llm.set_responses(["Thought: ok\nAction: FINISH\nAction Input: {}"])

        step1 = ReActStep(
            iteration=0,
            thought="First thought",
            action="search",
            action_input={"query": "test"},
            observation="Found results",
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        adapter(task="Test", history=[step1], tools=["search"])

        prompt = llm.call_history[0]
        assert "First thought" in prompt
        assert "search" in prompt
        assert "Found results" in prompt

    def test_prompt_limits_history(self) -> None:
        """Test prompt limits history to MAX_HISTORY_STEPS."""
        llm = MockLLM()
        llm.set_responses(["Thought: ok\nAction: FINISH\nAction Input: {}"])

        # Create more steps than MAX_HISTORY_STEPS
        history = [
            ReActStep(iteration=i, thought=f"Thought {i}")
            for i in range(MAX_HISTORY_STEPS + 5)
        ]

        adapter = LLMThinkAdapter(llm_client=llm)
        adapter(task="Test", history=history, tools=[])

        prompt = llm.call_history[0]
        # Should have recent thoughts but not very old ones
        assert f"Thought {MAX_HISTORY_STEPS + 4}" in prompt
        assert "Thought 0" not in prompt

    def test_prompt_length_limit(self) -> None:
        """Test prompt is truncated to MAX_PROMPT_LENGTH."""
        llm = MockLLM()
        llm.set_responses(["Thought: ok\nAction: FINISH\nAction Input: {}"])

        # Create very long task
        long_task = "x" * (MAX_PROMPT_LENGTH + 1000)

        adapter = LLMThinkAdapter(llm_client=llm)
        adapter(task=long_task, history=[], tools=[])

        prompt = llm.call_history[0]
        assert len(prompt) <= MAX_PROMPT_LENGTH


class TestErrorHandling:
    """Tests for error handling in adapter."""

    def test_empty_task_returns_default(self) -> None:
        """Test empty task returns default response."""
        llm = MockLLM()
        adapter = LLMThinkAdapter(llm_client=llm)

        thought, action, action_input = adapter(
            task="",
            history=[],
            tools=[],
        )

        assert "No task provided" in thought
        assert action is None

    def test_llm_error_returns_error_thought(self) -> None:
        """Test LLM generation error returns error thought."""
        llm = MockLLM()
        # Don't set responses - will cause error
        llm.set_responses(["response"])
        llm.generate("consume")  # Exhaust responses

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=[],
        )

        assert "Error:" in thought
        assert action is None


class TestIntegrationWithReActEngine:
    """Integration tests with ReActEngine."""

    def test_adapter_works_with_engine(self) -> None:
        """Test adapter works as ThinkFunction in ReActEngine."""
        from ingestforge.agent.react_engine import ReActEngine, SimpleTool

        llm = MockLLM()
        llm.set_responses(
            [
                "Thought: I should search\n"
                "Action: search\n"
                'Action Input: {"query": "test"}',
                "Thought: Found the answer\n" "Action: FINISH\n" "Action Input: {}",
            ]
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        engine = ReActEngine(think_fn=adapter)

        search_tool = SimpleTool(
            name="search",
            description="Search tool",
            _fn=lambda query: f"Results for {query}",
        )
        engine.register_tool(search_tool)

        result = engine.run("Find information")

        assert result.success is True
        assert result.iterations == 2
        assert llm.call_count == 2

    def test_adapter_handles_multiline_thought(self) -> None:
        """Test adapter handles multi-line thoughts."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Thought: This is a complex task.\n"
                "I need to break it down.\n"
                "First, I'll search.\n"
                "Action: search\n"
                'Action Input: {"query": "info"}'
            ]
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Complex task",
            history=[],
            tools=["search"],
        )

        assert "complex task" in thought.lower()
        assert "break it down" in thought.lower()
        assert action == "search"

    def test_adapter_case_insensitive_parsing(self) -> None:
        """Test adapter parses regardless of case."""
        llm = MockLLM()
        llm.set_responses(
            ["thought: lowercase works\n" "action: finish\n" "action input: {}"]
        )

        adapter = LLMThinkAdapter(llm_client=llm)
        thought, action, action_input = adapter(
            task="Test",
            history=[],
            tools=[],
        )

        assert "lowercase" in thought
        assert action is None  # FINISH becomes None


class TestConstants:
    """Tests for module constants."""

    def test_max_prompt_length(self) -> None:
        """Test MAX_PROMPT_LENGTH is reasonable."""
        assert MAX_PROMPT_LENGTH > 0
        assert MAX_PROMPT_LENGTH == 16000

    def test_max_history_steps(self) -> None:
        """Test MAX_HISTORY_STEPS is reasonable."""
        assert MAX_HISTORY_STEPS > 0
        assert MAX_HISTORY_STEPS == 10
