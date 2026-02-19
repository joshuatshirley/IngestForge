"""Tests for ReAct engine.

Tests the core reasoning loop implementation.
Autonomous Self-Correction tests included."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional, List

import pytest

from ingestforge.agent.react_engine import (
    AgentState,
    AgentResult,
    ReActEngine,
    ReActStep,
    ToolOutput,
    ToolResult,
    SimpleTool,
    create_engine,
    MAX_ITERATIONS,
    MAX_TOOLS,
    # imports
    ErrorCategory,
    ErrorClassifier,
    ToolRetryPolicy,
    CorrectionEvent,
    MAX_TOOL_RETRIES,
    MAX_RETRY_DELAY_MS,
    BASE_RETRY_DELAY_MS,
)

# Test fixtures


def simple_think_fn(
    task: str,
    history: list[ReActStep],
    tools: list[str],
) -> tuple[str, Optional[str], dict[str, Any]]:
    """Simple thinking function for tests."""
    if len(history) >= 2:
        return ("Task complete", None, {})

    if history:
        return ("Continuing work", "search", {"query": "more info"})

    return ("Starting task", "search", {"query": task})


def immediate_complete_fn(
    task: str,
    history: list[ReActStep],
    tools: list[str],
) -> tuple[str, Optional[str], dict[str, Any]]:
    """Thinking function that completes immediately."""
    return (f"Answer: {task}", None, {})


def always_fail_fn(
    task: str,
    history: list[ReActStep],
    tools: list[str],
) -> tuple[str, Optional[str], dict[str, Any]]:
    """Thinking function that always calls missing tool."""
    return ("Trying", "nonexistent_tool", {})


# AgentState tests


class TestAgentState:
    """Tests for AgentState enum."""

    def test_states_defined(self) -> None:
        """Test all states are defined."""
        states = [s.value for s in AgentState]

        assert "idle" in states
        assert "thinking" in states
        assert "complete" in states
        assert "failed" in states

    def test_state_count(self) -> None:
        """Test correct number of states."""
        # States: IDLE, THINKING, ACTING, OBSERVING, AUDITING, REVISING, PAUSED, COMPLETE, FAILED
        assert len(AgentState) == 9


# ToolResult tests


class TestToolResult:
    """Tests for ToolResult enum."""

    def test_results_defined(self) -> None:
        """Test all results defined."""
        results = [r.value for r in ToolResult]

        assert "success" in results
        assert "error" in results
        assert "timeout" in results


# ToolOutput tests


class TestToolOutput:
    """Tests for ToolOutput dataclass."""

    def test_success_output(self) -> None:
        """Test successful output."""
        output = ToolOutput(status=ToolResult.SUCCESS, data="result")

        assert output.is_success is True
        assert output.data == "result"

    def test_error_output(self) -> None:
        """Test error output."""
        output = ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message="Something failed",
        )

        assert output.is_success is False
        assert output.error_message == "Something failed"


# ReActStep tests


class TestReActStep:
    """Tests for ReActStep dataclass."""

    def test_step_creation(self) -> None:
        """Test creating a step."""
        step = ReActStep(
            iteration=0,
            thought="Thinking about task",
            action="search",
            action_input={"query": "test"},
        )

        assert step.iteration == 0
        assert step.thought == "Thinking about task"
        assert step.action == "search"

    def test_step_to_dict(self) -> None:
        """Test converting step to dict."""
        step = ReActStep(
            iteration=1,
            thought="Working",
            action="fetch",
            action_input={"url": "http://example.com"},
            observation="Got data",
        )

        d = step.to_dict()

        assert d["iteration"] == 1
        assert d["action"] == "fetch"
        assert d["observation"] == "Got data"

    def test_step_no_action(self) -> None:
        """Test step with no action (completion)."""
        step = ReActStep(iteration=2, thought="Final answer")

        assert step.action is None
        assert step.action_input == {}


# AgentResult tests


class TestAgentResult:
    """Tests for AgentResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = AgentResult(
            success=True,
            final_answer="Answer found",
            steps=[],
            iterations=3,
            state=AgentState.COMPLETE,
        )

        assert result.success is True
        assert result.final_answer == "Answer found"

    def test_failure_result(self) -> None:
        """Test failure result."""
        result = AgentResult(
            success=False,
            final_answer="Could not complete",
            steps=[],
            iterations=5,
            state=AgentState.FAILED,
        )

        assert result.success is False
        assert result.state == AgentState.FAILED

    def test_step_count(self) -> None:
        """Test step count property."""
        steps = [
            ReActStep(iteration=0, thought="A"),
            ReActStep(iteration=1, thought="B"),
        ]
        result = AgentResult(
            success=True,
            final_answer="Done",
            steps=steps,
            iterations=2,
            state=AgentState.COMPLETE,
        )

        assert result.step_count == 2

    def test_to_dict(self) -> None:
        """Test converting result to dict."""
        result = AgentResult(
            success=True,
            final_answer="Answer",
            steps=[],
            iterations=1,
            state=AgentState.COMPLETE,
        )

        d = result.to_dict()

        assert d["success"] is True
        assert d["state"] == "complete"


# SimpleTool tests


class TestSimpleTool:
    """Tests for SimpleTool class."""

    def test_tool_execution(self) -> None:
        """Test executing a simple tool."""
        tool = SimpleTool(
            name="add",
            description="Add numbers",
            _fn=lambda a, b: a + b,
        )

        output = tool.execute(a=2, b=3)

        assert output.is_success is True
        assert output.data == 5

    def test_tool_error(self) -> None:
        """Test tool that raises exception."""
        tool = SimpleTool(
            name="fail",
            description="Always fails",
            _fn=lambda: 1 / 0,
        )

        output = tool.execute()

        assert output.is_success is False
        assert "division" in output.error_message.lower()


# ReActEngine tests


class TestReActEngine:
    """Tests for ReActEngine class."""

    def test_engine_creation(self) -> None:
        """Test creating an engine."""
        engine = ReActEngine(think_fn=simple_think_fn)

        assert engine.state == AgentState.IDLE
        assert engine.tool_names == []

    def test_register_tool(self) -> None:
        """Test registering a tool."""
        engine = ReActEngine(think_fn=simple_think_fn)
        tool = SimpleTool(
            name="search",
            description="Search",
            _fn=lambda query: f"Results for {query}",
        )

        result = engine.register_tool(tool)

        assert result is True
        assert "search" in engine.tool_names

    def test_unregister_tool(self) -> None:
        """Test unregistering a tool."""
        engine = ReActEngine(think_fn=simple_think_fn)
        tool = SimpleTool(name="test", description="Test", _fn=lambda: None)
        engine.register_tool(tool)

        result = engine.unregister_tool("test")

        assert result is True
        assert "test" not in engine.tool_names

    def test_unregister_nonexistent(self) -> None:
        """Test unregistering nonexistent tool."""
        engine = ReActEngine(think_fn=simple_think_fn)

        result = engine.unregister_tool("nonexistent")

        assert result is False


class TestReActExecution:
    """Tests for ReAct execution."""

    def test_run_simple_task(self) -> None:
        """Test running a simple task."""
        engine = ReActEngine(think_fn=simple_think_fn)
        tool = SimpleTool(
            name="search",
            description="Search",
            _fn=lambda query: f"Found: {query}",
        )
        engine.register_tool(tool)

        result = engine.run("Find information")

        assert result.success is True
        assert result.iterations > 0

    def test_run_empty_task(self) -> None:
        """Test running empty task."""
        engine = ReActEngine(think_fn=simple_think_fn)

        result = engine.run("")

        assert result.success is False
        assert result.state == AgentState.FAILED

    def test_immediate_completion(self) -> None:
        """Test task that completes immediately."""
        engine = ReActEngine(think_fn=immediate_complete_fn)

        result = engine.run("Simple question")

        assert result.success is True
        assert result.iterations == 1
        assert "Simple question" in result.final_answer

    def test_missing_tool(self) -> None:
        """Test calling missing tool."""
        engine = ReActEngine(think_fn=always_fail_fn, max_iterations=3)

        result = engine.run("Try something")

        # Should fail but not crash
        assert result.iterations <= 3

    def test_max_iterations(self) -> None:
        """Test max iterations limit."""

        def infinite_fn(
            task: str,
            history: list[ReActStep],
            tools: list[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            return ("Keep going", "search", {"query": "more"})

        engine = ReActEngine(think_fn=infinite_fn, max_iterations=5)
        tool = SimpleTool(
            name="search",
            description="Search",
            _fn=lambda query: "data",
        )
        engine.register_tool(tool)

        result = engine.run("Endless task")

        assert result.iterations == 5
        assert result.success is False


class TestToolExecution:
    """Tests for tool execution within engine."""

    def test_tool_receives_args(self) -> None:
        """Test that tools receive arguments."""
        received: dict[str, Any] = {}

        def capture_fn(**kwargs: Any) -> str:
            received.update(kwargs)
            return "captured"

        def single_action_fn(
            task: str,
            history: list[ReActStep],
            tools: list[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Acting", "capture", {"key": "value", "num": 42})

        engine = ReActEngine(think_fn=single_action_fn)
        engine.register_tool(
            SimpleTool(name="capture", description="Capture", _fn=capture_fn)
        )

        engine.run("Test args")

        assert received.get("key") == "value"
        assert received.get("num") == 42

    def test_tool_error_handling(self) -> None:
        """Test error handling in tools."""

        def error_fn(
            task: str,
            history: list[ReActStep],
            tools: list[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done despite error", None, {})
            return ("Trying", "broken", {})

        engine = ReActEngine(think_fn=error_fn)
        engine.register_tool(
            SimpleTool(
                name="broken",
                description="Broken",
                _fn=lambda: exec('raise ValueError("broken")'),
            )
        )

        result = engine.run("Handle error")

        # Should complete despite tool error
        assert result.iterations > 0


# Factory function tests


class TestCreateEngine:
    """Tests for create_engine factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        engine = create_engine(think_fn=simple_think_fn)

        assert engine.state == AgentState.IDLE

    def test_create_custom_iterations(self) -> None:
        """Test creating with custom iterations."""
        engine = create_engine(think_fn=simple_think_fn, max_iterations=10)

        assert engine._max_iterations == 10


# Constants tests


class TestConstants:
    """Tests for module constants."""

    def test_max_iterations(self) -> None:
        """Test MAX_ITERATIONS is reasonable."""
        assert MAX_ITERATIONS > 0
        assert MAX_ITERATIONS == 20

    def test_max_tools(self) -> None:
        """Test MAX_TOOLS is reasonable."""
        assert MAX_TOOLS > 0
        assert MAX_TOOLS == 50


class TestMaxToolsLimit:
    """Tests for tool registration limits."""

    def test_max_tools_enforced(self) -> None:
        """Test that MAX_TOOLS limit is enforced."""
        engine = ReActEngine(think_fn=simple_think_fn)

        # Register MAX_TOOLS tools
        for i in range(MAX_TOOLS):
            tool = SimpleTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                _fn=lambda: None,
            )
            result = engine.register_tool(tool)
            assert result is True

        # Next registration should fail
        extra_tool = SimpleTool(
            name="extra",
            description="Extra",
            _fn=lambda: None,
        )
        result = engine.register_tool(extra_tool)

        assert result is False
        assert len(engine.tool_names) == MAX_TOOLS


# ---------------------------------------------------------------------------
# Self-Correction Tests
# ---------------------------------------------------------------------------


class TestErrorCategory:
    """Tests for ErrorCategory enum ()."""

    def test_categories_defined(self) -> None:
        """Test all categories are defined."""
        categories = [c.value for c in ErrorCategory]
        assert "retryable" in categories
        assert "parameter" in categories
        assert "fatal" in categories

    def test_category_count(self) -> None:
        """Test correct number of categories."""
        assert len(ErrorCategory) == 3

    def test_category_enum_values(self) -> None:
        """Test enum values are strings."""
        assert ErrorCategory.RETRYABLE.value == "retryable"
        assert ErrorCategory.PARAMETER_ERROR.value == "parameter"
        assert ErrorCategory.FATAL.value == "fatal"

    def test_category_comparison(self) -> None:
        """Test categories can be compared."""
        assert ErrorCategory.RETRYABLE != ErrorCategory.FATAL
        assert ErrorCategory.RETRYABLE == ErrorCategory.RETRYABLE


class TestErrorClassifier:
    """Tests for ErrorClassifier (GWT-1, GWT-2, GWT-3)."""

    def test_classify_timeout(self) -> None:
        """GWT-1: Timeout errors are retryable."""
        classifier = ErrorClassifier()
        result = classifier.classify("Connection timeout after 30s")
        assert result == ErrorCategory.RETRYABLE

    def test_classify_rate_limit(self) -> None:
        """GWT-1: Rate limit errors are retryable."""
        classifier = ErrorClassifier()
        result = classifier.classify("429 Too Many Requests")
        assert result == ErrorCategory.RETRYABLE

    def test_classify_invalid_parameter(self) -> None:
        """GWT-2: Parameter errors are classified correctly."""
        classifier = ErrorClassifier()
        result = classifier.classify("Invalid parameter: expected integer")
        assert result == ErrorCategory.PARAMETER_ERROR

    def test_classify_missing_field(self) -> None:
        """GWT-2: Missing field errors are parameter errors."""
        classifier = ErrorClassifier()
        result = classifier.classify("Required field 'query' is missing")
        assert result == ErrorCategory.PARAMETER_ERROR

    def test_classify_permission_denied(self) -> None:
        """GWT-3: Permission denied is fatal."""
        classifier = ErrorClassifier()
        result = classifier.classify("Permission denied: Access forbidden")
        assert result == ErrorCategory.FATAL

    def test_classify_auth_error(self) -> None:
        """GWT-3: Auth errors are fatal."""
        classifier = ErrorClassifier()
        result = classifier.classify("401 Unauthorized")
        assert result == ErrorCategory.FATAL

    def test_classify_empty_message(self) -> None:
        """Test empty message defaults to fatal."""
        classifier = ErrorClassifier()
        result = classifier.classify("")
        assert result == ErrorCategory.FATAL

    def test_classify_unknown_error(self) -> None:
        """Test unknown error defaults to fatal."""
        classifier = ErrorClassifier()
        result = classifier.classify("Something strange happened")
        assert result == ErrorCategory.FATAL


class TestToolRetryPolicy:
    """Tests for ToolRetryPolicy (GWT-1)."""

    def test_default_policy(self) -> None:
        """Test default policy values."""
        policy = ToolRetryPolicy()
        assert policy.max_retries == MAX_TOOL_RETRIES
        assert policy.get_delay_ms(0) == BASE_RETRY_DELAY_MS

    def test_custom_policy(self) -> None:
        """Test custom policy values."""
        policy = ToolRetryPolicy(max_retries=2, base_delay_ms=50)
        assert policy.max_retries == 2
        assert policy.get_delay_ms(0) == 50

    def test_exponential_backoff(self) -> None:
        """Test delay increases exponentially."""
        policy = ToolRetryPolicy(base_delay_ms=100, max_delay_ms=5000)
        assert policy.get_delay_ms(0) == 100
        assert policy.get_delay_ms(1) == 200
        assert policy.get_delay_ms(2) == 400

    def test_max_delay_bounded(self) -> None:
        """Test delay is capped at max_delay_ms."""
        policy = ToolRetryPolicy(base_delay_ms=100, max_delay_ms=500)
        assert policy.get_delay_ms(10) == 500

    def test_jpl_rule2_max_retries_bounded(self) -> None:
        """JPL Rule #2: max_retries cannot exceed MAX_TOOL_RETRIES."""
        with pytest.raises(AssertionError):
            ToolRetryPolicy(max_retries=MAX_TOOL_RETRIES + 1)

    def test_jpl_rule2_max_delay_bounded(self) -> None:
        """JPL Rule #2: max_delay is capped at MAX_RETRY_DELAY_MS."""
        policy = ToolRetryPolicy(max_delay_ms=MAX_RETRY_DELAY_MS + 1000)
        # Should be capped
        assert policy.get_delay_ms(100) <= MAX_RETRY_DELAY_MS


class TestCorrectionEvent:
    """Tests for CorrectionEvent dataclass (GWT-4)."""

    def test_event_creation(self) -> None:
        """Test creating a correction event."""
        event = CorrectionEvent(
            event_type="retry",
            tool_name="search",
            attempts=2,
            outcome="success",
            duration_ms=150.5,
            error_messages=["timeout"],
        )
        assert event.event_type == "retry"
        assert event.tool_name == "search"
        assert event.attempts == 2

    def test_event_to_dict(self) -> None:
        """Test conversion to dictionary."""
        event = CorrectionEvent(
            event_type="parameter_adjust",
            tool_name="fetch",
            attempts=1,
            outcome="success",
            duration_ms=50.0,
        )
        d = event.to_dict()
        assert d["event_type"] == "parameter_adjust"
        assert d["tool_name"] == "fetch"
        assert d["outcome"] == "success"

    def test_event_default_errors(self) -> None:
        """Test default empty error list."""
        event = CorrectionEvent(
            event_type="retry",
            tool_name="test",
            attempts=1,
            outcome="success",
            duration_ms=10.0,
        )
        assert event.error_messages == []


class TestGWT1RetryableErrorRecovery:
    """GWT-1: Retryable tool error recovery tests."""

    def test_retry_succeeds_on_second_attempt(self) -> None:
        """GWT-1: Tool retry succeeds after transient failure."""
        call_count = [0]

        def flaky_fn(**kwargs: Any) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Connection timeout")
            return "success"

        def single_action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "flaky", {"key": "value"})

        engine = ReActEngine(
            think_fn=single_action_fn,
            retry_policy=ToolRetryPolicy(max_retries=2, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="flaky", description="Flaky tool", _fn=flaky_fn)
        )

        result = engine.run("Test retry")
        assert result.success is True
        assert call_count[0] == 2

    def test_retry_exhausted_after_max_attempts(self) -> None:
        """GWT-1: Tool marked unavailable after exhausting retries."""

        def always_timeout(**kwargs: Any) -> str:
            raise Exception("Connection timeout")

        def retry_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done despite failure", None, {})
            return ("Trying", "timeout_tool", {})

        engine = ReActEngine(
            think_fn=retry_fn,
            retry_policy=ToolRetryPolicy(max_retries=2, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="timeout_tool", description="Times out", _fn=always_timeout)
        )

        result = engine.run("Test exhaustion")
        # Tool should be marked unavailable
        assert "timeout_tool" in engine._unavailable_tools

    def test_backoff_delay_applied(self) -> None:
        """GWT-1: Exponential backoff is applied between retries."""
        timestamps: List[float] = []

        def timing_fn(**kwargs: Any) -> str:
            timestamps.append(time.time())
            if len(timestamps) < 3:
                raise Exception("Rate limit exceeded")
            return "ok"

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "timing", {})

        engine = ReActEngine(
            think_fn=action_fn,
            retry_policy=ToolRetryPolicy(max_retries=3, base_delay_ms=50),
        )
        engine.register_tool(
            SimpleTool(name="timing", description="Times calls", _fn=timing_fn)
        )

        engine.run("Test backoff")
        assert len(timestamps) >= 2
        # Second call should be delayed
        if len(timestamps) >= 2:
            gap1 = (timestamps[1] - timestamps[0]) * 1000
            assert gap1 >= 40  # Allow some timing variance


class TestGWT2ParameterAdjustment:
    """GWT-2: Parameter adjustment on error tests."""

    def test_parameter_correction_with_llm(self) -> None:
        """GWT-2: LLM suggests corrected parameters."""

        class MockLLM:
            def generate(self, prompt: str) -> str:
                return '{"query": "corrected_query"}'

        call_inputs: List[dict] = []

        def param_fn(**kwargs: Any) -> str:
            call_inputs.append(dict(kwargs))
            if kwargs.get("query") == "bad_query":
                raise Exception("Invalid parameter: query format wrong")
            return "success"

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "param_tool", {"query": "bad_query"})

        engine = ReActEngine(
            think_fn=action_fn,
            llm_client=MockLLM(),
            retry_policy=ToolRetryPolicy(max_retries=2, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="param_tool", description="Param tool", _fn=param_fn)
        )

        result = engine.run("Test param correction")
        # Should have tried with corrected parameters
        assert len(call_inputs) >= 2
        assert any(c.get("query") == "corrected_query" for c in call_inputs)

    def test_no_correction_without_llm(self) -> None:
        """GWT-2: No parameter correction when LLM unavailable."""

        def param_fn(**kwargs: Any) -> str:
            raise Exception("Invalid parameter")

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "param_tool", {"query": "bad"})

        # No LLM client
        engine = ReActEngine(
            think_fn=action_fn,
            retry_policy=ToolRetryPolicy(max_retries=1, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="param_tool", description="Param tool", _fn=param_fn)
        )

        result = engine.run("Test no correction")
        # Should fail without correction
        assert len(engine.correction_events) > 0


class TestGWT3GracefulDegradation:
    """GWT-3: Graceful degradation tests."""

    def test_tool_marked_unavailable(self) -> None:
        """GWT-3: Failed tool marked as unavailable."""

        def always_fail(**kwargs: Any) -> str:
            raise Exception("Service unavailable")

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "failing", {})

        engine = ReActEngine(
            think_fn=action_fn,
            retry_policy=ToolRetryPolicy(max_retries=1, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="failing", description="Failing", _fn=always_fail)
        )

        engine.run("Test degradation")
        assert "failing" in engine._unavailable_tools

    def test_unavailable_tool_returns_error(self) -> None:
        """GWT-3: Unavailable tool returns immediate error on second call."""
        call_count = [0]

        def fails_then_unavailable(**kwargs: Any) -> str:
            call_count[0] += 1
            raise Exception("Service unavailable")

        def multi_action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            # First call exhausts retries, second call should be immediate error
            if len(history) >= 2:
                return ("Done", None, {})
            return ("Trying", "unreliable", {})

        engine = ReActEngine(
            think_fn=multi_action_fn,
            retry_policy=ToolRetryPolicy(max_retries=1, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(
                name="unreliable", description="Unreliable", _fn=fails_then_unavailable
            )
        )

        result = engine.run("Test unavailable")
        # First call: 2 attempts (original + 1 retry)
        # Second call: 0 attempts (tool is unavailable)
        # So total should be 2
        assert call_count[0] == 2
        # Tool should be marked unavailable after first exhaustion
        assert "unreliable" in engine._unavailable_tools

    def test_fatal_error_no_retry(self) -> None:
        """GWT-3: Fatal errors do not trigger retry."""

        def auth_fail(**kwargs: Any) -> str:
            raise Exception("401 Unauthorized")

        call_count = [0]

        def counting_fn(**kwargs: Any) -> str:
            call_count[0] += 1
            raise Exception("401 Unauthorized")

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "auth_tool", {})

        engine = ReActEngine(
            think_fn=action_fn,
            retry_policy=ToolRetryPolicy(max_retries=3, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="auth_tool", description="Auth tool", _fn=counting_fn)
        )

        engine.run("Test fatal")
        # Should only call once (fatal = no retry)
        assert call_count[0] == 1


class TestGWT4CorrectionTelemetry:
    """GWT-4: Correction telemetry tests."""

    def test_retry_event_logged(self) -> None:
        """GWT-4: Retry events are logged."""

        def timeout_once(**kwargs: Any) -> str:
            if not hasattr(timeout_once, "called"):
                timeout_once.called = True
                raise Exception("Connection timeout")
            return "ok"

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "retry_tool", {})

        engine = ReActEngine(
            think_fn=action_fn,
            retry_policy=ToolRetryPolicy(max_retries=2, base_delay_ms=1),
        )
        engine.register_tool(
            SimpleTool(name="retry_tool", description="Retry tool", _fn=timeout_once)
        )

        engine.run("Test telemetry")
        events = engine.correction_events
        assert len(events) > 0
        assert events[0].event_type == "retry"
        assert events[0].tool_name == "retry_tool"

    def test_event_includes_duration(self) -> None:
        """GWT-4: Events include duration_ms."""

        def slow_fail(**kwargs: Any) -> str:
            time.sleep(0.05)  # 50ms
            raise Exception("Timeout")

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            if history:
                return ("Done", None, {})
            return ("Trying", "slow", {})

        engine = ReActEngine(
            think_fn=action_fn,
            retry_policy=ToolRetryPolicy(max_retries=1, base_delay_ms=1),
        )
        engine.register_tool(SimpleTool(name="slow", description="Slow", _fn=slow_fail))

        engine.run("Test duration")
        events = engine.correction_events
        assert len(events) > 0
        assert events[0].duration_ms >= 50  # At least 50ms

    def test_events_cleared_on_new_run(self) -> None:
        """GWT-4: Events cleared at start of new run."""

        def action_fn(
            task: str,
            history: List[ReActStep],
            tools: List[str],
        ) -> tuple[str, Optional[str], dict[str, Any]]:
            return ("Done", None, {})

        engine = ReActEngine(think_fn=action_fn)
        # Add a fake event
        engine._correction_events.append(
            CorrectionEvent("test", "tool", 1, "success", 10.0)
        )

        engine.run("New task")
        # Events should be cleared
        assert len(engine.correction_events) == 0


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance ()."""

    def test_rule2_max_retries_constant(self) -> None:
        """JPL Rule #2: MAX_TOOL_RETRIES is bounded."""
        assert MAX_TOOL_RETRIES == 3
        assert MAX_TOOL_RETRIES > 0

    def test_rule2_max_delay_constant(self) -> None:
        """JPL Rule #2: MAX_RETRY_DELAY_MS is bounded."""
        assert MAX_RETRY_DELAY_MS == 5000
        assert MAX_RETRY_DELAY_MS > 0

    def test_rule5_policy_assertions(self) -> None:
        """JPL Rule #5: ToolRetryPolicy asserts preconditions."""
        with pytest.raises(AssertionError):
            ToolRetryPolicy(max_retries=-1)
        with pytest.raises(AssertionError):
            ToolRetryPolicy(base_delay_ms=0)

    def test_rule5_delay_assertions(self) -> None:
        """JPL Rule #5: get_delay_ms asserts non-negative attempt."""
        policy = ToolRetryPolicy()
        with pytest.raises(AssertionError):
            policy.get_delay_ms(-1)

    def test_rule9_type_hints_correction_event(self) -> None:
        """JPL Rule #9: CorrectionEvent has complete type hints."""
        hints = CorrectionEvent.__dataclass_fields__
        required_fields = [
            "event_type",
            "tool_name",
            "attempts",
            "outcome",
            "duration_ms",
            "error_messages",
        ]
        for field_name in required_fields:
            assert field_name in hints

    def test_rule9_type_hints_error_category(self) -> None:
        """JPL Rule #9: ErrorCategory is properly typed enum."""
        assert issubclass(ErrorCategory, Enum)
