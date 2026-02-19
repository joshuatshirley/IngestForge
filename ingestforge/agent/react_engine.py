"""ReAct Engine Core for autonomous agent reasoning.

Implements the Reason-Act-Observe loop for autonomous
research and task completion.

Autonomous Self-Correction for tool-level resilience.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Callable, List, Optional, Protocol

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_ITERATIONS = 20
MAX_TOOLS = 50
MAX_THOUGHT_LENGTH = 2000
MAX_OBSERVATION_LENGTH = 5000

# JPL Rule #2 - Fixed upper bounds for retry logic
MAX_TOOL_RETRIES = 3
MAX_RETRY_DELAY_MS = 5000
BASE_RETRY_DELAY_MS = 100


class AgentState(Enum):
    """Agent execution states."""

    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    AUDITING = "auditing"  # New Adversarial State
    REVISING = "revising"  # Added for US-REVISE.1
    PAUSED = "paused"  # Added for US-HITL.1
    COMPLETE = "complete"
    FAILED = "failed"


class ToolResult(Enum):
    """Tool execution result status."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class ToolOutput:
    """Output from a tool execution."""

    status: ToolResult
    data: Any
    error_message: str = ""

    @property
    def is_success(self) -> bool:
        """Check if tool succeeded."""
        return self.status == ToolResult.SUCCESS


# ---------------------------------------------------------------------------
# Self-Correction Components
# ---------------------------------------------------------------------------


class ErrorCategory(Enum):
    """Error classification for retry strategy.

    GWT-1, GWT-2, GWT-3: Error categorization.
    Rule #9: Complete type hints via Enum.
    """

    RETRYABLE = "retryable"  # Timeout, rate limit, temporary failure
    PARAMETER_ERROR = "parameter"  # Invalid input, missing required field
    FATAL = "fatal"  # Permission denied, auth, resource not found


@dataclass
class CorrectionEvent:
    """Telemetry for self-correction events.

    GWT-4: Correction telemetry.
    Rule #9: Complete type hints.
    """

    event_type: str  # "retry", "parameter_adjust", "fallback"
    tool_name: str  # Tool that failed
    attempts: int  # Number of attempts made
    outcome: str  # "success", "exhausted", "fatal"
    duration_ms: float  # Total correction time
    error_messages: List[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "attempts": self.attempts,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "error_messages": self.error_messages,
        }


class ToolRetryPolicy:
    """Retry policy for tool failures.

    GWT-1: Retryable error recovery with backoff.
    Rule #2: Bounded delays and retries.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        max_retries: int = MAX_TOOL_RETRIES,
        base_delay_ms: int = BASE_RETRY_DELAY_MS,
        max_delay_ms: int = MAX_RETRY_DELAY_MS,
    ) -> None:
        """Initialize retry policy with bounds."""
        assert max_retries >= 0, "max_retries must be non-negative"
        assert (
            max_retries <= MAX_TOOL_RETRIES
        ), f"max_retries exceeds {MAX_TOOL_RETRIES}"
        assert base_delay_ms > 0, "base_delay_ms must be positive"

        self._max_retries = max_retries
        self._base_delay_ms = base_delay_ms
        self._max_delay_ms = min(max_delay_ms, MAX_RETRY_DELAY_MS)

    @property
    def max_retries(self) -> int:
        """Maximum retry attempts."""
        return self._max_retries

    def get_delay_ms(self, attempt: int) -> int:
        """Calculate delay for attempt using exponential backoff.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in milliseconds.
        """
        assert attempt >= 0, "attempt must be non-negative"
        # Exponential backoff: base * 2^attempt
        delay = self._base_delay_ms * (2**attempt)
        return min(delay, self._max_delay_ms)


class ErrorClassifier:
    """Classifies errors for retry strategy.

    GWT-1, GWT-2, GWT-3: Error classification.
    Rule #9: Complete type hints.
    """

    # Pattern lists for classification
    RETRYABLE_PATTERNS: List[str] = [
        "timeout",
        "timed out",
        "rate limit",
        "too many requests",
        "temporary",
        "unavailable",
        "connection reset",
        "network",
        "retry",
        "503",
        "502",
        "429",
        "EAGAIN",
    ]

    PARAMETER_PATTERNS: List[str] = [
        "invalid",
        "missing",
        "required",
        "parameter",
        "argument",
        "type error",
        "validation",
        "format",
        "expected",
        "not found",
        "unknown field",
        "bad request",
        "400",
    ]

    FATAL_PATTERNS: List[str] = [
        "permission denied",
        "unauthorized",
        "forbidden",
        "auth",
        "credentials",
        "access denied",
        "403",
        "401",
        "fatal",
        "does not exist",
        "deleted",
        "expired",
        "revoked",
    ]

    def classify(self, error_message: str) -> ErrorCategory:
        """Classify an error message.

        Args:
            error_message: Error message to classify.

        Returns:
            ErrorCategory for the error.
        """
        if not error_message:
            return ErrorCategory.FATAL

        lower_msg = error_message.lower()

        # Check fatal first (most restrictive)
        for pattern in self.FATAL_PATTERNS:
            if pattern in lower_msg:
                return ErrorCategory.FATAL

        # Check retryable (transient failures)
        for pattern in self.RETRYABLE_PATTERNS:
            if pattern in lower_msg:
                return ErrorCategory.RETRYABLE

        # Check parameter errors
        for pattern in self.PARAMETER_PATTERNS:
            if pattern in lower_msg:
                return ErrorCategory.PARAMETER_ERROR

        # Default to fatal for unknown errors (safe approach)
        return ErrorCategory.FATAL


class Tool(Protocol):
    """Protocol for agent tools."""

    @property
    def name(self) -> str:
        """Tool identifier."""
        ...

    @property
    def description(self) -> str:
        """Tool description for LLM."""
        ...

    def execute(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool with given arguments."""
        ...


@dataclass
class ReActStep:
    """Single step in the ReAct loop."""

    iteration: int
    thought: str
    action: Optional[str] = None
    action_input: dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    tool_result: Optional[ToolOutput] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert step to dictionary."""
        return {
            "iteration": self.iteration,
            "thought": self.thought[:MAX_THOUGHT_LENGTH],
            "action": self.action,
            "action_input": self.action_input,
            "observation": self.observation[:MAX_OBSERVATION_LENGTH],
            "success": self.tool_result.is_success if self.tool_result else None,
        }


@dataclass
class AgentResult:
    """Result of agent execution."""

    success: bool
    final_answer: str
    steps: list[ReActStep]
    iterations: int
    state: AgentState
    verification: Optional[dict] = None  # Added for US-REAL-CRITIC.1

    @property
    def step_count(self) -> int:
        """Number of steps taken."""
        return len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "iterations": self.iterations,
            "state": self.state.value,
            "verification": self.verification,
            "steps": [s.to_dict() for s in self.steps],
        }


class ThinkFunction(Protocol):
    """Protocol for thinking function."""

    def __call__(
        self,
        task: str,
        history: list[ReActStep],
        tools: list[str],
    ) -> tuple[str, Optional[str], dict[str, Any]]:
        """Generate thought and action.

        Returns:
            Tuple of (thought, action_name, action_input)
            If action_name is None, agent is done.
        """
        ...


class ReActEngine:
    """ReAct reasoning engine for autonomous agents.

    Implements Reason-Act-Observe loop with configurable
    tools and thinking function.
    """

    def __init__(
        self,
        think_fn: ThinkFunction,
        max_iterations: int = MAX_ITERATIONS,
        memory_searcher: Optional[Any] = None,  # Added for Task 3.2.2
        llm_client: Optional[Any] = None,  # For audit/revision cycles
        retry_policy: Optional[ToolRetryPolicy] = None,  #
    ) -> None:
        """Initialize the engine."""
        if max_iterations < 1:
            max_iterations = 1
        if max_iterations > MAX_ITERATIONS:
            max_iterations = MAX_ITERATIONS

        self._think_fn = think_fn
        self._max_iterations = max_iterations
        self._tools: dict[str, Tool] = {}
        self._state = AgentState.IDLE
        self._memory_searcher = memory_searcher
        self._recalled_context: str = ""
        self._should_pause: bool = False  # Added for interruption
        self._llm_client = llm_client  # For FactChecker and revision

        # Self-correction components
        self._error_classifier = ErrorClassifier()
        self._retry_policy = retry_policy or ToolRetryPolicy()
        self._correction_events: List[CorrectionEvent] = []
        self._unavailable_tools: set[str] = set()  # Temporarily unavailable

    @property
    def state(self) -> AgentState:
        """Current agent state."""
        return self._state

    @property
    def tool_names(self) -> list[str]:
        """Names of registered tools."""
        return list(self._tools.keys())

    def register_tool(self, tool: Tool) -> bool:
        """Register a tool with the engine.

        Args:
            tool: Tool to register

        Returns:
            True if registered
        """
        if len(self._tools) >= MAX_TOOLS:
            logger.warning(f"Max tools ({MAX_TOOLS}) reached")
            return False

        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")
        return True

    def unregister_tool(self, name: str) -> bool:
        """Unregister a tool."""
        if name not in self._tools:
            return False

        del self._tools[name]
        return True

    def request_pause(self) -> None:
        """Signal the engine to pause after the current step."""
        self._should_pause = True

    def resume(self) -> None:
        """Clear pause signal and resume state."""
        self._should_pause = False
        if self._state == AgentState.PAUSED:
            self._state = AgentState.THINKING

    @property
    def correction_events(self) -> List[CorrectionEvent]:
        """List of correction events from this run (GWT-4)."""
        return list(self._correction_events)

    def clear_correction_events(self) -> None:
        """Clear correction events for a new run."""
        self._correction_events.clear()

    def _check_resources_safe(self) -> None:
        """Check if system resources are within safe limits.

        Rule #1: Simple control flow with early return
        Rule #7: Guard against resource exhaustion

        Raises:
            ResourceExhaustedError: If resources exceed 80% threshold
        """
        try:
            from ingestforge.core.system import check_resources

            check_resources()
        except ImportError:
            # Resource guard not available, skip check
            pass

    def run(self, task: str) -> AgentResult:
        """Execute the ReAct loop for a task."""
        if not task.strip():
            return self._make_failure("Empty task provided")

        # Clear correction events and reset unavailable tools
        self._correction_events.clear()
        self._unavailable_tools.clear()

        # Perform Historical Recall (Task 3.2.2)
        if self._memory_searcher:
            self._recalled_context = self._perform_recall(task)

        self._state = AgentState.THINKING
        steps: list[ReActStep] = []
        final_answer = ""
        verification_data = None

        for iteration in range(self._max_iterations):
            # Resource safety check before each iteration
            self._check_resources_safe()

            step = self._execute_iteration(task, steps, iteration)
            steps.append(step)

            # Check for Interruption (US-HITL.1)
            if self._should_pause:
                self._state = AgentState.PAUSED
                logger.info(f"Agent mission paused after iteration {iteration}")
                break

            # Check for completion
            if step.action is None:
                final_answer = step.thought
                MAX_REVISIONS = 2
                for cycle in range(MAX_REVISIONS):
                    self._state = AgentState.AUDITING
                    verification_data = self._run_audit(final_answer, steps)

                    # If score is high enough, we are done
                    if verification_data["score"] >= 0.8:
                        break

                    # Otherwise, trigger self-correction
                    logger.warning(
                        f"Audit failed (score {verification_data['score']}). Triggering revision cycle {cycle+1}"
                    )
                    self._state = AgentState.REVISING
                    final_answer = self._run_revision(
                        final_answer, verification_data, steps
                    )

                self._state = AgentState.COMPLETE
                break

            # Check for failure
            if step.tool_result and not step.tool_result.is_success:
                if self._should_abort(step):
                    self._state = AgentState.FAILED
                    break

        # Handle max iterations
        if self._state == AgentState.THINKING:
            self._state = AgentState.FAILED
            final_answer = "Max iterations reached"

        return AgentResult(
            success=self._state == AgentState.COMPLETE,
            final_answer=final_answer,
            steps=steps,
            iterations=len(steps),
            state=self._state,
            verification=verification_data,  # US-REAL-CRITIC.1
        )

    def _execute_iteration(
        self,
        task: str,
        history: list[ReActStep],
        iteration: int,
    ) -> ReActStep:
        """Execute single ReAct iteration."""
        # Think phase
        self._state = AgentState.THINKING

        # Inject memory context if available (Task 8.2.2)
        effective_task = task
        if self._recalled_context and iteration == 0:
            effective_task = f"{task}\n{self._recalled_context}"

        thought, action, action_input = self._think_fn(
            effective_task, history, self.tool_names
        )

        step = ReActStep(
            iteration=iteration,
            thought=thought[:MAX_THOUGHT_LENGTH],
            action=action,
            action_input=action_input,
        )

        # No action means agent is done thinking
        if action is None:
            return step

        # Act phase - with self-correction retry logic ()
        self._state = AgentState.ACTING
        tool_output = self._execute_action_with_retry(action, action_input)
        step.tool_result = tool_output

        # Observe phase
        self._state = AgentState.OBSERVING
        step.observation = self._format_observation(tool_output)

        return step

    def _execute_action(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> ToolOutput:
        """Execute a tool action (single attempt).

        Args:
            action: Tool name
            action_input: Tool arguments

        Returns:
            Tool output
        """
        if action not in self._tools:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=f"Unknown tool: {action}",
            )

        tool = self._tools[action]
        try:
            return tool.execute(**action_input)
        except Exception as e:
            logger.error(f"Tool {action} failed: {e}")
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=str(e),
            )

    def _execute_action_with_retry(
        self,
        action: str,
        action_input: dict[str, Any],
    ) -> ToolOutput:
        """Execute a tool action with self-correction retry logic.

        GWT-1: Retryable error recovery.
        GWT-2: Parameter adjustment on error.
        GWT-3: Graceful degradation.

        Rule #1: No recursion (bounded loop).
        Rule #2: MAX_TOOL_RETRIES bound.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            action: Tool name.
            action_input: Tool arguments.

        Returns:
            Final tool output after retry attempts.
        """
        assert action, "action cannot be empty"

        # Check if tool is temporarily unavailable
        if action in self._unavailable_tools:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=f"Tool temporarily unavailable: {action}",
            )

        start_time = time.time()
        errors: List[str] = []
        current_input = dict(action_input)

        # Bounded retry loop (JPL Rule #1, #2)
        for attempt in range(self._retry_policy.max_retries + 1):
            output = self._execute_action(action, current_input)

            if output.is_success:
                if attempt > 0:
                    self._log_correction_event(
                        "retry", action, attempt, "success", start_time, errors
                    )
                return output

            # Classify the error
            errors.append(output.error_message)
            category = self._error_classifier.classify(output.error_message)

            # Handle based on category
            if category == ErrorCategory.FATAL:
                self._log_correction_event(
                    "retry", action, attempt + 1, "fatal", start_time, errors
                )
                return output

            if category == ErrorCategory.PARAMETER_ERROR:
                corrected = self._request_parameter_correction(
                    action, current_input, output.error_message
                )
                if corrected is not None:
                    current_input = corrected
                    continue
                # Could not correct parameters, treat as exhausted
                break

            # RETRYABLE: apply backoff and retry
            if attempt < self._retry_policy.max_retries:
                delay_ms = self._retry_policy.get_delay_ms(attempt)
                time.sleep(delay_ms / 1000.0)

        # All retries exhausted - graceful degradation
        self._unavailable_tools.add(action)
        self._log_correction_event(
            "retry",
            action,
            self._retry_policy.max_retries + 1,
            "exhausted",
            start_time,
            errors,
        )
        logger.warning(f"Tool {action} marked temporarily unavailable")

        return ToolOutput(
            status=ToolResult.ERROR,
            data=None,
            error_message=f"Tool exhausted after {len(errors)} attempts: {errors[-1]}",
        )

    def _request_parameter_correction(
        self,
        action: str,
        action_input: dict[str, Any],
        error_message: str,
    ) -> Optional[dict[str, Any]]:
        """Request LLM to suggest corrected parameters.

        GWT-2: Parameter adjustment on error.
        Rule #4: Function < 60 lines.

        Args:
            action: Tool name.
            action_input: Original parameters.
            error_message: Error that occurred.

        Returns:
            Corrected parameters or None if correction not possible.
        """
        if self._llm_client is None:
            return None

        try:
            import json

            prompt = (
                f"A tool call failed with a parameter error.\n\n"
                f"Tool: {action}\n"
                f"Parameters: {json.dumps(action_input, default=str)}\n"
                f"Error: {error_message}\n\n"
                f"Suggest corrected parameters as valid JSON only. "
                f"If you cannot fix it, respond with 'null'."
            )

            response = self._llm_client.generate(prompt)
            if not response or response.strip().lower() == "null":
                return None

            corrected = json.loads(response.strip())
            if isinstance(corrected, dict):
                logger.info(f"LLM suggested parameter correction for {action}")
                return corrected

        except Exception as e:
            logger.debug(f"Parameter correction failed: {e}")

        return None

    def _log_correction_event(
        self,
        event_type: str,
        tool_name: str,
        attempts: int,
        outcome: str,
        start_time: float,
        errors: List[str],
    ) -> None:
        """Log a correction event for telemetry.

        GWT-4: Correction telemetry.

        Args:
            event_type: Type of correction.
            tool_name: Tool that failed.
            attempts: Number of attempts.
            outcome: Final outcome.
            start_time: Start time for duration calculation.
            errors: List of error messages.
        """
        duration_ms = (time.time() - start_time) * 1000
        event = CorrectionEvent(
            event_type=event_type,
            tool_name=tool_name,
            attempts=attempts,
            outcome=outcome,
            duration_ms=duration_ms,
            error_messages=list(errors),
        )
        self._correction_events.append(event)
        logger.info(f"Correction event: {event.to_dict()}")

    def _format_observation(self, output: ToolOutput) -> str:
        """Format tool output as observation.

        Args:
            output: Tool output

        Returns:
            Observation string
        """
        if output.is_success:
            data_str = str(output.data)
            return data_str[:MAX_OBSERVATION_LENGTH]

        return f"Error: {output.error_message}"

    def _should_abort(self, step: ReActStep) -> bool:
        """Check if agent should abort after failure.

        Args:
            step: Failed step

        Returns:
            True if should abort
        """
        # Abort on critical errors only
        if step.tool_result is None:
            return False

        error = step.tool_result.error_message.lower()
        critical = ["permission denied", "auth", "fatal"]

        return any(c in error for c in critical)

    def _make_failure(self, message: str) -> AgentResult:
        """Create a failure result.

        Args:
            message: Failure message

        Returns:
            Failed result
        """
        self._state = AgentState.FAILED
        return AgentResult(
            success=False,
            final_answer=message,
            steps=[],
            iterations=0,
            state=AgentState.FAILED,
        )

    def _perform_recall(self, task: str) -> str:
        """Fetch relevant historical facts to prime the reasoning loop."""
        if not self._memory_searcher:
            return ""

        try:
            # We need a session to query the memory table.
            # In a real app this might be passed in, but for self-contained logic:
            from ingestforge.core.config_loaders import load_config
            from ingestforge.storage.postgres.repository import PostgresRepository

            # Temporary session for recall (optimized for read-only)
            config = load_config()
            # Assuming connection string is available in config or env
            # For MVP simplicity we re-use the standard connection pattern
            repo = PostgresRepository(config.storage.postgres.connection_string)

            with repo.SessionLocal() as session:
                facts = self._memory_searcher.recall_relevant(session, task)

                if not facts:
                    return ""

                logger.info(f"Recalled {len(facts)} historical facts for mission.")

                # Format for prompt injection
                formatted = "\n".join(
                    [f"- {f.fact_text} (Source: {f.source_title})" for f in facts]
                )
                return f"\n### HISTORICAL FINDINGS\nThe following facts from previous missions may be relevant:\n{formatted}\n"

        except Exception as e:
            logger.debug(f"Recall phase failed: {e}")
            return ""

    def _run_audit(self, draft: str, steps: list[ReActStep]) -> dict:
        """Execute the adversarial audit on the final draft using live components."""
        if self._llm_client is None:
            logger.warning("Audit skipped: no LLM client available")
            return {
                "score": 1.0,
                "claims_verified": 0,
                "critic_notes": "Audit skipped (no LLM client).",
            }

        from ingestforge.agent.critic.claim_extractor import ClaimExtractor
        from ingestforge.agent.critic.fact_checker import FactChecker

        extractor = ClaimExtractor()
        checker = FactChecker(self._llm_client)

        claims = extractor.extract(draft)
        if not claims:
            return {
                "score": 1.0,
                "claims_verified": 0,
                "critic_notes": "No atomic claims found to audit.",
            }

        logger.info(f"Auditing {len(claims)} atomic claims in synthesis...")
        verified_results = []
        for claim in claims[:10]:
            # Retrieve evidence text from previous steps
            evidence = self._get_evidence_for_claim(claim, steps)
            result = checker.verify_claim(claim, evidence)
            verified_results.append(result)

        # Calculate final confidence
        valid_count = len([r for r in verified_results if r.status.value == "verified"])
        score = valid_count / len(verified_results) if verified_results else 1.0

        return {
            "score": score,
            "claims_verified": len(verified_results),
            "critic_notes": f"Verified {valid_count} of {len(verified_results)} claims against corpus.",
        }

    def _run_revision(self, draft: str, audit: dict, steps: list[ReActStep]) -> str:
        """Ask the agent to correct its draft based on audit feedback."""
        if self._llm_client is None:
            logger.warning("Revision skipped: no LLM client available")
            return draft

        revision_prompt = (
            "You are in REVISION mode. Your previous draft failed a factual audit.\n\n"
            f"CRITIC FEEDBACK: {audit['critic_notes']}\n\n"
            f"ORIGINAL DRAFT: {draft}\n\n"
            "TASK: Rewrite the report to correct the errors identified by the critic. "
            "Ensure every claim is strictly supported by the source chunks in your history. "
            "If a claim cannot be verified, remove it."
        )

        try:
            return self._llm_client.generate(revision_prompt)
        except Exception as e:
            logger.error(f"Revision cycle failed: {e}")
            return draft  # Fallback to original

    def _get_evidence_for_claim(self, claim: Any, steps: list[ReActStep]) -> str:
        """Helper to find the text associated with a claim's citation."""
        if not claim.citation_id:
            return ""
        for step in steps:
            if step.observation and claim.citation_id in step.observation:
                return step.observation
        return ""


@dataclass
class SimpleTool:
    """Simple tool implementation for testing."""

    name: str
    description: str
    _fn: Callable[..., Any]

    def execute(self, **kwargs: Any) -> ToolOutput:
        """Execute the tool function."""
        try:
            result = self._fn(**kwargs)
            return ToolOutput(status=ToolResult.SUCCESS, data=result)
        except Exception as e:
            return ToolOutput(
                status=ToolResult.ERROR,
                data=None,
                error_message=str(e),
            )


def create_engine(
    think_fn: ThinkFunction,
    max_iterations: int = MAX_ITERATIONS,
    llm_client: Optional[Any] = None,
    retry_policy: Optional[ToolRetryPolicy] = None,
) -> ReActEngine:
    """Factory function to create ReAct engine.

    Args:
        think_fn: Thinking function.
        max_iterations: Max iterations.
        llm_client: Optional LLM client for audit/revision.
        retry_policy: Optional retry policy for tool failures ().

    Returns:
        Configured engine.
    """
    return ReActEngine(
        think_fn=think_fn,
        max_iterations=max_iterations,
        llm_client=llm_client,
        retry_policy=retry_policy,
    )
