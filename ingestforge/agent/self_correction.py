"""
Autonomous Self-Correction for Agent Tool Execution.

Provides error classification, retry policies, and telemetry
for autonomous recovery from tool failures.

NASA JPL Power of Ten compliant.
"""

import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_TOOL_RETRIES = 3
MAX_RETRY_DELAY_MS = 5000
MAX_ERROR_MESSAGE_LENGTH = 500
MAX_CORRECTION_EVENTS = 100


class ErrorCategory(Enum):
    """
    Categories for error classification.

    GWT-1, GWT-2, GWT-3: Error type determines recovery strategy.
    """

    RETRYABLE = "retryable"  # Timeout, rate limit, temporary
    PARAMETER_ERROR = "parameter"  # Invalid input, type mismatch
    FATAL = "fatal"  # Permission, auth, not found


@dataclass
class ToolRetryPolicy:
    """
    Retry configuration for tool execution.

    Defines bounded retry behavior.
    Rule #2: All bounds are fixed constants.
    Rule #9: Complete type hints.
    """

    max_retries: int = MAX_TOOL_RETRIES
    base_delay_ms: int = 100
    max_delay_ms: int = MAX_RETRY_DELAY_MS
    exponential_base: float = 2.0

    def __post_init__(self) -> None:
        """Validate and cap bounds."""
        assert self.max_retries >= 0, "max_retries must be non-negative"
        self.max_retries = min(self.max_retries, MAX_TOOL_RETRIES)
        self.max_delay_ms = min(self.max_delay_ms, MAX_RETRY_DELAY_MS)

    def get_delay_ms(self, attempt: int) -> int:
        """
        Calculate delay for retry attempt using exponential backoff.

        Rule #2: Bounded by max_delay_ms.
        Rule #4: Function < 60 lines.

        Args:
            attempt: Current attempt number (0-indexed).

        Returns:
            Delay in milliseconds.
        """
        if attempt < 0:
            return 0
        delay = self.base_delay_ms * (self.exponential_base**attempt)
        return min(int(delay), self.max_delay_ms)

    def should_retry(self, attempt: int, category: ErrorCategory) -> bool:
        """
        Determine if retry should be attempted.

        Rule #4: Function < 60 lines.

        Args:
            attempt: Current attempt number.
            category: Error category.

        Returns:
            True if should retry.
        """
        if category == ErrorCategory.FATAL:
            return False
        return attempt < self.max_retries


@dataclass
class CorrectionEvent:
    """
    Telemetry for a self-correction event.

    GWT-4: Captures correction details for observability.
    Rule #9: Complete type hints.
    """

    event_type: str  # "retry", "parameter_adjust", "fallback", "exhausted"
    tool_name: str
    attempts: int
    outcome: str  # "success", "exhausted", "fatal"
    duration_ms: float
    error_category: ErrorCategory
    error_messages: List[str] = field(default_factory=list)
    corrected_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "event_type": self.event_type,
            "tool_name": self.tool_name,
            "attempts": self.attempts,
            "outcome": self.outcome,
            "duration_ms": self.duration_ms,
            "error_category": self.error_category.value,
            "error_messages": self.error_messages[:5],  # Limit for logging
        }


class ErrorClassifier:
    """
    Classifies error messages into categories.

    GWT-1, GWT-2, GWT-3: Determines recovery strategy.
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    # Pattern definitions for each category
    RETRYABLE_PATTERNS: List[re.Pattern] = [
        re.compile(r"timeout", re.IGNORECASE),
        re.compile(r"rate.?limit", re.IGNORECASE),
        re.compile(r"too many requests", re.IGNORECASE),
        re.compile(r"temporarily unavailable", re.IGNORECASE),
        re.compile(r"service unavailable", re.IGNORECASE),
        re.compile(r"connection reset", re.IGNORECASE),
        re.compile(r"connection refused", re.IGNORECASE),
        re.compile(r"network error", re.IGNORECASE),
        re.compile(r"retry", re.IGNORECASE),
        re.compile(r"503", re.IGNORECASE),
        re.compile(r"429", re.IGNORECASE),
    ]

    PARAMETER_PATTERNS: List[re.Pattern] = [
        re.compile(r"invalid.*(param|argument|input)", re.IGNORECASE),
        re.compile(r"missing.*(param|argument|field)", re.IGNORECASE),
        re.compile(r"type.?error", re.IGNORECASE),
        re.compile(r"validation.?error", re.IGNORECASE),
        re.compile(r"expected.*(type|format)", re.IGNORECASE),
        re.compile(r"bad request", re.IGNORECASE),
        re.compile(r"400", re.IGNORECASE),
        re.compile(r"422", re.IGNORECASE),
    ]

    FATAL_PATTERNS: List[re.Pattern] = [
        re.compile(r"permission denied", re.IGNORECASE),
        re.compile(r"unauthorized", re.IGNORECASE),
        re.compile(r"forbidden", re.IGNORECASE),
        re.compile(r"not found", re.IGNORECASE),
        re.compile(r"authentication", re.IGNORECASE),
        re.compile(r"auth.?fail", re.IGNORECASE),
        re.compile(r"access denied", re.IGNORECASE),
        re.compile(r"401", re.IGNORECASE),
        re.compile(r"403", re.IGNORECASE),
        re.compile(r"404", re.IGNORECASE),
    ]

    def classify(self, error_message: str) -> ErrorCategory:
        """
        Classify an error message into a category.

        GWT-1, GWT-2, GWT-3: Determines recovery strategy.
        Rule #4: Function < 60 lines.

        Args:
            error_message: Error message to classify.

        Returns:
            ErrorCategory for the message.
        """
        if not error_message:
            return ErrorCategory.RETRYABLE  # Default to retryable

        # Truncate for safety
        msg = error_message[:MAX_ERROR_MESSAGE_LENGTH]

        # Check fatal first (highest priority)
        for pattern in self.FATAL_PATTERNS:
            if pattern.search(msg):
                return ErrorCategory.FATAL

        # Check parameter errors
        for pattern in self.PARAMETER_PATTERNS:
            if pattern.search(msg):
                return ErrorCategory.PARAMETER_ERROR

        # Check retryable
        for pattern in self.RETRYABLE_PATTERNS:
            if pattern.search(msg):
                return ErrorCategory.RETRYABLE

        # Default: assume retryable for unknown errors
        return ErrorCategory.RETRYABLE


class CorrectionTracker:
    """
    Tracks correction events for telemetry.

    GWT-4: Observability for self-correction.
    Rule #2: Bounded event storage.
    Rule #9: Complete type hints.
    """

    def __init__(self, max_events: int = MAX_CORRECTION_EVENTS) -> None:
        """Initialize tracker with bounded storage."""
        assert max_events > 0, "max_events must be positive"
        self._max_events = min(max_events, MAX_CORRECTION_EVENTS)
        self._events: List[CorrectionEvent] = []

    def record(self, event: CorrectionEvent) -> None:
        """
        Record a correction event.

        Rule #2: Bounded storage.

        Args:
            event: Event to record.
        """
        self._events.append(event)
        # Trim if exceeds max
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

        # Log the event
        logger.info(
            f"Correction event: {event.event_type} on {event.tool_name} "
            f"({event.attempts} attempts, {event.outcome})"
        )

    def get_events(self) -> List[CorrectionEvent]:
        """Get all recorded events."""
        return list(self._events)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics.

        Returns:
            Dictionary with correction statistics.
        """
        if not self._events:
            return {"total": 0, "success_rate": 1.0}

        total = len(self._events)
        successes = sum(1 for e in self._events if e.outcome == "success")

        return {
            "total": total,
            "successes": successes,
            "success_rate": successes / total if total > 0 else 1.0,
            "by_type": self._count_by_type(),
            "avg_attempts": sum(e.attempts for e in self._events) / total,
        }

    def _count_by_type(self) -> Dict[str, int]:
        """Count events by type."""
        counts: Dict[str, int] = {}
        for event in self._events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts

    def clear(self) -> None:
        """Clear all events."""
        self._events.clear()


class ToolExecutor:
    """
    Executes tools with self-correction capabilities.

    Main entry point for resilient tool execution.
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        policy: Optional[ToolRetryPolicy] = None,
        classifier: Optional[ErrorClassifier] = None,
        tracker: Optional[CorrectionTracker] = None,
        param_corrector: Optional[Callable[[str, Dict, str], Dict]] = None,
    ) -> None:
        """
        Initialize executor.

        Args:
            policy: Retry policy (default created if None).
            classifier: Error classifier (default created if None).
            tracker: Correction tracker (default created if None).
            param_corrector: Optional callback for parameter correction.
        """
        self._policy = policy or ToolRetryPolicy()
        self._classifier = classifier or ErrorClassifier()
        self._tracker = tracker or CorrectionTracker()
        self._param_corrector = param_corrector

    @property
    def tracker(self) -> CorrectionTracker:
        """Access the correction tracker."""
        return self._tracker

    def execute_with_retry(
        self,
        tool_fn: Callable[..., Any],
        tool_name: str,
        params: Dict[str, Any],
    ) -> Tuple[Any, Optional[CorrectionEvent]]:
        """
        Execute a tool with automatic retry on failure.

        GWT-1, GWT-2, GWT-3: Implements recovery strategies.
        Rule #2: Bounded retries.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            tool_fn: Function to execute.
            tool_name: Name of the tool (for logging).
            params: Parameters to pass to tool.

        Returns:
            Tuple of (result, correction_event).
            Result is None if all attempts failed.
        """
        assert tool_fn is not None, "tool_fn cannot be None"
        assert tool_name, "tool_name cannot be empty"

        start_time = time.time()
        errors: List[str] = []
        current_params = dict(params)
        last_category = ErrorCategory.RETRYABLE

        for attempt in range(self._policy.max_retries + 1):
            try:
                result = tool_fn(**current_params)
                # Success - record if this was a retry
                if attempt > 0:
                    event = self._create_event(
                        "retry",
                        tool_name,
                        attempt + 1,
                        "success",
                        start_time,
                        last_category,
                        errors,
                    )
                    self._tracker.record(event)
                    return (result, event)
                return (result, None)

            except Exception as e:
                error_msg = str(e)[:MAX_ERROR_MESSAGE_LENGTH]
                errors.append(error_msg)
                last_category = self._classifier.classify(error_msg)

                logger.warning(
                    f"Tool {tool_name} failed (attempt {attempt + 1}): "
                    f"{error_msg[:100]} [{last_category.value}]"
                )

                # Check if should retry
                if not self._policy.should_retry(attempt, last_category):
                    break

                # Apply correction strategy
                current_params = self._apply_correction(
                    tool_name, current_params, error_msg, last_category
                )

                # Wait before retry
                delay_ms = self._policy.get_delay_ms(attempt)
                if delay_ms > 0:
                    time.sleep(delay_ms / 1000.0)

        # All attempts exhausted
        outcome = "fatal" if last_category == ErrorCategory.FATAL else "exhausted"
        event = self._create_event(
            "exhausted",
            tool_name,
            len(errors),
            outcome,
            start_time,
            last_category,
            errors,
        )
        self._tracker.record(event)
        return (None, event)

    def _apply_correction(
        self,
        tool_name: str,
        params: Dict[str, Any],
        error_msg: str,
        category: ErrorCategory,
    ) -> Dict[str, Any]:
        """
        Apply correction strategy based on error category.

        GWT-2: Parameter adjustment.
        Rule #4: Function < 60 lines.

        Args:
            tool_name: Tool name.
            params: Current parameters.
            error_msg: Error message.
            category: Error category.

        Returns:
            Corrected parameters.
        """
        if category == ErrorCategory.PARAMETER_ERROR and self._param_corrector:
            try:
                corrected = self._param_corrector(tool_name, params, error_msg)
                logger.info(f"Parameters corrected for {tool_name}")
                return corrected
            except Exception as e:
                logger.warning(f"Parameter correction failed: {e}")

        # No correction applied - return original
        return params

    def _create_event(
        self,
        event_type: str,
        tool_name: str,
        attempts: int,
        outcome: str,
        start_time: float,
        category: ErrorCategory,
        errors: List[str],
    ) -> CorrectionEvent:
        """Create a CorrectionEvent."""
        duration_ms = (time.time() - start_time) * 1000
        return CorrectionEvent(
            event_type=event_type,
            tool_name=tool_name,
            attempts=attempts,
            outcome=outcome,
            duration_ms=duration_ms,
            error_category=category,
            error_messages=errors,
        )


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_executor(
    max_retries: int = MAX_TOOL_RETRIES,
    param_corrector: Optional[Callable[[str, Dict, str], Dict]] = None,
) -> ToolExecutor:
    """
    Create a ToolExecutor with default configuration.

    Args:
        max_retries: Maximum retry attempts.
        param_corrector: Optional parameter correction callback.

    Returns:
        Configured ToolExecutor.
    """
    policy = ToolRetryPolicy(max_retries=max_retries)
    return ToolExecutor(policy=policy, param_corrector=param_corrector)


def classify_error(error_message: str) -> ErrorCategory:
    """
    Convenience function to classify an error.

    Args:
        error_message: Error message to classify.

    Returns:
        ErrorCategory.
    """
    return ErrorClassifier().classify(error_message)
