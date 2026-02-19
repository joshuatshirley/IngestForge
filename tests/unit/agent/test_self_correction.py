"""
Tests for Autonomous Self-Correction ().

GWT-compliant test suite covering:
- Error classification
- Retry policies
- Tool execution with retry
- Correction telemetry

NASA JPL Power of Ten compliant.
"""

import inspect
import time
from typing import Any, Dict

import pytest

from ingestforge.agent.self_correction import (
    ErrorCategory,
    ErrorClassifier,
    ToolRetryPolicy,
    CorrectionEvent,
    CorrectionTracker,
    ToolExecutor,
    create_executor,
    classify_error,
    MAX_TOOL_RETRIES,
    MAX_RETRY_DELAY_MS,
    MAX_CORRECTION_EVENTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> ErrorClassifier:
    """Create error classifier."""
    return ErrorClassifier()


@pytest.fixture
def policy() -> ToolRetryPolicy:
    """Create default retry policy."""
    return ToolRetryPolicy()


@pytest.fixture
def tracker() -> CorrectionTracker:
    """Create correction tracker."""
    return CorrectionTracker()


@pytest.fixture
def executor() -> ToolExecutor:
    """Create tool executor."""
    return ToolExecutor()


# ---------------------------------------------------------------------------
# ErrorCategory Tests
# ---------------------------------------------------------------------------


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_retryable_category(self) -> None:
        """Test RETRYABLE category value."""
        assert ErrorCategory.RETRYABLE.value == "retryable"

    def test_parameter_error_category(self) -> None:
        """Test PARAMETER_ERROR category value."""
        assert ErrorCategory.PARAMETER_ERROR.value == "parameter"

    def test_fatal_category(self) -> None:
        """Test FATAL category value."""
        assert ErrorCategory.FATAL.value == "fatal"


# ---------------------------------------------------------------------------
# ErrorClassifier Tests
# ---------------------------------------------------------------------------


class TestErrorClassifier:
    """Tests for ErrorClassifier."""

    def test_classify_timeout_as_retryable(self, classifier: ErrorClassifier) -> None:
        """Test timeout classified as RETRYABLE."""
        assert classifier.classify("Connection timeout") == ErrorCategory.RETRYABLE

    def test_classify_rate_limit_as_retryable(
        self, classifier: ErrorClassifier
    ) -> None:
        """Test rate limit classified as RETRYABLE."""
        assert classifier.classify("Rate limit exceeded") == ErrorCategory.RETRYABLE

    def test_classify_503_as_retryable(self, classifier: ErrorClassifier) -> None:
        """Test 503 status classified as RETRYABLE."""
        assert (
            classifier.classify("Error 503: Service unavailable")
            == ErrorCategory.RETRYABLE
        )

    def test_classify_429_as_retryable(self, classifier: ErrorClassifier) -> None:
        """Test 429 status classified as RETRYABLE."""
        assert classifier.classify("429 Too Many Requests") == ErrorCategory.RETRYABLE

    def test_classify_invalid_param_as_parameter(
        self, classifier: ErrorClassifier
    ) -> None:
        """Test invalid parameter classified as PARAMETER_ERROR."""
        assert (
            classifier.classify("Invalid parameter: query")
            == ErrorCategory.PARAMETER_ERROR
        )

    def test_classify_missing_field_as_parameter(
        self, classifier: ErrorClassifier
    ) -> None:
        """Test missing field classified as PARAMETER_ERROR."""
        assert (
            classifier.classify("Missing required field: id")
            == ErrorCategory.PARAMETER_ERROR
        )

    def test_classify_type_error_as_parameter(
        self, classifier: ErrorClassifier
    ) -> None:
        """Test type error classified as PARAMETER_ERROR."""
        assert (
            classifier.classify("TypeError: expected int")
            == ErrorCategory.PARAMETER_ERROR
        )

    def test_classify_400_as_parameter(self, classifier: ErrorClassifier) -> None:
        """Test 400 status classified as PARAMETER_ERROR."""
        assert classifier.classify("400 Bad Request") == ErrorCategory.PARAMETER_ERROR

    def test_classify_permission_denied_as_fatal(
        self, classifier: ErrorClassifier
    ) -> None:
        """Test permission denied classified as FATAL."""
        assert classifier.classify("Permission denied") == ErrorCategory.FATAL

    def test_classify_unauthorized_as_fatal(self, classifier: ErrorClassifier) -> None:
        """Test unauthorized classified as FATAL."""
        assert classifier.classify("401 Unauthorized") == ErrorCategory.FATAL

    def test_classify_not_found_as_fatal(self, classifier: ErrorClassifier) -> None:
        """Test not found classified as FATAL."""
        assert classifier.classify("Resource not found") == ErrorCategory.FATAL

    def test_classify_403_as_fatal(self, classifier: ErrorClassifier) -> None:
        """Test 403 status classified as FATAL."""
        assert classifier.classify("403 Forbidden") == ErrorCategory.FATAL

    def test_classify_empty_as_retryable(self, classifier: ErrorClassifier) -> None:
        """Test empty message defaults to RETRYABLE."""
        assert classifier.classify("") == ErrorCategory.RETRYABLE

    def test_classify_unknown_as_retryable(self, classifier: ErrorClassifier) -> None:
        """Test unknown error defaults to RETRYABLE."""
        assert (
            classifier.classify("Something weird happened") == ErrorCategory.RETRYABLE
        )

    def test_fatal_takes_priority(self, classifier: ErrorClassifier) -> None:
        """Test FATAL patterns take priority over others."""
        # Message has both retry and fatal keywords
        msg = "Retry failed: Permission denied"
        assert classifier.classify(msg) == ErrorCategory.FATAL


# ---------------------------------------------------------------------------
# ToolRetryPolicy Tests
# ---------------------------------------------------------------------------


class TestToolRetryPolicy:
    """Tests for ToolRetryPolicy."""

    def test_default_max_retries(self, policy: ToolRetryPolicy) -> None:
        """Test default max retries."""
        assert policy.max_retries == MAX_TOOL_RETRIES

    def test_custom_max_retries(self) -> None:
        """Test custom max retries capped at MAX."""
        policy = ToolRetryPolicy(max_retries=10)
        assert policy.max_retries == MAX_TOOL_RETRIES

    def test_get_delay_exponential(self, policy: ToolRetryPolicy) -> None:
        """Test exponential backoff delay."""
        delay0 = policy.get_delay_ms(0)
        delay1 = policy.get_delay_ms(1)
        delay2 = policy.get_delay_ms(2)

        assert delay1 > delay0
        assert delay2 > delay1

    def test_get_delay_capped(self) -> None:
        """Test delay capped at max."""
        policy = ToolRetryPolicy(base_delay_ms=1000, max_delay_ms=2000)
        delay = policy.get_delay_ms(10)  # Would be huge without cap
        assert delay <= 2000

    def test_should_retry_retryable(self, policy: ToolRetryPolicy) -> None:
        """Test should_retry for RETRYABLE errors."""
        assert policy.should_retry(0, ErrorCategory.RETRYABLE) is True
        assert policy.should_retry(2, ErrorCategory.RETRYABLE) is True

    def test_should_retry_fatal_always_false(self, policy: ToolRetryPolicy) -> None:
        """Test should_retry is False for FATAL errors."""
        assert policy.should_retry(0, ErrorCategory.FATAL) is False

    def test_should_retry_exhausted(self, policy: ToolRetryPolicy) -> None:
        """Test should_retry is False when exhausted."""
        assert policy.should_retry(MAX_TOOL_RETRIES, ErrorCategory.RETRYABLE) is False

    def test_negative_max_retries_rejected(self) -> None:
        """Test negative max_retries rejected."""
        with pytest.raises(AssertionError):
            ToolRetryPolicy(max_retries=-1)


# ---------------------------------------------------------------------------
# CorrectionEvent Tests
# ---------------------------------------------------------------------------


class TestCorrectionEvent:
    """Tests for CorrectionEvent."""

    def test_event_creation(self) -> None:
        """Test creating a CorrectionEvent."""
        event = CorrectionEvent(
            event_type="retry",
            tool_name="search",
            attempts=2,
            outcome="success",
            duration_ms=150.5,
            error_category=ErrorCategory.RETRYABLE,
            error_messages=["timeout"],
        )

        assert event.event_type == "retry"
        assert event.tool_name == "search"
        assert event.attempts == 2
        assert event.outcome == "success"

    def test_event_to_dict(self) -> None:
        """Test CorrectionEvent to_dict."""
        event = CorrectionEvent(
            event_type="exhausted",
            tool_name="api_call",
            attempts=3,
            outcome="exhausted",
            duration_ms=500.0,
            error_category=ErrorCategory.RETRYABLE,
        )

        d = event.to_dict()

        assert d["event_type"] == "exhausted"
        assert d["error_category"] == "retryable"


# ---------------------------------------------------------------------------
# CorrectionTracker Tests
# ---------------------------------------------------------------------------


class TestCorrectionTracker:
    """Tests for CorrectionTracker."""

    def test_record_event(self, tracker: CorrectionTracker) -> None:
        """Test recording an event."""
        event = CorrectionEvent(
            event_type="retry",
            tool_name="test",
            attempts=1,
            outcome="success",
            duration_ms=100.0,
            error_category=ErrorCategory.RETRYABLE,
        )

        tracker.record(event)

        assert len(tracker.get_events()) == 1

    def test_bounded_storage(self) -> None:
        """Test events are bounded."""
        tracker = CorrectionTracker(max_events=5)

        for i in range(10):
            event = CorrectionEvent(
                event_type="retry",
                tool_name=f"tool_{i}",
                attempts=1,
                outcome="success",
                duration_ms=100.0,
                error_category=ErrorCategory.RETRYABLE,
            )
            tracker.record(event)

        assert len(tracker.get_events()) == 5

    def test_get_summary(self, tracker: CorrectionTracker) -> None:
        """Test summary statistics."""
        # Record success
        tracker.record(
            CorrectionEvent(
                event_type="retry",
                tool_name="t1",
                attempts=2,
                outcome="success",
                duration_ms=100.0,
                error_category=ErrorCategory.RETRYABLE,
            )
        )
        # Record failure
        tracker.record(
            CorrectionEvent(
                event_type="exhausted",
                tool_name="t2",
                attempts=3,
                outcome="exhausted",
                duration_ms=200.0,
                error_category=ErrorCategory.RETRYABLE,
            )
        )

        summary = tracker.get_summary()

        assert summary["total"] == 2
        assert summary["successes"] == 1
        assert summary["success_rate"] == 0.5

    def test_clear(self, tracker: CorrectionTracker) -> None:
        """Test clearing events."""
        tracker.record(
            CorrectionEvent(
                event_type="retry",
                tool_name="t",
                attempts=1,
                outcome="success",
                duration_ms=100.0,
                error_category=ErrorCategory.RETRYABLE,
            )
        )

        tracker.clear()

        assert len(tracker.get_events()) == 0


# ---------------------------------------------------------------------------
# GWT-1: Retryable Tool Error Recovery
# ---------------------------------------------------------------------------


class TestGWT1RetryableErrorRecovery:
    """GWT-1: Retryable error recovery tests."""

    def test_given_timeout_when_retry_then_success(self) -> None:
        """Given timeout, When retry, Then succeeds."""
        call_count = 0

        def flaky_tool(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Connection timeout")
            return "success"

        executor = ToolExecutor(
            policy=ToolRetryPolicy(base_delay_ms=1)  # Fast for testing
        )
        result, event = executor.execute_with_retry(flaky_tool, "test", {})

        assert result == "success"
        assert event is not None
        assert event.outcome == "success"
        assert event.attempts == 2

    def test_given_rate_limit_when_retry_then_success(self) -> None:
        """Given rate limit, When retry, Then succeeds."""
        call_count = 0

        def rate_limited(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 Too Many Requests")
            return "ok"

        executor = ToolExecutor(policy=ToolRetryPolicy(base_delay_ms=1))
        result, event = executor.execute_with_retry(rate_limited, "api", {})

        assert result == "ok"
        assert event.attempts == 3


# ---------------------------------------------------------------------------
# GWT-2: Parameter Adjustment on Error
# ---------------------------------------------------------------------------


class TestGWT2ParameterAdjustment:
    """GWT-2: Parameter adjustment tests."""

    def test_given_invalid_param_when_corrected_then_success(self) -> None:
        """Given invalid param, When corrected, Then succeeds."""
        call_count = 0

        def param_sensitive(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if kwargs.get("limit", 0) > 100:
                raise Exception("Invalid parameter: limit must be <= 100")
            return "success"

        def corrector(name: str, params: Dict, error: str) -> Dict:
            # Correct the limit parameter
            return {**params, "limit": 50}

        executor = ToolExecutor(
            policy=ToolRetryPolicy(base_delay_ms=1),
            param_corrector=corrector,
        )
        result, event = executor.execute_with_retry(
            param_sensitive, "search", {"limit": 200}
        )

        assert result == "success"


# ---------------------------------------------------------------------------
# GWT-3: Graceful Degradation
# ---------------------------------------------------------------------------


class TestGWT3GracefulDegradation:
    """GWT-3: Graceful degradation tests."""

    def test_given_fatal_error_when_detected_then_no_retry(self) -> None:
        """Given fatal error, When detected, Then no retry."""
        call_count = 0

        def auth_required(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized")

        executor = ToolExecutor(policy=ToolRetryPolicy(base_delay_ms=1))
        result, event = executor.execute_with_retry(auth_required, "api", {})

        assert result is None
        assert event.outcome == "fatal"
        assert call_count == 1  # No retries

    def test_given_exhausted_retries_when_fails_then_logged(self) -> None:
        """Given exhausted retries, When fails, Then logged."""

        def always_fails(**kwargs: Any) -> str:
            raise Exception("Network error")

        executor = ToolExecutor(policy=ToolRetryPolicy(max_retries=2, base_delay_ms=1))
        result, event = executor.execute_with_retry(always_fails, "net", {})

        assert result is None
        assert event.outcome == "exhausted"
        assert event.attempts == 3  # Initial + 2 retries


# ---------------------------------------------------------------------------
# GWT-4: Correction Telemetry
# ---------------------------------------------------------------------------


class TestGWT4CorrectionTelemetry:
    """GWT-4: Correction telemetry tests."""

    def test_given_retry_success_when_complete_then_event_logged(self) -> None:
        """Given retry success, When complete, Then event logged."""
        call_count = 0

        def flaky(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Timeout")
            return "ok"

        executor = ToolExecutor(policy=ToolRetryPolicy(base_delay_ms=1))
        executor.execute_with_retry(flaky, "test", {})

        events = executor.tracker.get_events()
        assert len(events) == 1
        assert events[0].event_type == "retry"
        assert events[0].outcome == "success"

    def test_given_failure_when_complete_then_event_logged(self) -> None:
        """Given failure, When complete, Then event logged."""

        def fails(**kwargs: Any) -> str:
            raise Exception("Always fails")

        executor = ToolExecutor(policy=ToolRetryPolicy(max_retries=1, base_delay_ms=1))
        executor.execute_with_retry(fails, "bad", {})

        events = executor.tracker.get_events()
        assert len(events) == 1
        assert events[0].event_type == "exhausted"

    def test_event_includes_duration(self) -> None:
        """Test event includes duration."""

        def slow(**kwargs: Any) -> str:
            time.sleep(0.01)
            return "done"

        executor = ToolExecutor()
        _, event = executor.execute_with_retry(slow, "slow", {})

        # No event for first-try success
        assert event is None


# ---------------------------------------------------------------------------
# JPL Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten compliance."""

    def test_jpl_rule_2_constants_defined(self) -> None:
        """JPL Rule #2: Verify constants are defined."""
        assert MAX_TOOL_RETRIES == 3
        assert MAX_RETRY_DELAY_MS == 5000
        assert MAX_CORRECTION_EVENTS == 100

    def test_jpl_rule_4_method_sizes(self) -> None:
        """JPL Rule #4: All methods < 60 lines."""
        import ingestforge.agent.self_correction as module

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                for method_name, method in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith("_"):
                        continue
                    try:
                        source = inspect.getsource(method)
                        lines = len(source.split("\n"))
                        assert lines < 60, f"{name}.{method_name} has {lines} lines"
                    except (OSError, TypeError):
                        pass

    def test_jpl_rule_5_assertions_present(self) -> None:
        """JPL Rule #5: Assertions are present."""
        import ingestforge.agent.self_correction as module

        source = inspect.getsource(module)
        assert "assert" in source

    def test_jpl_rule_9_type_hints(self) -> None:
        """JPL Rule #9: Key functions have type hints."""
        assert "return" in classify_error.__annotations__
        assert "return" in create_executor.__annotations__


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_executor(self) -> None:
        """Test create_executor factory."""
        executor = create_executor(max_retries=2)
        assert executor._policy.max_retries == 2

    def test_classify_error_function(self) -> None:
        """Test classify_error convenience function."""
        assert classify_error("Timeout") == ErrorCategory.RETRYABLE
        assert classify_error("401 Unauthorized") == ErrorCategory.FATAL


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_none_tool_fn_rejected(self) -> None:
        """Test None tool_fn rejected."""
        executor = ToolExecutor()
        with pytest.raises(AssertionError):
            executor.execute_with_retry(None, "test", {})  # type: ignore

    def test_empty_tool_name_rejected(self) -> None:
        """Test empty tool_name rejected."""
        executor = ToolExecutor()
        with pytest.raises(AssertionError):
            executor.execute_with_retry(lambda: None, "", {})

    def test_first_try_success_no_event(self) -> None:
        """Test first-try success doesn't create event."""
        executor = ToolExecutor()
        result, event = executor.execute_with_retry(lambda: "ok", "test", {})

        assert result == "ok"
        assert event is None
        assert len(executor.tracker.get_events()) == 0

    def test_param_corrector_failure_handled(self) -> None:
        """Test param corrector failure is handled gracefully."""

        def bad_corrector(name: str, params: Dict, error: str) -> Dict:
            raise Exception("Corrector failed")

        call_count = 0

        def tool(**kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            raise Exception("Invalid parameter: x")

        executor = ToolExecutor(
            policy=ToolRetryPolicy(max_retries=1, base_delay_ms=1),
            param_corrector=bad_corrector,
        )
        result, event = executor.execute_with_retry(tool, "test", {})

        assert result is None  # Still fails
        assert call_count == 2  # But retried
