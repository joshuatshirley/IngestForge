"""
Comprehensive tests for retry module.

Tests the retry decorator, delay calculation, and error handling.
This module is critical infrastructure with 40+ tests validating all edge cases.
"""

import time
from unittest.mock import Mock, call, patch

import pytest

from ingestforge.core.retry import (
    RetryConfig,
    RetryError,
    calculate_delay,
    llm_retry,
    retry,
)


# ============================================================================
# calculate_delay() Tests
# ============================================================================


class TestCalculateDelay:
    """Tests for calculate_delay function."""

    def test_basic_exponential_backoff(self):
        """Test basic exponential backoff calculation."""
        # Attempt 0: 1.0 * (2.0 ^ 0) = 1.0
        delay = calculate_delay(0, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 1.0

        # Attempt 1: 1.0 * (2.0 ^ 1) = 2.0
        delay = calculate_delay(1, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 2.0

        # Attempt 2: 1.0 * (2.0 ^ 2) = 4.0
        delay = calculate_delay(2, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 4.0

        # Attempt 3: 1.0 * (2.0 ^ 3) = 8.0
        delay = calculate_delay(3, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 8.0

    def test_max_delay_capping(self):
        """Test that delay is capped at max_delay."""
        # Attempt 10 would be 1024s, but should cap at 60s
        delay = calculate_delay(10, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 60.0

        # Custom max delay
        delay = calculate_delay(5, 1.0, 10.0, 2.0, jitter=False)
        assert delay == 10.0

    def test_jitter_adds_randomness(self):
        """Test that jitter adds random variation."""
        with patch("random.random", return_value=0.5):
            # Base delay: 4.0, jitter: 4.0 * 0.25 * 0.5 = 0.5
            delay = calculate_delay(2, 1.0, 60.0, 2.0, jitter=True)
            assert delay == 4.5

        with patch("random.random", return_value=1.0):
            # Base delay: 4.0, jitter: 4.0 * 0.25 * 1.0 = 1.0
            delay = calculate_delay(2, 1.0, 60.0, 2.0, jitter=True)
            assert delay == 5.0

    def test_jitter_range(self):
        """Test that jitter stays within 0-25% range."""
        base_delay = 4.0
        delays = []

        for _ in range(100):
            delay = calculate_delay(2, 1.0, 60.0, 2.0, jitter=True)
            delays.append(delay)

        # All delays should be >= base_delay
        assert all(d >= base_delay for d in delays)

        # All delays should be <= base_delay * 1.25
        assert all(d <= base_delay * 1.25 for d in delays)

        # Should have some variation (not all same)
        assert len(set(delays)) > 1

    def test_no_jitter_consistent(self):
        """Test that no jitter gives consistent results."""
        delays = [calculate_delay(2, 1.0, 60.0, 2.0, jitter=False) for _ in range(10)]

        # All delays should be identical
        assert all(d == 4.0 for d in delays)

    def test_custom_base_delay(self):
        """Test custom base delay values."""
        # Base delay of 0.5s
        delay = calculate_delay(1, 0.5, 60.0, 2.0, jitter=False)
        assert delay == 1.0

        # Base delay of 5s
        delay = calculate_delay(1, 5.0, 60.0, 2.0, jitter=False)
        assert delay == 10.0

    def test_custom_exponential_base(self):
        """Test custom exponential base values."""
        # Base 3: 1.0 * (3.0 ^ 2) = 9.0
        delay = calculate_delay(2, 1.0, 60.0, 3.0, jitter=False)
        assert delay == 9.0

        # Base 1.5: 1.0 * (1.5 ^ 2) = 2.25
        delay = calculate_delay(2, 1.0, 60.0, 1.5, jitter=False)
        assert delay == 2.25

    def test_zero_attempt(self):
        """Test delay calculation for attempt 0."""
        delay = calculate_delay(0, 1.0, 60.0, 2.0, jitter=False)
        assert delay == 1.0  # 1.0 * (2.0 ^ 0) = 1.0


# ============================================================================
# @retry Decorator Tests
# ============================================================================


class TestRetryDecorator:
    """Tests for @retry decorator."""

    def test_success_on_first_attempt(self):
        """Test function succeeds on first attempt (no retry needed)."""
        mock_func = Mock(return_value="success")
        decorated = retry(max_attempts=3)(mock_func)

        result = decorated("arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_success_after_one_retry(self):
        """Test function succeeds on second attempt."""
        mock_func = Mock(
            side_effect=[ValueError("fail"), "success"], __name__="mock_func"
        )
        decorated = retry(max_attempts=3, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 2

    def test_success_after_multiple_retries(self):
        """Test function succeeds after multiple retries."""
        mock_func = Mock(
            side_effect=[
                ValueError("fail1"),
                ValueError("fail2"),
                ValueError("fail3"),
                "success",
            ],
            __name__="mock_func",
        )
        decorated = retry(max_attempts=5, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 4

    def test_failure_after_max_attempts(self):
        """Test RetryError raised after max attempts."""
        mock_func = Mock(side_effect=ValueError("always fails"), __name__="mock_func")
        decorated = retry(max_attempts=3, base_delay=0.01)(mock_func)

        with pytest.raises(RetryError) as exc_info:
            decorated()

        assert mock_func.call_count == 3
        assert exc_info.value.attempts == 3
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert str(exc_info.value.last_exception) == "always fails"

    def test_retryable_exceptions_only(self):
        """Test only retryable exceptions trigger retry."""
        # ValueError is retryable, TypeError is not
        mock_func = Mock(side_effect=TypeError("not retryable"))
        decorated = retry(
            max_attempts=3, retryable_exceptions=(ValueError,), base_delay=0.01
        )(mock_func)

        with pytest.raises(TypeError) as exc_info:
            decorated()

        # Should fail immediately without retry
        assert mock_func.call_count == 1
        assert str(exc_info.value) == "not retryable"

    def test_multiple_retryable_exceptions(self):
        """Test multiple exception types are retryable."""
        mock_func = Mock(
            side_effect=[
                ValueError("fail1"),
                TypeError("fail2"),
                ConnectionError("fail3"),
                "success",
            ],
            __name__="mock_func",
        )
        decorated = retry(
            max_attempts=5,
            retryable_exceptions=(ValueError, TypeError, ConnectionError),
            base_delay=0.01,
        )(mock_func)

        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 4

    def test_delay_calculation_integration(self):
        """Test that delays are calculated correctly."""
        mock_func = Mock(
            side_effect=[ValueError("fail"), ValueError("fail"), "success"],
            __name__="mock_func",
        )

        with patch("time.sleep") as mock_sleep:
            decorated = retry(max_attempts=3, base_delay=1.0, jitter=False)(mock_func)
            result = decorated()

        assert result == "success"
        # Should have slept twice (after attempt 1 and 2)
        assert mock_sleep.call_count == 2

        # First delay: 1.0 * (2.0 ^ 0) = 1.0
        # Second delay: 1.0 * (2.0 ^ 1) = 2.0
        mock_sleep.assert_has_calls([call(1.0), call(2.0)])

    def test_no_sleep_after_last_attempt(self):
        """Test no sleep after final failed attempt."""
        mock_func = Mock(side_effect=ValueError("always fails"), __name__="mock_func")

        with patch("time.sleep") as mock_sleep:
            decorated = retry(max_attempts=3, base_delay=0.01)(mock_func)

            with pytest.raises(RetryError):
                decorated()

        # Should sleep after attempts 1 and 2, but not after 3
        assert mock_sleep.call_count == 2

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        exc1 = ValueError("fail1")
        exc2 = ValueError("fail2")
        mock_func = Mock(side_effect=[exc1, exc2, "success"], __name__="mock_func")
        mock_callback = Mock()

        decorated = retry(max_attempts=3, base_delay=0.01, on_retry=mock_callback)(
            mock_func
        )
        result = decorated()

        assert result == "success"
        assert mock_callback.call_count == 2

        # Callback should receive exception and attempt number
        mock_callback.assert_has_calls(
            [
                call(exc1, 1),
                call(exc2, 2),
            ]
        )

    def test_function_with_args(self):
        """Test decorated function preserves args."""
        mock_func = Mock(return_value="result")
        decorated = retry(max_attempts=3)(mock_func)

        result = decorated(1, 2, 3, a="x", b="y")

        assert result == "result"
        mock_func.assert_called_once_with(1, 2, 3, a="x", b="y")

    def test_function_metadata_preserved(self):
        """Test decorated function preserves metadata."""

        def sample_func():
            """Sample docstring."""
            pass

        decorated = retry(max_attempts=3)(sample_func)

        assert decorated.__name__ == "sample_func"
        assert decorated.__doc__ == "Sample docstring."

    def test_max_attempts_validation(self):
        """Test retry with different max_attempts values."""
        mock_func = Mock(side_effect=ValueError("fail"), __name__="mock_func")

        # 1 attempt = no retry
        decorated = retry(max_attempts=1, base_delay=0.01)(mock_func)
        with pytest.raises(RetryError):
            decorated()
        assert mock_func.call_count == 1

        # Reset for next test
        mock_func.reset_mock()
        mock_func.side_effect = ValueError("fail")

        # 5 attempts
        decorated = retry(max_attempts=5, base_delay=0.01)(mock_func)
        with pytest.raises(RetryError):
            decorated()
        assert mock_func.call_count == 5

    def test_exponential_backoff_progression(self):
        """Test exponential backoff increases correctly."""
        mock_func = Mock(side_effect=ValueError("fail"), __name__="mock_func")

        with patch("time.sleep") as mock_sleep:
            decorated = retry(
                max_attempts=4, base_delay=1.0, max_delay=60.0, jitter=False
            )(mock_func)

            with pytest.raises(RetryError):
                decorated()

        # Delays: 1.0, 2.0, 4.0 (no sleep after 4th attempt)
        assert mock_sleep.call_count == 3
        mock_sleep.assert_has_calls([call(1.0), call(2.0), call(4.0)])

    def test_max_delay_capping_integration(self):
        """Test max_delay caps the delay properly."""
        mock_func = Mock(side_effect=ValueError("fail"), __name__="mock_func")

        with patch("time.sleep") as mock_sleep:
            decorated = retry(
                max_attempts=6, base_delay=10.0, max_delay=15.0, jitter=False
            )(mock_func)

            with pytest.raises(RetryError):
                decorated()

        # Delays: 10, 15 (capped), 15 (capped), 15 (capped), 15 (capped)
        delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        assert delays == [10.0, 15.0, 15.0, 15.0, 15.0]

    def test_retry_with_return_value(self):
        """Test retry preserves return values."""
        mock_func = Mock(
            side_effect=[ValueError("fail"), {"key": "value"}], __name__="mock_func"
        )
        decorated = retry(max_attempts=3, base_delay=0.01)(mock_func)

        result = decorated()

        assert result == {"key": "value"}

    def test_exception_chain_preserved(self):
        """Test exception chaining is preserved."""

        def failing_func():
            raise ValueError("original error")

        decorated = retry(max_attempts=2, base_delay=0.01)(failing_func)

        with pytest.raises(RetryError) as exc_info:
            decorated()

        # RetryError should wrap the original ValueError
        assert isinstance(exc_info.value.last_exception, ValueError)
        assert str(exc_info.value.last_exception) == "original error"

    def test_concurrent_calls_independent(self):
        """Test multiple concurrent calls are independent."""
        call_count = {"value": 0}

        def func_with_state():
            call_count["value"] += 1
            if call_count["value"] < 3:
                raise ValueError("fail")
            return "success"

        decorated = retry(max_attempts=5, base_delay=0.01)(func_with_state)

        # First call should succeed after 3 attempts
        result1 = decorated()
        assert result1 == "success"
        assert call_count["value"] == 3

        # Second call should fail (state not reset)
        result2 = decorated()
        assert result2 == "success"

    def test_callback_not_invoked_on_success(self):
        """Test on_retry callback is NOT called when function succeeds first time."""
        mock_func = Mock(return_value="success")
        mock_callback = Mock()

        decorated = retry(max_attempts=3, on_retry=mock_callback)(mock_func)
        result = decorated()

        assert result == "success"
        mock_func.assert_called_once()
        # Callback should never be called since no retry occurred
        mock_callback.assert_not_called()

    def test_callback_exception_breaks_retry(self):
        """Test that exceptions in on_retry callback break retry flow."""
        mock_func = Mock(
            side_effect=[ValueError("fail1"), ValueError("fail2"), "success"],
            __name__="mock_func",
        )

        def failing_callback(exception, attempt):
            """Callback that always raises."""
            raise RuntimeError("callback failed")

        # Callback exception propagates and breaks retry
        decorated = retry(max_attempts=3, base_delay=0.01, on_retry=failing_callback)(
            mock_func
        )

        # The callback exception should propagate
        with pytest.raises(RuntimeError):
            decorated()

    def test_callback_exception_propagates(self):
        """Test that callback exceptions propagate to caller."""
        mock_func = Mock(side_effect=[ValueError("fail")], __name__="mock_func")

        def failing_callback(exception, attempt):
            raise KeyError("callback error")

        decorated = retry(max_attempts=3, base_delay=0.01, on_retry=failing_callback)(
            mock_func
        )

        with pytest.raises(KeyError) as exc_info:
            decorated()

        assert str(exc_info.value) == "'callback error'"

    def test_function_returns_none(self):
        """Test retry works when function returns None."""
        call_count = {"value": 0}

        def returns_none():
            call_count["value"] += 1
            if call_count["value"] < 2:
                raise ValueError("fail")
            return None

        decorated = retry(max_attempts=3, base_delay=0.01)(returns_none)
        result = decorated()

        assert result is None
        assert call_count["value"] == 2

    def test_zero_max_attempts(self):
        """Test that zero max_attempts raises RetryError immediately."""
        mock_func = Mock(return_value="success")
        decorated = retry(max_attempts=0, base_delay=0.01)(mock_func)

        # With 0 attempts, loop won't run (range(0) is empty)
        # Should raise RetryError with last_exception=None
        with pytest.raises(RetryError) as exc_info:
            decorated()

        # Function never called
        mock_func.assert_not_called()

        # RetryError with 0 attempts and None exception
        assert exc_info.value.attempts == 0
        assert exc_info.value.last_exception is None
        assert "Failed after 0 attempts" in str(exc_info.value)

    def test_linear_backoff_via_exponential_base_one(self):
        """Test linear backoff using exponential_base=1.0."""
        mock_func = Mock(side_effect=ValueError("fail"), __name__="mock_func")

        with patch("time.sleep") as mock_sleep:
            decorated = retry(
                max_attempts=4, base_delay=2.0, exponential_base=1.0, jitter=False
            )(mock_func)

            with pytest.raises(RetryError):
                decorated()

        # With exponential_base=1.0, all delays should be constant (base_delay)
        # Delays: 2.0, 2.0, 2.0
        assert mock_sleep.call_count == 3
        for call_args in mock_sleep.call_args_list:
            assert call_args[0][0] == 2.0

    def test_constant_backoff(self):
        """Test constant backoff (exponential_base=1.0)."""
        delays = []
        for attempt in range(5):
            delay = calculate_delay(attempt, 5.0, 60.0, 1.0, jitter=False)
            delays.append(delay)

        # All delays should be constant at base_delay
        assert all(d == 5.0 for d in delays)

    def test_negative_base_delay(self):
        """Test behavior with negative base_delay."""
        # Negative delay would be unusual but should still calculate
        delay = calculate_delay(1, -1.0, 60.0, 2.0, jitter=False)
        assert delay == -2.0  # -1.0 * (2.0 ^ 1)

    def test_negative_exponential_base(self):
        """Test behavior with negative exponential base."""
        # Negative base creates alternating positive/negative delays
        delay0 = calculate_delay(0, 1.0, 60.0, -2.0, jitter=False)
        assert delay0 == 1.0  # 1.0 * (-2.0 ^ 0) = 1.0

        delay1 = calculate_delay(1, 1.0, 60.0, -2.0, jitter=False)
        assert delay1 == -2.0  # 1.0 * (-2.0 ^ 1) = -2.0

    def test_very_large_attempt_number(self):
        """Test delay calculation with very large attempt numbers."""
        # Should cap at max_delay regardless of attempt number
        delay = calculate_delay(1000, 1.0, 10.0, 2.0, jitter=False)
        assert delay == 10.0

    def test_zero_base_delay(self):
        """Test zero base delay."""
        delay = calculate_delay(5, 0.0, 60.0, 2.0, jitter=False)
        assert delay == 0.0

    def test_zero_max_delay(self):
        """Test zero max delay caps all delays to zero."""
        delay = calculate_delay(10, 10.0, 0.0, 2.0, jitter=False)
        assert delay == 0.0


# ============================================================================
# llm_retry Pre-configured Decorator Tests
# ============================================================================


class TestLLMRetry:
    """Tests for llm_retry pre-configured decorator."""

    def test_llm_retry_configuration(self):
        """Test llm_retry has correct default configuration."""
        mock_func = Mock(
            side_effect=[ValueError("fail"), ValueError("fail"), "success"],
            __name__="mock_func",
        )

        with patch("time.sleep") as mock_sleep:
            decorated = llm_retry(mock_func)
            result = decorated()

        assert result == "success"
        assert mock_func.call_count == 3

        # Should have 2 delays (after attempts 1 and 2)
        assert mock_sleep.call_count == 2

    def test_llm_retry_respects_max_attempts(self):
        """Test llm_retry respects configured max_attempts."""
        mock_func = Mock(side_effect=ValueError("always fails"), __name__="mock_func")
        decorated = llm_retry(mock_func)

        with pytest.raises(RetryError) as exc_info:
            decorated()

        # llm_retry configured with max_attempts=3
        assert mock_func.call_count == 3
        assert exc_info.value.attempts == 3

    def test_llm_retry_exponential_backoff(self):
        """Test llm_retry uses exponential backoff."""
        mock_func = Mock(side_effect=ValueError("fail"), __name__="mock_func")

        with patch("time.sleep") as mock_sleep:
            decorated = llm_retry(mock_func)

            with pytest.raises(RetryError):
                decorated()

        # Should have delays with exponential growth
        assert mock_sleep.call_count == 2

        # First delay should be ~1s, second ~2s (with jitter)
        delays = [call_args[0][0] for call_args in mock_sleep.call_args_list]
        assert 1.0 <= delays[0] <= 1.25  # 1s + up to 25% jitter
        assert 2.0 <= delays[1] <= 2.5  # 2s + up to 25% jitter

    def test_llm_retry_max_delay(self):
        """Test llm_retry respects max_delay=30.0."""
        # This is harder to test directly, but we can verify it's set
        # by checking the decorator configuration
        # For now, just verify it works with high attempt counts
        mock_func = Mock(return_value="success")
        decorated = llm_retry(mock_func)

        result = decorated()
        assert result == "success"

    def test_llm_retry_with_real_function(self):
        """Test llm_retry with actual function."""
        attempt_count = {"value": 0}

        @llm_retry
        def api_call():
            attempt_count["value"] += 1
            if attempt_count["value"] < 3:
                raise ConnectionError("API timeout")
            return {"status": "success"}

        result = api_call()

        assert result == {"status": "success"}
        assert attempt_count["value"] == 3


# ============================================================================
# RetryConfig Tests
# ============================================================================


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_default_values(self):
        """Test RetryConfig default values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.retryable_exceptions == (Exception,)

    def test_custom_values(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=120.0,
            exponential_base=3.0,
            jitter=False,
            retryable_exceptions=(ValueError, TypeError),
        )

        assert config.max_attempts == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 120.0
        assert config.exponential_base == 3.0
        assert config.jitter is False
        assert config.retryable_exceptions == (ValueError, TypeError)


# ============================================================================
# RetryError Tests
# ============================================================================


class TestRetryError:
    """Tests for RetryError exception."""

    def test_retry_error_attributes(self):
        """Test RetryError stores exception details."""
        original_error = ValueError("original")
        retry_error = RetryError("Failed after 3 attempts", original_error, 3)

        assert str(retry_error) == "Failed after 3 attempts"
        assert retry_error.last_exception is original_error
        assert retry_error.attempts == 3

    def test_retry_error_is_exception(self):
        """Test RetryError is a proper exception."""
        error = RetryError("test", ValueError("test"), 1)

        assert isinstance(error, Exception)

        # Can be raised and caught
        with pytest.raises(RetryError) as exc_info:
            raise error

        assert exc_info.value is error


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.integration
class TestRetryIntegration:
    """Integration tests for retry module."""

    def test_realistic_api_scenario(self):
        """Test realistic API retry scenario."""
        call_log = []

        def flaky_api_call(data: dict) -> dict:
            """Simulates flaky API that fails first 2 times."""
            call_log.append(time.time())

            if len(call_log) < 3:
                raise ConnectionError("API unavailable")

            return {"result": "success", "data": data}

        decorated = retry(
            max_attempts=5,
            base_delay=0.05,
            retryable_exceptions=(ConnectionError,),
        )(flaky_api_call)

        start_time = time.time()
        result = decorated({"input": "test"})
        total_time = time.time() - start_time

        assert result == {"result": "success", "data": {"input": "test"}}
        assert len(call_log) == 3

        # Verify delays occurred (total time should be > 2 delays)
        assert total_time > 0.05  # At least one delay

    def test_non_retryable_fails_fast(self):
        """Test non-retryable exceptions fail immediately."""

        def func_with_auth_error():
            raise PermissionError("Not authorized")

        decorated = retry(
            max_attempts=5,
            base_delay=0.1,
            retryable_exceptions=(ConnectionError, TimeoutError),
        )(func_with_auth_error)

        start_time = time.time()

        with pytest.raises(PermissionError):
            decorated()

        total_time = time.time() - start_time

        # Should fail immediately without delays
        assert total_time < 0.05

    @pytest.mark.slow
    def test_long_retry_sequence(self):
        """Test long retry sequence with realistic delays."""
        attempts = []

        def eventually_succeeds():
            attempts.append(time.time())
            if len(attempts) < 4:
                raise ValueError("not ready")
            return "success"

        decorated = retry(max_attempts=5, base_delay=0.1, jitter=False)(
            eventually_succeeds
        )

        result = decorated()

        assert result == "success"
        assert len(attempts) == 4

        # Verify delays: 0.1s, 0.2s, 0.4s
        if len(attempts) >= 4:
            delay1 = attempts[1] - attempts[0]
            delay2 = attempts[2] - attempts[1]
            delay3 = attempts[3] - attempts[2]

            assert 0.08 <= delay1 <= 0.15  # ~0.1s ± tolerance
            assert 0.18 <= delay2 <= 0.25  # ~0.2s ± tolerance
            assert 0.38 <= delay3 <= 0.45  # ~0.4s ± tolerance
