"""
Retry Utilities for Resilient Operations.

This module provides retry logic with exponential backoff and jitter for handling
transient failures in external service calls (LLM APIs, embedding services, etc.).

Architecture Context
--------------------
Retry sits in the Core layer and is used by feature modules that call external
services. The LLM, enrichment, and ingest modules all depend on these utilities
to handle rate limits and temporary failures gracefully.

    ┌─────────────────┐
    │  LLM Clients    │──┐
    ├─────────────────┤  │
    │  Embeddings     │──┼──→  @retry decorator
    ├─────────────────┤  │     (exponential backoff + jitter)
    │  Web Fetching   │──┘
    └─────────────────┘

Components
----------
**@retry decorator**
    Wraps synchronous functions with retry logic:

        @retry(max_attempts=3, retryable_exceptions=(APIError,))
        def call_api(data: Any):
            return api.post(data)

**Pre-configured Decorator**
    Ready-to-use decorator for LLM API calls:

    - llm_retry: For LLM API calls (rate limits, timeouts)

Backoff Strategy
----------------
Delay increases exponentially: `base_delay * (exponential_base ^ attempt)`

    Attempt 1: 1.0s  (+ jitter)
    Attempt 2: 2.0s  (+ jitter)
    Attempt 3: 4.0s  (+ jitter)
    Attempt 4: 8.0s  (+ jitter)
    ... capped at max_delay

Jitter (random 0-25% variation) prevents the "thundering herd" problem where
many clients retry simultaneously after a shared failure.

Design Decisions
----------------
1. **Decorator pattern**: Minimal code changes to add retry logic.
2. **Jitter by default**: Prevents cascading retry storms.
3. **Configurable exceptions**: Only retry on expected transient errors.
4. **Async-first**: Native async support, not just sync-wrapped.
"""

import random
import time
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Optional, Tuple, Type

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)


class RetryError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Exception, attempts: int) -> None:
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def calculate_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
) -> float:
    """Calculate delay for next retry attempt."""
    # Exponential backoff: base_delay * (exponential_base ^ attempt)
    delay = base_delay * (exponential_base**attempt)

    # Cap at max delay
    delay = min(delay, max_delay)

    # Add jitter (0-25% random variation)
    if jitter:
        jitter_amount = delay * 0.25 * random.random()
        delay += jitter_amount

    return delay


def _log_retry_attempt(
    exception: Exception, attempt: int, max_attempts: int, delay: float, func_name: str
) -> None:
    """
    Log retry attempt with context.

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        exception: Exception that triggered retry
        attempt: Current attempt number (0-based)
        max_attempts: Maximum attempts allowed
        delay: Delay before next retry
        func_name: Name of function being retried
    """
    logger.warning(
        f"Attempt {attempt + 1}/{max_attempts} failed, " f"retrying in {delay:.2f}s",
        error=str(exception),
        function=func_name,
    )


def _log_final_failure(exception: Exception, max_attempts: int, func_name: str) -> None:
    """
    Log final failure after all retries exhausted.

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        exception: Final exception
        max_attempts: Maximum attempts that were made
        func_name: Name of function that failed
    """
    logger.error(
        f"All {max_attempts} attempts failed",
        error=str(exception),
        function=func_name,
    )


def _handle_retry_attempt(
    exception: Exception,
    attempt: int,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
    func_name: str,
    on_retry: Optional[Callable[[Exception, int], None]],
) -> None:
    """
    Handle logic for a retry attempt (logging, delay, callback).

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        exception: Exception that triggered retry
        attempt: Current attempt number (0-based)
        max_attempts: Maximum attempts allowed
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Base for exponential backoff
        jitter: Whether to add jitter
        func_name: Name of function being retried
        on_retry: Optional callback to invoke
    """
    is_last_attempt = attempt >= max_attempts - 1
    if is_last_attempt:
        _log_final_failure(exception, max_attempts, func_name)
        return

    # Calculate delay and log
    delay = calculate_delay(attempt, base_delay, max_delay, exponential_base, jitter)
    _log_retry_attempt(exception, attempt, max_attempts, delay, func_name)

    # Invoke callback if provided
    if on_retry:
        on_retry(exception, attempt + 1)

    # Sleep before next attempt
    time.sleep(delay)


def _execute_with_retry(
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    exponential_base: float,
    jitter: bool,
    retryable_exceptions: Tuple[Type[Exception], ...],
    on_retry: Optional[Callable[[Exception, int], None]],
) -> Any:
    """
    Execute function with retry logic.

    Rule #1: Extracted from decorator to reduce nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        func: Function to execute
        args: Positional arguments
        kwargs: Keyword arguments
        max_attempts: Maximum attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay cap
        exponential_base: Exponential base
        jitter: Add jitter to delays
        retryable_exceptions: Exceptions to retry on
        on_retry: Optional retry callback

    Returns:
        Function return value

    Raises:
        RetryError: If all attempts fail
    """
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            _handle_retry_attempt(
                e,
                attempt,
                max_attempts,
                base_delay,
                max_delay,
                exponential_base,
                jitter,
                func.__name__,
                on_retry,
            )

    raise RetryError(
        f"Failed after {max_attempts} attempts: {last_exception}",
        last_exception,
        max_attempts,
    )


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = None,
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for retrying functions with exponential backoff.

    Rule #1: Reduced nesting via helper extraction
    Rule #4: Function <60 lines (refactored from 63)
    Rule #9: Full type hints

    Args:
        max_attempts: Maximum number of attempts (including first try)
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Add random jitter to delays
        retryable_exceptions: Exception types to retry on
        on_retry: Callback(exception, attempt) called before each retry

    Example:
        @retry(max_attempts=3, base_delay=1.0)
        def call_api():
            return requests.get("https://api.example.com")

        @retry(retryable_exceptions=(ConnectionError, TimeoutError))
        def fetch_data() -> None:
            ...
    """
    if retryable_exceptions is None:
        retryable_exceptions = (Exception,)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return _execute_with_retry(
                func,
                args,
                kwargs,
                max_attempts,
                base_delay,
                max_delay,
                exponential_base,
                jitter,
                retryable_exceptions,
                on_retry,
            )

        return wrapper

    return decorator


# =============================================================================
# Pre-configured retry decorators for common use cases (DEAD-D06)
# =============================================================================

# For LLM API calls (rate limits, timeouts)
# Delays: 1s, 2s, 4s with exponential backoff (base 2.0)
llm_retry = retry(
    max_attempts=3,
    base_delay=1.0,  # 1s for first retry
    max_delay=30.0,
    exponential_base=2.0,  # 1s -> 2s -> 4s
    jitter=True,
)

# For embedding API calls (model loading, GPU memory)
# More conservative: 2 attempts, longer delays to allow memory recovery
embedding_retry = retry(
    max_attempts=2,
    base_delay=2.0,  # 2s for first retry (model may need reload)
    max_delay=10.0,
    exponential_base=2.0,  # 2s -> 4s
    jitter=True,
)

# For network calls (web fetching, HTTP requests)
# More aggressive: 4 attempts for transient network issues
network_retry = retry(
    max_attempts=4,
    base_delay=0.5,  # 500ms for first retry
    max_delay=15.0,
    exponential_base=2.0,  # 0.5s -> 1s -> 2s -> 4s
    jitter=True,
)
