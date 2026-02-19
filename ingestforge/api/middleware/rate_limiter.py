"""
Rate Limiting Middleware for API Protection.

Rate Limiting Middleware.
Protects LLM costs with token quotas using token bucket algorithm.

Follows NASA JPL Power of Ten:
- Rule #1: No recursion
- Rule #2: Fixed bounds on all data structures
- Rule #4: Functions under 60 lines
- Rule #5: Assertions at entry points
- Rule #9: Complete type hints
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Awaitable

from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# JPL Rule #2: Fixed upper bounds
MAX_BUCKETS = 10000
DEFAULT_RATE_LIMIT = 60  # requests per minute
DEFAULT_BURST_LIMIT = 10  # burst capacity
TOKEN_REFILL_INTERVAL_SECONDS = 1.0
MAX_TOKENS_PER_BUCKET = 1000

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class TokenBucket:
    """
    Token bucket for rate limiting.

    Rule #5: Post-init validation.
    Rule #9: Complete type hints.
    """

    tokens: float
    max_tokens: float
    refill_rate: float  # tokens per second
    last_update: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        """Validate bucket parameters."""
        assert 0 <= self.tokens <= MAX_TOKENS_PER_BUCKET, "tokens out of range"
        assert 0 < self.max_tokens <= MAX_TOKENS_PER_BUCKET, "max_tokens out of range"
        assert self.refill_rate > 0, "refill_rate must be positive"


@dataclass
class RateLimitConfig:
    """
    Rate limiting configuration.

    Rule #9: Complete type hints.
    """

    requests_per_minute: int = DEFAULT_RATE_LIMIT
    burst_limit: int = DEFAULT_BURST_LIMIT
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.requests_per_minute > 0, "requests_per_minute must be positive"
        assert self.burst_limit > 0, "burst_limit must be positive"


@dataclass
class RateLimitResult:
    """
    Result of rate limit check.

    Rule #9: Complete type hints.
    """

    allowed: bool
    remaining: int
    reset_seconds: float
    retry_after: Optional[float] = None

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset_seconds)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


# =============================================================================
# Rate Limiter
# =============================================================================


class RateLimiter:
    """
    Token bucket rate limiter.

    Rule #2: Fixed bounds on bucket count.
    Rule #9: Complete type hints.
    """

    def __init__(self, config: Optional[RateLimitConfig] = None) -> None:
        """
        Initialize rate limiter.

        Args:
            config: Rate limiting configuration.
        """
        self._config = config or RateLimitConfig()
        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    @property
    def bucket_count(self) -> int:
        """Return current bucket count."""
        return len(self._buckets)

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """
        Check if request is allowed and consume tokens.

        Rule #2: Enforces MAX_BUCKETS limit.
        Rule #4: Under 60 lines.
        Rule #5: Validates inputs.

        Args:
            key: Identifier for rate limiting (user ID, IP, API key).
            cost: Number of tokens to consume.

        Returns:
            RateLimitResult indicating if request is allowed.
        """
        assert key, "key must be non-empty"
        assert cost > 0, "cost must be positive"

        if not self._config.enabled:
            return RateLimitResult(allowed=True, remaining=999, reset_seconds=0)

        async with self._lock:
            bucket = self._get_or_create_bucket(key)
            self._refill_bucket(bucket)

            if bucket.tokens >= cost:
                bucket.tokens -= cost
                remaining = int(bucket.tokens)
                reset_seconds = (bucket.max_tokens - bucket.tokens) / bucket.refill_rate
                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    reset_seconds=reset_seconds,
                )
            else:
                retry_after = (cost - bucket.tokens) / bucket.refill_rate
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_seconds=bucket.max_tokens / bucket.refill_rate,
                    retry_after=retry_after,
                )

    def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """
        Get or create a token bucket for a key.

        Rule #2: Enforces MAX_BUCKETS limit by evicting oldest.
        Rule #4: Under 60 lines.

        Args:
            key: The bucket identifier.

        Returns:
            The token bucket for the key.
        """
        if key in self._buckets:
            return self._buckets[key]

        # Evict oldest buckets if limit reached
        if len(self._buckets) >= MAX_BUCKETS:
            self._evict_oldest_buckets(int(MAX_BUCKETS * 0.1))

        # Calculate refill rate from config
        refill_rate = self._config.requests_per_minute / 60.0

        bucket = TokenBucket(
            tokens=float(self._config.burst_limit),
            max_tokens=float(self._config.burst_limit),
            refill_rate=refill_rate,
        )
        self._buckets[key] = bucket
        return bucket

    def _evict_oldest_buckets(self, count: int) -> None:
        """
        Evict oldest buckets by last update time.

        Rule #4: Under 60 lines.

        Args:
            count: Number of buckets to evict.
        """
        if count <= 0 or not self._buckets:
            return

        # Sort by last_update and remove oldest
        sorted_keys = sorted(
            self._buckets.keys(),
            key=lambda k: self._buckets[k].last_update,
        )

        for key in sorted_keys[:count]:
            del self._buckets[key]

        logger.info(f"Evicted {count} rate limit buckets")

    def _refill_bucket(self, bucket: TokenBucket) -> None:
        """
        Refill tokens based on elapsed time.

        Rule #4: Under 60 lines.

        Args:
            bucket: The bucket to refill.
        """
        now = time.time()
        elapsed = now - bucket.last_update
        bucket.last_update = now

        if elapsed > 0:
            new_tokens = bucket.tokens + (elapsed * bucket.refill_rate)
            bucket.tokens = min(new_tokens, bucket.max_tokens)

    def reset(self, key: str) -> bool:
        """
        Reset rate limit for a key.

        Args:
            key: The key to reset.

        Returns:
            True if key existed and was reset.
        """
        if key in self._buckets:
            del self._buckets[key]
            return True
        return False

    def get_stats(self) -> Dict[str, int]:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with stats.
        """
        return {
            "bucket_count": self.bucket_count,
            "max_buckets": MAX_BUCKETS,
            "requests_per_minute": self._config.requests_per_minute,
            "burst_limit": self._config.burst_limit,
        }


# =============================================================================
# Middleware
# =============================================================================


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for rate limiting.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        app: ASGIApp,
        limiter: Optional[RateLimiter] = None,
        key_func: Optional[Callable[[Request], str]] = None,
        protected_paths: Optional[list] = None,
    ) -> None:
        """
        Initialize rate limit middleware.

        Args:
            app: The ASGI application.
            limiter: Rate limiter instance.
            key_func: Function to extract rate limit key from request.
            protected_paths: List of path prefixes to protect.
        """
        super().__init__(app)
        self._limiter = limiter or RateLimiter()
        self._key_func = key_func or self._default_key_func
        self._protected_paths = protected_paths or ["/v1/agent", "/v1/synthesis"]

    def _default_key_func(self, request: Request) -> str:
        """
        Extract rate limit key from request.

        Uses Authorization header if present, otherwise client IP.

        Args:
            request: The incoming request.

        Returns:
            Rate limit key string.
        """
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return f"token:{auth[7:20]}"  # Use first 13 chars of token

        client_ip = request.client.host if request.client else "unknown"
        return f"ip:{client_ip}"

    def _is_protected_path(self, path: str) -> bool:
        """
        Check if path should be rate limited.

        Args:
            path: The request path.

        Returns:
            True if path should be protected.
        """
        for prefix in self._protected_paths:
            if path.startswith(prefix):
                return True
        return False

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """
        Process request with rate limiting.

        Rule #4: Under 60 lines.

        Args:
            request: The incoming request.
            call_next: The next middleware/handler.

        Returns:
            The response.
        """
        # Skip rate limiting for non-protected paths
        if not self._is_protected_path(request.url.path):
            return await call_next(request)

        key = self._key_func(request)
        result = await self._limiter.check(key)

        if not result.allowed:
            logger.warning(f"Rate limit exceeded for {key}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded",
                headers=result.to_headers(),
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for header, value in result.to_headers().items():
            response.headers[header] = value

        return response


# =============================================================================
# Convenience Functions
# =============================================================================


def create_rate_limiter(
    requests_per_minute: int = DEFAULT_RATE_LIMIT,
    burst_limit: int = DEFAULT_BURST_LIMIT,
    enabled: bool = True,
) -> RateLimiter:
    """
    Create a configured rate limiter.

    Args:
        requests_per_minute: Maximum requests per minute.
        burst_limit: Burst capacity.
        enabled: Whether rate limiting is enabled.

    Returns:
        Configured RateLimiter instance.
    """
    config = RateLimitConfig(
        requests_per_minute=requests_per_minute,
        burst_limit=burst_limit,
        enabled=enabled,
    )
    return RateLimiter(config)


def create_rate_limit_middleware(
    app: ASGIApp,
    requests_per_minute: int = DEFAULT_RATE_LIMIT,
    burst_limit: int = DEFAULT_BURST_LIMIT,
    protected_paths: Optional[list] = None,
) -> RateLimitMiddleware:
    """
    Create rate limit middleware.

    Args:
        app: The ASGI application.
        requests_per_minute: Maximum requests per minute.
        burst_limit: Burst capacity.
        protected_paths: Paths to protect.

    Returns:
        Configured middleware instance.
    """
    limiter = create_rate_limiter(requests_per_minute, burst_limit)
    return RateLimitMiddleware(
        app,
        limiter=limiter,
        protected_paths=protected_paths,
    )
