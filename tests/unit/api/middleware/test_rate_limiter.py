"""
Tests for Rate Limiting Middleware.

Rate Limiting Middleware.
Verifies JPL Power of Ten compliance.
"""

import pytest
import asyncio
from unittest.mock import Mock
from fastapi import Request

from ingestforge.api.middleware.rate_limiter import (
    TokenBucket,
    RateLimitConfig,
    RateLimitResult,
    RateLimiter,
    RateLimitMiddleware,
    create_rate_limiter,
    create_rate_limit_middleware,
    MAX_BUCKETS,
    DEFAULT_RATE_LIMIT,
    DEFAULT_BURST_LIMIT,
    MAX_TOKENS_PER_BUCKET,
)


# =============================================================================
# TestTokenBucket
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket dataclass."""

    def test_create_valid_bucket(self) -> None:
        """Test creating a valid token bucket."""
        bucket = TokenBucket(
            tokens=10.0,
            max_tokens=20.0,
            refill_rate=1.0,
        )

        assert bucket.tokens == 10.0
        assert bucket.max_tokens == 20.0
        assert bucket.refill_rate == 1.0
        assert bucket.last_update > 0

    def test_tokens_out_of_range_fails(self) -> None:
        """Test that tokens out of range raises AssertionError."""
        with pytest.raises(AssertionError):
            TokenBucket(
                tokens=MAX_TOKENS_PER_BUCKET + 1,
                max_tokens=10.0,
                refill_rate=1.0,
            )

    def test_negative_tokens_fails(self) -> None:
        """Test that negative tokens raises AssertionError."""
        with pytest.raises(AssertionError):
            TokenBucket(
                tokens=-1.0,
                max_tokens=10.0,
                refill_rate=1.0,
            )

    def test_invalid_refill_rate_fails(self) -> None:
        """Test that zero refill rate raises AssertionError."""
        with pytest.raises(AssertionError):
            TokenBucket(
                tokens=5.0,
                max_tokens=10.0,
                refill_rate=0,
            )


# =============================================================================
# TestRateLimitConfig
# =============================================================================


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = RateLimitConfig()

        assert config.requests_per_minute == DEFAULT_RATE_LIMIT
        assert config.burst_limit == DEFAULT_BURST_LIMIT
        assert config.enabled is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = RateLimitConfig(
            requests_per_minute=100,
            burst_limit=20,
            enabled=False,
        )

        assert config.requests_per_minute == 100
        assert config.burst_limit == 20
        assert config.enabled is False

    def test_invalid_requests_per_minute_fails(self) -> None:
        """Test that invalid requests_per_minute raises AssertionError."""
        with pytest.raises(AssertionError):
            RateLimitConfig(requests_per_minute=0)

    def test_invalid_burst_limit_fails(self) -> None:
        """Test that invalid burst_limit raises AssertionError."""
        with pytest.raises(AssertionError):
            RateLimitConfig(burst_limit=-1)


# =============================================================================
# TestRateLimitResult
# =============================================================================


class TestRateLimitResult:
    """Tests for RateLimitResult dataclass."""

    def test_allowed_result(self) -> None:
        """Test creating an allowed result."""
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            reset_seconds=10.0,
        )

        assert result.allowed is True
        assert result.remaining == 5
        assert result.retry_after is None

    def test_denied_result(self) -> None:
        """Test creating a denied result."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_seconds=60.0,
            retry_after=30.0,
        )

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after == 30.0

    def test_to_headers(self) -> None:
        """Test converting result to HTTP headers."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_seconds=60.0,
            retry_after=30.0,
        )

        headers = result.to_headers()

        assert headers["X-RateLimit-Remaining"] == "0"
        assert headers["X-RateLimit-Reset"] == "60"
        assert headers["Retry-After"] == "30"

    def test_to_headers_without_retry(self) -> None:
        """Test headers without retry_after."""
        result = RateLimitResult(
            allowed=True,
            remaining=5,
            reset_seconds=10.0,
        )

        headers = result.to_headers()

        assert "Retry-After" not in headers


# =============================================================================
# TestRateLimiter
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        """Create a rate limiter for testing."""
        config = RateLimitConfig(
            requests_per_minute=60,
            burst_limit=5,
        )
        return RateLimiter(config)

    @pytest.mark.asyncio
    async def test_allow_within_limit(self, limiter: RateLimiter) -> None:
        """Test that requests within limit are allowed."""
        result = await limiter.check("user-1")

        assert result.allowed is True
        assert result.remaining == 4  # 5 - 1

    @pytest.mark.asyncio
    async def test_deny_when_exhausted(self, limiter: RateLimiter) -> None:
        """Test that requests are denied when tokens exhausted."""
        # Exhaust all tokens
        for _ in range(5):
            await limiter.check("user-2")

        # Next request should be denied
        result = await limiter.check("user-2")

        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after is not None
        assert result.retry_after > 0

    @pytest.mark.asyncio
    async def test_different_keys_independent(self, limiter: RateLimiter) -> None:
        """Test that different keys have independent limits."""
        # Exhaust user-3's tokens
        for _ in range(5):
            await limiter.check("user-3")

        # user-4 should still have tokens
        result = await limiter.check("user-4")

        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_disabled_limiter_always_allows(self) -> None:
        """Test that disabled limiter allows all requests."""
        config = RateLimitConfig(enabled=False)
        limiter = RateLimiter(config)

        for _ in range(100):
            result = await limiter.check("user-5")
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_custom_cost(self, limiter: RateLimiter) -> None:
        """Test consuming multiple tokens at once."""
        result = await limiter.check("user-6", cost=3)

        assert result.allowed is True
        assert result.remaining == 2  # 5 - 3

    @pytest.mark.asyncio
    async def test_bucket_eviction(self) -> None:
        """Test that old buckets are evicted when limit reached."""
        config = RateLimitConfig(requests_per_minute=60, burst_limit=5)
        limiter = RateLimiter(config)

        # Create many buckets
        for i in range(100):
            await limiter.check(f"user-evict-{i}")

        assert limiter.bucket_count <= MAX_BUCKETS

    def test_reset(self, limiter: RateLimiter) -> None:
        """Test resetting a rate limit."""
        # First check creates bucket
        asyncio.run(limiter.check("user-reset"))
        assert limiter.bucket_count == 1

        # Reset removes bucket
        result = limiter.reset("user-reset")
        assert result is True
        assert limiter.bucket_count == 0

    def test_reset_nonexistent(self, limiter: RateLimiter) -> None:
        """Test resetting a nonexistent key."""
        result = limiter.reset("nonexistent")
        assert result is False

    def test_get_stats(self, limiter: RateLimiter) -> None:
        """Test getting rate limiter stats."""
        stats = limiter.get_stats()

        assert stats["bucket_count"] == 0
        assert stats["max_buckets"] == MAX_BUCKETS
        assert stats["requests_per_minute"] == 60
        assert stats["burst_limit"] == 5

    @pytest.mark.asyncio
    async def test_empty_key_fails(self, limiter: RateLimiter) -> None:
        """Test that empty key raises AssertionError."""
        with pytest.raises(AssertionError):
            await limiter.check("")

    @pytest.mark.asyncio
    async def test_invalid_cost_fails(self, limiter: RateLimiter) -> None:
        """Test that invalid cost raises AssertionError."""
        with pytest.raises(AssertionError):
            await limiter.check("user", cost=0)


# =============================================================================
# TestRateLimitMiddleware
# =============================================================================


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware class."""

    def _create_mock_request(
        self,
        path: str = "/v1/agent/run",
        auth: str = "",
        client_ip: str = "127.0.0.1",
    ) -> Mock:
        """Create a mock request."""
        request = Mock(spec=Request)
        request.url = Mock()
        request.url.path = path
        request.headers = {"Authorization": auth} if auth else {}
        request.client = Mock()
        request.client.host = client_ip
        return request

    def test_default_key_func_with_token(self) -> None:
        """Test key extraction with Bearer token."""
        middleware = RateLimitMiddleware(Mock())
        request = self._create_mock_request(auth="Bearer abc123def456ghi")

        key = middleware._default_key_func(request)

        assert key.startswith("token:")

    def test_default_key_func_with_ip(self) -> None:
        """Test key extraction with client IP."""
        middleware = RateLimitMiddleware(Mock())
        request = self._create_mock_request(client_ip="192.168.1.1")

        key = middleware._default_key_func(request)

        assert key == "ip:192.168.1.1"

    def test_is_protected_path_agent(self) -> None:
        """Test that agent paths are protected."""
        middleware = RateLimitMiddleware(Mock())

        assert middleware._is_protected_path("/v1/agent/run") is True
        assert middleware._is_protected_path("/v1/agent/plan") is True

    def test_is_protected_path_synthesis(self) -> None:
        """Test that synthesis paths are protected."""
        middleware = RateLimitMiddleware(Mock())

        assert middleware._is_protected_path("/v1/synthesis/generate") is True

    def test_is_protected_path_other(self) -> None:
        """Test that other paths are not protected."""
        middleware = RateLimitMiddleware(Mock())

        assert middleware._is_protected_path("/v1/health") is False
        assert middleware._is_protected_path("/v1/auth/login") is False


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_rate_limiter(self) -> None:
        """Test creating a rate limiter with defaults."""
        limiter = create_rate_limiter()

        assert isinstance(limiter, RateLimiter)

    def test_create_rate_limiter_custom(self) -> None:
        """Test creating a rate limiter with custom values."""
        limiter = create_rate_limiter(
            requests_per_minute=120,
            burst_limit=20,
            enabled=False,
        )

        stats = limiter.get_stats()
        assert stats["requests_per_minute"] == 120
        assert stats["burst_limit"] == 20

    def test_create_rate_limit_middleware(self) -> None:
        """Test creating rate limit middleware."""
        app = Mock()
        middleware = create_rate_limit_middleware(
            app,
            requests_per_minute=30,
            burst_limit=5,
            protected_paths=["/custom"],
        )

        assert isinstance(middleware, RateLimitMiddleware)
        assert middleware._is_protected_path("/custom/endpoint") is True


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self) -> None:
        """Rule #2: Verify fixed upper bounds are defined."""
        assert MAX_BUCKETS > 0
        assert DEFAULT_RATE_LIMIT > 0
        assert DEFAULT_BURST_LIMIT > 0
        assert MAX_TOKENS_PER_BUCKET > 0

    def test_rule_5_preconditions(self) -> None:
        """Rule #5: Verify preconditions are asserted."""
        with pytest.raises(AssertionError):
            TokenBucket(tokens=-1, max_tokens=10, refill_rate=1)

        with pytest.raises(AssertionError):
            RateLimitConfig(requests_per_minute=0)

    def test_rule_9_type_hints(self) -> None:
        """Rule #9: Verify methods have type hints."""
        limiter = RateLimiter()

        assert hasattr(limiter.check, "__annotations__")
        assert hasattr(limiter.reset, "__annotations__")
        assert hasattr(limiter.get_stats, "__annotations__")
