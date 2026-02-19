"""Tests for async selector waiting strategies.

Tests wait conditions and timeout handling."""

from __future__ import annotations

import pytest

from ingestforge.core.scraping.wait_strategy import (
    SelectorSpec,
    WaitCondition,
    WaitConfig,
    WaitResult,
    WaitStatus,
    WaitStrategy,
    MAX_WAIT_TIMEOUT_MS,
    MAX_POLL_INTERVAL_MS,
    MIN_POLL_INTERVAL_MS,
    DEFAULT_TIMEOUT_MS,
)

# WaitCondition tests


class TestWaitCondition:
    """Tests for WaitCondition enum."""

    def test_conditions_defined(self) -> None:
        """Test all wait conditions are defined."""
        conditions = [c.value for c in WaitCondition]

        assert "visible" in conditions
        assert "hidden" in conditions
        assert "present" in conditions
        assert "text_contains" in conditions
        assert "attr_equals" in conditions
        assert "network_idle" in conditions
        assert "custom" in conditions


# WaitStatus tests


class TestWaitStatus:
    """Tests for WaitStatus enum."""

    def test_statuses_defined(self) -> None:
        """Test all statuses are defined."""
        statuses = [s.value for s in WaitStatus]

        assert "success" in statuses
        assert "timeout" in statuses
        assert "error" in statuses


# WaitResult tests


class TestWaitResult:
    """Tests for WaitResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = WaitResult(
            status=WaitStatus.SUCCESS,
            condition=WaitCondition.SELECTOR_VISIBLE,
            elapsed_ms=150.0,
            matched_selector=".my-element",
        )

        assert result.success is True
        assert result.condition == WaitCondition.SELECTOR_VISIBLE
        assert result.matched_selector == ".my-element"

    def test_timeout_result(self) -> None:
        """Test timeout result."""
        result = WaitResult(
            status=WaitStatus.TIMEOUT,
            condition=WaitCondition.SELECTOR_VISIBLE,
            elapsed_ms=10000.0,
            error="Element not found",
        )

        assert result.success is False
        assert result.status == WaitStatus.TIMEOUT

    def test_error_result(self) -> None:
        """Test error result."""
        result = WaitResult(
            status=WaitStatus.ERROR,
            condition=WaitCondition.SELECTOR_VISIBLE,
            error="Invalid selector",
        )

        assert result.success is False
        assert "Invalid" in result.error


# WaitConfig tests


class TestWaitConfig:
    """Tests for WaitConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = WaitConfig()

        assert config.timeout_ms == DEFAULT_TIMEOUT_MS
        assert config.poll_interval_ms == 100
        assert config.retry_on_error is True
        assert config.max_retries == 3

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = WaitConfig(
            timeout_ms=5000,
            poll_interval_ms=200,
            retry_on_error=False,
            max_retries=1,
        )

        assert config.timeout_ms == 5000
        assert config.poll_interval_ms == 200
        assert config.retry_on_error is False


# SelectorSpec tests


class TestSelectorSpec:
    """Tests for SelectorSpec dataclass."""

    def test_simple_spec(self) -> None:
        """Test simple selector spec."""
        spec = SelectorSpec(selector=".my-class")

        assert spec.selector == ".my-class"
        assert spec.condition == WaitCondition.SELECTOR_VISIBLE

    def test_text_spec(self) -> None:
        """Test text match spec."""
        spec = SelectorSpec(
            selector="#content",
            condition=WaitCondition.TEXT_CONTAINS,
            text_match="Loading complete",
        )

        assert spec.condition == WaitCondition.TEXT_CONTAINS
        assert spec.text_match == "Loading complete"

    def test_attribute_spec(self) -> None:
        """Test attribute match spec."""
        spec = SelectorSpec(
            selector="button",
            condition=WaitCondition.ATTRIBUTE_EQUALS,
            attribute_name="disabled",
            attribute_value="false",
        )

        assert spec.condition == WaitCondition.ATTRIBUTE_EQUALS
        assert spec.attribute_name == "disabled"


# WaitStrategy tests


class TestWaitStrategy:
    """Tests for WaitStrategy."""

    def test_strategy_creation(self) -> None:
        """Test creating strategy."""
        strategy = WaitStrategy()

        assert strategy.config is not None
        assert strategy.config.timeout_ms == DEFAULT_TIMEOUT_MS

    def test_strategy_with_config(self) -> None:
        """Test strategy with custom config."""
        config = WaitConfig(timeout_ms=5000)
        strategy = WaitStrategy(config=config)

        assert strategy.config.timeout_ms == 5000


class TestConfigValidation:
    """Tests for config validation."""

    def test_timeout_capped(self) -> None:
        """Test timeout is capped to max."""
        config = WaitConfig(timeout_ms=60000)  # Above max
        strategy = WaitStrategy(config=config)

        assert strategy.config.timeout_ms == MAX_WAIT_TIMEOUT_MS

    def test_poll_interval_max_capped(self) -> None:
        """Test poll interval is capped to max."""
        config = WaitConfig(poll_interval_ms=1000)  # Above max
        strategy = WaitStrategy(config=config)

        assert strategy.config.poll_interval_ms == MAX_POLL_INTERVAL_MS

    def test_poll_interval_min_enforced(self) -> None:
        """Test poll interval has minimum."""
        config = WaitConfig(poll_interval_ms=10)  # Below min
        strategy = WaitStrategy(config=config)

        assert strategy.config.poll_interval_ms == MIN_POLL_INTERVAL_MS


class TestEmptyInputHandling:
    """Tests for empty input handling."""

    @pytest.mark.asyncio
    async def test_empty_selector(self) -> None:
        """Test empty selector returns error."""
        strategy = WaitStrategy()

        result = await strategy.wait_for_selector(
            page=None,  # Would fail anyway
            selector="",
        )

        assert result.success is False
        assert result.status == WaitStatus.ERROR
        assert "Empty" in result.error

    @pytest.mark.asyncio
    async def test_empty_text_search(self) -> None:
        """Test empty text search returns error."""
        strategy = WaitStrategy()

        result = await strategy.wait_for_text(
            page=None,
            selector="",
            text="",
        )

        assert result.success is False
        assert "Missing" in result.error

    @pytest.mark.asyncio
    async def test_empty_attribute_search(self) -> None:
        """Test empty attribute search returns error."""
        strategy = WaitStrategy()

        result = await strategy.wait_for_attribute(
            page=None,
            selector="",
            attribute="",
            value="",
        )

        assert result.success is False
        assert "Missing" in result.error

    @pytest.mark.asyncio
    async def test_empty_selectors_list(self) -> None:
        """Test empty selectors list returns error."""
        strategy = WaitStrategy()

        result = await strategy.wait_for_any(
            page=None,
            specs=[],
        )

        assert result.success is False
        assert "No selectors" in result.error


class TestTimeoutBehavior:
    """Tests for timeout handling behavior."""

    def test_default_timeout(self) -> None:
        """Test default timeout value."""
        config = WaitConfig()

        assert config.timeout_ms == DEFAULT_TIMEOUT_MS
        assert config.timeout_ms == 10000

    def test_max_timeout_constant(self) -> None:
        """Test max timeout constant."""
        assert MAX_WAIT_TIMEOUT_MS == 30000

    def test_poll_interval_range(self) -> None:
        """Test poll interval bounds."""
        assert MIN_POLL_INTERVAL_MS == 50
        assert MAX_POLL_INTERVAL_MS == 500


class TestMaxSelectors:
    """Tests for selector limits."""

    @pytest.mark.asyncio
    async def test_many_selectors_limited(self) -> None:
        """Test that many selectors are limited."""
        strategy = WaitStrategy()

        # Create more than MAX_SELECTORS specs
        specs = [SelectorSpec(selector=f".element-{i}") for i in range(30)]

        # This would normally timeout but we just verify it doesn't
        # crash with too many selectors
        result = await strategy.wait_for_any(
            page=None,  # Will fail but tests the limit
            specs=specs,
        )

        # Should have attempted (up to limit) but failed due to no page
        assert result.success is False


# Note: Full integration tests with actual Playwright would require
# the playwright package and browser to be installed.
# These unit tests verify the structure, logic, and error handling.
