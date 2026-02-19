"""Tests for Playwright browser integration.

Tests browser management and page fetching."""

from __future__ import annotations

import pytest

from ingestforge.core.scraping.browser import (
    BrowserConfig,
    BrowserContext,
    BrowserManager,
    BrowserType,
    PageResult,
    PageStatus,
    create_browser_manager,
    MAX_PAGE_LOAD_TIMEOUT_MS,
)

# BrowserType tests


class TestBrowserType:
    """Tests for BrowserType enum."""

    def test_browser_types_defined(self) -> None:
        """Test all browser types are defined."""
        types = [t.value for t in BrowserType]

        assert "chromium" in types
        assert "firefox" in types
        assert "webkit" in types


# PageStatus tests


class TestPageStatus:
    """Tests for PageStatus enum."""

    def test_statuses_defined(self) -> None:
        """Test all statuses are defined."""
        statuses = [s.value for s in PageStatus]

        assert "success" in statuses
        assert "timeout" in statuses
        assert "error" in statuses
        assert "blocked" in statuses


# BrowserConfig tests


class TestBrowserConfig:
    """Tests for BrowserConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BrowserConfig()

        assert config.browser_type == BrowserType.CHROMIUM
        assert config.headless is True
        assert config.timeout_ms == MAX_PAGE_LOAD_TIMEOUT_MS
        assert config.viewport_width == 1920
        assert config.viewport_height == 1080

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BrowserConfig(
            browser_type=BrowserType.FIREFOX,
            headless=False,
            timeout_ms=5000,
            user_agent="Test Agent",
        )

        assert config.browser_type == BrowserType.FIREFOX
        assert config.headless is False
        assert config.timeout_ms == 5000
        assert config.user_agent == "Test Agent"


# PageResult tests


class TestPageResult:
    """Tests for PageResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        result = PageResult(
            url="https://example.com",
            status=PageStatus.SUCCESS,
            content="Page content",
            html="<html>...</html>",
            title="Example",
            load_time_ms=500.0,
        )

        assert result.success is True
        assert result.content == "Page content"
        assert result.title == "Example"

    def test_timeout_result(self) -> None:
        """Test timeout result."""
        result = PageResult(
            url="https://example.com",
            status=PageStatus.TIMEOUT,
            error="Page load timed out",
        )

        assert result.success is False
        assert result.status == PageStatus.TIMEOUT

    def test_error_result(self) -> None:
        """Test error result."""
        result = PageResult(
            url="https://example.com",
            status=PageStatus.ERROR,
            error="Network error",
        )

        assert result.success is False
        assert "Network error" in result.error

    def test_blocked_result(self) -> None:
        """Test blocked result."""
        result = PageResult(
            url="https://example.com",
            status=PageStatus.BLOCKED,
            error="HTTP 403",
        )

        assert result.success is False
        assert result.status == PageStatus.BLOCKED


# BrowserContext tests


class TestBrowserContext:
    """Tests for BrowserContext dataclass."""

    def test_empty_context(self) -> None:
        """Test empty context."""
        context = BrowserContext()

        assert context.browser is None
        assert context.context is None
        assert context.is_active is False

    def test_active_context(self) -> None:
        """Test active context."""
        context = BrowserContext(
            browser=object(),  # Mock browser
            context=object(),  # Mock context
        )

        assert context.is_active is True


# BrowserManager tests


class TestBrowserManager:
    """Tests for BrowserManager."""

    def test_manager_creation(self) -> None:
        """Test creating manager."""
        manager = BrowserManager()

        assert manager.config is not None
        assert manager.config.headless is True

    def test_manager_with_config(self) -> None:
        """Test manager with custom config."""
        config = BrowserConfig(timeout_ms=5000)
        manager = BrowserManager(config=config)

        assert manager.config.timeout_ms == 5000

    def test_manager_not_started_initially(self) -> None:
        """Test manager is not started initially."""
        manager = BrowserManager()

        assert manager._playwright is None
        assert manager._context is None

    @pytest.mark.asyncio
    async def test_fetch_empty_url(self) -> None:
        """Test fetching empty URL returns error."""
        manager = BrowserManager()

        result = await manager.fetch_page("")

        assert result.success is False
        assert result.status == PageStatus.ERROR
        assert "Empty" in result.error


# Factory function tests


class TestCreateBrowserManager:
    """Tests for create_browser_manager factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        manager = create_browser_manager()

        assert manager.config.headless is True
        assert manager.config.block_images is True

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        manager = create_browser_manager(
            headless=False,
            timeout_ms=5000,
            block_images=False,
        )

        assert manager.config.headless is False
        assert manager.config.timeout_ms == 5000
        assert manager.config.block_images is False


# Note: Integration tests with actual Playwright would require
# the playwright package and browser to be installed.
# These are unit tests that verify the structure and logic.


class TestBrowserManagerResourceBlocking:
    """Tests for resource blocking configuration."""

    def test_block_images_default(self) -> None:
        """Test images blocked by default."""
        config = BrowserConfig()

        assert config.block_images is True

    def test_block_fonts_default(self) -> None:
        """Test fonts blocked by default."""
        config = BrowserConfig()

        assert config.block_fonts is True

    def test_disable_blocking(self) -> None:
        """Test disabling resource blocking."""
        config = BrowserConfig(block_images=False, block_fonts=False)

        assert config.block_images is False
        assert config.block_fonts is False


class TestProxyConfiguration:
    """Tests for proxy configuration."""

    def test_no_proxy_default(self) -> None:
        """Test no proxy by default."""
        config = BrowserConfig()

        assert config.proxy_server == ""

    def test_proxy_configured(self) -> None:
        """Test proxy configuration."""
        config = BrowserConfig(proxy_server="http://proxy:8080")

        assert config.proxy_server == "http://proxy:8080"
