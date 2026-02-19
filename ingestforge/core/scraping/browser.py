"""Playwright-based headless browser for JS-rendered content.

Provides headless browser support for scraping React/Vue SPAs
and other JavaScript-heavy websites."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_CONCURRENT_PAGES = 10
MAX_PAGE_LOAD_TIMEOUT_MS = 30000
MAX_SCRIPT_TIMEOUT_MS = 10000
MAX_NAVIGATION_TIMEOUT_MS = 30000
MAX_CONTENT_SIZE_BYTES = 50 * 1024 * 1024  # 50MB


class BrowserType(str, Enum):
    """Supported browser types."""

    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


class PageStatus(str, Enum):
    """Page load status."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"
    BLOCKED = "blocked"


@dataclass
class BrowserConfig:
    """Configuration for headless browser."""

    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    timeout_ms: int = MAX_PAGE_LOAD_TIMEOUT_MS
    user_agent: str = ""
    viewport_width: int = 1920
    viewport_height: int = 1080
    proxy_server: str = ""
    ignore_https_errors: bool = False
    block_images: bool = True
    block_fonts: bool = True


@dataclass
class PageResult:
    """Result from page fetch."""

    url: str
    status: PageStatus
    content: str = ""
    html: str = ""
    title: str = ""
    error: str = ""
    load_time_ms: float = 0.0
    final_url: str = ""

    @property
    def success(self) -> bool:
        """Check if fetch was successful."""
        return self.status == PageStatus.SUCCESS


@dataclass
class BrowserContext:
    """Browser context for managing pages."""

    browser: object = None  # playwright.async_api.Browser
    context: object = None  # playwright.async_api.BrowserContext
    pages: List[object] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        """Check if context is active."""
        return self.browser is not None and self.context is not None


class BrowserManager:
    """Manages headless browser instances for scraping.

    Provides context management for proper resource cleanup
    per Rule #3 (No Dynamic Memory After Init).
    """

    def __init__(self, config: Optional[BrowserConfig] = None) -> None:
        """Initialize browser manager.

        Args:
            config: Browser configuration
        """
        self.config = config or BrowserConfig()
        self._playwright: Optional[object] = None
        self._context: Optional[BrowserContext] = None
        self._lock = asyncio.Lock()

    async def start(self) -> bool:
        """Start the browser.

        Returns:
            True if started successfully
        """
        async with self._lock:
            if self._playwright is not None:
                return True

            try:
                # Lazy import playwright
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                browser = await self._launch_browser()
                context = await self._create_context(browser)

                self._context = BrowserContext(browser=browser, context=context)

                logger.info(f"Browser started: {self.config.browser_type}")
                return True

            except ImportError:
                logger.error("Playwright not installed. Run: pip install playwright")
                return False
            except Exception as e:
                logger.exception(
                    f"Failed to start browser ({self.config.browser_type}): {e}"
                )
                return False

    async def stop(self) -> None:
        """Stop the browser and clean up resources."""
        async with self._lock:
            if self._context and self._context.context:
                try:
                    await self._context.context.close()
                except Exception as e:
                    logger.exception(
                        f"Error closing browser context during cleanup: {e}"
                    )

            if self._context and self._context.browser:
                try:
                    await self._context.browser.close()
                except Exception as e:
                    logger.exception(
                        f"Error closing browser instance during cleanup: {e}"
                    )

            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception as e:
                    logger.exception(f"Error stopping playwright during cleanup: {e}")

            self._playwright = None
            self._context = None
            logger.info("Browser stopped")

    async def fetch_page(self, url: str) -> PageResult:
        """Fetch a page with JavaScript rendering.

        Args:
            url: URL to fetch

        Returns:
            PageResult with content
        """
        if not url:
            return PageResult(url=url, status=PageStatus.ERROR, error="Empty URL")

        if not self._context or not self._context.is_active:
            started = await self.start()
            if not started:
                return PageResult(
                    url=url,
                    status=PageStatus.ERROR,
                    error="Browser not available",
                )

        try:
            return await self._fetch_with_page(url)
        except Exception as e:
            logger.exception(f"Error fetching {url}: {e}")
            return PageResult(url=url, status=PageStatus.ERROR, error=str(e))

    async def _launch_browser(self) -> object:
        """Launch browser based on config.

        Returns:
            Browser instance
        """
        launcher = getattr(self._playwright, self.config.browser_type.value)
        return await launcher.launch(headless=self.config.headless)

    async def _create_context(self, browser: object) -> object:
        """Create browser context with config.

        Args:
            browser: Browser instance

        Returns:
            Browser context
        """
        options = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height,
            },
            "ignore_https_errors": self.config.ignore_https_errors,
        }

        if self.config.user_agent:
            options["user_agent"] = self.config.user_agent

        if self.config.proxy_server:
            options["proxy"] = {"server": self.config.proxy_server}

        return await browser.new_context(**options)

    async def _fetch_with_page(self, url: str) -> PageResult:
        """Fetch URL with a new page.

        Args:
            url: URL to fetch

        Returns:
            PageResult
        """
        import time

        page = await self._context.context.new_page()

        try:
            # Block resources if configured
            if self.config.block_images or self.config.block_fonts:
                await self._setup_resource_blocking(page)

            # Navigate to page
            start_time = time.time()
            response = await page.goto(
                url,
                timeout=self.config.timeout_ms,
                wait_until="networkidle",
            )
            load_time = (time.time() - start_time) * 1000

            # Check response status
            if response is None:
                return PageResult(
                    url=url,
                    status=PageStatus.ERROR,
                    error="No response received",
                    load_time_ms=load_time,
                )

            if response.status >= 400:
                return PageResult(
                    url=url,
                    status=PageStatus.BLOCKED,
                    error=f"HTTP {response.status}",
                    load_time_ms=load_time,
                    final_url=page.url,
                )

            # Get content
            html = await page.content()
            content = await page.evaluate("() => document.body.innerText")
            title = await page.title()

            return PageResult(
                url=url,
                status=PageStatus.SUCCESS,
                content=content[:MAX_CONTENT_SIZE_BYTES],
                html=html[:MAX_CONTENT_SIZE_BYTES],
                title=title,
                load_time_ms=load_time,
                final_url=page.url,
            )

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                logger.warning(f"Page load timeout for {url}: {error_msg}")
                return PageResult(url=url, status=PageStatus.TIMEOUT, error=error_msg)
            logger.exception(f"Error loading page {url}: {e}")
            return PageResult(url=url, status=PageStatus.ERROR, error=error_msg)

        finally:
            await page.close()

    async def _setup_resource_blocking(self, page: object) -> None:
        """Set up resource blocking for faster loads.

        Args:
            page: Page to configure
        """
        blocked_types = []
        if self.config.block_images:
            blocked_types.extend(["image", "media"])
        if self.config.block_fonts:
            blocked_types.append("font")

        if blocked_types:
            await page.route(
                "**/*",
                lambda route: route.abort()
                if route.request.resource_type in blocked_types
                else route.continue_(),
            )

    @asynccontextmanager
    async def session(self) -> AsyncIterator["BrowserManager"]:
        """Context manager for browser session.

        Yields:
            Self for use in async with block
        """
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


def create_browser_manager(
    headless: bool = True,
    timeout_ms: int = MAX_PAGE_LOAD_TIMEOUT_MS,
    block_images: bool = True,
) -> BrowserManager:
    """Factory function to create browser manager.

    Args:
        headless: Run in headless mode
        timeout_ms: Page load timeout
        block_images: Block image loading

    Returns:
        Configured BrowserManager
    """
    config = BrowserConfig(
        headless=headless,
        timeout_ms=timeout_ms,
        block_images=block_images,
    )
    return BrowserManager(config=config)


async def fetch_with_js(
    url: str, timeout_ms: int = MAX_PAGE_LOAD_TIMEOUT_MS
) -> PageResult:
    """Convenience function to fetch a URL with JS rendering.

    Args:
        url: URL to fetch
        timeout_ms: Timeout in milliseconds

    Returns:
        PageResult
    """
    manager = create_browser_manager(timeout_ms=timeout_ms)

    async with manager.session():
        return await manager.fetch_page(url)
