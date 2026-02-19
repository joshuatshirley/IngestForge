"""Async selector waiting strategies for dynamic content.

Provides configurable waiting strategies to handle data loading
latency in JavaScript-heavy pages."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_WAIT_TIMEOUT_MS = 30000
MAX_POLL_INTERVAL_MS = 500
MIN_POLL_INTERVAL_MS = 50
MAX_SELECTORS = 20
DEFAULT_TIMEOUT_MS = 10000


class WaitCondition(str, Enum):
    """Types of wait conditions."""

    SELECTOR_VISIBLE = "visible"  # Element is visible
    SELECTOR_HIDDEN = "hidden"  # Element is hidden/removed
    SELECTOR_PRESENT = "present"  # Element exists in DOM
    TEXT_CONTAINS = "text_contains"  # Text appears in element
    ATTRIBUTE_EQUALS = "attr_equals"  # Attribute has value
    NETWORK_IDLE = "network_idle"  # No network activity
    CUSTOM = "custom"  # Custom predicate


class WaitStatus(str, Enum):
    """Wait operation status."""

    SUCCESS = "success"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass
class WaitResult:
    """Result from wait operation."""

    status: WaitStatus
    condition: WaitCondition
    elapsed_ms: float = 0.0
    matched_selector: str = ""
    error: str = ""

    @property
    def success(self) -> bool:
        """Check if wait was successful."""
        return self.status == WaitStatus.SUCCESS


@dataclass
class WaitConfig:
    """Configuration for wait strategy."""

    timeout_ms: int = DEFAULT_TIMEOUT_MS
    poll_interval_ms: int = 100
    retry_on_error: bool = True
    max_retries: int = 3


@dataclass
class SelectorSpec:
    """Specification for a selector to wait for."""

    selector: str
    condition: WaitCondition = WaitCondition.SELECTOR_VISIBLE
    text_match: str = ""
    attribute_name: str = ""
    attribute_value: str = ""


class WaitStrategy:
    """Implements waiting strategies for dynamic content.

    Provides methods to wait for selectors, text content,
    and custom conditions with configurable timeouts.
    """

    def __init__(self, config: Optional[WaitConfig] = None) -> None:
        """Initialize wait strategy.

        Args:
            config: Wait configuration
        """
        self.config = config or WaitConfig()
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration bounds."""
        if self.config.timeout_ms > MAX_WAIT_TIMEOUT_MS:
            self.config.timeout_ms = MAX_WAIT_TIMEOUT_MS
        if self.config.poll_interval_ms > MAX_POLL_INTERVAL_MS:
            self.config.poll_interval_ms = MAX_POLL_INTERVAL_MS
        if self.config.poll_interval_ms < MIN_POLL_INTERVAL_MS:
            self.config.poll_interval_ms = MIN_POLL_INTERVAL_MS

    async def wait_for_selector(
        self,
        page: object,
        selector: str,
        condition: WaitCondition = WaitCondition.SELECTOR_VISIBLE,
    ) -> WaitResult:
        """Wait for a selector to match condition.

        Args:
            page: Playwright page object
            selector: CSS selector
            condition: Wait condition

        Returns:
            WaitResult
        """
        if not selector:
            return WaitResult(
                status=WaitStatus.ERROR,
                condition=condition,
                error="Empty selector",
            )

        start_time = time.time()

        try:
            if condition == WaitCondition.SELECTOR_VISIBLE:
                await page.wait_for_selector(
                    selector,
                    state="visible",
                    timeout=self.config.timeout_ms,
                )
            elif condition == WaitCondition.SELECTOR_HIDDEN:
                await page.wait_for_selector(
                    selector,
                    state="hidden",
                    timeout=self.config.timeout_ms,
                )
            elif condition == WaitCondition.SELECTOR_PRESENT:
                await page.wait_for_selector(
                    selector,
                    state="attached",
                    timeout=self.config.timeout_ms,
                )
            else:
                return WaitResult(
                    status=WaitStatus.ERROR,
                    condition=condition,
                    error=f"Unsupported condition for selector: {condition}",
                )

            elapsed = (time.time() - start_time) * 1000

            return WaitResult(
                status=WaitStatus.SUCCESS,
                condition=condition,
                elapsed_ms=elapsed,
                matched_selector=selector,
            )

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            error_msg = str(e)

            if "timeout" in error_msg.lower():
                return WaitResult(
                    status=WaitStatus.TIMEOUT,
                    condition=condition,
                    elapsed_ms=elapsed,
                    error=f"Timeout waiting for {selector}",
                )

            return WaitResult(
                status=WaitStatus.ERROR,
                condition=condition,
                elapsed_ms=elapsed,
                error=error_msg,
            )

    async def wait_for_text(
        self,
        page: object,
        selector: str,
        text: str,
    ) -> WaitResult:
        """Wait for text to appear in element.

        Args:
            page: Playwright page object
            selector: CSS selector
            text: Text to wait for

        Returns:
            WaitResult
        """
        if not selector or not text:
            return WaitResult(
                status=WaitStatus.ERROR,
                condition=WaitCondition.TEXT_CONTAINS,
                error="Missing selector or text",
            )

        start_time = time.time()
        end_time = start_time + (self.config.timeout_ms / 1000)

        while time.time() < end_time:
            try:
                element = await page.query_selector(selector)
                if element:
                    content = await element.text_content()
                    if content and text in content:
                        elapsed = (time.time() - start_time) * 1000
                        return WaitResult(
                            status=WaitStatus.SUCCESS,
                            condition=WaitCondition.TEXT_CONTAINS,
                            elapsed_ms=elapsed,
                            matched_selector=selector,
                        )
            except Exception as e:
                logger.debug(f"Polling for text '{text}' in selector '{selector}': {e}")

            await asyncio.sleep(self.config.poll_interval_ms / 1000)

        elapsed = (time.time() - start_time) * 1000
        return WaitResult(
            status=WaitStatus.TIMEOUT,
            condition=WaitCondition.TEXT_CONTAINS,
            elapsed_ms=elapsed,
            error=f"Text '{text}' not found in {selector}",
        )

    async def wait_for_attribute(
        self,
        page: object,
        selector: str,
        attribute: str,
        value: str,
    ) -> WaitResult:
        """Wait for attribute to have value.

        Args:
            page: Playwright page object
            selector: CSS selector
            attribute: Attribute name
            value: Expected value

        Returns:
            WaitResult
        """
        if not selector or not attribute:
            return WaitResult(
                status=WaitStatus.ERROR,
                condition=WaitCondition.ATTRIBUTE_EQUALS,
                error="Missing selector or attribute",
            )

        start_time = time.time()
        end_time = start_time + (self.config.timeout_ms / 1000)

        while time.time() < end_time:
            try:
                element = await page.query_selector(selector)
                if element:
                    attr_value = await element.get_attribute(attribute)
                    if attr_value == value:
                        elapsed = (time.time() - start_time) * 1000
                        return WaitResult(
                            status=WaitStatus.SUCCESS,
                            condition=WaitCondition.ATTRIBUTE_EQUALS,
                            elapsed_ms=elapsed,
                            matched_selector=selector,
                        )
            except Exception as e:
                logger.debug(
                    f"Polling for attribute '{attribute}={value}' in selector '{selector}': {e}"
                )

            await asyncio.sleep(self.config.poll_interval_ms / 1000)

        elapsed = (time.time() - start_time) * 1000
        return WaitResult(
            status=WaitStatus.TIMEOUT,
            condition=WaitCondition.ATTRIBUTE_EQUALS,
            elapsed_ms=elapsed,
            error=f"Attribute {attribute}={value} not found",
        )

    async def wait_for_network_idle(
        self,
        page: object,
        idle_time_ms: int = 500,
    ) -> WaitResult:
        """Wait for network to become idle.

        Args:
            page: Playwright page object
            idle_time_ms: Time without requests to consider idle

        Returns:
            WaitResult
        """
        start_time = time.time()

        try:
            await page.wait_for_load_state(
                "networkidle",
                timeout=self.config.timeout_ms,
            )
            elapsed = (time.time() - start_time) * 1000

            return WaitResult(
                status=WaitStatus.SUCCESS,
                condition=WaitCondition.NETWORK_IDLE,
                elapsed_ms=elapsed,
            )

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            error_msg = str(e)

            if "timeout" in error_msg.lower():
                return WaitResult(
                    status=WaitStatus.TIMEOUT,
                    condition=WaitCondition.NETWORK_IDLE,
                    elapsed_ms=elapsed,
                    error="Network did not become idle",
                )

            return WaitResult(
                status=WaitStatus.ERROR,
                condition=WaitCondition.NETWORK_IDLE,
                elapsed_ms=elapsed,
                error=error_msg,
            )

    async def wait_for_any(
        self,
        page: object,
        specs: List[SelectorSpec],
    ) -> WaitResult:
        """Wait for any of multiple selectors.

        Args:
            page: Playwright page object
            specs: List of selector specifications

        Returns:
            WaitResult for first matching selector
        """
        if not specs:
            return WaitResult(
                status=WaitStatus.ERROR,
                condition=WaitCondition.SELECTOR_VISIBLE,
                error="No selectors provided",
            )

        # Limit number of selectors
        specs = specs[:MAX_SELECTORS]

        start_time = time.time()
        end_time = start_time + (self.config.timeout_ms / 1000)

        while time.time() < end_time:
            for spec in specs:
                try:
                    element = await page.query_selector(spec.selector)
                    if element and await self._check_spec(element, spec):
                        elapsed = (time.time() - start_time) * 1000
                        return WaitResult(
                            status=WaitStatus.SUCCESS,
                            condition=spec.condition,
                            elapsed_ms=elapsed,
                            matched_selector=spec.selector,
                        )
                except Exception as e:
                    logger.debug(
                        f"Polling selector '{spec.selector}' with condition '{spec.condition}': {e}"
                    )

            await asyncio.sleep(self.config.poll_interval_ms / 1000)

        elapsed = (time.time() - start_time) * 1000
        return WaitResult(
            status=WaitStatus.TIMEOUT,
            condition=WaitCondition.SELECTOR_VISIBLE,
            elapsed_ms=elapsed,
            error="No selectors matched",
        )

    async def _check_spec(self, element: object, spec: SelectorSpec) -> bool:
        """Check if element matches specification.

        Args:
            element: DOM element
            spec: Selector specification

        Returns:
            True if matches
        """
        if spec.condition == WaitCondition.SELECTOR_VISIBLE:
            return await element.is_visible()

        if spec.condition == WaitCondition.TEXT_CONTAINS:
            text = await element.text_content()
            return text and spec.text_match in text

        if spec.condition == WaitCondition.ATTRIBUTE_EQUALS:
            value = await element.get_attribute(spec.attribute_name)
            return value == spec.attribute_value

        return True  # SELECTOR_PRESENT


def wait_for_selector(
    page: object,
    selector: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> WaitResult:
    """Synchronous wrapper for wait_for_selector.

    Args:
        page: Playwright page object
        selector: CSS selector
        timeout_ms: Timeout in milliseconds

    Returns:
        WaitResult
    """
    config = WaitConfig(timeout_ms=timeout_ms)
    strategy = WaitStrategy(config=config)
    return asyncio.get_event_loop().run_until_complete(
        strategy.wait_for_selector(page, selector)
    )


async def wait_for_content(
    page: object,
    selector: str,
    text: str,
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
) -> WaitResult:
    """Convenience function to wait for text content.

    Args:
        page: Playwright page object
        selector: CSS selector
        text: Text to wait for
        timeout_ms: Timeout in milliseconds

    Returns:
        WaitResult
    """
    config = WaitConfig(timeout_ms=timeout_ms)
    strategy = WaitStrategy(config=config)
    return await strategy.wait_for_text(page, selector, text)
