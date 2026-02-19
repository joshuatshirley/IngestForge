#!/usr/bin/env python3
"""User agent and proxy rotation logic.

Manages rotation of user agents and proxies for anti-detection.
"""

import hashlib
import random
from datetime import datetime
from typing import Any, Optional

from ingestforge.ingest.anti_detection.data import USER_AGENTS
from ingestforge.ingest.anti_detection.models import (
    BrowserType,
    DeviceType,
    ProxyConfig,
    UserAgent,
)


class UserAgentRotator:
    """Manages user agent rotation."""

    def __init__(
        self,
        browser_types: list[Any] = None,
        device_types: list[Any] = None,
    ):
        self.browser_types = browser_types or [BrowserType.CHROME, BrowserType.FIREFOX]
        self.device_types = device_types or [DeviceType.DESKTOP]
        self._build_pool()
        self._last_used = None
        self._usage_count = {}

    def _build_pool(self) -> None:
        """Build the user agent pool based on configuration.

        Rule #1: Reduced nesting via helper extraction
        """
        self.pool = []
        for browser in self.browser_types:
            self._add_browser_agents(browser)

        if not self.pool:
            # Fallback to a basic Chrome UA
            self.pool = [USER_AGENTS[BrowserType.CHROME][0]]

    def _add_browser_agents(self, browser: "BrowserType") -> None:
        """Add user agents for a browser type if they match device types.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if browser not in USER_AGENTS:
            return

        for ua in USER_AGENTS[browser]:
            if ua.device in self.device_types:
                self.pool.append(ua)

    def get_random(self) -> UserAgent:
        """Get a random user agent from the pool."""
        return random.choice(self.pool)

    def get_weighted(self) -> UserAgent:
        """Get user agent with lower usage getting priority."""
        if not self._usage_count:
            ua = random.choice(self.pool)
        else:
            # Prefer less-used user agents
            weights = []
            for ua in self.pool:
                count = self._usage_count.get(ua.string, 0)
                weights.append(1.0 / (count + 1))

            ua = random.choices(self.pool, weights=weights)[0]

        self._usage_count[ua.string] = self._usage_count.get(ua.string, 0) + 1
        self._last_used = ua
        return ua

    def get_consistent_for_domain(self, domain: str) -> UserAgent:
        """Get consistent user agent for a domain (session-like behavior)."""
        # Hash domain to get consistent index
        hash_val = int(hashlib.md5(domain.encode()).hexdigest(), 16)
        index = hash_val % len(self.pool)
        return self.pool[index]

    @property
    def pool_size(self) -> int:
        """Get the size of the user agent pool."""
        return len(self.pool)


class ProxyRotator:
    """Manages proxy rotation."""

    def __init__(self, proxies: list[Any] = None) -> None:
        self.proxies = proxies or []
        self._current_index = 0
        self._domain_proxy_map = {}

    def add_proxy(self, proxy: ProxyConfig) -> None:
        """Add a proxy to the rotation pool."""
        self.proxies.append(proxy)

    def remove_proxy(self, proxy: ProxyConfig) -> None:
        """Remove a proxy from the pool."""
        self.proxies = [p for p in self.proxies if p.url != proxy.url]

    def get_next(self) -> Optional[ProxyConfig]:
        """Get next proxy in rotation."""
        if not self.proxies:
            return None

        proxy = self.proxies[self._current_index]
        self._current_index = (self._current_index + 1) % len(self.proxies)
        proxy.last_used = datetime.now()
        return proxy

    def get_random(self) -> Optional[ProxyConfig]:
        """Get a random proxy."""
        if not self.proxies:
            return None
        return random.choice(self.proxies)

    def get_best(self) -> Optional[ProxyConfig]:
        """Get the proxy with highest success rate."""
        if not self.proxies:
            return None

        return max(self.proxies, key=lambda p: p.success_rate)

    def get_for_domain(self, domain: str) -> Optional[ProxyConfig]:
        """Get consistent proxy for a domain."""
        if not self.proxies:
            return None

        if domain not in self._domain_proxy_map:
            hash_val = int(hashlib.md5(domain.encode()).hexdigest(), 16)
            index = hash_val % len(self.proxies)
            self._domain_proxy_map[domain] = self.proxies[index]

        return self._domain_proxy_map[domain]

    def report_success(self, proxy: ProxyConfig) -> None:
        """Report successful use of proxy."""
        proxy.success_count += 1

    def report_failure(self, proxy: ProxyConfig) -> None:
        """Report failed use of proxy."""
        proxy.failure_count += 1

    @property
    def pool_size(self) -> int:
        return len(self.proxies)
