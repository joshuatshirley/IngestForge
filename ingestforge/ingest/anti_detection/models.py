#!/usr/bin/env python3
"""Data models for anti-detection handling.

Defines enums and dataclasses for user agents, fingerprints, delays, proxies,
and anti-detection configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class BrowserType(Enum):
    """Browser types for user agent generation."""

    CHROME = "chrome"
    FIREFOX = "firefox"
    SAFARI = "safari"
    EDGE = "edge"
    MOBILE_CHROME = "mobile_chrome"
    MOBILE_SAFARI = "mobile_safari"


class DeviceType(Enum):
    """Device types for fingerprinting."""

    DESKTOP = "desktop"
    MOBILE = "mobile"
    TABLET = "tablet"


@dataclass
class UserAgent:
    """User agent information."""

    string: str
    browser: BrowserType
    browser_version: str
    os: str
    os_version: str
    device: DeviceType
    is_mobile: bool = False


@dataclass
class RequestFingerprint:
    """Browser fingerprint for a request."""

    user_agent: str
    accept_language: str
    accept_encoding: str
    accept: str
    connection: str
    cache_control: str
    sec_fetch_mode: str
    sec_fetch_site: str
    sec_fetch_dest: str
    sec_ch_ua: str
    sec_ch_ua_mobile: str
    sec_ch_ua_platform: str
    dnt: str

    def to_headers(self) -> dict[str, Any]:
        """Convert to HTTP headers dictionary."""
        headers = {
            "User-Agent": self.user_agent,
            "Accept-Language": self.accept_language,
            "Accept-Encoding": self.accept_encoding,
            "Accept": self.accept,
            "Connection": self.connection,
        }

        if self.cache_control:
            headers["Cache-Control"] = self.cache_control

        if self.sec_fetch_mode:
            headers["Sec-Fetch-Mode"] = self.sec_fetch_mode
            headers["Sec-Fetch-Site"] = self.sec_fetch_site
            headers["Sec-Fetch-Dest"] = self.sec_fetch_dest

        if self.sec_ch_ua:
            headers["Sec-CH-UA"] = self.sec_ch_ua
            headers["Sec-CH-UA-Mobile"] = self.sec_ch_ua_mobile
            headers["Sec-CH-UA-Platform"] = self.sec_ch_ua_platform

        if self.dnt:
            headers["DNT"] = self.dnt

        return headers


@dataclass
class DelayConfig:
    """Configuration for request delays."""

    min_delay: float = 1.0  # Minimum delay in seconds
    max_delay: float = 3.0  # Maximum delay in seconds
    burst_limit: int = 5  # Max requests before longer delay
    burst_delay: float = 10.0  # Longer delay after burst
    jitter_factor: float = 0.3  # Randomization factor
    respect_robots_delay: bool = True
    robots_delay_default: float = 1.0

    def get_delay(self, request_count: int = 0) -> float:
        """Calculate delay for next request."""
        import random

        # Check if burst limit reached
        if (
            self.burst_limit > 0
            and request_count > 0
            and request_count % self.burst_limit == 0
        ):
            base_delay = self.burst_delay
        else:
            base_delay = random.uniform(self.min_delay, self.max_delay)

        # Add jitter
        jitter = base_delay * self.jitter_factor * (random.random() * 2 - 1)
        return max(0.1, base_delay + jitter)


@dataclass
class ProxyConfig:
    """Proxy configuration."""

    host: str
    port: int
    protocol: str = "http"
    username: str = ""
    password: str = ""
    country: str = ""
    is_residential: bool = False
    last_used: Optional[datetime] = None
    failure_count: int = 0
    success_count: int = 0

    @property
    def url(self) -> str:
        """Get proxy URL."""
        auth = ""
        if self.username and self.password:
            auth = f"{self.username}:{self.password}@"
        return f"{self.protocol}://{auth}{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 1.0


@dataclass
class AntiDetectionConfig:
    """Complete anti-detection configuration."""

    rotate_user_agents: bool = True
    browser_types: list[Any] = field(
        default_factory=lambda: [BrowserType.CHROME, BrowserType.FIREFOX]
    )
    device_types: list[Any] = field(default_factory=lambda: [DeviceType.DESKTOP])

    delay_config: DelayConfig = field(default_factory=DelayConfig)

    use_referer: bool = True
    referer_policy: str = "same-origin"  # same-origin, cross-origin, none

    rotate_proxies: bool = False
    proxies: list[Any] = field(default_factory=list)  # List of ProxyConfig

    randomize_fingerprint: bool = True
    send_dnt: bool = False

    session_cookies: bool = True
    max_retries_per_proxy: int = 3
