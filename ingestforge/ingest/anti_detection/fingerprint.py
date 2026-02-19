#!/usr/bin/env python3
"""Browser fingerprint generation and referer management.

Generates realistic browser fingerprints and manages referer headers.
"""

import random
from typing import Optional
from urllib.parse import urlparse

from ingestforge.ingest.anti_detection.models import (
    BrowserType,
    RequestFingerprint,
    UserAgent,
)


class FingerprintGenerator:
    """Generates browser fingerprints for requests."""

    ACCEPT_LANGUAGES = [
        "en-US,en;q=0.9",
        "en-GB,en;q=0.9",
        "en-US,en;q=0.9,es;q=0.8",
        "en-US,en;q=0.9,fr;q=0.8",
    ]

    def __init__(self, send_dnt: bool = False) -> None:
        self.send_dnt = send_dnt

    def generate(self, user_agent: UserAgent) -> RequestFingerprint:
        """Generate a fingerprint matching the user agent."""
        is_chrome = user_agent.browser in [
            BrowserType.CHROME,
            BrowserType.MOBILE_CHROME,
            BrowserType.EDGE,
        ]
        is_firefox = user_agent.browser == BrowserType.FIREFOX

        # Chrome-specific headers
        sec_ch_ua = ""
        sec_ch_ua_mobile = ""
        sec_ch_ua_platform = ""

        if is_chrome:
            version = user_agent.browser_version.split(".")[0]
            if user_agent.browser == BrowserType.EDGE:
                sec_ch_ua = f'"Not_A Brand";v="8", "Chromium";v="{version}", "Microsoft Edge";v="{version}"'
            else:
                sec_ch_ua = f'"Not_A Brand";v="8", "Chromium";v="{version}", "Google Chrome";v="{version}"'

            sec_ch_ua_mobile = "?1" if user_agent.is_mobile else "?0"

            platform_map = {
                "Windows": '"Windows"',
                "macOS": '"macOS"',
                "Linux": '"Linux"',
                "Android": '"Android"',
                "iOS": '"iOS"',
            }
            sec_ch_ua_platform = platform_map.get(user_agent.os, '"Unknown"')

        # Accept header varies by request type
        accept = "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8"

        # Accept-Encoding
        accept_encoding = "gzip, deflate, br"
        if is_firefox:
            accept_encoding = "gzip, deflate, br, zstd"

        return RequestFingerprint(
            user_agent=user_agent.string,
            accept_language=random.choice(self.ACCEPT_LANGUAGES),
            accept_encoding=accept_encoding,
            accept=accept,
            connection="keep-alive",
            cache_control="max-age=0" if random.random() > 0.5 else "",
            sec_fetch_mode="navigate" if is_chrome else "",
            sec_fetch_site="none" if is_chrome else "",
            sec_fetch_dest="document" if is_chrome else "",
            sec_ch_ua=sec_ch_ua,
            sec_ch_ua_mobile=sec_ch_ua_mobile,
            sec_ch_ua_platform=sec_ch_ua_platform,
            dnt="1" if self.send_dnt else "",
        )


class RefererManager:
    """Manages referer headers."""

    def __init__(self, policy: str = "same-origin") -> None:
        self.policy = policy
        self._last_url: dict[str, str] = {}  # domain -> last URL

    def get_referer(self, url: str, previous_url: Optional[str] = None) -> str:
        """Get appropriate referer for URL."""
        if self.policy == "none":
            return ""

        parsed = urlparse(url)
        domain = parsed.netloc

        if previous_url:
            prev_parsed = urlparse(previous_url)

            if self.policy == "same-origin":
                if prev_parsed.netloc == domain:
                    return previous_url
                return ""

            elif self.policy == "cross-origin":
                return previous_url

        # Use last URL for this domain if no previous URL
        return self._last_url.get(domain, "")

    def update_last_url(self, url: str) -> None:
        """Update last visited URL for domain."""
        parsed = urlparse(url)
        self._last_url[parsed.netloc] = url
