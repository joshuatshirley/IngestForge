#!/usr/bin/env python3
"""User agent database for anti-detection.

Contains user agent strings for various browsers and devices.
"""

from ingestforge.ingest.anti_detection.models import BrowserType, DeviceType, UserAgent


# User Agent Database
USER_AGENTS = {
    BrowserType.CHROME: [
        UserAgent(
            string="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            browser=BrowserType.CHROME,
            browser_version="120.0.0.0",
            os="Windows",
            os_version="10",
            device=DeviceType.DESKTOP,
        ),
        UserAgent(
            string="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
            browser=BrowserType.CHROME,
            browser_version="119.0.0.0",
            os="Windows",
            os_version="10",
            device=DeviceType.DESKTOP,
        ),
        UserAgent(
            string="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            browser=BrowserType.CHROME,
            browser_version="120.0.0.0",
            os="macOS",
            os_version="10.15.7",
            device=DeviceType.DESKTOP,
        ),
        UserAgent(
            string="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            browser=BrowserType.CHROME,
            browser_version="120.0.0.0",
            os="Linux",
            os_version="",
            device=DeviceType.DESKTOP,
        ),
    ],
    BrowserType.FIREFOX: [
        UserAgent(
            string="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            browser=BrowserType.FIREFOX,
            browser_version="121.0",
            os="Windows",
            os_version="10",
            device=DeviceType.DESKTOP,
        ),
        UserAgent(
            string="Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
            browser=BrowserType.FIREFOX,
            browser_version="121.0",
            os="macOS",
            os_version="10.15",
            device=DeviceType.DESKTOP,
        ),
        UserAgent(
            string="Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
            browser=BrowserType.FIREFOX,
            browser_version="121.0",
            os="Linux",
            os_version="",
            device=DeviceType.DESKTOP,
        ),
    ],
    BrowserType.SAFARI: [
        UserAgent(
            string="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
            browser=BrowserType.SAFARI,
            browser_version="17.2",
            os="macOS",
            os_version="10.15.7",
            device=DeviceType.DESKTOP,
        ),
    ],
    BrowserType.EDGE: [
        UserAgent(
            string="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            browser=BrowserType.EDGE,
            browser_version="120.0.0.0",
            os="Windows",
            os_version="10",
            device=DeviceType.DESKTOP,
        ),
    ],
    BrowserType.MOBILE_CHROME: [
        UserAgent(
            string="Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            browser=BrowserType.MOBILE_CHROME,
            browser_version="120.0.0.0",
            os="Android",
            os_version="13",
            device=DeviceType.MOBILE,
            is_mobile=True,
        ),
        UserAgent(
            string="Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
            browser=BrowserType.MOBILE_CHROME,
            browser_version="120.0.0.0",
            os="Android",
            os_version="12",
            device=DeviceType.MOBILE,
            is_mobile=True,
        ),
    ],
    BrowserType.MOBILE_SAFARI: [
        UserAgent(
            string="Mozilla/5.0 (iPhone; CPU iPhone OS 17_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1",
            browser=BrowserType.MOBILE_SAFARI,
            browser_version="17.2",
            os="iOS",
            os_version="17.2",
            device=DeviceType.MOBILE,
            is_mobile=True,
        ),
    ],
}
