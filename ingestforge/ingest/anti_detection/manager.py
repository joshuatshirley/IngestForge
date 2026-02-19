#!/usr/bin/env python3
"""Main anti-detection manager and convenience functions.

Coordinates all anti-detection components and provides simple factory functions.
"""

import random
from typing import Any, Optional
from urllib.parse import urlparse

from ingestforge.ingest.anti_detection.data import USER_AGENTS
from ingestforge.ingest.anti_detection.delay import RequestDelayer
from ingestforge.ingest.anti_detection.fingerprint import (
    FingerprintGenerator,
    RefererManager,
)
from ingestforge.ingest.anti_detection.models import (
    AntiDetectionConfig,
    BrowserType,
    DelayConfig,
    DeviceType,
    ProxyConfig,
    RequestFingerprint,
)
from ingestforge.ingest.anti_detection.rotators import ProxyRotator, UserAgentRotator


class AntiDetectionManager:
    """Main manager for anti-detection measures."""

    def __init__(self, config: AntiDetectionConfig = None) -> None:
        self.config = config or AntiDetectionConfig()

        self.ua_rotator = UserAgentRotator(
            browser_types=self.config.browser_types,
            device_types=self.config.device_types,
        )

        self.fingerprint_gen = FingerprintGenerator(
            send_dnt=self.config.send_dnt,
        )

        self.proxy_rotator = ProxyRotator(
            proxies=self.config.proxies,
        )

        self.referer_manager = RefererManager(
            policy=self.config.referer_policy,
        )

        self.delayer = RequestDelayer(
            config=self.config.delay_config,
        )

        self._session_data = {}  # domain -> session info

    def _get_user_agent_for_request(self, domain: str, use_session: bool) -> Any:
        """
        Get or create user agent for request.

        Rule #1: Early return for existing session
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            domain: Target domain
            use_session: Whether to use consistent session

        Returns:
            User agent object
        """
        if use_session and domain in self._session_data:
            return self._session_data[domain]["user_agent"]

        # Determine user agent
        if self.config.rotate_user_agents:
            if use_session:
                user_agent = self.ua_rotator.get_consistent_for_domain(domain)
            else:
                user_agent = self.ua_rotator.get_weighted()
        else:
            user_agent = self.ua_rotator.pool[0]

        # Store in session
        if use_session:
            self._session_data[domain] = {"user_agent": user_agent}

        return user_agent

    def _generate_request_fingerprint(self, user_agent: Any) -> RequestFingerprint:
        """
        Generate request fingerprint.

        Rule #1: Early return for randomize
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            user_agent: User agent object

        Returns:
            RequestFingerprint
        """
        if self.config.randomize_fingerprint:
            return self.fingerprint_gen.generate(user_agent)

        # Default fingerprint
        return RequestFingerprint(
            user_agent=user_agent.string,
            accept_language="en-US,en;q=0.9",
            accept_encoding="gzip, deflate, br",
            accept="text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            connection="keep-alive",
            cache_control="",
            sec_fetch_mode="",
            sec_fetch_site="",
            sec_fetch_dest="",
            sec_ch_ua="",
            sec_ch_ua_mobile="",
            sec_ch_ua_platform="",
            dnt="",
        )

    def _add_referer_to_headers(
        self, headers: dict[str, Any], url: str, previous_url: Optional[str]
    ) -> None:
        """
        Add referer header if configured.

        Rule #1: Early return if disabled
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            headers: Headers dict to mutate
            url: Current URL
            previous_url: Previous URL
        """
        if not self.config.use_referer:
            return

        referer = self.referer_manager.get_referer(url, previous_url)
        if referer:
            headers["Referer"] = referer

    def _get_proxy_for_request(self, domain: str, use_session: bool) -> Optional[Any]:
        """
        Get proxy for request.

        Rule #1: Early return if disabled
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            domain: Target domain
            use_session: Whether to use consistent session

        Returns:
            ProxyConfig or None
        """
        if not (self.config.rotate_proxies and self.proxy_rotator.pool_size > 0):
            return None

        if use_session:
            return self.proxy_rotator.get_for_domain(domain)
        return self.proxy_rotator.get_next()

    def prepare_request(
        self,
        url: str,
        previous_url: str = None,
        use_session: bool = True,
    ) -> dict[str, Any]:
        """
        Prepare headers and settings for a request.

        Rule #1: Linear control flow
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            url: Target URL
            previous_url: Previous URL for referer
            use_session: Use consistent session for domain

        Returns:
            Dictionary with 'headers', 'proxy', and 'delay' keys
        """
        parsed = urlparse(url)
        domain = parsed.netloc

        # Get components using helpers
        user_agent = self._get_user_agent_for_request(domain, use_session)
        fingerprint = self._generate_request_fingerprint(user_agent)
        headers = fingerprint.to_headers()
        self._add_referer_to_headers(headers, url, previous_url)
        proxy = self._get_proxy_for_request(domain, use_session)
        delay = self.delayer.get_delay(domain)

        return {
            "headers": headers,
            "proxy": proxy.url if proxy else None,
            "proxy_config": proxy,
            "delay": delay,
            "user_agent": user_agent,
        }

    def wait_for_request(self, url: str) -> None:
        """Wait the appropriate delay before making a request."""
        parsed = urlparse(url)
        self.delayer.wait(parsed.netloc)

    def update_after_request(
        self, url: str, success: bool, proxy: ProxyConfig = None
    ) -> None:
        """Update state after a request."""
        # Update referer tracking
        self.referer_manager.update_last_url(url)

        # Update proxy stats
        if proxy:
            if success:
                self.proxy_rotator.report_success(proxy)
            else:
                self.proxy_rotator.report_failure(proxy)

    def clear_session(self, domain: str = None) -> None:
        """Clear session data."""
        if domain:
            self._session_data.pop(domain, None)
            self.delayer.reset_domain(domain)
        else:
            self._session_data.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get anti-detection statistics."""
        return {
            "user_agent_pool_size": self.ua_rotator.pool_size,
            "proxy_pool_size": self.proxy_rotator.pool_size,
            "active_sessions": len(self._session_data),
            "domains_tracked": list(self._session_data.keys()),
        }


# Convenience functions


def create_anti_detection_manager(
    rotate_user_agents: bool = True,
    min_delay: float = 1.0,
    max_delay: float = 3.0,
    use_mobile: bool = False,
) -> AntiDetectionManager:
    """
    Create an anti-detection manager with common settings.

    Args:
        rotate_user_agents: Enable user agent rotation
        min_delay: Minimum delay between requests
        max_delay: Maximum delay between requests
        use_mobile: Include mobile user agents

    Returns:
        Configured AntiDetectionManager
    """
    device_types = [DeviceType.DESKTOP]
    browser_types = [BrowserType.CHROME, BrowserType.FIREFOX]

    if use_mobile:
        device_types.append(DeviceType.MOBILE)
        browser_types.extend([BrowserType.MOBILE_CHROME, BrowserType.MOBILE_SAFARI])

    config = AntiDetectionConfig(
        rotate_user_agents=rotate_user_agents,
        browser_types=browser_types,
        device_types=device_types,
        delay_config=DelayConfig(
            min_delay=min_delay,
            max_delay=max_delay,
        ),
    )

    return AntiDetectionManager(config)


def get_random_user_agent(
    browser: BrowserType = None,
    mobile: bool = False,
) -> str:
    """
    Get a random user agent string.

    Args:
        browser: Specific browser type (optional)
        mobile: Return mobile user agent

    Returns:
        User agent string
    """
    if browser:
        agents = USER_AGENTS.get(browser, USER_AGENTS[BrowserType.CHROME])
        return random.choice(agents).string

    if mobile:
        mobile_agents = USER_AGENTS.get(
            BrowserType.MOBILE_CHROME, []
        ) + USER_AGENTS.get(BrowserType.MOBILE_SAFARI, [])
        if mobile_agents:
            return random.choice(mobile_agents).string

    # Default: random desktop agent
    desktop_agents = []
    for browser_type in [
        BrowserType.CHROME,
        BrowserType.FIREFOX,
        BrowserType.SAFARI,
        BrowserType.EDGE,
    ]:
        desktop_agents.extend(USER_AGENTS.get(browser_type, []))

    return (
        random.choice(desktop_agents).string
        if desktop_agents
        else USER_AGENTS[BrowserType.CHROME][0].string
    )


def get_request_headers(
    url: str = None,
    user_agent: str = None,
    include_sec_headers: bool = True,
) -> dict[str, Any]:
    """
    Get a set of realistic request headers.

    Args:
        url: Target URL (for referer)
        user_agent: Specific user agent (or random)
        include_sec_headers: Include Sec-* headers

    Returns:
        Headers dictionary
    """
    if not user_agent:
        user_agent = get_random_user_agent()

    # Find matching UserAgent object
    ua_obj = None
    for browser_agents in USER_AGENTS.values():
        for agent in browser_agents:
            if agent.string == user_agent:
                ua_obj = agent
                break
        if ua_obj:
            break

    if not ua_obj:
        ua_obj = USER_AGENTS[BrowserType.CHROME][0]

    gen = FingerprintGenerator()
    fingerprint = gen.generate(ua_obj)

    headers = fingerprint.to_headers()

    if not include_sec_headers:
        headers = {k: v for k, v in headers.items() if not k.startswith("Sec-")}

    return headers


def calculate_delay(
    min_delay: float = 1.0,
    max_delay: float = 3.0,
    jitter: float = 0.3,
) -> float:
    """
    Calculate a randomized delay.

    Args:
        min_delay: Minimum delay
        max_delay: Maximum delay
        jitter: Jitter factor

    Returns:
        Delay in seconds
    """
    base = random.uniform(min_delay, max_delay)
    jitter_amount = base * jitter * (random.random() * 2 - 1)
    return max(0.1, base + jitter_amount)


if __name__ == "__main__":
    # Example usage
    manager = create_anti_detection_manager(
        rotate_user_agents=True,
        min_delay=1.0,
        max_delay=3.0,
    )

    print("Anti-Detection Manager Stats:")
    stats = manager.get_stats()
    print(f"  User Agent Pool: {stats['user_agent_pool_size']} agents")
    print(f"  Proxy Pool: {stats['proxy_pool_size']} proxies")

    # Prepare a request
    url = "https://example.com/page"
    request_info = manager.prepare_request(url)

    print(f"\nPrepared request for {url}:")
    print(f"  User-Agent: {request_info['headers']['User-Agent'][:50]}...")
    print(f"  Delay: {request_info['delay']:.2f}s")
    print(f"  Proxy: {request_info['proxy'] or 'None'}")

    print("\nSample headers:")
    for key, value in list(request_info["headers"].items())[:5]:
        print(f"  {key}: {value[:50] if len(str(value)) > 50 else value}")
