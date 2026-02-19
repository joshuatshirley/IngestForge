#!/usr/bin/env python3
"""Request delay management for anti-detection.

Manages request delays with randomization and burst protection.
"""

import time

from ingestforge.ingest.anti_detection.models import DelayConfig


class RequestDelayer:
    """Manages request delays."""

    def __init__(self, config: DelayConfig = None) -> None:
        self.config = config or DelayConfig()
        self._request_counts = {}  # domain -> count
        self._last_request_time = {}  # domain -> timestamp

    def wait(self, domain: str = "default") -> None:
        """Wait appropriate delay before next request."""
        count = self._request_counts.get(domain, 0)
        delay = self.config.get_delay(count)

        # Check if we need to wait from last request
        if domain in self._last_request_time:
            elapsed = time.time() - self._last_request_time[domain]
            remaining = delay - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._request_counts[domain] = count + 1
        self._last_request_time[domain] = time.time()

    def get_delay(self, domain: str = "default") -> float:
        """Get the delay that would be used (without waiting)."""
        count = self._request_counts.get(domain, 0)
        return self.config.get_delay(count)

    def reset_domain(self, domain: str) -> None:
        """Reset request count for a domain."""
        self._request_counts.pop(domain, None)
        self._last_request_time.pop(domain, None)
