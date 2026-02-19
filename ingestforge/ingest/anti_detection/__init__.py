#!/usr/bin/env python3
"""Anti-bot detection handling for web scraping.

Implements measures to avoid detection when scraping websites:
- User-agent rotation
- Request delays with randomization
- Referer header management
- Cookie session handling
- Fingerprint randomization
- Proxy rotation support

This module has been split into focused submodules for maintainability
while maintaining 100% backward compatibility via re-exports.
"""

# Models (enums and dataclasses)
from ingestforge.ingest.anti_detection.models import (
    AntiDetectionConfig,
    BrowserType,
    DelayConfig,
    DeviceType,
    ProxyConfig,
    RequestFingerprint,
    UserAgent,
)

# Data
from ingestforge.ingest.anti_detection.data import USER_AGENTS

# Rotators
from ingestforge.ingest.anti_detection.rotators import ProxyRotator, UserAgentRotator

# Fingerprint and referer
from ingestforge.ingest.anti_detection.fingerprint import (
    FingerprintGenerator,
    RefererManager,
)

# Delay
from ingestforge.ingest.anti_detection.delay import RequestDelayer

# Manager and convenience functions
from ingestforge.ingest.anti_detection.manager import (
    AntiDetectionManager,
    calculate_delay,
    create_anti_detection_manager,
    get_random_user_agent,
    get_request_headers,
)

__all__ = [
    # Enums
    "BrowserType",
    "DeviceType",
    # Models
    "UserAgent",
    "RequestFingerprint",
    "DelayConfig",
    "ProxyConfig",
    "AntiDetectionConfig",
    # Data
    "USER_AGENTS",
    # Rotators
    "UserAgentRotator",
    "ProxyRotator",
    # Fingerprint
    "FingerprintGenerator",
    "RefererManager",
    # Delay
    "RequestDelayer",
    # Manager
    "AntiDetectionManager",
    # Convenience functions
    "create_anti_detection_manager",
    "get_random_user_agent",
    "get_request_headers",
    "calculate_delay",
]
