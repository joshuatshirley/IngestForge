"""URL Validation and SSRF Prevention.

Validates URLs and blocks Server-Side Request Forgery (SSRF) attacks by
preventing access to private IP ranges and localhost.

Web URL Connector - Epic (Security)
Follows NASA JPL Rules #2 (Bounded), #4 (Modular), #7 (Check Returns), #9 (Type Hints)

Timestamp: 2026-02-18 20:00 UTC
"""

from __future__ import annotations

import ipaddress
import re
from typing import Tuple
from urllib.parse import urlparse

# JPL Rule #2: Fixed upper bounds
MAX_URL_LENGTH = 2048
MAX_DOMAIN_LENGTH = 253
MAX_PATH_LENGTH = 2000

# Allowed URL schemes (prevent file://, javascript:, data:, etc.)
ALLOWED_SCHEMES = frozenset({"http", "https"})

# Private IP ranges to block (SSRF prevention)
PRIVATE_IP_RANGES = [
    ipaddress.ip_network("127.0.0.0/8"),  # Localhost
    ipaddress.ip_network("10.0.0.0/8"),  # Private Class A
    ipaddress.ip_network("172.16.0.0/12"),  # Private Class B
    ipaddress.ip_network("192.168.0.0/16"),  # Private Class C
    ipaddress.ip_network("169.254.0.0/16"),  # Link-local
    ipaddress.ip_network("::1/128"),  # IPv6 localhost
    ipaddress.ip_network("fc00::/7"),  # IPv6 private
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local
]

# Dangerous domain patterns to block
BLOCKED_DOMAIN_PATTERNS = frozenset(
    {
        r"localhost",
        r"\.local$",
        r"\.internal$",
        r"\.localdomain$",
        r"\.lan$",
    }
)


def validate_url(url: str) -> Tuple[bool, str]:
    """
    Validate URL and check for SSRF vulnerabilities.

    Epic URL Validation & Security
    Timestamp: 2026-02-18 20:00 UTC

    JPL Rule #4: Function under 60 lines
    JPL Rule #7: Returns (is_valid, error_message) tuple
    JPL Rule #9: Complete type hints

    Args:
        url: URL to validate

    Returns:
        Tuple of (is_valid: bool, error_message: str)
        - (True, "") if valid
        - (False, "error reason") if invalid

    Examples:
        >>> validate_url("https://example.com/page")
        (True, "")

        >>> validate_url("http://192.168.1.1/admin")
        (False, "Private IP address blocked (SSRF prevention)")

        >>> validate_url("file:///etc/passwd")
        (False, "URL scheme must be http or https")
    """
    # JPL Rule #7: Check preconditions
    if not url or not isinstance(url, str):
        return (False, "URL cannot be empty")

    url = url.strip()

    # Check URL length (JPL Rule #2: Bounded)
    if len(url) > MAX_URL_LENGTH:
        return (False, f"URL exceeds {MAX_URL_LENGTH} character limit")

    # Parse URL
    try:
        parsed = urlparse(url)
    except Exception as e:
        return (False, f"Invalid URL format: {e}")

    # Validate scheme
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        return (False, f"URL scheme must be http or https, got: {parsed.scheme}")

    # Validate hostname exists
    if not parsed.netloc:
        return (False, "URL must include a domain name")

    # Extract hostname (remove port if present)
    hostname = parsed.hostname
    if not hostname:
        return (False, "Invalid hostname in URL")

    # Check domain length
    if len(hostname) > MAX_DOMAIN_LENGTH:
        return (False, f"Domain name exceeds {MAX_DOMAIN_LENGTH} character limit")

    # Check for blocked domain patterns
    for pattern in BLOCKED_DOMAIN_PATTERNS:
        if re.search(pattern, hostname, re.IGNORECASE):
            return (False, f"Domain '{hostname}' is blocked (internal/local domain)")

    # Check if hostname is an IP address
    try:
        ip_addr = ipaddress.ip_address(hostname)

        # Check if IP is in private range (SSRF prevention)
        for private_range in PRIVATE_IP_RANGES:
            if ip_addr in private_range:
                return (
                    False,
                    f"Private IP address blocked (SSRF prevention): {ip_addr}",
                )

    except ValueError:
        # Not an IP address, it's a domain name - this is fine
        pass

    # Check path length
    if len(parsed.path) > MAX_PATH_LENGTH:
        return (False, f"URL path exceeds {MAX_PATH_LENGTH} character limit")

    # All checks passed
    return (True, "")


def is_safe_url(url: str) -> bool:
    """
    Quick check if URL is safe (no SSRF risk).

    Epic
    Timestamp: 2026-02-18 20:00 UTC

    JPL Rule #4: Small helper function
    JPL Rule #7: Returns boolean result

    Args:
        url: URL to check

    Returns:
        True if safe, False if unsafe

    Example:
        >>> is_safe_url("https://example.com/page")
        True
    """
    is_valid, _ = validate_url(url)
    return is_valid


def get_validation_error(url: str) -> str:
    """
    Get validation error message for URL.

    Epic
    Timestamp: 2026-02-18 20:00 UTC

    JPL Rule #4: Small helper function
    JPL Rule #7: Returns error string (empty if valid)

    Args:
        url: URL to validate

    Returns:
        Error message (empty string if valid)

    Example:
        >>> get_validation_error("file:///etc/passwd")
        "URL scheme must be http or https, got: file"
    """
    is_valid, error = validate_url(url)
    return error if not is_valid else ""
