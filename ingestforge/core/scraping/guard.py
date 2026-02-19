"""Domain Security Guard for crawler protection.

Validates URLs against domain restrictions and provides
robots.txt parsing for ethical crawling."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from urllib.parse import urlparse

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_DOMAINS_ALLOWED = 100
MAX_URL_LENGTH = 2048
MAX_ROBOTS_RULES = 500
MAX_PATH_PATTERNS = 200


class ValidationResult(str, Enum):
    """Result of URL validation."""

    ALLOWED = "allowed"
    BLOCKED_DOMAIN = "blocked_domain"
    BLOCKED_ROBOTS = "blocked_robots"
    BLOCKED_PATTERN = "blocked_pattern"
    INVALID_URL = "invalid_url"
    INVALID_SCHEME = "invalid_scheme"


@dataclass
class RobotsRules:
    """Parsed robots.txt rules for a domain."""

    domain: str
    disallow_patterns: List[str] = field(default_factory=list)
    allow_patterns: List[str] = field(default_factory=list)
    crawl_delay: float = 0.0
    sitemap_urls: List[str] = field(default_factory=list)

    def is_allowed(self, path: str) -> bool:
        """Check if path is allowed by robots.txt.

        Args:
            path: URL path to check

        Returns:
            True if allowed
        """
        # Check allow patterns first (more specific)
        for pattern in self.allow_patterns[:MAX_PATH_PATTERNS]:
            if self._matches_pattern(path, pattern):
                return True

        # Check disallow patterns
        for pattern in self.disallow_patterns[:MAX_PATH_PATTERNS]:
            if self._matches_pattern(path, pattern):
                return False

        # Default: allowed
        return True

    def _matches_pattern(self, path: str, pattern: str) -> bool:
        """Check if path matches robots pattern.

        Args:
            path: URL path
            pattern: Robots pattern

        Returns:
            True if matches
        """
        if not pattern:
            return False

        # Handle wildcards
        if "*" in pattern:
            regex = pattern.replace("*", ".*")
            return bool(re.match(regex, path))

        # Handle $ (end anchor)
        if pattern.endswith("$"):
            return path == pattern[:-1]

        # Prefix match
        return path.startswith(pattern)


@dataclass
class GuardConfig:
    """Configuration for domain guard."""

    allowed_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    allow_subdomains: bool = False
    enforce_robots_txt: bool = True
    allowed_schemes: List[str] = field(default_factory=lambda: ["http", "https"])
    blocked_extensions: List[str] = field(
        default_factory=lambda: [
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".exe",
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".svg",
            ".mp3",
            ".mp4",
            ".avi",
            ".mov",
        ]
    )


@dataclass
class ValidationReport:
    """Detailed validation report."""

    url: str
    result: ValidationResult
    domain: str = ""
    path: str = ""
    reason: str = ""
    robots_checked: bool = False

    @property
    def is_allowed(self) -> bool:
        """Check if URL is allowed."""
        return self.result == ValidationResult.ALLOWED


class DomainGuard:
    """Guards against off-domain and disallowed URLs.

    Provides URL validation with domain locking, robots.txt
    respect, and extension filtering.
    """

    def __init__(self, config: Optional[GuardConfig] = None) -> None:
        """Initialize guard.

        Args:
            config: Guard configuration
        """
        self.config = config or GuardConfig()
        self._robots_cache: Dict[str, RobotsRules] = {}
        self._primary_domain: str = ""

    def set_primary_domain(self, url: str) -> None:
        """Set primary domain from URL.

        Args:
            url: URL to extract domain from
        """
        parsed = urlparse(url)
        self._primary_domain = parsed.netloc

    def validate(self, url: str) -> ValidationReport:
        """Validate a URL against security rules.

        Args:
            url: URL to validate

        Returns:
            ValidationReport with result
        """
        # Empty URL
        if not url:
            return ValidationReport(
                url=url,
                result=ValidationResult.INVALID_URL,
                reason="Empty URL",
            )

        # URL length
        if len(url) > MAX_URL_LENGTH:
            return ValidationReport(
                url=url,
                result=ValidationResult.INVALID_URL,
                reason="URL too long",
            )

        # Parse URL
        try:
            parsed = urlparse(url)
        except Exception as e:
            return ValidationReport(
                url=url,
                result=ValidationResult.INVALID_URL,
                reason=str(e),
            )

        domain = parsed.netloc
        path = parsed.path

        # Validate scheme
        if parsed.scheme not in self.config.allowed_schemes:
            return ValidationReport(
                url=url,
                result=ValidationResult.INVALID_SCHEME,
                domain=domain,
                path=path,
                reason=f"Scheme not allowed: {parsed.scheme}",
            )

        # Check blocked domains
        if self._is_blocked_domain(domain):
            return ValidationReport(
                url=url,
                result=ValidationResult.BLOCKED_DOMAIN,
                domain=domain,
                path=path,
                reason="Domain is blocked",
            )

        # Check domain is allowed
        if not self._is_allowed_domain(domain):
            return ValidationReport(
                url=url,
                result=ValidationResult.BLOCKED_DOMAIN,
                domain=domain,
                path=path,
                reason="Domain not in allowed list",
            )

        # Check blocked extensions
        if self._has_blocked_extension(path):
            return ValidationReport(
                url=url,
                result=ValidationResult.BLOCKED_PATTERN,
                domain=domain,
                path=path,
                reason="File extension blocked",
            )

        # Check robots.txt
        robots_checked = False
        if self.config.enforce_robots_txt:
            robots_checked = True
            if not self._check_robots(domain, path):
                return ValidationReport(
                    url=url,
                    result=ValidationResult.BLOCKED_ROBOTS,
                    domain=domain,
                    path=path,
                    reason="Disallowed by robots.txt",
                    robots_checked=True,
                )

        return ValidationReport(
            url=url,
            result=ValidationResult.ALLOWED,
            domain=domain,
            path=path,
            robots_checked=robots_checked,
        )

    def is_allowed(self, url: str) -> bool:
        """Quick check if URL is allowed.

        Args:
            url: URL to check

        Returns:
            True if allowed
        """
        return self.validate(url).is_allowed

    def add_robots_rules(self, domain: str, rules: RobotsRules) -> None:
        """Add robots.txt rules for a domain.

        Args:
            domain: Domain
            rules: Parsed rules
        """
        self._robots_cache[domain] = rules

    def parse_robots_txt(self, domain: str, content: str) -> RobotsRules:
        """Parse robots.txt content.

        Args:
            domain: Domain
            content: robots.txt content

        Returns:
            Parsed RobotsRules
        """
        rules = RobotsRules(domain=domain)
        current_applies = False

        lines = content.split("\n")
        for line in lines[:MAX_ROBOTS_RULES]:
            current_applies = self._process_robots_line(line, rules, current_applies)

        # Cache the rules
        self._robots_cache[domain] = rules
        return rules

    def _process_robots_line(
        self,
        line: str,
        rules: RobotsRules,
        current_applies: bool,
    ) -> bool:
        """Process a single robots.txt line.

        Args:
            line: Line to process
            rules: RobotsRules to update
            current_applies: Whether current user-agent applies

        Returns:
            Updated current_applies flag
        """
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith("#"):
            return current_applies

        # Parse directive
        if ":" not in line:
            return current_applies

        directive, value = line.split(":", 1)
        directive = directive.strip().lower()
        value = value.strip()

        # User-agent directive
        if directive == "user-agent":
            return value == "*" or "bot" in value.lower()

        # Only process rules that apply to us
        if not current_applies:
            return current_applies

        # Process applicable directives
        _apply_robots_directive(directive, value, rules)
        return current_applies

    def _is_allowed_domain(self, domain: str) -> bool:
        """Check if domain is allowed.

        Args:
            domain: Domain to check

        Returns:
            True if allowed
        """
        # If no primary domain set and no allowed list, block all
        if not self._primary_domain and not self.config.allowed_domains:
            return False

        # Check primary domain
        if self._check_primary_domain(domain):
            return True

        # Check allowed list
        if self._check_allowed_list(domain):
            return True

        return False

    def _check_primary_domain(self, domain: str) -> bool:
        """Check if domain matches primary domain.

        Args:
            domain: Domain to check

        Returns:
            True if matches
        """
        if not self._primary_domain:
            return False

        if domain == self._primary_domain:
            return True

        if self.config.allow_subdomains:
            if domain.endswith(f".{self._primary_domain}"):
                return True

        return False

    def _check_allowed_list(self, domain: str) -> bool:
        """Check if domain is in allowed list.

        Args:
            domain: Domain to check

        Returns:
            True if in list
        """
        if not self.config.allowed_domains:
            return False

        for allowed in self.config.allowed_domains[:MAX_DOMAINS_ALLOWED]:
            if domain == allowed:
                return True
            if self.config.allow_subdomains:
                if domain.endswith(f".{allowed}"):
                    return True

        return False

    def _is_blocked_domain(self, domain: str) -> bool:
        """Check if domain is explicitly blocked.

        Args:
            domain: Domain to check

        Returns:
            True if blocked
        """
        for blocked in self.config.blocked_domains[:MAX_DOMAINS_ALLOWED]:
            if domain == blocked:
                return True
            if domain.endswith(f".{blocked}"):
                return True
        return False

    def _has_blocked_extension(self, path: str) -> bool:
        """Check if path has blocked extension.

        Args:
            path: URL path

        Returns:
            True if blocked
        """
        path_lower = path.lower()
        for ext in self.config.blocked_extensions:
            if path_lower.endswith(ext):
                return True
        return False

    def _check_robots(self, domain: str, path: str) -> bool:
        """Check robots.txt rules.

        Args:
            domain: Domain
            path: URL path

        Returns:
            True if allowed by robots.txt
        """
        rules = self._robots_cache.get(domain)
        if not rules:
            # No rules cached, allow by default
            return True

        return rules.is_allowed(path)


def _apply_robots_directive(directive: str, value: str, rules: RobotsRules) -> None:
    """Apply a robots.txt directive to rules.

    Args:
        directive: Directive name (lowercased)
        value: Directive value
        rules: RobotsRules to update
    """
    # Early return for empty values on allow/disallow
    if directive in ("disallow", "allow") and not value:
        return

    if directive == "disallow":
        rules.disallow_patterns.append(value)
        return

    if directive == "allow":
        rules.allow_patterns.append(value)
        return

    if directive == "crawl-delay":
        _set_crawl_delay(value, rules)
        return

    if directive == "sitemap":
        rules.sitemap_urls.append(value)


def _set_crawl_delay(value: str, rules: RobotsRules) -> None:
    """Set crawl delay from string value.

    Args:
        value: Delay value string
        rules: RobotsRules to update
    """
    try:
        rules.crawl_delay = float(value)
    except ValueError:
        logger.debug(f"Invalid crawl-delay value: {value}")


def create_guard(
    primary_url: str,
    allow_subdomains: bool = False,
    enforce_robots: bool = True,
) -> DomainGuard:
    """Factory function to create domain guard.

    Args:
        primary_url: Primary URL to lock to
        allow_subdomains: Allow subdomains
        enforce_robots: Enforce robots.txt

    Returns:
        Configured DomainGuard
    """
    config = GuardConfig(
        allow_subdomains=allow_subdomains,
        enforce_robots_txt=enforce_robots,
    )
    guard = DomainGuard(config=config)
    guard.set_primary_domain(primary_url)
    return guard


def validate_url(url: str, primary_domain: str = "") -> ValidationReport:
    """Convenience function to validate URL.

    Args:
        url: URL to validate
        primary_domain: Primary domain

    Returns:
        ValidationReport
    """
    guard = DomainGuard()
    if primary_domain:
        guard.set_primary_domain(f"https://{primary_domain}")
    return guard.validate(url)
