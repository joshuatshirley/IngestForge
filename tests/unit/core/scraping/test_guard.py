"""Tests for domain security guard.

Tests URL validation, domain restriction, and robots.txt parsing."""

from __future__ import annotations


from ingestforge.core.scraping.guard import (
    DomainGuard,
    GuardConfig,
    RobotsRules,
    ValidationReport,
    ValidationResult,
    create_guard,
    validate_url,
)

# ValidationResult tests


class TestValidationResult:
    """Tests for ValidationResult enum."""

    def test_results_defined(self) -> None:
        """Test all results are defined."""
        results = [r.value for r in ValidationResult]

        assert "allowed" in results
        assert "blocked_domain" in results
        assert "blocked_robots" in results
        assert "blocked_pattern" in results
        assert "invalid_url" in results
        assert "invalid_scheme" in results


# RobotsRules tests


class TestRobotsRules:
    """Tests for RobotsRules dataclass."""

    def test_rules_creation(self) -> None:
        """Test creating rules."""
        rules = RobotsRules(domain="example.com")

        assert rules.domain == "example.com"
        assert len(rules.disallow_patterns) == 0

    def test_is_allowed_empty_rules(self) -> None:
        """Test is_allowed with no rules."""
        rules = RobotsRules(domain="example.com")

        assert rules.is_allowed("/any/path") is True

    def test_is_allowed_disallow_match(self) -> None:
        """Test is_allowed with matching disallow."""
        rules = RobotsRules(
            domain="example.com",
            disallow_patterns=["/admin/", "/private/"],
        )

        assert rules.is_allowed("/admin/page") is False
        assert rules.is_allowed("/public/page") is True

    def test_is_allowed_allow_overrides(self) -> None:
        """Test that allow overrides disallow."""
        rules = RobotsRules(
            domain="example.com",
            disallow_patterns=["/api/"],
            allow_patterns=["/api/public/"],
        )

        assert rules.is_allowed("/api/private/") is False
        assert rules.is_allowed("/api/public/") is True

    def test_pattern_wildcard(self) -> None:
        """Test wildcard pattern matching."""
        rules = RobotsRules(
            domain="example.com",
            disallow_patterns=["/admin/*"],
        )

        assert rules.is_allowed("/admin/page") is False
        assert rules.is_allowed("/admin/") is False

    def test_pattern_end_anchor(self) -> None:
        """Test end anchor pattern matching."""
        rules = RobotsRules(
            domain="example.com",
            disallow_patterns=["/page$"],
        )

        assert rules.is_allowed("/page") is False
        assert rules.is_allowed("/page/subpage") is True


# GuardConfig tests


class TestGuardConfig:
    """Tests for GuardConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = GuardConfig()

        assert config.allow_subdomains is False
        assert config.enforce_robots_txt is True
        assert "http" in config.allowed_schemes
        assert "https" in config.allowed_schemes

    def test_blocked_extensions_default(self) -> None:
        """Test default blocked extensions."""
        config = GuardConfig()

        assert ".pdf" in config.blocked_extensions
        assert ".exe" in config.blocked_extensions
        assert ".jpg" in config.blocked_extensions


# ValidationReport tests


class TestValidationReport:
    """Tests for ValidationReport dataclass."""

    def test_allowed_report(self) -> None:
        """Test allowed report."""
        report = ValidationReport(
            url="https://example.com",
            result=ValidationResult.ALLOWED,
            domain="example.com",
        )

        assert report.is_allowed is True

    def test_blocked_report(self) -> None:
        """Test blocked report."""
        report = ValidationReport(
            url="https://other.com",
            result=ValidationResult.BLOCKED_DOMAIN,
            reason="Domain not allowed",
        )

        assert report.is_allowed is False


# DomainGuard tests


class TestDomainGuard:
    """Tests for DomainGuard."""

    def test_guard_creation(self) -> None:
        """Test creating guard."""
        guard = DomainGuard()

        assert guard.config is not None

    def test_guard_with_config(self) -> None:
        """Test guard with custom config."""
        config = GuardConfig(allow_subdomains=True)
        guard = DomainGuard(config=config)

        assert guard.config.allow_subdomains is True

    def test_set_primary_domain(self) -> None:
        """Test setting primary domain."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com/page")

        assert guard._primary_domain == "example.com"


class TestURLValidation:
    """Tests for URL validation."""

    def test_validate_empty_url(self) -> None:
        """Test validating empty URL."""
        guard = DomainGuard()

        report = guard.validate("")

        assert report.result == ValidationResult.INVALID_URL

    def test_validate_invalid_scheme(self) -> None:
        """Test validating invalid scheme."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com")

        report = guard.validate("ftp://example.com/file")

        assert report.result == ValidationResult.INVALID_SCHEME

    def test_validate_allowed_domain(self) -> None:
        """Test validating allowed domain."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com")

        report = guard.validate("https://example.com/page")

        assert report.result == ValidationResult.ALLOWED

    def test_validate_blocked_domain(self) -> None:
        """Test validating blocked domain."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com")

        report = guard.validate("https://other.com/page")

        assert report.result == ValidationResult.BLOCKED_DOMAIN

    def test_validate_blocked_extension(self) -> None:
        """Test validating blocked extension."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com")

        report = guard.validate("https://example.com/file.pdf")

        assert report.result == ValidationResult.BLOCKED_PATTERN

    def test_is_allowed_shorthand(self) -> None:
        """Test is_allowed shorthand method."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com")

        assert guard.is_allowed("https://example.com/page") is True
        assert guard.is_allowed("https://other.com/page") is False


class TestSubdomainHandling:
    """Tests for subdomain handling."""

    def test_subdomain_blocked_default(self) -> None:
        """Test subdomain blocked by default."""
        guard = DomainGuard()
        guard.set_primary_domain("https://example.com")

        assert guard.is_allowed("https://www.example.com") is False
        assert guard.is_allowed("https://blog.example.com") is False

    def test_subdomain_allowed_when_enabled(self) -> None:
        """Test subdomain allowed when enabled."""
        config = GuardConfig(allow_subdomains=True)
        guard = DomainGuard(config=config)
        guard.set_primary_domain("https://example.com")

        assert guard.is_allowed("https://www.example.com") is True
        assert guard.is_allowed("https://blog.example.com") is True


class TestRobotsTxtParsing:
    """Tests for robots.txt parsing."""

    def test_parse_simple_robots(self) -> None:
        """Test parsing simple robots.txt."""
        guard = DomainGuard()
        content = """
User-agent: *
Disallow: /admin/
Disallow: /private/
Allow: /admin/public/
Crawl-delay: 1
"""
        rules = guard.parse_robots_txt("example.com", content)

        assert len(rules.disallow_patterns) == 2
        assert "/admin/" in rules.disallow_patterns
        assert "/private/" in rules.disallow_patterns
        assert len(rules.allow_patterns) == 1
        assert rules.crawl_delay == 1.0

    def test_parse_robots_with_sitemap(self) -> None:
        """Test parsing robots.txt with sitemap."""
        guard = DomainGuard()
        content = """
User-agent: *
Sitemap: https://example.com/sitemap.xml
"""
        rules = guard.parse_robots_txt("example.com", content)

        assert "https://example.com/sitemap.xml" in rules.sitemap_urls

    def test_robots_rules_cached(self) -> None:
        """Test that robots rules are cached."""
        guard = DomainGuard()
        content = "User-agent: *\nDisallow: /admin/"

        guard.parse_robots_txt("example.com", content)

        assert "example.com" in guard._robots_cache

    def test_validate_respects_robots(self) -> None:
        """Test that validation respects robots.txt."""
        config = GuardConfig(enforce_robots_txt=True)
        guard = DomainGuard(config=config)
        guard.set_primary_domain("https://example.com")

        # Add robots rules
        rules = RobotsRules(
            domain="example.com",
            disallow_patterns=["/admin/"],
        )
        guard.add_robots_rules("example.com", rules)

        report = guard.validate("https://example.com/admin/page")

        assert report.result == ValidationResult.BLOCKED_ROBOTS


class TestBlockedDomains:
    """Tests for explicitly blocked domains."""

    def test_blocked_domain_rejected(self) -> None:
        """Test explicitly blocked domain."""
        config = GuardConfig(
            blocked_domains=["malicious.com"],
        )
        guard = DomainGuard(config=config)
        guard.set_primary_domain("https://example.com")

        report = guard.validate("https://malicious.com/page")

        assert report.result == ValidationResult.BLOCKED_DOMAIN

    def test_blocked_subdomain_rejected(self) -> None:
        """Test subdomain of blocked domain."""
        config = GuardConfig(
            blocked_domains=["malicious.com"],
        )
        guard = DomainGuard(config=config)
        guard.set_primary_domain("https://example.com")

        report = guard.validate("https://sub.malicious.com/page")

        assert report.result == ValidationResult.BLOCKED_DOMAIN


# Factory function tests


class TestCreateGuard:
    """Tests for create_guard factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        guard = create_guard("https://example.com")

        assert guard._primary_domain == "example.com"
        assert guard.config.allow_subdomains is False

    def test_create_with_subdomains(self) -> None:
        """Test creating with subdomains."""
        guard = create_guard(
            "https://example.com",
            allow_subdomains=True,
        )

        assert guard.config.allow_subdomains is True


class TestValidateURLFunction:
    """Tests for validate_url function."""

    def test_validate_without_primary(self) -> None:
        """Test validation without primary domain."""
        report = validate_url("https://example.com")

        # Without primary domain set, should block
        assert report.is_allowed is False

    def test_validate_with_primary(self) -> None:
        """Test validation with primary domain."""
        report = validate_url(
            "https://example.com/page",
            primary_domain="example.com",
        )

        assert report.is_allowed is True
