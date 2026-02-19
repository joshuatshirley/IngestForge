"""Comprehensive Unit Tests for Security Configuration.

Security Shield CI
Tests all configuration loading, parsing, and validation functions.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints

Test Coverage Target: >80%
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ingestforge.core.security.config import (
    MAX_CUSTOM_RULES,
    MAX_EXCLUDE_PATTERNS,
    MAX_FILE_EXTENSIONS,
    MAX_CONFIG_FILE_SIZE,
    SecurityConfig,
    _parse_severity,
    _parse_category,
    _parse_custom_rules,
    load_config,
    _build_config_from_dict,
    apply_config_to_scanner,
    get_default_config_path,
)
from ingestforge.core.security.scanner import (
    FindingCategory,
    SecurityRule,
    Severity,
    create_scanner,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def minimal_config_yaml() -> str:
    """Minimal valid YAML config."""
    return 'version: "1.0"'


@pytest.fixture
def full_config_yaml() -> str:
    """Complete YAML config with all sections."""
    return """
version: "1.0"

rules:
  enabled:
    - SEC001
    - SEC002
  disabled:
    - SEC040
    - SEC041

exclude_patterns:
  - "**/tests/**"
  - "**/*.md"
  - "**/fixtures/**"

severity_threshold: "high"

fail_on:
  - "critical"
  - "high"

scanner:
  max_file_size: 5000000
  max_findings: 500
  file_extensions:
    - ".py"
    - ".js"

custom_rules:
  - rule_id: "TEST001"
    title: "Test Secret"
    description: "Test secret pattern"
    pattern: "SECRET_[A-Z0-9]{16}"
    severity: "high"
    category: "secrets"
    recommendation: "Use env vars"
    file_extensions:
      - ".py"
      - ".yaml"
  - rule_id: "TEST002"
    title: "Test Injection"
    description: "Test injection pattern"
    pattern: 'execute\\([^)]+\\+[^)]+\\)'
    severity: "critical"
    category: "injection"
    recommendation: "Use parameterized queries"

ci:
  enabled: true
  fail_on_warning: true
  github_comment: true
  max_display_findings: 15

precommit:
  enabled: true
  staged_only: false
  allow_warnings: false

output:
  format: "json"
  verbose: true
"""


@pytest.fixture
def invalid_severity_yaml() -> str:
    """YAML with invalid severity value."""
    return """
version: "1.0"
severity_threshold: "invalid_severity"
"""


@pytest.fixture
def invalid_category_yaml() -> str:
    """YAML with invalid category in custom rule."""
    return """
version: "1.0"
custom_rules:
  - rule_id: "TEST001"
    title: "Test"
    pattern: "test"
    severity: "high"
    category: "invalid_category"
    recommendation: "test"
"""


# =============================================================================
# TestSecurityConfigDataclass
# =============================================================================


class TestSecurityConfigDataclass:
    """Tests for SecurityConfig dataclass creation and validation."""

    def test_create_security_config_with_defaults(self) -> None:
        """
        GIVEN: No parameters
        WHEN: Creating SecurityConfig
        THEN: Default values are set correctly
        """
        config = SecurityConfig()

        assert config.version == "1.0"
        assert config.rules_enabled == []
        assert config.rules_disabled == []
        assert config.exclude_patterns == []
        assert config.severity_threshold == Severity.MEDIUM
        assert Severity.CRITICAL in config.fail_on
        assert Severity.HIGH in config.fail_on
        assert config.max_file_size == 10_000_000
        assert config.max_findings == 1000
        assert config.custom_rules == []
        assert config.ci_enabled is True
        assert config.precommit_enabled is True

    def test_create_security_config_with_custom_values(self) -> None:
        """
        GIVEN: Custom parameter values
        WHEN: Creating SecurityConfig
        THEN: Custom values are set correctly
        """
        config = SecurityConfig(
            version="2.0",
            rules_enabled=["SEC001"],
            rules_disabled=["SEC040"],
            severity_threshold=Severity.HIGH,
            max_file_size=5_000_000,
            ci_fail_on_warning=True,
        )

        assert config.version == "2.0"
        assert config.rules_enabled == ["SEC001"]
        assert config.rules_disabled == ["SEC040"]
        assert config.severity_threshold == Severity.HIGH
        assert config.max_file_size == 5_000_000
        assert config.ci_fail_on_warning is True

    def test_security_config_validates_too_many_enabled_rules(self) -> None:
        """
        GIVEN: Too many enabled rules (>MAX_CUSTOM_RULES)
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        too_many_rules = [f"RULE{i:03d}" for i in range(MAX_CUSTOM_RULES + 1)]

        with pytest.raises(AssertionError, match="Too many enabled rules"):
            SecurityConfig(rules_enabled=too_many_rules)

    def test_security_config_validates_too_many_disabled_rules(self) -> None:
        """
        GIVEN: Too many disabled rules (>MAX_CUSTOM_RULES)
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        too_many_rules = [f"RULE{i:03d}" for i in range(MAX_CUSTOM_RULES + 1)]

        with pytest.raises(AssertionError, match="Too many disabled rules"):
            SecurityConfig(rules_disabled=too_many_rules)

    def test_security_config_validates_too_many_exclusions(self) -> None:
        """
        GIVEN: Too many exclusion patterns (>MAX_EXCLUDE_PATTERNS)
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        too_many_patterns = [f"pattern{i}" for i in range(MAX_EXCLUDE_PATTERNS + 1)]

        with pytest.raises(AssertionError, match="Too many exclusions"):
            SecurityConfig(exclude_patterns=too_many_patterns)

    def test_security_config_validates_too_many_custom_rules(self) -> None:
        """
        GIVEN: Too many custom rules (>MAX_CUSTOM_RULES)
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        pattern = re.compile("test")
        too_many_rules = [
            SecurityRule(
                rule_id=f"CUSTOM{i:03d}",
                title="Test",
                description="Test",
                category=FindingCategory.SECRETS,
                severity=Severity.HIGH,
                pattern=pattern,
                recommendation="Test",
            )
            for i in range(MAX_CUSTOM_RULES + 1)
        ]

        with pytest.raises(AssertionError, match="Too many custom rules"):
            SecurityConfig(custom_rules=too_many_rules)

    def test_security_config_validates_too_many_extensions(self) -> None:
        """
        GIVEN: Too many file extensions (>MAX_FILE_EXTENSIONS)
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        too_many_exts = [f".ext{i}" for i in range(MAX_FILE_EXTENSIONS + 1)]

        with pytest.raises(AssertionError, match="Too many file extensions"):
            SecurityConfig(file_extensions=too_many_exts)

    def test_security_config_validates_negative_max_file_size(self) -> None:
        """
        GIVEN: Negative max_file_size
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        with pytest.raises(AssertionError, match="max_file_size must be positive"):
            SecurityConfig(max_file_size=-1)

    def test_security_config_validates_negative_max_findings(self) -> None:
        """
        GIVEN: Negative max_findings
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        with pytest.raises(AssertionError, match="max_findings must be positive"):
            SecurityConfig(max_findings=-1)

    def test_security_config_validates_negative_ci_max_display(self) -> None:
        """
        GIVEN: Negative ci_max_display
        WHEN: Creating SecurityConfig
        THEN: AssertionError is raised
        """
        with pytest.raises(AssertionError, match="ci_max_display must be positive"):
            SecurityConfig(ci_max_display=-1)


# =============================================================================
# TestParseSeverity
# =============================================================================


class TestParseSeverity:
    """Tests for _parse_severity helper function."""

    def test_parse_severity_critical(self) -> None:
        """
        GIVEN: String "critical"
        WHEN: Parsing severity
        THEN: Severity.CRITICAL is returned
        """
        result = _parse_severity("critical")
        assert result == Severity.CRITICAL

    def test_parse_severity_high(self) -> None:
        """
        GIVEN: String "high"
        WHEN: Parsing severity
        THEN: Severity.HIGH is returned
        """
        result = _parse_severity("high")
        assert result == Severity.HIGH

    def test_parse_severity_medium(self) -> None:
        """
        GIVEN: String "medium"
        WHEN: Parsing severity
        THEN: Severity.MEDIUM is returned
        """
        result = _parse_severity("medium")
        assert result == Severity.MEDIUM

    def test_parse_severity_low(self) -> None:
        """
        GIVEN: String "low"
        WHEN: Parsing severity
        THEN: Severity.LOW is returned
        """
        result = _parse_severity("low")
        assert result == Severity.LOW

    def test_parse_severity_info(self) -> None:
        """
        GIVEN: String "info"
        WHEN: Parsing severity
        THEN: Severity.INFO is returned
        """
        result = _parse_severity("info")
        assert result == Severity.INFO

    def test_parse_severity_case_insensitive(self) -> None:
        """
        GIVEN: Mixed case string "CrItIcAl"
        WHEN: Parsing severity
        THEN: Severity.CRITICAL is returned
        """
        result = _parse_severity("CrItIcAl")
        assert result == Severity.CRITICAL

    def test_parse_severity_with_whitespace(self) -> None:
        """
        GIVEN: String with whitespace "  high  "
        WHEN: Parsing severity
        THEN: Severity.HIGH is returned
        """
        result = _parse_severity("  high  ")
        assert result == Severity.HIGH

    def test_parse_severity_invalid_raises_value_error(self) -> None:
        """
        GIVEN: Invalid severity string
        WHEN: Parsing severity
        THEN: ValueError is raised
        """
        with pytest.raises(ValueError, match="Invalid severity"):
            _parse_severity("invalid")


# =============================================================================
# TestParseCategory
# =============================================================================


class TestParseCategory:
    """Tests for _parse_category helper function."""

    def test_parse_category_secrets(self) -> None:
        """
        GIVEN: String "secrets"
        WHEN: Parsing category
        THEN: FindingCategory.SECRETS is returned
        """
        result = _parse_category("secrets")
        assert result == FindingCategory.SECRETS

    def test_parse_category_injection(self) -> None:
        """
        GIVEN: String "injection"
        WHEN: Parsing category
        THEN: FindingCategory.INJECTION is returned
        """
        result = _parse_category("injection")
        assert result == FindingCategory.INJECTION

    def test_parse_category_crypto(self) -> None:
        """
        GIVEN: String "crypto"
        WHEN: Parsing category
        THEN: FindingCategory.CRYPTO is returned
        """
        result = _parse_category("crypto")
        assert result == FindingCategory.CRYPTO

    def test_parse_category_auth(self) -> None:
        """
        GIVEN: String "auth"
        WHEN: Parsing category
        THEN: FindingCategory.AUTH is returned
        """
        result = _parse_category("auth")
        assert result == FindingCategory.AUTH

    def test_parse_category_config(self) -> None:
        """
        GIVEN: String "config"
        WHEN: Parsing category
        THEN: FindingCategory.CONFIG is returned
        """
        result = _parse_category("config")
        assert result == FindingCategory.CONFIG

    def test_parse_category_data(self) -> None:
        """
        GIVEN: String "data"
        WHEN: Parsing category
        THEN: FindingCategory.DATA is returned
        """
        result = _parse_category("data")
        assert result == FindingCategory.DATA

    def test_parse_category_dependencies(self) -> None:
        """
        GIVEN: String "dependencies"
        WHEN: Parsing category
        THEN: FindingCategory.DEPENDENCIES is returned
        """
        result = _parse_category("dependencies")
        assert result == FindingCategory.DEPENDENCIES

    def test_parse_category_case_insensitive(self) -> None:
        """
        GIVEN: Mixed case string "SeCrEtS"
        WHEN: Parsing category
        THEN: FindingCategory.SECRETS is returned
        """
        result = _parse_category("SeCrEtS")
        assert result == FindingCategory.SECRETS

    def test_parse_category_with_whitespace(self) -> None:
        """
        GIVEN: String with whitespace "  injection  "
        WHEN: Parsing category
        THEN: FindingCategory.INJECTION is returned
        """
        result = _parse_category("  injection  ")
        assert result == FindingCategory.INJECTION

    def test_parse_category_invalid_raises_value_error(self) -> None:
        """
        GIVEN: Invalid category string
        WHEN: Parsing category
        THEN: ValueError is raised
        """
        with pytest.raises(ValueError, match="Invalid category"):
            _parse_category("invalid_category")


# =============================================================================
# TestParseCustomRules
# =============================================================================


class TestParseCustomRules:
    """Tests for _parse_custom_rules helper function."""

    def test_parse_custom_rules_empty_list(self) -> None:
        """
        GIVEN: Empty list
        WHEN: Parsing custom rules
        THEN: Empty list is returned
        """
        result = _parse_custom_rules([])
        assert result == []

    def test_parse_custom_rules_single_valid_rule(self) -> None:
        """
        GIVEN: Single valid rule dictionary
        WHEN: Parsing custom rules
        THEN: List with one SecurityRule is returned
        """
        rules_data = [
            {
                "rule_id": "TEST001",
                "title": "Test Rule",
                "description": "Test description",
                "pattern": "TEST_[A-Z]+",
                "severity": "high",
                "category": "secrets",
                "recommendation": "Fix it",
                "file_extensions": [".py"],
            }
        ]

        result = _parse_custom_rules(rules_data)

        assert len(result) == 1
        assert result[0].rule_id == "TEST001"
        assert result[0].title == "Test Rule"
        assert result[0].severity == Severity.HIGH
        assert result[0].category == FindingCategory.SECRETS

    def test_parse_custom_rules_multiple_valid_rules(self) -> None:
        """
        GIVEN: Multiple valid rule dictionaries
        WHEN: Parsing custom rules
        THEN: List with multiple SecurityRules is returned
        """
        rules_data = [
            {
                "rule_id": "TEST001",
                "title": "Rule 1",
                "pattern": "PATTERN1",
                "severity": "high",
                "category": "secrets",
                "recommendation": "Fix 1",
            },
            {
                "rule_id": "TEST002",
                "title": "Rule 2",
                "pattern": "PATTERN2",
                "severity": "medium",
                "category": "injection",
                "recommendation": "Fix 2",
            },
        ]

        result = _parse_custom_rules(rules_data)

        assert len(result) == 2
        assert result[0].rule_id == "TEST001"
        assert result[1].rule_id == "TEST002"

    def test_parse_custom_rules_with_missing_fields_uses_defaults(self) -> None:
        """
        GIVEN: Rule with missing optional fields
        WHEN: Parsing custom rules
        THEN: Default values are used
        """
        rules_data = [
            {
                "pattern": "TEST_PATTERN",
            }
        ]

        result = _parse_custom_rules(rules_data)

        assert len(result) == 1
        assert result[0].rule_id == "CUSTOM000"  # default
        assert result[0].title == "Custom Rule"  # default
        assert result[0].severity == Severity.MEDIUM  # default
        assert result[0].category == FindingCategory.SECRETS  # default

    def test_parse_custom_rules_invalid_pattern_skips_rule(self) -> None:
        """
        GIVEN: Rule with invalid regex pattern
        WHEN: Parsing custom rules
        THEN: Invalid rule is skipped (logged)
        """
        rules_data = [
            {
                "rule_id": "VALID",
                "pattern": "VALID_PATTERN",
                "severity": "high",
                "category": "secrets",
            },
            {
                "rule_id": "INVALID",
                "pattern": "[invalid(regex",  # Invalid regex
                "severity": "high",
                "category": "secrets",
            },
        ]

        result = _parse_custom_rules(rules_data)

        # Only valid rule should be included
        assert len(result) == 1
        assert result[0].rule_id == "VALID"

    def test_parse_custom_rules_exceeding_max_raises_assertion(self) -> None:
        """
        GIVEN: More than MAX_CUSTOM_RULES rules
        WHEN: Parsing custom rules
        THEN: AssertionError is raised
        """
        too_many_rules = [
            {"pattern": f"PATTERN{i}"} for i in range(MAX_CUSTOM_RULES + 1)
        ]

        with pytest.raises(AssertionError, match="Too many custom rules"):
            _parse_custom_rules(too_many_rules)


# =============================================================================
# TestLoadConfig
# =============================================================================


class TestLoadConfig:
    """Tests for load_config main function."""

    def test_load_config_with_none_returns_defaults(self) -> None:
        """
        GIVEN: None config path
        WHEN: Loading config
        THEN: Default SecurityConfig is returned
        """
        config = load_config(None)

        assert isinstance(config, SecurityConfig)
        assert config.version == "1.0"
        assert config.severity_threshold == Severity.MEDIUM

    def test_load_config_with_nonexistent_file_returns_defaults(self) -> None:
        """
        GIVEN: Path to non-existent file
        WHEN: Loading config
        THEN: Default SecurityConfig is returned
        """
        config = load_config(Path("/nonexistent/config.yaml"))

        assert isinstance(config, SecurityConfig)
        assert config.version == "1.0"

    def test_load_config_with_minimal_yaml(
        self,
        minimal_config_yaml: str,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: Minimal valid YAML file
        WHEN: Loading config
        THEN: Config with defaults is returned
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(minimal_config_yaml, encoding="utf-8")

        config = load_config(config_file)

        assert config.version == "1.0"
        assert config.severity_threshold == Severity.MEDIUM  # default

    def test_load_config_with_full_yaml(
        self,
        full_config_yaml: str,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: Complete YAML with all sections
        WHEN: Loading config
        THEN: All values are parsed correctly
        """
        config_file = tmp_path / "config.yaml"
        config_file.write_text(full_config_yaml, encoding="utf-8")

        config = load_config(config_file)

        assert config.version == "1.0"
        assert "SEC001" in config.rules_enabled
        assert "SEC040" in config.rules_disabled
        assert "**/tests/**" in config.exclude_patterns
        assert config.severity_threshold == Severity.HIGH
        assert Severity.CRITICAL in config.fail_on
        assert config.max_file_size == 5_000_000
        assert config.max_findings == 500
        assert ".py" in config.file_extensions
        assert len(config.custom_rules) == 2
        assert config.ci_enabled is True
        assert config.ci_fail_on_warning is True
        assert config.precommit_staged_only is False

    def test_load_config_with_invalid_yaml_returns_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: File with invalid YAML syntax
        WHEN: Loading config
        THEN: Default config is returned
        """
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: [unclosed", encoding="utf-8")

        config = load_config(config_file)

        assert isinstance(config, SecurityConfig)
        assert config.version == "1.0"

    def test_load_config_with_non_dict_root_returns_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: YAML file with non-dict root (list)
        WHEN: Loading config
        THEN: Default config is returned
        """
        config_file = tmp_path / "list.yaml"
        config_file.write_text("- item1\n- item2", encoding="utf-8")

        config = load_config(config_file)

        assert isinstance(config, SecurityConfig)

    def test_load_config_with_oversized_file_returns_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: File larger than MAX_CONFIG_FILE_SIZE
        WHEN: Loading config
        THEN: Default config is returned
        """
        config_file = tmp_path / "huge.yaml"
        # Create file larger than MAX_CONFIG_FILE_SIZE
        huge_content = "x" * (MAX_CONFIG_FILE_SIZE + 1)
        config_file.write_text(huge_content, encoding="utf-8")

        config = load_config(config_file)

        assert isinstance(config, SecurityConfig)
        assert config.version == "1.0"


# =============================================================================
# TestBuildConfigFromDict
# =============================================================================


class TestBuildConfigFromDict:
    """Tests for _build_config_from_dict helper function."""

    def test_build_config_from_empty_dict(self) -> None:
        """
        GIVEN: Empty dictionary
        WHEN: Building config
        THEN: Config with defaults is returned
        """
        config = _build_config_from_dict({})

        assert config.version == "1.0"
        assert config.severity_threshold == Severity.MEDIUM

    def test_build_config_with_version(self) -> None:
        """
        GIVEN: Dict with version
        WHEN: Building config
        THEN: Version is set correctly
        """
        config = _build_config_from_dict({"version": "2.0"})

        assert config.version == "2.0"

    def test_build_config_with_rules_section(self) -> None:
        """
        GIVEN: Dict with rules section
        WHEN: Building config
        THEN: Rules are parsed correctly
        """
        data = {
            "rules": {
                "enabled": ["SEC001", "SEC002"],
                "disabled": ["SEC040"],
            }
        }

        config = _build_config_from_dict(data)

        assert config.rules_enabled == ["SEC001", "SEC002"]
        assert config.rules_disabled == ["SEC040"]

    def test_build_config_with_exclude_patterns(self) -> None:
        """
        GIVEN: Dict with exclude_patterns
        WHEN: Building config
        THEN: Patterns are set correctly
        """
        data = {"exclude_patterns": ["**/tests/**", "**/*.md"]}

        config = _build_config_from_dict(data)

        assert config.exclude_patterns == ["**/tests/**", "**/*.md"]

    def test_build_config_with_invalid_severity_uses_default(self) -> None:
        """
        GIVEN: Dict with invalid severity_threshold
        WHEN: Building config
        THEN: Default severity is used
        """
        data = {"severity_threshold": "invalid_severity"}

        config = _build_config_from_dict(data)

        assert config.severity_threshold == Severity.MEDIUM

    def test_build_config_with_fail_on_list(self) -> None:
        """
        GIVEN: Dict with fail_on list
        WHEN: Building config
        THEN: Severities are parsed correctly
        """
        data = {"fail_on": ["critical", "high", "medium"]}

        config = _build_config_from_dict(data)

        assert Severity.CRITICAL in config.fail_on
        assert Severity.HIGH in config.fail_on
        assert Severity.MEDIUM in config.fail_on

    def test_build_config_with_invalid_fail_on_skips_invalid(self) -> None:
        """
        GIVEN: Dict with invalid severity in fail_on
        WHEN: Building config
        THEN: Invalid severity is skipped
        """
        data = {"fail_on": ["critical", "invalid", "high"]}

        config = _build_config_from_dict(data)

        assert Severity.CRITICAL in config.fail_on
        assert Severity.HIGH in config.fail_on
        # Should only have 2 valid severities
        assert len(config.fail_on) == 2

    def test_build_config_with_scanner_section(self) -> None:
        """
        GIVEN: Dict with scanner section
        WHEN: Building config
        THEN: Scanner settings are set correctly
        """
        data = {
            "scanner": {
                "max_file_size": 5_000_000,
                "max_findings": 500,
                "file_extensions": [".py", ".js"],
            }
        }

        config = _build_config_from_dict(data)

        assert config.max_file_size == 5_000_000
        assert config.max_findings == 500
        assert config.file_extensions == [".py", ".js"]

    def test_build_config_with_ci_section(self) -> None:
        """
        GIVEN: Dict with ci section
        WHEN: Building config
        THEN: CI settings are set correctly
        """
        data = {
            "ci": {
                "enabled": True,
                "fail_on_warning": True,
                "github_comment": False,
                "max_display_findings": 15,
            }
        }

        config = _build_config_from_dict(data)

        assert config.ci_enabled is True
        assert config.ci_fail_on_warning is True
        assert config.ci_github_comment is False
        assert config.ci_max_display == 15

    def test_build_config_with_precommit_section(self) -> None:
        """
        GIVEN: Dict with precommit section
        WHEN: Building config
        THEN: Pre-commit settings are set correctly
        """
        data = {
            "precommit": {
                "enabled": False,
                "staged_only": False,
                "allow_warnings": False,
            }
        }

        config = _build_config_from_dict(data)

        assert config.precommit_enabled is False
        assert config.precommit_staged_only is False
        assert config.precommit_allow_warnings is False


# =============================================================================
# TestApplyConfigToScanner
# =============================================================================


class TestApplyConfigToScanner:
    """Tests for apply_config_to_scanner function."""

    def test_apply_config_adds_custom_rules(self) -> None:
        """
        GIVEN: Scanner and config with custom rules
        WHEN: Applying config
        THEN: Custom rules are added to scanner
        """
        scanner = create_scanner()
        initial_count = len(scanner.get_rules())

        pattern = re.compile("TEST_[A-Z]+")
        custom_rule = SecurityRule(
            rule_id="CUSTOM001",
            title="Test Rule",
            description="Test",
            category=FindingCategory.SECRETS,
            severity=Severity.HIGH,
            pattern=pattern,
            recommendation="Fix it",
        )
        config = SecurityConfig(custom_rules=[custom_rule])

        apply_config_to_scanner(scanner, config)

        assert len(scanner.get_rules()) == initial_count + 1

    def test_apply_config_disables_rules(self) -> None:
        """
        GIVEN: Scanner and config with disabled rules
        WHEN: Applying config
        THEN: Specified rules are disabled
        """
        scanner = create_scanner()
        config = SecurityConfig(rules_disabled=["SEC001", "SEC002"])

        apply_config_to_scanner(scanner, config)

        rules_dict = {r.rule_id: r for r in scanner.get_rules()}
        assert rules_dict["SEC001"].enabled is False
        assert rules_dict["SEC002"].enabled is False

    def test_apply_config_with_empty_config_changes_nothing(self) -> None:
        """
        GIVEN: Scanner and empty config
        WHEN: Applying config
        THEN: Scanner unchanged
        """
        scanner = create_scanner()
        initial_rules = scanner.get_rules()
        config = SecurityConfig()

        apply_config_to_scanner(scanner, config)

        assert len(scanner.get_rules()) == len(initial_rules)


# =============================================================================
# TestGetDefaultConfigPath
# =============================================================================


class TestGetDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_get_default_config_path_when_file_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        GIVEN: .ingestforge-security.yaml exists in cwd
        WHEN: Getting default config path
        THEN: Path to config is returned
        """
        config_file = tmp_path / ".ingestforge-security.yaml"
        config_file.write_text("version: '1.0'", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        result = get_default_config_path()

        assert result is not None
        assert result.name == ".ingestforge-security.yaml"
        assert result.exists()

    def test_get_default_config_path_when_file_not_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        GIVEN: No config file in cwd
        WHEN: Getting default config path
        THEN: None is returned
        """
        monkeypatch.chdir(tmp_path)

        result = get_default_config_path()

        assert result is None


# =============================================================================
# TestEdgeCases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_config_with_unicode_content(self, tmp_path: Path) -> None:
        """
        GIVEN: Config file with unicode characters
        WHEN: Loading config
        THEN: Config is loaded successfully
        """
        config_file = tmp_path / "unicode.yaml"
        unicode_yaml = """
version: "1.0"
custom_rules:
  - rule_id: "TEST001"
    title: "Test Rule with émojis 🔒"
    pattern: "SECRET"
    severity: "high"
    category: "secrets"
    recommendation: "Fix it"
"""
        config_file.write_text(unicode_yaml, encoding="utf-8")

        config = load_config(config_file)

        assert len(config.custom_rules) == 1

    def test_config_with_very_long_pattern(self, tmp_path: Path) -> None:
        """
        GIVEN: Config with very long regex pattern
        WHEN: Loading config
        THEN: Config is loaded successfully
        """
        long_pattern = "A" * 500
        config_file = tmp_path / "long.yaml"
        yaml_content = f"""
version: "1.0"
custom_rules:
  - rule_id: "TEST001"
    title: "Long Pattern"
    pattern: "{long_pattern}"
    severity: "high"
    category: "secrets"
    recommendation: "Fix"
"""
        config_file.write_text(yaml_content, encoding="utf-8")

        config = load_config(config_file)

        assert len(config.custom_rules) == 1

    def test_config_with_empty_sections(self, tmp_path: Path) -> None:
        """
        GIVEN: Config with empty sections
        WHEN: Loading config
        THEN: Defaults are used for empty sections
        """
        config_file = tmp_path / "empty.yaml"
        yaml_content = """
version: "1.0"
rules:
exclude_patterns:
custom_rules:
"""
        config_file.write_text(yaml_content, encoding="utf-8")

        config = load_config(config_file)

        assert config.rules_enabled == []
        assert config.exclude_patterns == []
        assert config.custom_rules == []
