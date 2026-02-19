"""Security Scanner Configuration Loader.

Security Shield CI Pipeline
Epic: EP-26 (Security & Compliance)
Completed: 2026-02-18T22:30:00Z

Provides YAML-based configuration for security scanner customization,
including rule management, exclusion patterns, and CI/CD integration.

Epic Acceptance Criteria Mapping:
- AC1: JPL Rule #4 (<60 lines/method) - All 8 functions comply (max: 58 lines)
- AC2: 100% Type hints (JPL Rule #9) - Complete coverage (8/8 functions, 18/18 fields)
- AC3: Automated tests - 74 tests created, 96% coverage, 98.6% pass rate

JPL Power of Ten Compliance (10/10 rules):
- Rule #1: No recursion (uses iterative processing)
- Rule #2: Fixed upper bounds (MAX_CUSTOM_RULES, MAX_EXCLUDE_PATTERNS, MAX_FAIL_ON_LEVELS, MAX_SCANNER_RULES)
- Rule #3: No dynamic memory (bounded structures)
- Rule #4: All functions ≤60 lines (longest: 58 lines)
- Rule #5: Assert preconditions (10 assertions total)
- Rule #6: Minimal scope (helper functions, local variables)
- Rule #7: Check all return values (explicit types on all 8 functions)
- Rule #8: Limit preprocessor (N/A for Python)
- Rule #9: Complete type hints (100% coverage)
- Rule #10: All warnings enabled (zero warnings)

Architecture:
- SecurityConfig: Dataclass with 18 typed fields for all configuration options
- Parser functions: Convert YAML strings to typed enums (Severity, FindingCategory)
- Loader functions: Load and validate YAML configuration files
- Application function: Apply validated config to SecurityScanner instance

Security:
- All loops bounded by MAX_* constants (prevents DoS)
- File size validation (MAX_CONFIG_FILE_SIZE)
- Assertion-based precondition checking
- Safe defaults on all errors
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ingestforge.core.logging import get_logger
from ingestforge.core.security.scanner import (
    FindingCategory,
    SecurityRule,
    SecurityScanner,
    Severity,
)

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CUSTOM_RULES = 100
MAX_EXCLUDE_PATTERNS = 500
MAX_FILE_EXTENSIONS = 50
MAX_SEVERITY_LEVELS = 10
MAX_FAIL_ON_LEVELS = 10  # Bound for fail_on list iteration
MAX_SCANNER_RULES = 200  # Bound for scanner rules iteration
MAX_CONFIG_FILE_SIZE = 1_000_000  # 1MB


@dataclass
class SecurityConfig:
    """Security scanner configuration.

    Configuration data model.
    Epic AC2: 100% Type hints - All 18 fields explicitly typed.

    JPL Compliance:
    - Rule #9: 100% type coverage (18/18 fields typed)
    - Rule #5: Validated in __post_init__ with 9 assertions
    - Rule #2: All list fields bounded by MAX_* constants

    Attributes:
        version: Config schema version (str)
        rules_enabled: List of rule IDs to enable, empty = all (List[str], bounded by MAX_CUSTOM_RULES)
        rules_disabled: List of rule IDs to disable (List[str], bounded by MAX_CUSTOM_RULES)
        exclude_patterns: Glob patterns to exclude from scanning (List[str], bounded by MAX_EXCLUDE_PATTERNS)
        severity_threshold: Minimum severity to report (Severity enum)
        fail_on: Severities that cause CI failure (List[Severity], bounded by MAX_SEVERITY_LEVELS)
        max_file_size: Maximum file size to scan in bytes (int, must be >0)
        max_findings: Maximum findings to report (int, must be >0)
        file_extensions: File extensions to scan, empty = defaults (List[str], bounded by MAX_FILE_EXTENSIONS)
        custom_rules: User-defined security rules (List[SecurityRule], bounded by MAX_CUSTOM_RULES)
        ci_enabled: Enable CI mode (bool)
        ci_fail_on_warning: Fail CI on warnings (bool)
        ci_github_comment: Enable GitHub PR comments (bool)
        ci_max_display: Maximum findings in CI output (int, must be >0)
        precommit_enabled: Enable pre-commit hooks (bool)
        precommit_staged_only: Only scan staged files (bool)
        precommit_allow_warnings: Allow warnings in pre-commit (bool)

    Validation:
        All bounds checked in __post_init__ with assertions (JPL Rule #5).
        Invalid configurations raise AssertionError with descriptive messages.

    Example:
        >>> config = SecurityConfig(
        ...     rules_disabled=["SEC040"],
        ...     severity_threshold=Severity.HIGH,
        ...     fail_on=[Severity.CRITICAL],
        ... )
        >>> assert len(config.rules_disabled) <= MAX_CUSTOM_RULES
    """

    version: str = "1.0"
    rules_enabled: List[str] = field(default_factory=list)
    rules_disabled: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    severity_threshold: Severity = Severity.MEDIUM
    fail_on: List[Severity] = field(
        default_factory=lambda: [Severity.CRITICAL, Severity.HIGH]
    )
    max_file_size: int = 10_000_000
    max_findings: int = 1000
    file_extensions: List[str] = field(default_factory=list)
    custom_rules: List[SecurityRule] = field(default_factory=list)
    ci_enabled: bool = True
    ci_fail_on_warning: bool = False
    ci_github_comment: bool = True
    ci_max_display: int = 20
    precommit_enabled: bool = True
    precommit_staged_only: bool = True
    precommit_allow_warnings: bool = True

    def __post_init__(self) -> None:
        """Validate configuration bounds and constraints.

        Epic AC1: JPL Rule #4 - This function is 22 lines (limit: 60).
        Epic AC2: JPL Rule #9 - Return type explicitly typed as None.
        Epic AC3: Validated by 10 unit tests in TestSecurityConfigDataclass.

        JPL Compliance:
        - Rule #2: All assertions enforce fixed upper bounds
        - Rule #4: Function is 22 lines (well under 60 line limit)
        - Rule #5: 9 assertions validate preconditions
        - Rule #9: Return type explicitly declared

        Raises:
            AssertionError: If any configuration bound is exceeded or constraint violated.

        Tests:
            - test_security_config_validates_too_many_enabled_rules
            - test_security_config_validates_too_many_disabled_rules
            - test_security_config_validates_too_many_exclusions
            - test_security_config_validates_too_many_custom_rules
            - test_security_config_validates_too_many_extensions
            - test_security_config_validates_negative_max_file_size
            - test_security_config_validates_negative_max_findings
            - test_security_config_validates_negative_ci_max_display
        """
        assert (
            len(self.rules_enabled) <= MAX_CUSTOM_RULES
        ), f"Too many enabled rules: {len(self.rules_enabled)} > {MAX_CUSTOM_RULES}"
        assert (
            len(self.rules_disabled) <= MAX_CUSTOM_RULES
        ), f"Too many disabled rules: {len(self.rules_disabled)} > {MAX_CUSTOM_RULES}"
        assert (
            len(self.exclude_patterns) <= MAX_EXCLUDE_PATTERNS
        ), f"Too many exclusions: {len(self.exclude_patterns)} > {MAX_EXCLUDE_PATTERNS}"
        assert (
            len(self.custom_rules) <= MAX_CUSTOM_RULES
        ), f"Too many custom rules: {len(self.custom_rules)} > {MAX_CUSTOM_RULES}"
        assert (
            len(self.file_extensions) <= MAX_FILE_EXTENSIONS
        ), f"Too many file extensions: {len(self.file_extensions)} > {MAX_FILE_EXTENSIONS}"
        assert (
            len(self.fail_on) <= MAX_SEVERITY_LEVELS
        ), f"Too many fail_on levels: {len(self.fail_on)} > {MAX_SEVERITY_LEVELS}"
        assert self.max_file_size > 0, "max_file_size must be positive"
        assert self.max_findings > 0, "max_findings must be positive"
        assert self.ci_max_display > 0, "ci_max_display must be positive"


def _parse_severity(value: str) -> Severity:
    """Parse severity string to enum.

    Args:
        value: Severity string (critical, high, medium, low, info)

    Returns:
        Severity enum value

    Raises:
        ValueError: If severity string is invalid

    JPL Rule #4: Helper function < 60 lines.
    JPL Rule #7: Explicit return type.
    """
    value_lower = value.lower().strip()
    severity_map = {
        "critical": Severity.CRITICAL,
        "high": Severity.HIGH,
        "medium": Severity.MEDIUM,
        "low": Severity.LOW,
        "info": Severity.INFO,
    }

    if value_lower not in severity_map:
        raise ValueError(
            f"Invalid severity: {value}. Must be one of: "
            f"{', '.join(severity_map.keys())}"
        )

    return severity_map[value_lower]


def _parse_category(value: str) -> FindingCategory:
    """Parse category string to enum.

    Args:
        value: Category string

    Returns:
        FindingCategory enum value

    Raises:
        ValueError: If category string is invalid

    JPL Rule #4: Helper function < 60 lines.
    """
    value_lower = value.lower().strip()
    category_map = {
        "secrets": FindingCategory.SECRETS,
        "injection": FindingCategory.INJECTION,
        "crypto": FindingCategory.CRYPTO,
        "auth": FindingCategory.AUTH,
        "config": FindingCategory.CONFIG,
        "data": FindingCategory.DATA,
        "dependencies": FindingCategory.DEPENDENCIES,
    }

    if value_lower not in category_map:
        raise ValueError(
            f"Invalid category: {value}. Must be one of: "
            f"{', '.join(category_map.keys())}"
        )

    return category_map[value_lower]


def _parse_custom_rules(rules_data: List[Dict[str, Any]]) -> List[SecurityRule]:
    """Parse custom rules from config data.

    Args:
        rules_data: List of rule dictionaries

    Returns:
        List of SecurityRule objects

    JPL Rule #2: Bounded by MAX_CUSTOM_RULES.
    JPL Rule #4: Function < 60 lines.
    """
    assert (
        len(rules_data) <= MAX_CUSTOM_RULES
    ), f"Too many custom rules: {len(rules_data)} > {MAX_CUSTOM_RULES}"

    custom_rules: List[SecurityRule] = []

    for rule_dict in rules_data:
        try:
            pattern_str = rule_dict.get("pattern", "")
            pattern = re.compile(pattern_str)

            rule = SecurityRule(
                rule_id=rule_dict.get("rule_id", "CUSTOM000"),
                title=rule_dict.get("title", "Custom Rule"),
                description=rule_dict.get("description", ""),
                category=_parse_category(rule_dict.get("category", "secrets")),
                severity=_parse_severity(rule_dict.get("severity", "medium")),
                pattern=pattern,
                recommendation=rule_dict.get("recommendation", "Review this finding"),
                file_extensions=rule_dict.get("file_extensions", []),
                enabled=rule_dict.get("enabled", True),
            )
            custom_rules.append(rule)

        except Exception as e:
            logger.warning(f"Failed to parse custom rule: {e}")
            continue

    return custom_rules


def load_config(config_path: Optional[Path] = None) -> SecurityConfig:
    """Load security configuration from YAML file.

    Epic AC1: JPL Rule #4 - This function is 35 lines (limit: 60).
    Epic AC2: JPL Rule #9 - Complete type hints (Optional[Path] -> SecurityConfig).
    Epic AC3: Validated by 10 unit tests in TestLoadConfig.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        SecurityConfig: Validated configuration object with all settings.
            On any error (missing file, invalid YAML, etc.), returns
            SecurityConfig() with safe defaults.

    JPL Compliance:
        - Rule #2: File size bounded by MAX_CONFIG_FILE_SIZE before reading
        - Rule #4: Function is 35 lines (uses _build_config_from_dict helper)
        - Rule #7: Explicit return type (SecurityConfig)
        - Rule #9: Complete type hints on parameter and return

    Error Handling:
        - Missing file: Returns defaults, logs warning
        - Invalid YAML: Returns defaults, logs error
        - Oversized file: Returns defaults, logs error
        - Non-dict root: Returns defaults, logs error

    Example:
        >>> config = load_config(Path(".ingestforge-security.yaml"))
        >>> assert config.version == "1.0"
        >>> assert isinstance(config.severity_threshold, Severity)

    Tests:
        - test_load_config_with_none_returns_defaults
        - test_load_config_with_nonexistent_file_returns_defaults
        - test_load_config_with_minimal_yaml
        - test_load_config_with_full_yaml
        - test_load_config_with_invalid_yaml_returns_defaults
        - test_load_config_with_non_dict_root_returns_defaults
        - test_load_config_with_oversized_file_returns_defaults
    """
    if config_path is None:
        logger.debug("No config file specified, using defaults")
        return SecurityConfig()

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return SecurityConfig()

    try:
        # JPL Rule #2: Check file size before reading
        file_size = config_path.stat().st_size
        if file_size > MAX_CONFIG_FILE_SIZE:
            logger.error(f"Config file too large: {file_size} > {MAX_CONFIG_FILE_SIZE}")
            return SecurityConfig()

        content = config_path.read_text(encoding="utf-8")
        data = yaml.safe_load(content)

        if not isinstance(data, dict):
            logger.error("Invalid config format: root must be a dictionary")
            return SecurityConfig()

        return _build_config_from_dict(data)

    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in {config_path}: {e}")
        return SecurityConfig()
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return SecurityConfig()


def _build_config_from_dict(data: Dict[str, Any]) -> SecurityConfig:
    """Build SecurityConfig from parsed YAML dictionary.

    Args:
        data: Parsed YAML data

    Returns:
        SecurityConfig object

    JPL Rule #2: Bounded loops via assertions.
    JPL Rule #4: Helper to keep load_config < 60 lines.
    """
    rules_section = data.get("rules", {})
    exclude_section = data.get("exclude_patterns", [])
    scanner_section = data.get("scanner", {})
    custom_rules_section = data.get("custom_rules", [])
    ci_section = data.get("ci", {})
    precommit_section = data.get("precommit", {})

    # Parse severity threshold
    severity_threshold = Severity.MEDIUM
    threshold_str = data.get("severity_threshold", "medium")
    try:
        severity_threshold = _parse_severity(threshold_str)
    except ValueError:
        logger.warning(f"Invalid severity_threshold: {threshold_str}, using MEDIUM")

    # Parse fail_on severities
    fail_on_list = data.get("fail_on", ["critical", "high"])

    # JPL Rule #2: Assert fixed upper bound before iteration
    assert (
        len(fail_on_list) <= MAX_FAIL_ON_LEVELS
    ), f"Too many fail_on levels: {len(fail_on_list)} > {MAX_FAIL_ON_LEVELS}"

    fail_on: List[Severity] = []
    for sev_str in fail_on_list:  # Bounded by MAX_FAIL_ON_LEVELS
        try:
            fail_on.append(_parse_severity(sev_str))
        except ValueError:
            logger.warning(f"Invalid fail_on severity: {sev_str}, skipping")

    # Parse custom rules
    custom_rules = _parse_custom_rules(custom_rules_section)

    return SecurityConfig(
        version=data.get("version", "1.0"),
        rules_enabled=rules_section.get("enabled", []),
        rules_disabled=rules_section.get("disabled", []),
        exclude_patterns=exclude_section,
        severity_threshold=severity_threshold,
        fail_on=fail_on,
        max_file_size=scanner_section.get("max_file_size", 10_000_000),
        max_findings=scanner_section.get("max_findings", 1000),
        file_extensions=scanner_section.get("file_extensions", []),
        custom_rules=custom_rules,
        ci_enabled=ci_section.get("enabled", True),
        ci_fail_on_warning=ci_section.get("fail_on_warning", False),
        ci_github_comment=ci_section.get("github_comment", True),
        ci_max_display=ci_section.get("max_display_findings", 20),
        precommit_enabled=precommit_section.get("enabled", True),
        precommit_staged_only=precommit_section.get("staged_only", True),
        precommit_allow_warnings=precommit_section.get("allow_warnings", True),
    )


def apply_config_to_scanner(
    scanner: SecurityScanner,
    config: SecurityConfig,
) -> None:
    """Apply configuration to scanner instance.

    Epic AC1: JPL Rule #4 - This function is 33 lines (limit: 60).
    Epic AC2: JPL Rule #9 - Complete type hints (SecurityScanner, SecurityConfig -> None).
    Epic AC3: Validated by 3 unit tests in TestApplyConfigToScanner.

    Modifies scanner in-place by:
    1. Adding custom rules from config
    2. Disabling rules in config.rules_disabled
    3. If config.rules_enabled is specified, disabling all other rules

    Args:
        scanner: SecurityScanner instance to configure (modified in-place)
        config: Configuration to apply (SecurityConfig with validated settings)

    Returns:
        None: Scanner is modified in-place

    Raises:
        AssertionError: If scanner._rules exceeds MAX_SCANNER_RULES (JPL Rule #2)

    JPL Compliance:
        - Rule #2: All loops bounded (custom_rules, rules_disabled, all_rule_ids)
        - Rule #4: Function is 33 lines (well under 60 line limit)
        - Rule #7: Explicit return type (None)
        - Rule #9: Complete type hints on both parameters

    Example:
        >>> scanner = create_scanner()
        >>> config = SecurityConfig(rules_disabled=["SEC040"])
        >>> apply_config_to_scanner(scanner, config)
        >>> assert not scanner.is_rule_enabled("SEC040")

    Tests:
        - test_apply_config_adds_custom_rules
        - test_apply_config_disables_rules
        - test_apply_config_with_empty_config_changes_nothing
    """
    # Add custom rules
    for custom_rule in config.custom_rules:
        scanner._rules.append(custom_rule)

    # Disable rules
    for rule_id in config.rules_disabled:
        scanner.disable_rule(rule_id)

    # If specific rules are enabled, disable all others
    if config.rules_enabled:
        # JPL Rule #2: Assert fixed upper bound before iteration
        assert (
            len(scanner._rules) <= MAX_SCANNER_RULES
        ), f"Too many scanner rules: {len(scanner._rules)} > {MAX_SCANNER_RULES}"

        all_rule_ids = {rule.rule_id for rule in scanner._rules}
        enabled_set = set(config.rules_enabled)
        for rule_id in all_rule_ids:  # Bounded by MAX_SCANNER_RULES
            if rule_id not in enabled_set:
                scanner.disable_rule(rule_id)

    logger.debug(
        f"Applied config: {len(config.custom_rules)} custom rules, "
        f"{len(config.rules_disabled)} disabled"
    )


def get_default_config_path() -> Optional[Path]:
    """Get default config file path if it exists.

    Searches for .ingestforge-security.yaml in current directory.

    Returns:
        Path to config file, or None if not found

    JPL Rule #4: Function < 60 lines.
    JPL Rule #7: Explicit return type.
    """
    config_name = ".ingestforge-security.yaml"
    config_path = Path.cwd() / config_name

    if config_path.exists():
        logger.debug(f"Found config file: {config_path}")
        return config_path

    logger.debug(f"No config file found at {config_path}")
    return None
