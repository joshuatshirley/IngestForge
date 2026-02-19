"""Integration Tests for Security CI Pipeline.

Security Shield CI
Tests configuration loading, pre-commit hooks, and CI integration.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestforge.core.security.config import (
    SecurityConfig,
    apply_config_to_scanner,
    get_default_config_path,
    load_config,
)
from ingestforge.core.security.scanner import (
    FindingCategory,
    Severity,
    create_scanner,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_config_yaml() -> str:
    """Create sample YAML config content."""
    return """
version: "1.0"

rules:
  disabled:
    - SEC040

exclude_patterns:
  - "**/tests/**"
  - "**/*.md"

severity_threshold: "high"

fail_on:
  - "critical"

scanner:
  max_file_size: 5000000
  max_findings: 500

custom_rules:
  - rule_id: "TEST001"
    title: "Test Pattern"
    description: "Test rule"
    pattern: "TEST_SECRET_[A-Z0-9]{16}"
    severity: "high"
    category: "secrets"
    recommendation: "Remove test secret"
    file_extensions:
      - ".py"

ci:
  enabled: true
  fail_on_warning: true
  github_comment: true
  max_display_findings: 10

precommit:
  enabled: true
  staged_only: false
  allow_warnings: false
"""


@pytest.fixture
def temp_config_file(sample_config_yaml: str) -> Path:
    """Create temporary config file."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(sample_config_yaml)
        return Path(f.name)


# =============================================================================
# TestConfigLoader
# =============================================================================


class TestConfigLoader:
    """Tests for configuration loading."""

    def test_load_config_from_file(self, temp_config_file: Path) -> None:
        """
        GIVEN: Valid YAML config file
        WHEN: Loading config
        THEN: Config is parsed correctly
        """
        config = load_config(temp_config_file)

        assert config.version == "1.0"
        assert "SEC040" in config.rules_disabled
        assert "**/tests/**" in config.exclude_patterns
        assert config.severity_threshold == Severity.HIGH
        assert Severity.CRITICAL in config.fail_on
        assert config.max_file_size == 5000000
        assert config.max_findings == 500
        assert len(config.custom_rules) == 1
        assert config.custom_rules[0].rule_id == "TEST001"
        assert config.ci_enabled is True
        assert config.ci_fail_on_warning is True

    def test_load_config_with_missing_file_uses_defaults(self) -> None:
        """
        GIVEN: Non-existent config file path
        WHEN: Loading config
        THEN: Default config is returned
        """
        config = load_config(Path("/nonexistent/config.yaml"))

        assert config.version == "1.0"
        assert config.severity_threshold == Severity.MEDIUM
        assert Severity.CRITICAL in config.fail_on
        assert Severity.HIGH in config.fail_on
        assert len(config.custom_rules) == 0

    def test_load_config_with_none_uses_defaults(self) -> None:
        """
        GIVEN: None config path
        WHEN: Loading config
        THEN: Default config is returned
        """
        config = load_config(None)

        assert isinstance(config, SecurityConfig)
        assert config.version == "1.0"
        assert config.ci_enabled is True

    def test_load_config_with_invalid_yaml_returns_defaults(
        self,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: Invalid YAML file
        WHEN: Loading config
        THEN: Default config is returned (error logged)
        """
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("invalid: yaml: content: [unclosed", encoding="utf-8")

        config = load_config(invalid_yaml)

        # Should return defaults on parse error
        assert isinstance(config, SecurityConfig)
        assert config.version == "1.0"

    def test_validate_custom_rules(self, temp_config_file: Path) -> None:
        """
        GIVEN: Config with custom rules
        WHEN: Loading config
        THEN: Custom rules are validated and loaded
        """
        config = load_config(temp_config_file)

        assert len(config.custom_rules) == 1
        rule = config.custom_rules[0]
        assert rule.rule_id == "TEST001"
        assert rule.title == "Test Pattern"
        assert rule.severity == Severity.HIGH
        assert rule.category == FindingCategory.SECRETS
        assert rule.pattern.pattern == "TEST_SECRET_[A-Z0-9]{16}"


# =============================================================================
# TestConfigApplication
# =============================================================================


class TestConfigApplication:
    """Tests for applying config to scanner."""

    def test_apply_config_to_scanner(self, temp_config_file: Path) -> None:
        """
        GIVEN: Scanner and config with custom rules
        WHEN: Applying config to scanner
        THEN: Scanner is configured correctly
        """
        scanner = create_scanner()
        config = load_config(temp_config_file)

        initial_rules = len(scanner.get_rules())
        apply_config_to_scanner(scanner, config)

        # Should have added custom rule
        assert len(scanner.get_rules()) == initial_rules + 1

        # Should have disabled SEC040
        for rule in scanner.get_rules():
            if rule.rule_id == "SEC040":
                assert rule.enabled is False

    def test_apply_config_disables_rules(self) -> None:
        """
        GIVEN: Config with disabled rules
        WHEN: Applying to scanner
        THEN: Rules are disabled
        """
        scanner = create_scanner()
        config = SecurityConfig(rules_disabled=["SEC001", "SEC002"])

        apply_config_to_scanner(scanner, config)

        disabled_rules = [r for r in scanner.get_rules() if not r.enabled]
        disabled_ids = {r.rule_id for r in disabled_rules}
        assert "SEC001" in disabled_ids
        assert "SEC002" in disabled_ids

    def test_apply_config_enables_specific_rules(self) -> None:
        """
        GIVEN: Config with specific enabled rules
        WHEN: Applying to scanner
        THEN: Only those rules are enabled
        """
        scanner = create_scanner()
        config = SecurityConfig(rules_enabled=["SEC001", "SEC002"])

        apply_config_to_scanner(scanner, config)

        enabled_rules = [r for r in scanner.get_rules() if r.enabled]
        enabled_ids = {r.rule_id for r in enabled_rules}

        # Only SEC001 and SEC002 should be enabled
        assert "SEC001" in enabled_ids
        assert "SEC002" in enabled_ids
        # Other rules should be disabled
        assert len(enabled_rules) == 2


# =============================================================================
# TestPreCommitIntegration
# =============================================================================


class TestPreCommitIntegration:
    """Tests for pre-commit hook integration."""

    def test_precommit_config_settings(self, temp_config_file: Path) -> None:
        """
        GIVEN: Config with pre-commit settings
        WHEN: Loading config
        THEN: Pre-commit settings are loaded
        """
        config = load_config(temp_config_file)

        assert config.precommit_enabled is True
        assert config.precommit_staged_only is False
        assert config.precommit_allow_warnings is False

    def test_precommit_default_settings(self) -> None:
        """
        GIVEN: Default config
        WHEN: Loading config
        THEN: Pre-commit defaults are correct
        """
        config = load_config(None)

        assert config.precommit_enabled is True
        assert config.precommit_staged_only is True
        assert config.precommit_allow_warnings is True


# =============================================================================
# TestCIIntegration
# =============================================================================


class TestCIIntegration:
    """Tests for CI/CD integration."""

    def test_ci_config_settings(self, temp_config_file: Path) -> None:
        """
        GIVEN: Config with CI settings
        WHEN: Loading config
        THEN: CI settings are loaded
        """
        config = load_config(temp_config_file)

        assert config.ci_enabled is True
        assert config.ci_fail_on_warning is True
        assert config.ci_github_comment is True
        assert config.ci_max_display == 10

    def test_ci_default_settings(self) -> None:
        """
        GIVEN: Default config
        WHEN: Loading config
        THEN: CI defaults are correct
        """
        config = load_config(None)

        assert config.ci_enabled is True
        assert config.ci_fail_on_warning is False
        assert config.ci_github_comment is True
        assert config.ci_max_display == 20


# =============================================================================
# TestDefaultConfigPath
# =============================================================================


class TestDefaultConfigPath:
    """Tests for default config path detection."""

    def test_get_default_config_path_when_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        GIVEN: .ingestforge-security.yaml exists in cwd
        WHEN: Getting default config path
        THEN: Path to config is returned
        """
        # Create config in temp directory
        config_file = tmp_path / ".ingestforge-security.yaml"
        config_file.write_text("version: '1.0'", encoding="utf-8")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        result = get_default_config_path()

        assert result is not None
        assert result.name == ".ingestforge-security.yaml"

    def test_get_default_config_path_when_not_exists(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        GIVEN: No config file in cwd
        WHEN: Getting default config path
        THEN: None is returned
        """
        # Change to temp directory with no config
        monkeypatch.chdir(tmp_path)

        result = get_default_config_path()

        assert result is None


# =============================================================================
# TestEndToEndCI
# =============================================================================


class TestEndToEndCI:
    """End-to-end tests for CI pipeline."""

    def test_full_ci_pipeline_with_config(self, temp_config_file: Path) -> None:
        """
        GIVEN: Config file and scanner
        WHEN: Running full CI pipeline
        THEN: Config is loaded, applied, and scan runs
        """
        # Load config
        config = load_config(temp_config_file)
        assert len(config.custom_rules) == 1

        # Create and configure scanner
        scanner = create_scanner()
        apply_config_to_scanner(scanner, config)

        # Verify custom rule was added
        rule_ids = {r.rule_id for r in scanner.get_rules()}
        assert "TEST001" in rule_ids

        # Run scan on temp directory (should find no issues)
        with tempfile.TemporaryDirectory() as tmpdir:
            report = scanner.scan_directory(Path(tmpdir))

            assert report.files_scanned >= 0
            assert report.exit_code == 0  # Clean scan
            assert report.completed_at is not None
