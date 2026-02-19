"""
Tests for Configuration Management.

This module tests the core configuration system including environment variable
expansion and config dataclasses.

Test Strategy
-------------
- Focus on public API: expand_env_vars(), Config dataclass, load_config()
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Avoid testing implementation details (validation internals, migrations)
- Each test should be self-contained and clear

Organization
------------
- TestExpandEnvVars: Environment variable expansion
- TestConfigDataclass: Config object creation and access
- TestConfigDefaults: Default values behavior
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from ingestforge.core.config import (
    Config,
    ProjectConfig,
    RedactionConfig,
)
from ingestforge.core.config_loaders import expand_env_vars


# ============================================================================
# Test Classes
# ============================================================================


class TestExpandEnvVars:
    """Tests for environment variable expansion.

    Rule #4: Focused test class - tests only expand_env_vars()
    """

    def test_simple_expansion(self):
        """Test basic ${VAR} expansion."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}):
            result = expand_env_vars("${TEST_VAR}")

            assert result == "test_value"

    def test_expansion_in_string(self):
        """Test ${VAR} expansion within a larger string."""
        with patch.dict(os.environ, {"API_KEY": "secret123"}):
            result = expand_env_vars("key: ${API_KEY}")

            assert result == "key: secret123"

    def test_multiple_expansions(self):
        """Test multiple ${VAR} expansions in one string."""
        with patch.dict(os.environ, {"HOST": "localhost", "PORT": "8080"}):
            result = expand_env_vars("http://${HOST}:${PORT}")

            assert result == "http://localhost:8080"

    def test_missing_var_empty_string(self):
        """Test expansion of missing variable returns empty string."""
        with patch.dict(os.environ, {}, clear=True):
            result = expand_env_vars("${NONEXISTENT_VAR}")

            assert result == ""

    def test_dict_expansion(self):
        """Test expansion in dictionary values."""
        with patch.dict(os.environ, {"KEY": "value123"}):
            data = {"api_key": "${KEY}", "host": "localhost"}

            result = expand_env_vars(data)

            assert result["api_key"] == "value123"
            assert result["host"] == "localhost"

    def test_list_expansion(self):
        """Test expansion in list items."""
        with patch.dict(os.environ, {"VAL1": "first", "VAL2": "second"}):
            data = ["${VAL1}", "${VAL2}", "third"]

            result = expand_env_vars(data)

            assert result == ["first", "second", "third"]

    def test_nested_expansion(self):
        """Test expansion in nested dict/list structures."""
        with patch.dict(os.environ, {"KEY": "secret"}):
            data = {"config": {"api": {"key": "${KEY}"}}}

            result = expand_env_vars(data)

            assert result["config"]["api"]["key"] == "secret"

    def test_non_string_passthrough(self):
        """Test non-string values pass through unchanged."""
        result_int = expand_env_vars(123)
        result_bool = expand_env_vars(True)
        result_none = expand_env_vars(None)

        assert result_int == 123
        assert result_bool is True
        assert result_none is None


class TestProjectConfig:
    """Tests for ProjectConfig dataclass.

    Rule #4: Focused test class - tests only ProjectConfig
    """

    def test_creation_with_defaults(self):
        """Test ProjectConfig with default values."""
        config = ProjectConfig()

        assert config.name == "my-knowledge-base"
        assert config.data_dir == ".data"
        assert config.ingest_dir == ".ingest"

    def test_creation_with_custom_values(self):
        """Test ProjectConfig with custom values."""
        config = ProjectConfig(
            name="my_project",
            data_dir="custom_data",
        )

        assert config.name == "my_project"
        assert config.data_dir == "custom_data"

    def test_name_attribute_access(self):
        """Test accessing name attribute."""
        config = ProjectConfig(name="test_name")

        assert config.name == "test_name"


class TestConfigDataclass:
    """Tests for Config dataclass.

    Rule #4: Focused test class - tests only Config creation
    """

    def test_creation_with_defaults(self):
        """Test Config with all default values."""
        config = Config()

        assert config.project is not None
        assert config.data_path is not None
        assert isinstance(config.data_path, Path)

    def test_project_config_nested(self):
        """Test accessing nested project config."""
        project = ProjectConfig(name="test_proj")
        config = Config(project=project)

        assert config.project.name == "test_proj"

    def test_data_path_is_path_object(self):
        """Test data_path is converted to Path object."""
        config = Config()

        assert isinstance(config.data_path, Path)

    def test_data_path_property(self):
        """Test data_path property returns a Path."""
        config = Config()

        # data_path is a computed property
        assert isinstance(config.data_path, Path)


class TestConfigFromYAML:
    """Tests for loading config from YAML.

    Rule #4: Focused test class - tests only YAML loading
    """

    def test_minimal_yaml_config(self, temp_dir):
        """Test loading minimal YAML config."""
        config_file = temp_dir / "config.yaml"
        config_data = {
            "project": {"name": "yaml_test"},
        }
        config_file.write_text(yaml.dump(config_data))

        # Just test that YAML can be loaded and parsed
        with open(config_file) as f:
            data = yaml.safe_load(f)

        assert data["project"]["name"] == "yaml_test"

    def test_yaml_with_env_vars(self, temp_dir):
        """Test YAML with environment variable placeholders."""
        config_file = temp_dir / "config.yaml"
        config_data = {
            "llm": {"api_key": "${TEST_API_KEY}"},
        }
        config_file.write_text(yaml.dump(config_data))

        with open(config_file) as f:
            data = yaml.safe_load(f)

        # Test env var expansion
        with patch.dict(os.environ, {"TEST_API_KEY": "secret"}):
            expanded = expand_env_vars(data)

            assert expanded["llm"]["api_key"] == "secret"


class TestRedactionConfig:
    """Tests for RedactionConfig dataclass (SEC-001.2).

    Rule #4: Focused test class - tests only RedactionConfig
    """

    def test_default_disabled(self):
        """Test redaction is disabled by default."""
        config = RedactionConfig()
        assert config.enabled is False

    def test_default_pii_types(self):
        """Test default PII types to redact."""
        config = RedactionConfig()
        assert "email" in config.types
        assert "phone" in config.types
        assert "ssn" in config.types
        assert "person_name" in config.types

    def test_empty_whitelist(self):
        """Test whitelist is empty by default."""
        config = RedactionConfig()
        assert config.whitelist == []

    def test_custom_types(self):
        """Test custom PII types."""
        config = RedactionConfig(types=["email", "credit_card"])
        assert config.types == ["email", "credit_card"]

    def test_invalid_type_raises(self):
        """Test invalid PII type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid PII type"):
            RedactionConfig(types=["invalid_type"])

    def test_whitelist_with_entries(self):
        """Test whitelist with entries."""
        config = RedactionConfig(whitelist=["safe@company.com", "John Smith"])
        assert len(config.whitelist) == 2
        assert "safe@company.com" in config.whitelist

    def test_whitelist_limit(self):
        """Test whitelist exceeds max raises error."""
        large_whitelist = [f"item{i}" for i in range(1001)]
        with pytest.raises(ValueError, match="Whitelist exceeds max"):
            RedactionConfig(whitelist=large_whitelist)

    def test_custom_patterns(self):
        """Test custom regex patterns."""
        config = RedactionConfig(custom_patterns={"employee_id": r"EMP-\d{6}"})
        assert "employee_id" in config.custom_patterns

    def test_custom_patterns_limit(self):
        """Test custom patterns exceeds max raises error."""
        many_patterns = {f"pattern{i}": r"\d+" for i in range(11)}
        with pytest.raises(ValueError, match="Custom patterns exceeds max"):
            RedactionConfig(custom_patterns=many_patterns)

    def test_mask_char(self):
        """Test custom mask character."""
        config = RedactionConfig(mask_char="X")
        assert config.mask_char == "X"

    def test_preserve_length(self):
        """Test preserve length option."""
        config = RedactionConfig(preserve_length=True)
        assert config.preserve_length is True

    def test_show_type(self):
        """Test show type option."""
        config = RedactionConfig(show_type=False)
        assert config.show_type is False


class TestConfigWithRedaction:
    """Tests for Config with redaction field."""

    def test_config_has_redaction(self):
        """Test Config includes redaction config."""
        config = Config()
        assert hasattr(config, "redaction")
        assert isinstance(config.redaction, RedactionConfig)

    def test_config_redaction_defaults(self):
        """Test redaction defaults in Config."""
        config = Config()
        assert config.redaction.enabled is False

    def test_config_from_dict_with_redaction(self):
        """Test Config.from_dict parses redaction config."""
        data = {
            "redaction": {
                "enabled": True,
                "types": ["email", "phone"],
                "whitelist": ["test@example.com"],
            }
        }
        config = Config.from_dict(data)
        assert config.redaction.enabled is True
        assert "email" in config.redaction.types
        assert "test@example.com" in config.redaction.whitelist


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Environment variable expansion: 8 tests
    - ProjectConfig dataclass: 3 tests
    - Config dataclass: 4 tests
    - YAML loading: 2 tests

    Total: 17 tests

Design Decisions:
    1. Focus on public API behaviors (expand_env_vars, Config creation)
    2. Don't test every nested config class - too much detail
    3. Don't test validation internals - implementation detail
    4. Don't test migrations or loaders - separate concern
    5. Simple, clear tests that verify expected behavior
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Justification:
    - Config is primarily a data container with defaults
    - Key behavior is env var expansion - thoroughly tested
    - YAML loading is library functionality - minimal testing
    - Validation and loaders can be tested separately if needed
    - Focus on what developers actually use: creating Config, env vars
"""
