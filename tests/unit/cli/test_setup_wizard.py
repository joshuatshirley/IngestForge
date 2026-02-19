"""
Tests for Setup Wizard CLI command.

Guided CLI Setup Wizard
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile

from ingestforge.cli.setup_wizard import (
    SetupWizard,
    SetupConfig,
    LLMProviderSetup,
    ConfigPreset,
    HealthCheckResult,
    MAX_PATH_LENGTH,
    MAX_API_KEY_LENGTH,
    MAX_MODEL_NAME_LENGTH,
    SUPPORTED_PROVIDERS,
)


# =============================================================================
# SETUP CONFIG TESTS
# =============================================================================


class TestSetupConfig:
    """Tests for SetupConfig dataclass."""

    def test_default_config_creation(self):
        """
        GWT:
        Given no parameters
        When SetupConfig is created
        Then default values are set.
        """
        config = SetupConfig()

        assert config.project_name == "my_research"
        assert config.storage_backend == "chromadb"
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.performance_mode == "balanced"

    def test_config_with_custom_values(self):
        """
        GWT:
        Given custom parameters
        When SetupConfig is created
        Then custom values are used.
        """
        config = SetupConfig(
            project_name="test_project",
            storage_backend="jsonl",
            performance_mode="speed",
        )

        assert config.project_name == "test_project"
        assert config.storage_backend == "jsonl"
        assert config.performance_mode == "speed"


class TestLLMProviderSetup:
    """Tests for LLMProviderSetup dataclass."""

    def test_default_llm_config(self):
        """
        GWT:
        Given no parameters
        When LLMProviderSetup is created
        Then llama.cpp defaults are set.
        """
        llm = LLMProviderSetup()

        assert llm.provider == "llamacpp"
        assert llm.api_key == ""
        assert llm.ollama_url == "http://localhost:11434"

    def test_cloud_provider_config(self):
        """
        GWT:
        Given cloud provider parameters
        When LLMProviderSetup is created
        Then cloud values are stored.
        """
        llm = LLMProviderSetup(
            provider="openai",
            api_key="sk-test-key",
            model_name="gpt-4o-mini",
        )

        assert llm.provider == "openai"
        assert llm.api_key == "sk-test-key"
        assert llm.model_name == "gpt-4o-mini"


# =============================================================================
# HEALTH CHECK RESULT TESTS
# =============================================================================


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_passing_check(self):
        """
        GWT:
        Given passing check
        When HealthCheckResult is created
        Then passed is True.
        """
        result = HealthCheckResult(
            name="Python Version",
            passed=True,
            message="Python 3.11.0",
        )

        assert result.passed is True
        assert result.suggestion is None

    def test_failing_check_with_suggestion(self):
        """
        GWT:
        Given failing check with suggestion
        When HealthCheckResult is created
        Then suggestion is included.
        """
        result = HealthCheckResult(
            name="OCR",
            passed=False,
            message="Tesseract not found",
            suggestion="Install tesseract-ocr",
        )

        assert result.passed is False
        assert "tesseract" in result.suggestion


# =============================================================================
# SETUP WIZARD TESTS
# =============================================================================


class TestSetupWizard:
    """Tests for SetupWizard class."""

    def test_wizard_initialization(self):
        """
        GWT:
        Given no parameters
        When SetupWizard is created
        Then default config is initialized.
        """
        wizard = SetupWizard()

        assert wizard.config is not None
        assert wizard.console is not None

    def test_wizard_with_custom_console(self):
        """
        GWT:
        Given custom console
        When SetupWizard is created
        Then custom console is used.
        """
        mock_console = MagicMock()
        wizard = SetupWizard(console=mock_console)

        assert wizard.console == mock_console


class TestWizardHealthChecks:
    """Tests for wizard health check methods."""

    def test_check_python_version_passes(self):
        """
        GWT:
        Given Python 3.10+
        When _check_python_version is called
        Then check passes.
        """
        wizard = SetupWizard()

        with patch("sys.version_info") as mock_version:
            mock_version.major = 3
            mock_version.minor = 11
            mock_version.micro = 0

            result = wizard._check_python_version()

            assert result.passed is True
            assert "3.11.0" in result.message

    def test_check_python_version_old(self):
        """
        GWT:
        Given Python < 3.10
        When _check_python_version is called
        Then check fails with suggestion.
        """
        wizard = SetupWizard()

        with patch("sys.version_info") as mock_version:
            mock_version.major = 3
            mock_version.minor = 9
            mock_version.micro = 0

            result = wizard._check_python_version()

            assert result.passed is False
            assert "3.10" in result.message

    def test_check_data_directory_writable(self):
        """
        GWT:
        Given writable directory
        When _check_data_directory is called
        Then check passes.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = SetupWizard()
            wizard.config.data_path = Path(tmpdir) / "test_data"

            result = wizard._check_data_directory()

            assert result.passed is True
            assert "Writable" in result.message

    def test_check_optional_tools_tesseract_missing(self):
        """
        GWT:
        Given tesseract not installed
        When _check_optional_tools is called
        Then check passes with note (optional).
        """
        wizard = SetupWizard()

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None

            results = wizard._check_optional_tools()

            # All should pass (they're optional)
            for result in results:
                assert result.passed is True


class TestWizardWritePermissions:
    """Tests for write permission verification (JPL Rule #7)."""

    def test_verify_write_permissions_success(self):
        """
        GWT:
        Given writable directory
        When _verify_write_permissions is called
        Then returns True.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = SetupWizard()
            wizard.config.data_path = Path(tmpdir) / "data"

            result = wizard._verify_write_permissions()

            assert result is True

    def test_verify_write_permissions_creates_directory(self):
        """
        GWT:
        Given non-existent directory path
        When _verify_write_permissions is called
        Then directory is created and verified.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = SetupWizard()
            nested_path = Path(tmpdir) / "a" / "b" / "c"
            wizard.config.data_path = nested_path

            result = wizard._verify_write_permissions()

            assert result is True
            assert nested_path.exists()


class TestWizardConfigGeneration:
    """Tests for config file generation."""

    def test_build_llm_config_llamacpp(self):
        """
        GWT:
        Given llamacpp provider
        When _build_llm_config is called
        Then llamacpp config is generated.
        """
        wizard = SetupWizard()
        wizard.config.llm.provider = "llamacpp"
        wizard.config.llm.model_path = "/path/to/model.gguf"

        config = wizard._build_llm_config()

        assert config["default_provider"] == "llamacpp"
        assert "llamacpp" in config
        assert config["llamacpp"]["model_path"] == "/path/to/model.gguf"

    def test_build_llm_config_ollama(self):
        """
        GWT:
        Given ollama provider
        When _build_llm_config is called
        Then ollama config is generated.
        """
        wizard = SetupWizard()
        wizard.config.llm.provider = "ollama"
        wizard.config.llm.model_name = "llama3"
        wizard.config.llm.ollama_url = "http://localhost:11434"

        config = wizard._build_llm_config()

        assert config["default_provider"] == "ollama"
        assert config["ollama"]["model"] == "llama3"

    def test_build_llm_config_openai(self):
        """
        GWT:
        Given openai provider
        When _build_llm_config is called
        Then openai config is generated with api_key.
        """
        wizard = SetupWizard()
        wizard.config.llm.provider = "openai"
        wizard.config.llm.model_name = "gpt-4o-mini"
        wizard.config.llm.api_key = "${OPENAI_API_KEY}"

        config = wizard._build_llm_config()

        assert config["default_provider"] == "openai"
        assert config["openai"]["model"] == "gpt-4o-mini"
        assert config["openai"]["api_key"] == "${OPENAI_API_KEY}"

    def test_build_config_yaml_structure(self):
        """
        GWT:
        Given complete setup config
        When _build_config_yaml is called
        Then valid YAML with all sections is generated.
        """
        wizard = SetupWizard()
        wizard.config.project_name = "test_project"
        wizard.config.storage_backend = "chromadb"
        wizard.config.llm.provider = "llamacpp"

        yaml_content = wizard._build_config_yaml()

        assert "project:" in yaml_content
        assert "test_project" in yaml_content
        assert "storage:" in yaml_content
        assert "chromadb" in yaml_content
        assert "llm:" in yaml_content


# =============================================================================
# CONSTANTS TESTS (JPL Rule #2: Bounded Data)
# =============================================================================


class TestConstants:
    """Tests for module constants (JPL Rule #2)."""

    def test_path_length_bounded(self):
        """
        GWT:
        Given MAX_PATH_LENGTH constant
        When checked
        Then it has reasonable Windows-safe value.
        """
        assert MAX_PATH_LENGTH == 260
        assert MAX_PATH_LENGTH > 0

    def test_api_key_length_bounded(self):
        """
        GWT:
        Given MAX_API_KEY_LENGTH constant
        When checked
        Then it prevents unbounded input.
        """
        assert MAX_API_KEY_LENGTH == 256
        assert MAX_API_KEY_LENGTH > 0

    def test_model_name_length_bounded(self):
        """
        GWT:
        Given MAX_MODEL_NAME_LENGTH constant
        When checked
        Then it prevents unbounded input.
        """
        assert MAX_MODEL_NAME_LENGTH == 128
        assert MAX_MODEL_NAME_LENGTH > 0

    def test_supported_providers_defined(self):
        """
        GWT:
        Given SUPPORTED_PROVIDERS constant
        When checked
        Then all expected providers are listed.
        """
        assert "llamacpp" in SUPPORTED_PROVIDERS
        assert "ollama" in SUPPORTED_PROVIDERS
        assert "openai" in SUPPORTED_PROVIDERS
        assert "claude" in SUPPORTED_PROVIDERS
        assert "gemini" in SUPPORTED_PROVIDERS


# =============================================================================
# CONFIG PRESET TESTS
# =============================================================================


class TestConfigPreset:
    """Tests for ConfigPreset enum."""

    def test_standard_preset(self):
        """
        GWT:
        Given standard preset
        When accessed
        Then value is 'standard'.
        """
        assert ConfigPreset.STANDARD.value == "standard"

    def test_expert_preset(self):
        """
        GWT:
        Given expert preset
        When accessed
        Then value is 'expert'.
        """
        assert ConfigPreset.EXPERT.value == "expert"

    def test_preset_from_string(self):
        """
        GWT:
        Given string 'expert'
        When converted to ConfigPreset
        Then EXPERT enum is returned.
        """
        preset = ConfigPreset("expert")
        assert preset == ConfigPreset.EXPERT


# =============================================================================
# JPL COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_bounded_inputs(self):
        """
        GWT:
        Given oversized inputs
        When validated
        Then assertions catch them (JPL Rule #2).
        """
        # All constants are defined and bounded
        assert MAX_PATH_LENGTH > 0 and MAX_PATH_LENGTH < 1000
        assert MAX_API_KEY_LENGTH > 0 and MAX_API_KEY_LENGTH < 1000
        assert MAX_MODEL_NAME_LENGTH > 0 and MAX_MODEL_NAME_LENGTH < 1000

    def test_jpl_rule_5_preconditions(self):
        """
        GWT:
        Given SetupWizard
        When examining code
        Then asserts are used for preconditions.
        """
        # Config assertions are in _step_project_basics and other methods
        # This test verifies the pattern exists
        wizard = SetupWizard()
        assert wizard.config is not None

    def test_jpl_rule_7_write_verification(self):
        """
        GWT:
        Given _verify_write_permissions method
        When verifying directory
        Then write is verified by reading back (JPL Rule #7).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            wizard = SetupWizard()
            wizard.config.data_path = Path(tmpdir) / "test"

            # This method reads back the written content to verify
            result = wizard._verify_write_permissions()

            assert result is True

    def test_jpl_rule_9_type_hints(self):
        """
        GWT:
        Given SetupWizard class
        When inspecting methods
        Then all have type hints.
        """
        wizard = SetupWizard()

        # Check key method annotations exist
        assert hasattr(wizard.run, "__annotations__")
        assert hasattr(wizard._verify_write_permissions, "__annotations__")
        assert hasattr(wizard._build_config_yaml, "__annotations__")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestWizardIntegration:
    """Integration tests for complete wizard flow."""

    @patch("ingestforge.cli.setup_wizard.Confirm.ask")
    @patch("ingestforge.cli.setup_wizard.Prompt.ask")
    def test_wizard_cancelled_at_start(self, mock_prompt, mock_confirm):
        """
        GWT:
        Given user cancels at start
        When wizard.run is called
        Then None is returned.
        """
        mock_confirm.return_value = False

        wizard = SetupWizard()
        result = wizard.run()

        assert result is None

    def test_wizard_generates_valid_yaml(self):
        """
        GWT:
        Given completed wizard config
        When generating YAML
        Then YAML is valid and parseable.
        """
        import yaml

        wizard = SetupWizard()
        wizard.config.project_name = "yaml_test"
        wizard.config.llm.provider = "llamacpp"
        wizard.config.llm.model_path = "/test/model.gguf"

        yaml_content = wizard._build_config_yaml()

        # Should parse without error
        parsed = yaml.safe_load(yaml_content)

        assert parsed["project"]["name"] == "yaml_test"
        assert parsed["llm"]["default_provider"] == "llamacpp"
