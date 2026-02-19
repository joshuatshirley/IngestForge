"""Tests for workflow pipeline command.

Tests PipelineExecutor, PipelineLoader, and pipeline functionality.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestforge.cli.workflow.pipeline import (
    PipelineStep,
    PipelineConfig,
    StepResult,
    PipelineResult,
    PipelineLoader,
    VariableResolver,
    PipelineExecutor,
    PipelineValidator,
    PipelineCommand,
)


class TestPipelineStep:
    """Tests for PipelineStep dataclass."""

    def test_step_creation_defaults(self) -> None:
        """Test step creation with defaults."""
        step = PipelineStep(name="test", command="echo test")

        assert step.name == "test"
        assert step.command == "echo test"
        assert step.on_error == "fail"
        assert step.timeout == 3600
        assert step.depends_on == []

    def test_step_with_condition(self) -> None:
        """Test step with condition."""
        step = PipelineStep(
            name="conditional",
            command="cmd",
            condition="exists:/tmp/file",
        )

        assert step.condition == "exists:/tmp/file"

    def test_step_with_dependencies(self) -> None:
        """Test step with dependencies."""
        step = PipelineStep(
            name="step2",
            command="cmd",
            depends_on=["step1"],
        )

        assert "step1" in step.depends_on


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test config creation."""
        config = PipelineConfig(
            name="test-pipeline",
            description="Test description",
            steps=[
                PipelineStep(name="step1", command="cmd1"),
                PipelineStep(name="step2", command="cmd2"),
            ],
        )

        assert config.name == "test-pipeline"
        assert len(config.steps) == 2
        assert config.on_failure == "stop"

    def test_config_with_variables(self) -> None:
        """Test config with variables."""
        config = PipelineConfig(
            name="test",
            variables={"TARGET": "/data", "SIZE": "1000"},
        )

        assert config.variables["TARGET"] == "/data"
        assert config.variables["SIZE"] == "1000"


class TestStepResult:
    """Tests for StepResult dataclass."""

    def test_successful_result(self) -> None:
        """Test successful step result."""
        result = StepResult(
            name="test",
            success=True,
            duration=1.5,
            output="Success output",
        )

        assert result.success is True
        assert result.skipped is False
        assert result.duration == 1.5

    def test_failed_result(self) -> None:
        """Test failed step result."""
        result = StepResult(
            name="test",
            success=False,
            error="Command failed",
        )

        assert result.success is False
        assert result.error == "Command failed"

    def test_skipped_result(self) -> None:
        """Test skipped step result."""
        result = StepResult(
            name="test",
            success=True,
            skipped=True,
        )

        assert result.skipped is True


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_result_counts(self) -> None:
        """Test result counting methods."""
        result = PipelineResult(
            pipeline_name="test",
            steps=[
                StepResult(name="s1", success=True, skipped=False),
                StepResult(name="s2", success=True, skipped=True),
                StepResult(name="s3", success=False, skipped=False),
            ],
        )

        assert result.successful_steps == 1
        assert result.failed_steps == 1
        assert result.skipped_steps == 1

    def test_result_all_successful(self) -> None:
        """Test all_successful property."""
        result = PipelineResult(
            pipeline_name="test",
            steps=[
                StepResult(name="s1", success=True),
                StepResult(name="s2", success=True),
            ],
            all_successful=True,
        )

        assert result.all_successful is True


class TestVariableResolver:
    """Tests for VariableResolver class."""

    def test_resolve_pipeline_variable(self) -> None:
        """Test resolving pipeline variables."""
        resolver = VariableResolver({"NAME": "test"}, env_vars=False)

        result = resolver.resolve("Hello ${NAME}")

        assert result == "Hello test"

    def test_resolve_multiple_variables(self) -> None:
        """Test resolving multiple variables."""
        resolver = VariableResolver(
            {"A": "1", "B": "2"},
            env_vars=False,
        )

        result = resolver.resolve("${A} + ${B}")

        assert result == "1 + 2"

    def test_resolve_step_variables_override(self) -> None:
        """Test step variables override pipeline variables."""
        resolver = VariableResolver({"VAR": "pipeline"}, env_vars=False)

        result = resolver.resolve("Value: ${VAR}", {"VAR": "step"})

        assert result == "Value: step"

    def test_resolve_unknown_variable(self) -> None:
        """Test unknown variables are left as-is."""
        resolver = VariableResolver({}, env_vars=False)

        result = resolver.resolve("${UNKNOWN}")

        assert result == "${UNKNOWN}"


class TestPipelineValidator:
    """Tests for PipelineValidator class."""

    def test_validate_valid_config(self) -> None:
        """Test validating valid configuration."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="step1", command="cmd1"),
                PipelineStep(name="step2", command="cmd2"),
            ],
        )

        validator = PipelineValidator()
        errors = validator.validate(config)

        assert errors == []

    def test_validate_missing_name(self) -> None:
        """Test validating config without name."""
        config = PipelineConfig(
            name="",
            steps=[PipelineStep(name="step1", command="cmd")],
        )

        validator = PipelineValidator()
        errors = validator.validate(config)

        assert any("name is required" in e.lower() for e in errors)

    def test_validate_no_steps(self) -> None:
        """Test validating config without steps."""
        config = PipelineConfig(name="test", steps=[])

        validator = PipelineValidator()
        errors = validator.validate(config)

        assert any("at least one step" in e.lower() for e in errors)

    def test_validate_duplicate_step_names(self) -> None:
        """Test validating config with duplicate step names."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="step1", command="cmd1"),
                PipelineStep(name="step1", command="cmd2"),
            ],
        )

        validator = PipelineValidator()
        errors = validator.validate(config)

        assert any("duplicate" in e.lower() for e in errors)

    def test_validate_unknown_dependency(self) -> None:
        """Test validating config with unknown dependency."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="step1", command="cmd", depends_on=["nonexistent"]),
            ],
        )

        validator = PipelineValidator()
        errors = validator.validate(config)

        assert any("unknown step" in e.lower() for e in errors)

    def test_validate_circular_dependency(self) -> None:
        """Test validating config with circular dependency."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="a", command="cmd", depends_on=["b"]),
                PipelineStep(name="b", command="cmd", depends_on=["a"]),
            ],
        )

        validator = PipelineValidator()
        errors = validator.validate(config)

        assert any("circular" in e.lower() for e in errors)


class TestPipelineExecutor:
    """Tests for PipelineExecutor class."""

    def test_execute_simple_pipeline(self) -> None:
        """Test executing simple pipeline."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="echo", command="echo hello"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        assert result.all_successful is True
        assert len(result.steps) == 1
        assert result.steps[0].success is True

    def test_execute_dry_run(self) -> None:
        """Test dry run mode."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="dangerous", command="rm -rf /"),
            ],
        )

        executor = PipelineExecutor(dry_run=True)
        result = executor.execute(config)

        assert result.all_successful is True
        assert "DRY RUN" in result.steps[0].output

    def test_execute_with_target(self, tmp_path: Path) -> None:
        """Test execution with target variable."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="ls", command="echo ${TARGET}"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config, target=tmp_path)

        assert result.all_successful is True
        assert str(tmp_path) in result.steps[0].output

    def test_execute_failing_step(self) -> None:
        """Test execution with failing step."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="fail", command="exit 1"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        assert result.all_successful is False
        assert result.steps[0].success is False

    def test_execute_stops_on_failure(self) -> None:
        """Test execution stops on failure with on_failure=stop."""
        config = PipelineConfig(
            name="test",
            on_failure="stop",
            steps=[
                PipelineStep(name="fail", command="exit 1"),
                PipelineStep(name="never", command="echo never"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        assert len(result.steps) == 1  # Second step not executed

    def test_execute_skipped_dependency(self) -> None:
        """Test step skipped when dependency fails."""
        config = PipelineConfig(
            name="test",
            on_failure="continue",
            steps=[
                PipelineStep(name="fail", command="exit 1"),
                PipelineStep(name="depends", command="echo hi", depends_on=["fail"]),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        # depends should be skipped
        assert result.steps[1].skipped is True

    def test_execute_conditional_step_true(self) -> None:
        """Test conditional step when condition is true."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="conditional", command="echo hi", condition="true"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        assert result.steps[0].skipped is False
        assert result.steps[0].success is True

    def test_execute_conditional_step_false(self) -> None:
        """Test conditional step when condition is false."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="conditional", command="echo hi", condition="false"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        assert result.steps[0].skipped is True

    def test_execute_on_error_skip(self) -> None:
        """Test step with on_error=skip."""
        config = PipelineConfig(
            name="test",
            steps=[
                PipelineStep(name="fail", command="exit 1", on_error="skip"),
                PipelineStep(name="next", command="echo hi"),
            ],
        )

        executor = PipelineExecutor()
        result = executor.execute(config)

        # First step should be marked as skipped (not failed)
        assert result.steps[0].skipped is True
        assert result.steps[1].success is True


class TestPipelineLoader:
    """Tests for PipelineLoader class."""

    def test_load_basic_yaml(self, tmp_path: Path) -> None:
        """Test loading basic YAML configuration."""
        yaml_content = """
name: test-pipeline
description: A test pipeline
steps:
  - step1
  - step2
"""
        yaml_file = tmp_path / "pipeline.yaml"
        yaml_file.write_text(yaml_content)

        loader = PipelineLoader()
        config = loader.load(yaml_file)

        assert config.name == "test-pipeline"
        assert len(config.steps) == 2

    def test_parse_step_string(self) -> None:
        """Test parsing step from string."""
        loader = PipelineLoader()
        step = loader._parse_step("ingest")

        assert step.name == "ingest"
        assert step.command == "ingest"

    def test_parse_step_dict(self) -> None:
        """Test parsing step from dictionary."""
        loader = PipelineLoader()
        step = loader._parse_step(
            {
                "name": "ingest",
                "command": "ingestforge ingest ./docs",
                "on_error": "continue",
            }
        )

        assert step.name == "ingest"
        assert step.command == "ingestforge ingest ./docs"
        assert step.on_error == "continue"

    def test_parse_config_with_variables(self) -> None:
        """Test parsing config with variables."""
        loader = PipelineLoader()
        config = loader._parse_config(
            {
                "name": "test",
                "variables": {"TARGET": "/data"},
                "steps": ["step1"],
            }
        )

        assert config.variables["TARGET"] == "/data"


class TestPipelineCommand:
    """Tests for PipelineCommand class."""

    @pytest.fixture
    def temp_project(self, tmp_path: Path) -> Path:
        """Create temporary project directory."""
        project = tmp_path / "project"
        project.mkdir()
        (project / ".ingestforge").mkdir()
        return project

    @pytest.fixture
    def temp_target(self, tmp_path: Path) -> Path:
        """Create temporary target file."""
        target = tmp_path / "target.txt"
        target.write_text("test content")
        return target

    def test_create_config_predefined(self) -> None:
        """Test creating config from predefined name."""
        cmd = PipelineCommand()

        config = cmd._create_config("full", None)

        assert config.name == "full"
        assert len(config.steps) == 4

    def test_create_config_custom_steps(self) -> None:
        """Test creating config with custom steps."""
        cmd = PipelineCommand()

        config = cmd._create_config("custom", ["step1", "step2", "step3"])

        assert config.name == "custom"
        assert len(config.steps) == 3

    def test_validate_pipeline(self, temp_target: Path) -> None:
        """Test pipeline validation."""
        cmd = PipelineCommand()
        config = PipelineConfig(
            name="valid",
            steps=[PipelineStep(name="s1", command="cmd")],
        )

        exit_code = cmd._validate_pipeline(config)

        assert exit_code == 0

    def test_validate_pipeline_invalid(self, temp_target: Path) -> None:
        """Test validating invalid pipeline."""
        cmd = PipelineCommand()
        config = PipelineConfig(name="", steps=[])

        exit_code = cmd._validate_pipeline(config)

        assert exit_code == 1
