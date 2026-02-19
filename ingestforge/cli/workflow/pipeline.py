"""Pipeline command - Execute multi-step workflows.

Executes predefined or custom multi-step pipelines with YAML support."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.table import Table

from ingestforge.cli.workflow.base import WorkflowCommand
from ingestforge.core.security.command import (
    CommandInjectionError,
    safe_run,
)


@dataclass
class PipelineStep:
    """Definition of a pipeline step."""

    name: str
    command: str
    description: str = ""
    condition: Optional[str] = None
    on_error: str = "fail"  # fail, skip, continue
    timeout: int = 3600
    variables: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Pipeline configuration from YAML."""

    name: str
    description: str = ""
    steps: List[PipelineStep] = field(default_factory=list)
    variables: Dict[str, str] = field(default_factory=dict)
    on_failure: str = "stop"  # stop, continue


@dataclass
class StepResult:
    """Result of executing a pipeline step."""

    name: str
    success: bool
    skipped: bool = False
    duration: float = 0.0
    output: str = ""
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Result of executing a pipeline."""

    pipeline_name: str
    steps: List[StepResult] = field(default_factory=list)
    all_successful: bool = True
    total_duration: float = 0.0

    @property
    def successful_steps(self) -> int:
        """Count successful steps."""
        return sum(1 for s in self.steps if s.success and not s.skipped)

    @property
    def failed_steps(self) -> int:
        """Count failed steps."""
        return sum(1 for s in self.steps if not s.success and not s.skipped)

    @property
    def skipped_steps(self) -> int:
        """Count skipped steps."""
        return sum(1 for s in self.steps if s.skipped)


class PipelineLoader:
    """Load pipeline configurations from YAML files."""

    def load(self, path: Path) -> PipelineConfig:
        """Load pipeline from YAML file.

        Args:
            path: Path to YAML file

        Returns:
            PipelineConfig object
        """
        try:
            import yaml
        except ImportError:
            # Fallback to basic parsing
            return self._load_basic(path)

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def _load_basic(self, path: Path) -> PipelineConfig:
        """Load pipeline with basic YAML parsing.

        Args:
            path: Path to YAML file

        Returns:
            PipelineConfig object
        """
        # Simple YAML parser for basic cases
        content = path.read_text(encoding="utf-8")
        data = self._parse_simple_yaml(content)
        return self._parse_config(data)

    def _parse_simple_yaml(self, content: str) -> Dict[str, Any]:
        """Parse simple YAML content.

        Args:
            content: YAML string

        Returns:
            Parsed dictionary
        """
        result: Dict[str, Any] = {}
        current_key = ""
        current_list: List[Any] = []
        in_list = False

        for line in content.split("\n"):
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            # Check for key-value
            if ":" in stripped and not stripped.startswith("-"):
                key, value = stripped.split(":", 1)
                key = key.strip()
                value = value.strip()

                if in_list and current_key:
                    result[current_key] = current_list
                    current_list = []
                    in_list = False

                if value:
                    result[key] = value
                else:
                    current_key = key
                    in_list = True

            elif stripped.startswith("-") and in_list:
                item = stripped[1:].strip()
                current_list.append(item)

        if in_list and current_key:
            result[current_key] = current_list

        return result

    def _parse_config(self, data: Dict[str, Any]) -> PipelineConfig:
        """Parse configuration dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            PipelineConfig object
        """
        name = data.get("name", "custom")
        description = data.get("description", "")
        variables = data.get("variables", {})
        on_failure = data.get("on_failure", "stop")

        steps_data = data.get("steps", [])
        steps = [self._parse_step(s) for s in steps_data]

        return PipelineConfig(
            name=name,
            description=description,
            steps=steps,
            variables=variables,
            on_failure=on_failure,
        )

    def _parse_step(self, data: Any) -> PipelineStep:
        """Parse step data.

        Args:
            data: Step data (string or dict)

        Returns:
            PipelineStep object
        """
        if isinstance(data, str):
            return PipelineStep(name=data, command=data)

        if isinstance(data, dict):
            return PipelineStep(
                name=data.get("name", "step"),
                command=data.get("command", data.get("name", "")),
                description=data.get("description", ""),
                condition=data.get("condition"),
                on_error=data.get("on_error", "fail"),
                timeout=data.get("timeout", 3600),
                variables=data.get("variables", {}),
                depends_on=data.get("depends_on", []),
            )

        return PipelineStep(name=str(data), command=str(data))


class VariableResolver:
    """Resolve variables in pipeline commands."""

    def __init__(
        self,
        pipeline_vars: Dict[str, str],
        env_vars: bool = True,
    ) -> None:
        """Initialize resolver.

        Args:
            pipeline_vars: Pipeline-defined variables
            env_vars: Whether to include environment variables
        """
        self.variables: Dict[str, str] = {}

        # Add environment variables
        if env_vars:
            self.variables.update(os.environ)

        # Pipeline variables override environment
        self.variables.update(pipeline_vars)

    def resolve(self, text: str, step_vars: Optional[Dict[str, str]] = None) -> str:
        """Resolve variables in text.

        Args:
            text: Text with variable placeholders
            step_vars: Step-specific variables

        Returns:
            Resolved text
        """
        # Merge step variables
        vars_to_use = {**self.variables}
        if step_vars:
            vars_to_use.update(step_vars)

        # Replace ${VAR} patterns
        result = text
        for name, value in vars_to_use.items():
            result = result.replace(f"${{{name}}}", value)

        return result


class PipelineExecutor:
    """Execute pipeline configurations."""

    def __init__(
        self,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize executor.

        Args:
            dry_run: If True, only show what would be done
            verbose: If True, show detailed output
        """
        self.dry_run = dry_run
        self.verbose = verbose
        self._completed_steps: Dict[str, StepResult] = {}

    def execute(
        self,
        config: PipelineConfig,
        target: Optional[Path] = None,
    ) -> PipelineResult:
        """Execute a pipeline configuration.

        Args:
            config: Pipeline configuration
            target: Optional target path

        Returns:
            PipelineResult
        """
        result = PipelineResult(pipeline_name=config.name)
        start_time = time.time()

        # Set up variable resolver with target
        extra_vars = {}
        if target:
            extra_vars["TARGET"] = str(target)
            extra_vars["TARGET_NAME"] = target.name
            extra_vars["TARGET_DIR"] = str(target.parent)

        resolver = VariableResolver({**config.variables, **extra_vars})

        # Execute steps
        for step in config.steps:
            step_result = self._execute_step(step, resolver, config.on_failure)
            result.steps.append(step_result)
            self._completed_steps[step.name] = step_result

            # Check if we should continue
            if not step_result.success and not step_result.skipped:
                if config.on_failure == "stop":
                    result.all_successful = False
                    break

        result.total_duration = time.time() - start_time
        result.all_successful = all(s.success or s.skipped for s in result.steps)

        return result

    def _execute_step(
        self,
        step: PipelineStep,
        resolver: VariableResolver,
        on_failure: str,
    ) -> StepResult:
        """Execute a single pipeline step.

        Args:
            step: Step to execute
            resolver: Variable resolver
            on_failure: Pipeline failure behavior

        Returns:
            StepResult
        """
        # Check dependencies
        for dep in step.depends_on:
            dep_result = self._completed_steps.get(dep)
            if not dep_result or not dep_result.success:
                return StepResult(
                    name=step.name,
                    success=False,
                    skipped=True,
                    error=f"Dependency not met: {dep}",
                )

        # Check condition
        if step.condition and not self._evaluate_condition(step.condition, resolver):
            return StepResult(
                name=step.name,
                success=True,
                skipped=True,
            )

        # Resolve command
        command = resolver.resolve(step.command, step.variables)

        if self.dry_run:
            return StepResult(
                name=step.name,
                success=True,
                skipped=False,
                output=f"[DRY RUN] Would execute: {command}",
            )

        # Execute command
        return self._run_command(step.name, command, step.timeout, step.on_error)

    def _evaluate_condition(self, condition: str, resolver: VariableResolver) -> bool:
        """Evaluate a step condition.

        Args:
            condition: Condition expression
            resolver: Variable resolver

        Returns:
            True if condition is met
        """
        resolved = resolver.resolve(condition)

        # Simple condition evaluation
        if resolved.lower() in ("true", "yes", "1"):
            return True
        if resolved.lower() in ("false", "no", "0", ""):
            return False

        # File existence check
        if resolved.startswith("exists:"):
            path = Path(resolved[7:].strip())
            return path.exists()

        return True

    def _run_command(
        self,
        name: str,
        command: str,
        timeout: int,
        on_error: str,
    ) -> StepResult:
        """Run a command.

        Args:
            name: Step name
            command: Command to run
            timeout: Timeout in seconds
            on_error: Error handling behavior

        Returns:
            StepResult

        Security:
            Uses safe_run to execute commands without shell=True,
            preventing shell injection attacks.
        """
        import subprocess

        start_time = time.time()

        try:
            # Execute command safely (no shell=True)
            result = safe_run(command, timeout=timeout)

            duration = time.time() - start_time
            success = result.returncode == 0

            if not success and on_error == "skip":
                return StepResult(
                    name=name,
                    success=True,
                    skipped=True,
                    duration=duration,
                    error=result.stderr,
                )

            return StepResult(
                name=name,
                success=success,
                duration=duration,
                output=result.stdout,
                error=result.stderr if not success else None,
            )

        except CommandInjectionError as e:
            # Command validation failed - potential injection attempt
            return StepResult(
                name=name,
                success=False,
                duration=time.time() - start_time,
                error=f"Command validation failed: {e}",
            )

        except subprocess.TimeoutExpired:
            return StepResult(
                name=name,
                success=False,
                duration=timeout,
                error=f"Command timed out after {timeout}s",
            )

        except Exception as e:
            return StepResult(
                name=name,
                success=False,
                duration=time.time() - start_time,
                error=str(e),
            )


class PipelineValidator:
    """Validate pipeline configurations."""

    def validate(self, config: PipelineConfig) -> List[str]:
        """Validate a pipeline configuration.

        Args:
            config: Configuration to validate

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []

        if not config.name:
            errors.append("Pipeline name is required")

        if not config.steps:
            errors.append("Pipeline must have at least one step")

        # Check for duplicate step names
        names = [s.name for s in config.steps]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            errors.append(f"Duplicate step names: {', '.join(set(duplicates))}")

        # Validate dependencies
        for step in config.steps:
            for dep in step.depends_on:
                if dep not in names:
                    errors.append(f"Step '{step.name}' depends on unknown step: {dep}")

        # Check for circular dependencies
        if self._has_circular_deps(config.steps):
            errors.append("Circular dependency detected in steps")

        return errors

    def _has_circular_deps(self, steps: List[PipelineStep]) -> bool:
        """Check for circular dependencies.

        Args:
            steps: List of steps

        Returns:
            True if circular dependency exists
        """
        visited: Dict[str, int] = {}  # 0=unvisited, 1=visiting, 2=visited
        step_map = {s.name: s for s in steps}

        def visit(name: str) -> bool:
            if name not in step_map:
                return False

            if visited.get(name, 0) == 1:
                return True  # Circular
            if visited.get(name, 0) == 2:
                return False  # Already checked

            visited[name] = 1
            for dep in step_map[name].depends_on:
                if visit(dep):
                    return True
            visited[name] = 2
            return False

        return any(visit(s.name) for s in steps)


class PipelineCommand(WorkflowCommand):
    """Execute multi-step workflows."""

    def execute(
        self,
        name: str,
        target: Path,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        steps: Optional[List[str]] = None,
        dry_run: bool = False,
        validate_only: bool = False,
    ) -> int:
        """Execute pipeline workflow.

        Args:
            name: Pipeline name or YAML file path
            target: Target file or directory
            project: Project directory
            output: Output file for report
            steps: Custom pipeline steps
            dry_run: If True, only show what would be done
            validate_only: If True, only validate the pipeline

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate target exists
            self.validate_file_path(target, must_exist=True, must_be_file=False)

            # Load or create pipeline config
            config = self._load_or_create_config(name, steps)

            if validate_only:
                return self._validate_pipeline(config)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Execute pipeline
            executor = PipelineExecutor(dry_run=dry_run)
            result = executor.execute(config, target)

            # Display results
            self._display_results(result, dry_run)

            # Save report
            if output:
                self._save_report(output, result)

            return 0 if result.all_successful else 1

        except Exception as e:
            return self.handle_error(e, "Pipeline execution failed")

    def _load_or_create_config(
        self, name: str, steps: Optional[List[str]]
    ) -> PipelineConfig:
        """Load config from file or create from predefined.

        Args:
            name: Pipeline name or file path
            steps: Custom steps (overrides predefined)

        Returns:
            PipelineConfig
        """
        # Check if name is a file path
        path = Path(name)
        if path.exists() and path.suffix in {".yaml", ".yml"}:
            loader = PipelineLoader()
            return loader.load(path)

        # Create from predefined or custom steps
        return self._create_config(name, steps)

    def _create_config(self, name: str, steps: Optional[List[str]]) -> PipelineConfig:
        """Create pipeline config from name or steps.

        Args:
            name: Pipeline name
            steps: Custom steps

        Returns:
            PipelineConfig
        """
        if steps:
            step_objs = [PipelineStep(name=s, command=s) for s in steps]
            return PipelineConfig(name="custom", steps=step_objs)

        # Predefined pipelines
        pipelines: Dict[str, List[str]] = {
            "full": ["ingest", "enrich", "index", "validate"],
            "basic": ["ingest", "index"],
            "analysis": ["analyze", "export"],
            "quality": ["validate", "analyze"],
        }

        step_names = pipelines.get(name.lower(), ["ingest"])
        step_objs = [
            PipelineStep(
                name=s,
                command=f"ingestforge {s} ${{TARGET}}",
            )
            for s in step_names
        ]

        return PipelineConfig(name=name, steps=step_objs)

    def _validate_pipeline(self, config: PipelineConfig) -> int:
        """Validate pipeline configuration.

        Args:
            config: Configuration to validate

        Returns:
            Exit code
        """
        validator = PipelineValidator()
        errors = validator.validate(config)

        if errors:
            self.print_error("Pipeline validation failed:")
            for error in errors:
                self.console.print(f"  - {error}")
            return 1

        self.print_success(f"Pipeline '{config.name}' is valid")
        self.console.print()
        self.print_info(f"Steps: {len(config.steps)}")
        for step in config.steps:
            self.console.print(f"  - {step.name}")

        return 0

    def _display_results(self, result: PipelineResult, dry_run: bool) -> None:
        """Display pipeline results.

        Args:
            result: Pipeline result
            dry_run: Whether this was a dry run
        """
        self.console.print()

        title = f"Pipeline: {result.pipeline_name}"
        if dry_run:
            title += " [DRY RUN]"

        table = Table(title=title)
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Duration", style="yellow")
        table.add_column("Details", style="dim")

        for step in result.steps:
            status, style = self._get_status_display(step)
            details = step.error[:40] if step.error else ""

            table.add_row(
                step.name.capitalize(),
                f"[{style}]{status}[/{style}]",
                f"{step.duration:.2f}s" if not step.skipped else "-",
                details,
            )

        self.console.print(table)

        # Summary
        self.console.print()
        self.print_info(f"Pipeline completed in {result.total_duration:.2f}s")
        self.print_info(
            f"Steps: {result.successful_steps} successful, "
            f"{result.failed_steps} failed, "
            f"{result.skipped_steps} skipped"
        )

        if not result.all_successful:
            self.print_warning("Pipeline completed with errors")

    def _get_status_display(self, step: StepResult) -> tuple[str, str]:
        """Get status display text and style.

        Args:
            step: Step result

        Returns:
            Tuple of (text, style)
        """
        if step.skipped:
            return "Skipped", "yellow"
        if step.success:
            return "Success", "green"
        return "Failed", "red"

    def _save_report(self, output: Path, result: PipelineResult) -> None:
        """Save pipeline report.

        Args:
            output: Output file path
            result: Pipeline result
        """
        workflow_data = {
            "name": f"Pipeline: {result.pipeline_name}",
            "total_ops": len(result.steps),
            "successful": result.successful_steps,
            "failed": result.failed_steps,
            "duration": result.total_duration,
            "errors": [
                {"operation": s.name, "error": s.error or ""}
                for s in result.steps
                if not s.success and not s.skipped
            ],
        }

        self.save_workflow_report(output, workflow_data)


def command(
    name: str = typer.Argument(
        ..., help="Pipeline name (full/basic/analysis/quality) or YAML file path"
    ),
    target: Path = typer.Argument(..., help="Target file or directory"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
    steps: Optional[List[str]] = typer.Option(
        None, "--step", "-s", help="Custom pipeline steps"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", "-n", help="Show what would be done without executing"
    ),
    validate: bool = typer.Option(
        False, "--validate", help="Validate pipeline configuration only"
    ),
) -> None:
    """Execute multi-step pipeline workflow.

    Runs predefined pipelines, custom step sequences, or YAML-defined pipelines.

    Predefined pipelines:
    - full: ingest -> enrich -> index -> validate
    - basic: ingest -> index
    - analysis: analyze -> export
    - quality: validate -> analyze

    YAML Pipeline Format:
        name: my-pipeline
        description: Custom ingestion pipeline
        variables:
          CHUNK_SIZE: "1000"
        steps:
          - name: ingest
            command: "ingestforge ingest ${TARGET} -c ${CHUNK_SIZE}"
          - name: analyze
            command: "ingestforge analyze ${TARGET}"
            depends_on: [ingest]
            on_error: continue

    Examples:
        # Run predefined pipeline
        ingestforge workflow pipeline full documents/paper.pdf

        # Run YAML-defined pipeline
        ingestforge workflow pipeline my-pipeline.yaml data/

        # Custom steps
        ingestforge workflow pipeline custom docs/ -s ingest -s analyze -s export

        # Dry run
        ingestforge workflow pipeline full data/ --dry-run

        # Validate only
        ingestforge workflow pipeline my-pipeline.yaml data/ --validate

        # With report
        ingestforge workflow pipeline full data/ -o report.md
    """
    cmd = PipelineCommand()
    exit_code = cmd.execute(name, target, project, output, steps, dry_run, validate)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
