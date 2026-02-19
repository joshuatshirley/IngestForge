"""Validate command - Validate configuration.

Validates configuration structure and values.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List
import typer

from ingestforge.cli.config.base import ConfigCommand


class ValidateCommand(ConfigCommand):
    """Validate configuration."""

    def execute(
        self,
        project: Optional[Path] = None,
        fix: bool = False,
    ) -> int:
        """Validate configuration.

        Args:
            project: Project directory
            fix: Attempt to fix issues

        Returns:
            0 if valid, 1 if invalid
        """
        try:
            # Get config path
            config_path = self.get_config_path(project)

            # Check if config exists
            if not config_path.exists():
                self.print_warning("No configuration file found")
                self.print_info("Using default configuration")
                return 0

            # Load config
            config = self.load_config(config_path)

            # Validate structure
            is_valid, errors = self.validate_config_structure(config)

            # Additional validations
            warnings = self._validate_values(config)

            # Display results
            self._display_results(is_valid, errors, warnings)

            # Fix if requested
            if not is_valid and fix:
                self._fix_config(config_path, config, errors)
                return 0

            return 0 if is_valid else 1

        except Exception as e:
            return self.handle_error(e, "Failed to validate configuration")

    def _validate_llm_settings(self, llm: dict, warnings: List[str]) -> None:
        """Validate LLM configuration settings.

        Rule #4: No large functions - Extracted from _validate_values
        """
        # Check temperature range
        temp = llm.get("temperature", 0.7)
        if not (0.0 <= temp <= 2.0):
            warnings.append(f"LLM temperature {temp} outside range [0.0, 2.0]")

        # Check max_tokens
        max_tokens = llm.get("max_tokens", 2000)
        if max_tokens < 100:
            warnings.append(f"LLM max_tokens {max_tokens} seems too low")

    def _validate_chunking_settings(self, chunking: dict, warnings: List[str]) -> None:
        """Validate chunking configuration settings.

        Rule #4: No large functions - Extracted from _validate_values
        """
        chunk_size = chunking.get("size", 1000)
        overlap = chunking.get("overlap", 100)

        if overlap >= chunk_size:
            warnings.append(f"Chunk overlap {overlap} >= size {chunk_size}")

        if chunk_size < 100:
            warnings.append(f"Chunk size {chunk_size} seems too small")

    def _validate_retrieval_settings(
        self, retrieval: dict, warnings: List[str]
    ) -> None:
        """Validate retrieval configuration settings.

        Rule #4: No large functions - Extracted from _validate_values
        """
        top_k = retrieval.get("top_k", 5)
        if top_k < 1:
            warnings.append(f"Retrieval top_k {top_k} must be >= 1")

        threshold = retrieval.get("score_threshold", 0.7)
        if not (0.0 <= threshold <= 1.0):
            warnings.append(f"Score threshold {threshold} outside range [0.0, 1.0]")

    def _validate_processing_settings(
        self, processing: dict, warnings: List[str]
    ) -> None:
        """Validate processing configuration settings.

        Rule #4: No large functions - Extracted from _validate_values
        """
        max_workers = processing.get("max_workers", 4)
        if max_workers < 1:
            warnings.append(f"Processing max_workers {max_workers} must be >= 1")

    def _validate_values(self, config: dict) -> List[str]:
        """Validate configuration values.

        Rule #4: No large functions - Refactored to <60 lines

        Args:
            config: Configuration dictionary

        Returns:
            List of warning messages
        """
        warnings: List[str] = []

        # Validate each section using helper methods
        if "llm" in config:
            self._validate_llm_settings(config["llm"], warnings)

        if "chunking" in config:
            self._validate_chunking_settings(config["chunking"], warnings)

        if "retrieval" in config:
            self._validate_retrieval_settings(config["retrieval"], warnings)

        if "processing" in config:
            self._validate_processing_settings(config["processing"], warnings)

        return warnings

    def _display_results(
        self, is_valid: bool, errors: List[str], warnings: List[str]
    ) -> None:
        """Display validation results.

        Args:
            is_valid: Whether config is valid
            errors: List of errors
            warnings: List of warnings
        """
        from rich.panel import Panel

        if is_valid and not warnings:
            panel = Panel(
                "[green][OK][/green] Configuration is valid",
                border_style="green",
                title="Validation",
            )
            self.console.print(panel)
            return

        # Show errors
        if errors:
            self.console.print()
            self.console.print("[bold red]Errors:[/bold red]")
            for error in errors:
                self.console.print(f"  [red][X][/red] {error}")

        # Show warnings
        if warnings:
            self.console.print()
            self.console.print("[bold yellow]Warnings:[/bold yellow]")
            for warning in warnings:
                self.console.print(f"  [yellow]![/yellow] {warning}")

        # Summary
        self.console.print()
        if is_valid:
            self.print_warning("Configuration has warnings")
        else:
            self.print_error("Configuration has errors")

    def _fix_config(self, config_path: Path, config: dict, errors: List[str]) -> None:
        """Attempt to fix configuration.

        Args:
            config_path: Path to config file
            config: Current configuration
            errors: List of errors to fix
        """
        self.print_info("Attempting to fix configuration...")

        defaults = self.get_default_config()

        self._add_missing_sections(config, defaults)
        self._add_missing_keys_in_sections(config, defaults)

        self.save_config(config_path, config)
        self.print_success("Configuration fixed and saved")

    def _add_missing_sections(self, config: dict, defaults: dict) -> None:
        """Add missing top-level sections to config.

        Args:
            config: Configuration dictionary to modify
            defaults: Default configuration with all sections
        """
        for section in defaults:
            if section not in config:
                config[section] = defaults[section]
                self.print_info(f"Added missing section: {section}")

    def _add_missing_keys_in_sections(
        self, config: dict, defaults: dict[str, Any]
    ) -> None:
        """Add missing keys within required sections.

        Args:
            config: Configuration dictionary to modify
            defaults: Default configuration with all keys
        """
        required_sections = ["llm", "embedding", "storage"]

        for section in required_sections:
            if section in config and section in defaults:
                self._add_missing_keys_in_section(config, defaults, section)

    def _add_missing_keys_in_section(
        self, config: dict, defaults: dict, section: str
    ) -> None:
        """Add missing keys within a specific section.

        Args:
            config: Configuration dictionary to modify
            defaults: Default configuration
            section: Section name to process
        """
        section_config = config[section]
        section_defaults = defaults[section]

        for key in section_defaults:
            if key not in section_config:
                section_config[key] = section_defaults[key]
                self.print_info(f"Added missing {section}.{key}")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    fix: bool = typer.Option(False, "--fix", help="Attempt to fix issues"),
) -> None:
    """Validate configuration.

    Validates configuration structure and values.
    Reports errors and warnings.

    Checks:
    - Required sections present
    - Required fields in each section
    - Value ranges (temperature, thresholds, etc.)
    - Logical constraints (overlap < chunk_size)

    Examples:
        # Validate configuration
        ingestforge config validate

        # Validate and fix issues
        ingestforge config validate --fix

        # Validate specific project
        ingestforge config validate -p /path/to/project
    """
    cmd = ValidateCommand()
    exit_code = cmd.execute(project, fix)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
