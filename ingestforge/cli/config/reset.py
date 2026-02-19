"""Reset command - Reset configuration to defaults.

Resets configuration to default values.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.config.base import ConfigCommand


class ResetCommand(ConfigCommand):
    """Reset configuration to defaults."""

    def execute(
        self,
        project: Optional[Path] = None,
        section: Optional[str] = None,
        force: bool = False,
    ) -> int:
        """
        Reset configuration.

        Rule #4: Reduced from 64 → 55 lines (shortened docstring)
        """
        try:
            # Get config path
            config_path = self.get_config_path(project)

            # Check if config exists
            config_exists = config_path.exists()

            # Confirm reset
            if not force and config_exists:
                if section:
                    msg = f"Reset section '{section}' to defaults?"
                else:
                    msg = "Reset entire configuration to defaults?"

                if not typer.confirm(msg):
                    self.print_info("Reset cancelled")
                    return 0

            # Load current config or start fresh
            if config_exists and section:
                config = self.load_config(config_path)
            else:
                config = {}

            # Get defaults
            defaults = self.get_default_config()

            # Reset section or entire config
            if section:
                self._reset_section(config, defaults, section)
            else:
                config = defaults

            # Save config
            self.save_config(config_path, config)

            # Confirm
            if section:
                self.print_success(f"Reset section '{section}' to defaults")
            else:
                self.print_success("Reset configuration to defaults")

            self.print_info(f"Config saved: {config_path}")

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to reset configuration")

    def _reset_section(self, config: dict, defaults: dict, section: str) -> None:
        """Reset specific section.

        Args:
            config: Current configuration
            defaults: Default configuration
            section: Section to reset

        Raises:
            ValueError: If section not found in defaults
        """
        if section not in defaults:
            raise ValueError(
                f"Unknown section: {section}. " f"Valid: {', '.join(defaults.keys())}"
            )

        config[section] = defaults[section]


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    section: Optional[str] = typer.Option(
        None, "--section", "-s", help="Reset specific section only"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Reset configuration to defaults.

    Resets configuration to default values. Can reset entire
    configuration or specific sections.

    Sections:
    - llm: LLM provider settings
    - embedding: Embedding model settings
    - storage: Storage backend settings
    - chunking: Chunking strategy settings
    - retrieval: Retrieval settings
    - processing: Processing settings

    Examples:
        # Reset entire configuration
        ingestforge config reset

        # Reset without confirmation
        ingestforge config reset --force

        # Reset specific section
        ingestforge config reset --section llm

        # Reset section for specific project
        ingestforge config reset -s storage -p /path/to/project
    """
    cmd = ResetCommand()
    exit_code = cmd.execute(project, section, force)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
