"""Set command - Set configuration values.

Sets configuration values with validation.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.config.base import ConfigCommand
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class SetCommand(ConfigCommand):
    """Set configuration values."""

    def execute(
        self,
        key: str,
        value: str,
        project: Optional[Path] = None,
        value_type: str = "auto",
    ) -> int:
        """Set configuration value.

        Args:
            key: Configuration key (dot notation)
            value: Value to set
            project: Project directory
            value_type: Type of value (auto/string/int/float/bool)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate key
            if not key or "." not in key:
                raise typer.BadParameter(
                    "Key must be in dot notation (e.g., llm.model)"
                )

            # Get config path
            config_path = self.get_config_path(project)

            # Load current config
            config = self.load_config(config_path)

            # Convert value to appropriate type
            typed_value = self._convert_value(value, value_type)

            # Show current value if exists
            current = self.get_config_value(config, key)
            if current is not None:
                self.print_info(f"Current value: {current}")

            # Set new value
            self.set_config_value(config, key, typed_value)

            # Validate config
            is_valid, errors = self.validate_config_structure(config)
            if not is_valid:
                self.print_warning("Configuration validation warnings:")
                for error in errors:
                    self.console.print(f"  • {error}")

            # Save config
            self.save_config(config_path, config)

            # Confirm
            self.print_success(f"Set {key} = {typed_value}")
            self.print_info(f"Config saved: {config_path}")

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to set configuration")

    def _convert_value(self, value: str, value_type: str) -> any:
        """Convert string value to appropriate type.

        Args:
            value: String value
            value_type: Target type

        Returns:
            Converted value

        Raises:
            ValueError: If conversion fails
        """
        if value_type == "string":
            return value

        if value_type == "int":
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to int")

        if value_type == "float":
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"Cannot convert '{value}' to float")

        if value_type == "bool":
            lower = value.lower()
            if lower in ("true", "yes", "1"):
                return True
            if lower in ("false", "no", "0"):
                return False
            raise ValueError(f"Cannot convert '{value}' to bool")

        # Auto-detect type
        return self._auto_convert(value)

    def _auto_convert(self, value: str) -> any:
        """Auto-detect and convert value type.

        Args:
            value: String value

        Returns:
            Converted value
        """
        # Try bool
        lower = value.lower()
        if lower in ("true", "yes"):
            return True
        if lower in ("false", "no"):
            return False

        # Try int
        try:
            if "." not in value:
                return int(value)
        except ValueError as e:
            logger.debug(f"Failed to parse value as int: {e}")

        # Try float
        try:
            return float(value)
        except ValueError as e:
            logger.debug(f"Failed to parse value as float: {e}")

        # Default to string
        return value


# Typer command wrapper
def command(
    key: str = typer.Argument(..., help="Configuration key (dot notation)"),
    value: str = typer.Argument(..., help="Value to set"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    value_type: str = typer.Option(
        "auto",
        "--type",
        "-t",
        help="Value type (auto/string/int/float/bool)",
    ),
) -> None:
    """Set configuration value.

    Sets a configuration value with automatic type detection.
    Creates config file if it doesn't exist.

    Key format: Use dot notation to specify nested keys
    (e.g., "llm.model", "chunking.size")

    Type detection:
    - Numbers without decimals → int
    - Numbers with decimals → float
    - true/false/yes/no → bool
    - Everything else → string

    Examples:
        # Set LLM model
        ingestforge config set llm.model gpt-4

        # Set chunk size (auto-detected as int)
        ingestforge config set chunking.size 1500

        # Set temperature (auto-detected as float)
        ingestforge config set llm.temperature 0.8

        # Set boolean flag
        ingestforge config set retrieval.rerank true

        # Force type
        ingestforge config set custom.value "123" --type string

        # Specific project
        ingestforge config set llm.model gpt-4 -p /path/to/project
    """
    cmd = SetCommand()
    exit_code = cmd.execute(key, value, project, value_type)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
