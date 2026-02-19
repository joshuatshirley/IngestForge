"""Show command - Display current configuration.

Shows the current configuration settings.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import typer

from ingestforge.cli.config.base import ConfigCommand


class ShowCommand(ConfigCommand):
    """Display current configuration."""

    def execute(
        self,
        project: Optional[Path] = None,
        key: Optional[str] = None,
        format: str = "summary",
    ) -> int:
        """Show configuration.

        Args:
            project: Project directory
            key: Specific key to show (dot notation)
            format: Output format (summary/full/json)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Get config path
            config_path = self.get_config_path(project)

            # Load config
            config = self.load_config(config_path)

            # Show specific key if requested
            if key:
                value = self.get_config_value(config, key)
                if value is None:
                    self.print_warning(f"Key not found: {key}")
                    return 1

                self.console.print(f"[cyan]{key}:[/cyan] {value}")
                return 0

            # Show configuration based on format
            if format == "json":
                self._show_json(config)
            elif format == "full":
                self._show_full(config)
            else:
                self._show_summary(config)

            # Show config path
            self.console.print()
            self.print_info(f"Config file: {config_path}")

            if not config_path.exists():
                self.print_warning("Using default configuration (no file found)")

            return 0

        except Exception as e:
            return self.handle_error(e, "Failed to show configuration")

    def _show_summary(self, config: dict) -> None:
        """Show configuration summary.

        Args:
            config: Configuration dictionary
        """
        from rich.panel import Panel

        lines = self.create_config_summary(config)
        panel = Panel("\n".join(lines), border_style="cyan", title="Configuration")
        self.console.print(panel)

    def _show_full(self, config: dict) -> None:
        """Show full configuration.

        Args:
            config: Configuration dictionary
        """
        self.console.print()
        self.console.print("[bold]Full Configuration:[/bold]")
        self.console.print()

        syntax = self.format_config_display(config)
        self.console.print(syntax)

    def _show_json(self, config: dict) -> None:
        """Show configuration as JSON.

        Args:
            config: Configuration dictionary
        """
        import json

        json_str = json.dumps(config, indent=2, ensure_ascii=False)
        self.console.print(json_str)


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    key: Optional[str] = typer.Option(
        None, "--key", "-k", help="Show specific key (dot notation)"
    ),
    format: str = typer.Option(
        "summary",
        "--format",
        "-f",
        help="Output format (summary/full/json)",
    ),
) -> None:
    """Show current configuration.

    Displays the current IngestForge configuration settings.
    Can show full config, summary, or specific keys.

    Formats:
    - summary: Key settings overview (default)
    - full: Complete configuration with syntax highlighting
    - json: Raw JSON output

    Examples:
        # Show configuration summary
        ingestforge config show

        # Show full configuration
        ingestforge config show --format full

        # Show as JSON
        ingestforge config show --format json

        # Show specific key
        ingestforge config show --key llm.model

        # Show for specific project
        ingestforge config show -p /path/to/project
    """
    cmd = ShowCommand()
    exit_code = cmd.execute(project, key, format)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
