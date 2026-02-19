"""List command - List all configuration settings."""

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.table import Table
from rich.tree import Tree
from ingestforge.cli.config.base import ConfigCommand


class ListCommand(ConfigCommand):
    """List all configuration settings."""

    def execute(
        self,
        section: Optional[str] = None,
        format_type: str = "table",
        project: Optional[Path] = None,
    ) -> int:
        """List configuration settings."""
        try:
            ctx = self.initialize_context(project, require_storage=False)
            config = ctx["config"]

            # Filter by section if specified
            if section:
                if section not in config:
                    self.print_error(f"Section not found: {section}")
                    return 1
                config_to_display = {section: config[section]}
            else:
                config_to_display = config

            # Display configuration
            if format_type == "tree":
                self._display_as_tree(config_to_display)
            else:
                self._display_as_table(config_to_display)

            return 0

        except Exception as e:
            return self.handle_error(e, "Configuration listing failed")

    def _display_as_table(self, config: Dict) -> None:
        """Display configuration as table."""
        self.console.print()
        self.console.print("[bold cyan]Configuration Settings[/bold cyan]\n")

        for section_name, section_data in config.items():
            if not isinstance(section_data, dict):
                continue

            table = Table(title=f"[{section_name}]", show_lines=True)
            table.add_column("Setting", width=30)
            table.add_column("Value", width=50)

            for key, value in section_data.items():
                # Format value
                if isinstance(value, (list, dict)):
                    value_str = str(value)[:80]
                else:
                    value_str = str(value)

                table.add_row(key, value_str)

            self.console.print(table)
            self.console.print()

    def _display_as_tree(self, config: Dict) -> None:
        """Display configuration as tree."""
        self.console.print()
        tree = Tree("[bold cyan]Configuration[/bold cyan]")

        for section_name, section_data in config.items():
            if isinstance(section_data, dict):
                section_branch = tree.add(f"[yellow]{section_name}[/yellow]")

                for key, value in section_data.items():
                    value_str = str(value)[:60]
                    section_branch.add(f"{key}: [cyan]{value_str}[/cyan]")
            else:
                tree.add(f"{section_name}: [cyan]{section_data}[/cyan]")

        self.console.print(tree)

    def _flatten_config(self, config: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested configuration."""
        flat = {}

        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                flat.update(self._flatten_config(value, full_key))
            else:
                flat[full_key] = value

        return flat


def command(
    section: Optional[str] = typer.Argument(
        None, help="Specific section to list (optional)"
    ),
    format_type: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Display format: table or tree",
    ),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """List all configuration settings.

    Displays current configuration in table or tree format.
    Optionally filter by specific section.

    Examples:
        # List all settings
        ingestforge config list

        # List specific section
        ingestforge config list llm

        # Display as tree
        ingestforge config list --format tree

        # List embedding settings
        ingestforge config list embedding
    """
    cmd = ListCommand()
    exit_code = cmd.execute(section, format_type, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
