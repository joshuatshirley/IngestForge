"""Base class for storage CLI commands.

Provides common functionality for storage operations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.table import Table

from ingestforge.core.config import Config


class StorageCommand:
    """Base class for storage commands."""

    def __init__(self) -> None:
        """Initialize storage command."""
        self.console = Console()

    def get_config(self, project: Optional[Path] = None) -> Config:
        """Load configuration.

        Args:
            project: Project directory

        Returns:
            Config object
        """
        from ingestforge.core.config_loaders import load_config

        return load_config(base_path=project)

    def get_storage(self, config: Config) -> Any:
        """Get storage backend.

        Args:
            config: Configuration

        Returns:
            Storage backend
        """
        from ingestforge.storage.factory import get_storage_backend

        return get_storage_backend(config)

    def print_success(self, message: str) -> None:
        """Print success message.

        Args:
            message: Message to print
        """
        self.console.print(f"[green]SUCCESS[/green]: {message}")

    def print_error(self, message: str) -> None:
        """Print error message.

        Args:
            message: Message to print
        """
        self.console.print(f"[red]ERROR[/red]: {message}")

    def print_info(self, message: str) -> None:
        """Print info message.

        Args:
            message: Message to print
        """
        self.console.print(f"[blue]INFO[/blue]: {message}")

    def print_warning(self, message: str) -> None:
        """Print warning message.

        Args:
            message: Message to print
        """
        self.console.print(f"[yellow]WARNING[/yellow]: {message}")

    def create_stats_table(self, stats: Dict[str, Any]) -> Table:
        """Create a table for statistics display.

        Args:
            stats: Statistics dictionary

        Returns:
            Rich Table object
        """
        table = Table(title="Storage Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            # Format the key nicely
            display_key = key.replace("_", " ").title()
            table.add_row(display_key, str(value))

        return table

    def handle_error(self, error: Exception, context: str) -> int:
        """Handle error and return exit code.

        Args:
            error: Exception that occurred
            context: Context message

        Returns:
            Exit code (1)
        """
        self.print_error(f"{context}: {error}")
        return 1
