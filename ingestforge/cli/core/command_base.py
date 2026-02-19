"""Base class for all CLI commands.

This module provides the foundation for all CLI commands, following
Commandments #4 (No Large Functions), #6 (Smallest Scope), and
#9 (Type Safety).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict

from rich.console import Console

from ingestforge.cli.core.progress import ProgressManager
from ingestforge.cli.core.error_handlers import CLIErrorHandler
from ingestforge.cli.core.initializers import CLIInitializer
from ingestforge.cli.core.validators import InputValidator


class IngestForgeCommand(ABC):
    """Abstract base class for all IngestForge CLI commands.

    Provides common functionality:
    - Console output (via ProgressManager)
    - Error handling (via CLIErrorHandler)
    - Project initialization (via CLIInitializer)
    - Input validation (via InputValidator)

    Subclasses must implement execute() method.

    Example:
        class MyCommand(IngestForgeCommand):
            def execute(self, arg1: str, arg2: int) -> int:
                # Command logic here
                return 0  # Success
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        """Initialize command.

        Args:
            console: Rich console (for testing, inject mock)
                    If None, uses default console from ProgressManager
        """
        self.console = console or Console()

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> int:
        """Execute the command.

        Subclasses must implement this method with their command logic.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Exit code (0 = success, non-zero = error)

        Example:
            def execute(self, name: str, count: int = 10) -> int:
                # Command implementation
                return 0
        """
        pass

    # === Project Management ===

    def get_project_path(self, project: Optional[Path] = None) -> Path:
        """Get project path from option or current directory.

        Args:
            project: Project path from --project option

        Returns:
            Resolved absolute project path
        """
        return CLIInitializer.get_project_path(project)

    def ensure_initialized(self, project_path: Path) -> None:
        """Verify project is initialized.

        Args:
            project_path: Project directory

        Raises:
            SystemExit: If project not initialized
        """
        CLIInitializer.ensure_project_initialized(project_path)

    def load_config(self, project_path: Optional[Path] = None) -> Any:
        """Load project configuration.

        Args:
            project_path: Project directory

        Returns:
            Config object

        Raises:
            SystemExit: If config loading fails
        """
        return CLIInitializer.load_config(project_path)

    def initialize_context(
        self,
        project_path: Optional[Path] = None,
        require_storage: bool = False,
        require_pipeline: bool = False,
    ) -> Dict[str, Any]:
        """Initialize command context with config, storage, and/or pipeline.

        This is the recommended way to initialize command dependencies.

        Args:
            project_path: Project directory
            require_storage: Whether to initialize storage backend
            require_pipeline: Whether to initialize pipeline

        Returns:
            Dict with 'config', 'project_path', 'storage', 'pipeline'

        Example:
            ctx = self.initialize_context(require_storage=True)
            results = ctx['storage'].search(query)
        """
        return CLIInitializer.initialize_for_command(
            project_path, require_storage, require_pipeline
        )

    # === Output Methods ===

    def print_success(self, message: str) -> None:
        """Print success message."""
        ProgressManager.print_success(message)

    def print_error(self, message: str) -> None:
        """Print error message."""
        ProgressManager.print_error(message)

    def print_warning(self, message: str) -> None:
        """Print warning message."""
        ProgressManager.print_warning(message)

    def print_info(self, message: str) -> None:
        """Print info message."""
        ProgressManager.print_info(message)

    # === Error Handling ===

    def handle_error(self, error: Exception, context: str = "") -> int:
        """Handle command error and return exit code.

        Does not exit - returns exit code for caller to decide.

        Args:
            error: Exception that occurred
            context: Optional context message

        Returns:
            Exit code (1 for error)
        """
        CLIErrorHandler.handle_error(error, context)
        return 1

    def exit_on_error(self, error: Exception, context: str = "") -> None:
        """Handle error and exit CLI.

        Args:
            error: Exception that occurred
            context: Optional context message

        Raises:
            SystemExit: Always exits
        """
        CLIErrorHandler.exit_on_error(error, context)

    # === Validation Methods ===

    def validate_project_name(self, name: str) -> None:
        """Validate project name.

        Args:
            name: Project name

        Raises:
            typer.BadParameter: If invalid
        """
        InputValidator.validate_project_name(name)

    def validate_file_path(
        self, path: Path, must_exist: bool = True, must_be_file: bool = True
    ) -> Path:
        """Validate file path.

        Args:
            path: File path
            must_exist: Whether file must exist
            must_be_file: Whether path must be a file

        Returns:
            Resolved absolute path

        Raises:
            typer.BadParameter: If invalid
        """
        return InputValidator.validate_file_path(path, must_exist, must_be_file)

    def validate_directory_path(
        self, path: Path, must_exist: bool = True, must_be_empty: bool = False
    ) -> Path:
        """Validate directory path.

        Args:
            path: Directory path
            must_exist: Whether directory must exist
            must_be_empty: Whether directory must be empty

        Returns:
            Resolved absolute path

        Raises:
            typer.BadParameter: If invalid
        """
        return InputValidator.validate_directory_path(path, must_exist, must_be_empty)

    def validate_k_value(self, k: int, min_k: int = 1, max_k: int = 100) -> None:
        """Validate k (number of results) parameter.

        Args:
            k: Number of results
            min_k: Minimum allowed
            max_k: Maximum allowed

        Raises:
            typer.BadParameter: If invalid
        """
        InputValidator.validate_k_value(k, min_k, max_k)

    def validate_non_empty_string(self, value: str, name: str = "value") -> None:
        """Validate string is non-empty.

        Args:
            value: String to validate
            name: Parameter name for error messages

        Raises:
            typer.BadParameter: If empty
        """
        InputValidator.validate_non_empty_string(value, name)


# Backwards compatibility alias
BaseCommand = IngestForgeCommand
