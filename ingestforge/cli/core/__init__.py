"""CLI core utilities package.

This package provides reusable components for CLI commands:
- error_handlers: Centralized error handling with consistent formatting
- progress: Progress indicators and status messages
- initializers: Standard initialization patterns (config, storage, pipeline)
- validators: Input validation utilities
- command_base: Base class for all CLI commands
- results: Standard result types for operations

Usage:
    from ingestforge.cli.core import CLIInitializer, InputValidator
    from ingestforge.cli.core import ProgressManager, CLIErrorHandler
    from ingestforge.cli.core.command_base import IngestForgeCommand
    from ingestforge.cli.core.results import IngestResult

Example:
    class MyCommand(IngestForgeCommand):
        def execute(self, name: str) -> int:
            # Validate
            self.validate_non_empty_string(name, "name")

            # Initialize
            ctx = self.initialize_context(require_storage=True)

            # Execute with progress
            result = ProgressManager.run_with_spinner(
                lambda: do_work(ctx['storage']),
                "Processing...",
                "Complete!"
            )

            return 0
"""

from __future__ import annotations

# Import main classes for convenient access
from ingestforge.cli.core.error_handlers import CLIErrorHandler, cli_exception_handler
from ingestforge.cli.core.progress import (
    ProgressManager,
    BatchProgressTracker,
    ProgressReporter,
    is_interactive,
)
from ingestforge.cli.core.initializers import CLIInitializer
from ingestforge.cli.core.validators import InputValidator
from ingestforge.cli.core.command_base import IngestForgeCommand
from ingestforge.cli.core.results import (
    OperationResult,
    IngestResult,
    QueryResult,
    ResearchResult,
    ValidationResult,
    BatchProcessingResult,
)

__all__ = [
    # Error handling
    "CLIErrorHandler",
    "cli_exception_handler",
    # Progress reporting
    "ProgressManager",
    "BatchProgressTracker",
    "ProgressReporter",
    "is_interactive",
    # Initialization
    "CLIInitializer",
    # Validation
    "InputValidator",
    # Base command
    "IngestForgeCommand",
    # Result types
    "OperationResult",
    "IngestResult",
    "QueryResult",
    "ResearchResult",
    "ValidationResult",
    "BatchProcessingResult",
]
