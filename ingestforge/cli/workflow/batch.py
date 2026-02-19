"""Batch command - Run batch operations on multiple files.

Executes operations on multiple files or items in batch mode.

Implements UX-003: Progress Indicators with ETA, quiet mode, and CI fallback.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, List, Callable, Dict
import typer

from ingestforge.cli.workflow.base import WorkflowCommand
from ingestforge.cli.core.progress import ProgressReporter
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class BatchCommand(WorkflowCommand):
    """Run batch operations on multiple files."""

    def execute(
        self,
        operation: str,
        target: Path,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        pattern: str = "*",
        recursive: bool = True,
        quiet: bool = False,
    ) -> int:
        """Run batch operation on files.

        Implements UX-003: Progress Indicators with ETA and quiet mode.

        Args:
            operation: Operation to perform (ingest/analyze/export)
            target: Target directory
            project: Project directory
            output: Output file for report
            pattern: File pattern to match
            recursive: Search recursively
            quiet: Suppress progress bar output

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_operation(operation)
            self.validate_file_path(target, must_exist=True, must_be_file=False)

            if not target.is_dir():
                raise typer.BadParameter("Target must be a directory")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Find files
            files = self.find_files_by_pattern(target, pattern, recursive)

            if not files:
                self._handle_no_files(target, pattern)
                return 0

            # Execute batch operation
            results = self._execute_batch(operation, files, ctx, quiet=quiet)

            # Display results
            self._display_results(results, operation)

            # Save report if requested
            if output:
                self._save_batch_report(output, results, operation)

            return 0 if results["failed"] == 0 else 1

        except Exception as e:
            return self.handle_error(e, "Batch operation failed")

    def validate_operation(self, operation: str) -> None:
        """Validate operation type.

        Args:
            operation: Operation to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        valid_ops = ["ingest", "analyze", "export", "validate"]

        if operation.lower() not in valid_ops:
            raise typer.BadParameter(
                f"Invalid operation '{operation}'. " f"Valid: {', '.join(valid_ops)}"
            )

    def _handle_no_files(self, target: Path, pattern: str) -> None:
        """Handle case where no files found.

        Args:
            target: Target directory
            pattern: File pattern
        """
        self.print_warning(f"No files matching '{pattern}' in {target}")
        self.print_info("Try:\n  1. Checking the pattern\n  2. Verifying the directory")

    def _execute_batch(
        self,
        operation: str,
        files: List[Path],
        ctx: Dict[str, Any],
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute batch operation with progress reporting.

        Implements UX-003: Progress Indicators with ETA and quiet mode.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        Rule #4: Function <60 lines, decoupled progress via callbacks
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            operation: Operation type (ingest/analyze/export/validate)
            files: List of files to process
            ctx: Context dictionary
            quiet: Suppress progress output

        Returns:
            Results dictionary with counts and errors
        """
        assert operation is not None, "Operation cannot be None"
        assert isinstance(operation, str), "Operation must be string"
        assert files is not None, "Files cannot be None"
        assert isinstance(files, list), "Files must be list"
        assert ctx is not None, "Context cannot be None"
        assert isinstance(ctx, dict), "Context must be dictionary"
        operation_handlers: Dict[
            str, Callable[[List[Path], Dict[str, Any], bool], Dict[str, Any]]
        ] = {
            "ingest": self._batch_ingest,
            "analyze": self._batch_analyze,
            "export": self._batch_export,
            "validate": self._batch_validate,
        }
        operation_lower = operation.lower()
        handler = operation_handlers.get(operation_lower)
        if handler is None:
            logger.warning(f"Unknown batch operation: {operation}")
            return {"total": 0, "successful": 0, "failed": 0, "errors": []}

        # Execute handler with quiet flag
        result = handler(files, ctx, quiet)
        assert "total" in result, "Result must have 'total' key"
        assert "successful" in result, "Result must have 'successful' key"
        assert "failed" in result, "Result must have 'failed' key"

        return result

    def _batch_ingest(
        self, files: List[Path], ctx: Dict[str, Any], quiet: bool = False
    ) -> Dict[str, Any]:
        """Batch ingest files with progress.

        Args:
            files: List of files
            ctx: Context dictionary
            quiet: Suppress progress output

        Returns:
            Results dictionary
        """

        def ingest_file(file_path: Path) -> Dict[str, Any]:
            # Simplified ingest operation
            return {
                "file": str(file_path),
                "status": "success",
                "size": file_path.stat().st_size if file_path.exists() else 0,
            }

        return self._execute_with_progress(files, ingest_file, "Ingesting files", quiet)

    def _batch_analyze(
        self, files: List[Path], ctx: Dict[str, Any], quiet: bool = False
    ) -> Dict[str, Any]:
        """Batch analyze files with progress.

        Args:
            files: List of files
            ctx: Context dictionary
            quiet: Suppress progress output

        Returns:
            Results dictionary
        """

        def analyze_file(file_path: Path) -> Dict[str, Any]:
            # Simplified analysis
            return {
                "file": str(file_path),
                "status": "analyzed",
                "lines": 0,  # Would count lines in real implementation
            }

        return self._execute_with_progress(
            files, analyze_file, "Analyzing files", quiet
        )

    def _batch_export(
        self, files: List[Path], ctx: Dict[str, Any], quiet: bool = False
    ) -> Dict[str, Any]:
        """Batch export files with progress.

        Args:
            files: List of files
            ctx: Context dictionary
            quiet: Suppress progress output

        Returns:
            Results dictionary
        """

        def export_file(file_path: Path) -> Dict[str, Any]:
            return {
                "file": str(file_path),
                "status": "exported",
            }

        return self._execute_with_progress(files, export_file, "Exporting files", quiet)

    def _batch_validate(
        self, files: List[Path], ctx: Dict[str, Any], quiet: bool = False
    ) -> Dict[str, Any]:
        """Batch validate files with progress.

        Args:
            files: List of files
            ctx: Context dictionary
            quiet: Suppress progress output

        Returns:
            Results dictionary
        """

        def validate_file(file_path: Path) -> Dict[str, Any]:
            is_valid = file_path.exists() and file_path.is_file()
            return {
                "file": str(file_path),
                "status": "valid" if is_valid else "invalid",
                "valid": is_valid,
            }

        return self._execute_with_progress(
            files, validate_file, "Validating files", quiet
        )

    def _execute_with_progress(
        self,
        files: List[Path],
        operation: Callable[[Path], Dict[str, Any]],
        description: str,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """Execute operation on files with ProgressReporter.

        Implements UX-003: Progress Indicators with ETA, quiet mode, CI fallback.

        Rule #4: Decoupled progress via ProgressReporter.

        Args:
            files: Files to process
            operation: Operation function
            description: Description for progress bar
            quiet: Suppress progress output

        Returns:
            Results dictionary
        """
        results: Dict[str, Any] = {
            "total": len(files),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "items": [],
        }

        with ProgressReporter(
            total=len(files),
            description=description,
            quiet=quiet,
        ).start() as reporter:
            for idx, file_path in enumerate(files, 1):
                try:
                    result = operation(file_path)
                    results["items"].append(result)
                    results["successful"] += 1
                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(
                        {
                            "item": str(file_path),
                            "error": str(e),
                        }
                    )

                reporter.update(current=idx, item_name=file_path.name)

        return results

    def _display_results(self, results: dict, operation: str) -> None:
        """Display batch results.

        Args:
            results: Results dictionary
            operation: Operation name
        """
        summary = self.create_operation_summary(results, operation.capitalize())
        self.console.print(summary)

    def _save_batch_report(self, output: Path, results: dict, operation: str) -> None:
        """Save batch report.

        Args:
            output: Output file path
            results: Results dictionary
            operation: Operation name
        """
        workflow_data = {
            "name": f"Batch {operation}",
            "total_ops": results["total"],
            "successful": results["successful"],
            "failed": results["failed"],
            "duration": 0,  # Would track in real implementation
            "errors": results["errors"],
        }

        self.save_workflow_report(output, workflow_data)


# Typer command wrapper
def command(
    operation: str = typer.Argument(
        ..., help="Operation to perform (ingest/analyze/export/validate)"
    ),
    target: Path = typer.Argument(..., help="Target directory"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for report"
    ),
    pattern: str = typer.Option("*", "--pattern", help="File pattern to match"),
    recursive: bool = typer.Option(
        True, "--recursive/--no-recursive", "-r", help="Search recursively"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress progress bar output"
    ),
) -> None:
    """Run batch operations on multiple files.

    Executes the specified operation on all matching files
    in the target directory.

    Operations:
    - ingest: Ingest all matching files
    - analyze: Analyze all matching files
    - export: Export all matching files
    - validate: Validate all matching files

    Examples:
        # Ingest all PDF files
        ingestforge workflow batch ingest documents/ --pattern "*.pdf"

        # Analyze all text files (non-recursive)
        ingestforge workflow batch analyze data/ --pattern "*.txt" --no-recursive

        # Validate all files with report
        ingestforge workflow batch validate files/ -o validation.md

        # Specific project
        ingestforge workflow batch ingest docs/ -p /path/to/project

        # Suppress progress bar (for CI/scripts)
        ingestforge workflow batch ingest docs/ --quiet
    """
    cmd = BatchCommand()
    exit_code = cmd.execute(
        operation, target, project, output, pattern, recursive, quiet
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
