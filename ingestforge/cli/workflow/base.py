"""Base class for workflow commands.

Provides common functionality for workflow automation and batch operations.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import List, Dict, Any, Callable
from pathlib import Path
import time

from ingestforge.cli.core import IngestForgeCommand


class WorkflowCommand(IngestForgeCommand):
    """Base class for workflow commands."""

    def execute_batch_operation(
        self,
        items: List[Any],
        operation: Callable[[Any], Dict[str, Any]],
        description: str,
    ) -> Dict[str, Any]:
        """Execute operation on batch of items.

        Args:
            items: List of items to process
            operation: Operation function
            description: Description for progress

        Returns:
            Results dictionary
        """
        from rich.progress import Progress, SpinnerColumn, TextColumn

        results = {
            "total": len(items),
            "successful": 0,
            "failed": 0,
            "errors": [],
            "items": [],
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task(f"{description}...", total=len(items))

            for item in items:
                try:
                    result = operation(item)
                    results["items"].append(result)
                    results["successful"] += 1

                except Exception as e:
                    results["failed"] += 1
                    results["errors"].append(
                        {
                            "item": str(item),
                            "error": str(e),
                        }
                    )

                progress.advance(task)

        return results

    def validate_file_list(self, files: List[Path]) -> None:
        """Validate list of files.

        Args:
            files: List of file paths

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if not files:
            raise typer.BadParameter("No files provided")

        # Check if any files exist
        existing = [f for f in files if f.exists()]

        if not existing:
            raise typer.BadParameter("None of the provided files exist")

    def find_files_by_pattern(
        self, directory: Path, pattern: str, recursive: bool = True
    ) -> List[Path]:
        """Find files matching pattern.

        Args:
            directory: Directory to search
            pattern: File pattern (e.g., "*.pdf")
            recursive: Search recursively

        Returns:
            List of matching file paths
        """
        if recursive:
            return sorted(directory.rglob(pattern))
        else:
            return sorted(directory.glob(pattern))

    def group_files_by_extension(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by extension.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping extension to files
        """
        grouped: Dict[str, List[Path]] = {}

        for file_path in files:
            ext = file_path.suffix or ".no_extension"

            if ext not in grouped:
                grouped[ext] = []

            grouped[ext].append(file_path)

        return grouped

    def create_operation_summary(
        self, results: Dict[str, Any], operation_name: str
    ) -> str:
        """Create summary of batch operation results.

        Args:
            results: Results dictionary
            operation_name: Name of operation

        Returns:
            Formatted summary string
        """
        lines = [
            f"\n{operation_name} Results:\n",
            f"  Total: {results['total']}\n",
            f"  Successful: {results['successful']}\n",
            f"  Failed: {results['failed']}\n",
        ]

        if results["errors"]:
            lines.append(f"\nErrors ({len(results['errors'])}):\n")
            for error in results["errors"][:5]:  # Show first 5
                lines.append(f"  - {error['item']}: {error['error']}\n")

            if len(results["errors"]) > 5:
                remaining = len(results["errors"]) - 5
                lines.append(f"  ... and {remaining} more errors\n")

        return "".join(lines)

    def save_workflow_report(self, output: Path, workflow_data: Dict[str, Any]) -> None:
        """Save workflow report to file.

        Args:
            output: Output file path
            workflow_data: Workflow execution data
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            lines = [
                "# Workflow Report\n\n",
                f"**Generated:** {timestamp}\n",
                f"**Workflow:** {workflow_data.get('name', 'Unknown')}\n\n",
                "---\n\n",
                "## Summary\n\n",
                f"- Total Operations: {workflow_data.get('total_ops', 0)}\n",
                f"- Successful: {workflow_data.get('successful', 0)}\n",
                f"- Failed: {workflow_data.get('failed', 0)}\n",
                f"- Duration: {workflow_data.get('duration', 0):.2f}s\n\n",
            ]

            if workflow_data.get("errors"):
                lines.append("## Errors\n\n")
                for error in workflow_data["errors"]:
                    lines.append(
                        f"- {error.get('operation', 'Unknown')}: "
                        f"{error.get('error', 'Unknown error')}\n"
                    )
                lines.append("\n")

            output.write_text("".join(lines), encoding="utf-8")
            self.print_success(f"Workflow report saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save report: {e}")

    def measure_operation_time(self, operation: Callable[[], Any]) -> tuple[Any, float]:
        """Measure operation execution time.

        Args:
            operation: Operation to measure

        Returns:
            Tuple of (result, duration_seconds)
        """
        start_time = time.time()
        result = operation()
        duration = time.time() - start_time

        return result, duration

    def validate_pipeline_steps(self, steps: List[str]) -> None:
        """Validate pipeline step names.

        Args:
            steps: List of step names

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if not steps:
            raise typer.BadParameter("Pipeline must have at least one step")

        valid_steps = {
            "ingest",
            "enrich",
            "analyze",
            "export",
            "validate",
            "transform",
            "index",
        }

        for step in steps:
            if step not in valid_steps:
                raise typer.BadParameter(
                    f"Invalid step '{step}'. "
                    f"Valid steps: {', '.join(sorted(valid_steps))}"
                )
