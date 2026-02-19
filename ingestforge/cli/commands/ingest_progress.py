"""Clean progress display for document ingestion.

Shows organized, aesthetically pleasing progress output.
Supports parallel processing for faster ingestion.

Implements UX-003: Progress Indicators with:
- ETA (Estimated Time Remaining)
- Quiet mode support (--quiet flag)
- CI/non-interactive mode fallback

NASA JPL Rule #9: Complete type hints on all functions.
NASA JPL Rule #4: Decoupled progress via callbacks.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    TaskID,
)

from ingestforge.cli.core.progress import ProgressReporter, is_interactive


console: Console = Console()

# Number of parallel workers (use half of CPU cores, minimum 2)
DEFAULT_WORKERS: int = max(2, (os.cpu_count() or 4) // 2)


class IngestProgressDisplay:
    """Clean progress display for ingestion."""

    def __init__(self) -> None:
        self.files_processed: List[Dict[str, Any]] = []
        self.current_file: Optional[str] = None
        self.current_stage: str = ""
        self.total_chunks: int = 0

    def _create_file_row(self, info: Dict[str, Any]) -> List[str]:
        """Create a row for the file table."""
        status = "[green]Done[/green]" if info["complete"] else "[yellow]...[/yellow]"
        chunks = str(info["chunks"]) if info["chunks"] > 0 else "-"
        return [info["name"], status, info["stage"], chunks]

    def _build_display(self) -> Table:
        """Build the progress table."""
        table = Table(
            title="[bold cyan]Document Ingestion[/bold cyan]",
            show_header=True,
            header_style="bold white",
            border_style="dim",
            expand=True,
        )

        table.add_column("File", style="white", no_wrap=True, max_width=40)
        table.add_column("Status", style="white", width=8, justify="center")
        table.add_column("Stage", style="dim", width=20)
        table.add_column("Chunks", style="cyan", width=8, justify="right")

        for info in self.files_processed:
            row = self._create_file_row(info)
            table.add_row(*row)

        return table

    def start_file(self, file_path: Path) -> None:
        """Mark file as started."""
        self.current_file = file_path.name
        self.files_processed.append(
            {
                "name": file_path.name,
                "complete": False,
                "stage": "Loading...",
                "chunks": 0,
            }
        )

    def update_stage(self, stage: str) -> None:
        """Update current file's stage."""
        if self.files_processed:
            self.files_processed[-1]["stage"] = stage

    def complete_file(self, chunks: int) -> None:
        """Mark current file as complete."""
        if self.files_processed:
            self.files_processed[-1]["complete"] = True
            self.files_processed[-1]["stage"] = "Complete"
            self.files_processed[-1]["chunks"] = chunks
            self.total_chunks += chunks

    def get_summary(self) -> str:
        """Get summary text."""
        completed = sum(1 for f in self.files_processed if f["complete"])
        return f"Processed {completed} files, {self.total_chunks} total chunks"


def _process_single_file(
    pipeline: Any, file_path: Path
) -> Tuple[Path, Optional[Any], Optional[str]]:
    """Process a single file. Returns (file_path, result, error_message)."""
    try:
        result = pipeline.process_file(file_path)
        return (file_path, result, None)
    except Exception as e:
        return (file_path, None, str(e))


def _log_file_result(
    file_path: Path, result: Any, error: Optional[str]
) -> Tuple[int, Optional[str]]:
    """Log result for a single file. Returns (chunks_count, error_string or None)."""
    if error:
        console.print(f"  [red][ERROR][/red] {file_path.name}: {error[:50]}")
        return 0, f"{file_path.name}: {error}"

    chunks = result.chunks_created if result else 0
    console.print(
        f"  [green][OK][/green] {file_path.name} [dim]({chunks} chunks)[/dim]"
    )
    return chunks, None


def _handle_file_result(
    file_path: Path,
    result: Any,
    error: Optional[str],
    errors: List[str],
    completed_files: List[str],
    skip_errors: bool,
    futures: Dict[Any, Path],
) -> int:
    """Handle result for a single processed file.

    JPL-003: Extracted to reduce nesting in parallel ingestion.

    Args:
        file_path: Path to processed file
        result: Processing result
        error: Error message if failed
        errors: List to append errors to
        completed_files: List to append successful files to
        skip_errors: Whether to skip errors
        futures: Futures dict for cancellation

    Returns:
        Number of chunks created
    """
    chunks, error_msg = _log_file_result(file_path, result, error)

    if error_msg:
        errors.append(error_msg)
        if not skip_errors:
            _cancel_futures(futures)
            raise RuntimeError(error)
        return 0

    completed_files.append(file_path.name)
    return chunks


def _run_parallel_ingestion(
    files: List[Path],
    pipeline: Any,
    workers: int,
    skip_errors: bool,
    progress: Progress,
    task_id: TaskID,
) -> Tuple[List[str], List[str], int]:
    """Run parallel ingestion. Returns (completed_files, errors, total_chunks)."""
    completed_files: List[str] = []
    errors: List[str] = []
    total_chunks = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_file, pipeline, f): f for f in files}

        for future in as_completed(futures):
            file_path, result, error = future.result()
            # JPL-003: Reduced nesting via helper function
            chunks = _handle_file_result(
                file_path, result, error, errors, completed_files, skip_errors, futures
            )
            total_chunks += chunks
            progress.advance(task_id)

    return completed_files, errors, total_chunks


def _cancel_futures(futures: Dict[Any, Path]) -> None:
    """Cancel all pending futures."""
    for f in futures:
        f.cancel()


def _run_sequential_ingestion(
    files: List[Path],
    pipeline: Any,
    skip_errors: bool,
    progress: Progress,
    task_id: TaskID,
) -> Tuple[List[str], List[str], int]:
    """Run sequential ingestion. Returns (completed_files, errors, total_chunks)."""
    completed_files: List[str] = []
    errors: List[str] = []
    total_chunks = 0

    for file_path in files:
        progress.update(task_id, description=f"[bold]{file_path.name}[/bold]")
        _, result, error = _process_single_file(pipeline, file_path)
        chunks, error_msg = _log_file_result(file_path, result, error)

        if error_msg:
            errors.append(error_msg)
            if not skip_errors:
                raise RuntimeError(error)
        else:
            total_chunks += chunks
            completed_files.append(file_path.name)

        progress.advance(task_id)

    return completed_files, errors, total_chunks


def _print_summary(
    completed_files: List[str], total_chunks: int, errors: List[str]
) -> None:
    """Print ingestion summary."""
    console.print()
    if errors:
        console.print(f"[yellow][WARN][/yellow] {len(errors)} errors occurred")
    console.print(
        f"[green][OK][/green] Completed: "
        f"[bold]{len(completed_files)}[/bold] files, "
        f"[bold]{total_chunks}[/bold] chunks"
    )
    console.print()


def run_ingestion_with_progress(
    files: List[Path],
    pipeline: Any,
    skip_errors: bool = False,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    quiet: bool = False,
) -> Dict[str, Any]:
    """Run ingestion with progress display.

    Implements UX-003: Progress Indicators with ETA, quiet mode, and CI fallback.

    Rule #4: <60 lines via helpers.

    Args:
        files: List of file paths to process
        pipeline: Pipeline instance for processing
        skip_errors: Continue on errors
        parallel: Enable parallel processing
        max_workers: Number of parallel workers
        quiet: Suppress progress output (--quiet flag)

    Returns:
        Dictionary with success, files_processed, chunks_created, errors
    """
    workers = max_workers or DEFAULT_WORKERS
    use_parallel = parallel and len(files) > 1
    interactive = is_interactive()

    # Print header unless quiet
    if not quiet:
        console.print()
        console.print("[bold cyan]Starting Ingestion[/bold cyan]")
        mode = (
            f"[green]Parallel[/green] ({workers} workers)"
            if use_parallel
            else "[dim]Sequential[/dim]"
        )
        console.print(f"[dim]Processing {len(files)} files... {mode}[/dim]")
        console.print()

    # Use ProgressReporter for unified progress handling (UX-003)
    with ProgressReporter(
        total=len(files),
        description="Ingesting",
        quiet=quiet,
    ).start() as reporter:
        if use_parallel:
            completed, errors, chunks = _run_parallel_ingestion_with_reporter(
                files, pipeline, workers, skip_errors, reporter
            )
        else:
            completed, errors, chunks = _run_sequential_ingestion_with_reporter(
                files, pipeline, skip_errors, reporter
            )

    if not quiet:
        _print_summary(completed, chunks, errors)

    return {
        "success": len(errors) == 0 or skip_errors,
        "files_processed": len(completed),
        "chunks_created": chunks,
        "errors": errors,
    }


def _run_parallel_ingestion_with_reporter(
    files: List[Path],
    pipeline: Any,
    workers: int,
    skip_errors: bool,
    reporter: ProgressReporter,
) -> Tuple[List[str], List[str], int]:
    """Run parallel ingestion with ProgressReporter.

    Rule #4: Function <60 lines.

    Args:
        files: Files to process
        pipeline: Pipeline instance
        workers: Number of workers
        skip_errors: Continue on errors
        reporter: ProgressReporter instance

    Returns:
        Tuple of (completed_files, errors, total_chunks)
    """
    completed_files: List[str] = []
    errors: List[str] = []
    total_chunks = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_single_file, pipeline, f): f for f in files}

        for future in as_completed(futures):
            file_path, result, error = future.result()
            # JPL-003: Reduced nesting via helper function
            chunks = _handle_file_result(
                file_path, result, error, errors, completed_files, skip_errors, futures
            )
            total_chunks += chunks

            # Update progress reporter
            reporter.update(
                current=len(completed_files) + len(errors),
                item_name=file_path.name,
            )

    return completed_files, errors, total_chunks


def _run_sequential_ingestion_with_reporter(
    files: List[Path],
    pipeline: Any,
    skip_errors: bool,
    reporter: ProgressReporter,
) -> Tuple[List[str], List[str], int]:
    """Run sequential ingestion with ProgressReporter.

    Rule #4: Function <60 lines.

    Args:
        files: Files to process
        pipeline: Pipeline instance
        skip_errors: Continue on errors
        reporter: ProgressReporter instance

    Returns:
        Tuple of (completed_files, errors, total_chunks)
    """
    completed_files: List[str] = []
    errors: List[str] = []
    total_chunks = 0

    for idx, file_path in enumerate(files, 1):
        # Update progress with current file name
        reporter.update(current=idx - 1, item_name=file_path.name)

        _, result, error = _process_single_file(pipeline, file_path)
        chunks, error_msg = _log_file_result(file_path, result, error)

        if error_msg:
            errors.append(error_msg)
            if not skip_errors:
                raise RuntimeError(error)
        else:
            total_chunks += chunks
            completed_files.append(file_path.name)

        # Final update for this file
        reporter.update(current=idx, item_name=file_path.name)

    return completed_files, errors, total_chunks
