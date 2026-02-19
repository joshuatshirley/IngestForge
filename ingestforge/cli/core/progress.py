"""Unified progress reporting for CLI commands.

This module provides consistent progress indicators and status messages
across all CLI commands, following Commandments #1 (Simple Control Flow)
and #4 (No Large Functions).

Implements UX-003: Progress Indicators with:
- ETA (Estimated Time Remaining)
- Quiet mode support (--quiet flag)
- CI/non-interactive mode fallback
- Callback-based decoupled progress reporting (Rule #4)
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Callable, Any, Iterator, Tuple, Protocol, List
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from rich.console import Console

# Shared console instance with safe encoding
console = Console(force_terminal=False, no_color=False)

# ASCII spinner for Windows compatibility
ASCII_SPINNER = ["[=  ]", "[== ]", "[===]", "[ ==]", "[  =]", "[ ==]", "[== ]", "[===]"]


def is_interactive() -> bool:
    """Check if we're running in an interactive terminal.

    Returns:
        True if interactive terminal, False in CI/non-interactive mode.

    Detection includes:
    - CI environment variables (CI, GITHUB_ACTIONS, JENKINS, etc.)
    - Non-TTY stdout
    - TERM=dumb
    """
    # Check for CI environment variables
    ci_vars = ["CI", "GITHUB_ACTIONS", "JENKINS_URL", "TRAVIS", "CIRCLECI", "GITLAB_CI"]
    if any(os.environ.get(var) for var in ci_vars):
        return False

    # Check for dumb terminal
    if os.environ.get("TERM") == "dumb":
        return False

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    return True


class ProgressCallback(Protocol):
    """Protocol for progress callbacks (Rule #4: Decouple progress from processing)."""

    def __call__(
        self,
        current: int,
        total: int,
        message: str = "",
        item_name: str = "",
    ) -> None:
        """Report progress update.

        Args:
            current: Current item number (1-indexed)
            total: Total number of items
            message: Optional status message
            item_name: Optional name of current item being processed
        """
        ...


@dataclass
class ProgressState:
    """Tracks progress state for ETA calculation."""

    __slots__ = ("total", "current", "start_time", "item_times")

    total: int
    current: int
    start_time: float
    item_times: List[float]

    @property
    def elapsed(self) -> float:
        """Seconds elapsed since start."""
        return time.time() - self.start_time

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated seconds remaining, or None if insufficient data."""
        if self.current == 0:
            return None

        avg_time = self.elapsed / self.current
        remaining = self.total - self.current
        return avg_time * remaining

    @property
    def eta_formatted(self) -> str:
        """Formatted ETA string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."

        if eta < 60:
            return f"{eta:.0f}s"
        elif eta < 3600:
            mins = int(eta // 60)
            secs = int(eta % 60)
            return f"{mins}m {secs}s"
        else:
            hours = int(eta // 3600)
            mins = int((eta % 3600) // 60)
            return f"{hours}h {mins}m"


class ProgressReporter:
    """Unified progress reporter with ETA, quiet mode, and CI fallback.

    Implements UX-003: Progress Indicators per BACKLOG.md.

    Features:
    - Rich progress bars with ETA when interactive
    - Simple text output in CI/non-interactive mode
    - Quiet mode for suppressing all progress output
    - Callback-based design for decoupling (Rule #4)

    Example:
        reporter = ProgressReporter(total=100, description="Processing")
        for i, item in enumerate(items):
            process(item)
            reporter.update(current=i+1, item_name=item.name)
        reporter.finish()

    Example with callback:
        def my_callback(current: int, total: int, message: str, item_name: str):
            print(f"Progress: {current}/{total}")

        reporter = ProgressReporter(total=100, callback=my_callback)
        # ... processing with reporter.update()
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        quiet: bool = False,
        callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Initialize progress reporter.

        Args:
            total: Total number of items to process
            description: Description shown in progress bar
            quiet: If True, suppress all progress output
            callback: Optional callback for decoupled progress (Rule #4)

        Raises:
            ValueError: If total is negative
        """
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}")

        self.total = total
        self.description = description
        self.quiet = quiet
        self.callback = callback
        self._interactive = is_interactive()
        self._state = ProgressState(
            total=total,
            current=0,
            start_time=time.time(),
            item_times=[],
        )
        self._progress: Optional[Progress] = None
        self._task_id: Optional[Any] = None
        self._started = False
        self._last_text_update = 0.0

    def start(self) -> "ProgressReporter":
        """Start progress display. Returns self for chaining.

        Example:
            with ProgressReporter(100, "Processing").start() as reporter:
                for i in range(100):
                    reporter.update(i + 1)
        """
        if self._started:
            return self

        self._started = True
        self._state.start_time = time.time()

        if self.quiet:
            return self

        if self._interactive:
            self._start_rich_progress()
        else:
            self._start_text_progress()

        return self

    def _start_rich_progress(self) -> None:
        """Start rich progress bar display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="green", finished_style="green"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TextColumn("ETA:"),
            TimeRemainingColumn(),
            console=console,
            expand=False,
            transient=False,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            f"[bold]{self.description}[/bold]", total=self.total
        )

    def _start_text_progress(self) -> None:
        """Start simple text progress for CI mode."""
        print(f"{self.description}: 0/{self.total} (0%)")

    def update(
        self,
        current: Optional[int] = None,
        advance: int = 1,
        message: str = "",
        item_name: str = "",
    ) -> None:
        """Update progress.

        Args:
            current: Set current position (1-indexed). If None, advances by `advance`.
            advance: Amount to advance if current is None. Default 1.
            message: Optional status message
            item_name: Optional name of current item

        Note:
            Either use `current` to set absolute position, or `advance` to increment.
            Using `current` ignores `advance`.
        """
        if current is not None:
            self._state.current = current
        else:
            self._state.current += advance

        # Record item completion time for ETA
        self._state.item_times.append(time.time())

        # Fire callback (Rule #4: Decoupled progress)
        if self.callback:
            self.callback(
                self._state.current,
                self._state.total,
                message,
                item_name,
            )

        if self.quiet:
            return

        if self._interactive:
            self._update_rich_progress(message, item_name)
        else:
            self._update_text_progress(message, item_name)

    def _update_rich_progress(self, message: str, item_name: str) -> None:
        """Update rich progress bar."""
        if self._progress is None or self._task_id is None:
            return

        desc = f"[bold]{self.description}[/bold]"
        if item_name:
            desc = f"[bold]{item_name}[/bold]"

        self._progress.update(
            self._task_id,
            completed=self._state.current,
            description=desc,
        )

    def _update_text_progress(self, message: str, item_name: str) -> None:
        """Update text progress (rate-limited to once per second)."""
        now = time.time()
        if (
            now - self._last_text_update < 1.0
            and self._state.current < self._state.total
        ):
            return

        self._last_text_update = now
        pct = (
            (self._state.current / self._state.total * 100)
            if self._state.total > 0
            else 0
        )
        eta = self._state.eta_formatted

        line = f"{self.description}: {self._state.current}/{self._state.total} ({pct:.0f}%)"
        if item_name:
            line += f" - {item_name}"
        if self._state.current < self._state.total:
            line += f" [ETA: {eta}]"

        print(line)

    def finish(self, final_message: Optional[str] = None) -> None:
        """Complete progress and clean up.

        Args:
            final_message: Optional message to display on completion
        """
        if not self._started:
            return

        if self.quiet:
            return

        if self._interactive:
            if self._progress:
                self._progress.stop()
        else:
            elapsed = self._state.elapsed
            print(
                f"{self.description}: Complete ({self._state.current}/{self._state.total}) in {elapsed:.1f}s"
            )

        if final_message:
            console.print(f"[green][OK][/green] {final_message}")

    def __enter__(self) -> "ProgressReporter":
        """Context manager entry."""
        return self.start()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.finish()

    @classmethod
    def create_callback(
        cls, quiet: bool = False
    ) -> Tuple["ProgressReporter", ProgressCallback]:
        """Create a progress reporter and its callback for decoupled use.

        This factory method supports Rule #4 (decouple progress from processing).
        Pass the callback to processing functions that don't know about Rich.

        Args:
            quiet: If True, progress is suppressed

        Returns:
            Tuple of (reporter, callback)

        Example:
            reporter, callback = ProgressReporter.create_callback()
            reporter.total = len(items)  # Set total when known
            reporter.start()

            # Pass callback to processing function
            process_items(items, progress_callback=callback)

            reporter.finish()
        """
        reporter = cls(total=0, quiet=quiet)

        def callback(
            current: int,
            total: int,
            message: str = "",
            item_name: str = "",
        ) -> None:
            # Update total if it changed
            if reporter.total != total:
                reporter.total = total
                reporter._state.total = total
                if reporter._progress and reporter._task_id:
                    reporter._progress.update(reporter._task_id, total=total)

            reporter.update(current=current, message=message, item_name=item_name)

        reporter.callback = callback
        return reporter, callback


class ProgressManager:
    """Unified progress and status reporting for CLI operations.

    Provides consistent progress indicators (spinners, bars) and status
    messages (success, error, warning, info) across all commands.

    All methods are static to allow usage without instantiation.

    Example:
        with ProgressManager.spinner("Processing..."):
            do_work()

        ProgressManager.print_success("Operation complete!")
    """

    @staticmethod
    @contextmanager
    def spinner(description: str = "Processing...") -> Iterator[Tuple[Progress, Any]]:
        """Create spinner progress indicator context.

        Args:
            description: Status message to display

        Yields:
            Tuple of (Progress instance, task_id)

        Example:
            with ProgressManager.spinner("Loading data...") as (progress, task):
                # Long-running operation
                load_data()
        """
        # Use simple text indicator for Windows compatibility (no Unicode spinners)
        with Progress(
            TextColumn("[cyan][...][/cyan]"),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(description, total=None)
            yield progress, task

    @staticmethod
    @contextmanager
    def bar(
        total: int, description: str = "Processing..."
    ) -> Iterator[Tuple[Progress, Any]]:
        """Create progress bar context.

        Args:
            total: Total number of items to process
            description: Status message to display

        Yields:
            Tuple of (Progress instance, task_id)

        Example:
            items = [1, 2, 3, 4, 5]
            with ProgressManager.bar(len(items), "Processing items") as (prog, task):
                for item in items:
                    process(item)
                    prog.update(task, advance=1)
        """
        # Validate input (Commandment #7)
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(description, total=total)
            yield progress, task

    @staticmethod
    def run_with_spinner(
        operation: Callable[[], Any],
        description: str = "Processing...",
        success_message: Optional[str] = None,
    ) -> Any:
        """Execute operation with spinner and optional success message.

        Args:
            operation: Callable to execute (must take no arguments)
            description: Status message during operation
            success_message: Message to display on success (optional)

        Returns:
            Result from operation

        Raises:
            Exception: Re-raises any exception from operation

        Example:
            result = ProgressManager.run_with_spinner(
                lambda: fetch_data(),
                "Fetching data...",
                "Data fetched successfully!"
            )
        """
        # Validate inputs (Commandment #7)
        if not callable(operation):
            raise TypeError("operation must be callable")

        # Execute with progress (Commandment #1: Simple control flow)
        with ProgressManager.spinner(description):
            result = operation()

        # Show success message if provided
        if success_message:
            ProgressManager.print_success(success_message)

        return result

    @staticmethod
    def print_success(message: str) -> None:
        """Print success message with checkmark.

        Args:
            message: Success message to display

        Example:
            ProgressManager.print_success("File saved successfully!")
        """
        console.print(f"[green][OK][/green] {message}")

    @staticmethod
    def print_error(message: str) -> None:
        """Print error message with X mark.

        Args:
            message: Error message to display

        Example:
            ProgressManager.print_error("Failed to load configuration")
        """
        console.print(f"[red][ERROR][/red] {message}")

    @staticmethod
    def print_warning(message: str) -> None:
        """Print warning message with warning symbol.

        Args:
            message: Warning message to display

        Example:
            ProgressManager.print_warning("Using default configuration")
        """
        console.print(f"[yellow][WARN][/yellow] {message}")

    @staticmethod
    def print_info(message: str) -> None:
        """Print info message with info symbol.

        Args:
            message: Info message to display

        Example:
            ProgressManager.print_info("Loading 5 documents...")
        """
        console.print(f"[blue][INFO][/blue] {message}")


class BatchProgressTracker:
    """Track progress for batch operations.

    Simplified progress tracking for operations processing multiple items.
    Maintains counts and calculates success rate.

    Example:
        tracker = BatchProgressTracker(total=100)
        for item in items:
            if process(item):
                tracker.increment_success()
            else:
                tracker.increment_failure()
        tracker.print_summary()
    """

    def __init__(self, total: int, operation_name: str = "items") -> None:
        """Initialize batch progress tracker.

        Args:
            total: Total number of items to process
            operation_name: Name of items being processed (for display)

        Raises:
            ValueError: If total is negative
        """
        # Validate input (Commandment #7)
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}")

        self.total = total
        self.operation_name = operation_name
        self.success_count = 0
        self.failure_count = 0

    def increment_success(self) -> None:
        """Record successful item processing."""
        self.success_count += 1

    def increment_failure(self) -> None:
        """Record failed item processing."""
        self.failure_count += 1

    @property
    def processed(self) -> int:
        """Get total items processed so far."""
        return self.success_count + self.failure_count

    @property
    def remaining(self) -> int:
        """Get number of items remaining."""
        return self.total - self.processed

    @property
    def success_rate(self) -> float:
        """Calculate success rate as fraction (0.0 to 1.0).

        Returns:
            Success rate, or 0.0 if no items processed
        """
        if self.processed == 0:
            return 0.0
        return self.success_count / self.processed

    def print_summary(self) -> None:
        """Print summary of batch operation results.

        Displays:
        - Total processed
        - Success count (green)
        - Failure count (red if any)
        - Success rate percentage

        Example output:
            ✓ Processed 100 items: 95 succeeded, 5 failed (95.0% success rate)
        """
        # Format success rate (Commandment #1: Simple flow)
        rate_pct = self.success_rate * 100

        # Build message parts
        if self.failure_count > 0:
            msg = (
                f"[green][OK][/green] Processed {self.processed} {self.operation_name}: "
                f"[green]{self.success_count} succeeded[/green], "
                f"[red]{self.failure_count} failed[/red] "
                f"({rate_pct:.1f}% success rate)"
            )
        else:
            msg = (
                f"[green][OK][/green] Processed {self.processed} {self.operation_name}: "
                f"all succeeded (100% success rate)"
            )

        console.print(msg)
