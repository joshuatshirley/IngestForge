"""
Parallel Batch Dispatcher for IngestForge (IF).

Scaling - Parallel Batch Dispatch.
Enables concurrent processing of multiple documents through the IF-Protocol pipeline.
Follows NASA JPL Power of Ten rules.
"""

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ingestforge.core.pipeline.interfaces import IFArtifact
from ingestforge.core.errors import SafeErrorMessage

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_WORKERS = 16  # Maximum concurrent workers
MAX_BATCH_SIZE = 1000  # Maximum documents per batch
MAX_QUEUE_SIZE = 10000  # Maximum pending tasks
WORKER_TIMEOUT_SEC = 300  # 5-minute timeout per document
MAX_RETRIES = 3  # Maximum retry attempts per task


class TaskStatus(Enum):
    """
    Status of a batch task.

    Rule #9: Complete type hints via enum.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class BatchTask:
    """
    A single document processing task in a batch.

    Scaling - Parallel Batch Dispatch.
    Rule #9: Complete type hints.

    Attributes:
        task_id: Unique identifier for this task.
        document_id: ID of the document being processed.
        input_artifact: The input artifact to process.
        priority: Task priority (lower = higher priority).
    """

    task_id: str
    document_id: str
    input_artifact: IFArtifact
    priority: int = 0

    def __post_init__(self) -> None:
        """Validate task constraints."""
        if not self.task_id:
            object.__setattr__(self, "task_id", str(uuid.uuid4()))


@dataclass
class TaskResult:
    """
    Result of processing a single task.

    Scaling - Parallel Batch Dispatch.
    Rule #7: Explicit return values.
    Rule #9: Complete type hints.

    Attributes:
        task_id: ID of the task.
        document_id: ID of the processed document.
        status: Final status of the task.
        output_artifact: Output artifact if successful.
        error_message: Error message if failed.
        duration_ms: Processing time in milliseconds.
        retries: Number of retry attempts.
    """

    task_id: str
    document_id: str
    status: TaskStatus
    output_artifact: Optional[IFArtifact] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    retries: int = 0
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def success(self) -> bool:
        """Check if task completed successfully."""
        return self.status == TaskStatus.COMPLETED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "document_id": self.document_id,
            "status": self.status.value,
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "retries": self.retries,
            "completed_at": self.completed_at,
        }


@dataclass
class BatchProgress:
    """
    Real-time progress snapshot of a batch operation.

    Scaling - Parallel Batch Dispatch.
    Rule #9: Complete type hints.

    Attributes:
        batch_id: ID of the batch.
        total: Total number of tasks.
        pending: Number of pending tasks.
        running: Number of currently running tasks.
        completed: Number of successfully completed tasks.
        failed: Number of failed tasks.
        cancelled: Number of cancelled tasks.
    """

    batch_id: str
    total: int
    pending: int
    running: int
    completed: int
    failed: int
    cancelled: int
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def finished(self) -> bool:
        """Check if all tasks are done."""
        return self.pending == 0 and self.running == 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        done = self.completed + self.failed + self.cancelled
        if done == 0:
            return 0.0
        return (self.completed / done) * 100

    @property
    def progress_percent(self) -> float:
        """Calculate overall progress as percentage."""
        if self.total == 0:
            return 100.0
        done = self.completed + self.failed + self.cancelled
        return (done / self.total) * 100

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "total": self.total,
            "pending": self.pending,
            "running": self.running,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "finished": self.finished,
            "success_rate": self.success_rate,
            "progress_percent": self.progress_percent,
            "started_at": self.started_at,
        }


@dataclass
class BatchResult:
    """
    Aggregated results for a batch operation.

    Scaling - Parallel Batch Dispatch.
    Rule #7: Explicit return values.
    Rule #9: Complete type hints.

    Attributes:
        batch_id: ID of the batch.
        results: List of individual task results.
        started_at: When batch started.
        completed_at: When batch finished.
        duration_ms: Total batch duration.
    """

    batch_id: str
    results: List[TaskResult] = field(default_factory=list)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def total(self) -> int:
        """Total number of tasks."""
        return len(self.results)

    @property
    def completed_count(self) -> int:
        """Number of successfully completed tasks."""
        return sum(1 for r in self.results if r.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        """Number of failed tasks."""
        return sum(1 for r in self.results if r.status == TaskStatus.FAILED)

    @property
    def cancelled_count(self) -> int:
        """Number of cancelled tasks."""
        return sum(1 for r in self.results if r.status == TaskStatus.CANCELLED)

    @property
    def success(self) -> bool:
        """Check if all tasks completed successfully."""
        return self.failed_count == 0 and self.cancelled_count == 0 and self.total > 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed_count / self.total) * 100

    def add_result(self, result: TaskResult) -> None:
        """Add a task result."""
        self.results.append(result)

    def complete(self, duration_ms: float) -> None:
        """Mark batch as completed."""
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self.duration_ms = duration_ms

    def get_failures(self) -> List[TaskResult]:
        """Get all failed task results."""
        return [r for r in self.results if r.status == TaskStatus.FAILED]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "batch_id": self.batch_id,
            "total": self.total,
            "completed": self.completed_count,
            "failed": self.failed_count,
            "cancelled": self.cancelled_count,
            "success": self.success,
            "success_rate": self.success_rate,
            "duration_ms": self.duration_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "results": [r.to_dict() for r in self.results],
        }


# Type for the processing function
ProcessFunc = Callable[[IFArtifact], IFArtifact]


class BatchDispatcher:
    """
    Orchestrates parallel processing of document batches.

    Scaling - Parallel Batch Dispatch.
    Rule #2: Bounded worker pool and queue sizes.
    Rule #7: Explicit results for each document.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        process_func: ProcessFunc,
        max_workers: int = 4,
        timeout_sec: float = WORKER_TIMEOUT_SEC,
    ) -> None:
        """
        Initialize the batch dispatcher.

        Args:
            process_func: Function to process each artifact.
            max_workers: Maximum concurrent workers (capped at MAX_WORKERS).
            timeout_sec: Timeout per document in seconds.
        """
        self._max_workers = min(max_workers, MAX_WORKERS)
        self._timeout_sec = min(timeout_sec, WORKER_TIMEOUT_SEC)
        self._process_func = process_func

        self._executor: Optional[ThreadPoolExecutor] = None
        self._lock = threading.Lock()
        self._cancelled = False
        self._current_batch_id: Optional[str] = None

        # Progress tracking
        self._pending = 0
        self._running = 0
        self._completed = 0
        self._failed = 0
        self._cancelled_count = 0

    def _reset_counters(self) -> None:
        """Reset progress counters for new batch."""
        self._pending = 0
        self._running = 0
        self._completed = 0
        self._failed = 0
        self._cancelled_count = 0
        self._cancelled = False

    def _process_task(self, task: BatchTask) -> TaskResult:
        """
        Process a single task.

        Rule #4: Function < 60 lines.
        Rule #7: Always return explicit result.
        """
        import time

        start_time = time.perf_counter()

        with self._lock:
            self._pending -= 1
            self._running += 1

        # Check for cancellation
        if self._cancelled:
            with self._lock:
                self._running -= 1
                self._cancelled_count += 1
            return TaskResult(
                task_id=task.task_id,
                document_id=task.document_id,
                status=TaskStatus.CANCELLED,
                error_message="Batch was cancelled",
            )

        try:
            output = self._process_func(task.input_artifact)
            duration_ms = (time.perf_counter() - start_time) * 1000

            with self._lock:
                self._running -= 1
                self._completed += 1

            return TaskResult(
                task_id=task.task_id,
                document_id=task.document_id,
                status=TaskStatus.COMPLETED,
                output_artifact=output,
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f"Task {task.task_id} failed: {e}")

            with self._lock:
                self._running -= 1
                self._failed += 1

            return TaskResult(
                task_id=task.task_id,
                document_id=task.document_id,
                status=TaskStatus.FAILED,
                error_message=str(e),
                duration_ms=duration_ms,
            )

    def dispatch(self, tasks: List[BatchTask]) -> BatchResult:
        """
        Process a batch of tasks in parallel.

        Rule #2: Bounded batch size.
        Rule #4: Function < 60 lines.
        Rule #7: Return explicit result.

        Args:
            tasks: List of tasks to process.

        Returns:
            BatchResult with all task results.
        """
        import time

        if len(tasks) > MAX_BATCH_SIZE:
            raise ValueError(
                f"Batch size {len(tasks)} exceeds maximum {MAX_BATCH_SIZE}"
            )

        batch_id = str(uuid.uuid4())
        self._current_batch_id = batch_id
        self._reset_counters()
        self._pending = len(tasks)

        result = BatchResult(batch_id=batch_id)
        start_time = time.perf_counter()

        if not tasks:
            result.complete(0.0)
            return result

        # Create executor
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)

        try:
            # Submit all tasks
            futures: Dict[Future[TaskResult], BatchTask] = {}
            for task in tasks:
                future = self._executor.submit(self._process_task, task)
                futures[future] = task

            # Collect results as they complete
            for future in as_completed(futures, timeout=self._timeout_sec * len(tasks)):
                try:
                    task_result = future.result(timeout=self._timeout_sec)
                    result.add_result(task_result)
                except Exception as e:
                    task = futures[future]
                    result.add_result(
                        TaskResult(
                            task_id=task.task_id,
                            document_id=task.document_id,
                            status=TaskStatus.TIMEOUT,
                            # SEC-002: Sanitize error message
                            error_message=SafeErrorMessage.sanitize(
                                e, "task_timed_out", logger
                            ),
                        )
                    )

        finally:
            self._executor.shutdown(wait=False)
            self._executor = None

        duration_ms = (time.perf_counter() - start_time) * 1000
        result.complete(duration_ms)
        self._current_batch_id = None

        return result

    def get_progress(self) -> Optional[BatchProgress]:
        """
        Get current batch progress.

        Returns:
            BatchProgress if batch is running, None otherwise.
        """
        if self._current_batch_id is None:
            return None

        with self._lock:
            return BatchProgress(
                batch_id=self._current_batch_id,
                total=self._pending
                + self._running
                + self._completed
                + self._failed
                + self._cancelled_count,
                pending=self._pending,
                running=self._running,
                completed=self._completed,
                failed=self._failed,
                cancelled=self._cancelled_count,
            )

    def cancel(self) -> bool:
        """
        Cancel the current batch.

        Returns:
            True if cancellation was initiated, False if no batch running.
        """
        if self._current_batch_id is None:
            return False

        self._cancelled = True
        logger.info(f"Batch {self._current_batch_id} cancellation requested")
        return True

    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the dispatcher and release resources.

        Args:
            wait: Whether to wait for pending tasks to complete.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=wait)
            self._executor = None
        logger.info("BatchDispatcher shutdown complete")

    @property
    def max_workers(self) -> int:
        """Get the maximum worker count."""
        return self._max_workers

    @property
    def is_running(self) -> bool:
        """Check if a batch is currently running."""
        return self._current_batch_id is not None


def create_batch_dispatcher(
    process_func: ProcessFunc,
    max_workers: int = 4,
    timeout_sec: float = WORKER_TIMEOUT_SEC,
) -> BatchDispatcher:
    """
    Factory function to create a BatchDispatcher.

    Args:
        process_func: Function to process each artifact.
        max_workers: Maximum concurrent workers.
        timeout_sec: Timeout per document in seconds.

    Returns:
        Configured BatchDispatcher instance.
    """
    return BatchDispatcher(
        process_func=process_func,
        max_workers=max_workers,
        timeout_sec=timeout_sec,
    )


def create_tasks_from_artifacts(
    artifacts: List[IFArtifact], document_ids: Optional[List[str]] = None
) -> List[BatchTask]:
    """
    Helper to create batch tasks from artifacts.

    Args:
        artifacts: List of input artifacts.
        document_ids: Optional list of document IDs.

    Returns:
        List of BatchTask instances.
    """
    if len(artifacts) > MAX_BATCH_SIZE:
        raise ValueError(f"Too many artifacts: {len(artifacts)} > {MAX_BATCH_SIZE}")

    tasks: List[BatchTask] = []
    for i, artifact in enumerate(artifacts):
        doc_id = (
            document_ids[i]
            if document_ids and i < len(document_ids)
            else artifact.artifact_id
        )
        tasks.append(
            BatchTask(
                task_id=str(uuid.uuid4()),
                document_id=doc_id,
                input_artifact=artifact,
                priority=i,
            )
        )
    return tasks
