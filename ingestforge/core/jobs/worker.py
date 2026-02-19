"""
Worker pool for background job processing.

Manages async workers that poll the queue and process jobs.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from ingestforge.core.jobs.models import Job, JobStatus, JobType
from ingestforge.core.jobs.queue import JobHandler, SQLiteJobQueue
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
DEFAULT_AUTO_CLEANUP_DAYS = 7


class WorkerPool:
    """
    Manages async workers for job processing.

    Runs background workers that poll the queue and process jobs.
    """

    def __init__(
        self,
        queue: SQLiteJobQueue,
        handlers: Dict[JobType, JobHandler],
        max_workers: int = 4,
        poll_interval: float = 1.0,
        auto_cleanup: bool = True,
        cleanup_days: int = DEFAULT_AUTO_CLEANUP_DAYS,
    ):
        """
        Initialize worker pool.

        Args:
            queue: Job queue to poll.
            handlers: Map of job type to handler function.
            max_workers: Maximum concurrent workers.
            poll_interval: Seconds between queue polls.
            auto_cleanup: Clean old jobs on startup (DEAD-D07).
            cleanup_days: Days to retain completed jobs.
        """
        self.queue = queue
        self.handlers = handlers
        self.max_workers = max_workers
        self.poll_interval = poll_interval
        self.auto_cleanup = auto_cleanup
        self.cleanup_days = cleanup_days
        self._running = False
        self._tasks: set[asyncio.Task[None]] = set()

    async def start(self) -> None:
        """Start the worker pool.

        DEAD-D07: Automatically cleans old jobs on startup.
        """
        self._running = True

        # DEAD-D07: Clean old jobs on startup
        if self.auto_cleanup:
            await self._cleanup_old_jobs()

        async with asyncio.TaskGroup() as tg:
            for i in range(self.max_workers):
                task = tg.create_task(self._worker_loop(i))
                self._tasks.add(task)

    async def _cleanup_old_jobs(self) -> None:
        """Clean up old completed/failed jobs on startup."""
        try:
            older_than = timedelta(days=self.cleanup_days)
            count = await self.queue.cleanup(older_than)
            if count > 0:
                logger.info(f"Cleaned up {count} old job records")
        except Exception as e:
            logger.warning(f"Failed to clean old jobs: {e}")

    async def stop(self) -> None:
        """Stop all workers gracefully."""
        self._running = False
        # Workers will exit on next poll cycle

    async def _worker_loop(self, worker_id: int) -> None:
        """Main worker loop."""
        while self._running:
            try:
                job = await self.queue.dequeue()
                if job:
                    await self._process_job(job, worker_id)
                else:
                    await asyncio.sleep(self.poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log and continue
                print(f"Worker {worker_id} error: {e}")
                await asyncio.sleep(self.poll_interval)

    async def _process_job(self, job: Job, worker_id: int) -> None:
        """Process a single job."""
        handler = self.handlers.get(job.type)
        if not handler:
            job.status = JobStatus.FAILED
            job.error = f"No handler for job type: {job.type.value}"
            job.completed_at = datetime.now(timezone.utc)
            await self.queue.update(job)
            return

        async def progress_callback(progress: float, message: str) -> None:
            """Update job progress."""
            job.progress = progress
            job.progress_message = message
            await self.queue.update(job)

        try:
            # Run handler (may be sync or async)
            if asyncio.iscoroutinefunction(handler):
                result = await handler(job, progress_callback)
            else:
                result = handler(
                    job, lambda p, m: asyncio.create_task(progress_callback(p, m))
                )

            job.status = JobStatus.COMPLETED
            job.result = result if isinstance(result, dict) else {"result": result}
            job.progress = 1.0
            job.completed_at = datetime.now(timezone.utc)

        except Exception as e:
            job.retries += 1
            if job.retries >= job.max_retries:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now(timezone.utc)
            else:
                # Retry - put back in pending
                job.status = JobStatus.PENDING
                job.started_at = None

        await self.queue.update(job)


async def process_queue_worker(
    queue: SQLiteJobQueue,
    handlers: Dict[JobType, JobHandler],
    poll_interval: float = 5.0,
    low_priority: bool = True,
    shutdown_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Poll the job queue and process jobs as a low-priority background worker.

    This is a single-worker async function designed for resource-constrained
    environments. It processes one job at a time with configurable poll
    intervals to minimize CPU and memory usage.

    Args:
        queue: Job queue to poll for work.
        handlers: Map of job type to handler function.
        poll_interval: Seconds between queue polls (default 5.0).
            Higher values reduce CPU usage on idle systems.
        low_priority: If True, yields control frequently to avoid
            starving other tasks (default True).
        shutdown_event: Optional asyncio.Event to signal graceful shutdown.
            When set, the worker finishes the current job and exits.

    Usage:
        queue = create_job_queue()
        handlers = {JobType.INGEST: my_ingest_handler}

        # Run as a background task
        shutdown = asyncio.Event()
        task = asyncio.create_task(
            process_queue_worker(queue, handlers, shutdown_event=shutdown)
        )

        # To stop gracefully:
        shutdown.set()
        await task
    """
    if shutdown_event is None:
        shutdown_event = asyncio.Event()

    while not shutdown_event.is_set():
        try:
            job = await queue.dequeue()

            if job is None:
                # No work available; wait before polling again
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval)
                except asyncio.TimeoutError:
                    logger.debug("Worker polling timeout (expected behavior)")
                continue

            handler = handlers.get(job.type)
            if not handler:
                job.status = JobStatus.FAILED
                job.error = f"No handler registered for job type: {job.type.value}"
                job.completed_at = datetime.now(timezone.utc)
                await queue.update(job)
                continue

            # Build progress callback
            async def _progress(progress: float, message: str) -> None:
                job.progress = progress
                job.progress_message = message
                await queue.update(job)

            try:
                # Yield control before heavy processing when low_priority
                if low_priority:
                    await asyncio.sleep(0)

                if asyncio.iscoroutinefunction(handler):
                    result = await handler(job, _progress)
                else:
                    result = handler(
                        job,
                        lambda p, m: asyncio.create_task(_progress(p, m)),
                    )

                job.status = JobStatus.COMPLETED
                job.result = result if isinstance(result, dict) else {"result": result}
                job.progress = 1.0
                job.completed_at = datetime.now(timezone.utc)

            except Exception as e:
                job.retries += 1
                if job.retries >= job.max_retries:
                    job.status = JobStatus.FAILED
                    job.error = str(e)
                    job.completed_at = datetime.now(timezone.utc)
                else:
                    # Put back in queue for retry
                    job.status = JobStatus.PENDING
                    job.started_at = None

            await queue.update(job)

            # Yield control after processing when low_priority
            if low_priority:
                await asyncio.sleep(0)

        except asyncio.CancelledError:
            break
        except Exception:
            # Unexpected error in worker loop; sleep and retry
            logger.exception("Unexpected error in worker loop, will retry after sleep")
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                logger.debug("Worker error recovery timeout (expected behavior)")
