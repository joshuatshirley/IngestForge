"""
Background Job Queue for IngestForge.

This module provides an asynchronous job processing system for long-running
operations that shouldn't block the API or CLI. It supports job persistence,
progress tracking, automatic retry, and priority ordering.

This module has been split into focused submodules for maintainability
while maintaining 100% backward compatibility via re-exports.

Architecture Context
--------------------
The job queue enables non-blocking document processing:

    ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   CLI / API      │────→│   Job Queue     │────→│  Worker Pool    │
    │  (enqueue job)   │     │   (SQLite)      │     │  (process jobs) │
    └──────────────────┘     └─────────────────┘     └─────────────────┘
           ↑                                                  │
           │              Progress Updates                    │
           └──────────────────────────────────────────────────┘
"""

# Models
from ingestforge.core.jobs.models import Job, JobStatus, JobType

# Queue
from ingestforge.core.jobs.queue import (
    JobHandler,
    JobQueue,
    ProgressCallback,
    SQLiteJobQueue,
)

# Worker
from ingestforge.core.jobs.worker import WorkerPool, process_queue_worker

# Factory
from ingestforge.core.jobs.factory import create_job, create_job_queue

__all__ = [
    # Enums
    "JobStatus",
    "JobType",
    # Models
    "Job",
    # Queue
    "JobQueue",
    "SQLiteJobQueue",
    # Worker
    "WorkerPool",
    "process_queue_worker",
    # Factory
    "create_job",
    "create_job_queue",
    # Type aliases
    "ProgressCallback",
    "JobHandler",
]
