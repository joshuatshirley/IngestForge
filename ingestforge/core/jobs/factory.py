"""
Factory functions for creating jobs and job queues.

Provides convenience functions for job creation and queue initialization.
"""

import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from ingestforge.core.jobs.models import Job, JobType
from ingestforge.core.jobs.queue import SQLiteJobQueue


def create_job(
    job_type: JobType, payload: Dict[str, Any], priority: int = 0, max_retries: int = 3
) -> Job:
    """
    Create a new job.

    Args:
        job_type: Type of job.
        payload: Job-specific data.
        priority: Higher values processed first.
        max_retries: Maximum retry attempts.

    Returns:
        New Job instance.
    """
    return Job(
        id=str(uuid.uuid4()),
        type=job_type,
        payload=payload,
        priority=priority,
        max_retries=max_retries,
    )


def create_job_queue(db_path: Optional[Path] = None) -> SQLiteJobQueue:
    """
    Create a job queue.

    Args:
        db_path: Path to SQLite database. Defaults to .data/jobs.db.

    Returns:
        SQLiteJobQueue instance.
    """
    if db_path is None:
        db_path = Path.cwd() / ".data" / "jobs.db"
    return SQLiteJobQueue(db_path)
