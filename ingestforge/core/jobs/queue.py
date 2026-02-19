"""
Job queue implementations for background processing.

Provides abstract JobQueue protocol and SQLite-backed implementation.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol, cast

from ingestforge.core.jobs.models import Job, JobStatus, JobType


# Type aliases
ProgressCallback = Callable[[float, str], None]
JobHandler = Callable[[Job, ProgressCallback], Any]


class JobQueue(Protocol):
    """Abstract interface for job queue backends."""

    async def enqueue(self, job: Job) -> str:
        """Add a job to the queue. Returns job ID."""
        ...

    async def dequeue(self) -> Optional[Job]:
        """Get next job to process (priority order)."""
        ...

    async def get(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        ...

    async def update(self, job: Job) -> None:
        """Update job status/progress."""
        ...

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[Job]:
        """List jobs with optional filters."""
        ...

    async def count_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
    ) -> int:
        """Count jobs matching filters."""
        ...

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending/running job."""
        ...

    async def cleanup(self, older_than: timedelta) -> int:
        """Remove old completed/failed jobs."""
        ...


class SQLiteJobQueue:
    """
    SQLite-backed job queue for lightweight deployments.

    Provides persistent job storage with async-compatible operations.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        priority INTEGER DEFAULT 0,
        payload TEXT,
        progress REAL DEFAULT 0.0,
        progress_message TEXT,
        result TEXT,
        error TEXT,
        retries INTEGER DEFAULT 0,
        max_retries INTEGER DEFAULT 3,
        created_at TEXT NOT NULL,
        started_at TEXT,
        completed_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
    CREATE INDEX IF NOT EXISTS idx_jobs_priority ON jobs(priority DESC, created_at ASC);
    CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type);
    """

    def __init__(self, db_path: Path) -> None:
        """
        Initialize the SQLite job queue.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            # Reset any jobs that were running when process died
            conn.execute(
                "UPDATE jobs SET status = ? WHERE status = ?",
                (JobStatus.PENDING.value, JobStatus.RUNNING.value),
            )
            conn.commit()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    async def enqueue(self, job: Job) -> str:
        """Add a job to the queue."""
        with self._get_connection() as conn:
            data = job.to_dict()
            columns = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            conn.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders})",
                list(data.values()),
            )
            conn.commit()
        return job.id

    async def dequeue(self) -> Optional[Job]:
        """Get the next pending job by priority."""
        with self._get_connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM jobs
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
                """,
                (JobStatus.PENDING.value,),
            ).fetchone()

            if not row:
                return None

            # Atomically mark as running
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ? WHERE id = ?",
                (
                    JobStatus.RUNNING.value,
                    datetime.now(timezone.utc).isoformat(),
                    row["id"],
                ),
            )
            conn.commit()

            job = Job.from_dict(dict(row))
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now(timezone.utc)
            return job

    async def get(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

            if not row:
                return None

            return Job.from_dict(dict(row))

    async def update(self, job: Job) -> None:
        """Update a job."""
        with self._get_connection() as conn:
            data = job.to_dict()
            updates = ", ".join(f"{k} = ?" for k in data.keys() if k != "id")
            values = [v for k, v in data.items() if k != "id"]
            values.append(job.id)

            conn.execute(f"UPDATE jobs SET {updates} WHERE id = ?", values)
            conn.commit()

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> List[Job]:
        """List jobs with optional filters and cursor-based pagination."""
        with self._get_connection() as conn:
            query = "SELECT * FROM jobs"
            params: List[Any] = []
            conditions = []

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            if job_type:
                conditions.append("type = ?")
                params.append(job_type.value)

            if cursor:
                conditions.append("created_at < ?")
                params.append(cursor)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()
            return [Job.from_dict(dict(row)) for row in rows]

    async def count_jobs(
        self,
        status: Optional[JobStatus] = None,
        job_type: Optional[JobType] = None,
    ) -> int:
        """Count jobs matching filters."""
        with self._get_connection() as conn:
            query = "SELECT COUNT(*) FROM jobs"
            params: List[Any] = []
            conditions = []

            if status:
                conditions.append("status = ?")
                params.append(status.value)

            if job_type:
                conditions.append("type = ?")
                params.append(job_type.value)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            row = conn.execute(query, params).fetchone()
            return cast(int, row[0]) if row else 0

    async def cancel(self, job_id: str) -> bool:
        """Cancel a job if pending or running."""
        with self._get_connection() as conn:
            result = conn.execute(
                """
                UPDATE jobs SET status = ?, completed_at = ?
                WHERE id = ? AND status IN (?, ?)
                """,
                (
                    JobStatus.CANCELLED.value,
                    datetime.now(timezone.utc).isoformat(),
                    job_id,
                    JobStatus.PENDING.value,
                    JobStatus.RUNNING.value,
                ),
            )
            conn.commit()
            return result.rowcount > 0

    async def cleanup(self, older_than: timedelta) -> int:
        """Remove old completed/failed jobs."""
        cutoff = (datetime.now(timezone.utc) - older_than).isoformat()
        with self._get_connection() as conn:
            result = conn.execute(
                """
                DELETE FROM jobs
                WHERE status IN (?, ?, ?)
                AND completed_at < ?
                """,
                (
                    JobStatus.COMPLETED.value,
                    JobStatus.FAILED.value,
                    JobStatus.CANCELLED.value,
                    cutoff,
                ),
            )
            conn.commit()
            return result.rowcount
