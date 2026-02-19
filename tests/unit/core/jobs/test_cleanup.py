"""Tests for job queue cleanup (DEAD-D07).

NASA JPL Commandments compliance:
- Rule #1: Linear test structure
- Rule #4: Functions <60 lines
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ingestforge.core.jobs import (
    Job,
    JobStatus,
    JobType,
    SQLiteJobQueue,
    WorkerPool,
)


class TestSQLiteJobQueueCleanup:
    """Tests for SQLiteJobQueue.cleanup() method."""

    @pytest.fixture
    def temp_db(self, tmp_path: Path) -> Path:
        """Create temporary database path."""
        return tmp_path / "test_jobs.db"

    @pytest.fixture
    def queue(self, temp_db: Path) -> SQLiteJobQueue:
        """Create test queue."""
        return SQLiteJobQueue(temp_db)

    def test_cleanup_removes_old_completed_jobs(self, queue: SQLiteJobQueue) -> None:
        """Old completed jobs are removed."""
        # Create old completed job
        old_job = Job(
            id="old-job-1",
            type=JobType.INGEST,
            status=JobStatus.COMPLETED,
            completed_at=datetime.utcnow() - timedelta(days=10),
        )
        asyncio.run(queue.enqueue(old_job))

        # Manually update completed_at to simulate old job
        with queue._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ? WHERE id = ?",
                (
                    (datetime.utcnow() - timedelta(days=10)).isoformat(),
                    "old-job-1",
                ),
            )
            conn.commit()

        # Cleanup jobs older than 7 days
        count = asyncio.run(queue.cleanup(timedelta(days=7)))

        assert count == 1

    def test_cleanup_keeps_recent_jobs(self, queue: SQLiteJobQueue) -> None:
        """Recent completed jobs are kept."""
        # Create recent completed job
        recent_job = Job(
            id="recent-job-1",
            type=JobType.INGEST,
            status=JobStatus.COMPLETED,
            completed_at=datetime.utcnow() - timedelta(days=2),
        )
        asyncio.run(queue.enqueue(recent_job))

        # Manually update completed_at
        with queue._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ? WHERE id = ?",
                (
                    (datetime.utcnow() - timedelta(days=2)).isoformat(),
                    "recent-job-1",
                ),
            )
            conn.commit()

        # Cleanup jobs older than 7 days
        count = asyncio.run(queue.cleanup(timedelta(days=7)))

        assert count == 0

        # Verify job still exists
        job = asyncio.run(queue.get("recent-job-1"))
        assert job is not None

    def test_cleanup_removes_old_failed_jobs(self, queue: SQLiteJobQueue) -> None:
        """Old failed jobs are removed."""
        # Create old failed job
        old_job = Job(
            id="old-failed-1",
            type=JobType.INGEST,
            status=JobStatus.FAILED,
            error="Test error",
        )
        asyncio.run(queue.enqueue(old_job))

        # Update completed_at and status
        with queue._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ?, status = ? WHERE id = ?",
                (
                    (datetime.utcnow() - timedelta(days=10)).isoformat(),
                    JobStatus.FAILED.value,
                    "old-failed-1",
                ),
            )
            conn.commit()

        count = asyncio.run(queue.cleanup(timedelta(days=7)))

        assert count == 1

    def test_cleanup_keeps_pending_jobs(self, queue: SQLiteJobQueue) -> None:
        """Pending jobs are never removed by cleanup."""
        # Create pending job
        pending_job = Job(
            id="pending-job-1",
            type=JobType.INGEST,
            status=JobStatus.PENDING,
        )
        asyncio.run(queue.enqueue(pending_job))

        count = asyncio.run(queue.cleanup(timedelta(days=0)))

        assert count == 0

        # Verify job still exists
        job = asyncio.run(queue.get("pending-job-1"))
        assert job is not None

    def test_cleanup_empty_queue_returns_zero(self, queue: SQLiteJobQueue) -> None:
        """Empty queue returns zero."""
        count = asyncio.run(queue.cleanup(timedelta(days=7)))
        assert count == 0


class TestWorkerPoolAutoCleanup:
    """Tests for WorkerPool automatic cleanup on startup."""

    def test_worker_pool_has_auto_cleanup_enabled_by_default(self) -> None:
        """Auto cleanup is enabled by default."""
        mock_queue = MagicMock(spec=SQLiteJobQueue)
        pool = WorkerPool(queue=mock_queue, handlers={})

        assert pool.auto_cleanup is True
        assert pool.cleanup_days == 7

    def test_worker_pool_can_disable_auto_cleanup(self) -> None:
        """Auto cleanup can be disabled."""
        mock_queue = MagicMock(spec=SQLiteJobQueue)
        pool = WorkerPool(queue=mock_queue, handlers={}, auto_cleanup=False)

        assert pool.auto_cleanup is False

    def test_worker_pool_custom_cleanup_days(self) -> None:
        """Custom cleanup days can be specified."""
        mock_queue = MagicMock(spec=SQLiteJobQueue)
        pool = WorkerPool(queue=mock_queue, handlers={}, cleanup_days=30)

        assert pool.cleanup_days == 30

    @pytest.mark.asyncio
    async def test_cleanup_old_jobs_called_on_start(self) -> None:
        """Cleanup is called when worker pool starts."""
        mock_queue = MagicMock(spec=SQLiteJobQueue)
        mock_queue.cleanup = AsyncMock(return_value=5)
        mock_queue.dequeue = AsyncMock(return_value=None)

        pool = WorkerPool(
            queue=mock_queue,
            handlers={},
            auto_cleanup=True,
        )

        # Start and stop immediately
        pool._running = True
        await pool._cleanup_old_jobs()

        mock_queue.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_error_does_not_fail_startup(self) -> None:
        """Cleanup errors don't prevent pool startup."""
        mock_queue = MagicMock(spec=SQLiteJobQueue)
        mock_queue.cleanup = AsyncMock(side_effect=RuntimeError("DB error"))

        pool = WorkerPool(
            queue=mock_queue,
            handlers={},
            auto_cleanup=True,
        )

        # Should not raise
        await pool._cleanup_old_jobs()


class TestMaintenanceCleanupJobs:
    """Tests for CLI cleanup command with --jobs flag."""

    def test_clean_jobs_with_missing_database(self, tmp_path: Path) -> None:
        """Clean jobs handles missing database gracefully."""
        from ingestforge.cli.maintenance.cleanup import CleanupCommand

        cmd = CleanupCommand()
        result = cmd._clean_jobs(tmp_path, retention_days=7)

        assert result["count"] == 0
        assert result["errors"] == []

    def test_clean_jobs_with_existing_database(self, tmp_path: Path) -> None:
        """Clean jobs works with existing database."""
        # Create database with old job
        db_path = tmp_path / ".data" / "jobs.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        queue = SQLiteJobQueue(db_path)

        # Add old completed job
        old_job = Job(
            id="old-job-1",
            type=JobType.INGEST,
            status=JobStatus.COMPLETED,
        )
        asyncio.run(queue.enqueue(old_job))

        # Update to old date
        with queue._get_connection() as conn:
            conn.execute(
                "UPDATE jobs SET completed_at = ?, status = ? WHERE id = ?",
                (
                    (datetime.utcnow() - timedelta(days=10)).isoformat(),
                    JobStatus.COMPLETED.value,
                    "old-job-1",
                ),
            )
            conn.commit()

        from ingestforge.cli.maintenance.cleanup import CleanupCommand

        cmd = CleanupCommand()
        result = cmd._clean_jobs(tmp_path, retention_days=7)

        assert result["count"] == 1
        assert result["errors"] == []
