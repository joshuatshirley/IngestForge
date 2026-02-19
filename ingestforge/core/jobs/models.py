"""
Data models for background job processing.

Defines enums and dataclasses for job status, types, and job objects.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class JobStatus(Enum):
    """Job execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(Enum):
    """Types of background jobs."""

    INGEST = "ingest"
    EMBED = "embed"
    INDEX = "index"
    EXPORT = "export"
    STUDIO = "studio"


@dataclass
class Job:
    """
    Represents a background job.

    Attributes:
        id: Unique job identifier.
        type: Type of job (ingest, embed, etc.).
        status: Current job status.
        payload: Job-specific data.
        priority: Higher values processed first.
        progress: Completion percentage (0.0 to 1.0).
        progress_message: Current operation description.
        result: Output data on completion.
        error: Error message if failed.
        retries: Number of retry attempts.
        max_retries: Maximum retry attempts.
        created_at: When the job was created.
        started_at: When processing began.
        completed_at: When processing finished.
    """

    id: str
    type: JobType
    status: JobStatus = JobStatus.PENDING
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    progress: float = 0.0
    progress_message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "type": self.type.value,
            "status": self.status.value,
            "payload": json.dumps(self.payload),
            "priority": self.priority,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "result": json.dumps(self.result) if self.result else None,
            "error": self.error,
            "retries": self.retries,
            "max_retries": self.max_retries,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create Job from dictionary."""
        return cls(
            id=data["id"],
            type=JobType(data["type"]),
            status=JobStatus(data["status"]),
            payload=json.loads(data["payload"]) if data["payload"] else {},
            priority=data["priority"],
            progress=data["progress"],
            progress_message=data["progress_message"] or "",
            result=json.loads(data["result"]) if data["result"] else None,
            error=data["error"],
            retries=data["retries"],
            max_retries=data["max_retries"],
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data["started_at"]
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data["completed_at"]
            else None,
        )
