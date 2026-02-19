"""Strategist Schemas.

Defines the data models for research roadmaps and atomic tasks.
Follows NASA JPL Rule #7 (Validation) and Rule #9 (Type Hints).
"""

from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ResearchTask(BaseModel):
    """A single atomic task in a research roadmap."""

    id: str = Field(..., description="Unique task identifier (e.g. T1)")
    description: str = Field(..., description="Clear objective of what to find")
    estimated_effort: str = Field("low", description="S/M/L estimate")
    status: TaskStatus = TaskStatus.PENDING
    assigned_tool: Optional[str] = None
    result_summary: Optional[str] = None


class ResearchRoadmap(BaseModel):
    """A sequence of tasks to solve a complex query."""

    objective: str
    tasks: List[ResearchTask]
    version: str = "1.0.0"

    def get_next_pending(self) -> Optional[ResearchTask]:
        """Retrieve the first task that is still pending."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                return task
        return None
