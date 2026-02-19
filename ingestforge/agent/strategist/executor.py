"""Roadmap Executor.

Executes a sequence of research tasks using the ReAct engine.
Follows NASA JPL Rule #1 (Logic) and Rule #2 (Bounds).
"""

from __future__ import annotations

from ingestforge.agent.react_engine import ReActEngine, AgentResult
from ingestforge.agent.strategist.models import ResearchRoadmap, TaskStatus
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class RoadmapExecutor:
    """Orchestrates the sequential execution of a research roadmap."""

    def __init__(self, engine: ReActEngine):
        self._engine = engine

    def execute(self, roadmap: ResearchRoadmap) -> ResearchRoadmap:
        """Run all tasks in the roadmap sequentially.

        Rule #1: Flat loop over tasks.
        Rule #2: Safety break if roadmap is too long.
        """
        MAX_TASKS = 10
        for i, task in enumerate(roadmap.tasks[:MAX_TASKS]):
            if task.status != TaskStatus.PENDING:
                continue

            logger.info(f"Executing Roadmap Task {i+1}: {task.id} - {task.description}")
            task.status = TaskStatus.IN_PROGRESS

            try:
                # Run the ReAct engine for this atomic task
                result: AgentResult = self._engine.run(task.description)

                if result.success:
                    task.status = TaskStatus.COMPLETED
                    task.result_summary = result.final_answer
                else:
                    task.status = TaskStatus.FAILED
                    task.result_summary = "Task failed to synthesize a result."

            except Exception as e:
                logger.error(f"Execution error on task {task.id}: {e}")
                task.status = TaskStatus.FAILED
                task.result_summary = str(e)

        return roadmap
