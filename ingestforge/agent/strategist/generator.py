"""Roadmap Generator.

Translates complex objectives into a structured sequence of tasks.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
import json
from typing import List

from ingestforge.agent.personas import get_persona, PersonaType
from ingestforge.agent.strategist.models import (
    ResearchRoadmap,
    ResearchTask,
    TaskStatus,
)
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient

logger = get_logger(__name__)


class RoadmapGenerator:
    """Orchestrates the creation of research plans."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._persona = get_persona(PersonaType.STRATEGIST)

    def generate(self, objective: str) -> ResearchRoadmap:
        """Generate a roadmap for a given research objective.

        Rule #1: Simple logic with early returns.
        Rule #7: Validate objective.
        """
        if not objective or len(objective.strip()) < 5:
            return ResearchRoadmap(objective=objective, tasks=[])

        prompt = self._build_planning_prompt(objective)

        try:
            response = self._llm.generate(prompt)
            tasks = self._parse_tasks(response)
            if not tasks:
                logger.warning(f"Generator produced empty plan for: {objective[:30]}")
                tasks = [
                    ResearchTask(
                        id="T1",
                        description=f"Research {objective}",
                        status=TaskStatus.PENDING,
                    )
                ]

            return ResearchRoadmap(objective=objective, tasks=tasks)

        except Exception as e:
            logger.error(f"Roadmap generation failed: {e}")
            return ResearchRoadmap(objective=objective, tasks=[])

    def _build_planning_prompt(self, objective: str) -> str:
        return (
            f"SYSTEM: {self._persona.system_prompt}\n\n"
            f"OBJECTIVE: {objective}\n\n"
            "Format each task as a JSON object with 'id', 'description', and 'estimated_effort'. "
            'Example: {"id": "T1", "description": "Find definition of X", "estimated_effort": "low"}'
        )

    def _parse_tasks(self, response: str) -> List[ResearchTask]:
        """Attempt to parse JSON tasks from LLM response."""
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return [ResearchTask(**item) for item in data[:8]]
            return []
        except (json.JSONDecodeError, TypeError) as e:
            logger.debug(f"JSON parse failed for roadmap: {e}")
            return []
