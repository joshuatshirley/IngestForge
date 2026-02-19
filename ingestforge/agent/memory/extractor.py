"""Agent Fact Extractor.

Extracts core findings from agent reports for long-term storage.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
import json
from typing import List, Dict, Any

from ingestforge.agent.memory.models import AgentFact
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient

logger = get_logger(__name__)


class FactExtractor:
    """Logic for distilling synthesis reports into atomic facts."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def extract_facts(self, synthesis: str, mission_id: str) -> List[AgentFact]:
        """Convert a synthesis report into a list of AgentFact objects.

        Rule #1: Linear logic with early returns.
        Rule #7: Parameter validation.
        """
        if not synthesis or len(synthesis.strip()) < 50:
            return []

        prompt = self._build_extraction_prompt(synthesis)

        try:
            response = self._llm.generate(prompt)
            raw_facts = self._parse_json_facts(response)

            # Convert to SQLAlchemy models (Rule #4: Separation of Concerns)
            return [
                AgentFact(
                    fact_text=rf.get("fact", ""),
                    evidence_chunk_id=rf.get("chunk_id"),
                    source_title=rf.get("source"),
                    mission_id=mission_id,
                    confidence_score=rf.get("confidence", 1.0),
                )
                for rf in raw_facts[:5]
            ]

        except Exception as e:
            logger.error(f"Fact extraction failed: {e}")
            return []

    def _build_extraction_prompt(self, text: str) -> str:
        return (
            "You are a knowledge distiller. Extract the top 5 most important, "
            "verifiable facts from the following research report. "
            "Output ONLY a JSON list of objects with keys: 'fact', 'chunk_id', 'source', 'confidence'.\n\n"
            f"REPORT:\n{text}"
        )

    def _parse_json_facts(self, response: str) -> List[Dict[str, Any]]:
        """Safely parse JSON from LLM response."""
        try:
            data = json.loads(response)
            return data if isinstance(data, list) else []
        except Exception:
            return []
