"""Literary Analysis Extractor.

Extracts characters, themes, and arcs using specialized agent personas.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
import json
from typing import List, Dict, Any

from ingestforge.agent.personas import get_persona, PersonaType
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient

logger = get_logger(__name__)


class LiteraryExtractor:
    """Logic for high-fidelity narrative analysis."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client
        self._persona = get_persona(PersonaType.LITERARY)

    def analyze_characters(self, text: str) -> List[Dict[str, Any]]:
        """Identify characters and their core traits."""
        prompt = self._build_prompt("Extract all characters and their traits.", text)
        response = self._llm.generate(prompt)
        return self._parse_json(response, "characters")

    def analyze_themes(self, text: str) -> List[Dict[str, Any]]:
        """Identify thematic motifs and their prominence."""
        prompt = self._build_prompt("Extract primary themes and their frequency.", text)
        response = self._llm.generate(prompt)
        return self._parse_json(response, "themes")

    def _build_prompt(self, task: str, context: str) -> str:
        return (
            f"SYSTEM: {self._persona.system_prompt}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"TASK: {task}\n"
            "Output valid JSON ONLY."
        )

    def _parse_json(self, response: str, key: str) -> List[Dict[str, Any]]:
        """Safely parse JSON output from LLM."""
        try:
            data = json.loads(response)
            if isinstance(data, list):
                return data[:20]
            if isinstance(data, dict):
                return data.get(key, [])[:20]
            return []
        except Exception as e:
            logger.debug(f"Literary JSON parse failed: {e}")
            return []
