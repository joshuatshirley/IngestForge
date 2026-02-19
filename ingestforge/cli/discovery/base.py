"""Base class for discovery commands."""

from typing import Any, Optional, Dict
import json
import logging
from ingestforge.cli.core import IngestForgeCommand, ProgressManager

logger = logging.getLogger(__name__)


class DiscoveryCommand(IngestForgeCommand):
    """Base class for resource discovery commands."""

    def get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client."""
        try:
            from ingestforge.llm.factory import get_best_available_client

            return get_best_available_client(ctx["config"])
        except Exception:
            return None

    def search_context(self, storage: Any, query: str, k: int = 20) -> list[Any]:
        """Search for context."""
        return ProgressManager.run_with_spinner(
            lambda: storage.search(query, k=k),
            f"Searching for '{query}'...",
            "Context retrieved",
        )

    def generate_with_llm(self, llm_client: Any, prompt: str, desc: str) -> str:
        """Generate with LLM."""
        return ProgressManager.run_with_spinner(
            lambda: llm_client.generate(prompt),
            f"Generating {desc}...",
            "Complete",
        )

    def parse_json(self, response: str) -> Optional[Dict]:
        """Parse JSON response."""
        try:
            return json.loads(response)
        except (json.JSONDecodeError, ValueError) as e:
            # Fallback: Try to extract JSON from text
            logger.warning(f"Initial JSON parse failed: {e}. Attempting extraction.")
            import re

            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except (json.JSONDecodeError, ValueError) as e2:
                    logger.warning(f"JSON extraction failed: {e2}")
                    pass
            return None
