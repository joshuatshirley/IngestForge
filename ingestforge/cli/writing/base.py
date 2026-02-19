"""Base class for writing commands."""

from typing import Any, Optional, Dict
import json
import logging
from ingestforge.cli.core import IngestForgeCommand, ProgressManager

logger = logging.getLogger(__name__)


class WritingCommand(IngestForgeCommand):
    """Base class for writing assistance commands."""

    def get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client.

        Rule #7: Proper exception handling with logging
        """
        try:
            from ingestforge.llm.factory import get_best_available_client

            return get_best_available_client(ctx["config"])
        except (ImportError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to get LLM client: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting LLM client: {e}")
            return None

    def search_context(self, storage: Any, query: str, k: int = 30) -> list[Any]:
        """Search for context."""
        return ProgressManager.run_with_spinner(
            lambda: storage.search(query, k=k),
            f"Searching for '{query}'...",
            "Context retrieved",
        )

    def format_context(self, chunks: list, max_length: int = 5000) -> str:
        """Format chunks as context."""
        if not chunks:
            return ""
        context_parts = []
        current_length = 0
        for chunk in chunks:
            text = getattr(chunk, "text", str(chunk))
            if current_length + len(text) > max_length:
                break
            context_parts.append(text)
            current_length += len(text)
        return "\n\n".join(context_parts)

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
                    return None
            return None
