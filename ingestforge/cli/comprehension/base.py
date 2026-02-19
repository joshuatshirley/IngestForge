"""Base class for comprehension commands.

Provides common functionality for explaining, comparing, and connecting concepts.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import json

from ingestforge.cli.core import IngestForgeCommand, ProgressManager

logger = logging.getLogger(__name__)


class ComprehensionCommand(IngestForgeCommand):
    """Base class for comprehension commands."""

    def get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client for generation.

        Args:
            ctx: Context dictionary with config

        Returns:
            LLM client instance or None
        """
        try:
            from ingestforge.llm.factory import get_best_available_client

            client = get_best_available_client(ctx["config"])

            if client is None:
                self.print_warning(
                    "No LLM available. Install a provider:\n"
                    "  pip install anthropic  # For Claude\n"
                    "  pip install openai     # For OpenAI"
                )

            return client

        except Exception as e:
            self.print_warning(f"Failed to load LLM client: {e}")
            return None

    def search_concept_context(
        self, storage: Any, concept: str, k: int = 20
    ) -> list[Any]:
        """Search for context about a concept.

        Args:
            storage: ChunkRepository instance
            concept: Concept to search for
            k: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: storage.search(concept, k=k),
            f"Searching for '{concept}'...",
            "Context retrieved",
        )

    def format_context_for_prompt(self, chunks: list, max_length: int = 4000) -> str:
        """Format chunks as context for LLM prompt.

        Args:
            chunks: List of chunks
            max_length: Maximum context length

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []
        current_length = 0

        for idx, chunk in enumerate(chunks, 1):
            chunk_text = getattr(chunk, "text", str(chunk))

            # Add source if available
            source = getattr(chunk, "source_file", "unknown")
            chunk_header = f"[Source {idx}: {source}]\n"

            chunk_content = f"{chunk_header}{chunk_text}\n\n"
            chunk_length = len(chunk_content)

            if current_length + chunk_length > max_length:
                break

            context_parts.append(chunk_content)
            current_length += chunk_length

        return "".join(context_parts)

    def generate_with_llm(
        self, llm_client: Any, prompt: str, task_description: str
    ) -> str:
        """Generate text using LLM.

        Args:
            llm_client: LLM provider instance
            prompt: Prompt text
            task_description: Description for spinner

        Returns:
            Generated text
        """
        return ProgressManager.run_with_spinner(
            lambda: llm_client.generate(prompt),
            f"Generating {task_description}...",
            "Generation complete",
        )

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict or None if failed
        """
        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            import re

            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL
            )

            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from markdown block: {e}")

            # Try to find JSON object
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse JSON from extracted object: {e}")

            self.print_warning("Could not parse JSON from response")
            return None

    def validate_concept(self, concept: str) -> None:
        """Validate concept parameter.

        Args:
            concept: Concept string to validate

        Raises:
            typer.BadParameter: If concept is invalid
        """
        import typer

        if not concept or not concept.strip():
            raise typer.BadParameter("Concept cannot be empty")

        if len(concept) < 2:
            raise typer.BadParameter("Concept too short")

        if len(concept) > 200:
            raise typer.BadParameter("Concept too long (max 200 characters)")
