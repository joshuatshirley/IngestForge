"""Base class for argument building commands.

Provides common functionality for debate, support, counter, conflicts, and gaps analysis.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope), and #9 (Type Safety).
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional
import json

from ingestforge.cli.core import IngestForgeCommand, ProgressManager

logger = logging.getLogger(__name__)


class ArgumentCommand(IngestForgeCommand):
    """Base class for argument building commands."""

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

    def search_claim_context(self, storage: Any, claim: str, k: int = 30) -> list[Any]:
        """Search for context about a claim or topic.

        Args:
            storage: ChunkRepository instance
            claim: Claim or topic to search for
            k: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: storage.search(claim, k=k),
            f"Searching for evidence about '{claim}'...",
            "Context retrieved",
        )

    def format_context_for_prompt(self, chunks: list, max_length: int = 5000) -> str:
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
            return json.loads(response)
        except json.JSONDecodeError:
            import re

            # Try to extract JSON from markdown code blocks
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

    def validate_claim(self, claim: str) -> None:
        """Validate claim parameter.

        Args:
            claim: Claim string to validate

        Raises:
            typer.BadParameter: If claim is invalid
        """
        import typer

        if not claim or not claim.strip():
            raise typer.BadParameter("Claim cannot be empty")

        if len(claim) < 5:
            raise typer.BadParameter("Claim too short (minimum 5 characters)")

        if len(claim) > 500:
            raise typer.BadParameter("Claim too long (maximum 500 characters)")
