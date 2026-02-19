"""Base class for literary analysis commands.

Provides common functionality for literary analysis operations:
- Access to storage and LLM
- Literary-specific result formatting
- Common validation patterns

Follows Commandments #4 (Small Functions) and #6 (Smallest Scope).
"""

from __future__ import annotations

from typing import Optional, Any, List

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class LiteraryCommand(IngestForgeCommand):
    """Base class for literary analysis commands.

    Extends IngestForgeCommand with literary-specific functionality.
    """

    def get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client from context.

        Args:
            ctx: Context dict with config

        Returns:
            LLM client or None if unavailable
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

    def search_literary_context(
        self, storage: Any, work: str, k: int = 20
    ) -> List[Any]:
        """Search for literary context about a work.

        Args:
            storage: ChunkRepository instance
            work: Name of literary work
            k: Number of results to retrieve

        Returns:
            List of relevant chunks
        """
        query = f"literary analysis {work}"

        return ProgressManager.run_with_spinner(
            lambda: storage.search(query, k=k),
            f"Searching for context about '{work}'...",
            f"Found {k} relevant chunks",
        )

    def build_literary_prompt(
        self, task: str, work: str, context: str, additional_instructions: str = ""
    ) -> str:
        """Build prompt for literary analysis task.

        Args:
            task: Type of analysis (themes, characters, etc.)
            work: Name of literary work
            context: Context from knowledge base
            additional_instructions: Extra instructions for LLM

        Returns:
            Formatted prompt string
        """
        prompt_parts = [
            f"Perform {task} analysis for the literary work: {work}",
            "",
            "Context from knowledge base:",
            context,
            "",
        ]

        if additional_instructions:
            prompt_parts.extend([additional_instructions, ""])

        prompt_parts.append(f"Provide detailed {task} analysis:")

        return "\n".join(prompt_parts)

    def generate_analysis(
        self, llm_client: Any, prompt: str, task_description: str
    ) -> str:
        """Generate analysis using LLM.

        Args:
            llm_client: LLM provider instance
            prompt: Formatted prompt
            task_description: Description for progress indicator

        Returns:
            Generated analysis text
        """
        return ProgressManager.run_with_spinner(
            lambda: llm_client.generate(prompt),
            f"Generating {task_description}...",
            f"{task_description.capitalize()} generated!",
        )

    def validate_work_name(self, work: str) -> None:
        """Validate literary work name.

        Args:
            work: Name of literary work

        Raises:
            ValueError: If work name is empty
        """
        self.validate_non_empty_string(work, "work name")

    def format_context_for_prompt(self, chunks: List[Any]) -> str:
        """Format search results into context string.

        Args:
            chunks: Retrieved chunks from search

        Returns:
            Formatted context string
        """
        if not chunks:
            return "No context available from knowledge base."

        # Limit to prevent prompt overflow (Commandment #3: Memory management)
        max_chunks = min(10, len(chunks))
        context_parts = []

        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = getattr(chunk, "text", str(chunk))
            # Limit chunk size (Commandment #2: Fixed upper bound)
            max_length = 500
            if len(text) > max_length:
                text = text[:max_length] + "..."

            context_parts.append(f"[{i}] {text}")

        return "\n\n".join(context_parts)
