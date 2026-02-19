"""Base class for study commands.

Provides common functionality for quiz and flashcard generation.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import json

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class StudyCommand(IngestForgeCommand):
    """Base class for study commands."""

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

    def search_topic_context(self, storage: Any, topic: str, k: int = 20) -> list[Any]:
        """Search for context about a topic.

        Args:
            storage: ChunkRepository instance
            topic: Topic to search for
            k: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: storage.search(topic, k=k),
            f"Searching knowledge base for '{topic}'...",
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
            # Extract text from chunk
            text = self._extract_chunk_text(chunk)

            # Format chunk with index
            chunk_text = f"[{idx}] {text}\n"

            # Check length limit (Commandment #3: Memory management)
            if current_length + len(chunk_text) > max_length:
                break

            context_parts.append(chunk_text)
            current_length += len(chunk_text)

        return "\n".join(context_parts)

    def _extract_chunk_text(self, chunk: Any) -> str:
        """Extract text from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Chunk text content
        """
        if isinstance(chunk, dict):
            return chunk.get("text", "")
        elif hasattr(chunk, "text"):
            return chunk.text
        else:
            return str(chunk)

    def generate_with_llm(
        self, llm_client: Any, prompt: str, task_description: str
    ) -> str:
        """Generate content using LLM.

        Args:
            llm_client: LLM provider instance
            prompt: Prompt to send to LLM
            task_description: Description for progress display

        Returns:
            Generated text
        """
        return ProgressManager.run_with_spinner(
            lambda: self._generate_content(llm_client, prompt),
            f"Generating {task_description}...",
            f"{task_description.capitalize()} generated",
        )

    def _generate_content(self, llm_client: Any, prompt: str) -> str:
        """Generate content (internal helper).

        Args:
            llm_client: LLM provider instance
            prompt: Prompt text

        Returns:
            Generated response
        """
        # Handle different LLM provider APIs
        if hasattr(llm_client, "generate"):
            return llm_client.generate(prompt)
        elif hasattr(llm_client, "complete"):
            return llm_client.complete(prompt)
        elif callable(llm_client):
            return llm_client(prompt)
        else:
            raise TypeError(f"Unknown LLM client type: {type(llm_client)}")

    def parse_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response.

        Args:
            response: LLM response text

        Returns:
            Parsed JSON dict or None if invalid
        """
        # Try to extract JSON from response
        response = response.strip()

        # Handle markdown code blocks
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            if end > start:
                response = response[start:end].strip()

        # Try to parse JSON
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            self.print_warning(f"Failed to parse JSON response: {e}")
            return None

    def validate_topic(self, topic: str) -> None:
        """Validate topic string.

        Args:
            topic: Topic to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if not topic or not topic.strip():
            raise typer.BadParameter("Topic cannot be empty")

        if len(topic) > 200:
            raise typer.BadParameter(
                f"Topic too long: {len(topic)} characters (max 200)"
            )

    def validate_count(self, count: int, min_val: int = 1, max_val: int = 100) -> None:
        """Validate count parameter.

        Args:
            count: Count to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if count < min_val:
            raise typer.BadParameter(f"Count must be at least {min_val}")

        if count > max_val:
            raise typer.BadParameter(f"Count cannot exceed {max_val}")

    def save_json_output(
        self, output_path: Any, data: Dict[str, Any], success_message: str
    ) -> None:
        """Save data to JSON file.

        Args:
            output_path: Path to output file
            data: Data to save
            success_message: Message to display on success
        """
        try:
            output_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.print_success(success_message)

        except Exception as e:
            self.print_warning(f"Failed to save to file: {e}")
