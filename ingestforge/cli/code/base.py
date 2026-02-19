"""Base class for code analysis commands.

Provides common functionality for analyzing code and generating documentation.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Set
from pathlib import Path

from ingestforge.cli.core import IngestForgeCommand, ProgressManager


class CodeAnalysisCommand(IngestForgeCommand):
    """Base class for code analysis commands."""

    def get_llm_client(self, ctx: dict) -> Optional[Any]:
        """Get LLM client for analysis.

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

    def search_code_context(self, storage: Any, query: str, k: int = 20) -> list[Any]:
        """Search for code-related context.

        Args:
            storage: ChunkRepository instance
            query: Search query
            k: Number of chunks to retrieve

        Returns:
            List of relevant chunks
        """
        return ProgressManager.run_with_spinner(
            lambda: storage.search(query, k=k),
            f"Searching for code: '{query}'...",
            "Code context retrieved",
        )

    def find_code_files(
        self, directory: Path, extensions: Optional[Set[str]] = None
    ) -> List[Path]:
        """Find code files in directory.

        Args:
            directory: Directory to search
            extensions: Set of file extensions to include

        Returns:
            List of code file paths
        """
        if extensions is None:
            extensions = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".go"}

        code_files = []

        for ext in extensions:
            code_files.extend(directory.rglob(f"*{ext}"))

        return sorted(code_files)

    def extract_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic information from code file.

        Args:
            file_path: Path to code file

        Returns:
            Dictionary with file information
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            return {
                "path": str(file_path),
                "name": file_path.name,
                "extension": file_path.suffix,
                "lines": len(lines),
                "size": len(content),
                "exists": True,
            }

        except Exception as e:
            return {
                "path": str(file_path),
                "name": file_path.name,
                "extension": file_path.suffix,
                "lines": 0,
                "size": 0,
                "exists": False,
                "error": str(e),
            }

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

    def validate_code_path(self, path: Path) -> None:
        """Validate code path.

        Args:
            path: Path to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if not path.exists():
            raise typer.BadParameter(f"Path does not exist: {path}")

        if not (path.is_file() or path.is_dir()):
            raise typer.BadParameter(f"Path is not a file or directory: {path}")

    def group_files_by_extension(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by extension.

        Args:
            files: List of file paths

        Returns:
            Dictionary mapping extension to file list
        """
        grouped: Dict[str, List[Path]] = {}

        for file_path in files:
            ext = file_path.suffix or "no_extension"

            if ext not in grouped:
                grouped[ext] = []

            grouped[ext].append(file_path)

        return grouped
