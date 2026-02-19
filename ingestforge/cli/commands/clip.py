"""Clip command - Copy context to clipboard for LLM use.

Copy-Paste Ready CLI Interfaces
Epic: EP-08 (Structured Data Foundry)
Feature: FE-08-04 (Copy-Paste Ready CLI Interfaces)

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_CHUNKS)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import typer

from ingestforge.core.clipboard import (
    ClipboardResult,
    copy_to_clipboard,
    is_clipboard_available,
)
from ingestforge.core.llm_formatter import (
    ContextChunk,
    FormatterContext,
    FormattedOutput,
    LLMFormat,
    get_formatter,
)

# JPL Rule #2: Fixed upper bounds
MAX_CHUNKS = 20
MAX_QUERY_LENGTH = 500


class ClipCommand:
    """Copy retrieved context to clipboard for LLM use.

    Retrieves relevant chunks for a query and copies them to the
    system clipboard in a format optimized for the target LLM.
    """

    def __init__(self) -> None:
        """Initialize clip command."""
        self._console_available = True

    def execute(
        self,
        query: str,
        format_type: str = "markdown",
        k: int = 10,
        cite: bool = True,
        stdout: bool = False,
        project: Optional[Path] = None,
    ) -> int:
        """Execute clip command.

        Args:
            query: Query to retrieve context for.
            format_type: Output format (chatgpt, claude, plain, markdown).
            k: Number of chunks to retrieve.
            cite: Include citations.
            stdout: Output to stdout instead of clipboard.
            project: Project directory.

        Returns:
            Exit code (0 = success).
        """
        # JPL Rule #5: Assert preconditions
        assert query is not None, "query cannot be None"
        assert k > 0, "k must be positive"

        # JPL Rule #2: Enforce bounds
        k = min(k, MAX_CHUNKS)
        query = query[:MAX_QUERY_LENGTH]

        try:
            # Parse format type
            llm_format = self._parse_format(format_type)

            # Initialize storage and retrieve chunks
            chunks = self._retrieve_chunks(query, k, project)

            if not chunks:
                self._print_warning(f"No context found for: {query}")
                return 0

            # Format output
            formatted = self._format_output(query, chunks, llm_format, cite)

            # Output to clipboard or stdout
            if stdout or not is_clipboard_available():
                self._output_stdout(formatted)
            else:
                result = self._output_clipboard(formatted)
                if not result.success:
                    self._print_error(f"Clipboard failed: {result.message}")
                    self._output_stdout(formatted)

            return 0

        except ValueError as e:
            self._print_error(str(e))
            return 1
        except Exception as e:
            self._print_error(f"Clip failed: {e}")
            return 1

    def _parse_format(self, format_type: str) -> LLMFormat:
        """Parse format string to LLMFormat enum.

        Args:
            format_type: Format string.

        Returns:
            LLMFormat enum value.

        Raises:
            ValueError: If format is invalid.
        """
        format_map = {
            "chatgpt": LLMFormat.CHATGPT,
            "gpt": LLMFormat.CHATGPT,
            "claude": LLMFormat.CLAUDE,
            "plain": LLMFormat.PLAIN,
            "text": LLMFormat.PLAIN,
            "markdown": LLMFormat.MARKDOWN,
            "md": LLMFormat.MARKDOWN,
        }

        llm_format = format_map.get(format_type.lower())
        if llm_format is None:
            valid = ", ".join(format_map.keys())
            raise ValueError(f"Invalid format '{format_type}'. Valid: {valid}")

        return llm_format

    def _retrieve_chunks(
        self, query: str, k: int, project: Optional[Path]
    ) -> list[dict[str, Any]]:
        """Retrieve relevant chunks for query.

        Args:
            query: Search query.
            k: Number of chunks.
            project: Project directory.

        Returns:
            List of chunk dictionaries.
        """
        # Try to initialize storage
        try:
            from ingestforge.storage.factory import create_storage

            storage = create_storage(project)
            results = storage.search(query, k=k)

            chunks = []
            for i, result in enumerate(results[:k], 1):
                chunk_data = {
                    "text": getattr(result, "text", str(result)),
                    "source": getattr(result, "metadata", {}).get("source", "Unknown"),
                    "score": getattr(result, "score", 1.0),
                    "page": getattr(result, "metadata", {}).get("page"),
                    "section": getattr(result, "metadata", {}).get("section"),
                }
                chunks.append(chunk_data)

            return chunks

        except ImportError:
            # Storage not available, return empty
            return []
        except Exception:
            # Storage error, return empty
            return []

    def _format_output(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        llm_format: LLMFormat,
        cite: bool,
    ) -> FormattedOutput:
        """Format chunks for output.

        Args:
            query: Original query.
            chunks: Retrieved chunks.
            llm_format: Output format.
            cite: Include citations.

        Returns:
            Formatted output.
        """
        context_chunks = []
        for i, chunk in enumerate(chunks, 1):
            context_chunks.append(
                ContextChunk(
                    text=chunk.get("text", ""),
                    source=chunk.get("source", "Unknown"),
                    rank=i,
                    score=chunk.get("score", 1.0),
                    page=chunk.get("page"),
                    section=chunk.get("section"),
                )
            )

        context = FormatterContext(
            query=query,
            chunks=context_chunks,
            include_citations=cite,
        )

        formatter = get_formatter(llm_format)
        return formatter.format(context)

    def _output_clipboard(self, formatted: FormattedOutput) -> ClipboardResult:
        """Copy formatted output to clipboard.

        Args:
            formatted: Formatted output.

        Returns:
            Clipboard operation result.
        """
        result = copy_to_clipboard(formatted.content)

        if result.success:
            self._print_success(
                f"Copied {formatted.chunk_count} chunks "
                f"({result.chars_copied} chars) to clipboard"
            )
            if formatted.truncated:
                self._print_warning("Content was truncated")

        return result

    def _output_stdout(self, formatted: FormattedOutput) -> None:
        """Output formatted content to stdout.

        Args:
            formatted: Formatted output.
        """
        print(formatted.content)
        self._print_info(
            f"\n[{formatted.chunk_count} chunks, {formatted.char_count} chars]"
        )

    def _print_success(self, message: str) -> None:
        """Print success message."""
        print(f"[OK] {message}")

    def _print_warning(self, message: str) -> None:
        """Print warning message."""
        print(f"[WARN] {message}")

    def _print_error(self, message: str) -> None:
        """Print error message."""
        print(f"[ERROR] {message}")

    def _print_info(self, message: str) -> None:
        """Print info message."""
        print(message)


# Typer command
def command(
    query: str = typer.Argument(..., help="Query to retrieve context for"),
    format_type: str = typer.Option(
        "markdown",
        "--format",
        "-f",
        help="Output format: chatgpt, claude, plain, markdown",
    ),
    k: int = typer.Option(10, "--top-k", "-k", help="Number of chunks"),
    cite: bool = typer.Option(True, "--cite/--no-cite", help="Include citations"),
    stdout: bool = typer.Option(
        False, "--stdout", "-s", help="Output to stdout instead of clipboard"
    ),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
) -> None:
    """Copy retrieved context to clipboard for LLM use.

    Retrieves relevant chunks for a query and copies them to the
    system clipboard in a format optimized for the target LLM.

    Examples:
        ingestforge clip "machine learning basics"
        ingestforge clip "Python async" --format chatgpt
        ingestforge clip "API design" -f claude --no-cite
        ingestforge clip "testing" --stdout
    """
    cmd = ClipCommand()
    exit_code = cmd.execute(query, format_type, k, cite, stdout, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
