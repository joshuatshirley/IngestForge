"""Base class for transform commands.

Provides shared functionality for transformation operations.

Follows Commandments #4 (Small Functions), #6 (Smallest Scope),
and #9 (Type Safety).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any
import re

from ingestforge.cli.core.command_base import BaseCommand


class TransformCommand(BaseCommand):
    """Base class for transform commands."""

    def validate_chunk_size(
        self, chunk_size: int, min_size: int = 100, max_size: int = 10000
    ) -> None:
        """Validate chunk size parameter.

        Args:
            chunk_size: Size to validate
            min_size: Minimum allowed size
            max_size: Maximum allowed size

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if chunk_size < min_size:
            raise typer.BadParameter(f"Chunk size must be at least {min_size}")

        if chunk_size > max_size:
            raise typer.BadParameter(f"Chunk size must not exceed {max_size}")

    def validate_overlap(self, overlap: int, chunk_size: int) -> None:
        """Validate overlap parameter.

        Args:
            overlap: Overlap to validate
            chunk_size: Chunk size for comparison

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if overlap < 0:
            raise typer.BadParameter("Overlap must be non-negative")

        if overlap >= chunk_size:
            raise typer.BadParameter("Overlap must be less than chunk size")

    def clean_text_simple(self, text: str) -> str:
        """Clean text using simple rules.

        Args:
            text: Text to clean

        Returns:
            Cleaned text
        """
        # Remove multiple spaces (Commandment #1: Simple control flow)
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text

    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace in text.

        Args:
            text: Text to normalize

        Returns:
            Normalized text
        """
        # Replace tabs with spaces
        text = text.replace("\t", "    ")

        # Remove trailing spaces from lines
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]

        # Remove multiple blank lines
        cleaned_lines: List[str] = []
        prev_blank = False

        for line in lines:
            is_blank = len(line.strip()) == 0

            if not is_blank:
                cleaned_lines.append(line)
                prev_blank = False
            elif not prev_blank:
                cleaned_lines.append("")
                prev_blank = True

        return "\n".join(cleaned_lines)

    def split_into_chunks(
        self, text: str, chunk_size: int, overlap: int = 0
    ) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to split
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        chunks: List[str] = []
        start = 0
        text_len = len(text)

        # Simple chunking with overlap (Commandment #1: Simple control flow)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]

            if len(chunk.strip()) > 0:
                chunks.append(chunk)

            start += chunk_size - overlap

        return chunks

    def merge_chunks(self, chunks: List[str], separator: str = "\n\n") -> str:
        """Merge chunks into single text.

        Args:
            chunks: Chunks to merge
            separator: Separator between chunks

        Returns:
            Merged text
        """
        return separator.join(chunks)

    def extract_metadata_simple(self, text: str) -> Dict[str, Any]:
        """Extract simple metadata from text.

        Args:
            text: Text to analyze

        Returns:
            Metadata dictionary
        """
        lines = text.split("\n")
        words = text.split()

        return {
            "char_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "avg_word_length": sum(len(w) for w in words) / max(len(words), 1),
            "blank_lines": sum(1 for line in lines if not line.strip()),
        }

    def create_transform_summary(self, results: Dict[str, Any], operation: str) -> str:
        """Create summary of transform operation.

        Args:
            results: Operation results
            operation: Operation name

        Returns:
            Summary text
        """
        from rich.panel import Panel

        lines = [
            f"[bold]{operation} Results[/bold]",
            "",
            f"Input items: {results.get('input_count', 0)}",
            f"Output items: {results.get('output_count', 0)}",
            f"Successful: {results.get('successful', 0)}",
            f"Failed: {results.get('failed', 0)}",
        ]

        if results.get("errors"):
            lines.append("")
            lines.append("[yellow]Errors:[/yellow]")
            for error in results["errors"][:3]:
                lines.append(f"  • {error}")

        return Panel("\n".join(lines), border_style="cyan")

    def save_transform_report(
        self, output: Path, results: Dict[str, Any], operation: str
    ) -> None:
        """Save transform operation report.

        Args:
            output: Output file path
            results: Operation results
            operation: Operation name
        """
        lines = [
            f"# Transform Report: {operation}",
            "",
            f"**Date:** {self._get_timestamp()}",
            "",
            "## Summary",
            "",
            f"- **Input Items:** {results.get('input_count', 0)}",
            f"- **Output Items:** {results.get('output_count', 0)}",
            f"- **Successful:** {results.get('successful', 0)}",
            f"- **Failed:** {results.get('failed', 0)}",
            "",
        ]

        if results.get("errors"):
            lines.extend(
                [
                    "## Errors",
                    "",
                ]
            )
            for error in results["errors"]:
                lines.append(f"- {error}")
            lines.append("")

        if results.get("details"):
            lines.extend(
                [
                    "## Details",
                    "",
                ]
            )
            for detail in results["details"]:
                lines.append(f"- {detail}")
            lines.append("")

        report_text = "\n".join(lines)

        try:
            output.write_text(report_text, encoding="utf-8")
            self.print_success(f"Report saved: {output}")
        except Exception as e:
            self.print_error(f"Failed to save report: {e}")

    def _get_timestamp(self) -> str:
        """Get current timestamp string.

        Returns:
            Formatted timestamp
        """
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
