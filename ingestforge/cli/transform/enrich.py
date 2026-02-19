"""Enrich command - Enrich text with metadata.

Enriches text files with extracted metadata and analysis.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer

from ingestforge.cli.transform.base import TransformCommand


class EnrichCommand(TransformCommand):
    """Enrich text with metadata."""

    def execute(
        self,
        input_file: Path,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        extract_keywords: bool = True,
        add_stats: bool = True,
    ) -> int:
        """Enrich text file with metadata.

        Args:
            input_file: Input file to enrich
            project: Project directory
            output: Output file for enriched data
            extract_keywords: Extract keywords
            add_stats: Add statistics

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_file_path(input_file, must_exist=True)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)

            # Read file
            content = self._read_file(input_file)

            # Enrich content
            enriched = self._enrich_content(
                content,
                input_file,
                extract_keywords,
                add_stats,
            )

            # Display results
            self._display_enrichment(enriched)

            # Save enriched data
            if output:
                self._save_enriched(output, enriched)

            return 0

        except Exception as e:
            return self.handle_error(e, "Enrich operation failed")

    def _read_file(self, file_path: Path) -> str:
        """Read file content.

        Args:
            file_path: File to read

        Returns:
            File content

        Raises:
            ValueError: If file cannot be read
        """
        try:
            return file_path.read_text(encoding="utf-8")
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}")

    def _enrich_content(
        self,
        content: str,
        file_path: Path,
        extract_keywords: bool,
        add_stats: bool,
    ) -> Dict[str, Any]:
        """Enrich content with metadata.

        Args:
            content: Content to enrich
            file_path: Source file path
            extract_keywords: Whether to extract keywords
            add_stats: Whether to add statistics

        Returns:
            Enriched data dictionary
        """
        enriched: Dict[str, Any] = {
            "file": str(file_path),
            "content": content,
            "metadata": {},
        }

        # Add basic metadata
        enriched["metadata"]["file_size"] = len(content)
        enriched["metadata"]["file_name"] = file_path.name

        # Extract keywords if requested
        if extract_keywords:
            keywords = self._extract_keywords(content)
            enriched["metadata"]["keywords"] = keywords[:20]

        # Add statistics if requested
        if add_stats:
            stats = self.extract_metadata_simple(content)
            enriched["metadata"]["statistics"] = stats

        return enriched

    def _extract_keywords(self, text: str) -> list[Any]:
        """Extract keywords from text.

        Args:
            text: Text to analyze

        Returns:
            List of keywords
        """
        import re
        from collections import Counter

        # Simple keyword extraction
        words = re.findall(r"\b[a-z]{4,}\b", text.lower())

        # Count frequency
        word_counts = Counter(words)

        # Filter common words
        common_words = {
            "that",
            "this",
            "with",
            "from",
            "have",
            "been",
            "will",
            "would",
            "could",
            "should",
            "their",
            "about",
            "which",
            "there",
            "these",
            "those",
            "when",
            "where",
        }

        filtered = {
            word: count
            for word, count in word_counts.items()
            if word not in common_words
        }

        # Sort by frequency
        sorted_words = sorted(filtered.items(), key=lambda x: x[1], reverse=True)

        return [word for word, _ in sorted_words]

    def _display_enrichment(self, enriched: Dict[str, Any]) -> None:
        """Display enrichment results.

        Args:
            enriched: Enriched data
        """
        from rich.panel import Panel

        metadata = enriched["metadata"]

        lines = [
            "[bold]Enrichment Results[/bold]",
            "",
            f"File: {enriched['file']}",
        ]

        if "statistics" in metadata:
            stats = metadata["statistics"]
            lines.extend(
                [
                    "",
                    "[bold]Statistics:[/bold]",
                    f"  Characters: {stats['char_count']:,}",
                    f"  Words: {stats['word_count']:,}",
                    f"  Lines: {stats['line_count']:,}",
                    f"  Avg word length: {stats['avg_word_length']:.1f}",
                ]
            )

        if "keywords" in metadata:
            keywords = metadata["keywords"]
            lines.extend(
                [
                    "",
                    "[bold]Top Keywords:[/bold]",
                    f"  {', '.join(keywords[:10])}",
                ]
            )

        panel = Panel("\n".join(lines), border_style="cyan")
        self.console.print(panel)

    def _save_enriched(self, output: Path, enriched: Dict[str, Any]) -> None:
        """Save enriched data.

        Args:
            output: Output file path
            enriched: Enriched data
        """
        import json

        try:
            with output.open("w", encoding="utf-8") as f:
                json.dump(enriched, f, indent=2, ensure_ascii=False)

            self.print_success(f"Enriched data saved: {output}")

        except Exception as e:
            self.print_error(f"Failed to save enriched data: {e}")


# Typer command wrapper
def command(
    input_file: Path = typer.Argument(..., help="Input file to enrich"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for enriched data (JSON)"
    ),
    extract_keywords: bool = typer.Option(
        True, "--keywords/--no-keywords", help="Extract keywords"
    ),
    add_stats: bool = typer.Option(True, "--stats/--no-stats", help="Add statistics"),
) -> None:
    """Enrich text file with metadata.

    Analyzes text and extracts metadata including keywords,
    statistics, and other enrichments.

    Features:
    - Keyword extraction
    - Text statistics
    - Metadata generation
    - JSON output format

    Examples:
        # Basic enrichment
        ingestforge transform enrich document.txt -o enriched.json

        # Keywords only
        ingestforge transform enrich doc.txt --no-stats -o keywords.json

        # Statistics only
        ingestforge transform enrich doc.txt --no-keywords -o stats.json

        # Preview without saving
        ingestforge transform enrich document.txt

        # Specific project
        ingestforge transform enrich doc.txt -p /path/to/project -o data.json
    """
    cmd = EnrichCommand()
    exit_code = cmd.execute(input_file, project, output, extract_keywords, add_stats)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
