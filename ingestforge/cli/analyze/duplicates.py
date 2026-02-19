"""Duplicates command - Detect duplicate or near-duplicate content."""

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.table import Table
from ingestforge.cli.analyze.base import AnalyzeCommand


class DuplicatesCommand(AnalyzeCommand):
    """Detect duplicate and near-duplicate content."""

    def execute(
        self,
        threshold: float = 0.9,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """Detect duplicates in knowledge base."""
        try:
            ctx = self.initialize_context(project, require_storage=True)

            # Get all chunks
            all_chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not all_chunks:
                self.print_warning("No content in knowledge base")
                return 0

            if len(all_chunks) < 2:
                self.print_warning("Need at least 2 chunks to detect duplicates")
                return 0

            # Detect duplicates
            duplicates = self._detect_duplicates(all_chunks, threshold)

            if not duplicates:
                self.print_success("No duplicates found!")
                return 0

            # Display duplicates
            self._display_duplicates(duplicates, threshold)

            # Save if requested
            if output:
                self.save_json_output(
                    output,
                    {"duplicates": duplicates, "threshold": threshold},
                    f"Duplicates report saved to: {output}",
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Duplicate detection failed")

    def _detect_duplicates(
        self, chunks: list, threshold: float
    ) -> List[Dict[str, Any]]:
        """Detect duplicate chunks."""
        duplicates = []

        # Compare each chunk with others
        for i in range(len(chunks)):
            chunk_i = chunks[i]
            text_i = getattr(chunk_i, "text", str(chunk_i))
            metadata_i = self.extract_chunk_metadata(chunk_i)

            for j in range(i + 1, len(chunks)):
                chunk_j = chunks[j]
                text_j = getattr(chunk_j, "text", str(chunk_j))
                metadata_j = self.extract_chunk_metadata(chunk_j)

                # Calculate similarity
                similarity = self._calculate_text_similarity(text_i, text_j)

                if similarity >= threshold:
                    duplicates.append(
                        {
                            "chunk1_source": metadata_i.get("source", "Unknown"),
                            "chunk2_source": metadata_j.get("source", "Unknown"),
                            "similarity": round(similarity, 3),
                            "preview1": text_i[:100],
                            "preview2": text_j[:100],
                        }
                    )

        return duplicates

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple Jaccard similarity on words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _display_duplicates(self, duplicates: List[Dict], threshold: float) -> None:
        """Display duplicate detection results."""
        self.console.print()
        self.console.print(
            f"[bold cyan]Duplicate Detection (threshold: {threshold})[/bold cyan]\n"
        )

        self.print_warning(f"Found {len(duplicates)} duplicate pairs")

        table = Table(title="Duplicate Pairs", show_lines=True)
        table.add_column("Source 1", width=25)
        table.add_column("Source 2", width=25)
        table.add_column("Similarity", width=12)
        table.add_column("Preview", width=40)

        for dup in duplicates[:20]:  # Limit display
            table.add_row(
                dup["chunk1_source"],
                dup["chunk2_source"],
                f"{dup['similarity']:.1%}",
                dup["preview1"][:40] + "...",
            )

        self.console.print(table)

        if len(duplicates) > 20:
            self.print_info(
                f"Showing 20 of {len(duplicates)} duplicates. "
                f"Use -o to save full report."
            )


def command(
    threshold: float = typer.Option(
        0.9,
        "--threshold",
        "-t",
        min=0.0,
        max=1.0,
        help="Similarity threshold (0.0-1.0)",
    ),
    project: Optional[Path] = typer.Option(None, "-p", help="Project directory"),
    output: Optional[Path] = typer.Option(None, "-o", help="Output file"),
) -> None:
    """Detect duplicate and near-duplicate content.

    Identifies chunks with high similarity that may be duplicates.
    Higher threshold means stricter duplicate detection.

    Examples:
        ingestforge analyze duplicates
        ingestforge analyze duplicates --threshold 0.95
        ingestforge analyze duplicates -t 0.8 -o duplicates.json
    """
    cmd = DuplicatesCommand()
    exit_code = cmd.execute(threshold, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
