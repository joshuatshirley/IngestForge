"""Similarity command - Find similar documents and duplicates.

Identifies similar content and potential duplicates in the knowledge base.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.table import Table

from ingestforge.cli.analyze.base import AnalyzeCommand


class SimilarityCommand(AnalyzeCommand):
    """Find similar documents and duplicates."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        threshold: float = 0.7,
        limit: int = 20,
    ) -> int:
        """Find similar and duplicate content.

        Args:
            project: Project directory
            output: Output file for results
            threshold: Similarity threshold (0-1)
            limit: Maximum number of pairs to show

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_threshold(threshold)
            self.validate_limit(limit)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve chunks
            chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not chunks:
                self._handle_no_chunks()
                return 0

            if len(chunks) < 2:
                self.print_warning("Need at least 2 chunks to compare")
                return 0

            # Find similar pairs
            similar_pairs = self._find_similar_pairs(chunks, threshold, limit)

            # Display results
            self._display_similar_pairs(similar_pairs, threshold)

            # Save to file if requested
            if output:
                self._save_similarity_results(output, similar_pairs, threshold)

            return 0

        except Exception as e:
            return self.handle_error(e, "Similarity analysis failed")

    def validate_threshold(self, threshold: float) -> None:
        """Validate similarity threshold.

        Args:
            threshold: Threshold to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if threshold < 0.0 or threshold > 1.0:
            raise typer.BadParameter("Threshold must be between 0.0 and 1.0")

    def validate_limit(self, limit: int) -> None:
        """Validate result limit.

        Args:
            limit: Limit to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if limit < 1:
            raise typer.BadParameter("Limit must be at least 1")

        if limit > 1000:
            raise typer.BadParameter("Limit cannot exceed 1000")

    def _handle_no_chunks(self) -> None:
        """Handle case where no chunks found."""
        self.print_warning("Knowledge base is empty")
        self.print_info("Try ingesting some documents first")

    def _find_similar_pairs(
        self, chunks: list, threshold: float, limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar chunk pairs.

        Args:
            chunks: List of chunks
            threshold: Similarity threshold
            limit: Maximum pairs to return

        Returns:
            List of similar pair data
        """
        from ingestforge.cli.core import ProgressManager

        return ProgressManager.run_with_spinner(
            lambda: self._compute_similarities(chunks, threshold, limit),
            "Computing similarities...",
            "Similarity analysis complete",
        )

    def _create_pair_data(
        self,
        i: int,
        j: int,
        text1: str,
        text2: str,
        meta1: Dict[str, Any],
        meta2: Dict[str, Any],
        similarity: float,
    ) -> Dict[str, Any]:
        """
        Create similarity pair data structure.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            i: First chunk index
            j: Second chunk index
            text1: First chunk text
            text2: Second chunk text
            meta1: First chunk metadata
            meta2: Second chunk metadata
            similarity: Similarity score

        Returns:
            Pair data dictionary
        """
        assert i >= 0, "First index must be non-negative"
        assert j > i, "Second index must be greater than first"
        assert 0.0 <= similarity <= 1.0, "Similarity must be in [0, 1]"

        return {
            "chunk1_idx": i,
            "chunk2_idx": j,
            "source1": meta1.get("source", "unknown"),
            "source2": meta2.get("source", "unknown"),
            "similarity": similarity,
            "text1_preview": text1[:100],
            "text2_preview": text2[:100],
        }

    def _should_stop_comparison(
        self,
        comparisons_done: int,
        max_comparisons: int,
        similar_pairs_count: int,
        limit: int,
    ) -> bool:
        """
        Check if similarity comparison should stop.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            comparisons_done: Number of comparisons completed
            max_comparisons: Maximum comparisons allowed
            similar_pairs_count: Number of similar pairs found
            limit: Maximum pairs to find

        Returns:
            True if should stop, False otherwise
        """
        assert comparisons_done >= 0, "Comparisons done must be non-negative"
        assert max_comparisons > 0, "Max comparisons must be positive"
        assert similar_pairs_count >= 0, "Similar pairs count must be non-negative"
        assert limit > 0, "Limit must be positive"
        return comparisons_done >= max_comparisons or similar_pairs_count >= limit

    def _compare_chunk_pair(
        self,
        i: int,
        j: int,
        chunks: list,
        threshold: float,
    ) -> Optional[Dict[str, Any]]:
        """
        Compare a single pair of chunks.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            i: First chunk index
            j: Second chunk index
            chunks: List of all chunks
            threshold: Similarity threshold

        Returns:
            Pair data if similar enough, None otherwise
        """
        assert i >= 0, "First index must be non-negative"
        assert j > i, "Second index must be greater than first"
        assert 0 <= i < len(chunks), "First index out of range"
        assert 0 <= j < len(chunks), "Second index out of range"

        # Extract chunk data
        text1 = self.extract_chunk_text(chunks[i])
        meta1 = self.extract_chunk_metadata(chunks[i])
        text2 = self.extract_chunk_text(chunks[j])
        meta2 = self.extract_chunk_metadata(chunks[j])

        # Calculate similarity
        similarity = self.calculate_text_similarity(text1, text2)
        if similarity < threshold:
            return None

        # Create pair data
        return self._create_pair_data(i, j, text1, text2, meta1, meta2, similarity)

    def _compare_pairs_pairwise(
        self, chunks: list, threshold: float, max_comparisons: int, limit: int
    ) -> List[Dict[str, Any]]:
        """Find similar chunk pairs using pairwise comparison.

        Rule #4: No large functions - Extracted from _compute_similarities
        """
        similar_pairs: List[Dict[str, Any]] = []
        comparisons_done: int = 0
        for i in range(len(chunks)):
            if self._should_stop_comparison(
                comparisons_done, max_comparisons, len(similar_pairs), limit
            ):
                break

            for j in range(i + 1, len(chunks)):
                if self._should_stop_comparison(
                    comparisons_done, max_comparisons, len(similar_pairs), limit
                ):
                    break
                pair_data = self._compare_chunk_pair(i, j, chunks, threshold)
                comparisons_done += 1
                if pair_data is not None:
                    similar_pairs.append(pair_data)

        return similar_pairs

    def _compute_similarities(
        self, chunks: list, threshold: float, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Compute similarities between chunks.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #2: Fixed upper bounds (max_comparisons)
        Rule #4: Function <60 lines (refactored to 29 lines)
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            chunks: List of chunks
            threshold: Similarity threshold
            limit: Maximum pairs

        Returns:
            List of similar pairs sorted by similarity
        """
        assert chunks is not None, "Chunks cannot be None"
        assert isinstance(chunks, list), "Chunks must be a list"
        assert 0.0 <= threshold <= 1.0, "Threshold must be in [0, 1]"
        assert limit > 0, "Limit must be positive"
        total_comparisons = (len(chunks) * (len(chunks) - 1)) // 2
        MAX_COMPARISONS: int = 10_000  # Safety limit
        max_comparisons = min(total_comparisons, MAX_COMPARISONS)

        # Find similar pairs
        similar_pairs = self._compare_pairs_pairwise(
            chunks, threshold, max_comparisons, limit
        )

        # Sort by similarity (descending)
        similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        result = similar_pairs[:limit]
        assert len(result) <= limit, "Result must not exceed limit"

        return result

    def _display_similar_pairs(
        self, similar_pairs: List[Dict[str, Any]], threshold: float
    ) -> None:
        """Display similar pairs.

        Args:
            similar_pairs: List of similar pairs
            threshold: Similarity threshold used
        """
        self.console.print()

        if not similar_pairs:
            self.print_info(f"No similar pairs found with threshold {threshold:.2f}")
            self.print_info("Try lowering the threshold (e.g., --threshold 0.5)")
            return

        self.print_info(
            f"Found {len(similar_pairs)} similar pairs " f"(threshold: {threshold:.2f})"
        )

        self.console.print()

        # Create table
        table = Table(title="Similar Content Pairs")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Similarity", style="green", width=12)
        table.add_column("Source 1", style="yellow", width=25)
        table.add_column("Source 2", style="magenta", width=25)
        table.add_column("Preview", style="dim", width=40)

        for idx, pair in enumerate(similar_pairs, 1):
            table.add_row(
                str(idx),
                f"{pair['similarity']:.2%}",
                pair["source1"][:24],
                pair["source2"][:24],
                pair["text1_preview"][:39] + "...",
            )

        self.console.print(table)

        # Show duplicates warning
        exact_duplicates = [p for p in similar_pairs if p["similarity"] > 0.95]

        if exact_duplicates:
            self.console.print()
            self.print_warning(
                f"Found {len(exact_duplicates)} potential duplicates " "(>95% similar)"
            )

    def _save_similarity_results(
        self, output: Path, similar_pairs: List[Dict[str, Any]], threshold: float
    ) -> None:
        """Save similarity results to file.

        Args:
            output: Output file path
            similar_pairs: List of similar pairs
            threshold: Similarity threshold
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            lines = [
                "# Similarity Analysis\n\n",
                f"**Generated:** {timestamp}\n",
                f"**Threshold:** {threshold:.2f}\n",
                f"**Pairs Found:** {len(similar_pairs)}\n\n",
                "---\n\n",
            ]

            if not similar_pairs:
                lines.append("No similar pairs found.\n")
            else:
                lines.append("## Similar Content Pairs\n\n")

                for idx, pair in enumerate(similar_pairs, 1):
                    lines.append(
                        f"### Pair {idx} - {pair['similarity']:.2%} similar\n\n"
                    )
                    lines.append(f"**Source 1:** {pair['source1']}\n")
                    lines.append(f"**Source 2:** {pair['source2']}\n\n")
                    lines.append(f"**Preview 1:** {pair['text1_preview']}...\n\n")
                    lines.append(f"**Preview 2:** {pair['text2_preview']}...\n\n")
                    lines.append("---\n\n")

            output.write_text("".join(lines), encoding="utf-8")
            self.print_success(f"Similarity results saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save results: {e}")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for results"
    ),
    threshold: float = typer.Option(
        0.7, "--threshold", "-t", help="Similarity threshold (0.0-1.0)"
    ),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum pairs to show"),
) -> None:
    """Find similar documents and potential duplicates.

    Analyzes content to identify similar chunks and potential
    duplicate content in your knowledge base.

    Similarity is calculated using text-based comparison.
    Higher threshold = more similar required.

    Examples:
        # Find very similar content (70% threshold)
        ingestforge analyze similarity

        # Find potential duplicates (95% threshold)
        ingestforge analyze similarity --threshold 0.95

        # Lower threshold for broader matches
        ingestforge analyze similarity --threshold 0.5

        # Save results
        ingestforge analyze similarity --output similar.md

        # Show more pairs
        ingestforge analyze similarity --limit 50

        # Specific project
        ingestforge analyze similarity -p /path/to/project
    """
    cmd = SimilarityCommand()
    exit_code = cmd.execute(project, output, threshold, limit)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
