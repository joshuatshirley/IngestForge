"""Audit command - Analyze knowledge base quality and coverage.

Audits the knowledge base to assess quality, coverage, and identify gaps.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.table import Table

from ingestforge.cli.research.base import ResearchCommand


class AuditCommand(ResearchCommand):
    """Audit knowledge base quality and coverage."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        detailed: bool = False,
    ) -> int:
        """Audit knowledge base quality and coverage.

        Args:
            project: Project directory
            output: Output file for audit report (optional)
            detailed: Include detailed per-source analysis

        Returns:
            0 on success, 1 on error
        """
        try:
            # Initialize with storage (Commandment #7: Check parameters)
            ctx = self.initialize_context(project, require_storage=True)

            # Retrieve all chunks
            chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not chunks:
                self._handle_empty_storage()
                return 0

            # Analyze chunks
            analysis = self._analyze_knowledge_base(chunks, detailed)

            # Display results
            self._display_audit_results(analysis, detailed)

            # Save to file if requested
            if output:
                self._save_audit_report(output, analysis, detailed)

            return 0

        except Exception as e:
            return self.handle_error(e, "Audit failed")

    def _handle_empty_storage(self) -> None:
        """Handle case where storage is empty."""
        self.print_warning("Knowledge base is empty")
        self.print_info(
            "Try:\n"
            "  1. Ingesting documents with 'ingestforge ingest'\n"
            "  2. Checking project configuration"
        )

    def _analyze_knowledge_base(self, chunks: list, detailed: bool) -> Dict[str, Any]:
        """Analyze knowledge base quality.

        Args:
            chunks: List of all chunks
            detailed: Whether to include detailed analysis

        Returns:
            Dictionary with analysis results
        """
        analysis = {}

        # Basic metadata analysis
        analysis["metadata"] = self.analyze_chunks_metadata(chunks)

        # Quality metrics
        analysis["quality"] = self._calculate_quality_metrics(chunks)

        # Coverage analysis
        analysis["coverage"] = self._analyze_coverage(chunks)

        # Per-source analysis if detailed
        if detailed:
            analysis["per_source"] = self._analyze_per_source(chunks)

        return analysis

    def _calculate_quality_metrics(self, chunks: list) -> Dict[str, Any]:
        """Calculate quality metrics for chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with quality metrics
        """
        if not chunks:
            return {"avg_quality": 0.0, "chunks_with_metadata": 0}

        chunks_with_metadata = 0
        chunks_with_citations = 0

        for chunk in chunks:
            # Check for metadata presence
            if self._has_metadata(chunk):
                chunks_with_metadata += 1

            # Check for citation presence
            if self._get_chunk_citation(chunk):
                chunks_with_citations += 1

        return {
            "chunks_with_metadata": chunks_with_metadata,
            "chunks_with_citations": chunks_with_citations,
            "metadata_coverage": chunks_with_metadata / len(chunks),
            "citation_coverage": chunks_with_citations / len(chunks),
        }

    def _has_metadata(self, chunk: Any) -> bool:
        """Check if chunk has metadata.

        Args:
            chunk: Chunk object or dict

        Returns:
            True if has meaningful metadata
        """
        if isinstance(chunk, dict):
            metadata = chunk.get("metadata", {})
            return bool(metadata and len(metadata) > 0)
        elif hasattr(chunk, "metadata"):
            metadata = chunk.metadata
            return metadata is not None
        else:
            return False

    def _analyze_coverage(self, chunks: list) -> Dict[str, Any]:
        """Analyze topic coverage.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary with coverage metrics
        """
        # Extract unique sources
        sources = self._extract_unique_sources(chunks)

        # Calculate size distribution
        sizes = [self._get_chunk_size(chunk) for chunk in chunks]
        sizes.sort()

        return {
            "unique_sources": len(sources),
            "source_list": sources[:10],  # Top 10 sources
            "min_chunk_size": min(sizes) if sizes else 0,
            "max_chunk_size": max(sizes) if sizes else 0,
            "median_chunk_size": sizes[len(sizes) // 2] if sizes else 0,
        }

    def _get_chunk_size(self, chunk: Any) -> int:
        """Get size of chunk text.

        Args:
            chunk: Chunk object or dict

        Returns:
            Size in characters
        """
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        elif hasattr(chunk, "text"):
            text = chunk.text
        else:
            text = str(chunk)

        return len(text)

    def _analyze_per_source(self, chunks: list) -> Dict[str, Dict[str, Any]]:
        """Analyze metrics per source.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary mapping source to metrics
        """
        per_source: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            source = self._get_chunk_source(chunk)

            if source not in per_source:
                per_source[source] = {
                    "chunk_count": 0,
                    "total_size": 0,
                    "has_citations": False,
                }

            per_source[source]["chunk_count"] += 1
            per_source[source]["total_size"] += self._get_chunk_size(chunk)

            if self._get_chunk_citation(chunk):
                per_source[source]["has_citations"] = True

        return per_source

    def _display_audit_results(self, analysis: Dict[str, Any], detailed: bool) -> None:
        """Display audit results.

        Args:
            analysis: Analysis results
            detailed: Whether to show detailed output
        """
        self.console.print()

        # Summary table
        self._display_summary_table(analysis)

        # Quality metrics
        self._display_quality_metrics(analysis["quality"])

        # Coverage info
        self._display_coverage_info(analysis["coverage"])

        # Per-source details if requested
        if detailed and "per_source" in analysis:
            self._display_per_source_details(analysis["per_source"])

    def _display_summary_table(self, analysis: Dict[str, Any]) -> None:
        """Display summary table.

        Args:
            analysis: Analysis results
        """
        metadata = analysis["metadata"]

        table = Table(title="Knowledge Base Summary", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", str(metadata["total_chunks"]))
        table.add_row("Unique Sources", str(metadata["unique_sources"]))
        table.add_row("Avg Chunk Size", f"{metadata['avg_chunk_size']:.0f} chars")

        self.console.print(table)
        self.console.print()

    def _display_quality_metrics(self, quality: Dict[str, Any]) -> None:
        """Display quality metrics.

        Args:
            quality: Quality metrics
        """
        table = Table(title="Quality Metrics", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Metadata Coverage", f"{quality['metadata_coverage']:.1%}")
        table.add_row("Citation Coverage", f"{quality['citation_coverage']:.1%}")

        self.console.print(table)
        self.console.print()

    def _display_coverage_info(self, coverage: Dict[str, Any]) -> None:
        """Display coverage information.

        Args:
            coverage: Coverage metrics
        """
        self.print_info(
            f"Source diversity: {coverage['unique_sources']} unique sources"
        )
        self.print_info(
            f"Chunk size range: {coverage['min_chunk_size']} - "
            f"{coverage['max_chunk_size']} chars"
        )

    def _display_per_source_details(
        self, per_source: Dict[str, Dict[str, Any]]
    ) -> None:
        """Display per-source details.

        Args:
            per_source: Per-source metrics
        """
        self.console.print()
        table = Table(title="Per-Source Analysis")
        table.add_column("Source", style="cyan")
        table.add_column("Chunks", style="green")
        table.add_column("Total Size", style="yellow")
        table.add_column("Citations", style="magenta")

        for source, metrics in sorted(per_source.items())[:20]:
            table.add_row(
                source[:50],  # Truncate long names
                str(metrics["chunk_count"]),
                f"{metrics['total_size']:,}",
                "Yes" if metrics["has_citations"] else "No",
            )

        self.console.print(table)

    def _save_audit_report(
        self, output: Path, analysis: Dict[str, Any], detailed: bool
    ) -> None:
        """Save audit report to file.

        Args:
            output: Output file path
            analysis: Analysis results
            detailed: Whether to include detailed info
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            content = self._format_audit_report(analysis, timestamp, detailed)

            output.write_text(content, encoding="utf-8")
            self.print_success(f"Audit report saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save report: {e}")

    def _format_audit_report(
        self, analysis: Dict[str, Any], timestamp: str, detailed: bool
    ) -> str:
        """Format audit report content.

        Args:
            analysis: Analysis results
            timestamp: Timestamp string
            detailed: Whether to include details

        Returns:
            Formatted report content
        """
        lines = [
            "# Knowledge Base Audit Report\n",
            f"Generated: {timestamp}\n",
            "Tool: IngestForge Research Tools\n\n",
            "---\n\n",
            "## Summary\n\n",
            f"- Total Chunks: {analysis['metadata']['total_chunks']}\n",
            f"- Unique Sources: {analysis['metadata']['unique_sources']}\n",
            f"- Average Chunk Size: {analysis['metadata']['avg_chunk_size']:.0f} chars\n\n",
            "## Quality Metrics\n\n",
            f"- Metadata Coverage: {analysis['quality']['metadata_coverage']:.1%}\n",
            f"- Citation Coverage: {analysis['quality']['citation_coverage']:.1%}\n\n",
        ]

        if detailed and "per_source" in analysis:
            lines.append("## Per-Source Analysis\n\n")
            for source, metrics in sorted(analysis["per_source"].items()):
                lines.append(f"### {source}\n\n")
                lines.append(f"- Chunks: {metrics['chunk_count']}\n")
                lines.append(f"- Total Size: {metrics['total_size']:,} chars\n")
                lines.append(
                    f"- Has Citations: {'Yes' if metrics['has_citations'] else 'No'}\n\n"
                )

        return "".join(lines)


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for audit report"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Include detailed per-source analysis"
    ),
) -> None:
    """Audit knowledge base quality and coverage.

    Analyzes the knowledge base to assess:
    - Overall statistics and coverage
    - Quality metrics (metadata, citations)
    - Source distribution
    - Potential gaps or issues

    Examples:
        # Basic audit
        ingestforge research audit

        # Detailed audit with report
        ingestforge research audit --detailed --output audit_report.md

        # Audit specific project
        ingestforge research audit -p /path/to/project
    """
    cmd = AuditCommand()
    exit_code = cmd.execute(project, output, detailed)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
