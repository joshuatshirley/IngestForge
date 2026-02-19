"""Verify command - Verify source citations and references.

Verifies that citations are properly formatted and sources are traceable.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Any
import typer
from rich.table import Table
from rich.panel import Panel

from ingestforge.cli.research.base import ResearchCommand


class VerifyCommand(ResearchCommand):
    """Verify source citations and references."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        show_missing: bool = False,
    ) -> int:
        """Verify source citations and references.

        Args:
            project: Project directory
            output: Output file for verification report (optional)
            show_missing: Show chunks missing citations

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

            # Verify citations
            verification = self._verify_citations(chunks)

            # Display results
            self._display_verification_results(verification, show_missing)

            # Save to file if requested
            if output:
                self._save_verification_report(output, verification, show_missing)

            return 0

        except Exception as e:
            return self.handle_error(e, "Verification failed")

    def _handle_empty_storage(self) -> None:
        """Handle case where storage is empty."""
        self.print_warning("Knowledge base is empty")
        self.print_info(
            "Try:\n"
            "  1. Ingesting documents with 'ingestforge ingest'\n"
            "  2. Checking project configuration"
        )

    def _verify_citations(self, chunks: list) -> Dict[str, Any]:
        """Verify citations in chunks.

        Args:
            chunks: List of all chunks

        Returns:
            Dictionary with verification results
        """
        verification = {
            "total_chunks": len(chunks),
            "chunks_with_citations": 0,
            "chunks_without_citations": 0,
            "valid_citations": 0,
            "invalid_citations": 0,
            "citations_by_source": {},
            "missing_citation_chunks": [],
        }

        for idx, chunk in enumerate(chunks):
            citation = self._get_chunk_citation(chunk)
            source = self._get_chunk_source(chunk)

            # Track citations by source
            if source not in verification["citations_by_source"]:
                verification["citations_by_source"][source] = {
                    "total": 0,
                    "with_citation": 0,
                    "citations": [],
                }

            verification["citations_by_source"][source]["total"] += 1

            if citation:
                verification["chunks_with_citations"] += 1
                verification["citations_by_source"][source]["with_citation"] += 1

                # Validate citation format
                if self.validate_citation_format(citation):
                    verification["valid_citations"] += 1
                    verification["citations_by_source"][source]["citations"].append(
                        citation
                    )
                else:
                    verification["invalid_citations"] += 1
            else:
                verification["chunks_without_citations"] += 1
                verification["missing_citation_chunks"].append(
                    {
                        "index": idx,
                        "source": source,
                        "preview": self._get_chunk_preview(chunk),
                    }
                )

        return verification

    def _get_chunk_preview(self, chunk: Any, max_length: int = 80) -> str:
        """Get preview of chunk text.

        Args:
            chunk: Chunk object or dict
            max_length: Maximum preview length

        Returns:
            Preview string
        """
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
        elif hasattr(chunk, "text"):
            text = chunk.text
        else:
            text = str(chunk)

        text = text.strip()

        if len(text) > max_length:
            return text[:max_length] + "..."
        return text

    def _display_verification_results(
        self, verification: Dict[str, Any], show_missing: bool
    ) -> None:
        """Display verification results.

        Args:
            verification: Verification results
            show_missing: Whether to show missing citations
        """
        self.console.print()

        # Summary statistics
        self._display_summary_statistics(verification)

        # Citation coverage by source
        self._display_citation_coverage(verification["citations_by_source"])

        # Missing citations if requested
        if show_missing and verification["missing_citation_chunks"]:
            self._display_missing_citations(verification["missing_citation_chunks"])

        # Overall assessment
        self._display_assessment(verification)

    def _display_summary_statistics(self, verification: Dict[str, Any]) -> None:
        """Display summary statistics.

        Args:
            verification: Verification results
        """
        table = Table(title="Citation Verification Summary", show_header=False)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Chunks", str(verification["total_chunks"]))
        table.add_row("With Citations", str(verification["chunks_with_citations"]))
        table.add_row(
            "Without Citations", str(verification["chunks_without_citations"])
        )
        table.add_row("Valid Citations", str(verification["valid_citations"]))
        table.add_row("Invalid Citations", str(verification["invalid_citations"]))

        coverage = (
            verification["chunks_with_citations"] / verification["total_chunks"]
            if verification["total_chunks"] > 0
            else 0
        )
        table.add_row("Coverage", f"{coverage:.1%}")

        self.console.print(table)
        self.console.print()

    def _display_citation_coverage(
        self, citations_by_source: Dict[str, Dict[str, Any]]
    ) -> None:
        """Display citation coverage by source.

        Args:
            citations_by_source: Citation data by source
        """
        if not citations_by_source:
            return

        table = Table(title="Citation Coverage by Source")
        table.add_column("Source", style="cyan")
        table.add_column("Total Chunks", style="green")
        table.add_column("With Citations", style="yellow")
        table.add_column("Coverage", style="magenta")

        for source, data in sorted(citations_by_source.items())[:20]:
            coverage = data["with_citation"] / data["total"] if data["total"] > 0 else 0
            table.add_row(
                source[:40],  # Truncate long source names
                str(data["total"]),
                str(data["with_citation"]),
                f"{coverage:.1%}",
            )

        self.console.print(table)
        self.console.print()

    def _display_missing_citations(self, missing_chunks: List[Dict[str, Any]]) -> None:
        """Display chunks missing citations.

        Args:
            missing_chunks: List of chunks without citations
        """
        # Limit display to first 20
        display_chunks = missing_chunks[:20]

        table = Table(title="Chunks Without Citations (First 20)")
        table.add_column("Index", style="cyan")
        table.add_column("Source", style="yellow")
        table.add_column("Preview", style="dim")

        for chunk_info in display_chunks:
            table.add_row(
                str(chunk_info["index"]),
                chunk_info["source"][:30],
                chunk_info["preview"][:60],
            )

        self.console.print(table)
        self.console.print()

        if len(missing_chunks) > 20:
            self.print_info(
                f"Showing 20 of {len(missing_chunks)} chunks without citations"
            )

    def _display_assessment(self, verification: Dict[str, Any]) -> None:
        """Display overall assessment.

        Args:
            verification: Verification results
        """
        coverage = (
            verification["chunks_with_citations"] / verification["total_chunks"]
            if verification["total_chunks"] > 0
            else 0
        )

        if coverage >= 0.9:
            status = "Excellent"
            style = "green"
        elif coverage >= 0.7:
            status = "Good"
            style = "yellow"
        elif coverage >= 0.5:
            status = "Fair"
            style = "yellow"
        else:
            status = "Poor"
            style = "red"

        self.console.print(
            Panel(
                f"Citation coverage: [{style}]{status}[/{style}] ({coverage:.1%})",
                title="Overall Assessment",
                border_style=style,
            )
        )

    def _save_verification_report(
        self, output: Path, verification: Dict[str, Any], show_missing: bool
    ) -> None:
        """Save verification report to file.

        Args:
            output: Output file path
            verification: Verification results
            show_missing: Whether to include missing citations
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            content = self._format_verification_report(
                verification, timestamp, show_missing
            )

            output.write_text(content, encoding="utf-8")
            self.print_success(f"Verification report saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save report: {e}")

    def _format_verification_report(
        self, verification: Dict[str, Any], timestamp: str, show_missing: bool
    ) -> str:
        """Format verification report content.

        Args:
            verification: Verification results
            timestamp: Timestamp string
            show_missing: Whether to include missing citations

        Returns:
            Formatted report content
        """
        coverage = (
            verification["chunks_with_citations"] / verification["total_chunks"]
            if verification["total_chunks"] > 0
            else 0
        )

        lines = [
            "# Citation Verification Report\n\n",
            f"Generated: {timestamp}\n",
            "Tool: IngestForge Research Tools\n\n",
            "---\n\n",
            "## Summary\n\n",
            f"- Total Chunks: {verification['total_chunks']}\n",
            f"- With Citations: {verification['chunks_with_citations']}\n",
            f"- Without Citations: {verification['chunks_without_citations']}\n",
            f"- Valid Citations: {verification['valid_citations']}\n",
            f"- Invalid Citations: {verification['invalid_citations']}\n",
            f"- Coverage: {coverage:.1%}\n\n",
            "## Coverage by Source\n\n",
        ]

        for source, data in sorted(verification["citations_by_source"].items()):
            src_coverage = (
                data["with_citation"] / data["total"] if data["total"] > 0 else 0
            )
            lines.append(f"### {source}\n\n")
            lines.append(f"- Total Chunks: {data['total']}\n")
            lines.append(f"- With Citations: {data['with_citation']}\n")
            lines.append(f"- Coverage: {src_coverage:.1%}\n\n")

        if show_missing and verification["missing_citation_chunks"]:
            lines.append("## Chunks Without Citations\n\n")
            for chunk_info in verification["missing_citation_chunks"]:
                lines.append(
                    f"- Index {chunk_info['index']} ({chunk_info['source']}): "
                    f"{chunk_info['preview']}\n"
                )
            lines.append("\n")

        return "".join(lines)


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for verification report"
    ),
    show_missing: bool = typer.Option(
        False, "--show-missing", "-m", help="Show chunks without citations"
    ),
) -> None:
    """Verify source citations and references.

    Checks that sources are properly cited and traceable:
    - Citation presence and validity
    - Coverage by source
    - Missing or invalid citations

    Examples:
        # Basic verification
        ingestforge research verify

        # Show missing citations
        ingestforge research verify --show-missing

        # Generate detailed report
        ingestforge research verify --show-missing --output verify_report.md

        # Verify specific project
        ingestforge research verify -p /path/to/project
    """
    cmd = VerifyCommand()
    exit_code = cmd.execute(project, output, show_missing)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
