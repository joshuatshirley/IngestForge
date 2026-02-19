"""Plagiarism Reporter CLI for similarity findings.

Displays visual highlights of overlapping passages and
generates plagiarism check reports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ingestforge.analysis.similarity_window import (
    ComparisonResult,
    SimilarityMatch,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_DISPLAY_MATCHES = 20
MAX_HIGHLIGHT_LENGTH = 200
MAX_CONTEXT_LENGTH = 100


class SeverityLevel:
    """Severity levels for plagiarism findings."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ReportConfig:
    """Configuration for plagiarism reports."""

    show_highlights: bool = True
    show_context: bool = True
    max_matches: int = MAX_DISPLAY_MATCHES
    min_similarity: float = 0.75


@dataclass
class ReportSummary:
    """Summary of plagiarism check results."""

    total_matches: int = 0
    high_similarity_count: int = 0
    unique_sources: int = 0
    overlap_percentage: float = 0.0
    severity: str = SeverityLevel.LOW
    recommendation: str = ""


class PlagiarismReporter:
    """Generates plagiarism check reports with visual highlights.

    Displays similarity findings in a formatted report with
    highlighted overlapping passages.
    """

    def __init__(
        self,
        config: Optional[ReportConfig] = None,
    ) -> None:
        """Initialize reporter.

        Args:
            config: Report configuration
        """
        self.config = config or ReportConfig()
        self.console = Console()

    def generate_report(
        self,
        result: ComparisonResult,
        source_text: str = "",
    ) -> ReportSummary:
        """Generate and display plagiarism report.

        Args:
            result: Comparison result from window comparator
            source_text: Original source text (for highlighting)

        Returns:
            ReportSummary
        """
        # Generate summary
        summary = self._create_summary(result)

        # Display header
        self._display_header(summary)

        # Display matches
        if result.matches and self.config.show_highlights:
            self._display_matches(result.matches)

        # Display recommendation
        self._display_recommendation(summary)

        return summary

    def _create_summary(self, result: ComparisonResult) -> ReportSummary:
        """Create report summary from results.

        Args:
            result: Comparison result

        Returns:
            ReportSummary
        """
        # Count unique sources
        sources = set(m.corpus_source for m in result.matches)

        # Calculate severity
        overlap = result.overlap_ratio
        severity = self._determine_severity(overlap, result.highest_similarity)

        # Generate recommendation
        recommendation = self._generate_recommendation(severity, overlap)

        return ReportSummary(
            total_matches=len(result.matches),
            high_similarity_count=result.flagged_count,
            unique_sources=len(sources),
            overlap_percentage=overlap * 100,
            severity=severity,
            recommendation=recommendation,
        )

    def _determine_severity(self, overlap: float, highest_sim: float) -> str:
        """Determine severity level.

        Args:
            overlap: Overlap ratio
            highest_sim: Highest similarity score

        Returns:
            Severity level string
        """
        if highest_sim >= 0.95 or overlap >= 0.5:
            return SeverityLevel.CRITICAL

        if highest_sim >= 0.85 or overlap >= 0.3:
            return SeverityLevel.HIGH

        if highest_sim >= 0.75 or overlap >= 0.15:
            return SeverityLevel.MODERATE

        return SeverityLevel.LOW

    def _generate_recommendation(self, severity: str, overlap: float) -> str:
        """Generate recommendation based on severity.

        Args:
            severity: Severity level
            overlap: Overlap ratio

        Returns:
            Recommendation text
        """
        if severity == SeverityLevel.CRITICAL:
            return (
                "Critical similarity detected. Review highlighted passages "
                "and ensure proper citations or significant rewrites."
            )

        if severity == SeverityLevel.HIGH:
            return (
                "High similarity in several passages. Consider paraphrasing "
                "and adding citations to matching content."
            )

        if severity == SeverityLevel.MODERATE:
            return (
                "Moderate similarity found. Review flagged sections and "
                "ensure ideas are properly attributed."
            )

        return "Low similarity detected. Document appears to be original."

    def _display_header(self, summary: ReportSummary) -> None:
        """Display report header panel.

        Args:
            summary: Report summary
        """
        # Choose color based on severity
        color = self._severity_color(summary.severity)

        # Build header content
        lines = [
            f"[bold]Total Matches:[/bold] {summary.total_matches}",
            f"[bold]High Similarity:[/bold] {summary.high_similarity_count}",
            f"[bold]Unique Sources:[/bold] {summary.unique_sources}",
            f"[bold]Overlap:[/bold] {summary.overlap_percentage:.1f}%",
            f"[bold]Severity:[/bold] [{color}]{summary.severity.upper()}[/{color}]",
        ]

        panel = Panel(
            "\n".join(lines),
            title="[bold]Plagiarism Check Report[/bold]",
            border_style=color,
        )
        self.console.print(panel)

    def _display_matches(self, matches: List[SimilarityMatch]) -> None:
        """Display match table with highlights.

        Args:
            matches: List of similarity matches
        """
        # Limit matches
        display_matches = matches[: self.config.max_matches]

        # Create table
        table = Table(title="Similar Passages Found")
        table.add_column("#", style="dim", width=4)
        table.add_column("Similarity", justify="center", width=12)
        table.add_column("Source", style="cyan", width=25)
        table.add_column("Matched Text", width=50)

        for i, match in enumerate(display_matches, 1):
            # Format similarity with color
            sim_color = self._similarity_color(match.similarity_score)
            sim_text = f"[{sim_color}]{match.similarity_score:.1%}[/{sim_color}]"

            # Truncate texts
            source = self._truncate(match.corpus_source, 25)
            text = self._truncate(match.source_window.text, 50)

            table.add_row(str(i), sim_text, source, text)

        self.console.print(table)

        # Show remaining count if any
        remaining = len(matches) - len(display_matches)
        if remaining > 0:
            self.console.print(f"[dim]... and {remaining} more matches[/dim]")

    def _display_recommendation(self, summary: ReportSummary) -> None:
        """Display recommendation panel.

        Args:
            summary: Report summary
        """
        color = self._severity_color(summary.severity)

        panel = Panel(
            summary.recommendation,
            title="[bold]Recommendation[/bold]",
            border_style=color,
        )
        self.console.print(panel)

    def _severity_color(self, severity: str) -> str:
        """Get color for severity level.

        Args:
            severity: Severity level

        Returns:
            Rich color string
        """
        colors = {
            SeverityLevel.LOW: "green",
            SeverityLevel.MODERATE: "yellow",
            SeverityLevel.HIGH: "orange3",
            SeverityLevel.CRITICAL: "red",
        }
        return colors.get(severity, "white")

    def _similarity_color(self, score: float) -> str:
        """Get color for similarity score.

        Args:
            score: Similarity score

        Returns:
            Rich color string
        """
        if score >= 0.95:
            return "red"
        if score >= 0.85:
            return "orange3"
        if score >= 0.75:
            return "yellow"
        return "green"

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncate text with ellipsis.

        Args:
            text: Text to truncate
            max_length: Maximum length

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + "..."

    def highlight_passage(
        self,
        text: str,
        matches: List[SimilarityMatch],
    ) -> Text:
        """Create highlighted text with matches marked.

        Args:
            text: Source text
            matches: Matches to highlight

        Returns:
            Rich Text object with highlights
        """
        result = Text(text)

        # Sort matches by position (reverse to apply from end)
        sorted_matches = sorted(
            matches,
            key=lambda m: m.source_window.start_pos,
            reverse=True,
        )

        for match in sorted_matches:
            start = match.source_window.start_pos
            end = match.source_window.end_pos

            # Validate positions
            if start < 0 or end > len(text):
                continue

            # Apply highlight style
            result.stylize("bold yellow on dark_red", start, end)

        return result


def create_reporter(
    show_highlights: bool = True,
    max_matches: int = MAX_DISPLAY_MATCHES,
) -> PlagiarismReporter:
    """Factory function to create reporter.

    Args:
        show_highlights: Show highlighted passages
        max_matches: Maximum matches to display

    Returns:
        Configured reporter
    """
    config = ReportConfig(
        show_highlights=show_highlights,
        max_matches=max_matches,
    )
    return PlagiarismReporter(config=config)


def print_plagiarism_report(
    result: ComparisonResult,
    source_text: str = "",
) -> ReportSummary:
    """Convenience function to print plagiarism report.

    Args:
        result: Comparison result
        source_text: Original source text

    Returns:
        ReportSummary
    """
    reporter = create_reporter()
    return reporter.generate_report(result, source_text)
