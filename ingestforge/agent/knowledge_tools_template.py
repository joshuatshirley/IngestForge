"""Synthesis & Report Engine for autonomous agents.

Generates structured reports from agent research results."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ingestforge.agent.react_engine import AgentResult, ReActStep
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_REPORT_SECTIONS = 20
MAX_SECTION_LENGTH = 10000
MAX_FINDINGS = 50
MAX_SOURCES = 100
MAX_TITLE_LENGTH = 200


class ReportFormat(Enum):
    """Output formats for reports."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    TEXT = "text"


class SectionType(Enum):
    """Types of report sections."""

    SUMMARY = "summary"
    FINDINGS = "findings"
    METHODOLOGY = "methodology"
    SOURCES = "sources"
    CONCLUSION = "conclusion"
    APPENDIX = "appendix"


@dataclass
class Finding:
    """Single finding from research."""

    content: str
    source: str = ""
    confidence: float = 1.0
    step_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content[:MAX_SECTION_LENGTH],
            "source": self.source,
            "confidence": self.confidence,
            "step_index": self.step_index,
        }


@dataclass
class ReportSection:
    """Section of a report."""

    section_type: SectionType
    title: str
    content: str
    subsections: list["ReportSection"] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate section on creation."""
        self.title = self.title[:MAX_TITLE_LENGTH]
        self.content = self.content[:MAX_SECTION_LENGTH]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.section_type.value,
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convert to markdown."""
        prefix = "#" * level
        lines = [f"{prefix} {self.title}", "", self.content, ""]

        for sub in self.subsections:
            lines.append(sub.to_markdown(level + 1))

        return "\n".join(lines)


@dataclass
class Report:
    """Complete research report."""

    title: str
    task: str
    sections: list[ReportSection] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    success: bool = True

    def __post_init__(self) -> None:
        """Validate report on creation."""
        self.title = self.title[:MAX_TITLE_LENGTH]
        self.sections = self.sections[:MAX_REPORT_SECTIONS]
        self.findings = self.findings[:MAX_FINDINGS]
        self.sources = self.sources[:MAX_SOURCES]

    @property
    def section_count(self) -> int:
        """Number of sections."""
        return len(self.sections)

    @property
    def finding_count(self) -> int:
        """Number of findings."""
        return len(self.findings)

    def add_section(self, section: ReportSection) -> bool:
        """Add a section to the report.

        Args:
            section: Section to add

        Returns:
            True if added
        """
        if len(self.sections) >= MAX_REPORT_SECTIONS:
            return False

        self.sections.append(section)
        return True

    def add_finding(self, finding: Finding) -> bool:
        """Add a finding to the report.

        Args:
            finding: Finding to add

        Returns:
            True if added
        """
        if len(self.findings) >= MAX_FINDINGS:
            return False

        self.findings.append(finding)
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "task": self.task,
            "sections": [s.to_dict() for s in self.sections],
            "findings": [f.to_dict() for f in self.findings],
            "sources": self.sources,
            "created_at": self.created_at.isoformat(),
            "success": self.success,
        }

    def to_markdown(self) -> str:
        """Convert to markdown format."""
        lines = [
            f"# {self.title}",
            "",
            f"*Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            f"**Task:** {self.task}",
            "",
        ]

        for section in self.sections:
            lines.append(section.to_markdown())

        return "\n".join(lines)

    def to_html(self) -> str:
        """Convert to HTML format."""
        sections_html = "\n".join(
            f"<section><h2>{s.title}</h2><p>{s.content}</p></section>"
            for s in self.sections
        )

        return f"""<!DOCTYPE html>
<html>
<head><title>{self.title}</title></head>
<body>
<h1>{self.title}</h1>
<p><em>Generated: {self.created_at.strftime('%Y-%m-%d %H:%M')}</em></p>
<p><strong>Task:</strong> {self.task}</p>
{sections_html}
</body>
</html>"""


class ReportSynthesizer:
    """Synthesizes research results into reports.

    Processes agent execution results and generates
    structured reports in various formats.
    """

    def __init__(self) -> None:
        """Initialize the synthesizer."""
        self._findings: list[Finding] = []
        self._sources: set[str] = set()

    def synthesize(
        self,
        result: AgentResult,
        title: Optional[str] = None,
    ) -> Report:
        """Synthesize agent result into report.

        Args:
            result: Agent execution result
            title: Report title (optional)

        Returns:
            Generated report
        """
        # Extract findings from steps
        self._extract_findings(result.steps)

        # Generate title if not provided
        if not title:
            title = self._generate_title(result)

        # Build report
        report = Report(
            title=title,
            task=result.final_answer,
            findings=self._findings[:MAX_FINDINGS],
            sources=list(self._sources)[:MAX_SOURCES],
            success=result.success,
        )

        # Add standard sections
        self._add_summary_section(report, result)
        self._add_findings_section(report)
        self._add_methodology_section(report, result)

        if self._sources:
            self._add_sources_section(report)

        self._add_conclusion_section(report, result)

        return report

    def _extract_findings(self, steps: list[ReActStep]) -> None:
        """Extract findings from agent steps.

        Args:
            steps: Agent execution steps
        """
        self._findings = []
        self._sources = set()

        for i, step in enumerate(steps):
            if not step.observation:
                continue

            # Skip error observations
            if step.observation.startswith("Error:"):
                continue

            finding = Finding(
                content=step.observation,
                source=step.action or "reasoning",
                step_index=i,
            )
            self._findings.append(finding)

            # Track tool as source
            if step.action:
                self._sources.add(step.action)

    def _generate_title(self, result: AgentResult) -> str:
        """Generate report title from result.

        Args:
            result: Agent result

        Returns:
            Generated title
        """
        answer = result.final_answer[:50]
        return f"Research Report: {answer}"

    def _add_summary_section(
        self,
        report: Report,
        result: AgentResult,
    ) -> None:
        """Add summary section to report.

        Args:
            report: Target report
            result: Agent result
        """
        content = f"""This report summarizes the findings from an automated research session.

**Status:** {"Completed Successfully" if result.success else "Incomplete"}
**Iterations:** {result.iterations}
**Findings:** {len(self._findings)}

{result.final_answer}"""

        section = ReportSection(
            section_type=SectionType.SUMMARY,
            title="Executive Summary",
            content=content,
        )
        report.add_section(section)

    def _add_findings_section(self, report: Report) -> None:
        """Add findings section to report.

        Args:
            report: Target report
        """
        if not self._findings:
            return

        lines = ["Key findings from the research:\n"]

        for i, finding in enumerate(self._findings[:10], 1):
            lines.append(f"{i}. {finding.content[:500]}")
            if finding.source:
                lines.append(f"   *Source: {finding.source}*\n")

        section = ReportSection(
            section_type=SectionType.FINDINGS,
            title="Key Findings",
            content="\n".join(lines),
        )
        report.add_section(section)

    def _add_methodology_section(
        self,
        report: Report,
        result: AgentResult,
    ) -> None:
        """Add methodology section to report.

        Args:
            report: Target report
            result: Agent result
        """
        tools_used = list(self._sources)
        tool_list = ", ".join(tools_used) if tools_used else "None"

        content = f"""This research was conducted using an autonomous ReAct agent.

**Tools Used:** {tool_list}
**Total Steps:** {len(result.steps)}
**Final State:** {result.state.value}"""

        section = ReportSection(
            section_type=SectionType.METHODOLOGY,
            title="Methodology",
            content=content,
        )
        report.add_section(section)

    def _add_sources_section(self, report: Report) -> None:
        """Add sources section to report.

        Args:
            report: Target report
        """
        lines = ["Sources consulted during research:\n"]

        for source in sorted(self._sources):
            lines.append(f"- {source}")

        section = ReportSection(
            section_type=SectionType.SOURCES,
            title="Sources",
            content="\n".join(lines),
        )
        report.add_section(section)

    def _add_conclusion_section(
        self,
        report: Report,
        result: AgentResult,
    ) -> None:
        """Add conclusion section to report.

        Args:
            report: Target report
            result: Agent result
        """
        if result.success:
            content = """The research task was completed successfully.
All relevant information has been gathered and synthesized above."""
        else:
            content = f"""The research task could not be fully completed.
Final state: {result.state.value}

Please review the findings and consider additional research if needed."""

        section = ReportSection(
            section_type=SectionType.CONCLUSION,
            title="Conclusion",
            content=content,
        )
        report.add_section(section)


class ReportExporter:
    """Exports reports to various formats."""

    def export(
        self,
        report: Report,
        output_path: Path,
        format: ReportFormat = ReportFormat.MARKDOWN,
    ) -> bool:
        """Export report to file.

        Args:
            report: Report to export
            output_path: Output file path
            format: Output format

        Returns:
            True if successful
        """
        try:
            content = self._format_report(report, format)
            output_path.write_text(content, encoding="utf-8")
            logger.info(f"Report exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export failed: {e}")
            return False

    def _format_report(
        self,
        report: Report,
        format: ReportFormat,
    ) -> str:
        """Format report for output.

        Args:
            report: Report to format
            format: Output format

        Returns:
            Formatted string
        """
        if format == ReportFormat.MARKDOWN:
            return report.to_markdown()

        if format == ReportFormat.HTML:
            return report.to_html()

        if format == ReportFormat.JSON:
            import json

            return json.dumps(report.to_dict(), indent=2)

        # Default to text
        return f"{report.title}\n\n{report.to_markdown()}"


def create_synthesizer() -> ReportSynthesizer:
    """Factory function to create synthesizer.

    Returns:
        New synthesizer
    """
    return ReportSynthesizer()


def synthesize_report(
    result: AgentResult,
    title: Optional[str] = None,
) -> Report:
    """Convenience function to synthesize a report.

    Args:
        result: Agent execution result
        title: Optional report title

    Returns:
        Generated report
    """
    synthesizer = create_synthesizer()
    return synthesizer.synthesize(result, title)
