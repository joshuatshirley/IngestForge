"""Tests for synthesis & report engine.

Tests report generation from agent results."""

from __future__ import annotations

from pathlib import Path


from ingestforge.agent.react_engine import (
    AgentState,
    AgentResult,
    ReActStep,
    ToolOutput,
    ToolResult,
)
from ingestforge.agent.synthesis import (
    ReportFormat,
    SectionType,
    Finding,
    ReportSection,
    Report,
    ReportSynthesizer,
    ReportExporter,
    create_synthesizer,
    synthesize_report,
    MAX_REPORT_SECTIONS,
)

# Test fixtures


def make_agent_result(
    success: bool = True,
    steps: int = 3,
) -> AgentResult:
    """Create a test agent result."""
    step_list = []

    for i in range(steps):
        step = ReActStep(
            iteration=i,
            thought=f"Thinking step {i}",
            action="search" if i < steps - 1 else None,
            action_input={"query": f"query {i}"},
            observation=f"Found data {i}",
            tool_result=ToolOutput(
                status=ToolResult.SUCCESS,
                data=f"result {i}",
            ),
        )
        step_list.append(step)

    return AgentResult(
        success=success,
        final_answer="Research complete",
        steps=step_list,
        iterations=steps,
        state=AgentState.COMPLETE if success else AgentState.FAILED,
    )


# ReportFormat tests


class TestReportFormat:
    """Tests for ReportFormat enum."""

    def test_formats_defined(self) -> None:
        """Test all formats are defined."""
        formats = [f.value for f in ReportFormat]

        assert "markdown" in formats
        assert "html" in formats
        assert "json" in formats

    def test_format_count(self) -> None:
        """Test correct number of formats."""
        assert len(ReportFormat) == 4


# SectionType tests


class TestSectionType:
    """Tests for SectionType enum."""

    def test_types_defined(self) -> None:
        """Test all types are defined."""
        types = [t.value for t in SectionType]

        assert "summary" in types
        assert "findings" in types
        assert "conclusion" in types


# Finding tests


class TestFinding:
    """Tests for Finding dataclass."""

    def test_finding_creation(self) -> None:
        """Test creating a finding."""
        finding = Finding(
            content="Important discovery",
            source="search",
            confidence=0.9,
        )

        assert finding.content == "Important discovery"
        assert finding.confidence == 0.9

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        finding = Finding(
            content="Data point",
            source="analyze",
            step_index=2,
        )

        d = finding.to_dict()

        assert d["content"] == "Data point"
        assert d["step_index"] == 2


# ReportSection tests


class TestReportSection:
    """Tests for ReportSection dataclass."""

    def test_section_creation(self) -> None:
        """Test creating a section."""
        section = ReportSection(
            section_type=SectionType.SUMMARY,
            title="Summary",
            content="This is the summary.",
        )

        assert section.title == "Summary"
        assert section.section_type == SectionType.SUMMARY

    def test_section_with_subsections(self) -> None:
        """Test section with subsections."""
        sub = ReportSection(
            section_type=SectionType.FINDINGS,
            title="Sub Finding",
            content="Detail",
        )
        section = ReportSection(
            section_type=SectionType.FINDINGS,
            title="Findings",
            content="Main findings",
            subsections=[sub],
        )

        assert len(section.subsections) == 1

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        section = ReportSection(
            section_type=SectionType.METHODOLOGY,
            title="Method",
            content="Process description",
        )

        d = section.to_dict()

        assert d["type"] == "methodology"
        assert d["title"] == "Method"

    def test_to_markdown(self) -> None:
        """Test converting to markdown."""
        section = ReportSection(
            section_type=SectionType.CONCLUSION,
            title="Conclusion",
            content="Final thoughts",
        )

        md = section.to_markdown()

        assert "## Conclusion" in md
        assert "Final thoughts" in md


# Report tests


class TestReport:
    """Tests for Report dataclass."""

    def test_report_creation(self) -> None:
        """Test creating a report."""
        report = Report(
            title="Test Report",
            task="Research topic",
        )

        assert report.title == "Test Report"
        assert report.section_count == 0

    def test_add_section(self) -> None:
        """Test adding a section."""
        report = Report(title="Test", task="Task")
        section = ReportSection(
            section_type=SectionType.SUMMARY,
            title="Summary",
            content="Content",
        )

        result = report.add_section(section)

        assert result is True
        assert report.section_count == 1

    def test_add_finding(self) -> None:
        """Test adding a finding."""
        report = Report(title="Test", task="Task")
        finding = Finding(content="Discovery", source="tool")

        result = report.add_finding(finding)

        assert result is True
        assert report.finding_count == 1

    def test_max_sections_enforced(self) -> None:
        """Test max sections limit."""
        report = Report(title="Test", task="Task")

        for i in range(MAX_REPORT_SECTIONS + 5):
            section = ReportSection(
                section_type=SectionType.APPENDIX,
                title=f"Section {i}",
                content="Content",
            )
            report.add_section(section)

        assert report.section_count == MAX_REPORT_SECTIONS

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        report = Report(
            title="Export Test",
            task="Do research",
            success=True,
        )

        d = report.to_dict()

        assert d["title"] == "Export Test"
        assert d["success"] is True
        assert "created_at" in d

    def test_to_markdown(self) -> None:
        """Test converting to markdown."""
        report = Report(title="MD Test", task="Research")
        report.add_section(
            ReportSection(
                section_type=SectionType.SUMMARY,
                title="Summary",
                content="Overview",
            )
        )

        md = report.to_markdown()

        assert "# MD Test" in md
        assert "## Summary" in md

    def test_to_html(self) -> None:
        """Test converting to HTML."""
        report = Report(title="HTML Test", task="Research")

        html = report.to_html()

        assert "<title>HTML Test</title>" in html
        assert "<h1>HTML Test</h1>" in html


# ReportSynthesizer tests


class TestReportSynthesizer:
    """Tests for ReportSynthesizer class."""

    def test_synthesizer_creation(self) -> None:
        """Test creating a synthesizer."""
        synth = ReportSynthesizer()

        assert synth is not None

    def test_synthesize_success(self) -> None:
        """Test synthesizing successful result."""
        synth = ReportSynthesizer()
        result = make_agent_result(success=True, steps=3)

        report = synth.synthesize(result)

        assert report.success is True
        assert report.section_count > 0

    def test_synthesize_failure(self) -> None:
        """Test synthesizing failed result."""
        synth = ReportSynthesizer()
        result = make_agent_result(success=False, steps=2)

        report = synth.synthesize(result)

        assert report.success is False

    def test_synthesize_with_title(self) -> None:
        """Test synthesizing with custom title."""
        synth = ReportSynthesizer()
        result = make_agent_result()

        report = synth.synthesize(result, title="Custom Title")

        assert report.title == "Custom Title"

    def test_synthesize_extracts_findings(self) -> None:
        """Test that findings are extracted."""
        synth = ReportSynthesizer()
        result = make_agent_result(steps=5)

        report = synth.synthesize(result)

        assert report.finding_count > 0

    def test_synthesize_tracks_sources(self) -> None:
        """Test that sources are tracked."""
        synth = ReportSynthesizer()
        result = make_agent_result(steps=4)

        report = synth.synthesize(result)

        assert len(report.sources) > 0


class TestReportSections:
    """Tests for report section generation."""

    def test_has_summary_section(self) -> None:
        """Test that summary section is added."""
        synth = ReportSynthesizer()
        result = make_agent_result()

        report = synth.synthesize(result)

        types = [s.section_type for s in report.sections]
        assert SectionType.SUMMARY in types

    def test_has_findings_section(self) -> None:
        """Test that findings section is added."""
        synth = ReportSynthesizer()
        result = make_agent_result(steps=3)

        report = synth.synthesize(result)

        types = [s.section_type for s in report.sections]
        assert SectionType.FINDINGS in types

    def test_has_methodology_section(self) -> None:
        """Test that methodology section is added."""
        synth = ReportSynthesizer()
        result = make_agent_result()

        report = synth.synthesize(result)

        types = [s.section_type for s in report.sections]
        assert SectionType.METHODOLOGY in types

    def test_has_conclusion_section(self) -> None:
        """Test that conclusion section is added."""
        synth = ReportSynthesizer()
        result = make_agent_result()

        report = synth.synthesize(result)

        types = [s.section_type for s in report.sections]
        assert SectionType.CONCLUSION in types


# ReportExporter tests


class TestReportExporter:
    """Tests for ReportExporter class."""

    def test_export_markdown(self, tmp_path: Path) -> None:
        """Test exporting to markdown."""
        exporter = ReportExporter()
        report = Report(title="Export Test", task="Research")
        output = tmp_path / "report.md"

        result = exporter.export(report, output, ReportFormat.MARKDOWN)

        assert result is True
        assert output.exists()
        content = output.read_text()
        assert "# Export Test" in content

    def test_export_html(self, tmp_path: Path) -> None:
        """Test exporting to HTML."""
        exporter = ReportExporter()
        report = Report(title="HTML Export", task="Research")
        output = tmp_path / "report.html"

        result = exporter.export(report, output, ReportFormat.HTML)

        assert result is True
        content = output.read_text()
        assert "<html>" in content

    def test_export_json(self, tmp_path: Path) -> None:
        """Test exporting to JSON."""
        exporter = ReportExporter()
        report = Report(title="JSON Export", task="Research")
        output = tmp_path / "report.json"

        result = exporter.export(report, output, ReportFormat.JSON)

        assert result is True
        content = output.read_text()
        assert '"title": "JSON Export"' in content

    def test_export_text(self, tmp_path: Path) -> None:
        """Test exporting to text."""
        exporter = ReportExporter()
        report = Report(title="Text Export", task="Research")
        output = tmp_path / "report.txt"

        result = exporter.export(report, output, ReportFormat.TEXT)

        assert result is True
        assert output.exists()


# Factory function tests


class TestCreateSynthesizer:
    """Tests for create_synthesizer factory."""

    def test_create(self) -> None:
        """Test creating synthesizer."""
        synth = create_synthesizer()

        assert isinstance(synth, ReportSynthesizer)


class TestSynthesizeReport:
    """Tests for synthesize_report function."""

    def test_synthesize(self) -> None:
        """Test convenience function."""
        result = make_agent_result()

        report = synthesize_report(result)

        assert report is not None
        assert report.section_count > 0

    def test_synthesize_with_title(self) -> None:
        """Test with custom title."""
        result = make_agent_result()

        report = synthesize_report(result, title="My Report")

        assert report.title == "My Report"


# Edge case tests


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_result(self) -> None:
        """Test synthesizing empty result."""
        result = AgentResult(
            success=True,
            final_answer="Done",
            steps=[],
            iterations=0,
            state=AgentState.COMPLETE,
        )
        synth = ReportSynthesizer()

        report = synth.synthesize(result)

        assert report is not None
        assert report.finding_count == 0

    def test_error_observations(self) -> None:
        """Test that error observations are skipped."""
        step = ReActStep(
            iteration=0,
            thought="Try",
            action="broken",
            observation="Error: Something failed",
        )
        result = AgentResult(
            success=False,
            final_answer="Failed",
            steps=[step],
            iterations=1,
            state=AgentState.FAILED,
        )
        synth = ReportSynthesizer()

        report = synth.synthesize(result)

        # Error observations should not become findings
        assert report.finding_count == 0
