"""Tests for plagiarism reporter CLI.

Tests plagiarism report generation and display."""

from __future__ import annotations


from ingestforge.analysis.similarity_window import (
    ComparisonResult,
    SimilarityMatch,
    TextWindow,
)
from ingestforge.cli.commands.plagiarism_report import (
    PlagiarismReporter,
    ReportConfig,
    ReportSummary,
    SeverityLevel,
    create_reporter,
    print_plagiarism_report,
    MAX_DISPLAY_MATCHES,
)

# SeverityLevel tests


class TestSeverityLevel:
    """Tests for SeverityLevel class."""

    def test_severity_levels_defined(self) -> None:
        """Test all severity levels are defined."""
        assert SeverityLevel.LOW == "low"
        assert SeverityLevel.MODERATE == "moderate"
        assert SeverityLevel.HIGH == "high"
        assert SeverityLevel.CRITICAL == "critical"


# ReportConfig tests


class TestReportConfig:
    """Tests for ReportConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = ReportConfig()

        assert config.show_highlights is True
        assert config.max_matches == MAX_DISPLAY_MATCHES

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = ReportConfig(
            show_highlights=False,
            max_matches=10,
            min_similarity=0.8,
        )

        assert config.show_highlights is False
        assert config.max_matches == 10


# ReportSummary tests


class TestReportSummary:
    """Tests for ReportSummary dataclass."""

    def test_default_summary(self) -> None:
        """Test default summary values."""
        summary = ReportSummary()

        assert summary.total_matches == 0
        assert summary.severity == SeverityLevel.LOW

    def test_custom_summary(self) -> None:
        """Test summary with custom values."""
        summary = ReportSummary(
            total_matches=5,
            high_similarity_count=2,
            unique_sources=3,
            overlap_percentage=25.5,
            severity=SeverityLevel.HIGH,
            recommendation="Review content",
        )

        assert summary.total_matches == 5
        assert summary.severity == SeverityLevel.HIGH


# PlagiarismReporter tests


class TestPlagiarismReporter:
    """Tests for PlagiarismReporter class."""

    def test_reporter_creation(self) -> None:
        """Test creating reporter."""
        reporter = PlagiarismReporter()

        assert reporter.config is not None
        assert reporter.console is not None

    def test_reporter_with_config(self) -> None:
        """Test reporter with custom config."""
        config = ReportConfig(show_highlights=False)
        reporter = PlagiarismReporter(config=config)

        assert reporter.config.show_highlights is False


class TestSeverityDetermination:
    """Tests for severity determination."""

    def test_low_severity(self) -> None:
        """Test low severity determination."""
        reporter = PlagiarismReporter()

        severity = reporter._determine_severity(0.1, 0.7)

        assert severity == SeverityLevel.LOW

    def test_moderate_severity(self) -> None:
        """Test moderate severity determination."""
        reporter = PlagiarismReporter()

        severity = reporter._determine_severity(0.2, 0.8)

        assert severity == SeverityLevel.MODERATE

    def test_high_severity(self) -> None:
        """Test high severity determination."""
        reporter = PlagiarismReporter()

        severity = reporter._determine_severity(0.35, 0.88)

        assert severity == SeverityLevel.HIGH

    def test_critical_severity(self) -> None:
        """Test critical severity determination."""
        reporter = PlagiarismReporter()

        severity = reporter._determine_severity(0.6, 0.96)

        assert severity == SeverityLevel.CRITICAL


class TestRecommendationGeneration:
    """Tests for recommendation generation."""

    def test_low_recommendation(self) -> None:
        """Test low severity recommendation."""
        reporter = PlagiarismReporter()

        rec = reporter._generate_recommendation(SeverityLevel.LOW, 0.1)

        assert "original" in rec.lower()

    def test_critical_recommendation(self) -> None:
        """Test critical severity recommendation."""
        reporter = PlagiarismReporter()

        rec = reporter._generate_recommendation(SeverityLevel.CRITICAL, 0.6)

        assert "critical" in rec.lower() or "review" in rec.lower()


class TestSummaryCreation:
    """Tests for summary creation."""

    def test_create_summary_empty(self) -> None:
        """Test creating summary from empty result."""
        reporter = PlagiarismReporter()
        result = ComparisonResult()

        summary = reporter._create_summary(result)

        assert summary.total_matches == 0
        assert summary.severity == SeverityLevel.LOW

    def test_create_summary_with_matches(self) -> None:
        """Test creating summary from result with matches."""
        reporter = PlagiarismReporter()

        window = TextWindow("test text", 0, 9, 0)
        match = SimilarityMatch(
            source_window=window,
            corpus_text="test text here",
            corpus_source="source.pdf",
            similarity_score=0.85,
        )

        result = ComparisonResult(
            total_windows=10,
            windows_with_matches=5,
            matches=[match],
            highest_similarity=0.85,
            flagged_count=1,
        )

        summary = reporter._create_summary(result)

        assert summary.total_matches == 1
        assert summary.unique_sources == 1


class TestTextTruncation:
    """Tests for text truncation."""

    def test_truncate_short_text(self) -> None:
        """Test truncating short text."""
        reporter = PlagiarismReporter()

        result = reporter._truncate("short", 10)

        assert result == "short"

    def test_truncate_long_text(self) -> None:
        """Test truncating long text."""
        reporter = PlagiarismReporter()

        result = reporter._truncate("this is a very long text", 10)

        assert len(result) == 10
        assert result.endswith("...")


class TestColorMapping:
    """Tests for color mapping functions."""

    def test_severity_colors(self) -> None:
        """Test severity color mapping."""
        reporter = PlagiarismReporter()

        assert reporter._severity_color(SeverityLevel.LOW) == "green"
        assert reporter._severity_color(SeverityLevel.CRITICAL) == "red"

    def test_similarity_colors(self) -> None:
        """Test similarity score color mapping."""
        reporter = PlagiarismReporter()

        assert reporter._similarity_color(0.96) == "red"
        assert reporter._similarity_color(0.87) == "orange3"
        assert reporter._similarity_color(0.76) == "yellow"
        assert reporter._similarity_color(0.70) == "green"


class TestReportGeneration:
    """Tests for report generation."""

    def test_generate_report_empty(self) -> None:
        """Test generating report for empty result."""
        reporter = PlagiarismReporter()
        result = ComparisonResult()

        summary = reporter.generate_report(result)

        assert isinstance(summary, ReportSummary)
        assert summary.total_matches == 0

    def test_generate_report_with_data(self, capsys) -> None:
        """Test generating report with data."""
        config = ReportConfig(show_highlights=True)
        reporter = PlagiarismReporter(config=config)

        window = TextWindow("test text content", 0, 17, 0)
        match = SimilarityMatch(
            source_window=window,
            corpus_text="test text content in corpus",
            corpus_source="source.pdf",
            similarity_score=0.9,
        )

        result = ComparisonResult(
            total_windows=5,
            windows_with_matches=2,
            matches=[match],
            highest_similarity=0.9,
            flagged_count=1,
        )

        summary = reporter.generate_report(result)

        assert summary.total_matches == 1
        assert summary.high_similarity_count == 1


class TestHighlighting:
    """Tests for text highlighting."""

    def test_highlight_passage(self) -> None:
        """Test highlighting matched passage."""
        reporter = PlagiarismReporter()

        text = "This is some text that contains a match here."
        window = TextWindow("a match", 30, 37, 0)
        match = SimilarityMatch(
            source_window=window,
            corpus_text="a match in corpus",
            corpus_source="source.txt",
            similarity_score=0.9,
        )

        highlighted = reporter.highlight_passage(text, [match])

        # Should return Rich Text object
        assert highlighted is not None


# Factory function tests


class TestCreateReporter:
    """Tests for create_reporter factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        reporter = create_reporter()

        assert reporter.config.show_highlights is True

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        reporter = create_reporter(
            show_highlights=False,
            max_matches=5,
        )

        assert reporter.config.show_highlights is False
        assert reporter.config.max_matches == 5


class TestPrintPlagiarismReport:
    """Tests for print_plagiarism_report function."""

    def test_print_report(self) -> None:
        """Test printing plagiarism report."""
        result = ComparisonResult()

        summary = print_plagiarism_report(result)

        assert isinstance(summary, ReportSummary)

    def test_print_report_with_source(self) -> None:
        """Test printing report with source text."""
        result = ComparisonResult()

        summary = print_plagiarism_report(result, "source text")

        assert isinstance(summary, ReportSummary)
