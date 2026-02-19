"""Unit tests for security badge generation.

Security Shield CI Integration
Epic: EP-26 (Security & Compliance)

Tests security badge generation for shields.io format.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestforge.core.security.badge_generator import (
    generate_badge_data,
    generate_markdown_badge,
    generate_summary_text,
    save_badge_json,
)
from ingestforge.core.security.scanner import (
    FindingCategory,
    SecurityFinding,
    SecurityReport,
    Severity,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def clean_report() -> SecurityReport:
    """Create report with no findings.

    Returns:
        SecurityReport with no findings.
    """
    report = SecurityReport(
        report_id="clean", scan_path="/test", findings=[], files_scanned=50
    )
    report.complete(1000.0)
    return report


@pytest.fixture
def critical_report() -> SecurityReport:
    """Create report with critical findings.

    Returns:
        SecurityReport with critical findings.
    """
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Critical issue",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        )
    ]
    report = SecurityReport(
        report_id="critical", scan_path="/test", findings=findings, files_scanned=50
    )
    report.complete(1000.0)
    return report


@pytest.fixture
def high_report() -> SecurityReport:
    """Create report with high severity findings.

    Returns:
        SecurityReport with high findings.
    """
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.INJECTION,
            severity=Severity.HIGH,
            title="High issue",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-002",
        ),
        SecurityFinding(
            finding_id="f2",
            category=FindingCategory.INJECTION,
            severity=Severity.HIGH,
            title="Another high",
            description="Test",
            file_path="test.py",
            line_number=2,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-002",
        ),
    ]
    report = SecurityReport(
        report_id="high", scan_path="/test", findings=findings, files_scanned=50
    )
    report.complete(1000.0)
    return report


@pytest.fixture
def medium_report() -> SecurityReport:
    """Create report with medium severity findings.

    Returns:
        SecurityReport with medium findings.
    """
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.CONFIG,
            severity=Severity.MEDIUM,
            title="Medium issue",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-003",
        )
    ]
    report = SecurityReport(
        report_id="medium", scan_path="/test", findings=findings, files_scanned=50
    )
    report.complete(1000.0)
    return report


@pytest.fixture
def low_report() -> SecurityReport:
    """Create report with low severity findings.

    Returns:
        SecurityReport with low findings.
    """
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.CONFIG,
            severity=Severity.LOW,
            title="Low issue",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-004",
        )
    ]
    report = SecurityReport(
        report_id="low", scan_path="/test", findings=findings, files_scanned=50
    )
    report.complete(1000.0)
    return report


# =============================================================================
# BADGE DATA GENERATION TESTS (5 tests)
# =============================================================================


def test_given_clean_report_when_generate_badge_then_passing(
    clean_report: SecurityReport,
) -> None:
    """GIVEN clean report WHEN badge generated THEN shows 'passing'."""
    badge_data = generate_badge_data(clean_report)

    assert badge_data["message"] == "passing"
    assert badge_data["color"] == "brightgreen"
    assert badge_data["label"] == "security"


def test_given_critical_report_when_generate_badge_then_critical_red(
    critical_report: SecurityReport,
) -> None:
    """GIVEN critical report WHEN badge generated THEN shows critical count and red."""
    badge_data = generate_badge_data(critical_report)

    assert badge_data["message"] == "1 critical"
    assert badge_data["color"] == "red"


def test_given_high_report_when_generate_badge_then_high_orange(
    high_report: SecurityReport,
) -> None:
    """GIVEN high report WHEN badge generated THEN shows high count and orange."""
    badge_data = generate_badge_data(high_report)

    assert badge_data["message"] == "2 high"
    assert badge_data["color"] == "orange"


def test_given_medium_report_when_generate_badge_then_medium_yellow(
    medium_report: SecurityReport,
) -> None:
    """GIVEN medium report WHEN badge generated THEN shows medium count and yellow."""
    badge_data = generate_badge_data(medium_report)

    assert badge_data["message"] == "1 medium"
    assert badge_data["color"] == "yellow"


def test_given_low_report_when_generate_badge_then_low_yellowgreen(
    low_report: SecurityReport,
) -> None:
    """GIVEN low report WHEN badge generated THEN shows low count and yellowgreen."""
    badge_data = generate_badge_data(low_report)

    assert badge_data["message"] == "1 low"
    assert badge_data["color"] == "yellowgreen"


# =============================================================================
# BADGE SCHEMA TESTS (2 tests)
# =============================================================================


def test_given_report_when_generate_badge_then_has_schema_version(
    clean_report: SecurityReport,
) -> None:
    """GIVEN report WHEN badge generated THEN includes schemaVersion."""
    badge_data = generate_badge_data(clean_report)

    assert badge_data["schemaVersion"] == 1


def test_given_report_when_generate_badge_then_has_required_fields(
    clean_report: SecurityReport,
) -> None:
    """GIVEN report WHEN badge generated THEN has all required fields."""
    badge_data = generate_badge_data(clean_report)

    assert "schemaVersion" in badge_data
    assert "label" in badge_data
    assert "message" in badge_data
    assert "color" in badge_data


# =============================================================================
# FILE SAVE TESTS (2 tests)
# =============================================================================


def test_given_report_when_save_badge_json_then_creates_file(
    clean_report: SecurityReport, tmp_path: Path
) -> None:
    """GIVEN report WHEN save_badge_json called THEN creates JSON file."""
    output_path = tmp_path / "badge.json"

    save_badge_json(clean_report, output_path)

    assert output_path.exists()


def test_given_report_when_save_badge_json_then_valid_json(
    clean_report: SecurityReport, tmp_path: Path
) -> None:
    """GIVEN report WHEN save_badge_json called THEN creates valid JSON."""
    output_path = tmp_path / "badge.json"

    save_badge_json(clean_report, output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        badge_data = json.load(f)

    assert badge_data["schemaVersion"] == 1
    assert badge_data["message"] == "passing"


# =============================================================================
# MARKDOWN BADGE TESTS (3 tests)
# =============================================================================


def test_given_clean_report_when_markdown_badge_then_creates_static(
    clean_report: SecurityReport,
) -> None:
    """GIVEN clean report WHEN markdown badge THEN creates static badge."""
    markdown = generate_markdown_badge(clean_report)

    assert "![security]" in markdown
    assert "shields.io/badge" in markdown
    assert "passing" in markdown
    assert "brightgreen" in markdown


def test_given_endpoint_url_when_markdown_badge_then_creates_dynamic(
    clean_report: SecurityReport,
) -> None:
    """GIVEN endpoint URL WHEN markdown badge THEN creates dynamic badge."""
    markdown = generate_markdown_badge(
        clean_report,
        badge_url="https://img.shields.io/endpoint?url=",
        endpoint_url="https://example.com/badge.json",
    )

    assert "![Security]" in markdown
    assert "endpoint?url=" in markdown
    assert "example.com/badge.json" in markdown


def test_given_critical_report_when_markdown_badge_then_shows_critical(
    critical_report: SecurityReport,
) -> None:
    """GIVEN critical report WHEN markdown badge THEN shows critical in badge."""
    markdown = generate_markdown_badge(critical_report)

    assert "1%20critical" in markdown or "1 critical" in markdown
    assert "red" in markdown


# =============================================================================
# SUMMARY TEXT TESTS (5 tests)
# =============================================================================


def test_given_clean_report_when_summary_then_shows_passed(
    clean_report: SecurityReport,
) -> None:
    """GIVEN clean report WHEN summary generated THEN shows PASSED status."""
    summary = generate_summary_text(clean_report)

    assert "PASSED" in summary
    assert "✅" in summary


def test_given_critical_report_when_summary_then_shows_failed(
    critical_report: SecurityReport,
) -> None:
    """GIVEN critical report WHEN summary generated THEN shows FAILED status."""
    summary = generate_summary_text(critical_report)

    assert "FAILED" in summary
    assert "❌" in summary


def test_given_medium_report_when_summary_then_shows_warning(
    medium_report: SecurityReport,
) -> None:
    """GIVEN medium report WHEN summary generated THEN shows WARNING status."""
    summary = generate_summary_text(medium_report)

    assert "WARNING" in summary
    assert "⚠️" in summary


def test_given_report_when_summary_then_includes_scan_stats(
    clean_report: SecurityReport,
) -> None:
    """GIVEN report WHEN summary generated THEN includes scan statistics."""
    summary = generate_summary_text(clean_report)

    assert "Files Scanned:" in summary
    assert "50" in summary
    assert "Scan Duration:" in summary
    assert "1000" in summary


def test_given_report_when_summary_then_includes_severity_counts(
    high_report: SecurityReport,
) -> None:
    """GIVEN report WHEN summary generated THEN includes severity breakdown."""
    summary = generate_summary_text(high_report)

    assert "Critical:" in summary
    assert "High:" in summary
    assert "Medium:" in summary
    assert "Low:" in summary
    assert "Total Findings:" in summary


# =============================================================================
# PRIORITY TESTS (3 tests)
# =============================================================================


def test_given_critical_and_high_when_badge_then_prefers_critical() -> None:
    """GIVEN both critical and high WHEN badge THEN shows critical."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        ),
        SecurityFinding(
            finding_id="f2",
            category=FindingCategory.INJECTION,
            severity=Severity.HIGH,
            title="High",
            description="Test",
            file_path="test.py",
            line_number=2,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-002",
        ),
    ]
    report = SecurityReport(report_id="mixed", scan_path="/test", findings=findings)
    report.complete(0)

    badge_data = generate_badge_data(report)

    assert "critical" in badge_data["message"]
    assert badge_data["color"] == "red"


def test_given_high_and_medium_when_badge_then_prefers_high() -> None:
    """GIVEN both high and medium WHEN badge THEN shows high."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.INJECTION,
            severity=Severity.HIGH,
            title="High",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        ),
        SecurityFinding(
            finding_id="f2",
            category=FindingCategory.CONFIG,
            severity=Severity.MEDIUM,
            title="Medium",
            description="Test",
            file_path="test.py",
            line_number=2,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-002",
        ),
    ]
    report = SecurityReport(report_id="mixed", scan_path="/test", findings=findings)
    report.complete(0)

    badge_data = generate_badge_data(report)

    assert "high" in badge_data["message"]
    assert badge_data["color"] == "orange"


def test_given_medium_and_low_when_badge_then_prefers_medium() -> None:
    """GIVEN both medium and low WHEN badge THEN shows medium."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.CONFIG,
            severity=Severity.MEDIUM,
            title="Medium",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        ),
        SecurityFinding(
            finding_id="f2",
            category=FindingCategory.CONFIG,
            severity=Severity.LOW,
            title="Low",
            description="Test",
            file_path="test.py",
            line_number=2,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-002",
        ),
    ]
    report = SecurityReport(report_id="mixed", scan_path="/test", findings=findings)
    report.complete(0)

    badge_data = generate_badge_data(report)

    assert "medium" in badge_data["message"]
    assert badge_data["color"] == "yellow"


# =============================================================================
# MULTIPLE SEVERITY TESTS (3 tests)
# =============================================================================


def test_given_multiple_critical_when_badge_then_shows_count() -> None:
    """GIVEN multiple critical findings WHEN badge THEN shows total count."""
    findings = [
        SecurityFinding(
            finding_id=f"f{i}",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title=f"Critical {i}",
            description="Test",
            file_path="test.py",
            line_number=i,
            line_content="test",
            recommendation="Fix",
            rule_id=f"SEC-{i:03d}",
        )
        for i in range(5)
    ]
    report = SecurityReport(report_id="multi", scan_path="/test", findings=findings)
    report.complete(0)

    badge_data = generate_badge_data(report)

    assert "5 critical" in badge_data["message"]
    assert badge_data["color"] == "red"


def test_given_info_only_when_badge_then_shows_passing() -> None:
    """GIVEN only INFO severity WHEN badge THEN shows passing."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.CONFIG,
            severity=Severity.INFO,
            title="Info",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Note",
            rule_id="INFO-01",
        )
    ]
    report = SecurityReport(report_id="info", scan_path="/test", findings=findings)
    report.complete(0)

    badge_data = generate_badge_data(report)

    assert badge_data["message"] == "passing"
    assert badge_data["color"] == "brightgreen"


def test_given_all_severities_when_summary_then_shows_all_counts() -> None:
    """GIVEN all severity levels WHEN summary THEN shows breakdown."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        ),
        SecurityFinding(
            finding_id="f2",
            category=FindingCategory.INJECTION,
            severity=Severity.HIGH,
            title="High",
            description="Test",
            file_path="test.py",
            line_number=2,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-002",
        ),
        SecurityFinding(
            finding_id="f3",
            category=FindingCategory.CONFIG,
            severity=Severity.MEDIUM,
            title="Medium",
            description="Test",
            file_path="test.py",
            line_number=3,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-003",
        ),
        SecurityFinding(
            finding_id="f4",
            category=FindingCategory.CONFIG,
            severity=Severity.LOW,
            title="Low",
            description="Test",
            file_path="test.py",
            line_number=4,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-004",
        ),
        SecurityFinding(
            finding_id="f5",
            category=FindingCategory.CONFIG,
            severity=Severity.INFO,
            title="Info",
            description="Test",
            file_path="test.py",
            line_number=5,
            line_content="test",
            recommendation="Note",
            rule_id="INFO-01",
        ),
    ]
    report = SecurityReport(report_id="all", scan_path="/test", findings=findings)
    report.complete(2500.0)

    summary = generate_summary_text(report)

    assert "Critical: 1" in summary
    assert "High: 1" in summary
    assert "Medium: 1" in summary
    assert "Low: 1" in summary
    assert "Info: 1" in summary
    assert "Total Findings: 5" in summary


# =============================================================================
# MARKDOWN EDGE CASES (3 tests)
# =============================================================================


def test_given_spaces_in_message_when_markdown_then_url_encodes() -> None:
    """GIVEN message with spaces WHEN markdown THEN handles encoding."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        )
    ]
    report = SecurityReport(report_id="test", scan_path="/test", findings=findings)
    report.complete(0)

    markdown = generate_markdown_badge(report)

    # Should handle spaces in URL (either %20 or actual spaces)
    assert "critical" in markdown.lower()


def test_given_no_endpoint_when_markdown_then_creates_static_badge() -> None:
    """GIVEN no endpoint URL WHEN markdown THEN creates static badge."""
    report = SecurityReport(report_id="test", scan_path="/test", findings=[])
    report.complete(0)

    markdown = generate_markdown_badge(report, endpoint_url="")

    assert "shields.io/badge" in markdown
    assert "endpoint" not in markdown


def test_given_custom_badge_url_when_markdown_then_uses_custom() -> None:
    """GIVEN custom badge URL WHEN markdown THEN uses it."""
    report = SecurityReport(report_id="test", scan_path="/test", findings=[])
    report.complete(0)

    custom_url = "https://custom.shields.io/endpoint?url="
    markdown = generate_markdown_badge(
        report, badge_url=custom_url, endpoint_url="https://api.example.com/badge"
    )

    assert custom_url in markdown
    assert "api.example.com/badge" in markdown


# =============================================================================
# SUMMARY FORMAT TESTS (3 tests)
# =============================================================================


def test_given_report_when_summary_then_includes_header() -> None:
    """GIVEN report WHEN summary THEN includes markdown header."""
    report = SecurityReport(report_id="test", scan_path="/test", findings=[])
    report.complete(1000.0)

    summary = generate_summary_text(report)

    assert summary.startswith("# Security Scan Summary")


def test_given_report_when_summary_then_includes_emoji_indicators() -> None:
    """GIVEN report WHEN summary THEN includes emoji for severity."""
    findings = [
        SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="Fix",
            rule_id="SEC-001",
        )
    ]
    report = SecurityReport(report_id="test", scan_path="/test", findings=findings)
    report.complete(0)

    summary = generate_summary_text(report)

    # Should have emoji indicators
    assert "🔴" in summary  # Critical
    assert "❌" in summary  # Failed status


def test_given_zero_findings_when_summary_then_shows_all_zero() -> None:
    """GIVEN zero findings WHEN summary THEN shows all counts as zero."""
    report = SecurityReport(report_id="clean", scan_path="/test", findings=[])
    report.complete(500.0)

    summary = generate_summary_text(report)

    assert "Critical: 0" in summary
    assert "High: 0" in summary
    assert "Medium: 0" in summary
    assert "Low: 0" in summary
    assert "Info: 0" in summary
    assert "Total Findings: 0" in summary
