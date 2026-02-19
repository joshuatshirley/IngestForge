"""Unit tests for SARIF formatter.

Security Shield CI Integration
Epic: EP-26 (Security & Compliance)

Tests SARIF 2.1.0 output formatting for GitHub Code Scanning.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestforge.core.security.sarif_formatter import (
    SARIF_VERSION,
    convert_to_sarif,
    save_sarif,
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
def sample_findings() -> list[SecurityFinding]:
    """Create sample security findings.

    Returns:
        List of SecurityFinding instances.
    """
    return [
        SecurityFinding(
            finding_id="finding-1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Hardcoded API key",
            description="API key found in source code",
            file_path="src/config.py",
            line_number=42,
            line_content="API_KEY = 'sk-abc123'",
            recommendation="Use environment variables for secrets",
            rule_id="SEC-001",
            metadata={"confidence": "HIGH"},
        ),
        SecurityFinding(
            finding_id="finding-2",
            category=FindingCategory.INJECTION,
            severity=Severity.HIGH,
            title="SQL Injection vulnerability",
            description="Unsafe string concatenation in SQL query",
            file_path="src/database.py",
            line_number=15,
            line_content="query = f'SELECT * FROM users WHERE id={user_id}'",
            recommendation="Use parameterized queries",
            rule_id="SEC-010",
            metadata={},
        ),
    ]


@pytest.fixture
def sample_report(sample_findings: list[SecurityFinding]) -> SecurityReport:
    """Create sample security report.

    Args:
        sample_findings: List of sample findings.

    Returns:
        SecurityReport instance.
    """
    report = SecurityReport(
        report_id="test-report-123",
        scan_path="/test/project",
        findings=sample_findings,
        files_scanned=100,
        scan_duration_ms=1500.5,
    )
    report.complete(1500.5)
    return report


# =============================================================================
# BASIC SARIF CONVERSION TESTS (5 tests)
# =============================================================================


def test_given_report_when_converted_then_creates_sarif_structure(
    sample_report: SecurityReport,
) -> None:
    """GIVEN security report WHEN converted THEN creates valid SARIF structure."""
    sarif = convert_to_sarif(sample_report)

    assert sarif["version"] == SARIF_VERSION
    assert "$schema" in sarif
    assert "runs" in sarif
    assert len(sarif["runs"]) == 1


def test_given_report_when_converted_then_includes_tool_section(
    sample_report: SecurityReport,
) -> None:
    """GIVEN security report WHEN converted THEN includes tool section."""
    sarif = convert_to_sarif(sample_report, tool_name="TestTool", tool_version="2.0.0")

    run = sarif["runs"][0]
    assert "tool" in run
    assert run["tool"]["driver"]["name"] == "TestTool"
    assert run["tool"]["driver"]["version"] == "2.0.0"


def test_given_report_when_converted_then_includes_results(
    sample_report: SecurityReport,
) -> None:
    """GIVEN security report WHEN converted THEN includes results section."""
    sarif = convert_to_sarif(sample_report)

    run = sarif["runs"][0]
    assert "results" in run
    assert len(run["results"]) == 2


def test_given_report_when_converted_then_includes_invocations(
    sample_report: SecurityReport,
) -> None:
    """GIVEN security report WHEN converted THEN includes invocations."""
    sarif = convert_to_sarif(sample_report)

    run = sarif["runs"][0]
    assert "invocations" in run
    assert len(run["invocations"]) == 1


def test_given_report_when_converted_then_includes_rules(
    sample_report: SecurityReport,
) -> None:
    """GIVEN security report WHEN converted THEN includes rules."""
    sarif = convert_to_sarif(sample_report)

    run = sarif["runs"][0]
    rules = run["tool"]["driver"]["rules"]
    assert len(rules) == 2  # Two unique rule IDs
    assert all("id" in rule for rule in rules)


# =============================================================================
# RESULTS SECTION TESTS (5 tests)
# =============================================================================


def test_given_findings_when_converted_then_each_has_rule_id(
    sample_report: SecurityReport,
) -> None:
    """GIVEN findings WHEN converted THEN each result has ruleId."""
    sarif = convert_to_sarif(sample_report)

    results = sarif["runs"][0]["results"]
    assert all("ruleId" in result for result in results)
    assert results[0]["ruleId"] == "SEC-001"
    assert results[1]["ruleId"] == "SEC-010"


def test_given_critical_finding_when_converted_then_level_is_error(
    sample_report: SecurityReport,
) -> None:
    """GIVEN CRITICAL finding WHEN converted THEN level is 'error'."""
    sarif = convert_to_sarif(sample_report)

    results = sarif["runs"][0]["results"]
    critical_result = results[0]  # First finding is CRITICAL
    assert critical_result["level"] == "error"


def test_given_high_finding_when_converted_then_level_is_error(
    sample_report: SecurityReport,
) -> None:
    """GIVEN HIGH finding WHEN converted THEN level is 'error'."""
    sarif = convert_to_sarif(sample_report)

    results = sarif["runs"][0]["results"]
    high_result = results[1]  # Second finding is HIGH
    assert high_result["level"] == "error"


def test_given_finding_when_converted_then_has_location(
    sample_report: SecurityReport,
) -> None:
    """GIVEN finding WHEN converted THEN has physicalLocation."""
    sarif = convert_to_sarif(sample_report)

    result = sarif["runs"][0]["results"][0]
    assert "locations" in result
    location = result["locations"][0]
    assert "physicalLocation" in location

    physical_loc = location["physicalLocation"]
    assert physical_loc["artifactLocation"]["uri"] == "src/config.py"
    assert physical_loc["region"]["startLine"] == 42


def test_given_finding_when_converted_then_has_message(
    sample_report: SecurityReport,
) -> None:
    """GIVEN finding WHEN converted THEN has message text."""
    sarif = convert_to_sarif(sample_report)

    result = sarif["runs"][0]["results"][0]
    assert "message" in result
    assert result["message"]["text"] == "API key found in source code"


# =============================================================================
# TOOL SECTION TESTS (3 tests)
# =============================================================================


def test_given_rules_when_converted_then_each_has_id(
    sample_report: SecurityReport,
) -> None:
    """GIVEN rules WHEN converted THEN each has unique ID."""
    sarif = convert_to_sarif(sample_report)

    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    rule_ids = {rule["id"] for rule in rules}
    assert rule_ids == {"SEC-001", "SEC-010"}


def test_given_rules_when_converted_then_each_has_description(
    sample_report: SecurityReport,
) -> None:
    """GIVEN rules WHEN converted THEN each has description."""
    sarif = convert_to_sarif(sample_report)

    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    assert all("shortDescription" in rule for rule in rules)
    assert all("fullDescription" in rule for rule in rules)


def test_given_rules_when_converted_then_each_has_help(
    sample_report: SecurityReport,
) -> None:
    """GIVEN rules WHEN converted THEN each has help text."""
    sarif = convert_to_sarif(sample_report)

    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    assert all("help" in rule for rule in rules)


# =============================================================================
# INVOCATION SECTION TESTS (3 tests)
# =============================================================================


def test_given_clean_report_when_converted_then_execution_successful(
    sample_findings: list[SecurityFinding],
) -> None:
    """GIVEN report with no critical WHEN converted THEN executionSuccessful."""
    # Create report with only MEDIUM findings
    medium_finding = SecurityFinding(
        finding_id="f1",
        category=FindingCategory.CONFIG,
        severity=Severity.MEDIUM,
        title="Test",
        description="Test",
        file_path="test.py",
        line_number=1,
        line_content="test",
        recommendation="Fix it",
        rule_id="TEST",
    )
    report = SecurityReport(
        report_id="r1", scan_path="/test", findings=[medium_finding]
    )
    report.complete(100)

    sarif = convert_to_sarif(report)

    invocation = sarif["runs"][0]["invocations"][0]
    assert invocation["executionSuccessful"] is True


def test_given_critical_report_when_converted_then_execution_failed(
    sample_report: SecurityReport,
) -> None:
    """GIVEN report with critical WHEN converted THEN executionSuccessful false."""
    sarif = convert_to_sarif(sample_report)

    invocation = sarif["runs"][0]["invocations"][0]
    assert invocation["executionSuccessful"] is False


def test_given_report_when_converted_then_includes_scan_metadata(
    sample_report: SecurityReport,
) -> None:
    """GIVEN report WHEN converted THEN includes scan metadata."""
    sarif = convert_to_sarif(sample_report)

    invocation = sarif["runs"][0]["invocations"][0]
    assert invocation["properties"]["files_scanned"] == 100
    assert invocation["properties"]["scan_duration_ms"] == 1500.5


# =============================================================================
# SEVERITY MAPPING TESTS (4 tests)
# =============================================================================


def test_given_critical_severity_when_mapped_then_returns_error() -> None:
    """GIVEN CRITICAL severity WHEN mapped THEN returns 'error'."""
    from ingestforge.core.security.sarif_formatter import _severity_to_sarif_level

    level = _severity_to_sarif_level(Severity.CRITICAL)
    assert level == "error"


def test_given_high_severity_when_mapped_then_returns_error() -> None:
    """GIVEN HIGH severity WHEN mapped THEN returns 'error'."""
    from ingestforge.core.security.sarif_formatter import _severity_to_sarif_level

    level = _severity_to_sarif_level(Severity.HIGH)
    assert level == "error"


def test_given_medium_severity_when_mapped_then_returns_warning() -> None:
    """GIVEN MEDIUM severity WHEN mapped THEN returns 'warning'."""
    from ingestforge.core.security.sarif_formatter import _severity_to_sarif_level

    level = _severity_to_sarif_level(Severity.MEDIUM)
    assert level == "warning"


def test_given_low_severity_when_mapped_then_returns_note() -> None:
    """GIVEN LOW severity WHEN mapped THEN returns 'note'."""
    from ingestforge.core.security.sarif_formatter import _severity_to_sarif_level

    level = _severity_to_sarif_level(Severity.LOW)
    assert level == "note"


# =============================================================================
# FILE SAVE TESTS (2 tests)
# =============================================================================


def test_given_report_when_save_sarif_then_creates_file(
    sample_report: SecurityReport, tmp_path: Path
) -> None:
    """GIVEN report WHEN save_sarif called THEN creates SARIF file."""
    output_path = tmp_path / "test.sarif"

    save_sarif(sample_report, output_path)

    assert output_path.exists()


def test_given_report_when_save_sarif_then_valid_json(
    sample_report: SecurityReport, tmp_path: Path
) -> None:
    """GIVEN report WHEN save_sarif called THEN creates valid JSON."""
    output_path = tmp_path / "test.sarif"

    save_sarif(sample_report, output_path)

    with open(output_path, "r", encoding="utf-8") as f:
        sarif = json.load(f)

    assert sarif["version"] == SARIF_VERSION
    assert len(sarif["runs"]) == 1


# =============================================================================
# EDGE CASE TESTS (3 tests)
# =============================================================================


def test_given_empty_report_when_converted_then_creates_valid_sarif() -> None:
    """GIVEN empty report WHEN converted THEN creates valid SARIF."""
    report = SecurityReport(report_id="empty", scan_path="/test", findings=[])
    report.complete(0)

    sarif = convert_to_sarif(report)

    assert sarif["version"] == SARIF_VERSION
    assert len(sarif["runs"][0]["results"]) == 0
    assert len(sarif["runs"][0]["tool"]["driver"]["rules"]) == 0


def test_given_long_line_content_when_converted_then_truncates(
    sample_report: SecurityReport,
) -> None:
    """GIVEN long line content WHEN converted THEN truncates in snippet."""
    # Add finding with very long line
    long_line = "x" * 500
    finding = SecurityFinding(
        finding_id="f-long",
        category=FindingCategory.CONFIG,
        severity=Severity.LOW,
        title="Long line",
        description="Test",
        file_path="test.py",
        line_number=1,
        line_content=long_line,
        recommendation="Fix",
        rule_id="LONG",
    )
    sample_report.findings.append(finding)

    sarif = convert_to_sarif(sample_report)

    # Find the result with long line
    results = sarif["runs"][0]["results"]
    long_result = next(r for r in results if r["ruleId"] == "LONG")
    snippet = long_result["locations"][0]["physicalLocation"]["region"]["snippet"][
        "text"
    ]

    # Should be truncated to 200 chars
    assert len(snippet) <= 200


def test_given_duplicate_rules_when_converted_then_deduplicates(
    sample_findings: list[SecurityFinding],
) -> None:
    """GIVEN duplicate rule IDs WHEN converted THEN deduplicates rules."""
    # Add another finding with same rule ID as first
    duplicate_finding = SecurityFinding(
        finding_id="finding-3",
        category=FindingCategory.SECRETS,
        severity=Severity.HIGH,
        title="Another hardcoded secret",
        description="Another API key",
        file_path="src/other.py",
        line_number=10,
        line_content="SECRET = 'xyz'",
        recommendation="Use env vars",
        rule_id="SEC-001",  # Same as first finding
    )
    sample_findings.append(duplicate_finding)

    report = SecurityReport(
        report_id="test", scan_path="/test", findings=sample_findings
    )
    report.complete(0)

    sarif = convert_to_sarif(report)

    # Should have 2 unique rules (SEC-001, SEC-010), not 3
    rules = sarif["runs"][0]["tool"]["driver"]["rules"]
    rule_ids = [rule["id"] for rule in rules]
    assert len(rule_ids) == 2
    assert "SEC-001" in rule_ids
    assert "SEC-010" in rule_ids


# =============================================================================
# TRUNCATION AND LIMITS TESTS (3 tests)
# =============================================================================


def test_given_max_results_when_converted_then_truncates() -> None:
    """GIVEN >MAX results WHEN converted THEN truncates to MAX_SARIF_RESULTS."""
    from ingestforge.core.security.sarif_formatter import MAX_SARIF_RESULTS

    # Create report with MAX + 100 findings
    many_findings = []
    for i in range(MAX_SARIF_RESULTS + 100):
        many_findings.append(
            SecurityFinding(
                finding_id=f"f-{i}",
                category=FindingCategory.CONFIG,
                severity=Severity.LOW,
                title=f"Issue {i}",
                description="Test",
                file_path="test.py",
                line_number=i,
                line_content="test",
                recommendation="Fix",
                rule_id=f"RULE-{i}",
            )
        )

    report = SecurityReport(
        report_id="large", scan_path="/test", findings=many_findings
    )
    report.complete(0)

    sarif = convert_to_sarif(report)

    results = sarif["runs"][0]["results"]
    assert len(results) == MAX_SARIF_RESULTS


def test_given_info_severity_when_converted_then_level_is_note() -> None:
    """GIVEN INFO severity WHEN converted THEN SARIF level is 'note'."""
    finding = SecurityFinding(
        finding_id="f-info",
        category=FindingCategory.CONFIG,
        severity=Severity.INFO,
        title="Info finding",
        description="Informational only",
        file_path="test.py",
        line_number=1,
        line_content="test",
        recommendation="Consider this",
        rule_id="INFO-01",
    )

    report = SecurityReport(report_id="info", scan_path="/test", findings=[finding])
    report.complete(0)

    sarif = convert_to_sarif(report)

    result = sarif["runs"][0]["results"][0]
    assert result["level"] == "note"


def test_given_findings_when_converted_then_includes_fingerprints() -> None:
    """GIVEN findings WHEN converted THEN includes partial fingerprints."""
    finding = SecurityFinding(
        finding_id="f1",
        category=FindingCategory.SECRETS,
        severity=Severity.HIGH,
        title="Secret",
        description="Test",
        file_path="src/config.py",
        line_number=42,
        line_content="test",
        recommendation="Fix",
        rule_id="SEC-001",
    )

    report = SecurityReport(report_id="test", scan_path="/test", findings=[finding])
    report.complete(0)

    sarif = convert_to_sarif(report)

    result = sarif["runs"][0]["results"][0]
    assert "partialFingerprints" in result
    assert "primaryLocationLineHash" in result["partialFingerprints"]


# =============================================================================
# PROPERTIES AND METADATA TESTS (4 tests)
# =============================================================================


def test_given_finding_when_converted_then_includes_properties() -> None:
    """GIVEN finding WHEN converted THEN includes category in properties."""
    finding = SecurityFinding(
        finding_id="f1",
        category=FindingCategory.INJECTION,
        severity=Severity.HIGH,
        title="SQL Injection",
        description="Test",
        file_path="test.py",
        line_number=1,
        line_content="test",
        recommendation="Fix",
        rule_id="INJ-01",
    )

    report = SecurityReport(report_id="test", scan_path="/test", findings=[finding])
    report.complete(0)

    sarif = convert_to_sarif(report)

    result = sarif["runs"][0]["results"][0]
    assert result["properties"]["category"] == "injection"
    assert result["properties"]["finding_id"] == "f1"


def test_given_rule_when_converted_then_includes_tags() -> None:
    """GIVEN rule WHEN converted THEN includes category tags."""
    finding = SecurityFinding(
        finding_id="f1",
        category=FindingCategory.CRYPTO,
        severity=Severity.MEDIUM,
        title="Weak crypto",
        description="Test",
        file_path="test.py",
        line_number=1,
        line_content="test",
        recommendation="Fix",
        rule_id="CRYPTO-01",
    )

    report = SecurityReport(report_id="test", scan_path="/test", findings=[finding])
    report.complete(0)

    sarif = convert_to_sarif(report)

    rule = sarif["runs"][0]["tool"]["driver"]["rules"][0]
    assert "crypto" in rule["properties"]["tags"]


def test_given_completed_report_when_converted_then_includes_timestamps() -> None:
    """GIVEN completed report WHEN converted THEN includes start/end times."""
    finding = SecurityFinding(
        finding_id="f1",
        category=FindingCategory.CONFIG,
        severity=Severity.LOW,
        title="Config issue",
        description="Test",
        file_path="test.py",
        line_number=1,
        line_content="test",
        recommendation="Fix",
        rule_id="CFG-01",
    )

    report = SecurityReport(report_id="test", scan_path="/test", findings=[finding])
    report.complete(1500.5)

    sarif = convert_to_sarif(report)

    invocation = sarif["runs"][0]["invocations"][0]
    assert "startTimeUtc" in invocation
    assert "endTimeUtc" in invocation
    assert invocation["startTimeUtc"] is not None
    assert invocation["endTimeUtc"] is not None


def test_given_tool_info_when_converted_then_includes_in_driver() -> None:
    """GIVEN custom tool info WHEN converted THEN includes in driver section."""
    finding = SecurityFinding(
        finding_id="f1",
        category=FindingCategory.CONFIG,
        severity=Severity.LOW,
        title="Test",
        description="Test",
        file_path="test.py",
        line_number=1,
        line_content="test",
        recommendation="Fix",
        rule_id="TST-01",
    )

    report = SecurityReport(report_id="test", scan_path="/test", findings=[finding])
    report.complete(0)

    sarif = convert_to_sarif(report, tool_name="CustomScanner", tool_version="3.2.1")

    driver = sarif["runs"][0]["tool"]["driver"]
    assert driver["name"] == "CustomScanner"
    assert driver["version"] == "3.2.1"
    assert "informationUri" in driver
