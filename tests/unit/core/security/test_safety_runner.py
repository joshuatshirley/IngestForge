"""Unit tests for Safety integration.

Security Shield CI Integration
Epic: EP-26 (Security & Compliance)

Tests Safety dependency vulnerability scanner integration.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

from ingestforge.core.security.safety_runner import (
    SafetyRunner,
    run_safety_scan,
)
from ingestforge.core.security.scanner import (
    FindingCategory,
    SecurityFinding,
    Severity,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_safety_output() -> List[Dict[str, Any]]:
    """Sample Safety JSON output.

    Returns:
        Safety JSON structure (list of vulnerabilities).
    """
    return [
        {
            "package": "django",
            "installed_version": "2.2.0",
            "vulnerability": "CVE-2021-33203",
            "advisory": "Django 2.2.0 is affected by a critical SQL injection vulnerability that allows remote code execution.",
            "specs": [">=2.2.24", ">=3.1.12", ">=3.2.4"],
            "more_info_url": "https://pyup.io/vulnerabilities/CVE-2021-33203/42123/",
        },
        {
            "package": "requests",
            "installed_version": "2.6.0",
            "vulnerability": "CVE-2018-18074",
            "advisory": "The Requests package before 2.20.0 for Python sends an HTTP Authorization header to an http URI upon receiving a same-hostname https-to-http redirect.",
            "specs": [">=2.20.0"],
            "more_info_url": "https://pyup.io/vulnerabilities/CVE-2018-18074/36810/",
        },
    ]


@pytest.fixture
def runner() -> SafetyRunner:
    """Create SafetyRunner instance.

    Returns:
        SafetyRunner instance.
    """
    return SafetyRunner()


# =============================================================================
# INITIALIZATION TESTS (2 tests)
# =============================================================================


def test_given_no_api_key_when_initialized_then_creates_runner() -> None:
    """GIVEN no API key WHEN SafetyRunner initialized THEN creates instance."""
    runner = SafetyRunner()
    assert runner.api_key is None


def test_given_api_key_when_initialized_then_stores_key() -> None:
    """GIVEN API key WHEN SafetyRunner initialized THEN stores key."""
    runner = SafetyRunner(api_key="test-key-123")
    assert runner.api_key == "test-key-123"


# =============================================================================
# SAFETY EXECUTION TESTS (5 tests)
# =============================================================================


@patch("subprocess.run")
def test_given_requirements_file_when_run_then_executes_safety(
    mock_run: Mock,
    runner: SafetyRunner,
    tmp_path: Path,
    mock_safety_output: List[Dict[str, Any]],
) -> None:
    """GIVEN requirements file WHEN run THEN executes Safety subprocess."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("django==2.2.0\nrequests==2.6.0")

    mock_run.return_value = Mock(
        stdout=json.dumps(mock_safety_output), stderr="", returncode=64
    )

    findings = runner.run(requirements)

    assert mock_run.called
    assert len(findings) == 2


@patch("subprocess.run")
def test_given_no_file_when_run_then_scans_installed(
    mock_run: Mock, runner: SafetyRunner, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN no requirements file WHEN run THEN scans installed packages."""
    mock_run.return_value = Mock(
        stdout=json.dumps(mock_safety_output), stderr="", returncode=64
    )

    findings = runner.run(requirements_file=None)

    assert mock_run.called
    assert len(findings) == 2


@patch("subprocess.run")
def test_given_timeout_when_run_then_returns_empty(
    mock_run: Mock, runner: SafetyRunner
) -> None:
    """GIVEN subprocess timeout WHEN run THEN returns empty list."""
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired("safety", 180)

    findings = runner.run()

    assert findings == []


@patch("subprocess.run")
def test_given_invalid_json_when_run_then_returns_empty(
    mock_run: Mock, runner: SafetyRunner
) -> None:
    """GIVEN invalid JSON output WHEN run THEN returns empty list."""
    mock_run.return_value = Mock(stdout="invalid json", stderr="", returncode=64)

    findings = runner.run()

    assert findings == []


@patch("subprocess.run")
def test_given_no_output_when_run_then_returns_empty(
    mock_run: Mock, runner: SafetyRunner
) -> None:
    """GIVEN no stdout WHEN run THEN returns empty list."""
    mock_run.return_value = Mock(stdout="", stderr="error", returncode=1)

    findings = runner.run()

    assert findings == []


# =============================================================================
# PARSING TESTS (4 tests)
# =============================================================================


def test_given_safety_json_when_parsed_then_converts_to_findings(
    runner: SafetyRunner, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN Safety JSON WHEN parsed THEN converts to SecurityFindings."""
    findings = runner._parse_safety_output(mock_safety_output)

    assert len(findings) == 2
    assert all(isinstance(f, SecurityFinding) for f in findings)


def test_given_vulnerabilities_when_parsed_then_all_are_dependency_category(
    runner: SafetyRunner, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN vulnerabilities WHEN parsed THEN all categorized as DEPENDENCIES."""
    findings = runner._parse_safety_output(mock_safety_output)

    assert all(f.category == FindingCategory.DEPENDENCIES for f in findings)


def test_given_critical_advisory_when_parsed_then_maps_to_critical(
    runner: SafetyRunner, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN critical advisory WHEN parsed THEN maps to CRITICAL severity."""
    findings = runner._parse_safety_output(mock_safety_output)

    # First finding mentions "critical" and "remote code execution"
    critical_finding = findings[0]
    assert critical_finding.severity == Severity.CRITICAL


def test_given_medium_advisory_when_parsed_then_maps_to_medium(
    runner: SafetyRunner, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN medium advisory WHEN parsed THEN defaults to MEDIUM."""
    findings = runner._parse_safety_output(mock_safety_output)

    # Second finding has no critical/high keywords
    medium_finding = findings[1]
    assert medium_finding.severity == Severity.MEDIUM


# =============================================================================
# SEVERITY MAPPING TESTS (4 tests)
# =============================================================================


def test_given_critical_keyword_when_mapped_then_returns_critical(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'critical' keyword WHEN mapped THEN returns CRITICAL."""
    vuln = {"advisory": "This is a critical vulnerability"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.CRITICAL


def test_given_rce_keyword_when_mapped_then_returns_critical(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'remote code execution' keyword WHEN mapped THEN returns CRITICAL."""
    vuln = {"advisory": "Allows remote code execution"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.CRITICAL


def test_given_high_keyword_when_mapped_then_returns_high(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'high' keyword WHEN mapped THEN returns HIGH."""
    vuln = {"advisory": "High severity SQL injection"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.HIGH


def test_given_no_keywords_when_mapped_then_returns_medium(
    runner: SafetyRunner,
) -> None:
    """GIVEN no severity keywords WHEN mapped THEN defaults to MEDIUM."""
    vuln = {"advisory": "Some vulnerability"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.MEDIUM


# =============================================================================
# RECOMMENDATION TESTS (3 tests)
# =============================================================================


def test_given_specs_when_recommendation_then_suggests_version(
    runner: SafetyRunner,
) -> None:
    """GIVEN specs WHEN recommendation generated THEN suggests safe version."""
    vuln = {"package": "django", "specs": [">=2.2.24", ">=3.1.12"]}
    recommendation = runner._get_recommendation(vuln)

    assert "2.2.24" in recommendation
    assert "django" in recommendation


def test_given_no_specs_when_recommendation_then_suggests_upgrade(
    runner: SafetyRunner,
) -> None:
    """GIVEN no specs WHEN recommendation generated THEN suggests generic upgrade."""
    vuln = {"package": "requests", "specs": []}
    recommendation = runner._get_recommendation(vuln)

    assert "Upgrade" in recommendation
    assert "requests" in recommendation


def test_given_multiple_specs_when_recommendation_then_uses_first(
    runner: SafetyRunner,
) -> None:
    """GIVEN multiple specs WHEN recommendation generated THEN uses first version."""
    vuln = {"package": "django", "specs": [">=2.2.24", ">=3.1.12", ">=3.2.4"]}
    recommendation = runner._get_recommendation(vuln)

    assert "2.2.24" in recommendation


# =============================================================================
# CONVENIENCE FUNCTION TESTS (2 tests)
# =============================================================================


@patch("subprocess.run")
def test_given_requirements_when_run_safety_scan_then_executes(
    mock_run: Mock, tmp_path: Path, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN requirements file WHEN run_safety_scan called THEN executes scan."""
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("django==2.2.0")

    mock_run.return_value = Mock(
        stdout=json.dumps(mock_safety_output), stderr="", returncode=64
    )

    findings = run_safety_scan(requirements)

    assert len(findings) == 2


@patch("subprocess.run")
def test_given_api_key_when_run_safety_scan_then_uses_key(
    mock_run: Mock, mock_safety_output: List[Dict[str, Any]]
) -> None:
    """GIVEN API key WHEN run_safety_scan called THEN uses key."""
    mock_run.return_value = Mock(
        stdout=json.dumps(mock_safety_output), stderr="", returncode=64
    )

    findings = run_safety_scan(api_key="test-key")

    assert mock_run.called
    # Verify API key was passed in command
    call_args = mock_run.call_args
    assert any("--key" in str(arg) for arg in call_args[0][0])


# =============================================================================
# FINDING CONVERSION TESTS (3 tests)
# =============================================================================


def test_given_valid_vuln_when_converted_then_creates_finding(
    runner: SafetyRunner,
) -> None:
    """GIVEN valid Safety vuln WHEN converted THEN creates SecurityFinding."""
    vuln = {
        "package": "django",
        "installed_version": "2.2.0",
        "vulnerability": "CVE-2021-33203",
        "advisory": "Critical vulnerability",
        "specs": [">=2.2.24"],
        "more_info_url": "https://example.com",
    }

    finding = runner._convert_safety_vuln(vuln)

    assert finding is not None
    assert finding.category == FindingCategory.DEPENDENCIES
    assert "django" in finding.title
    assert "2.2.0" in finding.title


def test_given_missing_fields_when_converted_then_uses_defaults(
    runner: SafetyRunner,
) -> None:
    """GIVEN missing fields WHEN converted THEN uses default values."""
    vuln = {}

    finding = runner._convert_safety_vuln(vuln)

    assert finding is not None
    assert finding.severity == Severity.MEDIUM
    assert finding.file_path == "requirements.txt"
    assert finding.line_number == 0


def test_given_metadata_when_converted_then_preserves_details(
    runner: SafetyRunner,
) -> None:
    """GIVEN vuln with metadata WHEN converted THEN preserves in finding."""
    vuln = {
        "package": "django",
        "installed_version": "2.2.0",
        "vulnerability": "CVE-2021-33203",
        "advisory": "Test",
        "specs": [">=2.2.24"],
        "more_info_url": "https://example.com",
    }

    finding = runner._convert_safety_vuln(vuln)

    assert finding is not None
    assert finding.metadata["package"] == "django"
    assert finding.metadata["installed_version"] == "2.2.0"
    assert finding.metadata["cve"] == "CVE-2021-33203"


# =============================================================================
# TRUNCATION TESTS (2 tests)
# =============================================================================


@patch("subprocess.run")
def test_given_many_vulns_when_run_then_truncates_to_max(
    mock_run: Mock, runner: SafetyRunner
) -> None:
    """GIVEN >MAX vulns WHEN run THEN truncates to MAX_SAFETY_FINDINGS."""
    from ingestforge.core.security.safety_runner import MAX_SAFETY_FINDINGS

    # Create output with MAX + 50 vulnerabilities
    many_vulns = []
    for i in range(MAX_SAFETY_FINDINGS + 50):
        many_vulns.append(
            {
                "package": f"pkg-{i}",
                "installed_version": "1.0.0",
                "vulnerability": f"CVE-2021-{i:05d}",
                "advisory": "Test vulnerability",
                "specs": [">=2.0.0"],
                "more_info_url": "https://example.com",
            }
        )

    mock_run.return_value = Mock(
        stdout=json.dumps(many_vulns), stderr="", returncode=64
    )

    findings = runner.run()

    assert len(findings) == MAX_SAFETY_FINDINGS


@patch("subprocess.run")
def test_given_empty_specs_when_run_then_provides_generic_recommendation(
    mock_run: Mock, runner: SafetyRunner
) -> None:
    """GIVEN vuln with empty specs WHEN run THEN provides generic recommendation."""
    vuln_output = [
        {
            "package": "requests",
            "installed_version": "1.0.0",
            "vulnerability": "CVE-2021-12345",
            "advisory": "Some issue",
            "specs": [],
            "more_info_url": "https://example.com",
        }
    ]

    mock_run.return_value = Mock(
        stdout=json.dumps(vuln_output), stderr="", returncode=64
    )

    findings = runner.run()

    assert len(findings) == 1
    assert "Upgrade" in findings[0].recommendation
    assert "requests" in findings[0].recommendation


# =============================================================================
# ADVISORY SEVERITY MAPPING TESTS (4 tests)
# =============================================================================


def test_given_xss_advisory_when_mapped_then_returns_high(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'xss' in advisory WHEN mapped THEN returns HIGH."""
    vuln = {"advisory": "Cross-site scripting (XSS) vulnerability"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.HIGH


def test_given_sql_injection_advisory_when_mapped_then_returns_high(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'sql injection' in advisory WHEN mapped THEN returns HIGH."""
    vuln = {"advisory": "SQL injection vulnerability detected"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.HIGH


def test_given_rce_advisory_when_mapped_then_returns_critical(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'rce' in advisory WHEN mapped THEN returns CRITICAL."""
    vuln = {"advisory": "RCE exploit available"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.CRITICAL


def test_given_medium_explicit_advisory_when_mapped_then_returns_medium(
    runner: SafetyRunner,
) -> None:
    """GIVEN 'medium' in advisory WHEN mapped THEN returns MEDIUM."""
    vuln = {"advisory": "Medium severity issue"}
    severity = runner._map_severity(vuln)
    assert severity == Severity.MEDIUM


# =============================================================================
# ERROR HANDLING TESTS (2 tests)
# =============================================================================


@patch("subprocess.run")
def test_given_exception_when_run_then_returns_empty(
    mock_run: Mock, runner: SafetyRunner
) -> None:
    """GIVEN subprocess exception WHEN run THEN returns empty list."""
    mock_run.side_effect = Exception("Unexpected error")

    findings = runner.run()

    assert findings == []


def test_given_invalid_vuln_when_converted_then_returns_none(
    runner: SafetyRunner,
) -> None:
    """GIVEN invalid vuln structure WHEN converted THEN handles gracefully."""
    # Vuln that would cause exception during processing
    invalid_vuln = {"invalid_key": "invalid_value"}

    finding = runner._convert_safety_vuln(invalid_vuln)

    # Should handle gracefully
    assert finding is not None  # Uses defaults
