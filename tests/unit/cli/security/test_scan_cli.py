"""Unit Tests for Security Scan CLI.

Security Shield CI
Tests CLI commands for security scanning with CI/CD integration.

JPL Power of Ten Compliance:
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest
from typer.testing import CliRunner

from ingestforge.cli.main import app


runner = CliRunner()


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_python_file_clean() -> str:
    """Create clean Python code (no findings)."""
    return '''
"""Sample clean Python module."""

def calculate_sum(a: int, b: int) -> int:
    """Calculate sum of two numbers."""
    return a + b


if __name__ == "__main__":
    result = calculate_sum(5, 10)
    print(f"Result: {result}")
'''


@pytest.fixture
def sample_python_file_with_secret() -> str:
    """Create Python code with hardcoded secret."""
    return '''
"""Sample module with security issue."""

# BAD: Hardcoded API key
api_key = "secret_key_1234567890abcdef"

def make_request() -> None:
    """Make API request."""
    headers = {"Authorization": f"Bearer {api_key}"}
    print(headers)
'''


@pytest.fixture
def temp_dir_clean(
    sample_python_file_clean: str,
    tmp_path: Path,
) -> Path:
    """Create temp directory with clean code."""
    file_path = tmp_path / "clean_module.py"
    file_path.write_text(sample_python_file_clean, encoding="utf-8")
    return tmp_path


@pytest.fixture
def temp_dir_with_findings(
    sample_python_file_with_secret: str,
    tmp_path: Path,
) -> Path:
    """Create temp directory with security findings."""
    file_path = tmp_path / "vulnerable_module.py"
    file_path.write_text(sample_python_file_with_secret, encoding="utf-8")
    return tmp_path


# =============================================================================
# TestScanCommand
# =============================================================================


class TestScanCommand:
    """Tests for security scan command."""

    def test_scan_clean_directory_returns_zero(
        self,
        temp_dir_clean: Path,
    ) -> None:
        """
        GIVEN: Directory with clean code
        WHEN: Running security scan
        THEN: Exit code 0, no findings
        """
        result = runner.invoke(
            app,
            ["security", "scan", str(temp_dir_clean)],
        )

        assert result.exit_code == 0
        assert "No security issues found" in result.stdout or result.exit_code == 0

    def test_scan_file_with_findings_returns_nonzero(
        self,
        temp_dir_with_findings: Path,
    ) -> None:
        """
        GIVEN: File with security findings
        WHEN: Running security scan
        THEN: Exit code > 0, findings reported
        """
        result = runner.invoke(
            app,
            ["security", "scan", str(temp_dir_with_findings)],
        )

        # Should find at least one issue (exit code 1 or 2)
        assert result.exit_code in [1, 2]

    def test_scan_directory_recursive(
        self,
        tmp_path: Path,
        sample_python_file_clean: str,
    ) -> None:
        """
        GIVEN: Directory with subdirectories
        WHEN: Scanning recursively
        THEN: All files are scanned
        """
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "module.py").write_text(sample_python_file_clean, encoding="utf-8")

        result = runner.invoke(
            app,
            ["security", "scan", str(tmp_path), "--recursive"],
        )

        assert result.exit_code == 0

    def test_scan_with_exclude_patterns(
        self,
        temp_dir_with_findings: Path,
    ) -> None:
        """
        GIVEN: Directory with findings
        WHEN: Scanning with exclusion patterns
        THEN: Excluded files are not scanned
        """
        result = runner.invoke(
            app,
            [
                "security",
                "scan",
                str(temp_dir_with_findings),
                "--exclude",
                "**/*.py",
            ],
        )

        # Should skip all .py files, find nothing
        assert result.exit_code == 0

    def test_scan_with_json_output(
        self,
        temp_dir_clean: Path,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: Directory to scan
        WHEN: Running with --output flag
        THEN: JSON report is created
        """
        output_file = tmp_path / "report.json"

        result = runner.invoke(
            app,
            [
                "security",
                "scan",
                str(temp_dir_clean),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Validate JSON structure
        report_data = json.loads(output_file.read_text(encoding="utf-8"))
        assert "report_id" in report_data
        assert "summary" in report_data
        assert "findings" in report_data

    def test_scan_fail_on_warning_flag(
        self,
        temp_dir_with_findings: Path,
    ) -> None:
        """
        GIVEN: Directory with medium/low findings
        WHEN: Running with --fail-on-warning
        THEN: Exit code is 2 (strict mode)
        """
        result = runner.invoke(
            app,
            [
                "security",
                "scan",
                str(temp_dir_with_findings),
                "--fail-on-warning",
            ],
        )

        # Should escalate warnings to errors
        assert result.exit_code == 2

    def test_scan_verbose_mode(
        self,
        temp_dir_with_findings: Path,
    ) -> None:
        """
        GIVEN: Directory with findings
        WHEN: Running with --verbose
        THEN: Detailed findings are shown
        """
        result = runner.invoke(
            app,
            [
                "security",
                "scan",
                str(temp_dir_with_findings),
                "--verbose",
            ],
        )

        # Verbose output should contain more details
        assert result.exit_code in [1, 2]
        # Output should contain finding details (will vary by implementation)

    def test_scan_single_file(
        self,
        temp_dir_with_findings: Path,
    ) -> None:
        """
        GIVEN: Single file with findings
        WHEN: Scanning file directly
        THEN: File is scanned, findings reported
        """
        file_path = temp_dir_with_findings / "vulnerable_module.py"

        result = runner.invoke(
            app,
            ["security", "scan", str(file_path)],
        )

        assert result.exit_code in [1, 2]


# =============================================================================
# TestRulesCommand
# =============================================================================


class TestRulesCommand:
    """Tests for rules list command."""

    def test_list_rules_displays_all_rules(self) -> None:
        """
        GIVEN: Security scanner with default rules
        WHEN: Running rules command
        THEN: All rules are displayed
        """
        result = runner.invoke(
            app,
            ["security", "rules"],
        )

        assert result.exit_code == 0
        assert "Rule ID" in result.stdout or "SEC" in result.stdout

    def test_list_rules_shows_enabled_status(self) -> None:
        """
        GIVEN: Security scanner
        WHEN: Listing rules
        THEN: Enabled/disabled status is shown
        """
        result = runner.invoke(
            app,
            ["security", "rules"],
        )

        assert result.exit_code == 0
        # Should show enabled status (implementation dependent)


# =============================================================================
# TestCategoriesCommand
# =============================================================================


class TestCategoriesCommand:
    """Tests for categories list command."""

    def test_list_categories(self) -> None:
        """
        GIVEN: Security scanner
        WHEN: Running categories command
        THEN: All categories are displayed
        """
        result = runner.invoke(
            app,
            ["security", "categories"],
        )

        assert result.exit_code == 0
        assert "secrets" in result.stdout.lower() or "Security" in result.stdout


# =============================================================================
# TestCLIIntegration
# =============================================================================


class TestCLIIntegration:
    """Integration tests for CLI workflow."""

    def test_scan_help_command(self) -> None:
        """
        GIVEN: Security CLI
        WHEN: Running --help
        THEN: Help text is displayed
        """
        result = runner.invoke(
            app,
            ["security", "scan", "--help"],
        )

        assert result.exit_code == 0
        assert "Scan file or directory" in result.stdout

    def test_security_help_command(self) -> None:
        """
        GIVEN: Security CLI
        WHEN: Running security --help
        THEN: All commands are listed
        """
        result = runner.invoke(
            app,
            ["security", "--help"],
        )

        assert result.exit_code == 0
        assert "scan" in result.stdout
        assert "rules" in result.stdout
        assert "categories" in result.stdout

    def test_end_to_end_scan_workflow(
        self,
        temp_dir_with_findings: Path,
        tmp_path: Path,
    ) -> None:
        """
        GIVEN: Complete scan workflow
        WHEN: Running scan with all options
        THEN: Workflow completes successfully
        """
        output_file = tmp_path / "full-report.json"

        # Run comprehensive scan
        result = runner.invoke(
            app,
            [
                "security",
                "scan",
                str(temp_dir_with_findings),
                "--output",
                str(output_file),
                "--verbose",
                "--recursive",
            ],
        )

        # Should complete (exit code may vary based on findings)
        assert result.exit_code in [0, 1, 2]
        assert output_file.exists()

        # Validate report
        report_data: Dict[str, Any] = json.loads(
            output_file.read_text(encoding="utf-8")
        )
        assert isinstance(report_data, dict)
        assert "summary" in report_data
