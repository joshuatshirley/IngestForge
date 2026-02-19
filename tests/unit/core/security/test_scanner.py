"""Tests for Security Scanner.

Security Shield CI Pipeline
Tests security scanning, finding detection, and CI integration.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ingestforge.core.security.scanner import (
    SecurityScanner,
    SecurityFinding,
    SecurityReport,
    Severity,
    FindingCategory,
    create_scanner,
    MAX_FINDINGS,
    MAX_FILE_SIZE,
)


class TestSecurityFinding:
    """Tests for SecurityFinding dataclass."""

    def test_create_finding(self) -> None:
        """Test creating a finding."""
        finding = SecurityFinding(
            finding_id="test-123",
            category=FindingCategory.SECRETS,
            severity=Severity.HIGH,
            title="Hardcoded API Key",
            description="API key found in source",
            file_path="/path/to/file.py",
            line_number=42,
            line_content="api_key = 'abc123'",
            recommendation="Use environment variables",
            rule_id="SEC001",
        )

        assert finding.finding_id == "test-123"
        assert finding.severity == Severity.HIGH
        assert finding.line_number == 42

    def test_finding_to_dict(self) -> None:
        """Test converting finding to dictionary."""
        finding = SecurityFinding(
            finding_id="test-456",
            category=FindingCategory.INJECTION,
            severity=Severity.CRITICAL,
            title="SQL Injection",
            description="SQL injection vulnerability",
            file_path="app.py",
            line_number=100,
            line_content="query = f'SELECT * FROM users WHERE id={user_id}'",
            recommendation="Use parameterized queries",
            rule_id="SEC010",
        )

        data = finding.to_dict()

        assert data["finding_id"] == "test-456"
        assert data["category"] == "injection"
        assert data["severity"] == "critical"
        assert data["rule_id"] == "SEC010"

    def test_finding_immutable(self) -> None:
        """Test that finding is frozen/immutable."""
        finding = SecurityFinding(
            finding_id="test",
            category=FindingCategory.CRYPTO,
            severity=Severity.MEDIUM,
            title="Test",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="test",
            rule_id="TEST",
        )

        with pytest.raises(AttributeError):
            finding.severity = Severity.LOW  # type: ignore


class TestSecurityReport:
    """Tests for SecurityReport class."""

    def test_create_report(self) -> None:
        """Test creating a report."""
        report = SecurityReport(
            report_id="report-123",
            scan_path="/path/to/scan",
        )

        assert report.report_id == "report-123"
        assert len(report.findings) == 0
        assert report.exit_code == 0

    def test_add_finding(self) -> None:
        """Test adding findings to report."""
        report = SecurityReport(report_id="test", scan_path="/test")

        finding = SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.HIGH,
            title="Test",
            description="Test",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="test",
            rule_id="TEST",
        )

        result = report.add_finding(finding)

        assert result is True
        assert len(report.findings) == 1

    def test_severity_counts(self) -> None:
        """Test severity counting."""
        report = SecurityReport(report_id="test", scan_path="/test")

        # Add various severity findings
        for severity in [
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.HIGH,
            Severity.MEDIUM,
        ]:
            finding = SecurityFinding(
                finding_id=f"f-{severity.value}",
                category=FindingCategory.SECRETS,
                severity=severity,
                title="Test",
                description="Test",
                file_path="test.py",
                line_number=1,
                line_content="test",
                recommendation="test",
                rule_id="TEST",
            )
            report.add_finding(finding)

        assert report.critical_count == 1
        assert report.high_count == 2
        assert report.medium_count == 1

    def test_exit_code_clean(self) -> None:
        """Test exit code for clean scan."""
        report = SecurityReport(report_id="test", scan_path="/test")

        assert report.exit_code == 0

    def test_exit_code_with_info(self) -> None:
        """Test exit code with info findings only."""
        report = SecurityReport(report_id="test", scan_path="/test")

        finding = SecurityFinding(
            finding_id="f1",
            category=FindingCategory.CONFIG,
            severity=Severity.INFO,
            title="Info",
            description="Info",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="test",
            rule_id="TEST",
        )
        report.add_finding(finding)

        assert report.exit_code == 0

    def test_exit_code_with_warning(self) -> None:
        """Test exit code with medium severity."""
        report = SecurityReport(report_id="test", scan_path="/test")

        finding = SecurityFinding(
            finding_id="f1",
            category=FindingCategory.CONFIG,
            severity=Severity.MEDIUM,
            title="Warning",
            description="Warning",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="test",
            rule_id="TEST",
        )
        report.add_finding(finding)

        assert report.exit_code == 1

    def test_exit_code_with_error(self) -> None:
        """Test exit code with high severity."""
        report = SecurityReport(report_id="test", scan_path="/test")

        finding = SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.HIGH,
            title="Error",
            description="Error",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="test",
            rule_id="TEST",
        )
        report.add_finding(finding)

        assert report.exit_code == 2

    def test_exit_code_with_critical(self) -> None:
        """Test exit code with critical severity."""
        report = SecurityReport(report_id="test", scan_path="/test")

        finding = SecurityFinding(
            finding_id="f1",
            category=FindingCategory.SECRETS,
            severity=Severity.CRITICAL,
            title="Critical",
            description="Critical",
            file_path="test.py",
            line_number=1,
            line_content="test",
            recommendation="test",
            rule_id="TEST",
        )
        report.add_finding(finding)

        assert report.exit_code == 2

    def test_report_to_dict(self) -> None:
        """Test converting report to dictionary."""
        report = SecurityReport(
            report_id="test",
            scan_path="/test",
            files_scanned=10,
        )

        data = report.to_dict()

        assert data["report_id"] == "test"
        assert data["files_scanned"] == 10
        assert "summary" in data
        assert data["summary"]["exit_code"] == 0


class TestSecurityScanner:
    """Tests for SecurityScanner class."""

    def test_create_scanner(self) -> None:
        """Test creating a scanner."""
        scanner = SecurityScanner()

        assert scanner is not None
        assert len(scanner.get_rules()) > 0

    def test_scan_file_not_found(self) -> None:
        """Test scanning non-existent file."""
        scanner = SecurityScanner()

        findings = scanner.scan_file(Path("/nonexistent/file.py"))

        assert len(findings) == 0

    def test_scan_file_detects_hardcoded_password(self) -> None:
        """Test detecting hardcoded password."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text('password = "secret123"\n')

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) >= 1
            assert any(f.rule_id == "SEC002" for f in findings)
            assert any(f.severity == Severity.CRITICAL for f in findings)

    def test_scan_file_detects_api_key(self) -> None:
        """Test detecting hardcoded API key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.py"
            test_file.write_text('api_key = "abcdefghijklmnopqrstuvwxyz123456"\n')

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) >= 1
            assert any(f.category == FindingCategory.SECRETS for f in findings)

    def test_scan_file_detects_aws_key(self) -> None:
        """Test detecting AWS access key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "aws.py"
            test_file.write_text('aws_key = "AKIAIOSFODNN7EXAMPLE"\n')

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) >= 1
            assert any(f.rule_id == "SEC003" for f in findings)

    def test_scan_file_detects_private_key(self) -> None:
        """Test detecting private key."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "key.py"
            test_file.write_text('key = """-----BEGIN PRIVATE KEY-----\nMIIE...\n"""\n')

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) >= 1
            assert any(f.rule_id == "SEC004" for f in findings)

    def test_scan_file_detects_eval(self) -> None:
        """Test detecting eval() usage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "unsafe.py"
            test_file.write_text("result = eval(user_input)\n")

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) >= 1
            assert any(f.rule_id == "SEC021" for f in findings)

    def test_scan_file_detects_debug_mode(self) -> None:
        """Test detecting debug mode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "config.py"
            test_file.write_text("DEBUG = True\n")

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) >= 1
            assert any(f.rule_id == "SEC040" for f in findings)

    def test_scan_file_clean(self) -> None:
        """Test scanning clean file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "clean.py"
            test_file.write_text('import os\n\ndef hello():\n    print("Hello")\n')

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            # May have zero findings or just info-level
            high_findings = [
                f for f in findings if f.severity in [Severity.HIGH, Severity.CRITICAL]
            ]
            assert len(high_findings) == 0

    def test_scan_file_skips_unsupported_extension(self) -> None:
        """Test that unsupported extensions are skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "data.png"
            test_file.write_bytes(b"\x89PNG\r\n\x1a\n")

            scanner = SecurityScanner()
            findings = scanner.scan_file(test_file)

            assert len(findings) == 0

    def test_scan_directory(self) -> None:
        """Test scanning a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            (Path(tmpdir) / "clean.py").write_text('print("hello")\n')
            (Path(tmpdir) / "secrets.py").write_text('password = "secret"\n')

            scanner = SecurityScanner()
            report = scanner.scan_directory(Path(tmpdir))

            assert report.files_scanned >= 1
            assert len(report.findings) >= 1

    def test_scan_directory_recursive(self) -> None:
        """Test recursive directory scanning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            subdir = base / "subdir"
            subdir.mkdir()

            (base / "main.py").write_text('print("main")\n')
            (subdir / "sub.py").write_text('password = "hidden"\n')

            scanner = SecurityScanner()
            report = scanner.scan_directory(base, recursive=True)

            assert report.files_scanned >= 2
            assert len(report.findings) >= 1

    def test_scan_directory_non_recursive(self) -> None:
        """Test non-recursive directory scanning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            subdir = base / "subdir"
            subdir.mkdir()

            (base / "main.py").write_text('print("main")\n')
            (subdir / "sub.py").write_text('password = "hidden"\n')

            scanner = SecurityScanner()
            report = scanner.scan_directory(base, recursive=False)

            # Should not find the subdirectory file
            assert report.files_scanned == 1

    def test_enable_disable_rule(self) -> None:
        """Test enabling and disabling rules."""
        scanner = SecurityScanner()

        # Disable a rule
        result = scanner.disable_rule("SEC002")
        assert result is True

        # Check it's disabled
        rules = scanner.get_rules()
        sec002 = next(r for r in rules if r.rule_id == "SEC002")
        assert sec002.enabled is False

        # Re-enable
        scanner.enable_rule("SEC002")
        rules = scanner.get_rules()
        sec002 = next(r for r in rules if r.rule_id == "SEC002")
        assert sec002.enabled is True

    def test_enable_nonexistent_rule(self) -> None:
        """Test enabling non-existent rule."""
        scanner = SecurityScanner()

        result = scanner.enable_rule("NONEXISTENT")

        assert result is False


class TestBoundsEnforcement:
    """Tests for JPL Rule #2 bounds enforcement."""

    def test_max_findings_enforced(self) -> None:
        """Test that MAX_FINDINGS is enforced."""
        report = SecurityReport(report_id="test", scan_path="/test")

        # Try to add more than MAX_FINDINGS
        for i in range(MAX_FINDINGS + 10):
            finding = SecurityFinding(
                finding_id=f"f{i}",
                category=FindingCategory.CONFIG,
                severity=Severity.INFO,
                title="Test",
                description="Test",
                file_path="test.py",
                line_number=i,
                line_content="test",
                recommendation="test",
                rule_id="TEST",
            )
            report.add_finding(finding)

        assert len(report.findings) == MAX_FINDINGS
        assert report.truncated is True

    def test_max_file_size_constant(self) -> None:
        """Test that MAX_FILE_SIZE constant exists."""
        assert MAX_FILE_SIZE > 0

    def test_scanner_respects_max_file_size(self) -> None:
        """Test scanner max file size parameter."""
        scanner = SecurityScanner(max_file_size=100)

        # Internal limit should be set
        assert scanner._max_file_size == 100


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_scanner_factory(self) -> None:
        """Test create_scanner factory function."""
        scanner = create_scanner()

        assert isinstance(scanner, SecurityScanner)

    def test_create_scanner_with_max_size(self) -> None:
        """Test create_scanner with custom max size."""
        scanner = create_scanner(max_file_size=5000)

        assert scanner._max_file_size == 5000


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_bounds_constants_exist(self) -> None:
        """Test that bound constants are defined."""
        from ingestforge.core.security.scanner import (
            MAX_FINDINGS,
            MAX_FILE_SIZE,
            MAX_LINE_LENGTH,
            MAX_FILES_PER_SCAN,
        )

        assert MAX_FINDINGS > 0
        assert MAX_FILE_SIZE > 0
        assert MAX_LINE_LENGTH > 0
        assert MAX_FILES_PER_SCAN > 0

    def test_rule_5_invalid_max_file_size(self) -> None:
        """Test that invalid max_file_size fails assertion."""
        with pytest.raises(AssertionError):
            SecurityScanner(max_file_size=0)

    def test_rule_7_scan_file_returns_list(self) -> None:
        """Test that scan_file always returns a list."""
        scanner = SecurityScanner()

        result = scanner.scan_file(Path("/nonexistent"))

        assert isinstance(result, list)

    def test_rule_7_scan_directory_returns_report(self) -> None:
        """Test that scan_directory always returns report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = SecurityScanner()

            result = scanner.scan_directory(Path(tmpdir))

            assert isinstance(result, SecurityReport)

    def test_rule_9_type_hints(self) -> None:
        """Test that key methods have type hints."""
        import inspect

        scanner = SecurityScanner()

        # Check scan_file
        sig = inspect.signature(scanner.scan_file)
        # Just verify it has a return annotation
        assert sig.return_annotation is not inspect.Parameter.empty

        # Check scan_directory
        sig = inspect.signature(scanner.scan_directory)
        assert sig.return_annotation == SecurityReport
