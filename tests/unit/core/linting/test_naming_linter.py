"""Tests for Naming Convention Linter.

Naming Convention Linter
Epic: EP-26 (Security & Compliance)
"""

import tempfile
from pathlib import Path

import pytest

from ingestforge.core.linting import (
    NamingLinter,
    NamingViolation,
    LintReport,
    ViolationSeverity,
    create_linter,
    MAX_VIOLATIONS,
)
from ingestforge.core.linting.naming_linter import ViolationCategory


class TestNamingViolation:
    """Tests for NamingViolation dataclass."""

    def test_create_violation(self) -> None:
        """Test creating a naming violation."""
        violation = NamingViolation(
            violation_id="test-123",
            category=ViolationCategory.FUNCTION,
            severity=ViolationSeverity.ERROR,
            rule_id="NAME001",
            name="badName",
            expected_pattern="snake_case",
            file_path="/path/to/file.py",
            line_number=10,
            message="Function 'badName' should be snake_case",
            recommendation="Rename to snake_case",
        )
        assert violation.violation_id == "test-123"
        assert violation.category == ViolationCategory.FUNCTION
        assert violation.severity == ViolationSeverity.ERROR

    def test_violation_to_dict(self) -> None:
        """Test converting violation to dictionary."""
        violation = NamingViolation(
            violation_id="test-123",
            category=ViolationCategory.CLASS,
            severity=ViolationSeverity.ERROR,
            rule_id="NAME002",
            name="my_class",
            expected_pattern="PascalCase",
            file_path="/path/to/file.py",
            line_number=5,
            message="Class should be PascalCase",
            recommendation="Rename to PascalCase",
        )
        d = violation.to_dict()
        assert d["category"] == "class"
        assert d["severity"] == "error"
        assert d["rule_id"] == "NAME002"

    def test_violation_immutable(self) -> None:
        """Test that violations are immutable."""
        violation = NamingViolation(
            violation_id="test-123",
            category=ViolationCategory.FUNCTION,
            severity=ViolationSeverity.ERROR,
            rule_id="NAME001",
            name="badName",
            expected_pattern="snake_case",
            file_path="/path/to/file.py",
            line_number=10,
            message="Test",
            recommendation="Test",
        )
        with pytest.raises(AttributeError):
            violation.name = "newName"


class TestLintReport:
    """Tests for LintReport dataclass."""

    def test_create_report(self) -> None:
        """Test creating a lint report."""
        report = LintReport(
            report_id="report-123",
            scan_path="/path/to/scan",
        )
        assert report.report_id == "report-123"
        assert report.violations == []
        assert report.files_scanned == 0

    def test_add_violation(self) -> None:
        """Test adding violations to report."""
        report = LintReport(report_id="test", scan_path="/test")
        violation = NamingViolation(
            violation_id="v1",
            category=ViolationCategory.FUNCTION,
            severity=ViolationSeverity.ERROR,
            rule_id="NAME001",
            name="badName",
            expected_pattern="snake_case",
            file_path="/test.py",
            line_number=1,
            message="Test",
            recommendation="Test",
        )
        result = report.add_violation(violation)
        assert result is True
        assert len(report.violations) == 1

    def test_error_count(self) -> None:
        """Test error count property."""
        report = LintReport(report_id="test", scan_path="/test")

        # Add error
        report.add_violation(
            NamingViolation(
                violation_id="v1",
                category=ViolationCategory.FUNCTION,
                severity=ViolationSeverity.ERROR,
                rule_id="NAME001",
                name="bad",
                expected_pattern="test",
                file_path="/test.py",
                line_number=1,
                message="Test",
                recommendation="Test",
            )
        )

        # Add warning
        report.add_violation(
            NamingViolation(
                violation_id="v2",
                category=ViolationCategory.VARIABLE,
                severity=ViolationSeverity.WARNING,
                rule_id="NAME004",
                name="bad",
                expected_pattern="test",
                file_path="/test.py",
                line_number=2,
                message="Test",
                recommendation="Test",
            )
        )

        assert report.error_count == 1
        assert report.warning_count == 1
        assert report.info_count == 0

    def test_exit_code_clean(self) -> None:
        """Test exit code when clean."""
        report = LintReport(report_id="test", scan_path="/test")
        assert report.exit_code == 0

    def test_exit_code_warnings(self) -> None:
        """Test exit code with warnings."""
        report = LintReport(report_id="test", scan_path="/test")
        report.add_violation(
            NamingViolation(
                violation_id="v1",
                category=ViolationCategory.VARIABLE,
                severity=ViolationSeverity.WARNING,
                rule_id="NAME004",
                name="bad",
                expected_pattern="test",
                file_path="/test.py",
                line_number=1,
                message="Test",
                recommendation="Test",
            )
        )
        assert report.exit_code == 1

    def test_exit_code_errors(self) -> None:
        """Test exit code with errors."""
        report = LintReport(report_id="test", scan_path="/test")
        report.add_violation(
            NamingViolation(
                violation_id="v1",
                category=ViolationCategory.FUNCTION,
                severity=ViolationSeverity.ERROR,
                rule_id="NAME001",
                name="bad",
                expected_pattern="test",
                file_path="/test.py",
                line_number=1,
                message="Test",
                recommendation="Test",
            )
        )
        assert report.exit_code == 2

    def test_report_to_dict(self) -> None:
        """Test converting report to dictionary."""
        report = LintReport(report_id="test", scan_path="/test")
        report.files_scanned = 5
        d = report.to_dict()
        assert d["report_id"] == "test"
        assert d["files_scanned"] == 5
        assert "summary" in d


class TestNamingLinter:
    """Tests for NamingLinter class."""

    def test_create_linter(self) -> None:
        """Test creating a linter."""
        linter = create_linter()
        assert linter is not None
        assert len(linter.get_rules()) > 0

    def test_lint_valid_file(self) -> None:
        """Test linting a valid Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def my_function():
    pass

class MyClass:
    pass

MAX_VALUE = 100
my_variable = 42
"""
            )
            f.flush()
            path = Path(f.name)

        try:
            linter = create_linter()
            violations = linter.lint_file(path)
            assert len(violations) == 0
        finally:
            path.unlink()

    def test_lint_invalid_function_name(self) -> None:
        """Test detecting invalid function name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
def badFunctionName():
    pass
"""
            )
            f.flush()
            path = Path(f.name)

        try:
            linter = create_linter()
            violations = linter.lint_file(path)
            assert len(violations) >= 1
            assert any(v.rule_id == "NAME001" for v in violations)
        finally:
            path.unlink()

    def test_lint_invalid_class_name(self) -> None:
        """Test detecting invalid class name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
class my_class:
    pass
"""
            )
            f.flush()
            path = Path(f.name)

        try:
            linter = create_linter()
            violations = linter.lint_file(path)
            assert len(violations) >= 1
            assert any(v.rule_id == "NAME002" for v in violations)
        finally:
            path.unlink()

    def test_lint_dunder_methods_allowed(self) -> None:
        """Test that dunder methods are allowed."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
class MyClass:
    def __init__(self):
        pass

    def __str__(self):
        return ""
"""
            )
            f.flush()
            path = Path(f.name)

        try:
            linter = create_linter()
            violations = linter.lint_file(path)
            # Should not flag dunder methods
            assert not any(v.name.startswith("__") for v in violations)
        finally:
            path.unlink()

    def test_lint_private_methods_allowed(self) -> None:
        """Test that private methods with underscore prefix are checked correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                """
class MyClass:
    def _private_method(self):
        pass

    def __double_private(self):
        pass
"""
            )
            f.flush()
            path = Path(f.name)

        try:
            linter = create_linter()
            violations = linter.lint_file(path)
            # Valid private methods should not be flagged
            assert len(violations) == 0
        finally:
            path.unlink()

    def test_lint_nonexistent_file(self) -> None:
        """Test linting a nonexistent file."""
        linter = create_linter()
        violations = linter.lint_file(Path("/nonexistent/file.py"))
        assert violations == []

    def test_lint_non_python_file(self) -> None:
        """Test linting a non-Python file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("This is not Python")
            f.flush()
            path = Path(f.name)

        try:
            linter = create_linter()
            violations = linter.lint_file(path)
            assert violations == []
        finally:
            path.unlink()

    def test_lint_directory(self) -> None:
        """Test linting a directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid Python file
            valid_file = Path(tmpdir) / "valid.py"
            valid_file.write_text(
                """
def my_function():
    pass
"""
            )

            # Create an invalid Python file
            invalid_file = Path(tmpdir) / "invalid.py"
            invalid_file.write_text(
                """
def badFunction():
    pass
"""
            )

            linter = create_linter()
            report = linter.lint_directory(Path(tmpdir))

            assert report.files_scanned == 2
            assert len(report.violations) >= 1

    def test_enable_disable_rule(self) -> None:
        """Test enabling and disabling rules."""
        linter = create_linter()

        # Disable rule
        result = linter.disable_rule("NAME001")
        assert result is True

        rules = linter.get_rules()
        name001 = next(r for r in rules if r.rule_id == "NAME001")
        assert name001.enabled is False

        # Enable rule
        result = linter.enable_rule("NAME001")
        assert result is True

        rules = linter.get_rules()
        name001 = next(r for r in rules if r.rule_id == "NAME001")
        assert name001.enabled is True

    def test_disable_nonexistent_rule(self) -> None:
        """Test disabling a nonexistent rule."""
        linter = create_linter()
        result = linter.disable_rule("NONEXISTENT")
        assert result is False


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_max_violations(self) -> None:
        """Test that MAX_VIOLATIONS is enforced."""
        report = LintReport(report_id="test", scan_path="/test")

        # Add violations up to limit
        for i in range(MAX_VIOLATIONS + 10):
            violation = NamingViolation(
                violation_id=f"v{i}",
                category=ViolationCategory.FUNCTION,
                severity=ViolationSeverity.ERROR,
                rule_id="NAME001",
                name=f"bad{i}",
                expected_pattern="test",
                file_path="/test.py",
                line_number=i,
                message="Test",
                recommendation="Test",
            )
            report.add_violation(violation)

        assert len(report.violations) == MAX_VIOLATIONS
        assert report.truncated is True

    def test_rule_5_preconditions(self) -> None:
        """Test that preconditions are checked."""
        with pytest.raises(AssertionError):
            NamingLinter(max_file_size=0)

        with pytest.raises(AssertionError):
            NamingLinter(max_file_size=-1)

    def test_rule_5_directory_exists(self) -> None:
        """Test that directory existence is checked."""
        linter = create_linter()
        with pytest.raises(AssertionError):
            linter.lint_directory(Path("/nonexistent/directory"))

    def test_rule_7_lint_file_returns_list(self) -> None:
        """Test that lint_file always returns a list."""
        linter = create_linter()

        # Nonexistent file
        result = linter.lint_file(Path("/nonexistent.py"))
        assert isinstance(result, list)

        # Non-Python file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = Path(f.name)
        try:
            result = linter.lint_file(path)
            assert isinstance(result, list)
        finally:
            path.unlink()

    def test_rule_9_type_hints(self) -> None:
        """Test that functions have type hints."""
        import inspect

        sig = inspect.signature(NamingLinter.lint_file)
        assert sig.return_annotation is not inspect.Parameter.empty

        sig = inspect.signature(NamingLinter.lint_directory)
        assert sig.return_annotation is not inspect.Parameter.empty


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_linter_default(self) -> None:
        """Test create_linter with defaults."""
        linter = create_linter()
        assert isinstance(linter, NamingLinter)
        assert len(linter.get_rules()) > 0

    def test_create_linter_custom_size(self) -> None:
        """Test create_linter with custom file size."""
        linter = create_linter(max_file_size=1000)
        assert linter._max_file_size == 1000
