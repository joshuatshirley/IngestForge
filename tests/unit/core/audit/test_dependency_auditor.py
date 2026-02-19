"""Tests for Dependency Integrity Auditor.

BUG002: Dependency Integrity Audit
Epic: EP-26 (Security & Compliance)

Tests cover:
- DependencyIssue dataclass
- DependencyReport dataclass
- DependencyAuditor operations
- Import scanning
- Requirements parsing
- JPL compliance
"""

from __future__ import annotations

import pytest
from pathlib import Path

from ingestforge.core.audit import (
    DependencyAuditor,
    DependencyReport,
    DependencyIssue,
    create_dependency_auditor,
    MAX_FILES_TO_SCAN,
    MAX_IMPORTS_PER_FILE,
)


class TestDependencyIssue:
    """Tests for DependencyIssue dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic issue."""
        issue = DependencyIssue(
            issue_type="missing",
            package_name="requests",
            details="Not declared",
        )
        assert issue.issue_type == "missing"
        assert issue.package_name == "requests"
        assert issue.files == []

    def test_with_files(self) -> None:
        """Test issue with file list."""
        issue = DependencyIssue(
            issue_type="missing",
            package_name="numpy",
            details="Imported but not declared",
            files=["module/file.py", "module/other.py"],
        )
        assert len(issue.files) == 2

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        issue = DependencyIssue(
            issue_type="unused",
            package_name="flask",
            details="Declared but not imported",
        )
        result = issue.to_dict()

        assert result["issue_type"] == "unused"
        assert result["package_name"] == "flask"
        assert "details" in result


class TestDependencyReport:
    """Tests for DependencyReport dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic report."""
        report = DependencyReport(
            report_id="test-001",
            timestamp="2026-02-17T12:00:00Z",
        )
        assert report.report_id == "test-001"
        assert report.files_scanned == 0
        assert len(report.missing) == 0
        assert len(report.unused) == 0

    def test_exit_code_clean(self) -> None:
        """Test exit code when no issues."""
        report = DependencyReport(
            report_id="test-clean",
            timestamp="2026-02-17T12:00:00Z",
        )
        assert report.exit_code == 0

    def test_exit_code_missing(self) -> None:
        """Test exit code with missing dependencies."""
        report = DependencyReport(
            report_id="test-missing",
            timestamp="2026-02-17T12:00:00Z",
            missing=[DependencyIssue("missing", "pkg", "details")],
        )
        assert report.exit_code == 2

    def test_exit_code_unused(self) -> None:
        """Test exit code with unused dependencies."""
        report = DependencyReport(
            report_id="test-unused",
            timestamp="2026-02-17T12:00:00Z",
            unused=[DependencyIssue("unused", "pkg", "details")],
        )
        assert report.exit_code == 1

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        report = DependencyReport(
            report_id="test-dict",
            timestamp="2026-02-17T12:00:00Z",
            declared_packages={"requests", "numpy"},
            imported_packages={"requests"},
            files_scanned=10,
            audit_duration_ms=50.5,
        )
        result = report.to_dict()

        assert result["report_id"] == "test-dict"
        assert result["declared_count"] == 2
        assert result["imported_count"] == 1
        assert result["files_scanned"] == 10
        assert "summary" in result


class TestDependencyAuditor:
    """Tests for DependencyAuditor class."""

    def test_init_default(self, tmp_path: Path) -> None:
        """Test default initialization."""
        auditor = DependencyAuditor(project_root=tmp_path)
        assert auditor._project_root == tmp_path

    def test_init_validates_path(self, tmp_path: Path) -> None:
        """Test that init validates project root exists.

        JPL Rule #5: Assert preconditions.
        """
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(AssertionError):
            DependencyAuditor(project_root=non_existent)

    def test_normalize_package_name(self, tmp_path: Path) -> None:
        """Test package name normalization."""
        auditor = DependencyAuditor(project_root=tmp_path)

        assert auditor._normalize_package_name("Requests") == "requests"
        assert auditor._normalize_package_name("scikit_learn") == "scikit-learn"
        assert auditor._normalize_package_name("PyYAML") == "pyyaml"

    def test_extract_package_name(self, tmp_path: Path) -> None:
        """Test extracting package name from requirement spec."""
        auditor = DependencyAuditor(project_root=tmp_path)

        assert auditor._extract_package_name("requests>=2.28") == "requests"
        assert auditor._extract_package_name("numpy==1.24.0") == "numpy"
        assert auditor._extract_package_name("flask[async]") == "flask"
        assert auditor._extract_package_name("django~=4.0") == "django"

    def test_is_internal_or_stdlib(self, tmp_path: Path) -> None:
        """Test stdlib and internal package detection."""
        auditor = DependencyAuditor(project_root=tmp_path)

        # Stdlib
        assert auditor._is_internal_or_stdlib("os") is True
        assert auditor._is_internal_or_stdlib("json") is True
        assert auditor._is_internal_or_stdlib("pathlib") is True

        # Internal
        assert auditor._is_internal_or_stdlib("ingestforge") is True

        # External
        assert auditor._is_internal_or_stdlib("requests") is False
        assert auditor._is_internal_or_stdlib("numpy") is False

    def test_get_top_level_package(self, tmp_path: Path) -> None:
        """Test extracting top-level package from module path."""
        auditor = DependencyAuditor(project_root=tmp_path)

        assert auditor._get_top_level_package("requests") == "requests"
        assert auditor._get_top_level_package("requests.auth") == "requests"
        assert auditor._get_top_level_package("numpy.core.array") == "numpy"

    def test_is_dev_dependency(self, tmp_path: Path) -> None:
        """Test dev dependency detection."""
        auditor = DependencyAuditor(project_root=tmp_path)

        assert auditor._is_dev_dependency("pytest") is True
        assert auditor._is_dev_dependency("black") is True
        assert auditor._is_dev_dependency("mypy") is True
        assert auditor._is_dev_dependency("requests") is False

    def test_is_known_alias(self, tmp_path: Path) -> None:
        """Test known import alias detection."""
        auditor = DependencyAuditor(project_root=tmp_path)
        packages = {"pillow", "pyyaml", "scikit-learn"}

        assert auditor._is_known_alias("PIL", packages) is True
        assert auditor._is_known_alias("yaml", packages) is True
        assert auditor._is_known_alias("sklearn", packages) is True
        assert auditor._is_known_alias("unknown", packages) is False


class TestRequirementsParsing:
    """Tests for requirements.txt parsing."""

    def test_parse_requirements_txt(self, tmp_path: Path) -> None:
        """Test parsing requirements.txt file."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text(
            """
# This is a comment
requests>=2.28.0
numpy==1.24.0
flask[async]~=2.0

# Another comment
pandas
-r other-requirements.txt
"""
        )

        auditor = DependencyAuditor(project_root=tmp_path)
        packages = auditor._parse_requirements_txt(req_file)

        assert "requests" in packages
        assert "numpy" in packages
        assert "flask" in packages
        assert "pandas" in packages
        # Should not include comments or -r lines
        assert len(packages) == 4

    def test_parse_empty_requirements(self, tmp_path: Path) -> None:
        """Test parsing empty requirements.txt."""
        req_file = tmp_path / "requirements.txt"
        req_file.write_text("# Just a comment\n")

        auditor = DependencyAuditor(project_root=tmp_path)
        packages = auditor._parse_requirements_txt(req_file)

        assert len(packages) == 0


class TestImportScanning:
    """Tests for Python import scanning."""

    def test_extract_imports_from_file(self, tmp_path: Path) -> None:
        """Test extracting imports from a Python file."""
        py_file = tmp_path / "test_module.py"
        py_file.write_text(
            """
import os
import json
from pathlib import Path
from typing import Optional

import requests
from numpy import array
from flask.views import View
"""
        )

        auditor = DependencyAuditor(project_root=tmp_path)
        imports = auditor._extract_imports_from_file(py_file)

        assert "os" in imports
        assert "json" in imports
        assert "pathlib" in imports
        assert "typing" in imports
        assert "requests" in imports
        assert "numpy" in imports
        assert "flask.views" in imports

    def test_extract_imports_invalid_syntax(self, tmp_path: Path) -> None:
        """Test handling files with syntax errors."""
        py_file = tmp_path / "invalid.py"
        py_file.write_text("this is not valid python {{{")

        auditor = DependencyAuditor(project_root=tmp_path)
        imports = auditor._extract_imports_from_file(py_file)

        assert imports == []  # Should return empty, not raise

    def test_collect_python_files(self, tmp_path: Path) -> None:
        """Test collecting Python files iteratively.

        JPL Rule #1: No recursion.
        """
        # Create directory structure
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "module.py").write_text("# module")
        (tmp_path / "pkg" / "sub").mkdir()
        (tmp_path / "pkg" / "sub" / "nested.py").write_text("# nested")
        (tmp_path / "other.py").write_text("# other")

        auditor = DependencyAuditor(project_root=tmp_path)
        files = auditor._collect_python_files(tmp_path)

        assert len(files) == 3
        assert all(f.suffix == ".py" for f in files)


class TestFullAudit:
    """Tests for full audit operations."""

    def test_audit_simple_project(self, tmp_path: Path) -> None:
        """Test auditing a simple project structure."""
        # Create requirements.txt
        (tmp_path / "requirements.txt").write_text("requests>=2.28\n")

        # Create source directory
        src = tmp_path / "ingestforge"
        src.mkdir()
        (src / "__init__.py").write_text("")
        (src / "module.py").write_text("import requests\n")

        auditor = DependencyAuditor(project_root=tmp_path)
        report = auditor.audit()

        assert report.files_scanned > 0
        assert "requests" in report.declared_packages
        assert report.exit_code == 0  # All declared

    def test_audit_missing_dependency(self, tmp_path: Path) -> None:
        """Test detecting missing dependencies."""
        # Create requirements.txt (empty)
        (tmp_path / "requirements.txt").write_text("")

        # Create source with import
        src = tmp_path / "ingestforge"
        src.mkdir()
        (src / "module.py").write_text("import nonexistent_package\n")

        auditor = DependencyAuditor(project_root=tmp_path)
        report = auditor.audit()

        # Should detect missing dependency
        assert len(report.missing) >= 1
        assert report.exit_code == 2


class TestFactoryFunction:
    """Tests for create_dependency_auditor factory."""

    def test_create_dependency_auditor(self, tmp_path: Path) -> None:
        """Test factory creates auditor."""
        auditor = create_dependency_auditor(project_root=tmp_path)
        assert isinstance(auditor, DependencyAuditor)

    def test_create_without_path(self) -> None:
        """Test factory with default path."""
        auditor = create_dependency_auditor()
        assert isinstance(auditor, DependencyAuditor)
        assert auditor._project_root.exists()


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_fixed_bounds_defined(self) -> None:
        """Test that fixed bounds are defined.

        JPL Rule #2: Fixed upper bounds.
        """
        assert MAX_FILES_TO_SCAN == 2000
        assert MAX_IMPORTS_PER_FILE == 200

    def test_iterative_file_collection(self, tmp_path: Path) -> None:
        """Test that file collection is iterative, not recursive.

        JPL Rule #1: No recursion.
        """
        # Create nested structure
        for i in range(5):
            nested = tmp_path / f"level{i}"
            nested.mkdir(parents=True)
            (nested / "file.py").write_text("# file")

        auditor = DependencyAuditor(project_root=tmp_path)
        files = auditor._collect_python_files(tmp_path)

        # Should find all files without stack overflow
        assert len(files) == 5

    def test_all_functions_return_explicit_values(self, tmp_path: Path) -> None:
        """Test functions return explicit values.

        JPL Rule #7: Check all return values.
        """
        auditor = DependencyAuditor(project_root=tmp_path)

        # audit returns DependencyReport
        (tmp_path / "requirements.txt").write_text("")
        report = auditor.audit()
        assert isinstance(report, DependencyReport)

        # Helper functions return explicit values
        name = auditor._normalize_package_name("Test")
        assert isinstance(name, str)

        is_stdlib = auditor._is_internal_or_stdlib("os")
        assert isinstance(is_stdlib, bool)
