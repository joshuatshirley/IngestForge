"""Tests for runtime dependency guard module.

Tests dependency checking and installation guidance."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from ingestforge.core.env_check import (
    Dependency,
    DependencyCategory,
    DependencyChecker,
    DependencyCheckResult,
    DependencyStatus,
    EnvironmentReport,
    check_environment,
    check_ocr_available,
    check_pdf_tools_available,
    get_missing_instructions,
    DEFAULT_DEPENDENCIES,
)

# Dependency tests


class TestDependency:
    """Tests for Dependency dataclass."""

    def test_dependency_creation(self) -> None:
        """Test creating a dependency."""
        dep = Dependency(
            name="Test Tool",
            command="test-cmd",
            category=DependencyCategory.OCR,
            required=True,
        )

        assert dep.name == "Test Tool"
        assert dep.command == "test-cmd"
        assert dep.category == DependencyCategory.OCR
        assert dep.required is True

    def test_dependency_with_instructions(self) -> None:
        """Test dependency with install instructions."""
        dep = Dependency(
            name="Test",
            command="test",
            category=DependencyCategory.BUILD,
            install_instructions={
                "windows": "choco install test",
                "macos": "brew install test",
                "linux": "apt install test",
            },
        )

        assert "choco" in dep.get_install_instruction("windows")
        assert "brew" in dep.get_install_instruction("macos")
        assert "apt" in dep.get_install_instruction("linux")

    def test_get_install_instruction_default(self) -> None:
        """Test fallback to default instruction."""
        dep = Dependency(
            name="Test",
            command="test",
            category=DependencyCategory.BUILD,
            install_instructions={"default": "Install Test manually"},
        )

        assert "Install Test manually" in dep.get_install_instruction("unknown_os")

    def test_get_install_instruction_darwin_maps_to_macos(self) -> None:
        """Test darwin maps to macos key."""
        dep = Dependency(
            name="Test",
            command="test",
            category=DependencyCategory.BUILD,
            install_instructions={"macos": "brew install test"},
        )

        # darwin should map to macos
        instruction = dep.get_install_instruction("darwin")
        assert "brew" in instruction


# DependencyCheckResult tests


class TestDependencyCheckResult:
    """Tests for DependencyCheckResult dataclass."""

    def test_result_available(self) -> None:
        """Test result for available dependency."""
        dep = Dependency(name="Test", command="test", category=DependencyCategory.OCR)
        result = DependencyCheckResult(
            dependency=dep,
            status=DependencyStatus.AVAILABLE,
            version="1.2.3",
            path=Path("/usr/bin/test"),
        )

        assert result.status == DependencyStatus.AVAILABLE
        assert result.version == "1.2.3"
        assert result.error is None

    def test_result_missing(self) -> None:
        """Test result for missing dependency."""
        dep = Dependency(name="Test", command="test", category=DependencyCategory.OCR)
        result = DependencyCheckResult(
            dependency=dep,
            status=DependencyStatus.MISSING,
            error="Not found in PATH",
        )

        assert result.status == DependencyStatus.MISSING
        assert result.error is not None


# EnvironmentReport tests


class TestEnvironmentReport:
    """Tests for EnvironmentReport dataclass."""

    def test_report_all_available(self) -> None:
        """Test report with all dependencies available."""
        dep1 = Dependency(
            name="Test1", command="t1", category=DependencyCategory.OCR, required=True
        )
        dep2 = Dependency(
            name="Test2", command="t2", category=DependencyCategory.OCR, required=True
        )

        report = EnvironmentReport(
            results=[
                DependencyCheckResult(
                    dependency=dep1, status=DependencyStatus.AVAILABLE
                ),
                DependencyCheckResult(
                    dependency=dep2, status=DependencyStatus.AVAILABLE
                ),
            ]
        )

        assert report.all_required_available is True
        assert len(report.missing_required) == 0

    def test_report_missing_required(self) -> None:
        """Test report with missing required dependency."""
        dep = Dependency(
            name="Required",
            command="req",
            category=DependencyCategory.OCR,
            required=True,
        )

        report = EnvironmentReport(
            results=[
                DependencyCheckResult(dependency=dep, status=DependencyStatus.MISSING),
            ]
        )

        assert report.all_required_available is False
        assert len(report.missing_required) == 1

    def test_report_missing_optional(self) -> None:
        """Test report with missing optional dependency."""
        dep = Dependency(
            name="Optional",
            command="opt",
            category=DependencyCategory.OCR,
            required=False,
        )

        report = EnvironmentReport(
            results=[
                DependencyCheckResult(
                    dependency=dep, status=DependencyStatus.OPTIONAL_MISSING
                ),
            ]
        )

        # All required are available (there are none)
        assert report.all_required_available is True
        assert len(report.missing_optional) == 1


# DependencyChecker tests


class TestDependencyChecker:
    """Tests for DependencyChecker."""

    def test_checker_creation(self) -> None:
        """Test creating checker."""
        checker = DependencyChecker()

        assert len(checker.dependencies) > 0

    def test_checker_custom_dependencies(self) -> None:
        """Test checker with custom dependencies."""
        custom = [
            Dependency(
                name="Custom", command="custom", category=DependencyCategory.BUILD
            )
        ]
        checker = DependencyChecker(dependencies=custom)

        assert len(checker.dependencies) == 1
        assert checker.dependencies[0].name == "Custom"

    def test_check_dependency_available(self) -> None:
        """Test checking an available dependency."""
        dep = Dependency(
            name="Python",
            command="python",  # Should be available
            category=DependencyCategory.BUILD,
        )
        checker = DependencyChecker(dependencies=[dep])

        result = checker.check_dependency(dep)

        # Python should be available since we're running Python
        assert result.status == DependencyStatus.AVAILABLE

    def test_check_dependency_missing(self) -> None:
        """Test checking a missing dependency."""
        dep = Dependency(
            name="Nonexistent",
            command="definitely_not_a_real_command_xyz123",
            category=DependencyCategory.BUILD,
            required=True,
        )
        checker = DependencyChecker(dependencies=[dep])

        result = checker.check_dependency(dep)

        assert result.status == DependencyStatus.MISSING

    def test_check_dependency_optional_missing(self) -> None:
        """Test checking missing optional dependency."""
        dep = Dependency(
            name="Optional",
            command="not_real_optional_xyz",
            category=DependencyCategory.BUILD,
            required=False,
        )
        checker = DependencyChecker(dependencies=[dep])

        result = checker.check_dependency(dep)

        assert result.status == DependencyStatus.OPTIONAL_MISSING

    def test_check_all(self) -> None:
        """Test checking all dependencies."""
        checker = DependencyChecker()
        report = checker.check_all()

        assert isinstance(report, EnvironmentReport)
        assert len(report.results) == len(checker.dependencies)
        assert report.platform != ""
        assert report.python_version != ""

    def test_check_category(self) -> None:
        """Test checking by category."""
        checker = DependencyChecker()
        ocr_results = checker.check_category(DependencyCategory.OCR)

        # Should only have OCR dependencies
        for result in ocr_results:
            assert result.dependency.category == DependencyCategory.OCR


class TestDependencyCheckerVersions:
    """Tests for version checking."""

    def test_version_satisfies(self) -> None:
        """Test version comparison."""
        checker = DependencyChecker()

        assert checker._version_satisfies("2.0.0", "1.0.0") is True
        assert checker._version_satisfies("1.0.0", "2.0.0") is False
        assert checker._version_satisfies("1.5.0", "1.5.0") is True


# Module-level function tests


class TestModuleFunctions:
    """Tests for module-level functions."""

    def test_check_environment(self) -> None:
        """Test check_environment function."""
        report = check_environment()

        assert isinstance(report, EnvironmentReport)
        assert len(report.results) > 0

    @patch("shutil.which")
    def test_check_ocr_available_true(self, mock_which: MagicMock) -> None:
        """Test OCR check when available."""
        mock_which.return_value = "/usr/bin/tesseract"

        result = check_ocr_available()

        assert result is True
        mock_which.assert_called_with("tesseract")

    @patch("shutil.which")
    def test_check_ocr_available_false(self, mock_which: MagicMock) -> None:
        """Test OCR check when not available."""
        mock_which.return_value = None

        result = check_ocr_available()

        assert result is False

    @patch("shutil.which")
    def test_check_pdf_tools_available_true(self, mock_which: MagicMock) -> None:
        """Test PDF tools check when available."""
        mock_which.return_value = "/usr/bin/pdftotext"

        result = check_pdf_tools_available()

        assert result is True

    @patch("shutil.which")
    def test_check_pdf_tools_available_false(self, mock_which: MagicMock) -> None:
        """Test PDF tools check when not available."""
        mock_which.return_value = None

        result = check_pdf_tools_available()

        assert result is False

    def test_get_missing_instructions(self) -> None:
        """Test getting missing instructions."""
        instructions = get_missing_instructions()

        assert isinstance(instructions, list)
        # Each instruction should be a tuple of (name, instruction)
        for name, instruction in instructions:
            assert isinstance(name, str)
            assert isinstance(instruction, str)


# Default dependencies tests


class TestDefaultDependencies:
    """Tests for default dependency list."""

    def test_default_dependencies_exist(self) -> None:
        """Test that default dependencies are defined."""
        assert len(DEFAULT_DEPENDENCIES) > 0

    def test_default_dependencies_have_instructions(self) -> None:
        """Test that all defaults have install instructions."""
        for dep in DEFAULT_DEPENDENCIES:
            assert len(dep.install_instructions) > 0

    def test_default_dependencies_categories(self) -> None:
        """Test that defaults cover multiple categories."""
        categories = {dep.category for dep in DEFAULT_DEPENDENCIES}
        assert len(categories) >= 3

    def test_tesseract_in_defaults(self) -> None:
        """Test that Tesseract is in defaults."""
        names = [dep.name.lower() for dep in DEFAULT_DEPENDENCIES]
        assert any("tesseract" in name for name in names)

    def test_poppler_in_defaults(self) -> None:
        """Test that Poppler is in defaults."""
        commands = [dep.command for dep in DEFAULT_DEPENDENCIES]
        assert "pdftotext" in commands
