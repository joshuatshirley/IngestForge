"""Runtime Dependency Guard for IngestForge.

Detects and handles missing system tools with graceful fallbacks
and user-friendly installation prompts."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class DependencyStatus(str, Enum):
    """Status of a dependency check."""

    AVAILABLE = "available"
    MISSING = "missing"
    OUTDATED = "outdated"
    OPTIONAL_MISSING = "optional_missing"


class DependencyCategory(str, Enum):
    """Categories of dependencies."""

    OCR = "ocr"
    DOCUMENT = "document"
    IMAGE = "image"
    DATABASE = "database"
    BUILD = "build"


@dataclass
class Dependency:
    """A system dependency."""

    name: str
    command: str  # Command to check for
    category: DependencyCategory
    required: bool = True
    version_flag: str = "--version"
    min_version: Optional[str] = None
    description: str = ""
    install_instructions: Dict[str, str] = field(default_factory=dict)

    def get_install_instruction(self, platform_name: Optional[str] = None) -> str:
        """Get installation instruction for platform.

        Args:
            platform_name: Platform (auto-detect if None)

        Returns:
            Installation instruction string
        """
        if platform_name is None:
            platform_name = platform.system().lower()

        # Map platform names
        if platform_name == "darwin":
            platform_name = "macos"

        return self.install_instructions.get(
            platform_name,
            self.install_instructions.get("default", f"Install {self.name}"),
        )


@dataclass
class DependencyCheckResult:
    """Result of a dependency check."""

    dependency: Dependency
    status: DependencyStatus
    version: Optional[str] = None
    path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class EnvironmentReport:
    """Report of all dependency checks."""

    results: List[DependencyCheckResult] = field(default_factory=list)
    platform: str = ""
    python_version: str = ""

    @property
    def all_required_available(self) -> bool:
        """Check if all required dependencies are available."""
        for result in self.results:
            if result.dependency.required:
                if result.status in (
                    DependencyStatus.MISSING,
                    DependencyStatus.OUTDATED,
                ):
                    return False
        return True

    @property
    def missing_required(self) -> List[DependencyCheckResult]:
        """Get list of missing required dependencies."""
        return [
            r
            for r in self.results
            if r.dependency.required and r.status == DependencyStatus.MISSING
        ]

    @property
    def missing_optional(self) -> List[DependencyCheckResult]:
        """Get list of missing optional dependencies."""
        return [
            r
            for r in self.results
            if not r.dependency.required
            and r.status == DependencyStatus.OPTIONAL_MISSING
        ]


# Default dependencies to check
DEFAULT_DEPENDENCIES = [
    Dependency(
        name="Tesseract OCR",
        command="tesseract",
        category=DependencyCategory.OCR,
        required=False,
        description="Required for OCR-based document processing",
        install_instructions={
            "windows": "Download from: https://github.com/UB-Mannheim/tesseract/wiki",
            "macos": "brew install tesseract",
            "linux": "sudo apt-get install tesseract-ocr",
            "default": "See: https://tesseract-ocr.github.io/tessdoc/Installation.html",
        },
    ),
    Dependency(
        name="Poppler",
        command="pdftotext",
        category=DependencyCategory.DOCUMENT,
        required=False,
        description="Required for PDF text extraction",
        install_instructions={
            "windows": "Download from: https://github.com/oschwartz10612/poppler-windows/releases",
            "macos": "brew install poppler",
            "linux": "sudo apt-get install poppler-utils",
            "default": "See: https://poppler.freedesktop.org/",
        },
    ),
    Dependency(
        name="ImageMagick",
        command="convert",
        category=DependencyCategory.IMAGE,
        required=False,
        description="Used for image preprocessing",
        install_instructions={
            "windows": "Download from: https://imagemagick.org/script/download.php",
            "macos": "brew install imagemagick",
            "linux": "sudo apt-get install imagemagick",
        },
    ),
    Dependency(
        name="FFmpeg",
        command="ffmpeg",
        category=DependencyCategory.IMAGE,
        required=False,
        description="Used for audio/video processing",
        install_instructions={
            "windows": "Download from: https://ffmpeg.org/download.html",
            "macos": "brew install ffmpeg",
            "linux": "sudo apt-get install ffmpeg",
        },
    ),
    Dependency(
        name="SQLite",
        command="sqlite3",
        category=DependencyCategory.DATABASE,
        required=False,
        description="Database for local storage",
        install_instructions={
            "windows": "Usually bundled with Python",
            "macos": "Built into macOS",
            "linux": "sudo apt-get install sqlite3",
        },
    ),
]


class DependencyChecker:
    """Checks for system dependencies and provides installation guidance.

    Detects missing tools and offers platform-specific installation
    instructions with graceful fallback support.
    """

    def __init__(
        self,
        dependencies: Optional[List[Dependency]] = None,
    ) -> None:
        """Initialize checker.

        Args:
            dependencies: List of dependencies to check (default: DEFAULT_DEPENDENCIES)
        """
        self.dependencies = dependencies or DEFAULT_DEPENDENCIES.copy()

    def check_dependency(self, dep: Dependency) -> DependencyCheckResult:
        """Check a single dependency.

        Args:
            dep: Dependency to check

        Returns:
            DependencyCheckResult with status
        """
        # Check if command exists
        path = shutil.which(dep.command)

        if path is None:
            status = (
                DependencyStatus.MISSING
                if dep.required
                else DependencyStatus.OPTIONAL_MISSING
            )
            return DependencyCheckResult(
                dependency=dep,
                status=status,
                error=f"{dep.name} not found in PATH",
            )

        # Get version if possible
        version = self._get_version(dep)

        # Check minimum version if specified
        if dep.min_version and version:
            if not self._version_satisfies(version, dep.min_version):
                return DependencyCheckResult(
                    dependency=dep,
                    status=DependencyStatus.OUTDATED,
                    version=version,
                    path=Path(path),
                    error=f"Version {version} < required {dep.min_version}",
                )

        return DependencyCheckResult(
            dependency=dep,
            status=DependencyStatus.AVAILABLE,
            version=version,
            path=Path(path),
        )

    def _get_version(self, dep: Dependency) -> Optional[str]:
        """Get version of dependency.

        Args:
            dep: Dependency

        Returns:
            Version string or None
        """
        try:
            result = subprocess.run(
                [dep.command, dep.version_flag],
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = result.stdout or result.stderr
            return self._extract_version_from_output(output)
        except Exception as e:
            logger.debug(f"Failed to detect version: {e}")
            return None

    def _extract_version_from_output(self, output: str) -> Optional[str]:
        """Extract version from command output.

        Args:
            output: Command output text

        Returns:
            Version string or None
        """
        import re

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Try to find version number pattern
            match = re.search(r"(\d+\.[\d.]+)", line)
            if match:
                return match.group(1)
            return line[:50]  # Return first 50 chars

        return None

    def _version_satisfies(self, version: str, min_version: str) -> bool:
        """Check if version satisfies minimum.

        Args:
            version: Current version
            min_version: Required minimum version

        Returns:
            True if version >= min_version
        """
        try:
            from packaging import version as pkg_version

            return pkg_version.parse(version) >= pkg_version.parse(min_version)
        except ImportError:
            # Simple string comparison fallback
            return version >= min_version

    def check_all(self) -> EnvironmentReport:
        """Check all dependencies.

        Returns:
            EnvironmentReport with all results
        """
        import sys

        results = [self.check_dependency(dep) for dep in self.dependencies]

        return EnvironmentReport(
            results=results,
            platform=platform.system(),
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        )

    def check_category(
        self, category: DependencyCategory
    ) -> List[DependencyCheckResult]:
        """Check dependencies in a category.

        Args:
            category: Category to check

        Returns:
            List of results for that category
        """
        deps = [d for d in self.dependencies if d.category == category]
        return [self.check_dependency(d) for d in deps]


def check_environment() -> EnvironmentReport:
    """Check all environment dependencies.

    Returns:
        EnvironmentReport with results
    """
    checker = DependencyChecker()
    return checker.check_all()


def check_ocr_available() -> bool:
    """Check if OCR tools are available.

    Returns:
        True if Tesseract is available
    """
    return shutil.which("tesseract") is not None


def check_pdf_tools_available() -> bool:
    """Check if PDF tools are available.

    Returns:
        True if poppler tools are available
    """
    return shutil.which("pdftotext") is not None


def get_missing_instructions() -> List[Tuple[str, str]]:
    """Get installation instructions for missing dependencies.

    Returns:
        List of (name, instruction) tuples
    """
    report = check_environment()
    instructions = []

    for result in report.missing_required + report.missing_optional:
        instruction = result.dependency.get_install_instruction()
        instructions.append((result.dependency.name, instruction))

    return instructions


def print_environment_report() -> None:
    """Print formatted environment report to console."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    report = check_environment()

    table = Table(title="IngestForge Environment Check")
    table.add_column("Dependency", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Version")
    table.add_column("Path")

    status_styles = {
        DependencyStatus.AVAILABLE: "[green]Available[/green]",
        DependencyStatus.MISSING: "[red]Missing (Required)[/red]",
        DependencyStatus.OUTDATED: "[yellow]Outdated[/yellow]",
        DependencyStatus.OPTIONAL_MISSING: "[dim]Missing (Optional)[/dim]",
    }

    for result in report.results:
        status = status_styles.get(result.status, str(result.status))
        table.add_row(
            result.dependency.name,
            status,
            result.version or "-",
            str(result.path) if result.path else "-",
        )

    console.print(table)
    console.print()

    # Print installation instructions for missing
    missing = report.missing_required + report.missing_optional
    if missing:
        console.print("[yellow]Installation Instructions:[/yellow]")
        for result in missing:
            console.print(f"  {result.dependency.name}:")
            console.print(f"    {result.dependency.get_install_instruction()}")
            console.print()
