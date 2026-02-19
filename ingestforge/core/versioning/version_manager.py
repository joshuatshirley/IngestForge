"""Semantic Version Manager Implementation.

Automated Semantic Versioning
Epic: EP-26 (Security & Compliance)

Provides semantic versioning management with commit analysis
for CI/CD integration.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_COMMITS_TO_ANALYZE)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_COMMITS_TO_ANALYZE = 100
MAX_VERSION_LENGTH = 50
MAX_PRERELEASE_LENGTH = 20


class BumpType(Enum):
    """Version bump types following SemVer."""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    NONE = "none"


# Commit message patterns (Conventional Commits)
BREAKING_PATTERNS = [
    re.compile(r"^[a-z]+!:", re.IGNORECASE),  # feat!:, fix!:
    re.compile(r"BREAKING CHANGE:", re.IGNORECASE),
    re.compile(r"BREAKING-CHANGE:", re.IGNORECASE),
]

MINOR_PATTERNS = [
    re.compile(r"^feat(\(.+\))?:", re.IGNORECASE),  # feat: or feat(scope):
]

PATCH_PATTERNS = [
    re.compile(r"^fix(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^docs(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^chore(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^refactor(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^test(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^style(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^perf(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^ci(\(.+\))?:", re.IGNORECASE),
    re.compile(r"^build(\(.+\))?:", re.IGNORECASE),
]


@dataclass(frozen=True)
class SemanticVersion:
    """Immutable semantic version following SemVer 2.0.0.

    Rule #9: Complete type hints.
    """

    major: int
    minor: int
    patch: int
    prerelease: Optional[str] = None
    build_metadata: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate version components.

        Rule #5: Assert preconditions.
        """
        assert self.major >= 0, "major must be non-negative"
        assert self.minor >= 0, "minor must be non-negative"
        assert self.patch >= 0, "patch must be non-negative"

    def __str__(self) -> str:
        """Format as version string.

        Returns:
            Formatted version string (e.g., "1.2.3-alpha+build123").
        """
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build_metadata:
            version += f"+{self.build_metadata}"
        return version

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "prerelease": self.prerelease,
            "build_metadata": self.build_metadata,
            "string": str(self),
        }

    def bump(self, bump_type: BumpType) -> "SemanticVersion":
        """Create new version with specified bump.

        Args:
            bump_type: Type of version bump.

        Returns:
            New SemanticVersion with bumped values.

        Rule #7: Return explicit result.
        """
        if bump_type == BumpType.MAJOR:
            return SemanticVersion(self.major + 1, 0, 0)
        elif bump_type == BumpType.MINOR:
            return SemanticVersion(self.major, self.minor + 1, 0)
        elif bump_type == BumpType.PATCH:
            return SemanticVersion(self.major, self.minor, self.patch + 1)
        return self


# Version parsing regex (SemVer 2.0.0 compliant)
VERSION_PATTERN = re.compile(
    r"^v?"  # Optional 'v' prefix
    r"(?P<major>0|[1-9]\d*)\."
    r"(?P<minor>0|[1-9]\d*)\."
    r"(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)


def parse_version(version_str: str) -> Optional[SemanticVersion]:
    """Parse a version string into SemanticVersion.

    Args:
        version_str: Version string to parse.

    Returns:
        SemanticVersion or None if invalid.

    Rule #7: Return explicit result.
    """
    if not version_str or len(version_str) > MAX_VERSION_LENGTH:
        return None

    match = VERSION_PATTERN.match(version_str.strip())
    if not match:
        return None

    try:
        prerelease = match.group("prerelease")
        if prerelease and len(prerelease) > MAX_PRERELEASE_LENGTH:
            prerelease = prerelease[:MAX_PRERELEASE_LENGTH]

        return SemanticVersion(
            major=int(match.group("major")),
            minor=int(match.group("minor")),
            patch=int(match.group("patch")),
            prerelease=prerelease,
            build_metadata=match.group("build"),
        )
    except (ValueError, TypeError):
        return None


def _detect_bump_from_message(message: str) -> BumpType:
    """Detect bump type from commit message.

    Args:
        message: Commit message to analyze.

    Returns:
        Detected BumpType.

    Rule #4: Helper function.
    """
    # Check for breaking changes first
    for pattern in BREAKING_PATTERNS:
        if pattern.search(message):
            return BumpType.MAJOR

    # Check for features
    for pattern in MINOR_PATTERNS:
        if pattern.match(message):
            return BumpType.MINOR

    # Check for fixes and other patch-level changes
    for pattern in PATCH_PATTERNS:
        if pattern.match(message):
            return BumpType.PATCH

    return BumpType.NONE


class VersionManager:
    """Semantic version manager for CI/CD integration.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        version_file: Optional[Path] = None,
    ) -> None:
        """Initialize version manager.

        Args:
            project_root: Root directory of the project.
            version_file: Path to VERSION file (optional).

        Rule #5: Assert preconditions.
        """
        self._project_root = project_root or Path.cwd()
        self._version_file = version_file
        self._init_file = self._project_root / "ingestforge" / "__init__.py"

        assert (
            self._project_root.exists()
        ), f"Project root does not exist: {self._project_root}"

    def get_current_version(self) -> Optional[SemanticVersion]:
        """Get current version from __init__.py or VERSION file.

        Returns:
            Current SemanticVersion or None if not found.

        Rule #7: Return explicit result.
        """
        # Try __init__.py first
        version = self._read_version_from_init()
        if version:
            return version

        # Try VERSION file
        if self._version_file and self._version_file.exists():
            return self._read_version_from_file(self._version_file)

        # Try default VERSION location
        default_version_file = self._project_root / "VERSION"
        if default_version_file.exists():
            return self._read_version_from_file(default_version_file)

        return None

    def _read_version_from_init(self) -> Optional[SemanticVersion]:
        """Read version from __init__.py.

        Returns:
            SemanticVersion or None.

        Rule #4: Helper function.
        """
        if not self._init_file.exists():
            return None

        try:
            content = self._init_file.read_text(encoding="utf-8")
            match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
            if match:
                return parse_version(match.group(1))
        except Exception as e:
            logger.debug(f"Failed to read version from init: {e}")

        return None

    def _read_version_from_file(self, file_path: Path) -> Optional[SemanticVersion]:
        """Read version from file.

        Args:
            file_path: Path to version file.

        Returns:
            SemanticVersion or None.

        Rule #4: Helper function.
        """
        try:
            content = file_path.read_text(encoding="utf-8").strip()
            return parse_version(content)
        except Exception as e:
            logger.debug(f"Failed to read version from file: {e}")
            return None

    def bump_version(self, bump_type: BumpType) -> Optional[SemanticVersion]:
        """Bump version and update files.

        Args:
            bump_type: Type of version bump.

        Returns:
            New SemanticVersion or None if failed.

        Rule #7: Return explicit result.
        """
        current = self.get_current_version()
        if not current:
            logger.warning("No current version found, starting at 0.1.0")
            current = SemanticVersion(0, 1, 0)

        new_version = current.bump(bump_type)

        # Update __init__.py
        if self._update_init_version(new_version):
            logger.info(f"Version bumped: {current} -> {new_version}")
            return new_version

        return None

    def _update_init_version(self, version: SemanticVersion) -> bool:
        """Update version in __init__.py.

        Args:
            version: New version to set.

        Returns:
            True if successful.

        Rule #4: Helper function.
        Rule #7: Return explicit result.
        """
        if not self._init_file.exists():
            return False

        try:
            content = self._init_file.read_text(encoding="utf-8")
            new_content = re.sub(
                r'(__version__\s*=\s*["\'])([^"\']+)(["\'])',
                f"\\g<1>{version}\\g<3>",
                content,
            )

            if new_content == content:
                # No match found, try to add it
                return False

            self._init_file.write_text(new_content, encoding="utf-8")
            return True
        except Exception as e:
            logger.error(f"Failed to update version: {e}")
            return False

    def analyze_commits(self, since_tag: Optional[str] = None) -> BumpType:
        """Analyze commits to determine recommended bump type.

        Args:
            since_tag: Analyze commits since this tag (optional).

        Returns:
            Recommended BumpType based on commit messages.

        Rule #2: Limit commits analyzed.
        Rule #7: Return explicit result.
        """
        commits = self._get_commit_messages(since_tag)

        highest_bump = BumpType.NONE
        for message in commits[:MAX_COMMITS_TO_ANALYZE]:
            bump = _detect_bump_from_message(message)
            if bump == BumpType.MAJOR:
                return BumpType.MAJOR  # Can't go higher
            if bump == BumpType.MINOR and highest_bump != BumpType.MAJOR:
                highest_bump = BumpType.MINOR
            if bump == BumpType.PATCH and highest_bump == BumpType.NONE:
                highest_bump = BumpType.PATCH

        return highest_bump

    def _get_commit_messages(self, since_tag: Optional[str] = None) -> List[str]:
        """Get commit messages from git.

        Args:
            since_tag: Get commits since this tag.

        Returns:
            List of commit messages.

        Rule #4: Helper function.
        """
        try:
            if since_tag:
                cmd = ["git", "log", f"{since_tag}..HEAD", "--pretty=format:%s"]
            else:
                cmd = [
                    "git",
                    "log",
                    "-n",
                    str(MAX_COMMITS_TO_ANALYZE),
                    "--pretty=format:%s",
                ]

            result = subprocess.run(
                cmd,
                cwd=self._project_root,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                return [m for m in result.stdout.strip().split("\n") if m]
        except Exception as e:
            logger.debug(f"Failed to get commits: {e}")

        return []

    def get_next_version(self) -> Tuple[SemanticVersion, BumpType]:
        """Get next version based on commit analysis.

        Returns:
            Tuple of (next_version, bump_type).

        Rule #7: Return explicit result.
        """
        current = self.get_current_version()
        if not current:
            current = SemanticVersion(0, 0, 0)

        bump_type = self.analyze_commits()
        if bump_type == BumpType.NONE:
            bump_type = BumpType.PATCH  # Default to patch

        next_version = current.bump(bump_type)
        return next_version, bump_type

    def validate_version(self, version_str: str) -> bool:
        """Validate a version string.

        Args:
            version_str: Version string to validate.

        Returns:
            True if valid SemVer format.

        Rule #7: Return explicit result.
        """
        return parse_version(version_str) is not None


def create_version_manager(
    project_root: Optional[Path] = None,
    version_file: Optional[Path] = None,
) -> VersionManager:
    """Factory function to create a version manager.

    Args:
        project_root: Root directory of the project.
        version_file: Path to VERSION file.

    Returns:
        Configured VersionManager instance.
    """
    return VersionManager(project_root=project_root, version_file=version_file)
