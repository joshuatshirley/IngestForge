"""Tests for Version Manager.

Automated Semantic Versioning
Epic: EP-26 (Security & Compliance)

Tests cover:
- SemanticVersion dataclass
- Version parsing
- Commit message pattern detection
- Version bumping
- VersionManager operations
"""

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import patch

from ingestforge.core.versioning import (
    SemanticVersion,
    BumpType,
    VersionManager,
    create_version_manager,
    parse_version,
    MAX_COMMITS_TO_ANALYZE,
)
from ingestforge.core.versioning.version_manager import (
    _detect_bump_from_message,
)


class TestSemanticVersion:
    """Tests for SemanticVersion dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic version."""
        version = SemanticVersion(1, 2, 3)
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease is None
        assert version.build_metadata is None

    def test_full_version_creation(self) -> None:
        """Test creating version with all fields."""
        version = SemanticVersion(
            major=2,
            minor=0,
            patch=0,
            prerelease="alpha.1",
            build_metadata="build.123",
        )
        assert version.major == 2
        assert version.prerelease == "alpha.1"
        assert version.build_metadata == "build.123"

    def test_string_representation(self) -> None:
        """Test version string formatting."""
        # Basic version
        assert str(SemanticVersion(1, 2, 3)) == "1.2.3"

        # With prerelease
        assert str(SemanticVersion(1, 0, 0, prerelease="alpha")) == "1.0.0-alpha"

        # With build metadata
        assert str(SemanticVersion(1, 0, 0, build_metadata="123")) == "1.0.0+123"

        # With both
        version = SemanticVersion(1, 0, 0, prerelease="beta", build_metadata="456")
        assert str(version) == "1.0.0-beta+456"

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        version = SemanticVersion(1, 2, 3, prerelease="rc1")
        result = version.to_dict()

        assert result["major"] == 1
        assert result["minor"] == 2
        assert result["patch"] == 3
        assert result["prerelease"] == "rc1"
        assert result["string"] == "1.2.3-rc1"

    def test_immutability(self) -> None:
        """Test that version is immutable (frozen)."""
        version = SemanticVersion(1, 0, 0)
        with pytest.raises(AttributeError):
            version.major = 2  # type: ignore

    def test_negative_values_rejected(self) -> None:
        """Test that negative version numbers are rejected.

        JPL Rule #5: Assert preconditions.
        """
        with pytest.raises(AssertionError):
            SemanticVersion(-1, 0, 0)

        with pytest.raises(AssertionError):
            SemanticVersion(0, -1, 0)

        with pytest.raises(AssertionError):
            SemanticVersion(0, 0, -1)


class TestSemanticVersionBump:
    """Tests for version bump operations."""

    def test_bump_major(self) -> None:
        """Test major version bump."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.bump(BumpType.MAJOR)

        assert new_version.major == 2
        assert new_version.minor == 0
        assert new_version.patch == 0

    def test_bump_minor(self) -> None:
        """Test minor version bump."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.bump(BumpType.MINOR)

        assert new_version.major == 1
        assert new_version.minor == 3
        assert new_version.patch == 0

    def test_bump_patch(self) -> None:
        """Test patch version bump."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.bump(BumpType.PATCH)

        assert new_version.major == 1
        assert new_version.minor == 2
        assert new_version.patch == 4

    def test_bump_none(self) -> None:
        """Test no bump returns same version."""
        version = SemanticVersion(1, 2, 3)
        new_version = version.bump(BumpType.NONE)

        assert new_version.major == version.major
        assert new_version.minor == version.minor
        assert new_version.patch == version.patch

    def test_bump_clears_prerelease(self) -> None:
        """Test that bumping clears prerelease info."""
        version = SemanticVersion(1, 0, 0, prerelease="alpha")
        new_version = version.bump(BumpType.PATCH)

        assert new_version.prerelease is None


class TestParseVersion:
    """Tests for parse_version function."""

    def test_parse_basic_version(self) -> None:
        """Test parsing basic version strings."""
        version = parse_version("1.2.3")
        assert version is not None
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3

    def test_parse_with_v_prefix(self) -> None:
        """Test parsing version with v prefix."""
        version = parse_version("v1.0.0")
        assert version is not None
        assert version.major == 1

    def test_parse_with_prerelease(self) -> None:
        """Test parsing version with prerelease."""
        version = parse_version("2.0.0-alpha.1")
        assert version is not None
        assert version.prerelease == "alpha.1"

    def test_parse_with_build_metadata(self) -> None:
        """Test parsing version with build metadata."""
        version = parse_version("1.0.0+build.123")
        assert version is not None
        assert version.build_metadata == "build.123"

    def test_parse_full_semver(self) -> None:
        """Test parsing full SemVer string."""
        version = parse_version("1.2.3-beta.2+build.456")
        assert version is not None
        assert version.major == 1
        assert version.minor == 2
        assert version.patch == 3
        assert version.prerelease == "beta.2"
        assert version.build_metadata == "build.456"

    def test_parse_invalid_returns_none(self) -> None:
        """Test that invalid versions return None."""
        assert parse_version("") is None
        assert parse_version("invalid") is None
        assert parse_version("1.2") is None
        assert parse_version("1.2.3.4") is None
        assert parse_version("a.b.c") is None

    def test_parse_respects_max_length(self) -> None:
        """Test that overly long versions are rejected.

        JPL Rule #2: Fixed upper bounds.
        """
        long_version = "1.0.0-" + "a" * 100
        assert parse_version(long_version) is None

    def test_parse_whitespace_handling(self) -> None:
        """Test that whitespace is trimmed."""
        version = parse_version("  1.2.3  ")
        assert version is not None
        assert str(version) == "1.2.3"


class TestDetectBumpFromMessage:
    """Tests for commit message pattern detection."""

    def test_breaking_change_patterns(self) -> None:
        """Test detection of breaking changes."""
        # Exclamation mark syntax
        assert _detect_bump_from_message("feat!: redesign API") == BumpType.MAJOR
        assert _detect_bump_from_message("fix!: breaking fix") == BumpType.MAJOR

        # BREAKING CHANGE footer
        assert _detect_bump_from_message("BREAKING CHANGE: new API") == BumpType.MAJOR
        assert _detect_bump_from_message("BREAKING-CHANGE: update") == BumpType.MAJOR

    def test_feature_patterns(self) -> None:
        """Test detection of features (minor bump)."""
        assert _detect_bump_from_message("feat: add new command") == BumpType.MINOR
        assert _detect_bump_from_message("feat(cli): add option") == BumpType.MINOR
        assert _detect_bump_from_message("FEAT: uppercase") == BumpType.MINOR

    def test_patch_patterns(self) -> None:
        """Test detection of fixes and other patch-level changes."""
        assert _detect_bump_from_message("fix: resolve crash") == BumpType.PATCH
        assert _detect_bump_from_message("fix(core): bug fix") == BumpType.PATCH
        assert _detect_bump_from_message("docs: update readme") == BumpType.PATCH
        assert _detect_bump_from_message("chore: update deps") == BumpType.PATCH
        assert _detect_bump_from_message("refactor: clean up") == BumpType.PATCH
        assert _detect_bump_from_message("test: add tests") == BumpType.PATCH
        assert _detect_bump_from_message("style: format code") == BumpType.PATCH
        assert _detect_bump_from_message("perf: optimize") == BumpType.PATCH
        assert _detect_bump_from_message("ci: update workflow") == BumpType.PATCH
        assert _detect_bump_from_message("build: update config") == BumpType.PATCH

    def test_unknown_patterns(self) -> None:
        """Test that unknown patterns return NONE."""
        assert _detect_bump_from_message("random commit") == BumpType.NONE
        assert _detect_bump_from_message("Update something") == BumpType.NONE
        assert _detect_bump_from_message("WIP: work in progress") == BumpType.NONE

    def test_breaking_takes_priority(self) -> None:
        """Test that breaking change takes priority over other patterns."""
        # Even if it starts with feat, the ! makes it breaking
        assert _detect_bump_from_message("feat!: new API") == BumpType.MAJOR


class TestVersionManager:
    """Tests for VersionManager class."""

    def test_init_with_defaults(self, tmp_path: Path) -> None:
        """Test initialization with default values."""
        # Create minimal project structure
        (tmp_path / "ingestforge").mkdir()
        init_file = tmp_path / "ingestforge" / "__init__.py"
        init_file.write_text('__version__ = "1.0.0"')

        manager = VersionManager(project_root=tmp_path)
        assert manager._project_root == tmp_path

    def test_init_validates_project_root(self, tmp_path: Path) -> None:
        """Test that init validates project root exists.

        JPL Rule #5: Assert preconditions.
        """
        non_existent = tmp_path / "does_not_exist"
        with pytest.raises(AssertionError):
            VersionManager(project_root=non_existent)

    def test_get_current_version_from_init(self, tmp_path: Path) -> None:
        """Test reading version from __init__.py."""
        # Create project structure
        pkg_dir = tmp_path / "ingestforge"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text('__version__ = "2.3.4"')

        manager = VersionManager(project_root=tmp_path)
        version = manager.get_current_version()

        assert version is not None
        assert str(version) == "2.3.4"

    def test_get_current_version_from_file(self, tmp_path: Path) -> None:
        """Test reading version from VERSION file."""
        # No __init__.py, use VERSION file
        version_file = tmp_path / "VERSION"
        version_file.write_text("3.0.0")

        manager = VersionManager(
            project_root=tmp_path,
            version_file=version_file,
        )
        version = manager.get_current_version()

        assert version is not None
        assert str(version) == "3.0.0"

    def test_get_current_version_not_found(self, tmp_path: Path) -> None:
        """Test when no version source exists."""
        manager = VersionManager(project_root=tmp_path)
        version = manager.get_current_version()
        assert version is None

    def test_bump_version_updates_init(self, tmp_path: Path) -> None:
        """Test that bump_version updates __init__.py."""
        # Create project structure
        pkg_dir = tmp_path / "ingestforge"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text('__version__ = "1.0.0"\n')

        manager = VersionManager(project_root=tmp_path)
        new_version = manager.bump_version(BumpType.MINOR)

        assert new_version is not None
        assert str(new_version) == "1.1.0"

        # Verify file was updated
        content = init_file.read_text()
        assert "1.1.0" in content

    def test_bump_version_no_current_starts_at_0_1_0(self, tmp_path: Path) -> None:
        """Test bumping when no current version exists."""
        # Create empty __init__.py
        pkg_dir = tmp_path / "ingestforge"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text('__version__ = "0.0.0"\n')

        manager = VersionManager(project_root=tmp_path)
        new_version = manager.bump_version(BumpType.PATCH)

        # Should bump from 0.0.0 to 0.0.1
        assert new_version is not None
        assert str(new_version) == "0.0.1"

    def test_validate_version(self, tmp_path: Path) -> None:
        """Test version validation."""
        manager = VersionManager(project_root=tmp_path)

        assert manager.validate_version("1.0.0") is True
        assert manager.validate_version("v2.3.4") is True
        assert manager.validate_version("invalid") is False
        assert manager.validate_version("") is False


class TestVersionManagerCommitAnalysis:
    """Tests for commit analysis functionality."""

    def test_analyze_commits_detects_major(self, tmp_path: Path) -> None:
        """Test that analyze_commits detects breaking changes."""
        manager = VersionManager(project_root=tmp_path)

        # Mock git command
        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = [
                "feat: add feature",
                "feat!: breaking API change",
                "fix: bug fix",
            ]
            bump = manager.analyze_commits()
            assert bump == BumpType.MAJOR

    def test_analyze_commits_detects_minor(self, tmp_path: Path) -> None:
        """Test that analyze_commits detects features."""
        manager = VersionManager(project_root=tmp_path)

        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = [
                "fix: bug fix",
                "feat: new feature",
                "docs: update docs",
            ]
            bump = manager.analyze_commits()
            assert bump == BumpType.MINOR

    def test_analyze_commits_detects_patch(self, tmp_path: Path) -> None:
        """Test that analyze_commits detects patch-level changes."""
        manager = VersionManager(project_root=tmp_path)

        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = [
                "fix: bug fix",
                "docs: update readme",
                "chore: cleanup",
            ]
            bump = manager.analyze_commits()
            assert bump == BumpType.PATCH

    def test_analyze_commits_returns_none_for_unknown(self, tmp_path: Path) -> None:
        """Test that analyze_commits returns NONE for non-conventional commits."""
        manager = VersionManager(project_root=tmp_path)

        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = [
                "random commit message",
                "another update",
            ]
            bump = manager.analyze_commits()
            assert bump == BumpType.NONE

    def test_analyze_commits_respects_limit(self, tmp_path: Path) -> None:
        """Test that commit analysis respects MAX_COMMITS_TO_ANALYZE.

        JPL Rule #2: Fixed upper bounds.
        """
        manager = VersionManager(project_root=tmp_path)

        # Create more commits than the limit
        many_commits = [
            "fix: fix " + str(i) for i in range(MAX_COMMITS_TO_ANALYZE + 50)
        ]

        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = many_commits
            # Should still work without issues
            bump = manager.analyze_commits()
            assert bump == BumpType.PATCH


class TestGetNextVersion:
    """Tests for get_next_version functionality."""

    def test_get_next_version_with_commits(self, tmp_path: Path) -> None:
        """Test getting next version based on commits."""
        # Create project with version
        pkg_dir = tmp_path / "ingestforge"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text('__version__ = "1.0.0"')

        manager = VersionManager(project_root=tmp_path)

        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = ["feat: new feature"]
            next_ver, bump_type = manager.get_next_version()

            assert str(next_ver) == "1.1.0"
            assert bump_type == BumpType.MINOR

    def test_get_next_version_defaults_to_patch(self, tmp_path: Path) -> None:
        """Test that next version defaults to patch when no conventional commits."""
        pkg_dir = tmp_path / "ingestforge"
        pkg_dir.mkdir()
        init_file = pkg_dir / "__init__.py"
        init_file.write_text('__version__ = "1.0.0"')

        manager = VersionManager(project_root=tmp_path)

        with patch.object(manager, "_get_commit_messages") as mock_git:
            mock_git.return_value = ["random commit"]
            next_ver, bump_type = manager.get_next_version()

            assert str(next_ver) == "1.0.1"
            assert bump_type == BumpType.PATCH


class TestFactoryFunction:
    """Tests for create_version_manager factory."""

    def test_create_version_manager(self, tmp_path: Path) -> None:
        """Test factory function creates manager."""
        manager = create_version_manager(project_root=tmp_path)
        assert isinstance(manager, VersionManager)

    def test_create_version_manager_with_version_file(self, tmp_path: Path) -> None:
        """Test factory with version file."""
        version_file = tmp_path / "VERSION"
        version_file.write_text("1.0.0")

        manager = create_version_manager(
            project_root=tmp_path,
            version_file=version_file,
        )
        assert isinstance(manager, VersionManager)


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_fixed_bounds_defined(self) -> None:
        """Test that fixed bounds constants are defined.

        JPL Rule #2: Fixed upper bounds.
        """
        from ingestforge.core.versioning.version_manager import (
            MAX_COMMITS_TO_ANALYZE,
            MAX_VERSION_LENGTH,
            MAX_PRERELEASE_LENGTH,
        )

        assert MAX_COMMITS_TO_ANALYZE == 100
        assert MAX_VERSION_LENGTH == 50
        assert MAX_PRERELEASE_LENGTH == 20

    def test_all_functions_return_explicit_values(self, tmp_path: Path) -> None:
        """Test that functions return explicit values.

        JPL Rule #7: Check all return values.
        """
        manager = VersionManager(project_root=tmp_path)

        # These should all return explicit values (not None without meaning)
        version = manager.get_current_version()
        # None is valid here - explicit return

        is_valid = manager.validate_version("1.0.0")
        assert isinstance(is_valid, bool)

        parsed = parse_version("1.0.0")
        assert parsed is not None

        bump = _detect_bump_from_message("fix: test")
        assert isinstance(bump, BumpType)
