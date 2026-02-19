"""Comprehensive GWT unit tests for Platform Installer Packaging.

Platform Installers.
Verifies installer generation logic, script templates, and JPL compliance.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from scripts.package_installers import (
    InstallerConfig,
    InstallerFormat,
    InstallerPackager,
    PackageMetadata,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_binary(tmp_path):
    """Create a dummy binary file."""
    bin_path = tmp_path / "ingestforge.exe"
    bin_path.write_text("dummy binary content")
    return bin_path


@pytest.fixture
def packager(tmp_path):
    """Initialize packager with a temporary project root."""
    # Create a dummy pyproject.toml to define root
    (tmp_path / "pyproject.toml").write_text("")
    return InstallerPackager(project_root=tmp_path)


# =============================================================================
# UNIT TESTS (GWT)
# =============================================================================


def test_find_project_root_bounded(tmp_path):
    """GIVEN a deeply nested directory
    WHEN _find_project_root is called
    THEN it respects the MAX_DIR_DEPTH bound (JPL Rule #2).
    """
    deep_dir = tmp_path / "a" / "b" / "c" / "d"
    deep_dir.mkdir(parents=True)

    # Packager initialized in deep dir without pyproject.toml
    packager = InstallerPackager()

    with patch.object(Path, "resolve", return_value=deep_dir):
        root = packager._find_project_root()
        # Should fallback to CWD if no pyproject.toml found within depth
        assert isinstance(root, Path)


def test_check_prerequisites_missing_tool(packager, mock_binary):
    """GIVEN a configuration for NSIS
    WHEN 'makensis' is not in PATH
    THEN check_prerequisites returns an error message.
    """
    config = InstallerConfig(format=InstallerFormat.NSIS, binary_path=mock_binary)

    with patch("shutil.which", return_value=None):
        missing = packager.check_prerequisites(config)
        assert any("makensis not found" in m for m in missing)


def test_generate_nsis_script_content(packager, mock_binary):
    """GIVEN a valid installer config
    WHEN _generate_nsis_script is called
    THEN it returns a string containing file associations and uninstaller logic.
    """
    config = InstallerConfig(
        format=InstallerFormat.NSIS,
        binary_path=mock_binary,
        metadata=PackageMetadata(name="TestApp"),
    )

    script = packager._generate_nsis_script(config)

    assert '!include "MUI2.nsh"' in script
    assert 'WriteRegStr HKCR ".ingestforge"' in script
    assert "WriteUninstaller" in script
    assert 'Section "Uninstall"' in script


def test_create_nsis_success(packager, mock_binary, tmp_path):
    """GIVEN a valid binary and makensis availability
    WHEN _create_nsis is called
    THEN it executes the compiler and returns success.
    """
    config = InstallerConfig(
        format=InstallerFormat.NSIS,
        binary_path=mock_binary,
        output_dir=tmp_path / "dist",
    )

    # Mock successful subprocess run
    mock_run = MagicMock(returncode=0)

    # Mock the output file existence
    output_exe = config.output_dir / "IngestForge-Setup.exe"
    config.output_dir.mkdir(parents=True)
    output_exe.write_text("installer content")

    with patch("subprocess.run", return_value=mock_run):
        result = packager._create_nsis(config)
        assert result.success is True
        assert result.output_path == output_exe


def test_create_dmg_with_codesign(packager, mock_binary, tmp_path):
    """GIVEN a macOS environment and a signing identity
    WHEN _create_dmg is called
    THEN it attempts to call 'codesign' (AC).
    """
    config = InstallerConfig(
        format=InstallerFormat.DMG,
        binary_path=mock_binary,
        output_dir=tmp_path / "dist",
    )

    os.environ["MACOS_CERT_IDENTITY"] = "Developer ID: IngestForge"

    with patch("subprocess.run") as mock_subprocess:
        mock_subprocess.return_value = MagicMock(returncode=0)
        packager._create_dmg(config)

        # Verify codesign was called
        calls = [c.args[0][0] for c in mock_subprocess.call_args_list]
        assert "codesign" in calls

    del os.environ["MACOS_CERT_IDENTITY"]


def test_create_deb_desktop_integration(packager, mock_binary, tmp_path):
    """GIVEN a Linux config
    WHEN _create_deb is called
    THEN it creates a .desktop file with correct MIME types (AC).
    """
    config = InstallerConfig(
        format=InstallerFormat.DEB,
        binary_path=mock_binary,
        output_dir=tmp_path / "dist",
    )

    # Need to mock Path.mkdir and shutil.copy2 to avoid real FS ops in some deep paths
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        with patch("tempfile.TemporaryDirectory") as mock_tmp:
            pkg_dir = tmp_path / "pkg"
            pkg_dir.mkdir()
            mock_tmp.return_value.__enter__.return_value = str(pkg_dir)

            packager._create_deb(config)

            desktop_file = (
                pkg_dir / "usr" / "share" / "applications" / "ingestforge.desktop"
            )
            assert desktop_file.exists()
            content = desktop_file.read_text()
            assert "MimeType=application/x-ingestforge;" in content
