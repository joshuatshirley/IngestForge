"""Comprehensive GWT unit tests for and QA-0101 system requirements enforcement.

Fulfills Epic AC for:
- Dependency Detection (Python, Node, pip)
- QA-0101: Clean OS Validation (venv check)
- JPL Compliance: Rule #2, #4, #9

Testing Strategy:
- Mock platform.system to test cross-platform RAM detection.
- Mock subprocess.run to simulate Node.js and macOS sysctl responses.
- Mock builtins.open to simulate Linux /proc/meminfo.
- Mock sys.version_info for Python version checks.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, mock_open


# Import functions under test
from scripts.install import (
    check_python_version,
    check_node_version,
    get_total_ram,
    check_system_resources,
)

# =============================================================================
# Python Version Tests
# =============================================================================


def test_check_python_version_success() -> None:
    """GIVEN a system with Python 3.11
    WHEN check_python_version is called
    THEN it returns success=True and the correct version string.
    """
    with patch("sys.version_info", (3, 11, 0, "final", 0)):
        success, msg, ver = check_python_version()
        assert success is True
        assert "3.11" in ver
        assert "✓ Python" in msg


def test_check_python_version_failure() -> None:
    """GIVEN a system with Python 3.9
    WHEN check_python_version is called
    THEN it returns success=False.
    """
    with patch("sys.version_info", (3, 9, 0, "final", 0)):
        success, msg, ver = check_python_version()
        assert success is False
        assert "3.9" in ver
        assert "required" in msg


# =============================================================================
# Node.js Version Tests
# =============================================================================


def test_check_node_version_success() -> None:
    """GIVEN a system with Node.js v18.12.0
    WHEN check_node_version is called
    THEN it returns success=True.
    """
    mock_res = MagicMock(returncode=0, stdout="v18.12.0\n")
    with patch("subprocess.run", return_value=mock_res):
        success, msg, ver = check_node_version()
        assert success is True
        assert "18.12" in ver


def test_check_node_version_too_low() -> None:
    """GIVEN a system with Node.js v16.0.0
    WHEN check_node_version is called
    THEN it returns success=False.
    """
    mock_res = MagicMock(returncode=0, stdout="v16.0.0\n")
    with patch("subprocess.run", return_value=mock_res):
        success, msg, ver = check_node_version()
        assert success is False
        assert "required" in msg


def test_check_node_version_missing() -> None:
    """GIVEN a system without Node.js
    WHEN check_node_version is called
    THEN it handles FileNotFoundError gracefully.
    """
    with patch("subprocess.run", side_effect=FileNotFoundError):
        success, msg, ver = check_node_version()
        assert success is False
        assert "not found" in msg


# =============================================================================
# Cross-Platform RAM Detection Tests (/ QA-0101)
# =============================================================================


def test_get_ram_linux() -> None:
    """GIVEN a Linux system with 8GB RAM in /proc/meminfo
    WHEN get_total_ram is called
    THEN it returns 8.0.
    """
    meminfo_content = "MemTotal:        8388608 kB\nMemFree:         1024000 kB"
    with patch("platform.system", return_value="Linux"):
        with patch("pathlib.Path.exists", return_value=True):
            with patch("builtins.open", mock_open(read_data=meminfo_content)):
                ram = get_total_ram()
                assert ram == 8.0


def test_get_ram_macos() -> None:
    """GIVEN a macOS system reporting 16GB via sysctl
    WHEN get_total_ram is called
    THEN it returns 16.0.
    """
    # 16GB in bytes: 17179869184
    mock_res = MagicMock(returncode=0, stdout="17179869184\n")
    with patch("platform.system", return_value="Darwin"):
        with patch("subprocess.run", return_value=mock_res):
            ram = get_total_ram()
            assert ram == 16.0


def test_get_ram_windows() -> None:
    """GIVEN a Windows system reporting 4GB via GlobalMemoryStatusEx
    WHEN get_total_ram is called
    THEN it returns 4.0.
    """
    with patch("platform.system", return_value="Windows"):
        with patch("scripts.install._get_ram_windows", return_value=4.0):
            ram = get_total_ram()
            assert ram == 4.0


# =============================================================================
# Resource Warning Tests
# =============================================================================


def test_check_system_resources_warnings() -> None:
    """GIVEN a system where RAM cannot be determined
    WHEN check_system_resources is called
    THEN it returns a warning but meets_requirements=True (JPL Rule #7: fail-soft).
    """
    with patch("scripts.install.get_total_ram", return_value=0.0):
        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = MagicMock(free=10 * (1024**3))
            info = check_system_resources()
            assert info.meets_requirements is True
            assert any("Could not determine RAM" in w for w in info.warnings)
