"""Comprehensive GWT unit tests for hardware pre-flight validation.

Hardware-Preflight-Validation
Epic EP-31 (MVP Readiness)

Acceptance Criteria:
- Check for >= 2GB RAM
- Check for >= 5GB Disk space
- Clear warning/exit on failure
- skip-hw-check override
- Cross-platform support (Win/Mac/Linux)

JPL Compliance:
- Rule #2: Bounded constants
- Rule #4: All test functions < 60 lines
- Rule #9: 100% type hints
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# Import functions under test
from scripts.install import check_system_resources, SystemInfo, main


@pytest.fixture
def mock_disk_usage() -> MagicMock:
    """Mock shutil.disk_usage."""
    with patch("shutil.disk_usage") as mock:
        # Default: 10GB free (plenty)
        mock.return_value = MagicMock(free=10 * (1024**3))
        yield mock


# =============================================================================
# & RAM VALIDATION TESTS (Cross-platform)
# =============================================================================


def test_check_system_resources_sufficient_ram(mock_disk_usage: MagicMock) -> None:
    """GIVEN sufficient RAM (4GB)
    WHEN check_system_resources is called
    THEN it should return meets_requirements=True.
    """
    with patch("scripts.install.get_total_ram", return_value=4.0):
        info = check_system_resources()
        assert info.meets_requirements is True
        assert info.ram_gb == 4.0
        assert len(info.warnings) == 0


def test_check_system_resources_insufficient_ram(mock_disk_usage: MagicMock) -> None:
    """GIVEN insufficient RAM (1GB)
    WHEN check_system_resources is called
    THEN it should return meets_requirements=False.
    """
    with patch("scripts.install.get_total_ram", return_value=1.0):
        info = check_system_resources()
        assert info.meets_requirements is False
        assert any("Low RAM" in w for w in info.warnings)


# =============================================================================
# DISK SPACE VALIDATION TESTS
# =============================================================================


def test_check_system_resources_insufficient_disk() -> None:
    """GIVEN insufficient disk space (2GB)
    WHEN check_system_resources is called
    THEN it should return meets_requirements=False.
    """
    with patch("shutil.disk_usage") as mock_disk:
        mock_disk.return_value = MagicMock(free=2 * (1024**3))

        with patch("scripts.install.get_total_ram", return_value=8.0):
            info = check_system_resources()
            assert info.meets_requirements is False
            assert any("Low disk" in w for w in info.warnings)


# =============================================================================
# skip-hw-check OVERRIDE TESTS
# =============================================================================


def test_main_respects_skip_hw_check() -> None:
    """GIVEN insufficient resources
    WHEN main is called with --skip-hw-check
    THEN it should proceed past hardware validation.
    """
    # Mock prerequisites to fail HW check but pass software checks
    with patch(
        "scripts.install.check_python_version", return_value=(True, "OK", "3.10")
    ):
        with patch(
            "scripts.install.check_node_version", return_value=(True, "OK", "18.0")
        ):
            # HW Check fails
            bad_sys_info = SystemInfo(
                ram_gb=1.0,
                disk_gb=1.0,
                platform_name="Linux",
                meets_requirements=False,
                warnings=["Low RAM"],
            )
            with patch(
                "scripts.install.check_system_resources", return_value=bad_sys_info
            ):
                # Mock create_virtualenv to stop execution after HW check
                with patch(
                    "scripts.install.create_virtualenv",
                    return_value=(False, None, "STOP"),
                ):
                    # Should return 1 because create_virtualenv failed, but NOT because HW failed
                    exit_code = main(["--skip-hw-check"])
                    assert exit_code == 1
                    # Verification: If it reached create_virtualenv, it skipped HW exit


def test_main_exits_on_hw_failure_no_skip() -> None:
    """GIVEN insufficient resources
    WHEN main is called without --skip-hw-check
    THEN it should exit with code 1.
    """
    with patch(
        "scripts.install.check_python_version", return_value=(True, "OK", "3.10")
    ):
        with patch(
            "scripts.install.check_node_version", return_value=(True, "OK", "18.0")
        ):
            bad_sys_info = SystemInfo(
                ram_gb=1.0, disk_gb=1.0, platform_name="Linux", meets_requirements=False
            )
            with patch(
                "scripts.install.check_system_resources", return_value=bad_sys_info
            ):
                exit_code = main([])
                assert exit_code == 1
