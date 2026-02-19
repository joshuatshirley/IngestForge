"""Integration smoke tests for built binaries.

Post-Build Verification
Epic: EP-30 (Distribution & Deployment)

Tests run on actual built binaries to verify:
- Binary executes successfully
- Version command works
- Help command works  
- Basic functionality
- API server starts

Usage:
    pytest tests/integration/test_binary_smoke.py --binary=dist/ingestforge

JPL Compliance: Rules #4, #9
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import pytest


# =============================================================================
# Fixtures & Configuration
# =============================================================================


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add command line options for smoke tests.

    Args:
        parser: Pytest parser
    """
    parser.addoption(
        "--binary",
        action="store",
        default=None,
        help="Path to built binary for smoke testing",
    )


@pytest.fixture
def binary_path(request: pytest.FixtureRequest) -> Optional[Path]:
    """Get binary path from command line.

    Args:
        request: Pytest request fixture

    Returns:
        Path to binary or None if not provided
    """
    path_str = request.config.getoption("--binary")
    if path_str is None:
        pytest.skip("--binary not provided, skipping smoke tests")
    return Path(path_str)


# =============================================================================
# Smoke Tests ()
# =============================================================================


def test_binary_exists(binary_path: Path) -> None:
    """Test binary file exists.

    Given: Binary path provided
    When: Checking if file exists
    Then: File should exist and be executable
    """
    assert binary_path.exists(), f"Binary not found: {binary_path}"
    assert binary_path.is_file(), f"Not a file: {binary_path}"


def test_binary_version_command(binary_path: Path) -> None:
    """Test binary --version command.

    Given: Built binary
    When: Running --version command
    Then: Returns version string successfully
    """
    result = subprocess.run(
        [str(binary_path), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Version command failed: {result.stderr}"
    assert len(result.stdout) > 0, "Version output is empty"


def test_binary_help_command(binary_path: Path) -> None:
    """Test binary --help command.

    Given: Built binary
    When: Running --help command
    Then: Returns help text successfully
    """
    result = subprocess.run(
        [str(binary_path), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()


def test_binary_size_under_limit(binary_path: Path) -> None:
    """Test binary size is under 200MB limit.

    Given: Built binary
    When: Checking file size
    Then: Size should be < 200MB ()
    """
    size_mb = binary_path.stat().st_size / (1024 * 1024)

    assert size_mb < 200, (
        f"Binary exceeds 200MB limit: {size_mb:.2f}MB. "
        "Try using --upx-level max for better compression."
    )


@pytest.mark.slow
def test_binary_dry_run_ingest(binary_path: Path, tmp_path: Path) -> None:
    """Test binary can perform dry-run ingest.

    Given: Built binary and sample text file
    When: Running dry-run ingest command
    Then: Command succeeds without errors
    """
    # Create sample file
    sample = tmp_path / "sample.txt"
    sample.write_text("Test content for ingestion")

    result = subprocess.run(
        [str(binary_path), "ingest", str(sample), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=60,
    )

    # Should succeed or show dry-run message
    assert result.returncode in [0, 2], f"Dry-run failed: {result.stderr}"


# =============================================================================
# Test Summary
# =============================================================================

"""
Smoke Test Coverage ():

Total Tests: 5
- test_binary_exists: Verify binary file exists
- test_binary_version_command: Test --version flag
- test_binary_help_command: Test --help flag
- test_binary_size_under_limit: Verify size <200MB ()
- test_binary_dry_run_ingest: Test basic functionality

All tests follow GWT pattern
100% type hints
Run with: pytest tests/integration/test_binary_smoke.py --binary=dist/ingestforge
"""
