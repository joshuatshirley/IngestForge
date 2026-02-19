"""Integration tests for documentation build integrity.

User Documentation Portal.
Verifies that the MkDocs configuration is valid and the site builds without errors.
"""

import shutil
import subprocess
import pytest
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================

DOCS_DIR = Path("docs")
MKDOCS_CONFIG = Path("mkdocs.yml")

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def clean_site_dir():
    """Ensure site directory is clean before and after tests."""
    site_dir = Path("site")
    if site_dir.exists():
        shutil.rmtree(site_dir)
    yield
    if site_dir.exists():
        shutil.rmtree(site_dir)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_mkdocs_config_exists():
    """GIVEN the project root
    WHEN checking for configuration
    THEN mkdocs.yml should exist.
    """
    assert MKDOCS_CONFIG.exists()


def test_docs_structure_integrity():
    """GIVEN the docs directory
    WHEN validating required files
    THEN all core markdown files should exist.
    """
    required_files = [
        "index.md",
        "user-guide/getting-started.md",
        "user-guide/workflows.md",
        "user-guide/agent-missions.md",
        "troubleshooting.md",
        "faq.md",
        "api-reference.md",
    ]

    for file_path in required_files:
        assert (
            DOCS_DIR / file_path
        ).exists(), f"Missing documentation file: {file_path}"


@pytest.mark.skipif(shutil.which("mkdocs") is None, reason="MkDocs not installed")
def test_mkdocs_build_process(clean_site_dir):
    """GIVEN a valid configuration and content
    WHEN running 'mkdocs build'
    THEN the build should succeed with exit code 0.
    """
    # Use --strict to fail on warnings (broken links, etc.)
    result = subprocess.run(
        ["mkdocs", "build", "--strict"], capture_output=True, text=True
    )

    if result.returncode != 0:
        error_msg = f"MkDocs build failed:\n{result.stderr}"
        pytest.fail(error_msg)

    assert result.returncode == 0
    assert Path("site/index.html").exists()
    assert Path("site/user-guide/getting-started/index.html").exists()
