"""Fixtures for OCR testing."""

import pytest
from pathlib import Path
from typing import Dict


@pytest.fixture(scope="session")
def golden_ocr_dir() -> Path:
    """Return path to golden OCR dataset directory.

    Returns:
        Path to tests/data/golden/ocr directory

    Raises:
        FileNotFoundError: If golden directory does not exist
    """
    tests_root = Path(__file__).parent.parent.parent
    golden_dir = tests_root / "data" / "golden" / "ocr"

    if not golden_dir.exists():
        raise FileNotFoundError(f"Golden OCR directory not found: {golden_dir}")

    return golden_dir


@pytest.fixture(scope="session")
def golden_datasets(golden_ocr_dir: Path) -> Dict[str, str]:
    """Load all golden OCR dataset files.

    Args:
        golden_ocr_dir: Path to golden dataset directory

    Returns:
        Dictionary mapping dataset name to file contents

    Example:
        >>> datasets = golden_datasets(Path("/path/to/golden"))
        >>> assert "sample_1col" in datasets
        >>> assert len(datasets["sample_1col"]) > 0
    """
    datasets = {}

    for filepath in golden_ocr_dir.glob("sample_*.md"):
        dataset_name = filepath.stem
        datasets[dataset_name] = filepath.read_text(encoding="utf-8")

    return datasets


@pytest.fixture
def golden_1col(golden_datasets: Dict[str, str]) -> str:
    """Return single-column academic paper golden output.

    Args:
        golden_datasets: Dictionary of all golden datasets

    Returns:
        Expected markdown output for 1-column PDF
    """
    return golden_datasets["sample_1col"]


@pytest.fixture
def golden_2col(golden_datasets: Dict[str, str]) -> str:
    """Return two-column journal article golden output.

    Args:
        golden_datasets: Dictionary of all golden datasets

    Returns:
        Expected markdown output for 2-column PDF
    """
    return golden_datasets["sample_2col"]


@pytest.fixture
def golden_table(golden_datasets: Dict[str, str]) -> str:
    """Return document with tables golden output.

    Args:
        golden_datasets: Dictionary of all golden datasets

    Returns:
        Expected markdown output for PDF with tables
    """
    return golden_datasets["sample_table"]
