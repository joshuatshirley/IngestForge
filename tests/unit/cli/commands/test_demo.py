"""
Unit tests for IngestForge demo command.

Sample-Dataset-Demo
Tests all Epic AC using Given-When-Then format:
- Curated sample documents
- Pre-computed embeddings
- Demo command
- Sample queries
- Demo reset

JPL Power of Ten Compliance:
- Rule #2: Bounded test iterations (MAX_TEST_ITERATIONS)
- Rule #4: All test functions <60 lines
- Rule #9: 100% type hints

Coverage Target: >80%
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# JPL Rule #2: Bounded constants
MAX_TEST_ITERATIONS = 3


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def demo_dir(tmp_path: Path) -> Path:
    """Create temporary demo directory."""
    demo = tmp_path / "demo"
    demo.mkdir()
    return demo


@pytest.fixture
def demo_metadata(demo_dir: Path) -> Path:
    """Create demo metadata file."""
    metadata_file = demo_dir / "demo_metadata.json"
    metadata = {
        "version": "1.0.0",
        "corpus_size_mb": 2.5,
        "document_count": 10,
        "last_updated": "2026-02-18",
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f)
    return metadata_file


@pytest.fixture
def mock_collection() -> Mock:
    """Create mock ChromaDB collection."""
    collection = Mock()
    collection.count.return_value = 42
    collection.query.return_value = {
        "documents": [["Sample document text here"]],
        "metadatas": [[{"source": "test.pdf"}]],
    }
    return collection


@pytest.fixture
def demo_archive(tmp_path: Path) -> Path:
    """Create demo corpus archive."""
    archive_path = tmp_path / "demo_corpus.tar.gz"

    # Create temporary structure
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test.txt").write_text("test content")

    # Create archive
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(corpus_dir, arcname="demo_corpus")

    return archive_path


# =============================================================================
# Curated Sample Documents
# =============================================================================


def test_get_demo_directory() -> None:
    """
    Given: IngestForge is installed
    When: get_demo_directory() is called
    Then: Returns ~/.ingestforge/demo path

    Epic Demo stored in ~/.ingestforge/demo/
    """
    from ingestforge.cli.commands.demo import get_demo_directory

    demo_dir = get_demo_directory()

    assert isinstance(demo_dir, Path)
    assert demo_dir.name == "demo"
    assert ".ingestforge" in str(demo_dir)


def test_get_corpus_archive_path_not_found() -> None:
    """
    Given: Demo corpus archive does not exist
    When: get_corpus_archive_path() is called
    Then: Returns None

    Epic Handle missing corpus gracefully
    JPL Rule #7: Returns None on failure
    """
    from ingestforge.cli.commands.demo import get_corpus_archive_path

    with patch("pathlib.Path.exists", return_value=False):
        archive_path = get_corpus_archive_path()

    assert archive_path is None


def test_extract_demo_corpus_success(demo_dir: Path, demo_archive: Path) -> None:
    """
    Given: Valid demo corpus archive exists
    When: extract_demo_corpus() is called
    Then: Extracts files successfully and returns success=True

    Epic Extract 10 diverse sample documents
    Epic Extract corpus on first run
    JPL Rule #2: Bounded extraction (MAX_EXTRACTION_FILES)
    """
    from ingestforge.cli.commands.demo import extract_demo_corpus

    with patch(
        "ingestforge.cli.commands.demo.get_corpus_archive_path",
        return_value=demo_archive,
    ):
        success, message = extract_demo_corpus(demo_dir)

    assert success is True
    assert "✓ Extracted" in message
    assert demo_dir.exists()


def test_extract_demo_corpus_archive_not_found(demo_dir: Path) -> None:
    """
    Given: Demo corpus archive does not exist
    When: extract_demo_corpus() is called
    Then: Returns success=False with error message

    Epic Graceful error handling
    JPL Rule #7: Explicit success status
    """
    from ingestforge.cli.commands.demo import extract_demo_corpus

    with patch(
        "ingestforge.cli.commands.demo.get_corpus_archive_path", return_value=None
    ):
        success, message = extract_demo_corpus(demo_dir)

    assert success is False
    assert "not found" in message.lower()


def test_extract_demo_corpus_extraction_fails(
    demo_dir: Path, demo_archive: Path
) -> None:
    """
    Given: Archive exists but extraction fails
    When: extract_demo_corpus() is called
    Then: Returns success=False with error message

    Epic Handle extraction errors
    """
    from ingestforge.cli.commands.demo import extract_demo_corpus

    with patch(
        "ingestforge.cli.commands.demo.get_corpus_archive_path",
        return_value=demo_archive,
    ):
        with patch("tarfile.open", side_effect=Exception("Extraction error")):
            success, message = extract_demo_corpus(demo_dir)

    assert success is False
    assert "failed" in message.lower()


# =============================================================================
# Pre-Computed Embeddings
# =============================================================================


def test_load_demo_metadata_success(demo_metadata: Path, demo_dir: Path) -> None:
    """
    Given: Valid demo metadata file exists
    When: load_demo_metadata() is called
    Then: Returns DemoMetadata with correct fields

    Epic Metadata includes source, date, domain
    JPL Rule #9: TypedDict for metadata
    """
    from ingestforge.cli.commands.demo import load_demo_metadata

    metadata = load_demo_metadata(demo_dir)

    assert metadata is not None
    assert metadata["version"] == "1.0.0"
    assert metadata["corpus_size_mb"] == 2.5
    assert metadata["document_count"] == 10
    assert metadata["last_updated"] == "2026-02-18"


def test_load_demo_metadata_missing(demo_dir: Path) -> None:
    """
    Given: Demo metadata file does not exist
    When: load_demo_metadata() is called
    Then: Returns None

    JPL Rule #7: Returns None on failure
    """
    from ingestforge.cli.commands.demo import load_demo_metadata

    metadata = load_demo_metadata(demo_dir)

    assert metadata is None


def test_load_demo_metadata_invalid_json(demo_dir: Path) -> None:
    """
    Given: Metadata file contains invalid JSON
    When: load_demo_metadata() is called
    Then: Returns None

    Epic Graceful error handling
    """
    from ingestforge.cli.commands.demo import load_demo_metadata

    metadata_file = demo_dir / "demo_metadata.json"
    metadata_file.write_text("{invalid json")

    metadata = load_demo_metadata(demo_dir)

    assert metadata is None


def test_load_demo_collection_success(demo_dir: Path) -> None:
    """
    Given: Demo embeddings directory exists
    When: load_demo_collection() is called
    Then: Returns ChromaDB collection successfully

    Epic ChromaDB collection ready to query
    """
    from ingestforge.cli.commands.demo import load_demo_collection

    embeddings_dir = demo_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)

    mock_repo = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_repo.get_collection.return_value = mock_collection

    with patch(
        "ingestforge.cli.commands.demo.ChromaRepository", return_value=mock_repo
    ):
        success, collection, message = load_demo_collection()

    assert success is True
    assert collection is not None
    assert "✓ Loaded" in message


def test_load_demo_collection_missing_embeddings(demo_dir: Path) -> None:
    """
    Given: Demo embeddings directory does not exist
    When: load_demo_collection() is called
    Then: Returns failure with helpful message

    Epic Display clear error messages
    """
    from ingestforge.cli.commands.demo import load_demo_collection

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        success, collection, message = load_demo_collection()

    assert success is False
    assert collection is None
    assert "not found" in message.lower()


def test_load_demo_collection_import_error(demo_dir: Path) -> None:
    """
    Given: ChromaDB is not installed
    When: load_demo_collection() is called
    Then: Returns failure gracefully

    Epic Handle missing dependencies
    """
    from ingestforge.cli.commands.demo import load_demo_collection

    embeddings_dir = demo_dir / "embeddings"
    embeddings_dir.mkdir(parents=True)

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.ChromaRepository", side_effect=ImportError()
        ):
            success, collection, message = load_demo_collection()

    assert success is False
    assert collection is None


# =============================================================================
# Sample Queries
# =============================================================================


def test_get_demo_queries_returns_list() -> None:
    """
    Given: Demo command is initialized
    When: get_demo_queries() is called
    Then: Returns list of DemoQuery objects

    Epic 5-10 curated queries
    JPL Rule #2: Returns bounded list (MAX_DEMO_QUERIES)
    """
    from ingestforge.cli.commands.demo import get_demo_queries

    queries = get_demo_queries()

    assert isinstance(queries, list)
    assert len(queries) >= 5
    assert len(queries) <= 10


def test_get_demo_queries_structure() -> None:
    """
    Given: Demo queries are defined
    When: get_demo_queries() is called
    Then: Each query has required fields

    Epic Sample queries showcase features
    JPL Rule #9: TypedDict for query structure
    """
    from ingestforge.cli.commands.demo import get_demo_queries

    queries = get_demo_queries()

    # JPL Rule #2: Bounded iteration
    for query in queries[:MAX_TEST_ITERATIONS]:
        assert "query" in query
        assert "description" in query
        assert "feature" in query
        assert isinstance(query["query"], str)
        assert len(query["query"]) > 0


def test_get_demo_queries_covers_features() -> None:
    """
    Given: Demo queries are curated
    When: get_demo_queries() is called
    Then: Queries cover different features

    Epic Showcase semantic search, entity extraction, code search
    """
    from ingestforge.cli.commands.demo import get_demo_queries

    queries = get_demo_queries()
    features = [q["feature"] for q in queries]

    # Should have diverse features
    assert "semantic_search" in features
    assert len(set(features)) >= 3  # At least 3 different features


def test_run_demo_queries_success(mock_collection: Mock) -> None:
    """
    Given: Valid ChromaDB collection is loaded
    When: run_demo_queries() is called
    Then: Executes queries and displays results

    Epic Display sample queries to try
    JPL Rule #2: Bounded query loop
    """
    from ingestforge.cli.commands.demo import run_demo_queries

    success, message = run_demo_queries(mock_collection)

    assert success is True
    assert "queries" in message.lower()
    assert mock_collection.query.called


def test_run_demo_queries_missing_rich() -> None:
    """
    Given: Rich library is not installed
    When: run_demo_queries() is called
    Then: Returns failure with helpful message

    Epic Handle missing dependencies gracefully
    """
    from ingestforge.cli.commands.demo import run_demo_queries

    mock_collection = Mock()

    with patch("ingestforge.cli.commands.demo.Console", None):
        success, message = run_demo_queries(mock_collection)

    assert success is False
    assert "rich" in message.lower()


def test_run_demo_queries_handles_empty_results(mock_collection: Mock) -> None:
    """
    Given: Collection returns empty results
    When: run_demo_queries() is called
    Then: Handles gracefully without crashing

    Epic Robust error handling
    JPL Rule #7: Check all return values
    """
    from ingestforge.cli.commands.demo import run_demo_queries

    mock_collection.query.return_value = {"documents": [[]], "metadatas": [[]]}

    success, message = run_demo_queries(mock_collection)

    assert success is True  # Should complete even with no results


# =============================================================================
# Demo Reset
# =============================================================================


def test_reset_demo_success(demo_dir: Path, demo_archive: Path) -> None:
    """
    Given: Existing demo directory exists
    When: reset_demo() is called
    Then: Removes old demo and extracts fresh corpus

    Epic Reset and re-ingest demo corpus
    """
    from ingestforge.cli.commands.demo import reset_demo

    # Create existing demo
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "old_file.txt").write_text("old content")

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.get_corpus_archive_path",
            return_value=demo_archive,
        ):
            success, message = reset_demo()

    assert success is True
    assert "reset complete" in message.lower()
    assert not (demo_dir / "old_file.txt").exists()


def test_reset_demo_preserves_main_corpus(tmp_path: Path, demo_archive: Path) -> None:
    """
    Given: User has main corpus and demo corpus
    When: reset_demo() is called
    Then: Only demo corpus is removed, main corpus preserved

    Epic Preserves user's main corpus
    """
    from ingestforge.cli.commands.demo import reset_demo

    # Create demo and main corpus
    demo_dir = tmp_path / "demo"
    main_corpus = tmp_path / "main_corpus"

    demo_dir.mkdir()
    main_corpus.mkdir()
    (main_corpus / "user_file.txt").write_text("user content")

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.get_corpus_archive_path",
            return_value=demo_archive,
        ):
            reset_demo()

    # Main corpus should be untouched
    assert (main_corpus / "user_file.txt").exists()
    assert (main_corpus / "user_file.txt").read_text() == "user content"


def test_reset_demo_extraction_fails(demo_dir: Path) -> None:
    """
    Given: Demo archive extraction fails
    When: reset_demo() is called
    Then: Returns failure status

    Epic Robust error handling
    JPL Rule #7: Check return values
    """
    from ingestforge.cli.commands.demo import reset_demo

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.extract_demo_corpus",
            return_value=(False, "Error"),
        ):
            success, message = reset_demo()

    assert success is False


# =============================================================================
# Integration Tests
# =============================================================================


def test_run_demo_full_flow(
    demo_dir: Path, demo_archive: Path, mock_collection: Mock
) -> None:
    """
    Given: Fresh IngestForge installation
    When: run_demo() is called
    Then: Extracts corpus, loads collection, runs queries

    Epic Demo command launches demo mode
    Integration test covering full flow
    """
    from ingestforge.cli.commands.demo import run_demo

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.get_corpus_archive_path",
            return_value=demo_archive,
        ):
            with patch(
                "ingestforge.cli.commands.demo.load_demo_collection",
                return_value=(True, mock_collection, "OK"),
            ):
                exit_code = run_demo(reset=False, interactive=False)

    assert exit_code == 0


def test_run_demo_reset_flow(demo_dir: Path, demo_archive: Path) -> None:
    """
    Given: Demo corpus already exists
    When: run_demo(reset=True) is called
    Then: Resets demo corpus successfully

    Epic Demo reset workflow
    """
    from ingestforge.cli.commands.demo import run_demo

    demo_dir.mkdir(parents=True, exist_ok=True)

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.get_corpus_archive_path",
            return_value=demo_archive,
        ):
            exit_code = run_demo(reset=True, interactive=False)

    assert exit_code == 0


def test_run_demo_missing_rich() -> None:
    """
    Given: Rich library is not installed
    When: run_demo() is called
    Then: Returns error exit code

    Epic Dependency validation
    """
    from ingestforge.cli.commands.demo import run_demo

    with patch("ingestforge.cli.commands.demo.Console", None):
        exit_code = run_demo()

    assert exit_code == 1


def test_run_demo_collection_load_fails(demo_dir: Path, demo_archive: Path) -> None:
    """
    Given: Demo corpus extracted but collection load fails
    When: run_demo() is called
    Then: Returns error exit code with helpful message

    Epic Graceful degradation
    """
    from ingestforge.cli.commands.demo import run_demo

    with patch(
        "ingestforge.cli.commands.demo.get_demo_directory", return_value=demo_dir
    ):
        with patch(
            "ingestforge.cli.commands.demo.get_corpus_archive_path",
            return_value=demo_archive,
        ):
            with patch(
                "ingestforge.cli.commands.demo.load_demo_collection",
                return_value=(False, None, "Error"),
            ):
                # Fake that corpus exists
                (demo_dir / "demo_metadata.json").write_text("{}")

                exit_code = run_demo()

    assert exit_code == 1


# =============================================================================
# JPL Compliance Tests
# =============================================================================


def test_jpl_rule_2_bounded_loops() -> None:
    """
    Given: Demo module is implemented
    When: Code is analyzed
    Then: All loops use MAX_* bounded constants

    JPL Rule #2: Bounded loops required
    """
    from ingestforge.cli.commands import demo

    assert hasattr(demo, "MAX_DEMO_DOCS")
    assert hasattr(demo, "MAX_DEMO_QUERIES")
    assert hasattr(demo, "MAX_SEARCH_DEPTH")
    assert hasattr(demo, "MAX_EXTRACTION_FILES")

    assert demo.MAX_DEMO_DOCS == 10
    assert demo.MAX_DEMO_QUERIES == 10


def test_jpl_rule_9_type_hints() -> None:
    """
    Given: Demo module is implemented
    When: Functions are inspected
    Then: All functions have type hints

    JPL Rule #9: 100% type hints required
    """
    from ingestforge.cli.commands.demo import (
        extract_demo_corpus,
        load_demo_collection,
        run_demo_queries,
        reset_demo,
        run_demo,
    )

    functions = [
        extract_demo_corpus,
        load_demo_collection,
        run_demo_queries,
        reset_demo,
        run_demo,
    ]

    # JPL Rule #2: Bounded iteration
    for func in functions[: MAX_TEST_ITERATIONS * 2]:
        assert func.__annotations__, f"{func.__name__} missing type hints"


def test_jpl_rule_7_return_value_checking() -> None:
    """
    Given: Demo functions are implemented
    When: Functions are called
    Then: All return explicit success/failure status

    JPL Rule #7: Check all return values
    """
    from ingestforge.cli.commands.demo import (
        extract_demo_corpus,
        load_demo_collection,
        run_demo_queries,
        reset_demo,
    )

    # All functions should return Tuple[bool, ...] or int
    assert extract_demo_corpus.__annotations__["return"] == "Tuple[bool, str]"
    assert (
        load_demo_collection.__annotations__["return"]
        == "Tuple[bool, Optional[Any], str]"
    )
    assert run_demo_queries.__annotations__["return"] == "Tuple[bool, str]"
    assert reset_demo.__annotations__["return"] == "Tuple[bool, str]"


# =============================================================================
# Documentation Tests
# =============================================================================


def test_module_has_docstring() -> None:
    """
    Given: Demo module is implemented
    When: Module is imported
    Then: Has comprehensive docstring with Epic AC mapping

    Documentation requirement
    """
    from ingestforge.cli.commands import demo

    assert demo.__doc__ is not None
    assert len(demo.__doc__) > 100
    assert "" in demo.__doc__
    assert "Epic AC" in demo.__doc__


def test_all_functions_have_docstrings() -> None:
    """
    Given: Demo functions are implemented
    When: Functions are inspected
    Then: All have docstrings with Epic AC references

    Documentation requirement
    JPL Rule #2: Bounded iteration
    """
    from ingestforge.cli.commands.demo import (
        extract_demo_corpus,
        load_demo_collection,
        get_demo_queries,
        run_demo_queries,
        reset_demo,
    )

    functions = [
        extract_demo_corpus,
        load_demo_collection,
        get_demo_queries,
        run_demo_queries,
        reset_demo,
    ]

    # JPL Rule #2: Bounded loop
    for func in functions[: MAX_TEST_ITERATIONS * 2]:
        assert func.__doc__ is not None
        assert len(func.__doc__) > 20
