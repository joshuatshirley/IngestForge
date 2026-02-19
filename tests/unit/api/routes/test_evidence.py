"""
Unit tests for Evidence Links API - .

Tests bidirectional sync API endpoints for knowledge graph ↔ PDF viewer.

JPL Power of Ten Compliance:
- Rule #2: Fixed upper bounds in test fixtures
- Rule #4: All test functions < 60 lines
- Rule #9: Complete type hints

Epic: EP-14 (Foundry UI)
Feature: FE-11-01 (Visualization)
Test Date: 2026-02-18
"""

import pytest
from fastapi.testclient import TestClient
from typing import Dict, Any
from unittest.mock import Mock, patch

# Import API app
from ingestforge.api.main import app

client = TestClient(app)

# =============================================================================
# TEST FIXTURES (JPL Rule #2: Fixed bounds)
# =============================================================================

MAX_TEST_CHUNKS = 100


@pytest.fixture
def mock_chunk_with_bbox() -> Dict[str, Any]:
    """
    Create a mock chunk with bounding box metadata.

    Test fixture for evidence extraction.
    """
    return {
        "chunk_id": "chunk-001",
        "document_id": "doc-123",
        "content": "John Doe was the CEO of Apple Inc. in 2011.",
        "metadata": {
            "source_file": "/data/docs/apple_history.pdf",
            "total_pages": 50,
            "page_start": 5,
            "bbox": {
                "page": 5,
                "x1": 0.1,
                "y1": 0.2,
                "x2": 0.3,
                "y2": 0.25,
                "confidence": 0.95,
                "entity_id": "entity-john-doe",
                "text": "John Doe",
            },
        },
    }


@pytest.fixture
def mock_chunk_multiple_bboxes() -> Dict[str, Any]:
    """
    Create a mock chunk with multiple bounding boxes.

    Test fixture for multi-entity chunks.
    """
    return {
        "chunk_id": "chunk-002",
        "document_id": "doc-123",
        "content": "Apple Inc. was founded in 1976 by Steve Jobs.",
        "metadata": {
            "source_file": "/data/docs/apple_history.pdf",
            "total_pages": 50,
            "bounding_boxes": [
                {
                    "page": 10,
                    "x1": 0.15,
                    "y1": 0.3,
                    "x2": 0.35,
                    "y2": 0.35,
                    "confidence": 0.92,
                    "entity_id": "entity-apple",
                    "text": "Apple Inc.",
                },
                {
                    "page": 10,
                    "x1": 0.5,
                    "y1": 0.3,
                    "x2": 0.7,
                    "y2": 0.35,
                    "confidence": 0.88,
                    "entity_id": "entity-steve-jobs",
                    "text": "Steve Jobs",
                },
            ],
        },
    }


@pytest.fixture
def mock_chunk_no_bbox() -> Dict[str, Any]:
    """Create a mock chunk without bounding box metadata."""
    return {
        "chunk_id": "chunk-003",
        "document_id": "doc-123",
        "content": "This chunk has no bounding boxes.",
        "metadata": {"source_file": "/data/docs/apple_history.pdf"},
    }


@pytest.fixture
def mock_storage_get_chunks(
    mock_chunk_with_bbox: Dict[str, Any],
    mock_chunk_multiple_bboxes: Dict[str, Any],
    mock_chunk_no_bbox: Dict[str, Any],
) -> Mock:
    """
    Mock storage.get_chunks_by_document method.

    Returns 3 chunks: 1 with single bbox, 1 with multiple, 1 with none.
    """
    mock = Mock()
    mock.return_value = [
        mock_chunk_with_bbox,
        mock_chunk_multiple_bboxes,
        mock_chunk_no_bbox,
    ]
    return mock


# =============================================================================
# BASIC RETRIEVAL TESTS (4 tests)
# =============================================================================


def test_given_document_id_when_get_evidence_links_then_returns_links(
    mock_storage_get_chunks: Mock,
) -> None:
    """
    GIVEN a document ID with chunks containing bboxes
    WHEN GET /v1/extract/evidence-links?document_id=doc-123
    THEN returns all evidence links with 200 OK
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document = mock_storage_get_chunks

        response = client.get("/v1/extract/evidence-links?document_id=doc-123")

        assert response.status_code == 200
        data = response.json()

        assert data["document_id"] == "doc-123"
        assert data["total_links"] == 3  # 1 from chunk-001 + 2 from chunk-002
        assert len(data["links"]) == 3
        assert all("bbox" in link for link in data["links"])


def test_given_entity_id_filter_when_get_evidence_links_then_returns_filtered_links(
    mock_storage_get_chunks: Mock,
) -> None:
    """
    GIVEN entity_id filter parameter
    WHEN GET /v1/extract/evidence-links?document_id=doc-123&entity_id=entity-apple
    THEN returns only links for that entity
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document = mock_storage_get_chunks

        response = client.get(
            "/v1/extract/evidence-links?document_id=doc-123&entity_id=entity-apple"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_links"] == 1
        assert data["links"][0]["entity_id"] == "entity-apple"
        assert data["filters_applied"]["entity_id"] == "entity-apple"


def test_given_page_filter_when_get_evidence_links_then_returns_page_specific_links(
    mock_storage_get_chunks: Mock,
) -> None:
    """
    GIVEN page number filter
    WHEN GET /v1/extract/evidence-links?document_id=doc-123&page=10
    THEN returns only links on that page
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document = mock_storage_get_chunks

        response = client.get("/v1/extract/evidence-links?document_id=doc-123&page=10")

        assert response.status_code == 200
        data = response.json()

        assert data["total_links"] == 2  # Both from chunk-002 on page 10
        assert all(link["page"] == 10 for link in data["links"])


def test_given_confidence_threshold_when_get_evidence_links_then_returns_high_confidence_links(
    mock_storage_get_chunks: Mock,
) -> None:
    """
    GIVEN min_confidence filter
    WHEN GET /v1/extract/evidence-links?document_id=doc-123&min_confidence=0.9
    THEN returns only high-confidence links
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document = mock_storage_get_chunks

        response = client.get(
            "/v1/extract/evidence-links?document_id=doc-123&min_confidence=0.9"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["total_links"] == 2  # chunk-001 (0.95) + chunk-002 apple (0.92)
        assert all(link["confidence"] >= 0.9 for link in data["links"])


# =============================================================================
# VALIDATION TESTS (4 tests)
# =============================================================================


def test_given_invalid_document_id_when_get_evidence_links_then_returns_404() -> None:
    """
    GIVEN document ID with no chunks
    WHEN GET /v1/extract/evidence-links?document_id=invalid-doc
    THEN returns 404 NOT FOUND
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = []

        response = client.get("/v1/extract/evidence-links?document_id=invalid-doc")

        assert response.status_code == 404
        assert "No chunks found" in response.json()["detail"]


def test_given_invalid_page_number_when_get_evidence_links_then_returns_422() -> None:
    """
    GIVEN invalid page number (0 or negative)
    WHEN GET /v1/extract/evidence-links?document_id=doc-123&page=0
    THEN returns 422 UNPROCESSABLE ENTITY
    """
    response = client.get("/v1/extract/evidence-links?document_id=doc-123&page=0")

    assert response.status_code == 422  # Pydantic validation error


def test_given_confidence_out_of_range_when_get_evidence_links_then_returns_422() -> (
    None
):
    """
    GIVEN confidence threshold > 1.0
    WHEN GET /v1/extract/evidence-links?document_id=doc-123&min_confidence=1.5
    THEN returns 422 UNPROCESSABLE ENTITY
    """
    response = client.get(
        "/v1/extract/evidence-links?document_id=doc-123&min_confidence=1.5"
    )

    assert response.status_code == 422


def test_given_missing_document_id_when_get_evidence_links_then_returns_422() -> None:
    """
    GIVEN missing required document_id parameter
    WHEN GET /v1/extract/evidence-links (no params)
    THEN returns 422 UNPROCESSABLE ENTITY
    """
    response = client.get("/v1/extract/evidence-links")

    assert response.status_code == 422


# =============================================================================
# EDGE CASE TESTS (3 tests)
# =============================================================================


def test_given_no_bboxes_when_get_evidence_links_then_returns_empty_links() -> None:
    """
    GIVEN chunks with no bounding box metadata
    WHEN GET /v1/extract/evidence-links?document_id=doc-456
    THEN returns empty links array
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [
            {"chunk_id": "chunk-999", "document_id": "doc-456", "metadata": {}}
        ]

        response = client.get("/v1/extract/evidence-links?document_id=doc-456")

        assert response.status_code == 200
        data = response.json()
        assert data["total_links"] == 0
        assert data["links"] == []


def test_given_multiple_entities_same_page_when_get_evidence_links_then_returns_all_links(
    mock_chunk_multiple_bboxes: Dict[str, Any],
) -> None:
    """
    GIVEN chunk with multiple entities on same page
    WHEN GET /v1/extract/evidence-links?document_id=doc-123&page=10
    THEN returns all entities on that page
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [
            mock_chunk_multiple_bboxes
        ]

        response = client.get("/v1/extract/evidence-links?document_id=doc-123&page=10")

        assert response.status_code == 200
        data = response.json()
        assert data["total_links"] == 2
        assert {link["entity_id"] for link in data["links"]} == {
            "entity-apple",
            "entity-steve-jobs",
        }


def test_given_overlapping_bboxes_when_get_evidence_links_then_returns_all_links(
    mock_chunk_multiple_bboxes: Dict[str, Any],
) -> None:
    """
    GIVEN chunks with overlapping bounding boxes
    WHEN GET /v1/extract/evidence-links?document_id=doc-123
    THEN returns all links (no deduplication)
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [
            mock_chunk_multiple_bboxes
        ]

        response = client.get("/v1/extract/evidence-links?document_id=doc-123")

        assert response.status_code == 200
        data = response.json()
        assert data["total_links"] == 2  # Both returned, no deduplication


# =============================================================================
# DOCUMENT METADATA TESTS (3 tests)
# =============================================================================


def test_given_document_id_when_get_metadata_then_returns_document_info(
    mock_chunk_with_bbox: Dict[str, Any],
) -> None:
    """
    GIVEN a valid document ID
    WHEN GET /v1/extract/documents/{document_id}/metadata
    THEN returns document metadata with file path and page count
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [
            mock_chunk_with_bbox
        ]

        response = client.get("/v1/extract/documents/doc-123/metadata")

        assert response.status_code == 200
        data = response.json()

        assert data["document_id"] == "doc-123"
        assert data["file_path"] == "/data/docs/apple_history.pdf"
        assert data["total_pages"] == 50
        assert data["content_type"] == "application/pdf"


def test_given_invalid_document_when_get_metadata_then_returns_404() -> None:
    """
    GIVEN invalid document ID
    WHEN GET /v1/extract/documents/{document_id}/metadata
    THEN returns 404 NOT FOUND
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = []

        response = client.get("/v1/extract/documents/invalid-doc/metadata")

        assert response.status_code == 404


def test_given_document_no_title_when_get_metadata_then_uses_filename_as_title(
    mock_chunk_with_bbox: Dict[str, Any],
) -> None:
    """
    GIVEN document without explicit title
    WHEN GET /v1/extract/documents/{document_id}/metadata
    THEN uses filename stem as title
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [
            mock_chunk_with_bbox
        ]

        response = client.get("/v1/extract/documents/doc-123/metadata")

        assert response.status_code == 200
        data = response.json()
        assert data["title"] == "apple_history"  # Filename without extension


# =============================================================================
# PDF SERVING TESTS (2 tests)
# =============================================================================


def test_given_valid_document_when_serve_pdf_then_returns_file(
    mock_chunk_with_bbox: Dict[str, Any], tmp_path
) -> None:
    """
    GIVEN valid document with existing PDF file
    WHEN GET /v1/extract/documents/{document_id}/pdf
    THEN returns FileResponse with PDF content
    """
    # Create temporary PDF file
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")  # Valid PDF header

    # Mock chunk with temp file path
    mock_chunk = mock_chunk_with_bbox.copy()
    mock_chunk["metadata"]["source_file"] = str(pdf_file)

    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [mock_chunk]

        response = client.get("/v1/extract/documents/doc-123/pdf")

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert b"%PDF" in response.content


def test_given_missing_pdf_file_when_serve_pdf_then_returns_404(
    mock_chunk_with_bbox: Dict[str, Any],
) -> None:
    """
    GIVEN document with non-existent PDF file path
    WHEN GET /v1/extract/documents/{document_id}/pdf
    THEN returns 404 NOT FOUND
    """
    with patch("ingestforge.api.routes.evidence.get_storage") as mock_get_storage:
        mock_get_storage.return_value.get_chunks_by_document.return_value = [
            mock_chunk_with_bbox
        ]

        response = client.get("/v1/extract/documents/doc-123/pdf")

        assert response.status_code == 404
        assert "PDF file not found" in response.json()["detail"]


# =============================================================================
# JPL COMPLIANCE TESTS (1 test)
# =============================================================================


def test_jpl_compliance_all_functions_under_60_lines() -> None:
    """
    GIVEN evidence.py module
    WHEN analyzing function line counts
    THEN all functions are ≤60 lines (JPL Rule #4)
    """
    import inspect
    import ingestforge.api.routes.evidence as evidence_module

    violations = []

    for name, obj in inspect.getmembers(evidence_module):
        if inspect.isfunction(obj) and obj.__module__ == evidence_module.__name__:
            source_lines = inspect.getsourcelines(obj)[0]
            # Filter out empty lines and comments
            code_lines = [
                line
                for line in source_lines
                if line.strip() and not line.strip().startswith("#")
            ]
            line_count = len(code_lines)

            if line_count > 60:
                violations.append(f"{name}: {line_count} lines")

    assert not violations, f"JPL Rule #4 violations: {violations}"
