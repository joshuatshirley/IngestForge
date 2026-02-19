"""Tests for Retrieval & Search Router.

TICKET-506: Search endpoint with citations
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Only import if fastapi is available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from fastapi import FastAPI
from ingestforge.api.routes.retrieval import (
    router,
    _map_search_result_to_item,
    _normalize_score,
    CitationInfo,
    SearchResultItem,
    SearchResponse,
)


@pytest.fixture
def app():
    """Create FastAPI test app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_search_result():
    """Create a mock SearchResult dataclass."""

    @dataclass
    class MockSearchResult:
        chunk_id: str = "chunk_001"
        content: str = "Test content about quantum physics."
        score: float = 0.85
        document_id: str = "doc_001"
        section_title: str = "Introduction"
        chunk_type: str = "text"
        source_file: str = "/path/to/source.pdf"
        word_count: int = 5
        page_start: Optional[int] = 1
        page_end: Optional[int] = 2
        library: str = "default"
        metadata: Optional[Dict[str, Any]] = None
        source_location: Optional[Dict[str, Any]] = None
        author_id: Optional[str] = "author_001"
        author_name: Optional[str] = "John Doe"

    return MockSearchResult()


class TestSearchEndpoint:
    """Test GET /v1/search endpoint."""

    def test_search_returns_structured_json_with_citations(self, client):
        """Test search returns structured JSON with citations."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.retrieval.HybridRetriever"
        ) as mock_retriever_class:
            # Setup mock search result
            mock_result = MagicMock()
            mock_result.chunk_id = "chunk_001"
            mock_result.content = "Test content"
            mock_result.score = 0.85
            mock_result.document_id = "doc_001"
            mock_result.section_title = "Introduction"
            mock_result.source_file = "/path/to/file.pdf"
            mock_result.word_count = 10
            mock_result.page_start = 1
            mock_result.page_end = 2
            mock_result.library = "default"
            mock_result.metadata = {}
            mock_result.author_id = None
            mock_result.author_name = None

            mock_retriever = MagicMock()
            mock_retriever.search.return_value = [mock_result]
            mock_retriever_class.return_value = mock_retriever

            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_pipeline.return_value.config = mock_config
            mock_pipeline.return_value.storage = mock_storage

            # Make request
            response = client.get("/v1/search?query=test")

            # Verify response structure
            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["query"] == "test"
            assert data["total_results"] == 1
            assert len(data["results"]) == 1

            # Verify citation fields
            result = data["results"][0]
            assert "citation" in result
            citation = result["citation"]

            assert citation["source_url"] == "/path/to/file.pdf"
            assert citation["chunk_id"] == "chunk_001"
            assert citation["relevance_score"] == 0.85
            assert citation["page_start"] == 1
            assert citation["page_end"] == 2
            assert citation["section_title"] == "Introduction"

    def test_top_k_default_value(self, client):
        """Test top_k defaults to 10."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.retrieval.HybridRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            mock_retriever_class.return_value = mock_retriever

            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_pipeline.return_value.config = mock_config
            mock_pipeline.return_value.storage = mock_storage

            # Make request without top_k
            response = client.get("/v1/search?query=test")

            assert response.status_code == 200

            # Verify top_k was passed as default (10)
            mock_retriever.search.assert_called_once()
            call_kwargs = mock_retriever.search.call_args.kwargs
            assert call_kwargs.get("top_k") == 10

    def test_top_k_lower_bound(self, client):
        """Test top_k minimum value is 1."""
        # top_k=0 should fail validation
        response = client.get("/v1/search?query=test&top_k=0")
        assert response.status_code == 422  # Validation error

        # top_k=-1 should fail validation
        response = client.get("/v1/search?query=test&top_k=-1")
        assert response.status_code == 422

    def test_top_k_upper_bound(self, client):
        """Test top_k maximum value is 100."""
        # top_k=101 should fail validation
        response = client.get("/v1/search?query=test&top_k=101")
        assert response.status_code == 422  # Validation error

        # top_k=1000 should fail validation
        response = client.get("/v1/search?query=test&top_k=1000")
        assert response.status_code == 422

    def test_top_k_valid_bounds(self, client):
        """Test top_k accepts valid values at boundaries."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.retrieval.HybridRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            mock_retriever_class.return_value = mock_retriever

            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_pipeline.return_value.config = mock_config
            mock_pipeline.return_value.storage = mock_storage

            # top_k=1 should work
            response = client.get("/v1/search?query=test&top_k=1")
            assert response.status_code == 200

            # top_k=100 should work
            response = client.get("/v1/search?query=test&top_k=100")
            assert response.status_code == 200

    def test_empty_query_handling(self, client):
        """Test empty query returns error."""
        # Empty query parameter
        response = client.get("/v1/search?query=")
        assert response.status_code == 422  # Validation error (min_length=1)

    def test_whitespace_query_handling(self, client):
        """Test whitespace-only query returns error."""
        # Whitespace-only query
        response = client.get("/v1/search?query=%20%20%20")  # URL-encoded spaces
        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_missing_query_handling(self, client):
        """Test missing query parameter returns error."""
        response = client.get("/v1/search")
        assert response.status_code == 422  # Missing required parameter

    def test_project_id_filtering(self, client):
        """Test project_id parameter is passed to retriever."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.retrieval.HybridRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            mock_retriever_class.return_value = mock_retriever

            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_pipeline.return_value.config = mock_config
            mock_pipeline.return_value.storage = mock_storage

            # Make request with project_id
            response = client.get("/v1/search?query=test&project_id=my_project")

            assert response.status_code == 200

            # Verify project_id was passed as library_filter
            mock_retriever.search.assert_called_once()
            call_kwargs = mock_retriever.search.call_args.kwargs
            assert call_kwargs.get("library_filter") == "my_project"

    def test_search_with_no_results(self, client):
        """Test search with no results returns empty list."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.retrieval.HybridRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever.search.return_value = []
            mock_retriever_class.return_value = mock_retriever

            mock_config = MagicMock()
            mock_storage = MagicMock()
            mock_pipeline.return_value.config = mock_config
            mock_pipeline.return_value.storage = mock_storage

            response = client.get("/v1/search?query=nonexistent")

            assert response.status_code == 200
            data = response.json()

            assert data["success"] is True
            assert data["results"] == []
            assert data["total_results"] == 0


class TestMapSearchResultToItem:
    """Test _map_search_result_to_item helper."""

    def test_maps_all_fields(self, mock_search_result):
        """Test all fields are mapped correctly."""
        result = _map_search_result_to_item(mock_search_result)

        assert result.content == "Test content about quantum physics."
        assert result.document_id == "doc_001"
        assert result.word_count == 5
        assert result.library == "default"
        assert result.author_id == "author_001"
        assert result.author_name == "John Doe"

    def test_maps_citation_fields(self, mock_search_result):
        """Test citation fields are mapped correctly."""
        result = _map_search_result_to_item(mock_search_result)

        assert result.citation.source_url == "/path/to/source.pdf"
        assert result.citation.chunk_id == "chunk_001"
        assert result.citation.relevance_score == 0.85
        assert result.citation.page_start == 1
        assert result.citation.page_end == 2
        assert result.citation.section_title == "Introduction"

    def test_handles_none_metadata(self, mock_search_result):
        """Test handles None metadata gracefully."""
        mock_search_result.metadata = None
        result = _map_search_result_to_item(mock_search_result)

        assert result.metadata == {}

    def test_handles_none_optional_fields(self, mock_search_result):
        """Test handles None optional fields."""
        mock_search_result.page_start = None
        mock_search_result.page_end = None
        mock_search_result.section_title = None
        mock_search_result.author_id = None
        mock_search_result.author_name = None

        result = _map_search_result_to_item(mock_search_result)

        assert result.citation.page_start is None
        assert result.citation.page_end is None
        assert result.citation.section_title is None
        assert result.author_id is None
        assert result.author_name is None


class TestNormalizeScore:
    """Test _normalize_score helper."""

    def test_normalize_within_range(self):
        """Test score normalization within range."""
        assert _normalize_score(0.5, 1.0) == 0.5
        assert _normalize_score(0.0, 1.0) == 0.0
        assert _normalize_score(1.0, 1.0) == 1.0

    def test_normalize_with_different_max(self):
        """Test normalization with different max score."""
        assert _normalize_score(50.0, 100.0) == 0.5
        assert _normalize_score(25.0, 100.0) == 0.25

    def test_normalize_clamps_above_one(self):
        """Test scores above 1.0 are clamped."""
        assert _normalize_score(1.5, 1.0) == 1.0
        assert _normalize_score(200.0, 100.0) == 1.0

    def test_normalize_clamps_below_zero(self):
        """Test negative scores are clamped to 0."""
        assert _normalize_score(-0.5, 1.0) == 0.0

    def test_normalize_with_zero_max(self):
        """Test handles zero max score gracefully."""
        assert _normalize_score(0.5, 0.0) == 0.0
        assert _normalize_score(0.0, 0.0) == 0.0

    def test_normalize_with_negative_max(self):
        """Test handles negative max score gracefully."""
        assert _normalize_score(0.5, -1.0) == 0.0


class TestCitationInfoModel:
    """Test CitationInfo Pydantic model."""

    def test_valid_citation(self):
        """Test valid citation creation."""
        citation = CitationInfo(
            source_url="/path/to/file.pdf",
            chunk_id="chunk_001",
            relevance_score=0.85,
            page_start=1,
            page_end=5,
            section_title="Introduction",
        )

        assert citation.source_url == "/path/to/file.pdf"
        assert citation.chunk_id == "chunk_001"
        assert citation.relevance_score == 0.85

    def test_score_validation(self):
        """Test relevance_score bounds validation."""
        # Valid scores
        CitationInfo(chunk_id="c1", relevance_score=0.0)
        CitationInfo(chunk_id="c2", relevance_score=1.0)
        CitationInfo(chunk_id="c3", relevance_score=0.5)

        # Invalid scores
        with pytest.raises(ValueError):
            CitationInfo(chunk_id="c4", relevance_score=-0.1)

        with pytest.raises(ValueError):
            CitationInfo(chunk_id="c5", relevance_score=1.1)


class TestSearchResponseModel:
    """Test SearchResponse Pydantic model."""

    def test_default_values(self):
        """Test default values are set correctly."""
        response = SearchResponse(success=True, query="test")

        assert response.results == []
        assert response.total_results == 0
        assert response.query_time_ms == 0.0
        assert response.message == ""

    def test_full_response(self):
        """Test full response with all fields."""
        citation = CitationInfo(
            chunk_id="c1",
            relevance_score=0.9,
        )
        result = SearchResultItem(
            content="Test content",
            document_id="doc_001",
            citation=citation,
        )
        response = SearchResponse(
            success=True,
            results=[result],
            total_results=1,
            query="test query",
            query_time_ms=15.5,
            message="Found 1 result",
        )

        assert response.success is True
        assert len(response.results) == 1
        assert response.query == "test query"
