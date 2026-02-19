"""Comprehensive GWT unit tests for POST search endpoint.

NL Search Redesign.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch

from ingestforge.api.main import app

client = TestClient(app)

# =============================================================================
# API ROUTE TESTS (POST /search)
# =============================================================================


def test_search_post_endpoint_success():
    """GIVEN a valid SearchRequest
    WHEN POST /v1/search is called
    THEN it returns 200 OK and valid results.
    """
    mock_results = []  # Empty results list for simple test

    with patch("ingestforge.core.pipeline.pipeline.Pipeline"), patch(
        "ingestforge.retrieval.HybridRetriever.search"
    ) as mock_search:
        mock_search.return_value = mock_results

        request_data = {
            "query": "What is quantum gravity?",
            "top_k": 5,
            "filters": {"doc_type": "PDF"},
            "sort_by": "date",
        }

        response = client.post("/v1/search", json=request_data)

        assert response.status_code == 200
        assert response.json()["success"] is True
        assert response.json()["query"] == "What is quantum gravity?"

        # Verify retriever was called with correct filter/sort
        mock_search.assert_called_once_with(
            query="What is quantum gravity?",
            top_k=5,
            library_filter=None,
            metadata_filter={"doc_type": "PDF"},
            sort_by="date",
        )


def test_search_post_validation_error():
    """GIVEN an invalid SearchRequest (empty query)
    WHEN POST /v1/search is called
    THEN it returns 422 Unprocessable Entity.
    """
    response = client.post("/v1/search", json={"query": "   ", "top_k": 5})
    assert response.status_code == 422  # Pydantic validator failure


def test_search_post_large_query_limit():
    """GIVEN a query exceeding 10,000 chars
    WHEN POST /v1/search is called
    THEN it returns 422.
    """
    long_query = "a" * 10001
    response = client.post("/v1/search", json={"query": long_query})
    assert response.status_code == 422
