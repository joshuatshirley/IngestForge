"""
GWT Unit Tests for Nexus API Routes - Task 272.
Verifies the remote search endpoint logic and result enrichment.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request
from ingestforge.api.routes.nexus import nexus_remote_search
from ingestforge.core.models.search import SearchQuery, SearchResult


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.state = MagicMock()
    request.state.authorized_library = "shared_lib"
    return request


@pytest.fixture
def mock_pipeline():
    with patch("ingestforge.api.routes.nexus.Pipeline") as mock:
        pipeline = mock.return_value
        pipeline.config.nexus.nexus_id = "local-node-1"
        yield pipeline


# =============================================================================
# SCENARIO: Receiving a Remote Search Request
# =============================================================================


@pytest.mark.asyncio
async def test_remote_search_given_valid_query_when_called_then_returns_enriched_results(
    mock_request, mock_pipeline
):
    # Given
    query = SearchQuery(text="federated data", top_k=5)
    mock_results = [
        SearchResult(
            content="result 1",
            score=0.9,
            confidence=1.0,
            artifact_id="art-1",
            document_id="doc-1",
        )
    ]

    with patch("ingestforge.api.routes.nexus.HybridRetriever") as mock_retriever_cls:
        retriever = mock_retriever_cls.return_value
        retriever.search.return_value = mock_results

        # When
        response = await nexus_remote_search(query, mock_request)

        # Then
        assert response.total_hits == 1
        assert response.results[0].nexus_id == "local-node-1"
        assert response.results[0].nexus_name == "Remote Nexus"

        # Verify retriever was called with the library from request state
        retriever.search.assert_called_once()
        args, kwargs = retriever.search.call_args
        assert kwargs["library_filter"] == "shared_lib"
