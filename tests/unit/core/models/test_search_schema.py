"""
Unit tests for Search Schema - Task 151.
"""

import pytest
from pydantic import ValidationError
from ingestforge.core.models.search import SearchQuery, SearchResult


def test_search_query_validation():
    # Valid query
    query = SearchQuery(text="test query", top_k=5)
    assert query.text == "test query"
    assert query.top_k == 5
    assert query.broadcast is False

    # Empty query (invalid)
    with pytest.raises(ValidationError):
        SearchQuery(text="   ")

    # Too many filters (invalid - Rule #2)
    with pytest.raises(ValidationError):
        bad_filters = {f"key_{i}": i for i in range(25)}
        SearchQuery(text="test", filters=bad_filters)


def test_search_result_schema():
    result = SearchResult(
        content="snippet",
        score=0.95,
        confidence=0.88,
        artifact_id="art_123",
        document_id="doc_456",
    )
    assert result.nexus_id == "local"
    assert result.score == 0.95
    assert result.confidence == 0.88
