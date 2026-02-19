"""
GWT Unit Tests for Search Schema - Task 151.
NASA JPL Power of Ten Compliance: Rule #4, Rule #9.
"""

import pytest
from pydantic import ValidationError
from ingestforge.core.models.search import SearchQuery, SearchResult, SearchResponse

# =============================================================================
# GIVEN: A SearchQuery model
# =============================================================================


def test_search_query_given_valid_data_when_initialized_then_success():
    # Given
    data = {
        "text": "Find legal precedents",
        "top_k": 20,
        "filters": {"domain": "legal"},
        "broadcast": True,
    }

    # When
    query = SearchQuery(**data)

    # Then
    assert query.text == "Find legal precedents"
    assert query.top_k == 20
    assert query.broadcast is True
    assert query.filters["domain"] == "legal"


def test_search_query_given_empty_text_when_initialized_then_raises_error():
    # Given
    data = {"text": "   "}

    # When / Then
    with pytest.raises(ValidationError) as exc:
        SearchQuery(**data)
    assert "Query text cannot be empty" in str(exc.value)


def test_search_query_given_too_many_filters_when_initialized_then_raises_error():
    # Given (JPL Rule #2 enforcement)
    data = {
        "text": "test",
        "filters": {f"key_{i}": i for i in range(21)},  # Max is 20
    }

    # When / Then
    with pytest.raises(ValidationError) as exc:
        SearchQuery(**data)
    assert "Too many filters" in str(exc.value)


# =============================================================================
# GIVEN: A SearchResult model
# =============================================================================


def test_search_result_given_remote_source_when_initialized_then_preserves_nexus_id():
    # Given
    data = {
        "content": "Evidence found in Peer Nexus",
        "score": 0.9,
        "confidence": 0.95,
        "artifact_id": "art_789",
        "document_id": "doc_000",
        "nexus_id": "nexus-alpha-123",
    }

    # When
    result = SearchResult(**data)

    # Then
    assert result.nexus_id == "nexus-alpha-123"
    assert result.score == 0.9


def test_search_result_given_missing_fields_when_initialized_then_raises_error():
    # Given
    incomplete_data = {"content": "no ids"}

    # When / Then
    with pytest.raises(ValidationError):
        SearchResult(**incomplete_data)


# =============================================================================
# GIVEN: A SearchResponse model
# =============================================================================


def test_search_response_given_empty_results_when_initialized_then_defaults_applied():
    # When
    response = SearchResponse()

    # Then
    assert response.results == []
    assert response.total_hits == 0
    assert response.nexus_count == 1  # Default local
