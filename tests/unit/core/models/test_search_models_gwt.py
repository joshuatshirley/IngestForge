"""
GWT Unit Tests for Unified Search Models - Task 272.
Verifies the target_peer_ids field and its interaction with the broadcast flag.
"""

import pytest
from pydantic import ValidationError
from ingestforge.core.models.search import SearchQuery

# =============================================================================
# SCENARIO: SearchQuery initialization with Peer Selection
# =============================================================================


def test_search_query_given_target_peers_when_initialized_then_holds_correct_ids():
    # Given
    target_ids = ["peer-a", "peer-b"]

    # When
    query = SearchQuery(text="test query", broadcast=True, target_peer_ids=target_ids)

    # Then
    assert query.target_peer_ids == target_ids
    assert query.broadcast is True


def test_search_query_given_null_targets_when_initialized_then_defaults_to_none():
    # Given / When
    query = SearchQuery(text="test query")

    # Then
    assert query.target_peer_ids is None
    assert query.broadcast is False


def test_search_query_given_empty_list_when_initialized_then_holds_empty_list():
    # Given
    target_ids = []

    # When
    query = SearchQuery(text="test query", broadcast=True, target_peer_ids=target_ids)

    # Then
    assert query.target_peer_ids == []


# =============================================================================
# SCENARIO: Validation Constraints (Rule #2)
# =============================================================================


def test_search_query_given_invalid_text_when_initialized_then_raises_error():
    # Given / When / Then
    with pytest.raises(ValidationError):
        SearchQuery(text="")  # Too short


def test_search_query_given_excessive_filters_when_initialized_then_raises_error():
    # Given
    heavy_filters = {f"key_{i}": i for i in range(21)}  # Max is 20

    # When / Then
    with pytest.raises(ValidationError):
        SearchQuery(text="valid", filters=heavy_filters)
