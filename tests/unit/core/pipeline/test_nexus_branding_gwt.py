"""
GWT Unit Tests for Nexus Branding & Attribution - Task 140.
NASA JPL Power of Ten: Rule #4, Rule #9.
"""

import pytest
from unittest.mock import MagicMock
from ingestforge.core.pipeline.nexus_broadcast import NexusBroadcaster
from ingestforge.core.models.search import SearchResult
from ingestforge.core.models.nexus import NexusPeer
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.config.nexus import NexusConfig


@pytest.fixture
def mock_registry():
    reg = MagicMock(spec=NexusRegistry)
    peer = NexusPeer(
        id="nexus-alpha", name="Team Alpha", url="http://alpha", api_key_hash="h"
    )
    reg.get_peer.side_effect = lambda pid: peer if pid == "nexus-alpha" else None
    return reg


@pytest.fixture
def mock_config():
    conf = MagicMock(spec=NexusConfig)
    conf.nexus_id = "local-nexus"
    return conf


# =============================================================================
# GIVEN: A collection of raw search results from multiple sources
# =============================================================================


def test_branding_given_remote_result_when_aggregated_then_enriches_nexus_name(
    mock_registry, mock_config
):
    # Given
    broadcaster = NexusBroadcaster(mock_registry, mock_config)
    raw_results = [
        SearchResult(
            content="match",
            artifact_id="1",
            document_id="d1",
            nexus_id="nexus-alpha",
            score=0.9,
            confidence=0.9,
        )
    ]

    # When
    response = broadcaster._aggregate_broadcast([raw_results])

    # Then
    assert response.results[0].nexus_id == "nexus-alpha"
    assert response.results[0].nexus_name == "Team Alpha"


def test_branding_given_local_result_when_aggregated_then_preserves_default_name(
    mock_registry, mock_config
):
    # Given
    broadcaster = NexusBroadcaster(mock_registry, mock_config)
    local_results = [
        SearchResult(
            content="local match",
            artifact_id="2",
            document_id="d2",
            nexus_id="local",
            score=0.95,
            confidence=0.9,
        )
    ]

    # When
    response = broadcaster._aggregate_broadcast([local_results])

    # Then
    assert response.results[0].nexus_id == "local"
    assert response.results[0].nexus_name == "Local Library"


def test_branding_given_unknown_peer_when_aggregated_then_graceful_fallback(
    mock_registry, mock_config
):
    # Given
    broadcaster = NexusBroadcaster(mock_registry, mock_config)
    unknown_results = [
        SearchResult(
            content="ghost match",
            artifact_id="3",
            document_id="d3",
            nexus_id="nexus-unknown",
            score=0.5,
            confidence=0.5,
        )
    ]

    # When
    response = broadcaster._aggregate_broadcast([unknown_results])

    # Then
    # Should still have default "Local Library" or maintain original if peer not found
    assert response.results[0].nexus_name == "Local Library"
