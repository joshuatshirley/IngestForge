"""
GWT Unit Tests for Nexus Broadcaster - Task 128.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
import httpx
from unittest.mock import MagicMock, patch
from ingestforge.core.pipeline.nexus_broadcast import NexusBroadcaster
from ingestforge.core.models.search import SearchQuery
from ingestforge.core.models.nexus_error import PeerErrorType
from ingestforge.core.models.nexus import NexusPeer
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.config.nexus import NexusConfig


@pytest.fixture
def mock_registry():
    reg = MagicMock(spec=NexusRegistry)
    peer = MagicMock(spec=NexusPeer)
    peer.url = "http://peer-a.local"
    peer.id = "peer-a"
    reg.list_active_peers.return_value = [peer]
    return reg


@pytest.fixture
def mock_config():
    conf = MagicMock(spec=NexusConfig)
    conf.nexus_id = "local-nexus"
    conf.max_peers = 10
    return conf


# =============================================================================
# GIVEN: A search broadcast request
# =============================================================================


@pytest.mark.asyncio
async def test_broadcast_given_active_peer_when_search_sent_then_returns_remote_results(
    mock_registry, mock_config
):
    # Given
    query = SearchQuery(text="test", broadcast=True)
    broadcaster = NexusBroadcaster(mock_registry, mock_config)

    remote_results = [
        {
            "content": "Match from A",
            "score": 0.9,
            "confidence": 0.8,
            "artifact_id": "1",
            "document_id": "d1",
        }
    ]

    # Mock the HTTP response
    with patch("httpx.AsyncClient.post") as mock_post:
        mock_post.return_value = MagicMock(
            status_code=200, json=lambda: {"results": remote_results, "total_hits": 1}
        )

        # When
        response = await broadcaster.broadcast(query)

        # Then
        assert len(response.results) == 1
        assert response.results[0].content == "Match from A"
        assert len(response.peer_failures) == 0
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_broadcast_given_peer_timeout_when_search_sent_then_returns_failure_data(
    mock_registry, mock_config
):
    # Given
    query = SearchQuery(text="test", broadcast=True)
    broadcaster = NexusBroadcaster(mock_registry, mock_config)

    # Mock a network timeout
    with patch(
        "httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Too slow")
    ):
        # When
        response = await broadcaster.broadcast(query)

        # Then
        assert len(response.results) == 0
        assert len(response.peer_failures) == 1
        assert response.peer_failures[0].error_type == PeerErrorType.TIMEOUT


@pytest.mark.asyncio
async def test_broadcast_given_broadcast_false_when_called_then_skips_network(
    mock_registry, mock_config
):
    # Given
    query = SearchQuery(text="test", broadcast=False)
    broadcaster = NexusBroadcaster(mock_registry, mock_config)

    # When
    response = await broadcaster.broadcast(query)

    # Then
    assert len(response.results) == 0
    mock_registry.list_active_peers.assert_not_called()
