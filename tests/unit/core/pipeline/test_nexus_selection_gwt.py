"""
GWT Unit Tests for Surgical Nexus Peer Selection - Task 272.
Verifies that the Broadcaster correctly filters targets based on SearchQuery.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from ingestforge.core.pipeline.nexus_broadcast import NexusBroadcaster
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.search import SearchQuery
from ingestforge.core.models.nexus import NexusPeer, NexusStatus
from ingestforge.core.config.nexus import NexusConfig


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=NexusRegistry)
    registry.is_silenced.return_value = False
    # Define 3 active peers
    peers = [
        NexusPeer(
            id="peer-1",
            name="Peer 1",
            url="http://p1.com/",
            api_key_hash="h",
            status=NexusStatus.ONLINE,
        ),
        NexusPeer(
            id="peer-2",
            name="Peer 2",
            url="http://p2.com/",
            api_key_hash="h",
            status=NexusStatus.ONLINE,
        ),
        NexusPeer(
            id="peer-3",
            name="Peer 3",
            url="http://p3.com/",
            api_key_hash="h",
            status=NexusStatus.ONLINE,
        ),
    ]
    registry.list_active_peers.return_value = peers
    registry.get_peer.side_effect = lambda pid: next(
        (p for p in peers if p.id == pid), None
    )
    return registry


@pytest.fixture
def broadcaster(mock_registry):
    config = NexusConfig(
        nexus_id="local",
        cert_file=Path("c"),
        key_file=Path("k"),
        trust_store=Path("t"),
        max_peers=5,
    )
    return NexusBroadcaster(mock_registry, config)


# =============================================================================
# GIVEN: A broadcaster and multiple active peers
# =============================================================================


@pytest.mark.asyncio
async def test_selection_given_specific_ids_when_broadcast_then_only_queries_targets(
    broadcaster, mock_registry
):
    # Given: User selected only peer-1 and peer-3
    query = SearchQuery(
        text="quantum", broadcast=True, target_peer_ids=["peer-1", "peer-3"]
    )

    # When
    with patch.object(broadcaster, "_query_peer", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = []  # Empty results list
        await broadcaster.broadcast(query)

        # Then: Only 2 calls should be made
        assert mock_query.call_count == 2
        called_ids = [call.args[0] for call in mock_query.call_args_list]
        assert "peer-1" in called_ids
        assert "peer-3" in called_ids
        assert "peer-2" not in called_ids


@pytest.mark.asyncio
async def test_selection_given_no_ids_when_broadcast_then_queries_all_active(
    broadcaster, mock_registry
):
    # Given: target_peer_ids is None (Default)
    query = SearchQuery(text="quantum", broadcast=True, target_peer_ids=None)

    # When
    with patch.object(broadcaster, "_query_peer", new_callable=AsyncMock) as mock_query:
        mock_query.return_value = []
        await broadcaster.broadcast(query)

        # Then: All 3 peers should be queried
        assert mock_query.call_count == 3


@pytest.mark.asyncio
async def test_selection_given_invalid_id_when_broadcast_then_queries_zero_peers(
    broadcaster, mock_registry
):
    # Given: User provided an ID that doesn't exist in registry
    query = SearchQuery(text="quantum", broadcast=True, target_peer_ids=["ghost-peer"])

    # When
    with patch.object(broadcaster, "_query_peer", new_callable=AsyncMock) as mock_query:
        response = await broadcaster.broadcast(query)

        # Then
        assert mock_query.call_count == 0
        assert len(response.results) == 0
