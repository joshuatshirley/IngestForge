"""
Integration Tests for Nexus End-to-End Lifecycle - Task 273.
Verifies Broadcast -> Remote Processing -> Fusion -> Attribution.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
import respx
from httpx import Response
from pathlib import Path
from ingestforge.core.models.search import SearchQuery, SearchResult, SearchResponse
from ingestforge.core.models.nexus import NexusPeer, NexusStatus
from ingestforge.core.pipeline.nexus_broadcast import NexusBroadcaster
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.config.nexus import NexusConfig

# Test Constants
MOCK_PEER_ID = "mock-nex"
MOCK_PEER_URL = "https://mock-peer.local"  # Removed trailing slash
LOCAL_NEXUS_ID = "local-nex"


@pytest.fixture
def mock_registry(tmp_path):
    registry = NexusRegistry(tmp_path)
    peer = NexusPeer(
        id=MOCK_PEER_ID,
        name="Mock University Nexus",
        url=MOCK_PEER_URL,
        api_key_hash="test-hash",
        status=NexusStatus.ONLINE,
    )
    registry.add_peer(peer)
    return registry


@pytest.fixture
def broadcaster(mock_registry):
    config = NexusConfig(
        nexus_id=LOCAL_NEXUS_ID,
        cert_file=Path("cert.pem"),
        key_file=Path("key.pem"),
        trust_store=Path("trust"),
    )
    return NexusBroadcaster(mock_registry, config)


# =============================================================================
# GIVEN: A registered online peer
# =============================================================================


@pytest.mark.asyncio
async def test_e2e_given_valid_query_when_broadcast_then_fuses_remote_results(
    broadcaster, mock_registry
):
    # Given
    query = SearchQuery(text="quantum propulsion", broadcast=True)

    # Mock Remote Response
    remote_result = SearchResult(
        chunk_id="remote-1",
        content="Evidence of quantum thrust.",
        score=0.95,
        nexus_id=MOCK_PEER_ID,
        confidence=0.95,
        artifact_id="art-1",
        document_id="doc-1",
    )
    remote_response = SearchResponse(results=[remote_result], peer_failures=[])

    with respx.mock(base_url=MOCK_PEER_URL) as respx_mock:
        respx_mock.post("/v1/nexus/search").mock(
            return_value=Response(200, json=remote_response.model_dump(mode="json"))
        )

        # When
        final_response = await broadcaster.broadcast(query)

        # Then
        assert len(final_response.results) == 1
        result = final_response.results[0]
        assert result.content == "Evidence of quantum thrust."
        assert result.nexus_id == MOCK_PEER_ID
        assert (
            result.nexus_name == "Mock University Nexus"
        )  # Attribution Logic Verified (Task 140)


@pytest.mark.asyncio
async def test_e2e_given_unauthorized_peer_when_broadcast_then_logs_failure(
    broadcaster, mock_registry
):
    # Given
    query = SearchQuery(text="restricted data", broadcast=True)

    with respx.mock(base_url=MOCK_PEER_URL) as respx_mock:
        # Simulate 403 from remote middleware (Task 120/121 logic)
        respx_mock.post("/v1/nexus/search").mock(
            return_value=Response(403, json={"detail": "Access Denied"})
        )

        # When
        final_response = await broadcaster.broadcast(query)

        # Then
        assert len(final_response.results) == 0
        assert len(final_response.peer_failures) == 1
        failure = final_response.peer_failures[0]
        assert failure.nexus_id == MOCK_PEER_ID
        assert "403" in failure.message


@pytest.mark.asyncio
async def test_e2e_given_timeout_when_broadcast_then_handles_gracefully(
    broadcaster, mock_registry
):
    # Given
    query = SearchQuery(text="slow query", broadcast=True)

    with respx.mock(base_url=MOCK_PEER_URL) as respx_mock:
        respx_mock.post("/v1/nexus/search").mock(side_effect=TimeoutError)

        # When
        final_response = await broadcaster.broadcast(query)

        # Then
        assert len(final_response.results) == 0
        assert len(final_response.peer_failures) == 1
        # Broadcaster should catch exception and wrap it
        assert final_response.peer_failures[0].nexus_id == MOCK_PEER_ID
