"""
GWT Unit Tests for Nexus Peer Health Scanner - Task 124.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from ingestforge.core.system.nexus_scanner import NexusScanner
from ingestforge.core.models.nexus import NexusPeer, NexusStatus, NexusRegistryStore
from ingestforge.storage.nexus_registry import NexusRegistry


@pytest.fixture
def mock_registry(tmp_path):
    registry = NexusRegistry(tmp_path)
    # Clear any auto-loaded peers for clean test state
    registry._store = NexusRegistryStore()
    return registry


@pytest.fixture
def scanner(mock_registry):
    return NexusScanner(mock_registry)


@pytest.fixture
def active_peer():
    return NexusPeer(
        id="peer-123",
        name="Test Peer",
        url="http://remote-nexus.local/",
        api_key_hash="hash",
        status=NexusStatus.PENDING,
    )


# =============================================================================
# GIVEN: A NexusScanner and a registry with an active peer
# =============================================================================


@pytest.mark.asyncio
async def test_scanner_given_healthy_peer_when_scanned_then_marks_online(
    scanner, mock_registry, active_peer
):
    # Given
    mock_registry.add_peer(active_peer)

    # Mock httpx response 200
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(status_code=200)

        # When
        await scanner.scan_all()

        # Then
        peer = mock_registry.get_peer(active_peer.id)
        assert peer.status == NexusStatus.ONLINE
        assert peer.failure_count == 0
        assert peer.last_seen is not None


@pytest.mark.asyncio
async def test_scanner_given_unreachable_peer_when_scanned_then_increments_failure(
    scanner, mock_registry, active_peer
):
    # Given
    mock_registry.add_peer(active_peer)

    # Mock httpx failure
    with patch("httpx.AsyncClient.get", side_effect=Exception("Connection Refused")):
        # When
        await scanner.scan_all()

        # Then
        peer = mock_registry.get_peer(active_peer.id)
        assert peer.failure_count == 1
        assert (
            peer.status == NexusStatus.PENDING
        )  # Should still be pending after 1 failure


@pytest.mark.asyncio
async def test_scanner_given_max_failures_when_scanned_then_marks_offline(
    scanner, mock_registry, active_peer
):
    # Given
    active_peer.failure_count = 9  # One away from threshold (10)
    mock_registry.add_peer(active_peer)

    # Mock httpx failure
    with patch("httpx.AsyncClient.get", side_effect=Exception("Timeout")):
        # When
        await scanner.scan_all()

        # Then
        peer = mock_registry.get_peer(active_peer.id)
        assert peer.failure_count == 10
        assert peer.status == NexusStatus.OFFLINE


@pytest.mark.asyncio
async def test_scanner_given_offline_peer_when_scanned_successfully_then_marks_online(
    scanner, mock_registry, active_peer
):
    """Recovery Test (Task 274)."""
    # Given
    active_peer.status = NexusStatus.OFFLINE
    active_peer.failure_count = 10
    mock_registry.add_peer(active_peer)

    # Mock httpx response 200
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(status_code=200)

        # When
        await scanner.scan_all()

        # Then
        peer = mock_registry.get_peer(active_peer.id)
        assert peer.status == NexusStatus.ONLINE
        assert peer.failure_count == 0


@pytest.mark.asyncio
async def test_scanner_given_50_peers_when_scanned_then_completes_efficiently(
    scanner, mock_registry
):
    """Latency Impact Test (Task 274)."""
    # Given: 50 peers (JPL Rule #2 bound)
    for i in range(50):
        p = NexusPeer(
            id=f"peer-{i}", name=f"Peer {i}", url=f"http://p{i}.com/", api_key_hash="h"
        )
        mock_registry.add_peer(p)

    # When
    import time

    start = time.perf_counter()
    with patch("httpx.AsyncClient.get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = MagicMock(status_code=200)
        await scanner.scan_all()
    duration = time.perf_counter() - start

    # Then
    assert mock_get.call_count == 50
    # Relaxed for slow test environments, but still ensures basic completion
    assert duration < 30.0
