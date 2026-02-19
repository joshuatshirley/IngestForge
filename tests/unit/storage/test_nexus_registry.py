"""
GWT Tests for Nexus Registry - Task 131.
"""

import pytest
import json
from pathlib import Path
from ingestforge.core.models.nexus import NexusPeer, NexusStatus
from ingestforge.storage.nexus_registry import NexusRegistry


@pytest.fixture
def temp_registry(tmp_path: Path) -> NexusRegistry:
    return NexusRegistry(tmp_path)


# =============================================================================
# GIVEN: A NexusRegistry and a valid NexusPeer
# =============================================================================


def test_registry_given_new_peer_when_added_then_persists_to_disk(
    temp_registry, tmp_path
):
    # Given
    peer = NexusPeer(
        id="nex", name="Team Alpha", url="http://nexus-a.local", api_key_hash="hash123"
    )

    # When
    temp_registry.add_peer(peer)

    # Then
    storage_file = tmp_path / "nexus_peers.json"
    assert storage_file.exists()

    with open(storage_file, "r") as f:
        data = json.load(f)
        assert "nex" in data
        assert data["nex"]["name"] == "Team Alpha"


def test_registry_given_offline_peer_when_status_updated_then_state_changes(
    temp_registry,
):
    # Given
    peer = NexusPeer(
        id="n1", name="N1", url="http://n1", api_key_hash="h", status=NexusStatus.ONLINE
    )
    temp_registry.add_peer(peer)

    # When
    temp_registry.update_status("n1", NexusStatus.OFFLINE)

    # Then
    updated = temp_registry.get_peer("n1")
    assert updated.status == NexusStatus.OFFLINE


def test_registry_given_revoked_peer_when_listing_active_then_excluded(temp_registry):
    # Given
    p1 = NexusPeer(
        id="p1", name="P1", url="http://p1", api_key_hash="h", status=NexusStatus.ONLINE
    )
    p2 = NexusPeer(
        id="p2",
        name="P2",
        url="http://p2",
        api_key_hash="h",
        status=NexusStatus.REVOKED,
    )
    temp_registry.add_peer(p1)
    temp_registry.add_peer(p2)

    # When
    active = temp_registry.list_active_peers()

    # Then
    assert len(active) == 1
    assert active[0].id == "p1"
