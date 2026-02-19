"""
GWT Unit Tests for Nexus Kill-Switch - Task 123.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
from ingestforge.core.security.nexus_acl import NexusACLManager
from ingestforge.core.security.nexus_blacklist import NexusBlacklist
from ingestforge.core.models.nexus_acl import NexusRole

# =============================================================================
# GIVEN: A system with registered peers
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path):
    return tmp_path


def test_kill_given_authorized_peer_when_revoked_then_all_permissions_cleared(
    temp_data_dir,
):
    # Given
    acl = NexusACLManager(temp_data_dir)
    peer_id = "rogue-peer-123"
    acl.grant_access(peer_id, "lib-a")
    acl.grant_access(peer_id, "lib-b")

    # When
    acl.revoke_all_access(peer_id)

    # Then
    entry = acl._store.entries[peer_id]
    assert len(entry.allowed_libraries) == 0
    assert entry.role == NexusRole.READ_ONLY
    assert acl.is_authorized(peer_id, "lib-a") is False


def test_kill_given_blacklist_when_peer_added_then_fails_check(temp_data_dir):
    # Given
    blacklist = NexusBlacklist(temp_data_dir)
    peer_id = "blocked-node"

    # When
    blacklist.add(peer_id)

    # Then
    assert blacklist.is_revoked(peer_id) is True

    # Verify Persistence
    new_blacklist = NexusBlacklist(temp_data_dir)
    assert new_blacklist.is_revoked(peer_id) is True


def test_kill_given_blacklist_when_removed_then_passes_check(temp_data_dir):
    # Given
    blacklist = NexusBlacklist(temp_data_dir)
    peer_id = "restored-node"
    blacklist.add(peer_id)

    # When
    blacklist.remove(peer_id)

    # Then
    assert blacklist.is_revoked(peer_id) is False


def test_kill_given_blacklist_bounds_when_exceeded_then_ignores_new_entries(
    temp_data_dir,
):
    # Given
    blacklist = NexusBlacklist(temp_data_dir)
    # Fill to bound (Simulate Rule #2 compliance)
    for i in range(1000):
        blacklist._revoked_ids.add(f"peer-{i}")

    # When
    blacklist.add("extra-peer")

    # Then
    assert blacklist.is_revoked("extra-peer") is False
    assert len(blacklist._revoked_ids) == 1000
