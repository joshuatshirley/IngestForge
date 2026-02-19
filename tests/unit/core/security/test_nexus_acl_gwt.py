"""
GWT Unit Tests for Nexus ACL - Task 120.
"""

import pytest
from pathlib import Path
from ingestforge.core.security.nexus_acl import NexusACLManager
from ingestforge.core.models.nexus_acl import NexusAccessScope


@pytest.fixture
def temp_acl_manager(tmp_path: Path) -> NexusACLManager:
    return NexusACLManager(tmp_path)


# =============================================================================
# GIVEN: A NexusACLManager
# =============================================================================


def test_acl_given_new_peer_when_checked_then_denies_by_default(temp_acl_manager):
    # Then
    assert temp_acl_manager.is_authorized("peer-999", "lib-any") is False


def test_acl_given_granted_access_when_checked_then_allows(temp_acl_manager):
    # When
    temp_acl_manager.grant_access("peer-1", "lib-alpha")

    # Then
    assert temp_acl_manager.is_authorized("peer-1", "lib-alpha") is True
    assert temp_acl_manager.is_authorized("peer-1", "lib-beta") is False


def test_acl_given_revoked_access_when_checked_then_denies(temp_acl_manager):
    # Given
    temp_acl_manager.grant_access("peer-1", "lib-alpha")
    assert temp_acl_manager.is_authorized("peer-1", "lib-alpha") is True

    # When
    temp_acl_manager.revoke_access("peer-1", "lib-alpha")

    # Then
    assert temp_acl_manager.is_authorized("peer-1", "lib-alpha") is False


def test_acl_given_persistence_when_reloaded_then_preserves_rules(tmp_path):
    # Given
    mgr1 = NexusACLManager(tmp_path)
    mgr1.grant_access("peer-5", "lib-secret", scope=NexusAccessScope.FULL_TEXT)

    # When
    mgr2 = NexusACLManager(tmp_path)

    # Then
    assert mgr2.is_authorized("peer-5", "lib-secret") is True
    entry = mgr2._store.entries["peer-5"]
    assert entry.scope == NexusAccessScope.FULL_TEXT
