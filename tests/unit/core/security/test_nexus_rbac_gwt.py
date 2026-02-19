"""
GWT Unit Tests for Nexus RBAC - Task 121.
NASA JPL Compliance: Rule #4, Rule #9.
"""

from ingestforge.core.security.nexus_rbac import NexusRBAC
from ingestforge.core.models.nexus_acl import NexusRole

# =============================================================================
# GIVEN: A peer with a specific role
# =============================================================================


def test_rbac_given_readonly_role_when_searching_then_allowed():
    # Given
    role = NexusRole.READ_ONLY
    action = "nexus:search"

    # When
    allowed = NexusRBAC.is_action_allowed(role, action)

    # Then
    assert allowed is True


def test_rbac_given_readonly_role_when_annotating_then_denied():
    # Given
    role = NexusRole.READ_ONLY
    action = "nexus:annotate"

    # When
    allowed = NexusRBAC.is_action_allowed(role, action)

    # Then
    assert allowed is False


def test_rbac_given_contributor_role_when_annotating_then_allowed():
    # Given
    role = NexusRole.CONTRIBUTOR
    action = "nexus:annotate"

    # When
    allowed = NexusRBAC.is_action_allowed(role, action)

    # Then
    assert allowed is True


def test_rbac_given_admin_role_when_purging_then_allowed():
    # Given
    role = NexusRole.ADMIN
    action = "nexus:purge"

    # When
    allowed = NexusRBAC.is_action_allowed(role, action)

    # Then
    assert allowed is True


def test_rbac_given_invalid_action_when_checked_then_denied():
    # Given
    role = NexusRole.ADMIN
    action = "nexus:destroy_universe"

    # When
    allowed = NexusRBAC.is_action_allowed(role, action)

    # Then
    assert allowed is False


def test_rbac_given_role_when_retrieving_actions_then_returns_correct_set():
    # Given
    role = NexusRole.READ_ONLY

    # When
    actions = NexusRBAC.get_actions_for_role(role)

    # Then
    assert "nexus:search" in actions
    assert "nexus:annotate" not in actions
    assert len(actions) == 2
