"""
Nexus Role-Based Access Control (RBAC) Service.

Task 121: Resolution of role-based permissions for federated peers.
JPL Power of Ten: Rule #4 (Small functions), Rule #9 (Type hints).
"""

from typing import Dict, Set
from ingestforge.core.models.nexus_acl import NexusRole

# Mapping of roles to authorized actions
ROLE_PERMISSIONS: Dict[NexusRole, Set[str]] = {
    NexusRole.READ_ONLY: {"nexus:search", "nexus:view_artifact"},
    NexusRole.CONTRIBUTOR: {
        "nexus:search",
        "nexus:view_artifact",
        "nexus:annotate",
        "nexus:link",
    },
    NexusRole.ADMIN: {
        "nexus:search",
        "nexus:view_artifact",
        "nexus:annotate",
        "nexus:link",
        "nexus:reindex",
        "nexus:purge",
    },
}


class NexusRBAC:
    """
    Validates if a peer role is authorized for a specific action.
    """

    @staticmethod
    def is_action_allowed(role: NexusRole, action: str) -> bool:
        """
        Check if action is in the role's permission set.
        Rule #4: Logic under 10 lines.
        Rule #5: Assert input validity at entry point.
        """
        assert action and len(action.strip()) > 0, "Action string cannot be empty"

        permissions = ROLE_PERMISSIONS.get(role, set())
        return action in permissions

    @staticmethod
    def get_actions_for_role(role: NexusRole) -> Set[str]:
        """List all actions for a role."""
        return ROLE_PERMISSIONS.get(role, set())
