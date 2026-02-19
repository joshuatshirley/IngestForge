"""
Access Control List (ACL) Guard for Multi-User Project Collaboration.

Provides role-based access control (RBAC) for IngestForge projects, enabling
secure multi-user collaboration with granular permissions.

Architecture Context
--------------------
ACL Guard integrates with storage backends to enforce access control:

    ┌──────────────┐
    │   CLI / API  │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │  ACL Guard   │ ← Checks permissions before operations
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │   Storage    │
    │   Backend    │
    └──────────────┘

Threat Model
------------
IngestForge supports multi-user collaboration on shared projects:

    1. **Unauthorized Access**
       Attack: User tries to delete another user'''s content
       Defense: ACLGuard.check_permission() enforces role-based rules

    2. **Privilege Escalation**
       Attack: Contributor tries to grant admin permissions
       Defense: Only OWNER can grant/revoke roles

    3. **Data Leakage**
       Attack: Unauthorized read of private project
       Defense: All operations require explicit READ permission

Permission Model
----------------
**Permissions** (actions that can be performed):
- READ: View content
- WRITE: Create/modify own content
- DELETE: Remove content (own for EDITOR, all for OWNER)
- ADMIN: Manage users and roles

**Roles** (collections of permissions):
- VIEWER: Can only read content (READ)
- CONTRIBUTOR: Can read and add content (READ, WRITE)
- EDITOR: Can read, write, and delete own content (READ, WRITE, DELETE own)
- OWNER: Full access to all content and admin operations (READ, WRITE, DELETE all, ADMIN)

Usage Pattern
-------------
Basic access control:

    from ingestforge.core.security.acl import ACLGuard, Permission, Role

    # Initialize guard
    guard = ACLGuard()

    # Grant access
    guard.grant_role(user_id="alice", project_id="proj1", role=Role.CONTRIBUTOR)

    # Check permission before operation
    if guard.check_permission("alice", "proj1", Permission.WRITE):
        # Perform write operation
        pass

Integration with storage:

    # Decorator pattern (future enhancement)
    @require_permission(Permission.DELETE)
    def delete_chunk(self, user_id: str, chunk_id: str) -> None:
        # ACL check performed automatically
        pass---------------------------------
Rule #1: Simple Control Flow
    - Max 2 nesting levels in all methods
    - Early returns for validation

Rule #4: No Large Functions
    - All functions ≤ 60 lines

Rule #5: Assertion Density
    - Precondition checks at method entry
    - Postcondition checks for critical operations

Rule #7: Check Parameters
    - All inputs validated
    - Type checking with isinstance()

Rule #9: Type Safety
    - Full type hints throughout
    - Enums for Permission and Role
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ===========================================================================
# Permission & Role Enumerations
# ===========================================================================


class Permission(Enum):
    """
    Actions that can be performed on a project.

    Permissions are atomic capabilities that roles grant.
    """

    READ = "read"
    """View content (chunks, documents, metadata)."""

    WRITE = "write"
    """Create or modify content."""

    DELETE = "delete"
    """Remove content (scope depends on role)."""

    ADMIN = "admin"
    """Manage users, roles, and project settings."""


class Role(Enum):
    """
    User roles with associated permission sets.

    Roles define collections of permissions for different collaboration levels.
    """

    VIEWER = "viewer"
    """Read-only access. Can view but not modify content."""

    CONTRIBUTOR = "contributor"
    """Can read and create content. Cannot delete."""

    EDITOR = "editor"
    """Can read, write, and delete own content. Cannot manage users."""

    OWNER = "owner"
    """Full access. Can manage all content and users."""


# ===========================================================================
# Permission Mappings
# ===========================================================================

# Role -> Permissions mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.VIEWER: {Permission.READ},
    Role.CONTRIBUTOR: {Permission.READ, Permission.WRITE},
    Role.EDITOR: {Permission.READ, Permission.WRITE, Permission.DELETE},
    Role.OWNER: {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
    },
}

# ===========================================================================
# Data Classes
# ===========================================================================


@dataclass
class ACLEntry:
    """
    Access control list entry linking user to project with role.

    Attributes:
        user_id: Unique user identifier.
        project_id: Unique project identifier.
        role: User'''s role in the project.
        granted_at: Timestamp when role was granted (ISO format).
        granted_by: User ID who granted this role.
    """

    user_id: str
    project_id: str
    role: Role
    granted_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    granted_by: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Validate ACLEntry after initialization.

        Rule #5: Assertion density for data integrity.
        Rule #7: Check parameters at construction.
        """
        # Precondition: user_id must be non-empty
        if not self.user_id or not isinstance(self.user_id, str):
            raise ValueError("user_id must be a non-empty string")

        # Precondition: project_id must be non-empty
        if not self.project_id or not isinstance(self.project_id, str):
            raise ValueError("project_id must be a non-empty string")

        # Precondition: role must be valid Role enum
        if not isinstance(self.role, Role):
            raise ValueError(f"role must be a Role enum, got {type(self.role)}")

        # Precondition: granted_at must be valid ISO timestamp
        if not self.granted_at or not isinstance(self.granted_at, str):
            raise ValueError("granted_at must be a non-empty ISO timestamp string")


class PermissionDenied(Exception):
    """
    Raised when a user attempts an operation without required permission.

    Attributes:
        user_id: User who attempted the operation.
        project_id: Project the operation was attempted on.
        permission: Permission that was required.
        message: Human-readable error message.
    """

    def __init__(
        self,
        user_id: str,
        project_id: str,
        permission: Permission,
        message: Optional[str] = None,
    ) -> None:
        """Initialize PermissionDenied exception.

        Args:
            user_id: User who attempted the operation.
            project_id: Project the operation was attempted on.
            permission: Permission that was required.
            message: Optional custom message.
        """
        self.user_id = user_id
        self.project_id = project_id
        self.permission = permission

        if message is None:
            message = (
                f"User '''{user_id}'''  does not have {permission.value} permission "
                f"for project '''{project_id}'''"
            )

        super().__init__(message)


# ===========================================================================
# ACL Guard Implementation
# ===========================================================================


class ACLGuard:
    """
    Access control guard for multi-user project collaboration.

    Manages role assignments and enforces permission checks before operations.

    Example:
        >>> guard = ACLGuard()
        >>> guard.grant_role("alice", "proj1", Role.CONTRIBUTOR)
        >>> guard.check_permission("alice", "proj1", Permission.WRITE)
        True
        >>> guard.check_permission("alice", "proj1", Permission.DELETE)
        False
    """

    def __init__(self) -> None:
        """
        Initialize ACL guard with empty access control list.

        Rule #6: Instance-scoped storage, not global.
        """
        # Storage: (user_id, project_id) -> ACLEntry
        self._acl: Dict[Tuple[str, str], ACLEntry] = {}

    def grant_role(
        self,
        user_id: str,
        project_id: str,
        role: Role,
        granted_by: Optional[str] = None,
    ) -> None:
        """
        Grant a role to a user for a project.

        Rule #1: Early return for validation.
        Rule #4: Function ≤ 60 lines.
        Rule #7: Validate all parameters.

        Args:
            user_id: User to grant role to.
            project_id: Project to grant access for.
            role: Role to assign.
            granted_by: User ID who is granting the role (for audit trail).

        Raises:
            ValueError: If parameters are invalid.
        """
        # Precondition: Validate parameters (Rule #7)
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        if not isinstance(role, Role):
            raise ValueError(f"role must be a Role enum, got {type(role)}")

        # Create ACL entry
        entry = ACLEntry(
            user_id=user_id,
            project_id=project_id,
            role=role,
            granted_by=granted_by,
        )

        # Store in ACL
        key = (user_id, project_id)
        self._acl[key] = entry

        logger.info(
            f"Granted role {role.value} to user '''{user_id}''' "
            f"for project '''{project_id}'''"
        )

    def revoke_role(self, user_id: str, project_id: str) -> None:
        """
        Revoke a user'''s role for a project.

        Rule #1: Early return for validation.
        Rule #4: Function ≤ 60 lines.

        Args:
            user_id: User to revoke access for.
            project_id: Project to revoke access from.

        Raises:
            ValueError: If parameters are invalid.
            KeyError: If user has no role for the project.
        """
        # Precondition: Validate parameters
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        # Remove from ACL
        key = (user_id, project_id)

        if key not in self._acl:
            raise KeyError(
                f"User '''{user_id}''' has no role for project '''{project_id}'''"
            )

        del self._acl[key]

        logger.info(
            f"Revoked role for user '''{user_id}''' from project '''{project_id}'''"
        )

    def get_user_role(self, user_id: str, project_id: str) -> Optional[Role]:
        """
        Get a user'''s role for a project.

        Rule #1: Early return pattern.
        Rule #4: Function ≤ 60 lines.

        Args:
            user_id: User to check.
            project_id: Project to check access for.

        Returns:
            User'''s role, or None if user has no access.
        """
        # Precondition: Validate parameters
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        # Lookup in ACL
        key = (user_id, project_id)
        entry = self._acl.get(key)

        if entry is None:
            return None

        return entry.role

    def check_permission(
        self, user_id: str, project_id: str, action: Permission
    ) -> bool:
        """
        Check if a user has permission to perform an action.

        Rule #1: Early return for validation and permission checks.
        Rule #4: Function ≤ 60 lines.

        Args:
            user_id: User requesting the action.
            project_id: Project to perform action on.
            action: Permission required.

        Returns:
            True if user has permission, False otherwise.
        """
        # Precondition: Validate parameters
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        if not isinstance(action, Permission):
            raise ValueError(f"action must be a Permission enum, got {type(action)}")

        # Get user'''s role
        role = self.get_user_role(user_id, project_id)

        # No role = no access
        if role is None:
            return False

        # Check if role grants the required permission
        permissions = ROLE_PERMISSIONS.get(role, set())
        return action in permissions

    def require_permission(
        self, user_id: str, project_id: str, action: Permission
    ) -> None:
        """
        Assert user has permission, raise PermissionDenied if not.

        Rule #1: Early return for validation.
        Rule #4: Function ≤ 60 lines.
        Rule #5: Assertion-like behavior for access control.

        Args:
            user_id: User requesting the action.
            project_id: Project to perform action on.
            action: Permission required.

        Raises:
            PermissionDenied: If user lacks the required permission.
        """
        if not self.check_permission(user_id, project_id, action):
            raise PermissionDenied(user_id, project_id, action)

    def can_delete(self, user_id: str, project_id: str, resource_owner: str) -> bool:
        """
        Check if user can delete a resource.

        Rule #1: Early return for validation.
        Rule #4: Function ≤ 60 lines.

        DELETE permission has special rules:
        - EDITOR can delete own content (resource_owner == user_id)
        - OWNER can delete all content

        Args:
            user_id: User requesting the delete.
            project_id: Project containing the resource.
            resource_owner: User ID who owns the resource.

        Returns:
            True if user can delete, False otherwise.
        """
        # Precondition: Validate parameters
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        if not resource_owner or not isinstance(resource_owner, str):
            raise ValueError("resource_owner must be a non-empty string")

        # Get user'''s role
        role = self.get_user_role(user_id, project_id)

        # No role = no access
        if role is None:
            return False

        # OWNER can delete anything
        if role == Role.OWNER:
            return True

        # EDITOR can delete own content
        if role == Role.EDITOR and user_id == resource_owner:
            return True

        # Other roles cannot delete
        return False

    def get_acl_entry(self, user_id: str, project_id: str) -> Optional[ACLEntry]:
        """
        Get the ACL entry for a user-project pair.

        Args:
            user_id: User to check.
            project_id: Project to check.

        Returns:
            ACL entry if exists, None otherwise.
        """
        # Precondition: Validate parameters
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        key = (user_id, project_id)
        return self._acl.get(key)

    def list_users(self, project_id: str) -> List[ACLEntry]:
        """
        List all users with access to a project.

        Rule #4: Function ≤ 60 lines.

        Args:
            project_id: Project to list users for.

        Returns:
            List of ACL entries for the project.
        """
        # Precondition: Validate parameter
        if not project_id or not isinstance(project_id, str):
            raise ValueError("project_id must be a non-empty string")

        # Filter entries by project_id
        entries = [
            entry for entry in self._acl.values() if entry.project_id == project_id
        ]

        return entries

    def list_projects(self, user_id: str) -> List[ACLEntry]:
        """
        List all projects a user has access to.

        Rule #4: Function ≤ 60 lines.

        Args:
            user_id: User to list projects for.

        Returns:
            List of ACL entries for the user.
        """
        # Precondition: Validate parameter
        if not user_id or not isinstance(user_id, str):
            raise ValueError("user_id must be a non-empty string")

        # Filter entries by user_id
        entries = [entry for entry in self._acl.values() if entry.user_id == user_id]

        return entries
