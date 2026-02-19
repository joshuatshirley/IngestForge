"""
Tests for ACL Guard (Access Control List) implementation.

Verifies role-based access control for multi-user project collaboration.

Test Coverage
-------------
- Permission enum values
- Role enum values
- Role-to-Permission mappings
- ACLEntry validation
- PermissionDenied exception
- ACLGuard.grant_role()
- ACLGuard.revoke_role()
- ACLGuard.get_user_role()
- ACLGuard.check_permission()
- ACLGuard.require_permission()
- ACLGuard.can_delete() (ownership-based deletion)
- ACLGuard.list_users()
- ACLGuard.list_projects()---------------------------------
Rule #1: Simple Control Flow
    - Max 2 nesting levels
    - Early returns for test setup

Rule #2: Fixed Upper Bound
    - No unbounded loops in tests

Rule #4: No Large Functions
    - All test functions ≤ 60 lines

Rule #5: Assertion Density
    - Multiple assertions per test
    - Precondition validation

Rule #9: Type Safety
    - Full type hints in test helpers
"""

from __future__ import annotations

import pytest
from datetime import datetime

from ingestforge.core.security.acl import (
    ACLEntry,
    ACLGuard,
    Permission,
    PermissionDenied,
    Role,
    ROLE_PERMISSIONS,
)

# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def acl_guard() -> ACLGuard:
    """Create a fresh ACL guard for each test."""
    return ACLGuard()


@pytest.fixture
def populated_guard() -> ACLGuard:
    """Create an ACL guard with sample users and roles."""
    guard = ACLGuard()
    guard.grant_role("alice", "proj1", Role.OWNER)
    guard.grant_role("bob", "proj1", Role.EDITOR)
    guard.grant_role("charlie", "proj1", Role.CONTRIBUTOR)
    guard.grant_role("diana", "proj1", Role.VIEWER)
    guard.grant_role("alice", "proj2", Role.CONTRIBUTOR)
    return guard


# ===========================================================================
# Permission & Role Enum Tests
# ===========================================================================


def test_permission_enum_values() -> None:
    """Verify Permission enum has expected values."""
    assert Permission.READ.value == "read"
    assert Permission.WRITE.value == "write"
    assert Permission.DELETE.value == "delete"
    assert Permission.ADMIN.value == "admin"


def test_role_enum_values() -> None:
    """Verify Role enum has expected values."""
    assert Role.VIEWER.value == "viewer"
    assert Role.CONTRIBUTOR.value == "contributor"
    assert Role.EDITOR.value == "editor"
    assert Role.OWNER.value == "owner"


def test_role_permissions_mapping() -> None:
    """Verify ROLE_PERMISSIONS mapping is correct."""
    assert ROLE_PERMISSIONS[Role.VIEWER] == {Permission.READ}
    assert ROLE_PERMISSIONS[Role.CONTRIBUTOR] == {Permission.READ, Permission.WRITE}
    assert ROLE_PERMISSIONS[Role.EDITOR] == {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
    }
    assert ROLE_PERMISSIONS[Role.OWNER] == {
        Permission.READ,
        Permission.WRITE,
        Permission.DELETE,
        Permission.ADMIN,
    }


# ===========================================================================
# ACLEntry Tests
# ===========================================================================


def test_acl_entry_creation() -> None:
    """Test ACLEntry can be created with valid parameters."""
    entry = ACLEntry(
        user_id="alice",
        project_id="proj1",
        role=Role.OWNER,
        granted_by="admin",
    )
    assert entry.user_id == "alice"
    assert entry.project_id == "proj1"
    assert entry.role == Role.OWNER
    assert entry.granted_by == "admin"
    assert entry.granted_at is not None
    assert isinstance(entry.granted_at, str)


def test_acl_entry_auto_timestamp() -> None:
    """Test ACLEntry automatically generates timestamp."""
    entry = ACLEntry(
        user_id="alice",
        project_id="proj1",
        role=Role.VIEWER,
    )
    assert entry.granted_at is not None
    # Should be parseable as datetime
    parsed = datetime.fromisoformat(entry.granted_at)
    assert isinstance(parsed, datetime)


def test_acl_entry_validation_empty_user_id() -> None:
    """Test ACLEntry validation rejects empty user_id."""
    with pytest.raises(ValueError, match="user_id must be a non-empty string"):
        ACLEntry(user_id="", project_id="proj1", role=Role.VIEWER)


def test_acl_entry_validation_empty_project_id() -> None:
    """Test ACLEntry validation rejects empty project_id."""
    with pytest.raises(ValueError, match="project_id must be a non-empty string"):
        ACLEntry(user_id="alice", project_id="", role=Role.VIEWER)


def test_acl_entry_validation_invalid_role() -> None:
    """Test ACLEntry validation rejects non-Role types."""
    with pytest.raises(ValueError, match="role must be a Role enum"):
        ACLEntry(user_id="alice", project_id="proj1", role="viewer")  # type: ignore


# ===========================================================================
# PermissionDenied Exception Tests
# ===========================================================================


def test_permission_denied_exception() -> None:
    """Test PermissionDenied exception creation."""
    exc = PermissionDenied("alice", "proj1", Permission.DELETE)

    assert exc.user_id == "alice"
    assert exc.project_id == "proj1"
    assert exc.permission == Permission.DELETE
    assert "alice" in str(exc)
    assert "proj1" in str(exc)
    assert "delete" in str(exc)


def test_permission_denied_custom_message() -> None:
    """Test PermissionDenied with custom message."""
    exc = PermissionDenied(
        "bob", "proj2", Permission.ADMIN, message="Custom denial message"
    )

    assert str(exc) == "Custom denial message"


# ===========================================================================
# ACLGuard.grant_role() Tests
# ===========================================================================


def test_grant_role_basic(acl_guard: ACLGuard) -> None:
    """Test granting a role to a user."""
    acl_guard.grant_role("alice", "proj1", Role.CONTRIBUTOR)

    role = acl_guard.get_user_role("alice", "proj1")
    assert role == Role.CONTRIBUTOR


def test_grant_role_with_granter(acl_guard: ACLGuard) -> None:
    """Test granting a role with audit trail."""
    acl_guard.grant_role("bob", "proj1", Role.EDITOR, granted_by="admin")

    entry = acl_guard.get_acl_entry("bob", "proj1")
    assert entry is not None
    assert entry.granted_by == "admin"


def test_grant_role_overwrites_existing(acl_guard: ACLGuard) -> None:
    """Test granting a role overwrites previous role."""
    acl_guard.grant_role("alice", "proj1", Role.VIEWER)
    acl_guard.grant_role("alice", "proj1", Role.OWNER)

    role = acl_guard.get_user_role("alice", "proj1")
    assert role == Role.OWNER


def test_grant_role_validation_empty_user_id(acl_guard: ACLGuard) -> None:
    """Test grant_role rejects empty user_id."""
    with pytest.raises(ValueError, match="user_id must be a non-empty string"):
        acl_guard.grant_role("", "proj1", Role.VIEWER)


def test_grant_role_validation_empty_project_id(acl_guard: ACLGuard) -> None:
    """Test grant_role rejects empty project_id."""
    with pytest.raises(ValueError, match="project_id must be a non-empty string"):
        acl_guard.grant_role("alice", "", Role.VIEWER)


def test_grant_role_validation_invalid_role(acl_guard: ACLGuard) -> None:
    """Test grant_role rejects non-Role types."""
    with pytest.raises(ValueError, match="role must be a Role enum"):
        acl_guard.grant_role("alice", "proj1", "contributor")  # type: ignore


# ===========================================================================
# ACLGuard.revoke_role() Tests
# ===========================================================================


def test_revoke_role_basic(acl_guard: ACLGuard) -> None:
    """Test revoking a user's role."""
    acl_guard.grant_role("alice", "proj1", Role.CONTRIBUTOR)
    acl_guard.revoke_role("alice", "proj1")

    role = acl_guard.get_user_role("alice", "proj1")
    assert role is None


def test_revoke_role_nonexistent_raises(acl_guard: ACLGuard) -> None:
    """Test revoking non-existent role raises KeyError."""
    with pytest.raises(KeyError, match="has no role for project"):
        acl_guard.revoke_role("alice", "proj1")


def test_revoke_role_validation_empty_user_id(acl_guard: ACLGuard) -> None:
    """Test revoke_role rejects empty user_id."""
    with pytest.raises(ValueError, match="user_id must be a non-empty string"):
        acl_guard.revoke_role("", "proj1")


def test_revoke_role_validation_empty_project_id(acl_guard: ACLGuard) -> None:
    """Test revoke_role rejects empty project_id."""
    with pytest.raises(ValueError, match="project_id must be a non-empty string"):
        acl_guard.revoke_role("alice", "")


# ===========================================================================
# ACLGuard.get_user_role() Tests
# ===========================================================================


def test_get_user_role_existing(acl_guard: ACLGuard) -> None:
    """Test getting an existing user role."""
    acl_guard.grant_role("alice", "proj1", Role.OWNER)

    role = acl_guard.get_user_role("alice", "proj1")
    assert role == Role.OWNER


def test_get_user_role_nonexistent(acl_guard: ACLGuard) -> None:
    """Test getting non-existent user role returns None."""
    role = acl_guard.get_user_role("alice", "proj1")
    assert role is None


def test_get_user_role_validation_empty_user_id(acl_guard: ACLGuard) -> None:
    """Test get_user_role rejects empty user_id."""
    with pytest.raises(ValueError, match="user_id must be a non-empty string"):
        acl_guard.get_user_role("", "proj1")


def test_get_user_role_validation_empty_project_id(acl_guard: ACLGuard) -> None:
    """Test get_user_role rejects empty project_id."""
    with pytest.raises(ValueError, match="project_id must be a non-empty string"):
        acl_guard.get_user_role("alice", "")


# ===========================================================================
# ACLGuard.check_permission() Tests
# ===========================================================================


def test_check_permission_viewer(acl_guard: ACLGuard) -> None:
    """Test VIEWER permissions (READ only)."""
    acl_guard.grant_role("alice", "proj1", Role.VIEWER)

    # VIEWER has READ
    assert acl_guard.check_permission("alice", "proj1", Permission.READ) is True

    # VIEWER lacks WRITE, DELETE, ADMIN
    assert acl_guard.check_permission("alice", "proj1", Permission.WRITE) is False
    assert acl_guard.check_permission("alice", "proj1", Permission.DELETE) is False
    assert acl_guard.check_permission("alice", "proj1", Permission.ADMIN) is False


def test_check_permission_contributor(acl_guard: ACLGuard) -> None:
    """Test CONTRIBUTOR permissions (READ, WRITE)."""
    acl_guard.grant_role("bob", "proj1", Role.CONTRIBUTOR)

    # CONTRIBUTOR has READ, WRITE
    assert acl_guard.check_permission("bob", "proj1", Permission.READ) is True
    assert acl_guard.check_permission("bob", "proj1", Permission.WRITE) is True

    # CONTRIBUTOR lacks DELETE, ADMIN
    assert acl_guard.check_permission("bob", "proj1", Permission.DELETE) is False
    assert acl_guard.check_permission("bob", "proj1", Permission.ADMIN) is False


def test_check_permission_editor(acl_guard: ACLGuard) -> None:
    """Test EDITOR permissions (READ, WRITE, DELETE)."""
    acl_guard.grant_role("charlie", "proj1", Role.EDITOR)

    # EDITOR has READ, WRITE, DELETE
    assert acl_guard.check_permission("charlie", "proj1", Permission.READ) is True
    assert acl_guard.check_permission("charlie", "proj1", Permission.WRITE) is True
    assert acl_guard.check_permission("charlie", "proj1", Permission.DELETE) is True

    # EDITOR lacks ADMIN
    assert acl_guard.check_permission("charlie", "proj1", Permission.ADMIN) is False


def test_check_permission_owner(acl_guard: ACLGuard) -> None:
    """Test OWNER permissions (READ, WRITE, DELETE, ADMIN)."""
    acl_guard.grant_role("diana", "proj1", Role.OWNER)

    # OWNER has all permissions
    assert acl_guard.check_permission("diana", "proj1", Permission.READ) is True
    assert acl_guard.check_permission("diana", "proj1", Permission.WRITE) is True
    assert acl_guard.check_permission("diana", "proj1", Permission.DELETE) is True
    assert acl_guard.check_permission("diana", "proj1", Permission.ADMIN) is True


def test_check_permission_no_role(acl_guard: ACLGuard) -> None:
    """Test user with no role has no permissions."""
    # User has no role for proj1
    assert acl_guard.check_permission("alice", "proj1", Permission.READ) is False
    assert acl_guard.check_permission("alice", "proj1", Permission.WRITE) is False


def test_check_permission_validation_empty_user_id(acl_guard: ACLGuard) -> None:
    """Test check_permission rejects empty user_id."""
    with pytest.raises(ValueError, match="user_id must be a non-empty string"):
        acl_guard.check_permission("", "proj1", Permission.READ)


def test_check_permission_validation_invalid_action(acl_guard: ACLGuard) -> None:
    """Test check_permission rejects non-Permission types."""
    with pytest.raises(ValueError, match="action must be a Permission enum"):
        acl_guard.check_permission("alice", "proj1", "read")  # type: ignore


# ===========================================================================
# ACLGuard.require_permission() Tests
# ===========================================================================


def test_require_permission_allowed(acl_guard: ACLGuard) -> None:
    """Test require_permission passes when permission granted."""
    acl_guard.grant_role("alice", "proj1", Role.CONTRIBUTOR)

    # Should not raise
    acl_guard.require_permission("alice", "proj1", Permission.WRITE)


def test_require_permission_denied(acl_guard: ACLGuard) -> None:
    """Test require_permission raises PermissionDenied when lacking permission."""
    acl_guard.grant_role("alice", "proj1", Role.VIEWER)

    with pytest.raises(PermissionDenied) as exc_info:
        acl_guard.require_permission("alice", "proj1", Permission.WRITE)

    # Verify exception attributes
    assert exc_info.value.user_id == "alice"
    assert exc_info.value.project_id == "proj1"
    assert exc_info.value.permission == Permission.WRITE


# ===========================================================================
# ACLGuard.can_delete() Tests
# ===========================================================================


def test_can_delete_owner_all_content(acl_guard: ACLGuard) -> None:
    """Test OWNER can delete all content."""
    acl_guard.grant_role("alice", "proj1", Role.OWNER)

    # OWNER can delete their own content
    assert acl_guard.can_delete("alice", "proj1", resource_owner="alice") is True

    # OWNER can delete other users' content
    assert acl_guard.can_delete("alice", "proj1", resource_owner="bob") is True


def test_can_delete_editor_own_content(acl_guard: ACLGuard) -> None:
    """Test EDITOR can delete own content only."""
    acl_guard.grant_role("bob", "proj1", Role.EDITOR)

    # EDITOR can delete their own content
    assert acl_guard.can_delete("bob", "proj1", resource_owner="bob") is True

    # EDITOR cannot delete other users' content
    assert acl_guard.can_delete("bob", "proj1", resource_owner="alice") is False


def test_can_delete_contributor_cannot_delete(acl_guard: ACLGuard) -> None:
    """Test CONTRIBUTOR cannot delete any content."""
    acl_guard.grant_role("charlie", "proj1", Role.CONTRIBUTOR)

    # CONTRIBUTOR cannot delete even own content
    assert acl_guard.can_delete("charlie", "proj1", resource_owner="charlie") is False


def test_can_delete_viewer_cannot_delete(acl_guard: ACLGuard) -> None:
    """Test VIEWER cannot delete any content."""
    acl_guard.grant_role("diana", "proj1", Role.VIEWER)

    # VIEWER cannot delete any content
    assert acl_guard.can_delete("diana", "proj1", resource_owner="diana") is False


def test_can_delete_no_role(acl_guard: ACLGuard) -> None:
    """Test user with no role cannot delete."""
    # User has no role
    assert acl_guard.can_delete("alice", "proj1", resource_owner="alice") is False


# ===========================================================================
# ACLGuard.list_users() Tests
# ===========================================================================


def test_list_users_populated(populated_guard: ACLGuard) -> None:
    """Test listing all users for a project."""
    users = populated_guard.list_users("proj1")

    # Should have 4 users for proj1
    assert len(users) == 4

    # Verify user IDs
    user_ids = {entry.user_id for entry in users}
    assert user_ids == {"alice", "bob", "charlie", "diana"}


def test_list_users_empty(acl_guard: ACLGuard) -> None:
    """Test listing users for project with no users."""
    users = acl_guard.list_users("proj1")
    assert users == []


# ===========================================================================
# ACLGuard.list_projects() Tests
# ===========================================================================


def test_list_projects_populated(populated_guard: ACLGuard) -> None:
    """Test listing all projects for a user."""
    projects = populated_guard.list_projects("alice")

    # Alice has access to 2 projects
    assert len(projects) == 2

    # Verify project IDs
    project_ids = {entry.project_id for entry in projects}
    assert project_ids == {"proj1", "proj2"}


def test_list_projects_empty(acl_guard: ACLGuard) -> None:
    """Test listing projects for user with no access."""
    projects = acl_guard.list_projects("alice")
    assert projects == []


# ===========================================================================
# Integration Tests
# ===========================================================================


def test_integration_workflow(acl_guard: ACLGuard) -> None:
    """Test complete ACL workflow."""
    # 1. Grant roles to multiple users
    acl_guard.grant_role("alice", "proj1", Role.OWNER, granted_by="system")
    acl_guard.grant_role("bob", "proj1", Role.EDITOR)
    acl_guard.grant_role("charlie", "proj1", Role.CONTRIBUTOR)

    # 2. Verify permissions
    assert acl_guard.check_permission("alice", "proj1", Permission.ADMIN) is True
    assert acl_guard.check_permission("bob", "proj1", Permission.DELETE) is True
    assert acl_guard.check_permission("charlie", "proj1", Permission.WRITE) is True
    assert acl_guard.check_permission("charlie", "proj1", Permission.DELETE) is False

    # 3. Verify delete capabilities
    assert acl_guard.can_delete("alice", "proj1", resource_owner="bob") is True
    assert acl_guard.can_delete("bob", "proj1", resource_owner="bob") is True
    assert acl_guard.can_delete("bob", "proj1", resource_owner="alice") is False

    # 4. Revoke a role
    acl_guard.revoke_role("charlie", "proj1")
    assert acl_guard.get_user_role("charlie", "proj1") is None

    # 5. List users
    users = acl_guard.list_users("proj1")
    assert len(users) == 2
    assert {u.user_id for u in users} == {"alice", "bob"}
