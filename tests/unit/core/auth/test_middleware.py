"""
Tests for JWT Authentication Middleware.

FastAPI middleware for JWT validation.
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from unittest.mock import AsyncMock, MagicMock
import pytest

from ingestforge.core.auth.middleware import (
    JWTAuthDependency,
    JWTAuthMiddleware,
    RequireRole,
    map_roles_to_acl,
    get_highest_role,
    get_token_from_header,
    MAX_TOKEN_LENGTH,
)
from ingestforge.core.auth.jwt_handler import JWTHandler
from ingestforge.core.security.acl import Role


# =============================================================================
# ROLE MAPPING TESTS
# =============================================================================


class TestRoleMapping:
    """Tests for role mapping functions."""

    def test_map_roles_to_acl_owner(self):
        """
        GWT:
        Given 'owner' role string
        When map_roles_to_acl is called
        Then Role.OWNER is returned.
        """
        roles = map_roles_to_acl(["owner"])

        assert Role.OWNER in roles

    def test_map_roles_to_acl_admin(self):
        """
        GWT:
        Given 'admin' role string
        When map_roles_to_acl is called
        Then Role.OWNER is returned (admin maps to owner).
        """
        roles = map_roles_to_acl(["admin"])

        assert Role.OWNER in roles

    def test_map_roles_to_acl_multiple(self):
        """
        GWT:
        Given multiple role strings
        When map_roles_to_acl is called
        Then all roles are mapped.
        """
        roles = map_roles_to_acl(["editor", "viewer", "contributor"])

        assert Role.EDITOR in roles
        assert Role.VIEWER in roles
        assert Role.CONTRIBUTOR in roles

    def test_map_roles_case_insensitive(self):
        """
        GWT:
        Given uppercase role string
        When map_roles_to_acl is called
        Then role is mapped correctly.
        """
        roles = map_roles_to_acl(["OWNER", "Editor", "VIEWER"])

        assert Role.OWNER in roles
        assert Role.EDITOR in roles
        assert Role.VIEWER in roles

    def test_map_roles_unknown_role(self):
        """
        GWT:
        Given unknown role string
        When map_roles_to_acl is called
        Then unknown role is ignored.
        """
        roles = map_roles_to_acl(["unknown_role", "viewer"])

        assert Role.VIEWER in roles
        assert len(roles) == 1


class TestGetHighestRole:
    """Tests for get_highest_role function."""

    def test_highest_role_owner(self):
        """
        GWT:
        Given list with owner
        When get_highest_role is called
        Then Role.OWNER is returned.
        """
        result = get_highest_role(["viewer", "owner", "editor"])

        assert result == Role.OWNER

    def test_highest_role_editor(self):
        """
        GWT:
        Given list without owner
        When get_highest_role is called
        Then highest available role is returned.
        """
        result = get_highest_role(["viewer", "contributor", "editor"])

        assert result == Role.EDITOR

    def test_highest_role_empty(self):
        """
        GWT:
        Given empty roles list
        When get_highest_role is called
        Then None is returned.
        """
        result = get_highest_role([])

        assert result is None

    def test_highest_role_unknown_only(self):
        """
        GWT:
        Given only unknown roles
        When get_highest_role is called
        Then None is returned.
        """
        result = get_highest_role(["unknown", "invalid"])

        assert result is None


# =============================================================================
# TOKEN EXTRACTION TESTS
# =============================================================================


class TestGetTokenFromHeader:
    """Tests for get_token_from_header function."""

    @pytest.mark.asyncio
    async def test_extract_bearer_token(self):
        """
        GWT:
        Given request with valid Authorization header
        When get_token_from_header is called
        Then token is extracted.
        """
        request = MagicMock()
        request.headers.get.return_value = "Bearer test_token_123"

        token = await get_token_from_header(request)

        assert token == "test_token_123"

    @pytest.mark.asyncio
    async def test_missing_header(self):
        """
        GWT:
        Given request without Authorization header
        When get_token_from_header is called
        Then None is returned.
        """
        request = MagicMock()
        request.headers.get.return_value = None

        token = await get_token_from_header(request)

        assert token is None

    @pytest.mark.asyncio
    async def test_invalid_format_no_space(self):
        """
        GWT:
        Given header without space
        When get_token_from_header is called
        Then None is returned.
        """
        request = MagicMock()
        request.headers.get.return_value = "Bearertoken"

        token = await get_token_from_header(request)

        assert token is None

    @pytest.mark.asyncio
    async def test_wrong_scheme(self):
        """
        GWT:
        Given non-Bearer scheme
        When get_token_from_header is called
        Then None is returned.
        """
        request = MagicMock()
        request.headers.get.return_value = "Basic abc123"

        token = await get_token_from_header(request)

        assert token is None

    @pytest.mark.asyncio
    async def test_token_too_long(self):
        """
        GWT:
        Given token exceeding MAX_TOKEN_LENGTH (JPL Rule #2)
        When get_token_from_header is called
        Then None is returned.
        """
        long_token = "x" * (MAX_TOKEN_LENGTH + 1)
        request = MagicMock()
        request.headers.get.return_value = f"Bearer {long_token}"

        token = await get_token_from_header(request)

        assert token is None


# =============================================================================
# JWT AUTH DEPENDENCY TESTS
# =============================================================================


class TestJWTAuthDependency:
    """Tests for JWTAuthDependency."""

    @pytest.fixture
    def handler(self):
        """Create a JWTHandler for testing."""
        return JWTHandler(secret_key="test_secret")

    @pytest.mark.asyncio
    async def test_valid_token(self, handler):
        """
        GWT:
        Given valid token in request
        When dependency is called
        Then TokenPayload is returned.
        """
        token = handler.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["editor"],
        )

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        request.state = MagicMock()

        auth = JWTAuthDependency(handler=handler)
        payload = await auth(request)

        assert payload is not None
        assert payload.user_id == "user123"

    @pytest.mark.asyncio
    async def test_missing_token_required(self, handler):
        """
        GWT:
        Given missing token with required=True
        When dependency is called
        Then HTTPException 401 is raised.
        """
        from fastapi import HTTPException

        request = MagicMock()
        request.headers.get.return_value = None

        auth = JWTAuthDependency(handler=handler, required=True)

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)

        assert exc_info.value.status_code == 401
        assert "Missing" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_missing_token_optional(self, handler):
        """
        GWT:
        Given missing token with required=False
        When dependency is called
        Then None is returned.
        """
        request = MagicMock()
        request.headers.get.return_value = None

        auth = JWTAuthDependency(handler=handler, required=False)
        payload = await auth(request)

        assert payload is None

    @pytest.mark.asyncio
    async def test_invalid_token_required(self, handler):
        """
        GWT:
        Given invalid token with required=True
        When dependency is called
        Then HTTPException 401 is raised (AC: clear 401).
        """
        from fastapi import HTTPException

        request = MagicMock()
        request.headers.get.return_value = "Bearer invalid.token.here"

        auth = JWTAuthDependency(handler=handler, required=True)

        with pytest.raises(HTTPException) as exc_info:
            await auth(request)

        assert exc_info.value.status_code == 401
        assert "Invalid" in exc_info.value.detail or "expired" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_attaches_user_to_request_state(self, handler):
        """
        GWT:
        Given valid token
        When dependency is called
        Then user info is attached to request.state.
        """
        token = handler.create_token(
            user_id="user123",
            workspace_id="ws456",
            roles=["editor"],
        )

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        request.state = MagicMock()

        auth = JWTAuthDependency(handler=handler)
        await auth(request)

        assert request.state.user_id == "user123"
        assert request.state.workspace_id == "ws456"
        assert "editor" in request.state.roles


# =============================================================================
# REQUIRE ROLE TESTS
# =============================================================================


class TestRequireRole:
    """Tests for RequireRole dependency."""

    @pytest.fixture
    def handler(self):
        """Create a JWTHandler for testing."""
        return JWTHandler(secret_key="test_secret")

    @pytest.mark.asyncio
    async def test_allowed_role(self, handler):
        """
        GWT:
        Given user with allowed role
        When RequireRole is called
        Then TokenPayload is returned.
        """
        token = handler.create_token(
            user_id="admin_user",
            workspace_id="ws",
            roles=["admin"],
        )

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        request.state = MagicMock()

        require = RequireRole(allowed_roles=["admin", "owner"], handler=handler)
        payload = await require(request)

        assert payload.user_id == "admin_user"

    @pytest.mark.asyncio
    async def test_disallowed_role(self, handler):
        """
        GWT:
        Given user without allowed role
        When RequireRole is called
        Then HTTPException 403 is raised.
        """
        from fastapi import HTTPException

        token = handler.create_token(
            user_id="viewer_user",
            workspace_id="ws",
            roles=["viewer"],
        )

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        request.state = MagicMock()

        require = RequireRole(allowed_roles=["admin", "owner"], handler=handler)

        with pytest.raises(HTTPException) as exc_info:
            await require(request)

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_role_case_insensitive(self, handler):
        """
        GWT:
        Given role with different case
        When RequireRole is called
        Then role matches case-insensitively.
        """
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["EDITOR"],  # Uppercase
        )

        request = MagicMock()
        request.headers.get.return_value = f"Bearer {token}"
        request.state = MagicMock()

        require = RequireRole(allowed_roles=["editor"], handler=handler)  # lowercase
        payload = await require(request)

        assert payload is not None


# =============================================================================
# MIDDLEWARE TESTS
# =============================================================================


class TestJWTAuthMiddleware:
    """Tests for JWTAuthMiddleware."""

    @pytest.fixture
    def handler(self):
        """Create a JWTHandler for testing."""
        return JWTHandler(secret_key="test_secret")

    def test_middleware_init(self, handler):
        """
        GWT:
        Given handler and exclude paths
        When middleware is created
        Then configuration is stored.
        """
        app = MagicMock()
        middleware = JWTAuthMiddleware(
            app,
            handler=handler,
            exclude_paths=["/custom"],
        )

        assert "/custom" in middleware._exclude_paths
        assert "/health" in middleware._exclude_paths  # Default

    @pytest.mark.asyncio
    async def test_dispatch_excluded_path(self, handler):
        """
        GWT:
        Given request to excluded path
        When dispatch is called
        Then request proceeds without auth.
        """
        app = MagicMock()
        middleware = JWTAuthMiddleware(app, handler=handler)

        request = MagicMock()
        request.url.path = "/health"
        request.method = "GET"

        call_next = AsyncMock(return_value=MagicMock())
        response = await middleware.dispatch(request, call_next)

        call_next.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatch_valid_token(self, handler):
        """
        GWT:
        Given valid token
        When dispatch is called
        Then request proceeds with user info.
        """
        token = handler.create_token(
            user_id="user",
            workspace_id="ws",
            roles=["viewer"],
        )

        app = MagicMock()
        middleware = JWTAuthMiddleware(app, handler=handler)

        request = MagicMock()
        request.url.path = "/api/data"
        request.method = "GET"
        request.headers.get.return_value = f"Bearer {token}"
        request.state = MagicMock()

        call_next = AsyncMock(return_value=MagicMock())
        await middleware.dispatch(request, call_next)

        call_next.assert_called_once()
        assert request.state.user_id == "user"

    @pytest.mark.asyncio
    async def test_dispatch_missing_token(self, handler):
        """
        GWT:
        Given missing token on protected path
        When dispatch is called
        Then 401 response is returned.
        """
        app = MagicMock()
        middleware = JWTAuthMiddleware(app, handler=handler)

        request = MagicMock()
        request.url.path = "/api/data"
        request.method = "GET"
        request.headers.get.return_value = None

        call_next = AsyncMock()
        response = await middleware.dispatch(request, call_next)

        assert response.status_code == 401
        call_next.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_options_request(self, handler):
        """
        GWT:
        Given OPTIONS preflight request
        When dispatch is called
        Then request proceeds without auth.
        """
        app = MagicMock()
        middleware = JWTAuthMiddleware(app, handler=handler)

        request = MagicMock()
        request.url.path = "/api/data"
        request.method = "OPTIONS"

        call_next = AsyncMock(return_value=MagicMock())
        response = await middleware.dispatch(request, call_next)

        call_next.assert_called_once()
