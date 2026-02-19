"""
GWT Unit Tests for Nexus Authorization Middleware.

Task 125: Rate limiting per peer.
Task 123: Kill-switch (Blacklist).
Task 256: Global Silence.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, HTTPException
from ingestforge.api.middleware.nexus_auth import NexusAuthMiddleware


@pytest.fixture
def mock_app():
    return MagicMock()


@pytest.fixture
def middleware(mock_app):
    with patch(
        "ingestforge.api.middleware.nexus_auth.NexusAuditLogger"
    ) as mock_audit_cls, patch(
        "ingestforge.api.middleware.nexus_auth.NexusBlacklist"
    ) as mock_bl_cls, patch(
        "ingestforge.api.middleware.nexus_auth.NexusRegistry"
    ) as mock_reg_cls:
        # Configure mocks
        mock_reg = mock_reg_cls.return_value
        mock_reg.is_silenced.return_value = False

        mock_audit = mock_audit_cls.return_value
        mock_audit.log_event = AsyncMock()

        mw = NexusAuthMiddleware(mock_app)
        # Ensure instances used by mw are our mocks
        mw.audit_logger = mock_audit
        mw.blacklist = mock_bl_cls.return_value
        mw.registry = mock_reg

        return mw


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.url.path = "/v1/nexus/search"
    request.method = "POST"
    request.headers = {"X-Nexus-ID": "peer-789", "X-Target-Library": "legal_vault"}
    request.state = MagicMock()
    return request


# =============================================================================
# SCENARIO: Global Silence Mode (Task 256)
# =============================================================================


@pytest.mark.asyncio
async def test_nexus_auth_given_silenced_registry_when_dispatched_then_raises_503(
    middleware, mock_request
):
    # Given
    middleware.registry.is_silenced.return_value = True

    # When / Then
    with pytest.raises(HTTPException) as exc:
        await middleware.dispatch(mock_request, AsyncMock())

    assert exc.value.status_code == 503
    assert "isolated" in exc.value.detail


# =============================================================================
# SCENARIO: Emergency Kill-Switch / Blacklist (Task 123)
# =============================================================================


@pytest.mark.asyncio
async def test_nexus_auth_given_revoked_peer_when_dispatched_then_raises_403(
    middleware, mock_request
):
    # Given
    middleware.blacklist.is_revoked.return_value = True

    # When / Then
    with pytest.raises(HTTPException) as exc:
        await middleware.dispatch(mock_request, AsyncMock())

    assert exc.value.status_code == 403
    assert "REVOKED" in exc.value.detail


# =============================================================================
# SCENARIO: Rate Limiting (Task 125)
# =============================================================================


@pytest.mark.asyncio
async def test_nexus_auth_given_rate_limit_exceeded_when_dispatched_then_raises_429(
    middleware, mock_request
):
    # Given
    middleware.blacklist.is_revoked.return_value = False
    # Mock the rate limiter inside the middleware
    middleware.peer_limiter = MagicMock()
    limit_result = MagicMock()
    limit_result.allowed = False
    limit_result.to_headers.return_value = {"Retry-After": "60"}
    middleware.peer_limiter.check = AsyncMock(return_value=limit_result)

    # When / Then
    with pytest.raises(HTTPException) as exc:
        await middleware.dispatch(mock_request, AsyncMock())

    assert exc.value.status_code == 429
    assert "rate limit exceeded" in exc.value.detail
    assert exc.value.headers["Retry-After"] == "60"


# =============================================================================
# SCENARIO: Successful Authorization (Task 120/121)
# =============================================================================


@pytest.mark.asyncio
async def test_nexus_auth_given_valid_peer_and_acl_when_dispatched_then_proceeds(
    middleware, mock_request
):
    # Given
    middleware.blacklist.is_revoked.return_value = False
    middleware.registry.is_silenced.return_value = False

    # Mock ACL and RBAC
    with patch(
        "ingestforge.api.middleware.nexus_auth.NexusACLManager"
    ) as mock_acl_cls, patch(
        "ingestforge.api.middleware.nexus_auth.NexusRBAC"
    ) as mock_rbac:
        mock_acl = mock_acl_cls.return_value
        mock_acl.is_authorized.return_value = True
        mock_acl.get_role.return_value = "reader"
        mock_rbac.is_action_allowed.return_value = True

        # Mock next call
        call_next = AsyncMock()
        call_next.return_value = "success_response"

        # When
        response = await middleware.dispatch(mock_request, call_next)

        # Then
        assert response == "success_response"
        assert mock_request.state.nexus_peer_id == "peer-789"
        assert mock_request.state.authorized_library == "legal_vault"
        assert mock_request.state.nexus_role == "reader"
        call_next.assert_called_once_with(mock_request)
