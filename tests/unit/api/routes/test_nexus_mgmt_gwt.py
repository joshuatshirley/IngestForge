"""
GWT Unit Tests for Nexus Management Routes.

Task 271: Peer registry management.
Task 283: MFA/High-privilege confirmation.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import Request, HTTPException
from ingestforge.api.routes.nexus_mgmt import revoke_peer, ping_peer
from ingestforge.core.models.nexus import NexusStatus


@pytest.fixture
def mock_request():
    request = MagicMock(spec=Request)
    request.headers = {}
    request.state = MagicMock()
    request.state.user_id = "test-admin"
    return request


# = =============================================================================
# SCENARIO: Emergency Peer Revocation (Task 283)
# = =============================================================================


@pytest.mark.asyncio
async def test_revoke_peer_given_missing_admin_session_when_called_then_raises_403(
    mock_request,
):
    # Given: No X-Admin-Session header

    # When / Then
    with pytest.raises(HTTPException) as exc:
        await revoke_peer("peer-1", mock_request)

    assert exc.value.status_code == 403
    assert "High-risk action" in exc.value.detail


@pytest.mark.asyncio
async def test_revoke_peer_given_valid_admin_session_when_called_then_revokes_access(
    mock_request,
):
    # Given
    mock_request.headers["X-Admin-Session"] = "short-lived-token"

    with patch("ingestforge.api.routes.nexus_mgmt._get_acl") as mock_get_acl:
        mock_acl = mock_get_acl.return_value

        # When
        response = await revoke_peer("peer-1", mock_request)

        # Then
        assert response["status"] == "success"
        mock_acl.revoke_all_access.assert_called_once_with("peer-1")


# = =============================================================================
# SCENARIO: Manual Peer Health Check (Task 271)
# = =============================================================================


@pytest.mark.asyncio
async def test_ping_peer_given_existing_peer_when_called_then_triggers_scanner(
    mock_request,
):
    # Given
    with patch(
        "ingestforge.api.routes.nexus_mgmt._get_registry"
    ) as mock_get_reg, patch(
        "ingestforge.core.system.nexus_scanner.NexusScanner"
    ) as mock_scanner_cls:
        mock_reg = mock_get_reg.return_value
        mock_peer = MagicMock()
        mock_peer.status = NexusStatus.ONLINE
        mock_reg.get_peer.return_value = mock_peer

        mock_scanner = mock_scanner_cls.return_value
        mock_scanner.check_peer = AsyncMock()

        # When
        response = await ping_peer("peer-1")

        # Then
        assert response["new_status"] == NexusStatus.ONLINE
        mock_scanner.check_peer.assert_called_once_with(mock_peer)
