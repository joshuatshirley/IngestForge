"""
Unit tests for Nexus Handshake - Task 127.
"""

import pytest
from unittest.mock import MagicMock
from ingestforge.core.models.nexus_ld import HandshakeRequest
from ingestforge.core.pipeline.nexus_handshake import HandshakeManager
from ingestforge.storage.nexus_registry import NexusRegistry


@pytest.fixture
def mock_registry():
    return MagicMock(spec=NexusRegistry)


@pytest.mark.asyncio
async def test_handshake_given_valid_request_when_processed_then_accepted(
    mock_registry,
):
    # Given
    request = HandshakeRequest(
        nexus_id="peer-123", version="1.2.0", supported_verticals=["legal"]
    )
    manager = HandshakeManager(mock_registry)

    # When
    response, success = await manager.process_request(request, "peer-123")

    # Then
    assert success is True
    assert response.status == "ACCEPTED"
    assert response.nexus_id == "local-nexus"
    mock_registry.get_peer.assert_called_once_with("peer-123")


@pytest.mark.asyncio
async def test_handshake_given_mismatched_cn_when_processed_then_denied(mock_registry):
    # Given
    request = HandshakeRequest(nexus_id="peer-123", version="1.2.0")
    manager = HandshakeManager(mock_registry)

    # When
    response, success = await manager.process_request(request, "wrong-cn")

    # Then
    assert success is False
    assert "Identity mismatch" in response.status


@pytest.mark.asyncio
async def test_handshake_given_incompatible_version_when_processed_then_denied(
    mock_registry,
):
    # Given
    request = HandshakeRequest(
        nexus_id="peer-123",
        version="2.0.0",  # Only 1.x supported
    )
    manager = HandshakeManager(mock_registry)

    # When
    response, success = await manager.process_request(request, "peer-123")

    # Then
    assert success is False
    assert "Incompatible version" in response.status
