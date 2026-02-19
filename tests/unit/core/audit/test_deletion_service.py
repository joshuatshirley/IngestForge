"""
Comprehensive GWT unit tests for the PurgeService.

Verifiable Deletion Cert
Verifies that purging triggers pre-deletion hashing and signed certificate generation.
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestforge.core.audit.deletion_service import PurgeService
from ingestforge.core.audit.models import DeletionCertificate

# =============================================================================
# UNIT TESTS (GWT)
# =============================================================================


@pytest.mark.asyncio
async def test_verifiable_purge_success():
    """
    GIVEN an existing chunk in storage
    WHEN execute_verifiable_purge is called
    THEN it returns success, a signed certificate, and verified=True.
    """
    # Use parenthesized context managers (Python 3.10+)
    with (
        patch("ingestforge.core.audit.deletion_service.load_config"),
        patch(
            "ingestforge.core.audit.deletion_service.get_storage_backend"
        ) as MockRepoFactory,
        patch("ingestforge.core.audit.deletion_service.log_operation") as MockLog,
    ):
        # Setup mocks
        mock_repo = MagicMock()
        mock_chunk = MagicMock()
        mock_chunk.content = "Sensitive data to be purged"

        # First call to get_chunk returns the chunk
        # Second call (verification) returns None
        mock_repo.get_chunk.side_effect = [mock_chunk, None]
        mock_repo.delete_chunk.return_value = True

        MockRepoFactory.return_value = mock_repo

        service = PurgeService(secret_key="test_secret")
        success, cert, msg = service.execute_verifiable_purge("chunk_123", "admin_user")

        # Then
        assert success is True
        assert isinstance(cert, DeletionCertificate)
        assert cert.artifact_id == "chunk_123"
        assert cert.storage_verification is True
        assert cert.signature is not None

        # Verify log was called
        assert MockLog.called
        assert mock_repo.delete_chunk.called


@pytest.mark.asyncio
async def test_verifiable_purge_not_found():
    """
    GIVEN a non-existent chunk ID
    WHEN execute_verifiable_purge is called
    THEN it returns False and an appropriate error message.
    """
    with (
        patch("ingestforge.core.audit.deletion_service.load_config"),
        patch(
            "ingestforge.core.audit.deletion_service.get_storage_backend"
        ) as MockRepoFactory,
    ):
        mock_repo = MagicMock()
        mock_repo.get_chunk.return_value = None
        MockRepoFactory.return_value = mock_repo

        service = PurgeService()
        success, cert, msg = service.execute_verifiable_purge("missing_id")

        assert success is False
        assert cert is None
        assert "not found" in msg.lower()


def test_signature_verification():
    """
    GIVEN a generated certificate
    WHEN its signature is recomputed manually
    THEN it matches the certificate signature.
    """
    service = PurgeService(secret_key="my_secret")

    # We use the internal _create_certificate to test signing
    cert = service._create_certificate(
        artifact_id="id1", content_hash="abc", operator_id="op1", verified=True
    )

    # Re-sign same data
    data_to_sign = f"{cert.certificate_id}|{cert.artifact_id}|{cert.purge_timestamp.isoformat()}|{cert.content_hash}|{cert.operator_id}"
    expected_sig = service._sign_data(data_to_sign)

    assert cert.signature == expected_sig
