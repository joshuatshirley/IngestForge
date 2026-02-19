"""
GWT Unit Tests for Nexus mTLS - Task 119.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
import ssl
from unittest.mock import MagicMock, patch
from pathlib import Path
from ingestforge.core.config.nexus import NexusConfig
from ingestforge.core.security.mtls import MTLSContextManager


@pytest.fixture
def mock_nexus_config() -> NexusConfig:
    config = MagicMock(spec=NexusConfig)
    config.nexus_id = "nex"
    # Ensure cert_file and other paths are mocks that can have return values set on methods
    config.cert_file = MagicMock(spec=Path)
    config.key_file = MagicMock(spec=Path)
    config.trust_store = MagicMock(spec=Path)
    return config


# =============================================================================
# GIVEN: A valid mTLS configuration
# =============================================================================


def test_mtls_given_valid_paths_when_server_context_created_then_enforces_tls13(
    mock_nexus_config,
):
    # Given
    mock_nexus_config.cert_file.exists.return_value = True
    mock_nexus_config.key_file.exists.return_value = True
    mock_nexus_config.trust_store.exists.return_value = True

    manager = MTLSContextManager(mock_nexus_config)

    # When
    with patch("ssl.SSLContext.load_cert_chain"), patch(
        "ssl.SSLContext.load_verify_locations"
    ):
        context = manager.get_server_context()

    # Then
    assert context.minimum_version == ssl.TLSVersion.TLSv1_3
    assert context.verify_mode == ssl.CERT_REQUIRED


def test_mtls_given_missing_cert_when_context_created_then_raises_filenotfound(
    mock_nexus_config,
):
    # Given
    mock_nexus_config.cert_file.exists.return_value = False
    manager = MTLSContextManager(mock_nexus_config)

    # When / Then
    with pytest.raises(FileNotFoundError) as exc:
        manager.get_server_context()
    assert "mTLS identity files missing" in str(exc.value)


def test_mtls_given_outbound_request_when_client_context_created_then_server_auth_purpose(
    mock_nexus_config,
):
    # Given
    mock_nexus_config.cert_file.exists.return_value = True
    mock_nexus_config.key_file.exists.return_value = True
    mock_nexus_config.trust_store.exists.return_value = True

    manager = MTLSContextManager(mock_nexus_config)

    # When
    with patch("ssl.SSLContext.load_cert_chain"), patch(
        "ssl.SSLContext.load_verify_locations"
    ):
        context = manager.get_client_context()

    # Then (Default for create_default_context(ssl.Purpose.SERVER_AUTH))
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True


def test_mtls_given_directory_trust_store_when_loaded_then_uses_capath(
    mock_nexus_config,
):
    # Given
    mock_nexus_config.cert_file.exists.return_value = True
    mock_nexus_config.key_file.exists.return_value = True
    mock_nexus_config.trust_store.exists.return_value = True
    mock_nexus_config.trust_store.is_dir.return_value = True

    manager = MTLSContextManager(mock_nexus_config)

    # When
    with patch("ssl.SSLContext.load_cert_chain"), patch(
        "ssl.SSLContext.load_verify_locations"
    ) as mock_verify:
        manager.get_server_context()

    # Then
    mock_verify.assert_called_once_with(capath=str(mock_nexus_config.trust_store))
