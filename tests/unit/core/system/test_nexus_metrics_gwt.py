"""
GWT Unit Tests for Nexus Prometheus Metrics.

Task 126: Integration of security and performance metrics.
"""

from fastapi.testclient import TestClient
from ingestforge.api.main import app

client = TestClient(app)

# =============================================================================
# SCENARIO: Metrics endpoint availability
# =============================================================================


def test_metrics_endpoint_given_app_running_when_called_then_returns_200():
    # When
    response = client.get("/metrics")

    # Then
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


# =============================================================================
# SCENARIO: Tracking handshake failures
# =============================================================================


def test_metrics_given_handshake_failure_when_tracked_then_appears_in_endpoint():
    # Given
    from ingestforge.core.system.nexus_metrics import track_handshake_failure

    track_handshake_failure("test_reason")

    # When
    response = client.get("/metrics")

    # Then
    assert 'nexus_handshake_failures_total{reason="test_reason"} 1.0' in response.text


# =============================================================================
# SCENARIO: Tracking unauthorized attempts
# =============================================================================


def test_metrics_given_unauthorized_attempt_when_tracked_then_appears_in_endpoint():
    # Given
    from ingestforge.core.system.nexus_metrics import track_unauthorized_attempt

    track_unauthorized_attempt("peer-999", "rate_limit")

    # When
    response = client.get("/metrics")

    # Then
    assert (
        'nexus_unauthorized_access_attempts_total{peer_id="peer-999",reason="rate_limit"} 1.0'
        in response.text
    )
