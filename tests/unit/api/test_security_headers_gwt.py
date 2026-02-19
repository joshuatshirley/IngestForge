"""
GWT Unit Tests for API Security Headers.

Task 125: Security hardening (HSTS, CSP).
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi import Request, Response
from ingestforge.api.main import add_security_headers


@pytest.fixture
def mock_request():
    return MagicMock(spec=Request)


# =============================================================================
# SCENARIO: Middleware adds security headers to all responses
# =============================================================================


@pytest.mark.asyncio
async def test_security_headers_given_any_request_when_dispatched_then_appends_hsts_and_csp(
    mock_request,
):
    # Given
    mock_response = Response()
    call_next = AsyncMock(return_value=mock_response)

    # When
    response = await add_security_headers(mock_request, call_next)

    # Then
    assert (
        response.headers["Strict-Transport-Security"]
        == "max-age=31536000; includeSubDomains"
    )
    assert "Content-Security-Policy" in response.headers
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    call_next.assert_called_once_with(mock_request)
