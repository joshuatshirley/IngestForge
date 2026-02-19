"""Tests for FastAPI main application.

US-BUG-003: Updated to match actual implementation.
JPL Rule #9: Complete type hints.
"""

from __future__ import annotations

import pytest


# =============================================================================
# UNIT TESTS - Models
# =============================================================================


@pytest.mark.unit
def test_token_model() -> None:
    """Test Token response model."""
    from ingestforge.api.main import Token

    token = Token(access_token="abc123", token_type="bearer")
    assert token.access_token == "abc123"
    assert token.token_type == "bearer"


@pytest.mark.unit
def test_agent_request_model() -> None:
    """Test AgentRequest model with defaults."""
    from ingestforge.api.main import AgentRequest

    req = AgentRequest(task="test task")
    assert req.task == "test task"
    assert req.max_steps == 10
    assert req.provider == "llamacpp"
    assert req.roadmap is None


@pytest.mark.unit
def test_agent_mission_status_model() -> None:
    """Test AgentMissionStatus model."""
    from ingestforge.api.main import AgentMissionStatus

    status = AgentMissionStatus(
        id="job-123",
        task="test",
        status="RUNNING",
        steps=[],
        progress=50,
    )
    assert status.id == "job-123"
    assert status.status == "RUNNING"
    assert status.progress == 50


@pytest.mark.unit
def test_remote_ingest_request_model() -> None:
    """Test RemoteIngestRequest model."""
    from ingestforge.api.main import RemoteIngestRequest

    req = RemoteIngestRequest(platform="gdrive", source_id="doc-123")
    assert req.platform == "gdrive"
    assert req.source_id == "doc-123"
    assert req.token is None


@pytest.mark.unit
def test_transform_request_model() -> None:
    """Test TransformRequest model."""
    from ingestforge.api.main import TransformRequest

    req = TransformRequest(operation="chunk")
    assert req.operation == "chunk"
    assert req.target_library == "default"
    assert req.params == {}


# =============================================================================
# UNIT TESTS - run_server validation
# =============================================================================


@pytest.mark.unit
def test_run_server_validates_port() -> None:
    """Test run_server validates port range."""
    from ingestforge.api.main import run_server

    with pytest.raises(ValueError, match="Invalid port"):
        run_server(host="localhost", port=99999, reload=False)

    with pytest.raises(ValueError, match="Invalid port"):
        run_server(host="localhost", port=0, reload=False)

    with pytest.raises(ValueError, match="Invalid port"):
        run_server(host="localhost", port=-1, reload=False)


@pytest.mark.unit
def test_run_server_validates_host() -> None:
    """Test run_server validates host."""
    from ingestforge.api.main import run_server

    with pytest.raises(ValueError, match="Invalid host"):
        run_server(host="", port=8000, reload=False)

    with pytest.raises(ValueError, match="Invalid host"):
        run_server(host="   ", port=8000, reload=False)


# =============================================================================
# UNIT TESTS - Job limit enforcement
# =============================================================================


@pytest.mark.unit
def test_max_agent_jobs_constant() -> None:
    """Test MAX_AGENT_JOBS constant exists."""
    from ingestforge.api.main import MAX_AGENT_JOBS

    assert MAX_AGENT_JOBS == 1000


@pytest.mark.unit
def test_enforce_job_limit_function() -> None:
    """Test _enforce_job_limit removes old jobs when limit reached."""
    from ingestforge.api.main import _enforce_job_limit, agent_jobs, MAX_AGENT_JOBS

    # Clear existing jobs
    agent_jobs.clear()

    # Fill to limit
    for i in range(MAX_AGENT_JOBS):
        agent_jobs[f"job-{i}"] = {"id": f"job-{i}", "status": "DONE"}

    assert len(agent_jobs) == MAX_AGENT_JOBS

    # Enforce limit should remove oldest 10%
    _enforce_job_limit()
    assert len(agent_jobs) < MAX_AGENT_JOBS
    assert len(agent_jobs) == int(MAX_AGENT_JOBS * 0.9)

    # Cleanup
    agent_jobs.clear()


# =============================================================================
# INTEGRATION TESTS - Endpoints
# =============================================================================


@pytest.mark.integration
def test_health_endpoint_returns_200() -> None:
    """Test /v1/health endpoint returns 200."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()
        # Health router returns HealthResponse with status field
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
def test_docs_endpoint_available() -> None:
    """Test OpenAPI docs endpoint is available."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/docs")

        assert response.status_code == 200
    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
def test_synthesis_router_registered() -> None:
    """Test that synthesis router () is registered."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)

        # The /v1/synthesize endpoint should exist (may fail without LLM, but not 404)
        response = client.post("/v1/synthesize", json={"query": "test"})
        # Should not be 404 (not found) - may be 500 if LLM not available
        assert response.status_code != 404
    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
def test_auth_required_for_protected_endpoints() -> None:
    """Test that protected endpoints require authentication."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)

        # These endpoints should require auth
        protected = [
            ("GET", "/v1/ingest/status/test"),
            ("POST", "/v1/corpus/transform"),
            ("GET", "/v1/agent/status/test"),
        ]

        for method, path in protected:
            if method == "GET":
                response = client.get(path)
            else:
                response = client.post(path, json={})

            # Should be 401 Unauthorized without token
            assert response.status_code == 401, f"{method} {path} should require auth"
    except ImportError:
        pytest.skip("FastAPI not installed")


# =============================================================================
# UNIT TESTS - Logger defined
# =============================================================================


@pytest.mark.unit
def test_logger_defined() -> None:
    """Test logger is defined at module level."""
    from ingestforge.api.main import logger

    assert logger is not None
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")


@pytest.mark.integration
def test_ws_status_router_registered() -> None:
    """Test that WebSocket status router () is registered."""
    from ingestforge.api.main import app

    # Check route is registered
    ws_routes = [r for r in app.routes if hasattr(r, "path") and "/ws/" in r.path]
    assert len(ws_routes) > 0, "WebSocket status router should be registered"
