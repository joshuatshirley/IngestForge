"""Tests for Health and Status API endpoints.

G-TICKET-103: Tests for enhanced health and status endpoints with
storage health checks and project statistics.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch


# =============================================================================
# UNIT TESTS - Response Models
# =============================================================================


@pytest.mark.unit
def test_storage_health_model():
    """Test StorageHealth Pydantic model."""
    from ingestforge.api.routes.health import StorageHealth

    # Test healthy storage
    health = StorageHealth(
        healthy=True,
        backend="chromadb",
        error=None,
        details={"collections": 3},
    )
    assert health.healthy is True
    assert health.backend == "chromadb"
    assert health.error is None
    assert health.details == {"collections": 3}

    # Test unhealthy storage
    unhealthy = StorageHealth(
        healthy=False,
        backend="postgres",
        error="Connection refused",
        details={},
    )
    assert unhealthy.healthy is False
    assert unhealthy.error == "Connection refused"


@pytest.mark.unit
def test_health_response_model():
    """Test HealthResponse Pydantic model."""
    from ingestforge.api.routes.health import HealthResponse, StorageHealth

    storage = StorageHealth(healthy=True, backend="jsonl")

    response = HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp="2024-01-01T00:00:00",
        uptime_seconds=123.45,
        storage=storage,
    )

    assert response.status == "healthy"
    assert response.version == "1.0.0"
    assert response.uptime_seconds == 123.45
    assert response.storage.backend == "jsonl"


@pytest.mark.unit
def test_library_stats_model():
    """Test LibraryStats Pydantic model."""
    from ingestforge.api.routes.health import LibraryStats

    stats = LibraryStats(name="research", chunk_count=150)
    assert stats.name == "research"
    assert stats.chunk_count == 150


@pytest.mark.unit
def test_status_response_model():
    """Test StatusResponse Pydantic model."""
    from ingestforge.api.routes.health import StatusResponse, LibraryStats

    libraries = [
        LibraryStats(name="default", chunk_count=100),
        LibraryStats(name="research", chunk_count=50),
    ]

    response = StatusResponse(
        status="ok",
        version="1.0.0",
        timestamp="2024-01-01T00:00:00",
        project_name="test-project",
        document_count=10,
        chunk_count=150,
        total_embeddings=150,
        storage_backend="chromadb",
        libraries=libraries,
        data_path="/path/to/data",
    )

    assert response.status == "ok"
    assert response.project_name == "test-project"
    assert response.document_count == 10
    assert response.chunk_count == 150
    assert len(response.libraries) == 2
    assert response.storage_backend == "chromadb"


# =============================================================================
# UNIT TESTS - Helper Functions
# =============================================================================


@pytest.mark.unit
def test_get_storage_health_success():
    """Test _get_storage_health returns correct structure on success."""
    from ingestforge.api.routes.health import _get_storage_health

    # Mock the dependencies
    mock_health_result = {
        "healthy": True,
        "backend": "jsonl",
        "error": None,
        "details": {"path": "/data", "writable": True},
    }

    with patch("ingestforge.core.config.Config") as mock_config_cls, patch(
        "ingestforge.storage.factory.check_health"
    ) as mock_check:
        mock_config = MagicMock()
        mock_config_cls.load.return_value = mock_config
        mock_check.return_value = mock_health_result

        result = _get_storage_health()

        assert result.healthy is True
        assert result.backend == "jsonl"
        assert result.error is None
        assert result.details["writable"] is True


@pytest.mark.unit
def test_get_storage_health_failure():
    """Test _get_storage_health handles errors gracefully."""
    from ingestforge.api.routes.health import _get_storage_health

    with patch("ingestforge.core.config.Config") as mock_config_cls:
        mock_config_cls.load.side_effect = Exception("Config not found")

        result = _get_storage_health()

        assert result.healthy is False
        assert result.backend == "unknown"
        assert "Config not found" in result.error


@pytest.mark.unit
def test_get_project_stats_success():
    """Test _get_project_stats returns correct structure."""
    from ingestforge.api.routes.health import _get_project_stats

    # Create mock objects
    mock_config = MagicMock()
    mock_config.project.name = "test-project"
    mock_config.storage.backend = "jsonl"
    mock_config.data_path = Path("/data/test")

    mock_state = MagicMock()
    mock_state.total_documents = 5
    mock_state.total_embeddings = 100

    mock_storage = MagicMock()
    mock_storage.count.return_value = 100
    mock_storage.get_libraries.return_value = ["default", "research"]
    mock_storage.count_by_library.side_effect = (
        lambda lib: 60 if lib == "default" else 40
    )

    with patch("ingestforge.core.config.Config") as mock_config_cls, patch(
        "ingestforge.core.state.ProcessingState"
    ) as mock_state_cls, patch(
        "ingestforge.storage.factory.get_storage_backend"
    ) as mock_get_storage:
        mock_config_cls.load.return_value = mock_config
        mock_state_cls.load.return_value = mock_state
        mock_get_storage.return_value = mock_storage

        result = _get_project_stats()

        assert result["project_name"] == "test-project"
        assert result["document_count"] == 5
        assert result["chunk_count"] == 100
        assert result["storage_backend"] == "jsonl"
        assert len(result["libraries"]) == 2


@pytest.mark.unit
def test_get_project_stats_library_not_supported():
    """Test _get_project_stats handles backends without library support."""
    from ingestforge.api.routes.health import _get_project_stats

    mock_config = MagicMock()
    mock_config.project.name = "test"
    mock_config.storage.backend = "jsonl"
    mock_config.data_path = Path("/data")

    mock_state = MagicMock()
    mock_state.total_documents = 1
    mock_state.total_embeddings = 10

    mock_storage = MagicMock()
    mock_storage.count.return_value = 10
    mock_storage.get_libraries.side_effect = NotImplementedError()

    with patch("ingestforge.core.config.Config") as mock_config_cls, patch(
        "ingestforge.core.state.ProcessingState"
    ) as mock_state_cls, patch(
        "ingestforge.storage.factory.get_storage_backend"
    ) as mock_get_storage:
        mock_config_cls.load.return_value = mock_config
        mock_state_cls.load.return_value = mock_state
        mock_get_storage.return_value = mock_storage

        result = _get_project_stats()

        # Should fall back to single default library
        assert len(result["libraries"]) == 1
        assert result["libraries"][0]["name"] == "default"
        assert result["libraries"][0]["chunk_count"] == 10


# =============================================================================
# INTEGRATION TESTS - API Endpoints
# =============================================================================


@pytest.mark.integration
@pytest.mark.requires_api
def test_health_endpoint_returns_200():
    """Test /v1/health returns 200 with expected fields."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        # Check required fields exist
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "version" in data
        assert "timestamp" in data
        assert "uptime_seconds" in data
        assert "storage" in data

        # Storage should have expected structure
        storage = data["storage"]
        assert "healthy" in storage
        assert "backend" in storage

    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
@pytest.mark.requires_api
def test_status_endpoint_returns_200():
    """Test /v1/status returns 200 with project statistics."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/v1/status")

        assert response.status_code == 200
        data = response.json()

        # Check required fields exist
        assert "status" in data
        assert data["status"] in ["ok", "degraded"]
        assert "version" in data
        assert "timestamp" in data
        assert "document_count" in data
        assert "chunk_count" in data
        assert "storage_backend" in data
        assert "libraries" in data

        # Libraries should be a list
        assert isinstance(data["libraries"], list)

    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
@pytest.mark.requires_api
def test_health_endpoint_includes_storage_details():
    """Test /v1/health includes storage health details."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/v1/health")

        assert response.status_code == 200
        data = response.json()

        storage = data.get("storage", {})
        # Should have storage health info
        assert "healthy" in storage
        assert isinstance(storage["healthy"], bool)

        # Should have backend type
        assert "backend" in storage
        assert isinstance(storage["backend"], str)

        # Should have details dict
        assert "details" in storage
        assert isinstance(storage["details"], dict)

    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
@pytest.mark.requires_api
def test_status_endpoint_returns_library_stats():
    """Test /v1/status includes per-library statistics."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/v1/status")

        assert response.status_code == 200
        data = response.json()

        libraries = data.get("libraries", [])
        # Should have at least default library or empty list
        assert isinstance(libraries, list)

        # If libraries exist, check structure
        for lib in libraries:
            assert "name" in lib
            assert "chunk_count" in lib
            assert isinstance(lib["chunk_count"], int)

    except ImportError:
        pytest.skip("FastAPI not installed")


@pytest.mark.integration
@pytest.mark.requires_api
@pytest.mark.skip(
    reason="Legacy /health endpoint not implemented - use /v1/health instead"
)
def test_root_health_still_works():
    """Test legacy /health endpoint still returns 200."""
    try:
        from fastapi.testclient import TestClient
        from ingestforge.api.main import app

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Legacy endpoint has simpler response
        assert "status" in data
        assert "version" in data

    except ImportError:
        pytest.skip("FastAPI not installed")


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


@pytest.mark.unit
def test_health_response_minimal():
    """Test HealthResponse with minimal fields."""
    from ingestforge.api.routes.health import HealthResponse

    # Should work with required fields only
    response = HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp="2024-01-01T00:00:00",
    )
    assert response.uptime_seconds == 0.0
    assert response.storage is None


@pytest.mark.unit
def test_status_response_empty_libraries():
    """Test StatusResponse with no libraries."""
    from ingestforge.api.routes.health import StatusResponse

    response = StatusResponse(
        status="ok",
        version="1.0.0",
        timestamp="2024-01-01T00:00:00",
    )

    assert response.libraries == []
    assert response.document_count == 0
    assert response.chunk_count == 0
