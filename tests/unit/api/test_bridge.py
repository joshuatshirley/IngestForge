"""
Tests for Integration Bridge API endpoints.

TICKET-402: REST API for IDE integration
"""

import pytest
from unittest.mock import MagicMock, patch

# Only import if fastapi is available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from ingestforge.api.bridge import router
from fastapi import FastAPI


@pytest.fixture
def app():
    """Create FastAPI test app."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestIngestEndpoint:
    """Test POST /v1/ingest/bridge endpoint."""

    def test_ingest_text_success(self, client):
        """Test successful text ingestion."""
        with patch("ingestforge.core.pipeline.pipeline.Pipeline") as mock_pipeline:
            mock_result = MagicMock()
            mock_result.chunks_created = 3
            mock_result.chunk_ids = ["c1", "c2", "c3"]
            mock_result.document_id = "test_doc"
            mock_pipeline.return_value.process_text.return_value = mock_result

            response = client.post(
                "/v1/ingest/bridge",
                json={"text": "Test content.", "source": "vscode"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["chunk_count"] == 3


class TestQueryEndpoint:
    """Test POST /v1/query endpoint."""

    def test_query_success(self, client):
        """Test successful query."""
        with patch("ingestforge.core.pipeline.pipeline.Pipeline") as mock_pipeline:
            mock_pipeline.return_value.query.return_value = [
                {
                    "chunk_id": "c1",
                    "content": "Match",
                    "source_file": "t.txt",
                    "score": 0.9,
                }
            ]

            response = client.post("/v1/query", json={"query": "test", "top_k": 5})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True


class TestStatusEndpoint:
    """Test GET /v1/status endpoint."""

    def test_status_success(self, client):
        """Test successful status retrieval."""
        with patch("ingestforge.core.pipeline.pipeline.Pipeline") as mock_pipeline:
            mock_pipeline.return_value.get_status.return_value = {
                "total_documents": 42,
                "total_chunks": 256,
            }

            response = client.get("/v1/status")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["document_count"] == 42
