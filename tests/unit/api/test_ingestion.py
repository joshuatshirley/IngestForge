"""
Tests for Ingestion Router with Background Tasks.

TICKET-505: File upload and background processing endpoints
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from io import BytesIO

# Only import if fastapi is available
pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from fastapi import FastAPI
from ingestforge.api.routes.ingestion import router, generate_job_id


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


class TestGenerateJobId:
    """Test job ID generation."""

    def test_generate_job_id_format(self):
        """Test job ID has correct format."""
        job_id = generate_job_id()

        assert job_id.startswith("job_")
        assert len(job_id) == 16  # "job_" + 12 hex chars

    def test_generate_job_id_unique(self):
        """Test job IDs are unique."""
        ids = [generate_job_id() for _ in range(100)]

        # All IDs should be unique
        assert len(set(ids)) == 100


class TestUploadEndpoint:
    """Test POST /v1/ingest/upload endpoint."""

    def test_upload_file_success(self, client):
        """Test successful file upload."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.core.state.StateManager"
        ) as mock_state_mgr:
            # Setup mocks
            mock_config = MagicMock()
            mock_config.ingest.supported_formats = {".pdf", ".txt", ".epub"}
            mock_config.pending_path = Path("/tmp/pending")
            mock_config.data_path = Path("/tmp/data")
            mock_config.project.name = "test_project"

            mock_pipeline.return_value.config = mock_config

            mock_doc_state = MagicMock()
            mock_state_mgr.return_value.get_or_create_document.return_value = (
                mock_doc_state
            )

            # Create test file
            file_content = b"Test PDF content"
            files = {"file": ("test.pdf", BytesIO(file_content), "application/pdf")}

            # Make request
            response = client.post("/v1/ingest/upload", files=files)

            # Verify response
            assert response.status_code == 200
            data = response.json()

            assert "job_id" in data
            assert data["job_id"].startswith("job_")
            assert data["filename"] == "test.pdf"
            assert data["status"] == "PENDING"
            assert "queued for processing" in data["message"].lower()

    def test_upload_file_unsupported_format(self, client):
        """Test upload with unsupported file format."""
        with patch("ingestforge.core.pipeline.pipeline.Pipeline") as mock_pipeline:
            # Setup mocks
            mock_config = MagicMock()
            mock_config.ingest.supported_formats = {".pdf", ".txt"}
            mock_config.pending_path = Path("/tmp/pending")

            mock_pipeline.return_value.config = mock_config

            # Create test file with unsupported extension
            file_content = b"Test content"
            files = {
                "file": (
                    "test.docx",
                    BytesIO(file_content),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            }

            # Make request
            response = client.post("/v1/ingest/upload", files=files)

            # Verify error response
            assert response.status_code == 400
            assert "unsupported file type" in response.json()["detail"].lower()


class TestStatusEndpoint:
    """Test GET /v1/ingest/status/{job_id} endpoint."""

    def test_get_status_pending(self, client):
        """Test status check for pending job."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.core.state.StateManager"
        ) as mock_state_mgr:
            from ingestforge.core.state import ProcessingStatus, DocumentState

            # Setup mocks
            mock_config = MagicMock()
            mock_config.data_path = Path("/tmp/data")
            mock_config.project.name = "test_project"
            mock_pipeline.return_value.config = mock_config

            # Create pending document state
            mock_doc_state = DocumentState(
                document_id="job_abc123",
                source_file="/tmp/test.pdf",
                status=ProcessingStatus.PENDING,
            )

            mock_state = MagicMock()
            mock_state.get_document.return_value = mock_doc_state
            mock_state_mgr.return_value.state = mock_state

            # Make request
            response = client.get("/v1/ingest/status/job_abc123")

            # Verify response
            assert response.status_code == 200
            data = response.json()

            assert data["job_id"] == "job_abc123"
            assert data["status"] == "PENDING"
            assert data["progress"] == 0.0
            assert data["result"] is None

    def test_get_status_completed(self, client):
        """Test status check for completed job."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.core.state.StateManager"
        ) as mock_state_mgr:
            from ingestforge.core.state import ProcessingStatus, DocumentState

            # Setup mocks
            mock_config = MagicMock()
            mock_config.data_path = Path("/tmp/data")
            mock_config.project.name = "test_project"
            mock_pipeline.return_value.config = mock_config

            # Create completed document state
            mock_doc_state = DocumentState(
                document_id="job_xyz789",
                source_file="/tmp/test.pdf",
                status=ProcessingStatus.COMPLETED,
                total_chunks=10,
                indexed_chunks=10,
            )

            mock_state = MagicMock()
            mock_state.get_document.return_value = mock_doc_state
            mock_state_mgr.return_value.state = mock_state

            # Make request
            response = client.get("/v1/ingest/status/job_xyz789")

            # Verify response
            assert response.status_code == 200
            data = response.json()

            assert data["job_id"] == "job_xyz789"
            assert data["status"] == "COMPLETED"
            assert data["progress"] == 1.0
            assert data["result"] is not None
            assert data["result"]["chunks_created"] == 10
            assert data["result"]["chunks_indexed"] == 10

    def test_get_status_not_found(self, client):
        """Test status check for non-existent job."""
        with patch(
            "ingestforge.core.pipeline.pipeline.Pipeline"
        ) as mock_pipeline, patch(
            "ingestforge.core.state.StateManager"
        ) as mock_state_mgr:
            # Setup mocks
            mock_config = MagicMock()
            mock_config.data_path = Path("/tmp/data")
            mock_config.project.name = "test_project"
            mock_pipeline.return_value.config = mock_config

            mock_state = MagicMock()
            mock_state.get_document.return_value = None
            mock_state_mgr.return_value.state = mock_state

            # Make request
            response = client.get("/v1/ingest/status/job_notfound")

            # Verify error response
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
