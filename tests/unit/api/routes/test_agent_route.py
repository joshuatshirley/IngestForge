"""GWT unit tests for the Agent API endpoint."""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from ingestforge.api.main import app

client = TestClient(app)


def test_generate_brief_endpoint():
    """GIVEN a valid BriefRequest
    WHEN POST /v1/agent/generate-brief is called
    THEN it returns 200 OK and the synthesized brief.
    """
    with patch("ingestforge.api.routes.agent.IFBriefGenerator") as MockGen:
        # Create a real-ish brief model for the mock return
        from ingestforge.agent.brief_models import IntelligenceBrief

        mock_brief = IntelligenceBrief(
            mission_id="M-123",
            title="Test",
            summary="Synthesized summary.",
            key_entities=[],
            evidence=[],
        )

        # Mocking the async method
        import asyncio

        future = asyncio.Future()
        future.set_result(mock_brief)
        MockGen.return_value.generate_brief.return_value = future

        # Mocking storage for validator
        MockGen.return_value.pipeline.storage = MagicMock()

        payload = {"mission_id": "M-123", "query": "Test"}
        response = client.post("/v1/agent/generate-brief", json=payload)

        assert response.status_code == 200
        assert "markdown" in response.json()
        assert response.json()["brief"]["mission_id"] == "M-123"
        assert response.json()["validation_passed"] is True
