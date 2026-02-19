"""GWT unit tests for the Chat API endpoint."""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from ingestforge.api.main import app

client = TestClient(app)


def test_chat_endpoint_valid_request():
    """GIVEN a valid ChatRequest with history
    WHEN POST /v1/chat is called
    THEN it returns 200 OK and a synthesized answer.
    """
    mock_result = MagicMock()
    mock_result.answer = "The results show..."
    mock_result.sources = []
    mock_result.context_query = "Context query"

    with patch("ingestforge.api.routes.chat.ChatService") as MockService:
        instance = MockService.return_value
        instance.chat.return_value = mock_result

        payload = {
            "messages": [
                {"role": "user", "content": "Tell me about stars."},
                {"role": "ai", "content": "Stars are hot."},
                {"role": "user", "content": "How hot?"},
            ]
        }
        response = client.post("/v1/chat", json=payload)

        assert response.status_code == 200
        assert response.json()["answer"] == "The results show..."


def test_chat_endpoint_invalid_last_role():
    """GIVEN a request where the last message is NOT from the user
    WHEN POST /v1/chat is called
    THEN it returns 400 Bad Request.
    """
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "ai", "content": "Hi there"},
        ]
    }
    response = client.post("/v1/chat", json=payload)
    assert response.status_code == 400
    assert "user" in response.json()["detail"].lower()
