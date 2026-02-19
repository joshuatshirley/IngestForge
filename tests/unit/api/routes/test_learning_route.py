"""
GWT unit tests for Learning API Router.

Few-Shot Registry
Verifies API endpoint integration.
"""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from ingestforge.api.main import app

client = TestClient(app)


def test_add_verified_example_endpoint():
    """GIVEN a valid FewShotExample
    WHEN POST /v1/learning/examples is called
    THEN it returns 201 Created and the example ID.
    """
    payload = {
        "id": "ex_api_1",
        "input_text": "API input",
        "output_json": {"key": "val"},
        "domain": "cyber",
    }

    with patch("ingestforge.api.routes.learning.FewShotRegistry") as MockRegistry:
        MockRegistry.return_value.add_example.return_value = True

        response = client.post("/v1/learning/examples", json=payload)

        assert response.status_code == 201
        assert response.json()["id"] == "ex_api_1"


def test_list_verified_examples_endpoint():
    """GIVEN verified examples in the registry
    WHEN GET /v1/learning/examples is called
    THEN it returns a list of examples.
    """
    with patch("ingestforge.api.routes.learning.FewShotRegistry") as MockRegistry:
        mock_ex = MagicMock()
        mock_ex.id = "ex_1"
        mock_ex.domain = "legal"
        MockRegistry.return_value.list_examples.return_value = [mock_ex]

        response = client.get("/v1/learning/examples?domain=legal")

        assert response.status_code == 200
        assert response.json()["count"] == 1
        assert response.json()["examples"][0]["id"] == "ex_1"


def test_delete_verified_example_endpoint():
    """GIVEN an existing example ID
    WHEN DELETE /v1/learning/examples/{id} is called
    THEN it returns success.
    """
    with patch("ingestforge.api.routes.learning.FewShotRegistry") as MockRegistry:
        MockRegistry.return_value.remove_example.return_value = True

        response = client.delete("/v1/learning/examples/ex_1")

        assert response.status_code == 200
        assert response.json()["status"] == "removed"
