"""GWT unit tests for the Legal Vertical API endpoint."""

from fastapi.testclient import TestClient
from unittest.mock import patch
from ingestforge.api.main import app

client = TestClient(app)


def test_generate_pleading_endpoint():
    """GIVEN a valid pleading request
    WHEN POST /v1/legal/generate-pleading is called
    THEN it returns 200 OK and generated markdown.
    """
    payload = {
        "metadata": {
            "court_name": "Test Court",
            "jurisdiction": "Test State",
            "plaintiffs": [{"name": "P", "role": "Plaintiff"}],
            "defendants": [{"name": "D", "role": "Defendant"}],
            "title": "Complaint",
        },
        "generate_argument": False,
    }

    with patch("ingestforge.api.routes.legal.LegalPleadingGenerator") as MockGen:
        MockGen.return_value.generate_markdown.return_value = (
            "Mocked Markdown Header\nTest Court"
        )

        response = client.post("/v1/legal/generate-pleading", json=payload)

        assert response.status_code == 200
        assert "markdown" in response.json()
        assert "Test Court" in response.json()["markdown"]
