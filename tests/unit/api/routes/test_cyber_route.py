"""GWT unit tests for the Cyber Vertical API endpoint."""

from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from ingestforge.api.main import app

client = TestClient(app)


def test_cyber_scan_endpoint():
    """GIVEN text with a CVE
    WHEN POST /v1/cyber/scan is called
    THEN it returns 200 OK and extracted models.
    """
    with patch("ingestforge.api.routes.cyber.CyberExtractor") as MockExt:
        mock_model = MagicMock()
        mock_model.cve_id = "CVE-2024-9999"
        MockExt.return_value.extract_from_text.return_value = [mock_model]

        response = client.post("/v1/cyber/scan", json={"text": "CVE-2024-9999 found"})

        assert response.status_code == 200
        # Check specific field to verify model serialization
        assert response.json()[0]["cve_id"] == "CVE-2024-9999"


def test_cyber_report_endpoint():
    """GIVEN a report request
    WHEN POST /v1/cyber/report is called
    THEN it returns 200 OK and generated markdown.
    """
    payload = {
        "mission_id": "M-TEST",
        "vulnerabilities": [
            {"cve_id": "CVE-2024-1234", "summary": "Test vuln", "cvss_score": 5.0}
        ],
    }

    with patch("ingestforge.api.routes.cyber.CyberSecurityReportGenerator") as MockGen:
        MockGen.return_value.generate_markdown_report.return_value = (
            "Mocked Report Header"
        )

        response = client.post("/v1/cyber/report", json=payload)

        assert response.status_code == 200
        assert "markdown" in response.json()
        assert "Mocked Report Header" in response.json()["markdown"]
