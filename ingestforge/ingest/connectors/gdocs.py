"""Google Docs Remote Connector.

Extracts text and metadata from Google Docs via official REST API.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from typing import Dict, Optional

from ingestforge.ingest.connectors.base import RemoteConnector, RemoteDocument
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class GoogleDocsConnector(RemoteConnector):
    """Connector for fetching documents from Google Docs."""

    def __init__(self):
        self._service = None
        self._creds = None

    def authenticate(self, credentials: Dict[str, str]) -> bool:
        """
        Initialize Google API client.
        Rule #7: Validate credentials dict keys.
        """
        if "token" not in credentials:
            logger.error("Missing Google OAuth token")
            return False

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            self._creds = Credentials(token=credentials["token"])
            self._service = build("docs", "v1", credentials=self._creds)
            return True
        except Exception as e:
            logger.error(f"GDocs authentication failed: {e}")
            return False

    def fetch_document(self, document_id: str) -> Optional[RemoteDocument]:
        """
        Retrieve a Google Doc and extract its text content.
        Rule #1: Linear control flow with early returns.
        """
        if not self._service:
            logger.error("Connector not authenticated")
            return None

        try:
            doc = self._service.documents().get(documentId=document_id).execute()
            title = doc.get("title", "Untitled GDoc")

            # Extract plain text from document body
            text_content = self._extract_text_from_doc(doc)

            return RemoteDocument(
                content=text_content,
                title=title,
                source_id=document_id,
                mime_type="application/vnd.google-apps.document",
                metadata={
                    "platform": "google_docs",
                    "revision_id": doc.get("revisionId"),
                    "suggestions_view_mode": doc.get("suggestionsViewMode"),
                },
            )
        except Exception as e:
            logger.error(f"Failed to fetch GDoc {document_id}: {e}")
            return None

    def test_connection(self) -> bool:
        """Verify the service is active."""
        return self._service is not None

    def _extract_text_from_doc(self, doc: dict) -> str:
        """
        Walk the document JSON to aggregate text.
        Rule #4: Logic isolated to small helper.
        """
        text = ""
        body = doc.get("body", {})
        content = body.get("content", [])

        for element in content:
            if "paragraph" in element:
                for part in element["paragraph"]["elements"]:
                    if "textRun" in part:
                        text += part["textRun"]["content"]

        return text
