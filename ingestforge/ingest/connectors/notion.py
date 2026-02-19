"""Notion Remote Connector.

Extracts text and metadata from Notion Pages and Databases.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from ingestforge.ingest.connectors.base import RemoteConnector, RemoteDocument
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class NotionConnector(RemoteConnector):
    """Connector for fetching documents from Notion."""

    def __init__(self):
        self._client = None

    def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Initialize Notion client with an Integration Token."""
        if "token" not in credentials:
            logger.error("Missing Notion Integration Token")
            return False

        try:
            from notion_client import Client

            self._client = Client(auth=credentials["token"])
            return True
        except Exception as e:
            logger.error(f"Notion client initialization failed: {e}")
            return False

    def fetch_document(self, document_id: str) -> Optional[RemoteDocument]:
        """Retrieve a Notion page and parse its blocks."""
        if not self._client:
            return None

        try:
            # 1. Fetch Page Metadata
            page = self._client.pages.retrieve(page_id=document_id)
            title = self._extract_page_title(page)

            # 2. Fetch and parse blocks
            blocks = self._client.blocks.children.list(block_id=document_id).get(
                "results", []
            )
            content = self._parse_blocks(blocks)

            return RemoteDocument(
                content=content,
                title=title,
                source_id=document_id,
                mime_type="text/markdown",
                metadata={
                    "platform": "notion",
                    "created_time": page.get("created_time"),
                    "last_edited": page.get("last_edited_time"),
                },
            )
        except Exception as e:
            logger.error(f"Failed to fetch Notion page {document_id}: {e}")
            return None

    def test_connection(self) -> bool:
        return self._client is not None

    def _extract_page_title(self, page: Dict[str, Any]) -> str:
        """Helper to extract title from various Notion property types."""
        properties = page.get("properties", {})
        # Title is usually in a property named 'title' or 'Name'
        title_prop = properties.get("title") or properties.get("Name") or {}
        title_list = title_prop.get("title", [])
        if title_list:
            return title_list[0].get("plain_text", "Untitled Notion Page")
        return "Untitled Notion Page"

    def _parse_blocks(self, blocks: List[Dict[str, Any]]) -> str:
        """Recursively walk blocks to aggregate text."""
        text_parts = []
        for block in blocks:
            b_type = block.get("type")
            if not b_type:
                continue

            # Extract plain text from supported block types
            block_data = block.get(b_type, {})
            rich_text = block_data.get("rich_text", [])

            for part in rich_text:
                text_parts.append(part.get("plain_text", ""))

            # Note: Recursive children handling would go here for multi-level nesting
            # Follows Rule #2: Max depth 1 for MVP to prevent runaway recursion

        return "\n".join(text_parts)
