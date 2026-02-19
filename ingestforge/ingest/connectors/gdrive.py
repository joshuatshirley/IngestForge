"""Google Drive IConnector Implementation.

Downloads files from Google Drive to a local pending directory.
Follows NASA JPL Rules #4 (Modular), #7 (Check Returns), #9 (Type Hints).

GDrive handler identifies and downloads new files into pending/.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.ingest.connectors.base import (
    IConnector,
    IFConnectorResult,
    MAX_DOWNLOAD_SIZE_MB,
    MAX_PENDING_FILES,
)

logger = get_logger(__name__)

GDRIVE_MIME_MAP: Dict[str, str] = {
    "application/vnd.google-apps.document": ".docx",
    "application/vnd.google-apps.spreadsheet": ".xlsx",
    "application/vnd.google-apps.presentation": ".pptx",
    "application/pdf": ".pdf",
    "text/plain": ".txt",
    "text/markdown": ".md",
}


class GDriveConnector(IConnector):
    """
    Google Drive connector implementing IF-Protocol.

    Rule #2: MAX_PENDING_FILES bounds discovery.
    Rule #7: Check HTTP status and return IFFailureArtifact on error.
    """

    def __init__(self) -> None:
        """Initialize connector state."""
        self._service: Any = None
        self._credentials: Any = None
        self._folder_id: Optional[str] = None

    def connect(self, config: Dict[str, Any]) -> bool:
        """
        Establish connection using OAuth credentials.

        Rule #7: Validate config keys before proceeding.
        """
        if "token" not in config and "credentials_file" not in config:
            logger.error("Missing 'token' or 'credentials_file' in config")
            return False

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build

            if "token" in config:
                self._credentials = Credentials(token=config["token"])
            else:
                from google.oauth2 import service_account

                self._credentials = (
                    service_account.Credentials.from_service_account_file(
                        config["credentials_file"]
                    )
                )

            self._service = build("drive", "v3", credentials=self._credentials)
            self._folder_id = config.get("folder_id")
            return True
        except ImportError:
            logger.error("google-api-python-client not installed")
            return False
        except Exception as e:
            logger.error(f"GDrive connect failed: {e}")
            return False

    def discover(self) -> List[Dict[str, Any]]:
        """
        List files in the configured folder.

        Rule #2: Limited to MAX_PENDING_FILES.
        """
        if not self._service:
            logger.error("Connector not connected")
            return []

        query = "trashed = false"
        if self._folder_id:
            query = f"'{self._folder_id}' in parents and {query}"

        try:
            results = (
                self._service.files()
                .list(
                    q=query,
                    pageSize=min(MAX_PENDING_FILES, 100),
                    fields="files(id, name, mimeType, modifiedTime, size)",
                )
                .execute()
            )

            files = results.get("files", [])
            return [
                {
                    "id": f["id"],
                    "title": f["name"],
                    "mime_type": f.get("mimeType", "application/octet-stream"),
                    "modified": f.get("modifiedTime"),
                    "size_bytes": int(f.get("size", 0)),
                }
                for f in files[:MAX_PENDING_FILES]
            ]
        except Exception as e:
            logger.error(f"GDrive discover failed: {e}")
            return []

    def fetch(self, document_id: str, output_dir: Path) -> IFConnectorResult:
        """
        Download file to output directory.

        Rule #7: Check file size before download; return failure on error.
        """
        if not self._service:
            return IFConnectorResult(
                success=False,
                error_message="Connector not connected",
                http_status=401,
            )

        try:
            # Get file metadata first
            file_meta = (
                self._service.files()
                .get(
                    fileId=document_id,
                    fields="id, name, mimeType, size",
                )
                .execute()
            )

            size_mb = int(file_meta.get("size", 0)) / (1024 * 1024)
            if size_mb > MAX_DOWNLOAD_SIZE_MB:
                return IFConnectorResult(
                    success=False,
                    error_message=f"File exceeds {MAX_DOWNLOAD_SIZE_MB}MB limit",
                    http_status=413,
                )

            mime_type = file_meta.get("mimeType", "")
            file_name = file_meta.get("name", document_id)
            export_ext = GDRIVE_MIME_MAP.get(mime_type, "")

            # Download content
            content = self._download_content(document_id, mime_type)
            if content is None:
                return IFConnectorResult(
                    success=False,
                    error_message="Download failed",
                    http_status=500,
                )

            # Write to output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"{file_name}{export_ext}"
            output_file.write_bytes(content)

            return IFConnectorResult(
                success=True,
                file_path=output_file,
                metadata={
                    "source": "gdrive",
                    "file_id": document_id,
                    "mime_type": mime_type,
                },
            )
        except Exception as e:
            logger.error(f"GDrive fetch failed: {e}")
            return IFConnectorResult(
                success=False,
                error_message=str(e),
                http_status=500,
            )

    def _download_content(self, file_id: str, mime_type: str) -> Optional[bytes]:
        """
        Download or export file content.

        Rule #4: Isolated download logic.
        """
        try:
            from googleapiclient.http import MediaIoBaseDownload
            import io

            # Google Docs need export, regular files use get_media
            if mime_type.startswith("application/vnd.google-apps"):
                export_mime = "application/pdf"
                if mime_type == "application/vnd.google-apps.document":
                    export_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif mime_type == "application/vnd.google-apps.spreadsheet":
                    export_mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

                request = self._service.files().export_media(
                    fileId=file_id, mimeType=export_mime
                )
            else:
                request = self._service.files().get_media(fileId=file_id)

            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Download content failed: {e}")
            return None

    def disconnect(self) -> None:
        """Clean up service connection."""
        self._service = None
        self._credentials = None
        self._folder_id = None
