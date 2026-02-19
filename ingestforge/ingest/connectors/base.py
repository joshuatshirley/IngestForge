"""Remote Connector Base.

Defines the interface for external source ingestion (GDocs, Notion, etc.).
Follows NASA JPL Rule #4 (Modular) and Rule #9 (Type Hints).

Added IConnector interface for IF-Protocol compliance.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

MAX_DOWNLOAD_SIZE_MB = 100
MAX_PENDING_FILES = 500
SUPPORTED_HTTP_CODES = frozenset({200, 201, 202, 204})


@dataclass
class RemoteDocument:
    """Standardized representation of a document fetched from a remote source."""

    content: str
    title: str
    source_id: str  # e.g. GDoc file ID
    metadata: Dict[str, Any]
    mime_type: str


class RemoteConnector(ABC):
    """Abstract base for all external platform connectors."""

    @abstractmethod
    def authenticate(self, credentials: Dict[str, str]) -> bool:
        """Initialize platform-specific authentication."""
        pass

    @abstractmethod
    def fetch_document(self, document_id: str) -> Optional[RemoteDocument]:
        """Retrieve and parse a document from the remote source."""
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify API connectivity and credentials."""
        pass


class IConnector(ABC):
    """
    IF-Protocol compliant connector interface.

    Automates intake from external sources (GDrive, Web URLs).
    Returns IFArtifact objects instead of raw documents.
    """

    @abstractmethod
    def connect(self, config: Dict[str, Any]) -> bool:
        """
        Establish connection to the external source.

        Rule #7: Returns False on authentication failure.
        """
        pass

    @abstractmethod
    def discover(self) -> List[Dict[str, Any]]:
        """
        Discover new documents available for ingestion.

        Returns list of document descriptors with 'id', 'title', 'modified'.
        """
        pass

    @abstractmethod
    def fetch(self, document_id: str, output_dir: Path) -> "IFConnectorResult":
        """
        Fetch document and save to output directory.

        Rule #7: Returns IFFailureArtifact on HTTP 404/500.
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Clean up connection resources."""
        pass


@dataclass
class IFConnectorResult:
    """
    Result of a connector fetch operation.

    Standardized result for all connectors.
    """

    success: bool
    file_path: Optional[Path] = None
    artifact_id: Optional[str] = None
    error_message: Optional[str] = None
    http_status: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    def to_artifact(self) -> "Union[IFFileArtifact, IFFailureArtifact]":
        """
        Convert result to appropriate IFArtifact.

        Rule #7: Explicit artifact type based on success.
        """
        from ingestforge.core.pipeline.artifacts import (
            IFFileArtifact,
            IFFailureArtifact,
        )

        if self.success and self.file_path:
            return IFFileArtifact(
                file_path=self.file_path,
                metadata=self.metadata or {},
            )
        return IFFailureArtifact(
            error_message=self.error_message or "Unknown fetch error",
            metadata={"http_status": self.http_status},
        )
