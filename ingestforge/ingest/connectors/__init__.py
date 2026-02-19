"""Remote source connectors.

IConnector interface and implementations for external sources.
GDrive incremental sync with state tracking.
"""

from ingestforge.ingest.connectors.base import (
    IConnector,
    IFConnectorResult,
    RemoteConnector,
    RemoteDocument,
)
from ingestforge.ingest.connectors.gdrive import GDriveConnector
from ingestforge.ingest.connectors.gdrive_sync import (
    GDriveSyncManager,
    GDriveSyncReport,
)
from ingestforge.ingest.connectors.web_scraper import WebScraperConnector

__all__ = [
    # Base interfaces
    "IConnector",
    "IFConnectorResult",
    "RemoteConnector",
    "RemoteDocument",
    # Implementations
    "GDriveConnector",
    "GDriveSyncManager",
    "GDriveSyncReport",
    "WebScraperConnector",
]
