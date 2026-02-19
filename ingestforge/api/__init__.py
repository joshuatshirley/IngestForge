"""IngestForge REST API.

Provides HTTP endpoints for IDE integration and external tool connectivity.
Follows NASA JPL Commandments for all endpoint handlers.

TICKET-504: FastAPI core application with middleware and error handling.
TICKET-505: Ingestion router with background tasks.
"""

from ingestforge.api.bridge import router as bridge_router
from ingestforge.api.routes.ingestion import router as ingestion_router
from ingestforge.api.main import app

__all__ = ["bridge_router", "ingestion_router", "app"]
