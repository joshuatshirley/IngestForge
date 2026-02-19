"""IngestForge API Routes.

Provides modular routing for different API endpoints.
"""

from ingestforge.api.routes.ingestion import router as ingestion_router
from ingestforge.api.routes.retrieval import router as retrieval_router
from ingestforge.api.routes.health import router as health_router
from ingestforge.api.routes.viz import router as viz_router

__all__ = ["ingestion_router", "retrieval_router", "health_router", "viz_router"]
