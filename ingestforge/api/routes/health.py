"""Health and Status API Endpoints.

G-TICKET-103: Provides comprehensive health checks and system status endpoints
for monitoring and observability.
Endpoints:
- GET /v1/health - Health check with storage backend status
- GET /v1/status - Project statistics and storage info
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# =============================================================================
# RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class StorageHealth(BaseModel):
    """Storage backend health status."""

    healthy: bool = Field(..., description="Whether storage is healthy")
    backend: str = Field(..., description="Storage backend type")
    error: Optional[str] = Field(default=None, description="Error message if unhealthy")
    details: Dict[str, Any] = Field(
        default_factory=dict, description="Backend-specific details"
    )


class HealthResponse(BaseModel):
    """Enhanced health check response with storage status."""

    status: str = Field(
        ..., description="Overall health status: healthy, degraded, unhealthy"
    )
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    uptime_seconds: float = Field(default=0.0, description="Server uptime in seconds")
    storage: Optional[StorageHealth] = Field(
        default=None, description="Storage health details"
    )


class LibraryStats(BaseModel):
    """Statistics for a single library."""

    name: str = Field(..., description="Library name")
    chunk_count: int = Field(..., description="Number of chunks in library")


class StatusResponse(BaseModel):
    """Project statistics and system status."""

    status: str = Field(..., description="System status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    project_name: str = Field(default="", description="Project name from config")
    document_count: int = Field(default=0, description="Total documents ingested")
    chunk_count: int = Field(default=0, description="Total chunks in storage")
    total_embeddings: int = Field(default=0, description="Total embeddings indexed")
    storage_backend: str = Field(default="", description="Active storage backend type")
    libraries: List[LibraryStats] = Field(
        default_factory=list, description="Per-library statistics"
    )
    data_path: str = Field(default="", description="Data storage path")


# =============================================================================
# LAZY IMPORTS (Rule #6: Avoid slow startup)
# =============================================================================


class _LazyDeps:
    """Lazy loader for heavy dependencies."""

    _router = None
    _logger = None

    @classmethod
    def get_router(cls):
        """Get FastAPI router (lazy-loaded)."""
        if cls._router is None:
            try:
                from fastapi import APIRouter

                cls._router = APIRouter(prefix="/v1", tags=["health"])
            except ImportError:
                raise ImportError(
                    "FastAPI is required for API functionality. "
                    "Install with: pip install fastapi uvicorn"
                )
        return cls._router

    @classmethod
    def get_logger(cls):
        """Get logger (lazy-loaded)."""
        if cls._logger is None:
            from ingestforge.core.logging import get_logger

            cls._logger = get_logger(__name__)
        return cls._logger


# Initialize router
router = _LazyDeps.get_router()

# Module-level startup time (set when module loads)
_module_start_time: float = time.time()

# =============================================================================
# HELPER FUNCTIONS (Rule #4: <60 lines each)
# =============================================================================


def _get_storage_health() -> StorageHealth:
    """Get storage backend health status.

    Rule #4: Function body <60 lines.
    Rule #1: Early returns for error cases.

    Returns:
        StorageHealth with backend status
    """
    try:
        from ingestforge.core.config import Config
        from ingestforge.storage.factory import check_health

        config = Config.load()
        health_result = check_health(config)

        return StorageHealth(
            healthy=health_result.get("healthy", False),
            backend=health_result.get("backend", "unknown"),
            error=health_result.get("error"),
            details=health_result.get("details", {}),
        )

    except Exception as e:
        return StorageHealth(
            healthy=False,
            backend="unknown",
            error=str(e),
            details={},
        )


def _get_project_stats() -> Dict[str, Any]:
    """Get project statistics from state and storage.

    Rule #4: Function body <60 lines.

    Returns:
        Dictionary with project statistics
    """
    try:
        from ingestforge.core.config import Config
        from ingestforge.core.state import ProcessingState
        from ingestforge.storage.factory import get_storage_backend

        config = Config.load()

        # Load processing state for document counts
        state_file = config.data_path / "pipeline_state.json"
        state = ProcessingState.load(state_file)

        # Get storage backend for chunk counts
        storage = get_storage_backend(config)
        chunk_count = storage.count()

        # Get library statistics
        libraries = []
        try:
            library_names = storage.get_libraries()
            for lib_name in library_names:
                lib_count = storage.count_by_library(lib_name)
                libraries.append({"name": lib_name, "chunk_count": lib_count})
        except NotImplementedError:
            # Backend doesn't support library operations
            libraries = [{"name": "default", "chunk_count": chunk_count}]

        return {
            "project_name": config.project.name,
            "document_count": state.total_documents,
            "chunk_count": chunk_count,
            "total_embeddings": state.total_embeddings,
            "storage_backend": config.storage.backend,
            "libraries": libraries,
            "data_path": str(config.data_path),
        }

    except Exception as e:
        _LazyDeps.get_logger().warning(f"Failed to get project stats: {e}")
        return {
            "project_name": "",
            "document_count": 0,
            "chunk_count": 0,
            "total_embeddings": 0,
            "storage_backend": "unknown",
            "libraries": [],
            "data_path": "",
            "error": str(e),
        }


# =============================================================================
# ENDPOINT HANDLERS (Rule #4: <60 lines each)
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint with storage backend status.

    Rule #4: Function body <60 lines.

    Returns 200 with status:
    - "healthy": All systems operational
    - "degraded": Storage unhealthy but API responding
    - "unhealthy": Critical systems down

    Returns:
        HealthResponse with comprehensive health status
    """
    logger = _LazyDeps.get_logger()

    # Get application version
    version = "1.0.0"  # Default version
    try:
        from importlib.metadata import version as get_version

        version = get_version("ingestforge")
    except Exception as e:
        _LazyDeps.get_logger().debug(f"Version detection failed: {e}")

    # Calculate uptime
    uptime = time.time() - _module_start_time

    # Check storage health
    storage_health = _get_storage_health()

    # Determine overall status
    if storage_health.healthy:
        status = "healthy"
    else:
        status = "degraded"
        logger.warning(f"Storage health degraded: {storage_health.error}")

    return HealthResponse(
        status=status,
        version=version,
        timestamp=datetime.now().isoformat(),
        uptime_seconds=round(uptime, 2),
        storage=storage_health,
    )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get project statistics and system status.

    Rule #4: Function body <60 lines.

    Returns project statistics including:
    - Document and chunk counts
    - Storage backend information
    - Per-library statistics

    Returns:
        StatusResponse with comprehensive project statistics
    """
    # Get application version
    version = "1.0.0"
    try:
        from importlib.metadata import version as get_version

        version = get_version("ingestforge")
    except Exception as e:
        _LazyDeps.get_logger().debug(f"Version detection failed: {e}")

    # Get project statistics
    stats = _get_project_stats()

    # Build library stats list
    library_stats = [
        LibraryStats(name=lib["name"], chunk_count=lib["chunk_count"])
        for lib in stats.get("libraries", [])
    ]

    # Determine status based on whether we got valid stats
    status = "ok" if "error" not in stats else "degraded"

    return StatusResponse(
        status=status,
        version=version,
        timestamp=datetime.now().isoformat(),
        project_name=stats.get("project_name", ""),
        document_count=stats.get("document_count", 0),
        chunk_count=stats.get("chunk_count", 0),
        total_embeddings=stats.get("total_embeddings", 0),
        storage_backend=stats.get("storage_backend", ""),
        libraries=library_stats,
        data_path=stats.get("data_path", ""),
    )
