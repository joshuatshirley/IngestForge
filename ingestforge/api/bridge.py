"""Integration Bridge API - REST endpoints for external tool connectivity.

TICKET-402: Enables IDE extensions and external tools to interact with
IngestForge via HTTP endpoints.
Endpoints:
- POST /v1/ingest/bridge - Ingest text directly
- POST /v1/query - Query the knowledge base
- GET /v1/status - Get system status
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# =============================================================================
# REQUEST/RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class IngestRequest(BaseModel):
    """Request model for text ingestion.

    Rule #7: All fields validated with Pydantic.
    """

    text: str = Field(
        ..., min_length=1, max_length=1_000_000, description="Text content to ingest"
    )
    source: str = Field(
        default="ide",
        max_length=256,
        description="Source identifier (e.g., 'vscode', 'ide')",
    )
    title: Optional[str] = Field(
        default=None, max_length=512, description="Optional title for the content"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional metadata"
    )

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Validate text is not just whitespace."""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace-only")
        return v


class IngestResponse(BaseModel):
    """Response model for ingestion result."""

    success: bool
    chunk_count: int = 0
    chunk_ids: List[str] = []
    document_id: Optional[str] = None
    message: str = ""
    processing_time_ms: float = 0.0


class QueryRequest(BaseModel):
    """Request model for knowledge base query."""

    query: str = Field(..., min_length=1, max_length=10_000, description="Search query")
    top_k: int = Field(
        default=5, ge=1, le=100, description="Number of results to return"
    )
    library: Optional[str] = Field(
        default=None, max_length=128, description="Filter by library"
    )


class QueryResult(BaseModel):
    """Single query result."""

    content: str
    source: str
    score: float
    chunk_id: str


class QueryResponse(BaseModel):
    """Response model for query result."""

    success: bool
    results: List[QueryResult] = []
    query_time_ms: float = 0.0
    message: str = ""


class StatusResponse(BaseModel):
    """System status response."""

    status: str = "ok"
    version: str = "1.2.0"
    document_count: int = 0
    chunk_count: int = 0
    timestamp: str = ""


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

                cls._router = APIRouter(prefix="/v1", tags=["bridge"])
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

# =============================================================================
# ENDPOINT HANDLERS (Rule #4: <60 lines each)
# =============================================================================


@router.post("/ingest/bridge", response_model=IngestResponse)
async def ingest_text(request: IngestRequest) -> IngestResponse:
    """Ingest text content directly into the knowledge base.

    Rule #4: Function body <60 lines.
    Rule #7: Input validated via Pydantic model.

    Args:
        request: IngestRequest with text and metadata

    Returns:
        IngestResponse with ingestion results
    """
    import time

    start_time = time.perf_counter()
    logger = _LazyDeps.get_logger()

    try:
        # Lazy import to avoid circular deps and slow startup
        from ingestforge.core.pipeline.pipeline import Pipeline

        # Initialize pipeline with default config
        pipeline = Pipeline()

        # Process the text
        result = pipeline.process_text(
            text=request.text,
            source=request.source,
            title=request.title,
            metadata=request.metadata or {},
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Bridge ingest completed: {result.chunks_created} chunks in {elapsed_ms:.2f}ms"
        )

        return IngestResponse(
            success=True,
            chunk_count=result.chunks_created,
            chunk_ids=result.chunk_ids,
            document_id=result.document_id,
            message=f"Successfully ingested {result.chunks_created} chunks",
            processing_time_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Bridge ingest failed: {e}")
        return IngestResponse(
            success=False,
            message=str(e),
            processing_time_ms=elapsed_ms,
        )


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest) -> QueryResponse:
    """Query the knowledge base.

    Rule #4: Function body <60 lines.

    Args:
        request: QueryRequest with search parameters

    Returns:
        QueryResponse with search results
    """
    import time

    start_time = time.perf_counter()
    logger = _LazyDeps.get_logger()

    try:
        from ingestforge.core.pipeline.pipeline import Pipeline

        # Initialize pipeline to access storage
        pipeline = Pipeline()
        raw_results = pipeline.query(
            query_text=request.query,
            top_k=request.top_k,
            library_filter=request.library,
        )

        results = [
            QueryResult(
                content=r.get("content", "")[:1000],  # Truncate for API response
                source=r.get("source_file", ""),
                score=r.get("score", 0.0),
                chunk_id=r.get("chunk_id", ""),
            )
            for r in raw_results
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(
            f"Bridge query completed: {len(results)} results in {elapsed_ms:.2f}ms"
        )

        return QueryResponse(
            success=True,
            results=results,
            query_time_ms=elapsed_ms,
        )

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Bridge query failed: {e}")
        return QueryResponse(
            success=False,
            message=str(e),
            query_time_ms=elapsed_ms,
        )


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    """Get system status and statistics.

    Rule #4: Function body <60 lines.

    Returns:
        StatusResponse with system info
    """
    logger = _LazyDeps.get_logger()

    try:
        from ingestforge.core.pipeline.pipeline import Pipeline

        pipeline = Pipeline()
        stats = pipeline.get_status()

        return StatusResponse(
            status="ok",
            version="1.2.0",
            document_count=stats.get("total_documents", 0),
            chunk_count=stats.get("total_chunks", 0),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.warning(f"Status check failed: {e}")
        return StatusResponse(
            status="degraded",
            timestamp=datetime.now().isoformat(),
        )


# =============================================================================
# SERVER RUNNER
# =============================================================================


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Run the API server.

    Args:
        host: Bind address
        port: Port number
    """
    try:
        import uvicorn
        from fastapi import FastAPI

        app = FastAPI(
            title="IngestForge API",
            description="REST API for IDE integration and external tools",
            version="1.2.0",
        )
        app.include_router(router)

        uvicorn.run(app, host=host, port=port)

    except ImportError:
        raise ImportError(
            "uvicorn is required to run the API server. "
            "Install with: pip install uvicorn"
        )
