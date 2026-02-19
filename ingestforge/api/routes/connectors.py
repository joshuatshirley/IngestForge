"""URL Connector Router with Background Processing.

Web URL Connector - Epic (REST API Endpoint)
Timestamp: 2026-02-18 21:00 UTC

Endpoints:
- POST /v1/connectors/url - Fetch single URL for background processing
- POST /v1/connectors/url/batch - Batch process URLs from list
- GET /v1/connectors/status/{job_id} - Check processing status

Follows NASA JPL Rules #4 (<60 lines/function), #7 (Check Returns), #9 (Type Hints).
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, HttpUrl

if TYPE_CHECKING:
    from fastapi import BackgroundTasks

# =============================================================================
# REQUEST/RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class URLIngestRequest(BaseModel):
    """Request model for single URL ingestion.

    Epic (API Request Schema)
    Timestamp: 2026-02-18 21:00 UTC
    """

    url: HttpUrl = Field(..., description="Web URL to ingest")
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom HTTP headers (optional)",
    )
    project: Optional[str] = Field(
        default=None,
        description="Project name for organization (optional)",
    )


class BatchURLIngestRequest(BaseModel):
    """Request model for batch URL ingestion.

    Epic (Batch API)
    Timestamp: 2026-02-18 21:00 UTC
    """

    urls: List[HttpUrl] = Field(
        ...,
        min_items=1,
        max_items=100,  # JPL Rule #2: Fixed upper bound
        description="List of URLs to ingest (max 100)",
    )
    headers: Optional[Dict[str, str]] = Field(
        default=None,
        description="Custom HTTP headers applied to all URLs (optional)",
    )
    skip_errors: bool = Field(
        default=False,
        description="Continue processing on URL errors",
    )
    project: Optional[str] = Field(
        default=None,
        description="Project name for organization (optional)",
    )


class URLIngestResponse(BaseModel):
    """Response model for URL ingestion.

    Epic (API Response Schema)
    Timestamp: 2026-02-18 21:00 UTC
    """

    job_id: str = Field(..., description="Unique job identifier")
    url: str = Field(..., description="Ingested URL")
    status: str = Field(..., description="Initial status (PENDING)")
    message: str = Field(..., description="Status message")


class BatchURLIngestResponse(BaseModel):
    """Response model for batch URL ingestion.

    Epic (Batch Response Schema)
    Timestamp: 2026-02-18 21:00 UTC
    """

    job_id: str = Field(..., description="Unique batch job identifier")
    urls_count: int = Field(..., description="Number of URLs queued")
    status: str = Field(..., description="Initial status (PENDING)")
    message: str = Field(..., description="Status message")


class ConnectorStatusResponse(BaseModel):
    """Response model for connector job status.

    Epic (Status API)
    Timestamp: 2026-02-18 21:00 UTC
    """

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Processing status")
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress (0.0-1.0)",
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Processing result if completed",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed",
    )


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

                cls._router = APIRouter(
                    prefix="/v1/connectors",
                    tags=["connectors"],
                )
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
# HELPER FUNCTIONS (Rule #4: <60 lines each)
# =============================================================================


def generate_connector_job_id() -> str:
    """Generate unique connector job ID.

    Epic
    Timestamp: 2026-02-18 21:00 UTC

    Rule #4: Simple helper function.

    Returns:
        Unique UUID-based job identifier
    """
    return f"conn_{uuid.uuid4().hex[:12]}"


def _validate_url_security(url: str) -> None:
    """Validate URL for SSRF vulnerabilities.

    Epic (Security Validation)
    Timestamp: 2026-02-18 21:00 UTC

    Rule #7: Input validation before processing.

    Args:
        url: URL to validate

    Raises:
        ValueError: If URL fails security validation
    """
    from ingestforge.core.security.url import validate_url

    is_valid, error_msg = validate_url(url)
    if not is_valid:
        raise ValueError(f"URL security validation failed: {error_msg}")


async def process_url_task(
    job_id: str,
    url: str,
    custom_headers: Optional[Dict[str, str]],
    state_manager: Any,
) -> None:
    """Background task for URL processing.

    Epic (Background Processing)
    Timestamp: 2026-02-18 21:00 UTC

    Rule #4: Function <60 lines.
    Rule #7: Validates inputs before processing.

    Args:
        job_id: Job identifier
        url: URL to fetch
        custom_headers: Custom HTTP headers
        state_manager: StateManager instance for tracking
    """
    logger = _LazyDeps.get_logger()

    try:
        # Update status to PROCESSING
        from ingestforge.core.state import ProcessingStatus

        doc_state = state_manager.get_or_create_document(job_id, url)
        doc_state.status = ProcessingStatus.SPLITTING  # Start processing
        state_manager.update_document(doc_state)

        # Import WebScraperConnector
        from ingestforge.ingest.connectors.web_scraper import WebScraperConnector
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact
        from ingestforge.core.pipeline.pipeline import Pipeline

        # Create connector
        connector = WebScraperConnector()

        # Build config with custom headers
        config: Dict[str, Any] = {}
        if custom_headers:
            config["headers"] = custom_headers

        # Connect
        if not connector.connect(config):
            raise Exception("Failed to initialize web connector")

        try:
            # Fetch URL content as artifact
            artifact = connector.fetch_to_artifact(url)

            # Check if fetch failed
            if isinstance(artifact, IFFailureArtifact):
                raise Exception(f"Failed to fetch URL: {artifact.error_message}")

            # Process artifact through pipeline
            pipeline = Pipeline()
            result = pipeline.process_artifact(artifact)

            # Update state based on result
            if result.success:
                doc_state.status = ProcessingStatus.COMPLETED
                doc_state.total_chunks = result.chunks_created
                doc_state.indexed_chunks = result.chunks_indexed
                doc_state.complete()
                logger.info(
                    f"Job {job_id} completed: {result.chunks_created} chunks from {url}"
                )
            else:
                doc_state.fail(result.error_message or "Unknown error")
                logger.error(f"Job {job_id} failed: {result.error_message}")

            state_manager.update_document(doc_state)

        finally:
            # Clean up connector
            connector.disconnect()

    except Exception as e:
        logger.exception(f"Background URL task failed for job {job_id}: {e}")
        # Update state to FAILED
        from ingestforge.core.state import ProcessingStatus

        doc_state = state_manager.get_or_create_document(job_id, url)
        doc_state.fail(str(e))
        state_manager.update_document(doc_state)


async def process_batch_url_task(
    job_id: str,
    urls: List[str],
    custom_headers: Optional[Dict[str, str]],
    skip_errors: bool,
    state_manager: Any,
) -> None:
    """Background task for batch URL processing.

    Epic (Batch Processing)
    Timestamp: 2026-02-18 21:00 UTC

    Rule #4: Function <60 lines.
    Rule #2: Bounded iteration over URLs.

    Args:
        job_id: Batch job identifier
        urls: List of URLs to process
        custom_headers: Custom HTTP headers
        skip_errors: Continue on errors
        state_manager: StateManager instance
    """
    logger = _LazyDeps.get_logger()

    success_count = 0
    error_count = 0

    # Process each URL (Rule #2: Bounded by list size)
    for idx, url in enumerate(urls, 1):
        sub_job_id = f"{job_id}_{idx}"

        try:
            # Validate URL
            _validate_url_security(url)

            # Process URL
            await process_url_task(sub_job_id, url, custom_headers, state_manager)
            success_count += 1

        except Exception as e:
            logger.error(f"Batch URL {idx}/{len(urls)} failed ({url}): {e}")
            error_count += 1
            if not skip_errors:
                break

    logger.info(
        f"Batch job {job_id} complete: {success_count}/{len(urls)} succeeded, "
        f"{error_count} failed"
    )


def _get_status_string(processing_status: Any) -> str:
    """Convert ProcessingStatus to API status string.

    Rule #4: Helper for status conversion.

    Args:
        processing_status: ProcessingStatus enum value

    Returns:
        API status string
    """
    from ingestforge.core.state import ProcessingStatus

    # Map internal statuses to public API statuses
    if processing_status == ProcessingStatus.PENDING:
        return "PENDING"
    elif processing_status == ProcessingStatus.COMPLETED:
        return "COMPLETED"
    elif processing_status == ProcessingStatus.FAILED:
        return "FAILED"
    else:
        # All intermediate stages map to PROCESSING
        return "PROCESSING"


# Import BackgroundTasks for dependency injection
from fastapi import BackgroundTasks

# =============================================================================
# ENDPOINT HANDLERS (Rule #4: <60 lines each)
# =============================================================================


@router.post("/url", response_model=URLIngestResponse)
async def ingest_url(
    request: URLIngestRequest,
    background_tasks: BackgroundTasks,
) -> URLIngestResponse:
    """Ingest single web URL.

    Epic (REST API Endpoint)
    Timestamp: 2026-02-18 21:00 UTC

    Rule #4: Function body <60 lines.
    Rule #7: Input validated via Pydantic and helpers.

    Args:
        request: URLIngestRequest with URL and headers
        background_tasks: FastAPI BackgroundTasks (injected)

    Returns:
        URLIngestResponse with job ID and status

    Raises:
        HTTPException: If URL validation or processing fails
    """
    from fastapi import HTTPException
    from ingestforge.core.pipeline.pipeline import Pipeline
    from ingestforge.core.state import StateManager, ProcessingStatus

    logger = _LazyDeps.get_logger()

    try:
        # Validate URL security (SSRF prevention)
        url_str = str(request.url)
        _validate_url_security(url_str)

        # Generate job ID
        job_id = generate_connector_job_id()

        # Initialize pipeline
        pipeline = Pipeline()
        state_file = pipeline.config.data_path / "pipeline_state.json"
        project_name = request.project or pipeline.config.project.name
        state_manager = StateManager(state_file, project_name=project_name)

        # Create document state as PENDING
        doc_state = state_manager.get_or_create_document(job_id, url_str)
        doc_state.status = ProcessingStatus.PENDING
        state_manager.update_document(doc_state)

        # Add background task
        background_tasks.add_task(
            process_url_task,
            job_id,
            url_str,
            request.headers,
            state_manager,
        )

        logger.info(f"URL ingestion queued: {url_str} -> {job_id}")

        return URLIngestResponse(
            job_id=job_id,
            url=url_str,
            status="PENDING",
            message=f"URL queued for processing: {url_str}",
        )

    except ValueError as e:
        logger.warning(f"URL validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"URL ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}") from e


@router.post("/url/batch", response_model=BatchURLIngestResponse)
async def ingest_url_batch(
    request: BatchURLIngestRequest,
    background_tasks: BackgroundTasks,
) -> BatchURLIngestResponse:
    """Batch ingest multiple web URLs.

    Epic (Batch API Endpoint)
    Timestamp: 2026-02-18 21:00 UTC

    Rule #4: Function body <60 lines.
    Rule #2: Bounded by max_items=100 in Pydantic model.

    Args:
        request: BatchURLIngestRequest with URLs and headers
        background_tasks: FastAPI BackgroundTasks (injected)

    Returns:
        BatchURLIngestResponse with job ID and count

    Raises:
        HTTPException: If validation or processing fails
    """
    from fastapi import HTTPException
    from ingestforge.core.pipeline.pipeline import Pipeline
    from ingestforge.core.state import StateManager

    logger = _LazyDeps.get_logger()

    try:
        # Validate all URLs for security
        urls_str = [str(url) for url in request.urls]
        for url in urls_str:
            _validate_url_security(url)

        # Generate batch job ID
        job_id = generate_connector_job_id()

        # Initialize pipeline
        pipeline = Pipeline()
        state_file = pipeline.config.data_path / "pipeline_state.json"
        project_name = request.project or pipeline.config.project.name
        state_manager = StateManager(state_file, project_name=project_name)

        # Add background task
        background_tasks.add_task(
            process_batch_url_task,
            job_id,
            urls_str,
            request.headers,
            request.skip_errors,
            state_manager,
        )

        logger.info(f"Batch URL ingestion queued: {len(urls_str)} URLs -> {job_id}")

        return BatchURLIngestResponse(
            job_id=job_id,
            urls_count=len(urls_str),
            status="PENDING",
            message=f"Batch job queued: {len(urls_str)} URLs",
        )

    except ValueError as e:
        logger.warning(f"Batch URL validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Batch URL ingestion failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Batch ingestion failed: {e}"
        ) from e


@router.get("/status/{job_id}", response_model=ConnectorStatusResponse)
async def get_connector_status(job_id: str) -> ConnectorStatusResponse:
    """Get processing status for a connector job.

    Epic (Status API)
    Timestamp: 2026-02-18 21:00 UTC

    Rule #4: Function body <60 lines.
    Rule #7: Validates job_id parameter.

    Args:
        job_id: Job identifier (conn_XXX format)

    Returns:
        ConnectorStatusResponse with current status and progress

    Raises:
        HTTPException: If job not found or status check fails
    """
    from fastapi import HTTPException
    from ingestforge.core.pipeline.pipeline import Pipeline
    from ingestforge.core.state import StateManager, ProcessingStatus

    logger = _LazyDeps.get_logger()

    # Validate job_id format
    if not job_id or not job_id.startswith("conn_"):
        raise HTTPException(status_code=400, detail="Invalid job ID format")

    try:
        # Initialize pipeline to get config
        pipeline = Pipeline()
        state_file = pipeline.config.data_path / "pipeline_state.json"
        state_manager = StateManager(
            state_file, project_name=pipeline.config.project.name
        )

        # Get document state
        doc_state = state_manager.state.get_document(job_id)

        if doc_state is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

        # Convert status to API format
        status = _get_status_string(doc_state.status)

        # Calculate progress based on status
        progress = 0.0
        if doc_state.status == ProcessingStatus.COMPLETED:
            progress = 1.0
        elif doc_state.status == ProcessingStatus.FAILED:
            progress = 0.0
        elif doc_state.status != ProcessingStatus.PENDING:
            # Intermediate status - estimate progress
            progress = 0.5

        # Build result if completed
        result = None
        if doc_state.status == ProcessingStatus.COMPLETED:
            result = {
                "chunks_created": doc_state.total_chunks,
                "chunks_indexed": doc_state.indexed_chunks,
                "source_url": doc_state.source_file,
            }

        return ConnectorStatusResponse(
            job_id=job_id,
            status=status,
            progress=progress,
            result=result,
            error=doc_state.error_message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Status check failed for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}") from e
