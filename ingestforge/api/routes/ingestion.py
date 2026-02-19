"""Ingestion Router with Background Tasks.

TICKET-505: Implements file upload and background processing endpoints.
Endpoints:
- POST /v1/ingest/upload - Upload file for background processing
- GET /v1/ingest/status/{job_id} - Check processing status
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from fastapi import UploadFile, BackgroundTasks

# =============================================================================
# REQUEST/RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class UploadResponse(BaseModel):
    """Response model for file upload."""

    job_id: str = Field(..., description="Unique job identifier")
    filename: str = Field(..., description="Uploaded filename")
    status: str = Field(..., description="Initial status (PENDING)")
    message: str = Field(..., description="Status message")


class StatusResponse(BaseModel):
    """Response model for job status check."""

    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Processing status")
    progress: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Progress (0.0-1.0)"
    )
    result: Optional[Dict[str, Any]] = Field(
        default=None, description="Processing result if completed"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


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

                cls._router = APIRouter(prefix="/v1/ingest", tags=["ingestion"])
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


def generate_job_id() -> str:
    """Generate unique job ID.

    Rule #4: Simple helper function.

    Returns:
        Unique UUID-based job identifier
    """
    return f"job_{uuid.uuid4().hex[:12]}"


def _validate_file_upload(file, allowed_extensions: set[str]) -> None:
    """Validate uploaded file.

    Rule #7: Input validation before processing.

    Args:
        file: UploadFile from FastAPI
        allowed_extensions: Set of allowed file extensions

    Raises:
        ValueError: If file is invalid
    """
    # Check filename exists
    if not file.filename:
        raise ValueError("Filename is required")

    # Check file extension
    file_path = Path(file.filename)
    if file_path.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    # Check file is not empty (size check happens during read)
    if not file.file:
        raise ValueError("File content is empty")


async def _save_upload_to_pending(file, pending_dir: Path, job_id: str) -> Path:
    """Save uploaded file to pending directory.

    Rule #4: Extracted helper for file saving.
    Rule #7: Validates paths to prevent traversal attacks.

    Args:
        file: UploadFile from FastAPI
        pending_dir: Pending directory path
        job_id: Job identifier for tracking

    Returns:
        Path to saved file

    Raises:
        ValueError: If file operations fail
    """
    from ingestforge.core.security import SafeFileOperations

    # Use SafeFileOperations to prevent path traversal
    safe_ops = SafeFileOperations(pending_dir)

    # Create job-specific filename to avoid collisions
    file_path = Path(file.filename)
    safe_filename = f"{job_id}_{file_path.name}"

    # Validate the filename (returns absolute path with base_dir)
    full_path = safe_ops.validate_path(Path(safe_filename))

    # Save file content
    try:
        content = await file.read()

        # Check file size (max 100MB)
        max_size_mb = 100
        max_bytes = max_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(
                f"File too large: {len(content) / 1024 / 1024:.1f}MB (max {max_size_mb}MB)"
            )

        # Write file
        full_path.write_bytes(content)
        return full_path

    except Exception as e:
        # Clean up on error
        if full_path.exists():
            full_path.unlink()
        raise ValueError(f"Failed to save file: {e}") from e


async def process_file_task(job_id: str, file_path: Path, state_manager) -> None:
    """Background task for file processing.

    Rule #4: Function <60 lines.
    Rule #7: Validates inputs before processing.

    Args:
        job_id: Job identifier
        file_path: Path to file to process
        state_manager: StateManager instance for tracking
    """
    logger = _LazyDeps.get_logger()

    try:
        # Update status to PROCESSING
        doc_state = state_manager.get_or_create_document(job_id, str(file_path))
        doc_state.status = _get_processing_status("PROCESSING")
        state_manager.update_document(doc_state)

        # Import Pipeline (lazy to avoid circular deps)
        from ingestforge.core.pipeline.pipeline import Pipeline

        # Process the file
        pipeline = Pipeline()
        result = pipeline.process_file(file_path)

        # Update state based on result
        if result.success:
            doc_state.status = _get_processing_status("COMPLETED")
            doc_state.total_chunks = result.chunks_created
            doc_state.indexed_chunks = result.chunks_indexed
            doc_state.complete()
            logger.info(f"Job {job_id} completed: {result.chunks_created} chunks")
        else:
            doc_state.fail(result.error_message or "Unknown error")
            logger.error(f"Job {job_id} failed: {result.error_message}")

        state_manager.update_document(doc_state)

    except Exception as e:
        logger.exception(f"Background task failed for job {job_id}: {e}")
        # Update state to FAILED
        doc_state = state_manager.get_or_create_document(job_id, str(file_path))
        doc_state.fail(str(e))
        state_manager.update_document(doc_state)


def _get_processing_status(status_str: str):
    """Get ProcessingStatus enum from string.

    Rule #4: Helper to reduce duplication.

    Args:
        status_str: Status string

    Returns:
        ProcessingStatus enum value
    """
    from ingestforge.core.state import ProcessingStatus

    status_map = {
        "PENDING": ProcessingStatus.PENDING,
        "PROCESSING": ProcessingStatus.SPLITTING,  # Start with first stage
        "COMPLETED": ProcessingStatus.COMPLETED,
        "FAILED": ProcessingStatus.FAILED,
    }

    return status_map.get(status_str, ProcessingStatus.PENDING)


def _get_status_string(processing_status) -> str:
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


# Import File and BackgroundTasks for dependency injection
from fastapi import File, BackgroundTasks, UploadFile
# =============================================================================
# ENDPOINT HANDLERS (Rule #4: <60 lines each)
# =============================================================================


@router.post("/upload", response_model=UploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> UploadResponse:
    """Upload file for background processing.

    Rule #4: Function body <60 lines.
    Rule #7: Input validated via helper functions.

    Args:
        background_tasks: FastAPI BackgroundTasks (injected)
        file: Uploaded file (FastAPI UploadFile from form data)

    Returns:
        UploadResponse with job ID and status

    Raises:
        HTTPException: If upload fails
    """
    from fastapi import HTTPException
    from ingestforge.core.pipeline.pipeline import Pipeline
    from ingestforge.core.state import StateManager

    logger = _LazyDeps.get_logger()

    try:
        # Get supported formats from config
        pipeline = Pipeline()
        allowed_extensions = set(pipeline.config.ingest.supported_formats)

        # Validate file
        _validate_file_upload(file, allowed_extensions)

        # Generate job ID
        job_id = generate_job_id()

        # Save file to pending directory
        pending_dir = pipeline.config.pending_path
        pending_dir.mkdir(parents=True, exist_ok=True)
        file_path = await _save_upload_to_pending(file, pending_dir, job_id)

        # Initialize state manager
        state_file = pipeline.config.data_path / "pipeline_state.json"
        state_manager = StateManager(
            state_file, project_name=pipeline.config.project.name
        )

        # Create document state as PENDING
        doc_state = state_manager.get_or_create_document(job_id, str(file_path))
        doc_state.status = _get_processing_status("PENDING")
        state_manager.update_document(doc_state)

        # Add background task
        background_tasks.add_task(process_file_task, job_id, file_path, state_manager)

        logger.info(f"File upload successful: {file.filename} -> {job_id}")

        return UploadResponse(
            job_id=job_id,
            filename=file.filename,
            status="PENDING",
            message=f"File queued for processing: {file.filename}",
        )

    except ValueError as e:
        logger.warning(f"File upload validation failed: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}") from e


@router.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str) -> StatusResponse:
    """Get processing status for a job.

    Rule #4: Function body <60 lines.
    Rule #7: Validates job_id parameter.

    Args:
        job_id: Job identifier

    Returns:
        StatusResponse with current status and progress

    Raises:
        HTTPException: If job not found or status check fails
    """
    from fastapi import HTTPException
    from ingestforge.core.pipeline.pipeline import Pipeline
    from ingestforge.core.state import StateManager, ProcessingStatus

    logger = _LazyDeps.get_logger()

    # Validate job_id format
    if not job_id or not job_id.startswith("job_"):
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
                "source_file": doc_state.source_file,
            }

        return StatusResponse(
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
