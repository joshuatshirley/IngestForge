"""
Evidence Links API - Evidence Highlight Overlay.

Provides bounding box coordinates for entity mentions in source documents.
Enables bidirectional sync between knowledge graph and PDF viewer.

JPL Power of Ten Compliance:
- Rule #2: Fixed upper bounds (MAX_LINKS_PER_QUERY = 1000)
- Rule #4: All functions < 60 lines
- Rule #5: Assertions at entry points
- Rule #7: Check return values
- Rule #9: Complete type hints

Epic: EP-14 (Foundry UI)
Feature: FE-11-01 (Visualization)
Implementation Date: 2026-02-18
"""

from fastapi import APIRouter, HTTPException, Query, status, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/v1/extract", tags=["evidence"])

# =============================================================================
# MODELS
# =============================================================================


class VerificationRequest(BaseModel):
    """
    Fact Verification Audit Trail.
    """

    highlight_id: str = Field(..., description="ID of the highlight or entity")
    status: str = Field(..., description="Status (verified or refuted)")
    document_id: str = Field(..., description="Document source ID")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional audit metadata"
    )


# ... existing code ...


@router.post("/verify", status_code=status.HTTP_201_CREATED)
async def verify_evidence(request: VerificationRequest = Body(...)):
    """
    Verify or refute a piece of evidence.
    Task 113: Backend verification audit endpoint.
    """
    if request.status not in ["verified", "refuted"]:
        raise HTTPException(status_code=400, detail="Invalid status")

    # Audit logging (JPL Rule #7: Return values aren't enough, we need logs)
    logger.info(
        f"AUDIT: {request.document_id} | {request.highlight_id} | {request.status}"
    )

    return {"status": "success", "message": f"Fact {request.status}"}


# =============================================================================
# CONSTANTS (JPL Rule #2: Fixed upper bounds)
# =============================================================================

MAX_LINKS_PER_QUERY = 1000  # Maximum evidence links returned
MAX_CONFIDENCE_THRESHOLD = 1.0
MIN_CONFIDENCE_THRESHOLD = 0.0
MAX_PAGE_NUMBER = 10000
MIN_PAGE_NUMBER = 1
MAX_CHUNKS_TO_PROCESS = 10000  # JPL Rule #2: Bounded chunk iteration
MAX_BBOX_PER_CHUNK = 100  # JPL Rule #2: Bounded bbox iteration per chunk
MAX_FILTER_FIELDS = 10  # JPL Rule #2: Bounded filter dict iteration

# =============================================================================
# SCHEMAS (JPL Rule #9: Complete type hints)
# =============================================================================


class BoundingBox(BaseModel):
    """
    Bounding box coordinates for visual elements.

    AC: Normalized 0-1 coordinates for page-independent positioning.
    """

    x1: float = Field(..., ge=0.0, le=1.0, description="Left edge (0-1 normalized)")
    y1: float = Field(..., ge=0.0, le=1.0, description="Top edge (0-1 normalized)")
    x2: float = Field(..., ge=0.0, le=1.0, description="Right edge (0-1 normalized)")
    y2: float = Field(..., ge=0.0, le=1.0, description="Bottom edge (0-1 normalized)")


class EvidenceLink(BaseModel):
    """
    Link from entity/chunk to source document location.

    AC: Evidence provenance for bidirectional sync.
    """

    chunk_id: str = Field(..., description="Chunk containing the entity mention")
    document_id: str = Field(..., description="Source document ID")
    page: int = Field(..., ge=1, description="Page number (1-indexed)")
    bbox: BoundingBox = Field(..., description="Bounding box on page")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence (0.0-1.0)"
    )
    entity_id: Optional[str] = Field(None, description="Related entity ID if known")
    text: Optional[str] = Field(None, description="Extracted text snippet")


class EvidenceLinksResponse(BaseModel):
    """
    Response containing evidence links for a document.

    AC: Paginated results with metadata.
    """

    document_id: str = Field(..., description="Document ID queried")
    total_links: int = Field(..., description="Total links found")
    links: List[EvidenceLink] = Field(..., description="Evidence links (max 1000)")
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Query filters applied"
    )


class DocumentMetadata(BaseModel):
    """
    Metadata about a source document.

    AC: Document info for PDF viewer.
    """

    document_id: str = Field(..., description="Unique document identifier")
    title: Optional[str] = Field(None, description="Document title")
    total_pages: int = Field(..., ge=1, description="Total number of pages")
    file_path: str = Field(..., description="Path to source PDF file")
    content_type: str = Field(default="application/pdf", description="MIME type")


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter(prefix="/v1/extract", tags=["evidence"])


# =============================================================================
# ENDPOINTS (JPL Rule #4: < 60 lines each)
# =============================================================================


@router.get(
    "/evidence-links",
    response_model=EvidenceLinksResponse,
    summary="Get evidence links for document",
    description="Retrieve bounding box coordinates for entity mentions in source documents",
    status_code=status.HTTP_200_OK,
)
async def get_evidence_links(
    document_id: str = Query(..., description="Document ID to query"),
    entity_id: Optional[str] = Query(None, description="Filter by entity ID"),
    page: Optional[int] = Query(
        None,
        ge=MIN_PAGE_NUMBER,
        le=MAX_PAGE_NUMBER,
        description="Filter by page number",
    ),
    min_confidence: float = Query(
        MIN_CONFIDENCE_THRESHOLD,
        ge=MIN_CONFIDENCE_THRESHOLD,
        le=MAX_CONFIDENCE_THRESHOLD,
        description="Minimum confidence threshold",
    ),
) -> EvidenceLinksResponse:
    """
    Get evidence links for a document with optional filters.

    API GET /v1/extract/evidence-links endpoint.

    Rule #4: Under 60 lines.
    Rule #5: Validate inputs (FastAPI + Pydantic handles this).
    Rule #7: Check return values.

    Args:
        document_id: Document ID to query.
        entity_id: Optional entity ID filter.
        page: Optional page number filter.
        min_confidence: Minimum confidence threshold (default 0.0).

    Returns:
        EvidenceLinksResponse with filtered links.

    Raises:
        HTTPException: If document not found or query fails.
    """
    try:
        # Import storage here to avoid circular dependency
        from ingestforge.storage.factory import get_storage

        storage = get_storage()

        # Retrieve all chunks for document
        chunks = storage.get_chunks_by_document(document_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chunks found for document_id='{document_id}'",
            )

        # Extract evidence links from chunk metadata
        links = _extract_evidence_links(chunks, entity_id, page, min_confidence)

        # Enforce upper bound (JPL Rule #2)
        if len(links) > MAX_LINKS_PER_QUERY:
            logger.warning(
                f"Truncating {len(links)} links to {MAX_LINKS_PER_QUERY} for document {document_id}"
            )
            links = links[:MAX_LINKS_PER_QUERY]

        # Build response
        # JPL Rule #2: Bounded iteration over filter fields
        raw_filters = {
            "entity_id": entity_id,
            "page": page,
            "min_confidence": min_confidence,
        }
        filters_applied: Dict[str, Any] = {}
        filter_count = 0
        for key, value in raw_filters.items():
            if filter_count >= MAX_FILTER_FIELDS:
                break
            if value is not None:
                filters_applied[key] = value
            filter_count += 1

        return EvidenceLinksResponse(
            document_id=document_id,
            total_links=len(links),
            links=links,
            filters_applied=filters_applied,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get evidence links for {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve evidence links: {str(e)}",
        ) from e


@router.get(
    "/documents/{document_id}/metadata",
    response_model=DocumentMetadata,
    summary="Get document metadata",
    description="Retrieve metadata about a source document for PDF viewer",
    status_code=status.HTTP_200_OK,
)
async def get_document_metadata(document_id: str) -> DocumentMetadata:
    """
    Get metadata for a document.

    Provides document info for PDF viewer component.

    Rule #4: Under 60 lines.
    Rule #7: Check return values.

    Args:
        document_id: Document ID to query.

    Returns:
        DocumentMetadata with file path and page count.

    Raises:
        HTTPException: If document not found.
    """
    try:
        from ingestforge.storage.factory import get_storage

        storage = get_storage()

        # Get chunks to verify document exists and extract metadata
        chunks = storage.get_chunks_by_document(document_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Extract metadata from first chunk
        first_chunk = chunks[0]
        metadata = _get_chunk_metadata(first_chunk)

        file_path = metadata.get("source_file", "")
        total_pages = metadata.get("total_pages", 1)
        title = (
            metadata.get("title") or Path(file_path).stem if file_path else document_id
        )

        return DocumentMetadata(
            document_id=document_id,
            title=title,
            total_pages=total_pages,
            file_path=file_path,
            content_type="application/pdf",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get metadata for {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document metadata: {str(e)}",
        ) from e


@router.get(
    "/documents/{document_id}/pdf",
    summary="Serve PDF file",
    description="Serve the source PDF file for a document",
    status_code=status.HTTP_200_OK,
    response_class=FileResponse,
)
async def serve_document_pdf(document_id: str) -> FileResponse:
    """
    Serve the source PDF file for a document.

    Provides PDF file access for PDF.js viewer.

    Rule #4: Under 60 lines.
    Rule #5: Validate file exists.
    Rule #7: Check return values.

    Args:
        document_id: Document ID to retrieve.

    Returns:
        FileResponse with PDF content.

    Raises:
        HTTPException: If document or file not found.
    """
    try:
        from ingestforge.storage.factory import get_storage

        storage = get_storage()

        # Get chunks to verify document exists
        chunks = storage.get_chunks_by_document(document_id)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document not found: {document_id}",
            )

        # Extract file path from first chunk
        metadata = _get_chunk_metadata(chunks[0])
        file_path = metadata.get("source_file", "")

        if not file_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No source file path for document: {document_id}",
            )

        # Validate file exists
        pdf_path = Path(file_path)

        if not pdf_path.exists() or not pdf_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"PDF file not found: {file_path}",
            )

        # Serve file
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=pdf_path.name,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve PDF for {document_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to serve PDF file: {str(e)}",
        ) from e


# =============================================================================
# HELPER FUNCTIONS (JPL Rule #4: < 60 lines each)
# =============================================================================


def _get_chunk_metadata(chunk: Any) -> Dict[str, Any]:
    """
    Extract metadata from chunk (supports dict and object access).

    Rule #4: Under 60 lines.

    Args:
        chunk: Chunk record (dict or object).

    Returns:
        Metadata dictionary.
    """
    if isinstance(chunk, dict):
        return chunk.get("metadata", {})
    return getattr(chunk, "metadata", {})


def _extract_evidence_links(
    chunks: List[Any],
    entity_id_filter: Optional[str],
    page_filter: Optional[int],
    min_confidence: float,
) -> List[EvidenceLink]:
    """
    Extract evidence links from chunk metadata.

    Parse chunk metadata for bounding boxes.

    Rule #2: Bounded iteration (MAX_CHUNKS_TO_PROCESS, MAX_BBOX_PER_CHUNK).
    Rule #4: Under 60 lines (split into helpers).

    Args:
        chunks: List of chunk records from storage.
        entity_id_filter: Optional entity ID to filter by.
        page_filter: Optional page number to filter by.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of EvidenceLink objects.
    """
    links: List[EvidenceLink] = []

    # JPL Rule #2: Bounded iteration over chunks
    chunks_to_process = min(len(chunks), MAX_CHUNKS_TO_PROCESS)
    for i in range(chunks_to_process):
        chunk = chunks[i]
        # Extract metadata (support both dict and object access)
        if isinstance(chunk, dict):
            metadata = chunk.get("metadata", {})
            chunk_id = chunk.get("chunk_id", "")
            document_id = chunk.get("document_id", "")
        else:
            metadata = getattr(chunk, "metadata", {})
            chunk_id = getattr(chunk, "chunk_id", "")
            document_id = getattr(chunk, "document_id", "")

        # Check for bounding box data
        bbox_data = metadata.get("bbox") or metadata.get("bounding_boxes")

        if not bbox_data:
            continue  # Skip chunks without bbox data

        # Handle multiple bboxes per chunk (list) or single bbox (dict)
        bbox_list = bbox_data if isinstance(bbox_data, list) else [bbox_data]

        # JPL Rule #2: Bounded iteration over bboxes
        bbox_count = min(len(bbox_list), MAX_BBOX_PER_CHUNK)
        for j in range(bbox_count):
            bbox_entry = bbox_list[j]

            link = _process_bbox_entry(
                bbox_entry,
                chunk_id,
                document_id,
                metadata,
                entity_id_filter,
                page_filter,
                min_confidence,
            )

            if link:
                links.append(link)

    return links


def _process_bbox_entry(
    bbox_entry: Any,
    chunk_id: str,
    document_id: str,
    metadata: Dict[str, Any],
    entity_id_filter: Optional[str],
    page_filter: Optional[int],
    min_confidence: float,
) -> Optional[EvidenceLink]:
    """
    Process a single bbox entry and create EvidenceLink if valid.

    Helper to keep _extract_evidence_links under 60 lines.

    Rule #4: Under 60 lines.

    Returns:
        EvidenceLink if valid, None otherwise.
    """
    # Parse bounding box
    if not isinstance(bbox_entry, dict):
        return None

    # Apply filters
    page_num = bbox_entry.get("page", metadata.get("page_start", 1))
    entity_id = bbox_entry.get("entity_id")
    confidence = bbox_entry.get("confidence", 1.0)

    if page_filter is not None and page_num != page_filter:
        return None
    if entity_id_filter is not None and entity_id != entity_id_filter:
        return None
    if confidence < min_confidence:
        return None

    # Extract bounding box coordinates
    try:
        bbox = BoundingBox(
            x1=bbox_entry.get("x1", 0.0),
            y1=bbox_entry.get("y1", 0.0),
            x2=bbox_entry.get("x2", 1.0),
            y2=bbox_entry.get("y2", 1.0),
        )
    except Exception as e:
        logger.warning(f"Invalid bbox in chunk {chunk_id}: {e}")
        return None

    # Create evidence link
    return EvidenceLink(
        chunk_id=chunk_id,
        document_id=document_id,
        page=page_num,
        bbox=bbox,
        confidence=confidence,
        entity_id=entity_id,
        text=bbox_entry.get("text"),
    )
