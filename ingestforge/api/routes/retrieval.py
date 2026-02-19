"""Retrieval & Search Router.

TICKET-506: Implements search endpoints with citation fields.
Endpoints:
- GET /v1/search - Search the knowledge base with citations
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    pass

# =============================================================================
# REQUEST/RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class CitationInfo(BaseModel):
    """Citation information for a search result.

    Rule #9: Complete type hints for all fields.
    """

    source_url: Optional[str] = Field(
        default=None, description="Source file URL or path"
    )
    chunk_id: str = Field(..., description="Unique chunk identifier")
    relevance_score: float = Field(
        ..., ge=0.0, le=1.0, description="Relevance score (0.0-1.0)"
    )
    page_start: Optional[int] = Field(
        default=None, ge=1, description="Starting page number"
    )
    page_end: Optional[int] = Field(
        default=None, ge=1, description="Ending page number"
    )
    section_title: Optional[str] = Field(
        default=None, description="Section or chapter title"
    )


class SearchResultItem(BaseModel):
    """Single search result with citation.

    Rule #9: Complete type hints for all fields.
    """

    content: str = Field(..., description="Chunk text content")
    document_id: str = Field(..., description="Parent document identifier")
    citation: CitationInfo = Field(..., description="Citation information")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    word_count: int = Field(default=0, ge=0, description="Word count")
    library: str = Field(default="default", description="Library name")
    author_id: Optional[str] = Field(default=None, description="Author identifier")
    author_name: Optional[str] = Field(default=None, description="Author name")


class SearchResponse(BaseModel):
    """Response model for search endpoint.

    Rule #9: Complete type hints for all fields.
    """

    success: bool = Field(..., description="Whether search succeeded")
    results: List[SearchResultItem] = Field(
        default_factory=list, description="Search results"
    )
    total_results: int = Field(default=0, ge=0, description="Number of results")
    query: str = Field(..., description="Original query")
    query_time_ms: float = Field(default=0.0, ge=0.0, description="Query time in ms")
    message: str = Field(default="", description="Status message")


class SearchQueryParams(BaseModel):
    """Query parameters for search endpoint.

    Rule #7: Input validation via Pydantic.
    """

    query: str = Field(..., min_length=1, max_length=10_000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    project_id: Optional[str] = Field(
        default=None, max_length=256, description="Project/library filter"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v.strip()


class SearchRequest(BaseModel):
    """Request model for POST search endpoint.

    Redesigned search with filters and sorting.
    Rule #9: Complete type hints.
    """

    query: str = Field(..., min_length=1, max_length=10_000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results")
    project_id: Optional[str] = Field(
        default=None, max_length=256, description="Project/library filter"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Metadata filters (e.g., date, type)"
    )
    sort_by: Optional[str] = Field(
        default="relevance", description="Field to sort by (relevance, date, author)"
    )
    broadcast: bool = Field(
        default=False, description="Whether to query remote Nexus peers"
    )
    nexus_ids: Optional[List[str]] = Field(
        default=None, description="Specific peer IDs to target"
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v.strip()


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

                cls._router = APIRouter(prefix="/v1", tags=["retrieval"])
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


def _map_search_result_to_item(result) -> SearchResultItem:
    """Map internal SearchResult to API SearchResultItem.

    Rule #4: Function <60 lines.
    Rule #9: Complete type hints.

    Args:
        result: SearchResult dataclass from storage.base

    Returns:
        SearchResultItem for API response
    """
    # Build citation info
    citation = CitationInfo(
        source_url=result.source_file,
        chunk_id=result.chunk_id,
        relevance_score=result.score,
        page_start=result.page_start,
        page_end=result.page_end,
        section_title=result.section_title,
    )

    # Build response item
    return SearchResultItem(
        content=result.content,
        document_id=result.document_id,
        citation=citation,
        metadata=result.metadata or {},
        word_count=result.word_count,
        library=result.library,
        author_id=result.author_id,
        author_name=result.author_name,
    )


def _normalize_score(score: float, max_score: float = 1.0) -> float:
    """Normalize score to 0.0-1.0 range.

    Rule #4: Simple helper function.

    Args:
        score: Raw score value
        max_score: Maximum possible score

    Returns:
        Normalized score between 0.0 and 1.0
    """
    if max_score <= 0:
        return 0.0
    normalized = score / max_score
    return max(0.0, min(1.0, normalized))


# =============================================================================
# ENDPOINT HANDLERS (Rule #4: <60 lines each)
# =============================================================================

from fastapi import HTTPException
from ingestforge.core.pipeline.nexus_broadcast import NexusBroadcaster
from ingestforge.core.pipeline.nexus_fusion import NexusResultFusion
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.search import SearchQuery as UnifiedQuery
from pathlib import Path


@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest) -> SearchResponse:
    """
    Search the knowledge base with filters, sorting, and optional federation.

    Coordinates local hybrid retrieval with parallel remote broadcasting
    to a selected set of Nexus peers (Task 272).

    Args:
        request: SearchRequest containing query, filters, and nexus_ids list.

    Returns:
        SearchResponse with fused results from all contributing sources.
    """
    import time

    start_time = time.perf_counter()
    logger = _LazyDeps.get_logger()

    try:
        from ingestforge.core.pipeline.pipeline import Pipeline
        from ingestforge.retrieval import HybridRetriever

        pipeline = Pipeline()
        retriever = HybridRetriever(pipeline.config, pipeline.storage)

        # 1. Local Search
        local_results = retriever.search(
            query=request.query,
            top_k=request.top_k,
            library_filter=request.project_id,
            metadata_filter=request.filters,
            sort_by=request.sort_by,
        )

        # 2. Federated Search (Task 272 Integration)
        all_results = {"local": local_results}
        peer_failures = []

        if request.broadcast:
            registry = NexusRegistry(Path(".data/nexus"))
            broadcaster = NexusBroadcaster(registry, pipeline.config.nexus)

            # Map API request to Unified Search Query
            unified_query = UnifiedQuery(
                text=request.query,
                top_k=request.top_k,
                filters=request.filters or {},
                broadcast=True,
                target_peer_ids=request.nexus_ids,
            )

            nexus_resp = await broadcaster.broadcast(unified_query)
            peer_failures = nexus_resp.peer_failures

            # Group remote results by source for fusion
            for res in nexus_resp.results:
                if res.nexus_id not in all_results:
                    all_results[res.nexus_id] = []
                all_results[res.nexus_id].append(res)

        # 3. Result Fusion (Task 129)
        fusion = NexusResultFusion()
        fused_results = fusion.merge(all_results)

        # Map results to API format
        results = [_map_search_result_to_item(r) for r in fused_results]

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return SearchResponse(
            success=True,
            results=results,
            total_results=len(results),
            query=request.query,
            query_time_ms=elapsed_ms,
            message=f"Found {len(results)} results from {len(all_results)} sources",
        )

    except ValueError as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.warning(f"Search validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e

    except Exception as e:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {e}") from e
