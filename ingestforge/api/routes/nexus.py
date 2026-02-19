"""
Nexus Federation API Router.

Task 127: Add /v1/nexus/handshake endpoint.
Provides semantic connectivity for Workspace Nexus.
"""

from fastapi import APIRouter, Request

from ingestforge.core.models.search import SearchQuery, SearchResponse
from ingestforge.retrieval import HybridRetriever
from ingestforge.core.pipeline.pipeline import Pipeline

router = APIRouter(prefix="/v1/nexus", tags=["nexus"])

# ... existing code ...


@router.post(
    "/search", response_model=SearchResponse, summary="Remote Federated Search"
)
async def nexus_remote_search(query: SearchQuery, request: Request):
    """
    Handle an incoming search request from a remote Nexus peer.
    Rule #4: Function logic strictly < 60 lines.
    """
    # Extract authorized state injected by NexusAuthMiddleware (Task 120/121)
    authorized_library = getattr(request.state, "authorized_library", "default")

    pipeline = Pipeline()
    retriever = HybridRetriever(pipeline.config, pipeline.storage)

    # Execute local search limited to the authorized library
    local_results = retriever.search(
        query=query.text,
        top_k=query.top_k,
        library_filter=authorized_library,
        metadata_filter=query.filters,
        min_confidence=query.min_confidence,
    )

    # Enrich results with local identity before returning
    for res in local_results:
        res.nexus_id = pipeline.config.nexus.nexus_id
        res.nexus_name = "Remote Nexus"  # Optional: Use config.name

    return SearchResponse(
        results=local_results, total_hits=len(local_results), nexus_count=1
    )
