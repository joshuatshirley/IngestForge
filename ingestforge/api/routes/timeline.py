"""
Timeline API Router.

Evidence Timeline UI
Exposes chronological fact visualization and event correlation.

JPL Compliance:
- Rule #2: Bounded loops (MAX_TIMELINE_CHUNKS, MAX_TIMELINE_EVENTS).
- Rule #4: All functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel

from ingestforge.analysis.timeline_builder import TimelineBuilder, TimelineEntry
from ingestforge.api.main import get_current_user
from ingestforge.storage.factory import get_storage_backend
from ingestforge.core.config_loaders import load_config

# JPL Rule #2: Fixed upper bounds for safety-critical processing
MAX_TIMELINE_CHUNKS = 1000
MAX_TIMELINE_EVENTS = 1000

router = APIRouter(prefix="/v1/analysis", tags=["analysis"])


class TimelineResponse(BaseModel):
    """API response for chronological timeline."""

    events: List[Dict[str, Any]]
    total_count: int
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@router.get("/timeline", response_model=TimelineResponse)
async def get_chronological_timeline(
    library: str = Query("default", description="Library to query"),
    document_id: Optional[str] = Query(None, description="Filter by specific document"),
    current_user: dict = Depends(get_current_user),
) -> TimelineResponse:
    """
    Retrieves a chronological timeline of extracted facts and events.

    Fact visualization endpoint.
    """
    try:
        config = load_config()
        storage = get_storage_backend(config)

        # 1. Fetch chunks from storage (Rule #2: Bound input)
        if document_id:
            chunks = storage.get_chunks_by_document(document_id)
        else:
            chunks = storage.search("", k=MAX_TIMELINE_CHUNKS)

        if not chunks:
            return TimelineResponse(events=[], total_count=0)

        # 2. Build timeline (Rule #2: Bound processing loop)
        builder = TimelineBuilder()
        for chunk in chunks[:MAX_TIMELINE_CHUNKS]:
            builder.add_chunk(chunk if isinstance(chunk, dict) else chunk.to_dict())

        entries = builder.build()

        return _format_timeline_response(entries)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build timeline: {str(e)}",
        )


def _format_timeline_response(entries: List[TimelineEntry]) -> TimelineResponse:
    """Helper to format builder entries into API response. Rule #2: Bounded iteration."""
    if not entries:
        return TimelineResponse(events=[], total_count=0)

    # JPL Rule #2: Bound formatting loop
    bounded_entries = entries[:MAX_TIMELINE_EVENTS]
    formatted_events = [entry.to_dict() for entry in bounded_entries]

    return TimelineResponse(
        events=formatted_events,
        total_count=len(formatted_events),
        start_time=bounded_entries[0].timestamp.isoformat(),
        end_time=bounded_entries[-1].timestamp.isoformat(),
    )
