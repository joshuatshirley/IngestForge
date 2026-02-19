"""
Human-in-the-loop web curation system.

Provides a "Search, Review, and Ingest" workflow for curated web content ingestion.
Users search for topics, review results one-by-one, and decide what to ingest.

Example:
    from ingestforge.curation import CurationEngine, CurationSessionManager

    # Create a new session
    manager = CurationSessionManager()
    engine = CurationEngine(manager)

    # Start curation
    session = engine.start_session("machine learning transformers", max_results=20)

    # Interactive loop
    while session.state not in ("complete", "idle"):
        preview = engine.preview_current()
        # Show to user, get decision
        if user_wants_ingest:
            engine.ingest_current()
        else:
            engine.skip_current()
"""

from ingestforge.curation.models import (
    CurationState,
    CurationDecision,
    CurationItem,
    CurationSession,
    CurationSessionManager,
)
from ingestforge.curation.engine import CurationEngine, PreviewResult, IngestResult

__all__ = [
    "CurationState",
    "CurationDecision",
    "CurationItem",
    "CurationSession",
    "CurationSessionManager",
    "CurationEngine",
    "PreviewResult",
    "IngestResult",
]
