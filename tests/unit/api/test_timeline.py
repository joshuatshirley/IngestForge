"""
Timeline API GWT Tests.

Evidence Timeline UI
Verifies event extraction, sorting, and API response formatting.
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone
from ingestforge.api.routes.timeline import _format_timeline_response
from ingestforge.analysis.timeline_builder import TimelineEntry

# =============================================================================
# UNIT TESTS (GWT)
# =============================================================================


def test_format_timeline_response_success():
    """
    GIVEN a list of TimelineEntry objects
    WHEN _format_timeline_response is called
    THEN it returns a correctly populated TimelineResponse.
    """
    # 1. Setup entries
    entry1 = TimelineEntry(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        event_type="LOGIN",
        description="User logged in",
        source="syslog",
    )
    entry2 = TimelineEntry(
        timestamp=datetime(2026, 1, 2, tzinfo=timezone.utc),
        event_type="FILE_ACCESS",
        description="Sensitive file read",
        source="audit",
    )

    # 2. Execute
    response = _format_timeline_response([entry1, entry2])

    # 3. Verify
    assert response.total_count == 2
    assert len(response.events) == 2
    assert response.events[0]["event_type"] == "LOGIN"
    assert response.start_time == "2026-01-01T00:00:00+00:00"
    assert response.end_time == "2026-01-02T00:00:00+00:00"


def test_format_timeline_response_empty():
    """
    GIVEN an empty list of entries
    WHEN _format_timeline_response is called
    THEN it returns a response with zero events.
    """
    response = _format_timeline_response([])

    assert response.total_count == 0
    assert response.events == []
    assert response.start_time is None


@pytest.mark.asyncio
async def test_get_chronological_timeline_integration():
    """
    GIVEN a document ID with existing chunks
    WHEN the timeline endpoint is called
    THEN it aggregates events from those chunks.
    """
    with (
        patch("ingestforge.api.routes.timeline.load_config"),
        patch(
            "ingestforge.api.routes.timeline.get_storage_backend"
        ) as MockStorageFactory,
        patch(
            "ingestforge.api.routes.timeline.get_current_user",
            return_value={"sub": "admin"},
        ),
    ):
        # Setup mocks
        mock_storage = MagicMock()
        mock_chunk = {
            "chunk_id": "c1",
            "metadata": {
                "timestamp": "2026-02-18T10:00:00Z",
                "event_type": "DETECTION",
                "source": "sensor_a",
            },
            "content": "Malware detected on host",
        }
        mock_storage.get_chunks_by_document.return_value = [mock_chunk]
        MockStorageFactory.return_value = mock_storage

        # Test the handler directly (skipping FastAPI routing overhead)
        from ingestforge.api.routes.timeline import get_chronological_timeline

        response = await get_chronological_timeline(
            document_id="doc_123", current_user={"sub": "admin"}
        )

        assert response.total_count == 1
        assert response.events[0]["event_type"] == "DETECTION"
        assert response.events[0]["source"] == "sensor_a"
