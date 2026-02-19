"""
Streaming Events for SSE (Server-Sent Events) - .

Defines event types and async generators for real-time pipeline streaming.

NASA JPL Power of Ten compliant.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Callable, Dict, List, Literal, Optional

from ingestforge.core.pipeline.artifacts import (
    IFArtifact,
    IFChunkArtifact,
    IFFailureArtifact,
)
from ingestforge.core.pipeline.backpressure import BackpressureBuffer

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_EVENT_SIZE = 65536  # 64KB max per event
MAX_KEEPALIVE_INTERVAL = 30  # seconds
MAX_IDLE_TIMEOUT = 300  # 5 minutes


@dataclass(frozen=True)
class StreamEvent:
    """
    SSE event for streaming pipeline execution.

    Streaming Foundry API.

    JPL Compliance:
    - Rule #6: Smallest scope (frozen dataclass)
    - Rule #9: 100% type hints
    """

    event_type: Literal["chunk", "progress", "error", "complete", "keepalive"]
    data: Dict[str, Any]
    timestamp: str
    event_id: Optional[str] = None

    def to_sse_format(self) -> str:
        """
        Convert event to SSE format.

        Returns:
            SSE-formatted string (event: TYPE\ndata: JSON\n\n)

        Rule #4: < 60 lines.
        Rule #5: Assert output format.
        """
        lines: List[str] = []

        if self.event_id:
            lines.append(f"id: {self.event_id}")

        if self.event_type != "chunk":
            lines.append(f"event: {self.event_type}")

        json_data = json.dumps(self.data, ensure_ascii=False)
        assert len(json_data) <= MAX_EVENT_SIZE, f"Event exceeds {MAX_EVENT_SIZE} bytes"

        lines.append(f"data: {json_data}")
        lines.append("")  # Empty line terminates event

        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ProgressMetadata:
    """
    Progress metadata for streaming events.

    JPL Compliance:
    - Rule #6: Smallest scope (frozen)
    - Rule #9: 100% type hints
    """

    current: int
    total: int
    stage: str
    percentage: float


def create_chunk_event(
    artifact: IFChunkArtifact,
    current: int,
    total: int,
    stage: str,
    is_final: bool = False,
) -> StreamEvent:
    """
    Create chunk event from artifact.

    Args:
        artifact: Chunk artifact to stream
        current: Current chunk number
        total: Total estimated chunks
        stage: Pipeline stage name
        is_final: Whether this is the final chunk

    Returns:
        StreamEvent for chunk

    Rule #4: < 60 lines.
    Rule #5: Assert preconditions.
    Rule #9: 100% type hints.
    """
    assert artifact is not None, "artifact cannot be None"
    assert current > 0, "current must be positive"
    assert total > 0, "total must be positive"
    assert stage, "stage cannot be empty"

    percentage = (current / total) * 100.0 if total > 0 else 0.0

    data = {
        "chunk_id": artifact.artifact_id,
        "content": artifact.text[:500],  # Preview only (500 chars)
        "citations": [],
        "progress": {
            "current": current,
            "total": total,
            "stage": stage,
            "percentage": round(percentage, 2),
        },
        "is_final": is_final,
        "metadata": {
            "start_char": artifact.start_char,
            "end_char": artifact.end_char,
        },
    }

    return StreamEvent(
        event_type="chunk",
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
        event_id=artifact.artifact_id,
    )


def create_progress_event(
    current: int,
    total: int,
    stage: str,
    message: str = "",
) -> StreamEvent:
    """
    Create progress event.

    Args:
        current: Current item number
        total: Total items
        stage: Pipeline stage name
        message: Optional progress message

    Returns:
        StreamEvent for progress

    Rule #4: < 60 lines.
    Rule #9: 100% type hints.
    """
    assert current >= 0, "current must be non-negative"
    assert total > 0, "total must be positive"

    percentage = (current / total) * 100.0 if total > 0 else 0.0

    data = {
        "current": current,
        "total": total,
        "stage": stage,
        "percentage": round(percentage, 2),
        "message": message,
    }

    return StreamEvent(
        event_type="progress",
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def create_error_event(
    error_code: str,
    message: str,
    suggestion: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> StreamEvent:
    """
    Create error event.

    Args:
        error_code: Error code (e.g., "PROC_001")
        message: Error message
        suggestion: Optional suggestion for user
        details: Optional additional details

    Returns:
        StreamEvent for error

    Rule #4: < 60 lines.
    Rule #9: 100% type hints.
    """
    assert error_code, "error_code cannot be empty"
    assert message, "message cannot be empty"

    data = {
        "error_code": error_code,
        "message": message,
        "suggestion": suggestion or "Check logs for details",
        "details": details or {},
    }

    return StreamEvent(
        event_type="error",
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def create_complete_event(
    total_chunks: int,
    success: bool = True,
    summary: str = "",
) -> StreamEvent:
    """
    Create completion event.

    Args:
        total_chunks: Total chunks processed
        success: Whether processing succeeded
        summary: Optional summary message

    Returns:
        StreamEvent for completion

    Rule #4: < 60 lines.
    Rule #9: 100% type hints.
    """
    assert total_chunks >= 0, "total_chunks must be non-negative"

    data = {
        "total_chunks": total_chunks,
        "success": success,
        "summary": summary or f"Processed {total_chunks} chunks successfully",
    }

    return StreamEvent(
        event_type="complete",
        data=data,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def create_keepalive_event() -> StreamEvent:
    """
    Create keepalive event.

    Returns:
        StreamEvent for keepalive

    Rule #4: < 60 lines.
    """
    return StreamEvent(
        event_type="keepalive",
        data={"message": "Connection alive"},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def stream_pipeline_execution(
    artifacts: List[IFArtifact],
    stage_name: str,
    buffer: Optional[BackpressureBuffer] = None,
    on_chunk: Optional[Callable[[IFChunkArtifact], None]] = None,
) -> AsyncGenerator[StreamEvent, None]:
    """
    Stream pipeline execution as SSE events.

    Core async generator for streaming.

    Args:
        artifacts: List of artifacts to stream
        stage_name: Current pipeline stage
        buffer: Optional backpressure buffer
        on_chunk: Optional callback for each chunk

    Yields:
        StreamEvent for each chunk and progress update

    Rule #2: Bounded by artifacts list length.
    Rule #4: < 60 lines.
    Rule #9: 100% type hints.
    """
    assert artifacts is not None, "artifacts cannot be None"
    assert stage_name, "stage_name cannot be empty"

    total = len(artifacts)
    current = 0

    try:
        for artifact in artifacts:
            current += 1

            # Handle chunk artifacts
            if isinstance(artifact, IFChunkArtifact):
                is_final = current == total
                event = create_chunk_event(
                    artifact=artifact,
                    current=current,
                    total=total,
                    stage=stage_name,
                    is_final=is_final,
                )

                if buffer:
                    await buffer.put(event)
                yield event

                if on_chunk:
                    on_chunk(artifact)

            # Handle failure artifacts
            elif isinstance(artifact, IFFailureArtifact):
                error_event = create_error_event(
                    error_code=artifact.error_code or "PROC_UNKNOWN",
                    message=artifact.error_msg or "Processing failed",
                    details={"artifact_id": artifact.artifact_id},
                )
                yield error_event
                break  # Stop on failure

        # Send completion event
        completion_event = create_complete_event(
            total_chunks=current,
            success=True,
        )
        yield completion_event

    except Exception as e:
        logger.error(f"Stream error: {e}")
        error_event = create_error_event(
            error_code="STREAM_ERROR",
            message=str(e),
            suggestion="Check server logs",
        )
        yield error_event
