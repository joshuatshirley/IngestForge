"""
Pipeline processing result.

Provides the PipelineResult dataclass for tracking pipeline execution outcomes.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(slots=True)
class PipelineResult:
    """Result of pipeline processing.

    Uses __slots__ (via slots=True) to reduce memory overhead when many
    instances are created during batch processing.

    Attributes:
        document_id: Unique identifier for the processed document
        source_file: Path to the original source file
        success: Whether processing completed successfully
        chunks_created: Number of chunks generated from the document
        chunks_indexed: Number of chunks successfully indexed to storage
        chunk_ids: List of chunk IDs that were created
        error_message: Error description if success is False
        processing_time_sec: Total processing time in seconds
    """

    document_id: str
    source_file: str
    success: bool
    chunks_created: int
    chunks_indexed: int
    chunk_ids: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    processing_time_sec: float = 0.0
