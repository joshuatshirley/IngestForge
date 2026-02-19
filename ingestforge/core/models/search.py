"""
Unified Search Models for IngestForge.

Task 151: Define unified Search/Result Pydantic models.
Provides 100% type safety for local and federated (Nexus) retrieval.

JPL Power of Ten Compliance:
- Rule #2: Fixed bounds on filter count and query length.
- Rule #9: Complete type hints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
from ingestforge.core.models.nexus_error import PeerFailure

# JPL Rule #2: Fixed upper bounds
MAX_QUERY_LENGTH = 10000
MAX_FILTER_FIELDS = 20
MAX_TOP_K = 100
MAX_PEER_FAILURES = 50


class SearchQuery(BaseModel):
    """
    Unified model for incoming search requests.
    Supports local filtering and federated broadcast.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Search query string",
    )
    top_k: int = Field(
        default=10, ge=1, le=MAX_TOP_K, description="Number of results to return"
    )
    filters: Dict[str, Any] = Field(
        default_factory=dict, description="Metadata filters (e.g., domain, date)"
    )
    broadcast: bool = Field(
        default=False, description="Whether to query remote Nexus peers"
    )
    target_peer_ids: Optional[List[str]] = Field(
        default=None,
        description="Specific peer IDs to target. If None and broadcast is True, all peers are queried.",
    )
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )

    @field_validator("filters")
    @classmethod
    def validate_filters_count(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """JPL Rule #2: Enforce fixed bound on filter complexity."""
        if len(v) > MAX_FILTER_FIELDS:
            raise ValueError(f"Too many filters (max {MAX_FILTER_FIELDS})")
        return v

    @field_validator("text")
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query text cannot be empty")
        return v.strip()


class SearchResult(BaseModel):
    """
    Unified model for a single search result.
    Includes provenance for federated attribution.
    """

    content: str = Field(..., description="Text snippet of the match")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized relevance score")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Extraction confidence score"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Artifact metadata"
    )

    # Federated Attribution (EP-21 enabler)
    nexus_id: str = Field(
        default="local", description="ID of the originating Nexus instance"
    )
    nexus_name: str = Field(
        default="Local Library", description="Human-readable name of the source"
    )
    artifact_id: str = Field(..., description="Unique ID of the source artifact")
    document_id: str = Field(..., description="Parent document identifier")

    # Coordinate Mapping (integration)
    page: Optional[int] = Field(default=None, description="Page number if applicable")

    class Config:
        """Pydantic configuration."""

        populate_by_name = True


class SearchResponse(BaseModel):
    """
    Unified container for search result sets.
    """

    results: List[SearchResult] = Field(default_factory=list)
    total_hits: int = Field(default=0, ge=0)
    query_time_ms: float = Field(default=0.0)
    nexus_count: int = Field(
        default=1, description="Number of Nexuses that contributed results"
    )
    peer_failures: List[PeerFailure] = Field(
        default_factory=list, description="List of peers that failed to respond"
    )

    @field_validator("results")
    @classmethod
    def validate_results_count(cls, v: List[SearchResult]) -> List[SearchResult]:
        """JPL Rule #2: Enforce fixed bound on result set size."""
        if len(v) > MAX_TOP_K:
            return v[:MAX_TOP_K]
        return v

    @field_validator("peer_failures")
    @classmethod
    def validate_failures_count(cls, v: List[PeerFailure]) -> List[PeerFailure]:
        """JPL Rule #2: Enforce fixed bound on failure list."""
        if len(v) > MAX_PEER_FAILURES:
            return v[:MAX_PEER_FAILURES]
        return v
