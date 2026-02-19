"""
Discovery API Routes for Proactive Scout.

Proactive Scout
Provides recommended discovery intents based on knowledge gap analysis.

NASA JPL Power of Ten compliant.
"""

import logging
from typing import List, Any, Dict
from fastapi import APIRouter, Query, HTTPException, status
from pydantic import BaseModel, Field

from ingestforge.core.pipeline.knowledge_manifest import IFKnowledgeManifest
from ingestforge.core.pipeline.artifacts import IFDiscoveryIntentArtifact
from ingestforge.agent.proactive_scout import run_scout_analysis

logger = logging.getLogger(__name__)

# =============================================================================
# ROUTER SETUP
# =============================================================================

router = APIRouter(
    prefix="/v1/discovery",
    tags=["discovery"],
)

# =============================================================================
# CONSTANTS (JPL Rule #2: Fixed upper bounds)
# =============================================================================

MAX_INTENTS_PER_REQUEST = 100  # Maximum intents to return per request
DEFAULT_INTENT_LIMIT = 20  # Default number of intents to return
MIN_CONFIDENCE_THRESHOLD = 0.0
MAX_CONFIDENCE_THRESHOLD = 1.0

# =============================================================================
# SCHEMAS (JPL Rule #9: Complete type hints)
# =============================================================================


class DiscoveryIntent(BaseModel):
    """
    Discovery intent recommendation.

    Proactive Scout
    Represents a suggested discovery task.
    """

    intent_id: str = Field(..., description="Unique intent ID")
    target_entity: str = Field(..., description="Entity that needs exploration")
    entity_type: str = Field(..., description="Type of the target entity")
    missing_link_type: str = Field(..., description="Type of missing connection")
    rationale: str = Field(..., description="Why this discovery is recommended")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Recommendation confidence"
    )
    priority_score: float = Field(..., ge=0.0, le=1.0, description="Priority ranking")
    current_references: int = Field(0, ge=0, description="Current reference count")
    suggested_searches: List[str] = Field(
        default_factory=list, description="Recommended search terms"
    )


class DiscoveryResponse(BaseModel):
    """
    Response containing discovery recommendations.

    Proactive Scout
    """

    total_intents: int = Field(..., ge=0, description="Total intents found")
    returned_intents: int = Field(..., ge=0, description="Number of intents returned")
    intents: List[DiscoveryIntent] = Field(
        ..., description="Discovery intent recommendations"
    )
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Filters applied to results"
    )


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get(
    "/recommended",
    response_model=DiscoveryResponse,
    summary="Get recommended discoveries",
    description="Get recommended discovery tasks based on knowledge gap analysis",
    status_code=status.HTTP_200_OK,
)
async def get_recommended_discoveries(
    limit: int = Query(DEFAULT_INTENT_LIMIT, ge=1, le=MAX_INTENTS_PER_REQUEST),
    min_confidence: float = Query(
        MIN_CONFIDENCE_THRESHOLD,
        ge=MIN_CONFIDENCE_THRESHOLD,
        le=MAX_CONFIDENCE_THRESHOLD,
    ),
) -> DiscoveryResponse:
    """
    Get recommended discoveries from proactive scout analysis.

    AC: Provides GET endpoint for discovery recommendations.

    Rule #4: Under 60 lines.
    Rule #7: Check return values.

    Args:
        limit: Maximum number of intents to return.
        min_confidence: Minimum confidence threshold for intents.

    Returns:
        DiscoveryResponse with recommended intents.

    Raises:
        HTTPException: If manifest is not active or analysis fails.
    """
    try:
        # Get knowledge manifest instance
        manifest = IFKnowledgeManifest()

        # JPL Rule #7: Check if manifest is active
        if not manifest.is_active:
            logger.warning("Knowledge manifest is not active")
            return DiscoveryResponse(
                total_intents=0,
                returned_intents=0,
                intents=[],
                filters_applied={"limit": limit, "min_confidence": min_confidence},
            )

        # Run scout analysis
        all_intents = run_scout_analysis(manifest)

        # Filter by confidence
        filtered_intents = _filter_by_confidence(all_intents, min_confidence)

        # Apply limit
        limited_intents = _apply_limit(filtered_intents, limit)

        # Convert to response schema
        intent_responses = _convert_to_schema(limited_intents)

        logger.info(
            f"Returned {len(intent_responses)} discovery intents "
            f"(filtered from {len(all_intents)} total)"
        )

        return DiscoveryResponse(
            total_intents=len(all_intents),
            returned_intents=len(intent_responses),
            intents=intent_responses,
            filters_applied={"limit": limit, "min_confidence": min_confidence},
        )

    except Exception as e:
        logger.error(f"Failed to get discovery recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve discovery recommendations: {str(e)}",
        ) from e


# =============================================================================
# HELPER FUNCTIONS (JPL Rule #4: <60 lines each)
# =============================================================================


def _filter_by_confidence(
    intents: List[IFDiscoveryIntentArtifact], min_confidence: float
) -> List[IFDiscoveryIntentArtifact]:
    """
    Filter intents by confidence threshold.

    Rule #2: Bounded iteration.
    Rule #4: Under 60 lines.

    Args:
        intents: List of discovery intent artifacts.
        min_confidence: Minimum confidence threshold.

    Returns:
        Filtered list of intents.
    """
    filtered: List[IFDiscoveryIntentArtifact] = []

    # JPL Rule #2: Bounded iteration
    intent_count = min(len(intents), MAX_INTENTS_PER_REQUEST)

    for i in range(intent_count):
        intent = intents[i]
        if intent.confidence >= min_confidence:
            filtered.append(intent)

    return filtered


def _apply_limit(
    intents: List[IFDiscoveryIntentArtifact], limit: int
) -> List[IFDiscoveryIntentArtifact]:
    """
    Apply limit to intents list.

    Rule #4: Under 60 lines.

    Args:
        intents: List of discovery intent artifacts.
        limit: Maximum number to return.

    Returns:
        Limited list of intents.
    """
    # Already sorted by priority in scout
    return intents[:limit]


def _convert_to_schema(
    intents: List[IFDiscoveryIntentArtifact],
) -> List[DiscoveryIntent]:
    """
    Convert artifact objects to response schema.

    Rule #2: Bounded iteration.
    Rule #4: Under 60 lines.

    Args:
        intents: List of discovery intent artifacts.

    Returns:
        List of DiscoveryIntent schema objects.
    """
    results: List[DiscoveryIntent] = []

    # JPL Rule #2: Bounded iteration
    intent_count = min(len(intents), MAX_INTENTS_PER_REQUEST)

    for i in range(intent_count):
        intent = intents[i]

        results.append(
            DiscoveryIntent(
                intent_id=intent.artifact_id,
                target_entity=intent.target_entity,
                entity_type=intent.entity_type,
                missing_link_type=intent.missing_link_type,
                rationale=intent.rationale,
                confidence=intent.confidence,
                priority_score=intent.priority_score,
                current_references=intent.current_reference_count,
                suggested_searches=intent.suggested_search_terms,
            )
        )

    return results
