"""Query Clarification & Refinement Router.

Implements query disambiguation endpoints to prevent low-quality RAG results.

Endpoints:
- POST /v1/query/clarify - Evaluate query clarity and return suggestions
- POST /v1/query/refine - Apply selected refinement to query

Follows NASA JPL Power of Ten:
- Rule #2: Fixed bounds on all data structures
- Rule #4: Functions under 60 lines
- Rule #5: Assertions at entry points
- Rule #7: Check model response; fail-fast if invalid
- Rule #9: Complete type hints
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from ingestforge.query.clarifier import (
    IFQueryClarifier,
    ClarifierConfig,
    ClarificationArtifact,
    CLARITY_THRESHOLD,
    MAX_QUERY_LENGTH,
)

# JPL Rule #2: Fixed upper bounds
MAX_CONTEXT_FIELDS = 10
MAX_REFINEMENT_LENGTH = 500

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/v1/query", tags=["query"])


# =============================================================================
# REQUEST/RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class ClarifyQueryRequest(BaseModel):
    """
    Request model for query clarification.

    Rule #9: Complete type hints for all fields.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="User query to evaluate for clarity",
    )
    threshold: float = Field(
        CLARITY_THRESHOLD,
        ge=0.0,
        le=1.0,
        description="Clarity threshold (0.0-1.0, higher = stricter)",
    )
    use_llm: bool = Field(
        False,
        description="Whether to use LLM for enhanced suggestions",
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional context for query evaluation",
    )

    @field_validator("context")
    @classmethod
    def validate_context(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Validate context dictionary size.

        Rule #2: Enforce fixed bounds on data structures.
        """
        if v is not None and len(v) > MAX_CONTEXT_FIELDS:
            raise ValueError(
                f"context cannot have more than {MAX_CONTEXT_FIELDS} fields"
            )
        return v


class ClarifyQueryResponse(BaseModel):
    """
    Response model for query clarification.

    Rule #9: Complete type hints for all fields.
    """

    original_query: str = Field(..., description="Original query text")
    clarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Clarity score (0.0-1.0)"
    )
    is_clear: bool = Field(..., description="Whether query meets clarity threshold")
    needs_clarification: bool = Field(
        ..., description="Whether user should refine query"
    )
    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested query refinements (max 5)",
    )
    reason: str = Field(..., description="Human-readable reason for score")
    factors: Dict[str, float] = Field(
        default_factory=dict,
        description="Clarity score factor breakdown",
    )
    evaluation_time_ms: float = Field(
        ..., description="Time taken to evaluate (milliseconds)"
    )


class RefineQueryRequest(BaseModel):
    """
    Request model for query refinement.

    Rule #9: Complete type hints for all fields.
    """

    original_query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Original user query",
    )
    selected_refinement: str = Field(
        ...,
        min_length=1,
        max_length=MAX_REFINEMENT_LENGTH,
        description="User-selected refinement suggestion",
    )


class RefineQueryResponse(BaseModel):
    """
    Response model for query refinement.

    Rule #9: Complete type hints for all fields.
    """

    refined_query: str = Field(..., description="Query with refinement applied")
    clarity_score: float = Field(
        ..., ge=0.0, le=1.0, description="Clarity score of refined query"
    )
    improvement: float = Field(
        ..., description="Clarity score improvement (refined - original)"
    )
    is_clear: bool = Field(..., description="Whether refined query is clear")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_llm_fn() -> Optional[Any]:
    """
    Get LLM function for enhanced clarification.

    Rule #4: Under 60 lines.
    Rule #7: Check imports; fail-fast if unavailable.

    Returns:
        LLM function if available, None otherwise.
    """
    try:
        from ingestforge.llm.factory import create_llm_client

        llm_client = create_llm_client()
        if llm_client is None:
            logger.warning("LLM client not available for clarification")
            return None

        def llm_fn(prompt: str) -> str:
            """Wrapper function for LLM calls."""
            try:
                response = llm_client.generate(prompt, max_tokens=200)
                return response.text if hasattr(response, "text") else str(response)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return ""

        return llm_fn

    except ImportError:
        logger.warning("LLM factory not available")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return None


def _artifact_to_response(
    artifact: ClarificationArtifact, elapsed_ms: float
) -> ClarifyQueryResponse:
    """
    Convert ClarificationArtifact to API response.

    Rule #4: Under 60 lines.
    Rule #9: Complete type hints.

    Args:
        artifact: ClarificationArtifact from clarifier.
        elapsed_ms: Time taken for evaluation.

    Returns:
        ClarifyQueryResponse for API.
    """
    return ClarifyQueryResponse(
        original_query=artifact.original_query,
        clarity_score=artifact.clarity_score.score,
        is_clear=artifact.clarity_score.is_clear,
        needs_clarification=artifact.needs_clarification,
        suggestions=artifact.suggestions,
        reason=artifact.reason,
        factors=artifact.clarity_score.factors,
        evaluation_time_ms=elapsed_ms,
    )


# =============================================================================
# API ENDPOINTS
# =============================================================================


@router.post(
    "/clarify",
    response_model=ClarifyQueryResponse,
    summary="Evaluate query clarity",
    description="Analyze query for ambiguity and return clarification suggestions",
    status_code=status.HTTP_200_OK,
)
async def clarify_query(request: ClarifyQueryRequest) -> ClarifyQueryResponse:
    """
    Evaluate query clarity and return suggestions.

    POST /v1/query/clarify endpoint.

    Rule #4: Under 60 lines.
    Rule #5: Validate inputs (Pydantic handles this).
    Rule #7: Check clarifier output; fail-fast if invalid.

    Args:
        request: ClarifyQueryRequest with query and options.

    Returns:
        ClarifyQueryResponse with clarity score and suggestions.

    Raises:
        HTTPException: If clarification fails.
    """
    start_time = time.time()

    try:
        # Create clarifier with config
        config = ClarifierConfig(
            threshold=request.threshold,
            use_llm=request.use_llm,
            max_suggestions=5,  # JPL Rule #2: Fixed bound
        )

        # Get LLM function if requested
        llm_fn = _get_llm_fn() if request.use_llm else None

        # Initialize clarifier
        clarifier = IFQueryClarifier(config, llm_fn)

        # Evaluate query clarity
        artifact = clarifier.evaluate(request.query)

        # JPL Rule #7: Validate output
        if artifact is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Clarification failed: No artifact returned",
            )

        # Calculate elapsed time
        elapsed_ms = (time.time() - start_time) * 1000

        # Convert to response
        response = _artifact_to_response(artifact, elapsed_ms)

        logger.info(
            f"Query clarified: score={response.clarity_score:.2f}, "
            f"needs_clarification={response.needs_clarification}, "
            f"time={elapsed_ms:.1f}ms"
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query clarification failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clarify query: {str(e)}",
        ) from e


@router.post(
    "/refine",
    response_model=RefineQueryResponse,
    summary="Refine query with selected suggestion",
    description="Apply user-selected refinement to original query",
    status_code=status.HTTP_200_OK,
)
async def refine_query(request: RefineQueryRequest) -> RefineQueryResponse:
    """
    Apply selected refinement to query.

    POST /v1/query/refine endpoint.

    Rule #4: Under 60 lines.
    Rule #5: Validate inputs (Pydantic handles this).

    Args:
        request: RefineQueryRequest with original query and refinement.

    Returns:
        RefineQueryResponse with refined query and new clarity score.

    Raises:
        HTTPException: If refinement fails.
    """
    try:
        # Combine original query with refinement
        # Format: "original query (refinement)"
        refined_query = f"{request.original_query} ({request.selected_refinement})"

        # Re-evaluate clarity of both queries
        clarifier = IFQueryClarifier()

        # Evaluate original
        original_artifact = clarifier.evaluate(request.original_query)
        original_score = original_artifact.clarity_score.score

        # Evaluate refined
        refined_artifact = clarifier.evaluate(refined_query)
        refined_score = refined_artifact.clarity_score.score

        # Calculate improvement
        improvement = refined_score - original_score

        logger.info(
            f"Query refined: original_score={original_score:.2f}, "
            f"refined_score={refined_score:.2f}, "
            f"improvement={improvement:.2f}"
        )

        return RefineQueryResponse(
            refined_query=refined_query,
            clarity_score=refined_score,
            improvement=improvement,
            is_clear=refined_artifact.clarity_score.is_clear,
        )

    except Exception as e:
        logger.error(f"Query refinement failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refine query: {str(e)}",
        ) from e


# =============================================================================
# HEALTH CHECK (Optional)
# =============================================================================


@router.get(
    "/clarifier/health",
    summary="Check clarifier health",
    description="Verify query clarifier is operational",
    status_code=status.HTTP_200_OK,
)
async def clarifier_health() -> Dict[str, Any]:
    """
    Check if query clarifier is operational.

    Rule #4: Under 60 lines.

    Returns:
        Health status dictionary.
    """
    try:
        # Test clarifier with simple query
        clarifier = IFQueryClarifier()
        test_artifact = clarifier.evaluate("test query")

        return {
            "status": "healthy",
            "clarifier_available": True,
            "llm_available": _get_llm_fn() is not None,
            "test_evaluation": {
                "query": "test query",
                "score": test_artifact.clarity_score.score,
            },
        }
    except Exception as e:
        logger.error(f"Clarifier health check failed: {e}")
        return {
            "status": "unhealthy",
            "clarifier_available": False,
            "error": str(e),
        }
