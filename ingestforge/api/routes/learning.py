"""
Learning API Router.

Few-Shot Registry
Exposes verified extraction management.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: Complete type hints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from ingestforge.learning.models import FewShotExample
from ingestforge.learning.registry import FewShotRegistry
from ingestforge.learning.matcher import SemanticExampleMatcher

router = APIRouter(prefix="/v1/learning", tags=["learning"])


class ExamplesResponse(BaseModel):
    """Response model for examples listing."""

    examples: List[FewShotExample]
    count: int


class MatchRequest(BaseModel):
    """Request for semantic matching."""

    text: str
    domain: Optional[str] = None
    limit: int = 3


@router.post("/match", response_model=List[FewShotExample])
async def match_examples(request: MatchRequest) -> List[FewShotExample]:
    """
    Find semantically similar verified examples for few-shot prompting.

    API entry point for example matching.
    """
    matcher = SemanticExampleMatcher()
    matches = matcher.find_matches(
        input_text=request.text, domain=request.domain, limit=request.limit
    )
    return matches


@router.post("/examples", status_code=status.HTTP_201_CREATED)
async def add_verified_example(example: FewShotExample) -> dict:
    """
    Store a new verified extraction pair.

    API endpoint for registry insertion.
    """
    registry = FewShotRegistry()
    success = registry.add_example(example)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store example in registry",
        )

    return {"status": "success", "id": example.id}


@router.get("/examples", response_model=ExamplesResponse)
async def list_verified_examples(
    domain: Optional[str] = None, limit: int = 10
) -> ExamplesResponse:
    """
    List verified examples from the registry.
    """
    registry = FewShotRegistry()
    examples = registry.list_examples(domain=domain, limit=limit)

    return ExamplesResponse(examples=examples, count=len(examples))


@router.delete("/examples/{example_id}")
async def delete_verified_example(example_id: str) -> dict:
    """
    Remove an example from the registry.
    """
    registry = FewShotRegistry()
    removed = registry.remove_example(example_id)

    if not removed:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Example with ID {example_id} not found",
        )

    return {"status": "removed"}
