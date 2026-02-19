"""
Legal Vertical API Router.

Legal Pleading Template
Exposes legal synthesis capabilities.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: Complete type hints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ingestforge.verticals.legal.models import LegalPleadingModel
from ingestforge.verticals.legal.generator import LegalPleadingGenerator
from ingestforge.verticals.legal.aggregator import LegalFactAggregator

router = APIRouter(prefix="/v1/legal", tags=["legal"])


class GeneratePleadingRequest(BaseModel):
    """API request for pleading generation."""

    metadata: LegalPleadingModel
    generate_argument: bool = True
    context_query: Optional[str] = None
    context_chunks: List[str] = []


class GeneratePleadingResponse(BaseModel):
    """API response for pleading generation."""

    markdown: str
    success: bool = True


@router.post("/generate-pleading", response_model=GeneratePleadingResponse)
async def generate_pleading(
    request: GeneratePleadingRequest,
) -> GeneratePleadingResponse:
    """
    Generate a formatted legal pleading.

    Backend API entry point.
    """
    try:
        generator = LegalPleadingGenerator()
        aggregator = LegalFactAggregator()
        model = request.metadata

        # If no facts provided, auto-aggregate from Knowledge Graph
        if not model.statement_of_facts and request.context_query:
            model.statement_of_facts = aggregator.aggregate_evidence(
                request.context_query
            )

        # Optionally synthesize argument if context is provided
        if (
            request.generate_argument
            and request.context_query
            and request.context_chunks
        ):
            arg = generator.synthesize_argument(
                request.context_query, request.context_chunks
            )
            model.legal_argument = arg

        md = generator.generate_markdown(model)

        return GeneratePleadingResponse(markdown=md)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
