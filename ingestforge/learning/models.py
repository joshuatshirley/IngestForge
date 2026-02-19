"""
Few-Shot Learning Models.

Few-Shot Registry
Defines the structure for human-verified extraction examples.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime


class FewShotExample(BaseModel):
    """A human-verified extraction example for few-shot prompting."""

    id: str = Field(..., description="Unique identifier for the example")
    input_text: str = Field(..., description="The raw input text to extract from")
    output_json: Dict[str, Any] = Field(
        ..., description="The expected extracted data in JSON format"
    )
    domain: str = Field(
        default="general", description="The vertical domain (e.g. legal, cyber)"
    )
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    verified_by: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class ExampleQuery(BaseModel):
    """Query parameters for retrieving examples."""

    domain: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
