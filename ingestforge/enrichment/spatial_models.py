"""
Spatial Linkage Models.

Spatial Linkage Engine
Defines the structure for linking visual evidence to graph nodes.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from typing import Dict, Any
from pydantic import BaseModel, Field


class SpatialRegion(BaseModel):
    """Bounding box region in an image."""

    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)
    image_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SpatialLink(BaseModel):
    """Link between a Knowledge Graph node and a visual region."""

    node_id: str
    region: SpatialRegion
    link_type: str = "visual_evidence"
    metadata: Dict[str, Any] = Field(default_factory=dict)
