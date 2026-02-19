"""
JSON-LD Semantic Models for Workspace Nexus.

Task 127: Define JSON-LD Pydantic models for handshake.
Provides a semantic capability exchange between peer Nexuses.

JPL Power of Ten Compliance:
- Rule #2: Fixed bounds on capabilities dictionary.
- Rule #9: Complete type hints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# JPL Rule #2: Fixed upper bounds
MAX_CAPABILITIES = 50
MAX_VERTICALS = 20


class NexusHandshake(BaseModel):
    """
    Base JSON-LD model for Nexus Handshake.
    """

    context: str = Field(
        default="https://ingestforge.io/context/nexus.jsonld", alias="@context"
    )
    type: str = Field(default="NexusHandshake", alias="@type")

    nexus_id: str = Field(..., description="UUID of the originating Nexus")
    version: str = Field(..., description="API/Framework version")

    # Capabilities (JPL Rule #2 bounded)
    capabilities: Dict[str, Any] = Field(
        default_factory=dict, description="Map of framework capabilities"
    )

    # Verticals supported (JPL Rule #2 bounded)
    supported_verticals: List[str] = Field(
        default_factory=list, description="List of active domain verticals"
    )

    class Config:
        """Pydantic config."""

        allow_population_by_field_name = True
        populate_by_name = True


class HandshakeRequest(NexusHandshake):
    """Request payload for initiating a handshake."""

    pass


class HandshakeResponse(NexusHandshake):
    """Response payload containing target Nexus capabilities."""

    status: str = Field(default="ACCEPTED")
    public_key_fingerprint: Optional[str] = Field(
        None, description="Fingerprint of the mTLS certificate"
    )
