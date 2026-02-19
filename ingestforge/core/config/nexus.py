"""
Nexus Configuration Models.

Task 119: Trust store configuration for mTLS.
Rule #9: Complete type hints.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class NexusConfig(BaseModel):
    """
    Configuration for Workspace Nexus federation.
    """

    enabled: bool = Field(default=False)
    nexus_id: str = Field(..., description="Unique ID for this instance")

    # mTLS Security Paths
    cert_file: Path = Field(..., description="Path to local Nexus certificate (PEM)")
    key_file: Path = Field(..., description="Path to local Nexus private key")
    trust_store: Path = Field(
        ..., description="Path to trusted peer certificates directory"
    )

    # Timeouts & Limits (JPL Rule #2)
    query_timeout_ms: int = Field(default=2000, ge=100, le=10000)
    max_peers: int = Field(default=10, ge=1, le=50)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
