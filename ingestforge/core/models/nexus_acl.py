"""
Nexus ACL Models for IngestForge.

Task 120: Define resource-level permissions for peer Nexuses.
JPL Power of Ten: Rule #2 (Bounds), Rule #9 (Type hints).
"""

from __future__ import annotations

from enum import Enum
from typing import List, Dict
from pydantic import BaseModel, Field


class NexusAccessScope(str, Enum):
    """Granularity of shared data."""

    METADATA = "METADATA"
    FULL_TEXT = "FULL_TEXT"


class NexusRole(str, Enum):
    """Fine-grained peer permissions."""

    READ_ONLY = "READ_ONLY"  # Search & view
    CONTRIBUTOR = "CONTRIBUTOR"  # Add annotations/links
    ADMIN = "ADMIN"  # Management actions


class NexusACLEntry(BaseModel):
    """
    Permission entry for a single Nexus peer.
    """

    peer_id: str = Field(..., description="UUID of the remote Nexus")
    allowed_libraries: List[str] = Field(
        default_factory=list, description="Whitelist of local library IDs"
    )
    scope: NexusAccessScope = Field(default=NexusAccessScope.METADATA)
    role: NexusRole = Field(default=NexusRole.READ_ONLY)

    class Config:
        """Pydantic config."""

        use_enum_values = True


class NexusACLStore(BaseModel):
    """
    Root container for all peer permissions.
    JPL Rule #2: Max 50 peer entries per instance.
    """

    entries: Dict[str, NexusACLEntry] = Field(default_factory=dict)
