"""
Nexus Audit Log Models.

Task 122: Cryptographically signed security ledger.
JPL Power of Ten: Rule #9 (Type hints).
"""

from __future__ import annotations

from enum import Enum
from datetime import datetime
from pydantic import BaseModel, Field


class AuditDirection(str, Enum):
    """Flow of the federated query."""

    INBOUND = "IN"
    OUTBOUND = "OUT"


class AuditAction(str, Enum):
    """Result of the access control check."""

    ALLOW = "ALLOW"
    DENY = "DENY"
    ERROR = "ERROR"


class NexusAuditEntry(BaseModel):
    """
    Immutable record of a federated interaction.
    """

    timestamp: datetime = Field(default_factory=datetime.utcnow)
    direction: AuditDirection = Field(..., description="IN from peer or OUT to peer")
    nexus_id: str = Field(..., description="Peer Nexus ID")
    query_hash: str = Field(..., description="SHA-256 hash of the search payload")
    action: AuditAction = Field(..., description="Authorization decision")
    resource_id: str = Field(default="*", description="Target library or artifact ID")
    signature: str = Field(default="", description="Integrity signature of this entry")

    class Config:
        """Pydantic config."""

        use_enum_values = True
