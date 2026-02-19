"""
Nexus Peer Models for IngestForge.

Task 131: Peer registry models.
Rule #9: Complete type hints.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional, Dict
from pydantic import BaseModel, Field, HttpUrl


class NexusStatus(str, Enum):
    """Peer health status."""

    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    REVOKED = "REVOKED"
    PENDING = "PENDING"


class NexusPeer(BaseModel):
    """
    Metadata for a trusted remote IngestForge instance.
    """

    id: str = Field(..., description="Unique Nexus identifier (UUID)")
    name: str = Field(..., description="Human-readable name")
    url: HttpUrl = Field(..., description="Base URL of the remote API")

    # Security metadata
    api_key_hash: str = Field(..., description="SHA-256 hash of the peer API key")
    handshake_hash: Optional[str] = Field(
        None, description="Hash of the last JSON-LD handshake"
    )

    # Health tracking
    status: NexusStatus = Field(default=NexusStatus.PENDING)
    last_seen: Optional[datetime] = Field(None)
    failure_count: int = Field(default=0, ge=0)

    class Config:
        """Pydantic config."""

        use_enum_values = True


class NexusRegistryStore(BaseModel):
    """
    Root container for Nexus peer registry and global flags.
    """

    peers: Dict[str, NexusPeer] = Field(default_factory=dict)
    global_silence: bool = Field(
        default=False, description="Global isolation flag (Kill-Switch)"
    )

    class Config:
        """Pydantic config."""

        use_enum_values = True


# Rebuild models to resolve forward references
NexusRegistryStore.model_rebuild()
