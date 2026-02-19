"""
Nexus Peer Failure Models.

Task 130: Categorize failures from remote Nexus peers.
NASA JPL Rule #9: Complete type hints.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class PeerErrorType(str, Enum):
    """Categories of federated network failures."""

    TIMEOUT = "TIMEOUT"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    SERVER_ERROR = "SERVER_ERROR"
    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    UNKNOWN = "UNKNOWN"


class PeerFailure(BaseModel):
    """
    Data container for a failed peer query attempt.
    """

    nexus_id: str = Field(..., description="ID of the peer that failed")
    error_type: PeerErrorType = Field(default=PeerErrorType.UNKNOWN)
    status_code: Optional[int] = Field(
        None, description="HTTP status code if applicable"
    )
    message: str = Field(default="An error occurred during broadcast")

    class Config:
        """Pydantic config."""

        use_enum_values = True
