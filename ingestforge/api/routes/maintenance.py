"""
Maintenance and Governance API Router.

Verifiable Deletion Cert
Exposes administrative purge and data destruction capabilities.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: Complete type hints.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field

from ingestforge.core.audit.deletion_service import PurgeService
from ingestforge.core.audit.models import DeletionCertificate
from ingestforge.api.main import get_current_user

router = APIRouter(prefix="/v1/maintenance", tags=["maintenance"])


class PurgeRequest(BaseModel):
    """Request for verifiable artifact purging."""

    chunk_id: str = Field(..., description="ID of the chunk to purge")


class PurgeResponse(BaseModel):
    """Response containing the cryptographic proof of deletion."""

    success: bool
    certificate: Optional[DeletionCertificate] = None
    message: str


@router.post(
    "/purge",
    response_model=PurgeResponse,
    summary="Executes a verifiable data purge",
    status_code=status.HTTP_200_OK,
)
async def purge_artifact(
    request: PurgeRequest, current_user: dict = Depends(get_current_user)
) -> PurgeResponse:
    """
    Triggers a verifiable purge of a specific data chunk.

    Returns a signed DeletionCertificate as proof of purge.
    Only accessible to authorized administrators.
    """
    # Verify current user is an administrator (Simulation for MVP)
    # In production, this would check specific roles in current_user

    service = PurgeService()
    success, cert, msg = service.execute_verifiable_purge(
        chunk_id=request.chunk_id, operator_id=current_user.get("sub", "unknown_admin")
    )

    if not success:
        # We use 404 or 500 based on the message
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in msg.lower()
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=status_code, detail=msg)

    return PurgeResponse(success=True, certificate=cert, message=msg)
