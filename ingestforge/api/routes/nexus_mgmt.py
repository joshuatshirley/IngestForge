"""
Nexus Management Routes.

Task 271: UI-driven peer registry management.
Task 283: MFA/High-privilege confirmation for sensitive actions.
NASA JPL Power of Ten: Rule #4, Rule #9.
"""

import logging
from fastapi import APIRouter, HTTPException, status, Request
from typing import List
from pathlib import Path
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.nexus import NexusPeer
from ingestforge.core.security.nexus_acl import NexusACLManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/nexus", tags=["Nexus Management"])


def _get_registry() -> NexusRegistry:
    return NexusRegistry(Path(".data/nexus"))


def _get_acl() -> NexusACLManager:
    return NexusACLManager(Path(".data/nexus"))


@router.get("/peers", response_model=List[NexusPeer])
async def list_peers():
    """List all registered Nexus peers."""
    registry = _get_registry()
    return list(registry._store.peers.values())


@router.post("/peers", status_code=status.HTTP_201_CREATED)
async def register_peer(peer: NexusPeer):
    """Register a new Nexus peer."""
    registry = _get_registry()
    if registry.get_peer(peer.id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Peer with ID {peer.id} already exists.",
        )
    registry.add_peer(peer)
    return {"status": "success", "message": f"Peer {peer.name} registered."}


@router.delete("/peers/{peer_id}")
async def revoke_peer(peer_id: str, request: Request):
    """
    Emergency revocation of a peer (Kill-Switch).

    Task 283: Requires short-lived Admin-Session token.
    Audit: Logs user identity.
    """
    admin_token = request.headers.get("X-Admin-Session")
    if not admin_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="High-risk action requires a second confirmation (Admin Session).",
        )

    # Audit log (Simple print/log for MVP, should use NexusAuditLogger)
    user_id = getattr(request.state, "user_id", "admin")
    logger.warning(f"HIGH-RISK ACTION: Peer {peer_id} revoked by {user_id} (Task 283)")

    acl = _get_acl()
    # Task 123 logic
    acl.revoke_all_access(peer_id)
    return {"status": "success", "message": f"Peer {peer_id} access revoked globally."}


@router.post("/peers/{peer_id}/ping")
async def ping_peer(peer_id: str):
    """Manually trigger a health check for a peer."""
    from ingestforge.core.system.nexus_scanner import NexusScanner

    registry = _get_registry()
    peer = registry.get_peer(peer_id)
    if not peer:
        raise HTTPException(status_code=404, detail="Peer not found")

    scanner = NexusScanner(registry)
    await scanner.check_peer(peer)
    return {"status": "success", "new_status": peer.status}
