"""
Nexus Authorization Middleware.

Task 120: Enforcement of library-level access control.
NASA JPL Rule #4: Enforcement logic < 60 lines.
"""

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from ingestforge.core.security.nexus_acl import NexusACLManager
from ingestforge.core.security.nexus_rbac import NexusRBAC
from ingestforge.core.security.nexus_audit import NexusAuditLogger
from ingestforge.core.security.nexus_blacklist import NexusBlacklist
from ingestforge.api.middleware.rate_limiter import create_rate_limiter
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.nexus_audit import AuditDirection, AuditAction
from ingestforge.core.config.nexus import NexusConfig
from ingestforge.core.system.nexus_metrics import track_unauthorized_attempt
from pathlib import Path
import hashlib


class NexusAuthMiddleware(BaseHTTPMiddleware):
    """
    Validates that incoming federated queries are authorized for the target resources.

    Task 125: Rate limiting per peer (100 req/min).
    """

    def __init__(self, app):
        super().__init__(app)
        data_dir = Path(".data/nexus")
        self.audit_logger = NexusAuditLogger(
            NexusConfig(
                nexus_id="local",
                cert_file=Path("cert.pem"),
                key_file=Path("key.pem"),
                trust_store=Path("trust"),
            ),
            data_dir,
        )
        self.blacklist = NexusBlacklist(data_dir)
        self.registry = NexusRegistry(data_dir)
        # Task 125: Rate limiter for authenticated peers
        self.peer_limiter = create_rate_limiter(requests_per_minute=100, burst_limit=10)

    async def dispatch(self, request: Request, call_next):
        # Only intercept Nexus-specific routes
        if not request.url.path.startswith("/v1/nexus"):
            return await call_next(request)

        # 1. Isolation & Security (Task 256, 123, 125)
        peer_id = request.headers.get("X-Nexus-ID")
        await self._check_isolation()
        await self._check_security(peer_id)

        library_id = request.headers.get("X-Target-Library")
        query_hash = hashlib.sha256(
            f"{request.method}{request.url}".encode()
        ).hexdigest()

        if not peer_id or not library_id:
            await self.audit_logger.log_event(
                AuditDirection.INBOUND, "unknown", query_hash, AuditAction.ERROR
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Missing Nexus-ID or Target-Library headers",
            )

        # 2. Access Control (Task 120, 121)
        role = await self._check_access(
            peer_id, library_id, request.url.path, request.method, query_hash
        )

        # Success - Log ALLOW & Inject state
        await self.audit_logger.log_event(
            AuditDirection.INBOUND, peer_id, query_hash, AuditAction.ALLOW, library_id
        )
        request.state.nexus_peer_id = peer_id
        request.state.authorized_library = library_id
        request.state.nexus_role = role

        return await call_next(request)

    async def _check_isolation(self) -> None:
        """Task 256: Global Silence Mode."""
        if self.registry.is_silenced():
            await self.audit_logger.log_event(
                AuditDirection.INBOUND, "ALL", "N/A", AuditAction.DENY, "GLOBAL_SILENCE"
            )
            track_unauthorized_attempt("ALL", "global_silence")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Federated services are isolated.",
            )

    async def _check_security(self, peer_id: str | None) -> None:
        """Task 123 (Kill-Switch) & Task 125 (Rate Limiting)."""
        if not peer_id:
            return

        if self.blacklist.is_revoked(peer_id):
            await self.audit_logger.log_event(
                AuditDirection.INBOUND, peer_id, "N/A", AuditAction.DENY, "BLACK_LISTED"
            )
            track_unauthorized_attempt(peer_id, "revoked")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Nexus Peer {peer_id} is REVOKED.",
            )

        limit_result = await self.peer_limiter.check(f"peer:{peer_id}")
        if not limit_result.allowed:
            await self.audit_logger.log_event(
                AuditDirection.INBOUND,
                peer_id,
                "N/A",
                AuditAction.DENY,
                "RATE_LIMIT_EXCEEDED",
            )
            track_unauthorized_attempt(peer_id, "rate_limit")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Nexus rate limit exceeded.",
                headers=limit_result.to_headers(),
            )

    async def _check_access(
        self, peer_id: str, library_id: str, path: str, method: str, query_hash: str
    ) -> str:
        """Task 120 (ACL) & Task 121 (RBAC)."""
        acl = NexusACLManager(Path(".data/nexus"))
        if not acl.is_authorized(peer_id, library_id):
            await self.audit_logger.log_event(
                AuditDirection.INBOUND,
                peer_id,
                query_hash,
                AuditAction.DENY,
                library_id,
            )
            track_unauthorized_attempt(peer_id, "acl_denied")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Peer {peer_id} not authorized for {library_id}",
            )

        role = acl.get_role(peer_id)
        action = self._map_route_to_action(path, method)
        if not NexusRBAC.is_action_allowed(role, action):
            await self.audit_logger.log_event(
                AuditDirection.INBOUND,
                peer_id,
                query_hash,
                AuditAction.DENY,
                f"{library_id}:{action}",
            )
            track_unauthorized_attempt(peer_id, "rbac_denied")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Peer {peer_id} lacking permission: {action}",
            )

        return role

    def _map_route_to_action(self, path: str, method: str) -> str:
        """Map API paths to RBAC actions. Rule #4."""
        if path.endswith("/search") or path.endswith("/handshake"):
            return "nexus:search"
        if "annotate" in path:
            return "nexus:annotate"
        return "nexus:unknown"
