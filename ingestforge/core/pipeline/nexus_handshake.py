"""
JSON-LD Handshake Logic for Workspace Nexus.

Task 127: Implement capability exchange and verification.
JPL Rule #4: Parser and validation logic < 60 lines.
"""

import json
from typing import Tuple
from ingestforge.core.models.nexus_ld import HandshakeRequest, HandshakeResponse
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.logging import get_logger
from ingestforge.core.system.nexus_metrics import (
    track_handshake_failure,
    get_handshake_timer,
)

logger = get_logger(__name__)


class HandshakeManager:
    """
    Manages the semantic capability exchange between Nexus nodes.
    """

    def __init__(self, registry: NexusRegistry):
        self.registry = registry

    async def process_request(
        self, request: HandshakeRequest, client_cert_cn: str
    ) -> Tuple[HandshakeResponse, bool]:
        """
        Validate an incoming handshake request.
        Rule #4: Logic under 60 lines.
        """
        with get_handshake_timer():
            # Security Rule: CN must match payload nexus_id
            if request.nexus_id != client_cert_cn:
                logger.warning(
                    f"Nexus ID mismatch: payload={request.nexus_id}, cert={client_cert_cn}"
                )
                track_handshake_failure("identity_mismatch")
                return self._deny_handshake(request, "Identity mismatch"), False

            # Capability Match Logic
            compatible = self._check_compatibility(request)

            if not compatible:
                track_handshake_failure("incompatible_version")
                return self._deny_handshake(request, "Incompatible version"), False

            # Update Registry
            self._update_peer_registry(request)

            response = HandshakeResponse(
                nexus_id="local-nexus",
                version="1.0.0",
                supported_verticals=["legal", "technical"],
                status="ACCEPTED",
            )

            return response, True

    def _check_compatibility(self, request: HandshakeRequest) -> bool:
        """Verify version and mandatory capabilities."""
        # Simple version match for MVP
        return request.version.startswith("1.")

    def _update_peer_registry(self, request: HandshakeRequest) -> None:
        """Persistence of handshake state. Rule #4 compliant."""
        peer = self.registry.get_peer(request.nexus_id)
        if peer:
            peer.handshake_hash = str(hash(json.dumps(request.dict())))
            peer.status = "ONLINE"
            self.registry.save()

    def _deny_handshake(
        self, request: HandshakeRequest, reason: str
    ) -> HandshakeResponse:
        """Create a rejection response."""
        return HandshakeResponse(
            nexus_id="local-nexus", version="1.0.0", status=f"REJECTED: {reason}"
        )
