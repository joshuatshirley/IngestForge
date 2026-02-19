"""
Nexus Peer Health Auto-Scanner.

Task 124: Proactive monitoring of federated peer availability.
Task 274: Automated state transition (Recovery) & Latency verification.
JPL Power of Ten: Rule #2 (Fixed bounds), Rule #4 (Small functions).
"""

import asyncio
import httpx
import logging
from datetime import datetime, timezone
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.nexus import NexusPeer, NexusStatus

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed bounds
MAX_SCAN_CONCURRENCY = 5
MAX_FAILURE_THRESHOLD = 10
SCAN_TIMEOUT_SEC = 3.0


class NexusScanner:
    """
    Background service that pings remote peers to maintain network health.
    """

    def __init__(self, registry: NexusRegistry):
        self.registry = registry
        self._semaphore = asyncio.Semaphore(MAX_SCAN_CONCURRENCY)

    async def scan_all(self) -> None:
        """
        Scan all active peers and update their status.
        Rule #2: Fixed upper bound (max 50 peers).
        Rule #4: Logic under 20 lines.
        """
        peers = self.registry.list_active_peers()
        # Rule #2: Bound loop to prevent resource exhaustion
        tasks = [self.check_peer(peer) for peer in peers[:50]]
        await asyncio.gather(*tasks)

    async def check_peer(self, peer: NexusPeer) -> None:
        """
        Ping a single peer and update its health state.
        Rule #4: Logic under 40 lines.
        """
        health_url = f"{peer.url}v1/nexus/health"

        async with self._semaphore:
            try:
                async with httpx.AsyncClient(timeout=SCAN_TIMEOUT_SEC) as client:
                    response = await client.get(health_url)
                    if response.status_code == 200:
                        self._handle_success(peer)
                    else:
                        self._handle_failure(peer, f"HTTP {response.status_code}")
            except Exception as e:
                self._handle_failure(peer, str(e))

    def _handle_success(self, peer: NexusPeer) -> None:
        """Update peer state on successful ping."""
        peer.status = NexusStatus.ONLINE
        peer.last_seen = datetime.now(timezone.utc)
        peer.failure_count = 0
        self.registry.save()
        logger.debug(f"Peer {peer.name} ({peer.id}) is ONLINE.")

    def _handle_failure(self, peer: NexusPeer, reason: str) -> None:
        """Update peer state on failed ping and enforce auto-deactivation."""
        peer.failure_count += 1
        logger.warning(
            f"Health check failed for {peer.name}: {reason} (Count: {peer.failure_count})"
        )

        if peer.failure_count >= MAX_FAILURE_THRESHOLD:
            peer.status = NexusStatus.OFFLINE
            logger.error(
                f"Peer {peer.name} marked as OFFLINE after {MAX_FAILURE_THRESHOLD} failures."
            )

        self.registry.save()
