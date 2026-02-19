"""
Nexus Health Monitoring Service.

Task 131: Background health checks for peer Nexuses.
JPL Rule #2: Bounded loops (max 10 peers).
"""

import asyncio
import httpx
import logging
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.nexus import NexusStatus

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CONCURRENT_PINGS = 10
PING_TIMEOUT_SECONDS = 5.0


class NexusHealthMonitor:
    """
    Background service to monitor peer availability.
    """

    def __init__(self, registry: NexusRegistry):
        self.registry = registry
        self._running = False

    async def start(self, interval_seconds: int = 60):
        """Start the monitoring loop."""
        self._running = True
        while self._running:
            await self.check_all_peers()
            await asyncio.sleep(interval_seconds)

    async def stop(self):
        """Stop the monitoring loop."""
        self._running = False

    async def check_all_peers(self) -> None:
        """Ping all active peers. Rule #2: Bounded loop."""
        peers = self.registry.list_active_peers()

        # JPL Rule #2: Slice to ensure we don't overwhelm network
        for i in range(0, len(peers), MAX_CONCURRENT_PINGS):
            batch = peers[i : i + MAX_CONCURRENT_PINGS]
            tasks = [self._ping_peer(p.id, str(p.url)) for p in batch]
            await asyncio.gather(*tasks)

    async def _ping_peer(self, peer_id: str, url: str) -> None:
        """Single peer ping logic. Rule #7: Check returns."""
        try:
            async with httpx.AsyncClient(timeout=PING_TIMEOUT_SECONDS) as client:
                response = await client.get(f"{url}/v1/health")

                if response.status_code == 200:
                    self.registry.update_status(peer_id, NexusStatus.ONLINE)
                    logger.debug(f"Nexus {peer_id} is ONLINE")
                else:
                    self._handle_failure(peer_id)
        except Exception as e:
            logger.warning(f"Failed to ping Nexus {peer_id}: {e}")
            self._handle_failure(peer_id)

    def _handle_failure(self, peer_id: str) -> None:
        """
        Logic for marking peers offline.
        Task 125: Circuit breaker opens after 3 consecutive failures.
        """
        peer = self.registry.get_peer(peer_id)
        if not peer:
            return

        peer.failure_count += 1
        if peer.failure_count >= 3:
            self.registry.update_status(peer_id, NexusStatus.OFFLINE)
            logger.info(
                f"Nexus {peer_id} marked as OFFLINE after 3 failures (Task 125)"
            )
        else:
            self.registry.save()
