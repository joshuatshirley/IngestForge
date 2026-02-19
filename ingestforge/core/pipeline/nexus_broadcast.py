"""
Nexus Search Broadcaster.

Task 128: Asynchronous broadcast engine for cross-instance search.
NASA JPL Power of Ten: Rule #2 (Bounds), Rule #4 (Small functions).
"""

import asyncio
import httpx
import logging
from typing import List, Any, Union
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.models.search import (
    SearchQuery,
    SearchResult,
    SearchResponse,
    MAX_TOP_K,
)
from ingestforge.core.models.nexus_error import PeerFailure, PeerErrorType
from ingestforge.core.config.nexus import NexusConfig

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CONCURRENT_REQUESTS = 10
# Task 125: Hard timeout of 2000ms for outbound federated queries.
DEFAULT_TIMEOUT_SEC = 2.0


class NexusBroadcaster:
    """
    Distributes local search queries to verified remote Nexus peers.
    """

    def __init__(self, registry: NexusRegistry, config: NexusConfig):
        self.registry = registry
        self.config = config

    async def broadcast(self, query: SearchQuery) -> SearchResponse:
        """
        Distribute a search query to a filtered set of remote Nexus peers.

        Supports surgical target selection via query.target_peer_ids (Task 272).
        Enforces Global Silence isolation (Task 256).

        Args:
            query: The unified search query containing text and target filters.

        Returns:
            SearchResponse containing merged results and any peer failures.
        """
        assert query.text, "Query text must not be empty (Rule #5)"
        if not query.broadcast:
            return SearchResponse()

        # 1. Global Silence Mode (Task 256)
        if self.registry.is_silenced():
            logger.warning("Nexus Broadcaster silenced by global flag.")
            return SearchResponse(
                results=[],
                peer_failures=[
                    PeerFailure(
                        nexus_id="local",
                        error_type=PeerErrorType.AUTHENTICATION_FAILED,
                        message="Outbound broadcast disabled by global silence mode.",
                    )
                ],
            )

        active_peers = self.registry.list_active_peers()
        if not active_peers:
            return SearchResponse()

        # 2. Filter by Target Peer IDs (Task 272)
        if query.target_peer_ids is not None:
            target_peers = [p for p in active_peers if p.id in query.target_peer_ids]
        else:
            target_peers = active_peers[: self.config.max_peers]

        if not target_peers:
            return SearchResponse()

        tasks = [
            self._query_peer(peer.id, str(peer.url), query) for peer in target_peers
        ]

        # Gather with exceptions=True to capture task-level crashes
        broadcast_outputs = await asyncio.gather(*tasks, return_exceptions=True)

        return self._aggregate_broadcast(broadcast_outputs)

    def _build_search_url(self, base_url: str) -> str:
        """Sanitize and construct the full search endpoint URL."""
        return f"{base_url.rstrip('/')}/v1/nexus/search"

    async def _query_peer(
        self, peer_id: str, url: str, query: SearchQuery
    ) -> Union[List[SearchResult], PeerFailure]:
        """Single peer query logic. Rule #4 & #7 compliant."""
        try:
            target_url = self._build_search_url(url)
            async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SEC) as client:
                response = await client.post(
                    target_url,
                    json=query.dict(),
                    headers={"X-Nexus-ID": self.config.nexus_id},
                )

                if response.status_code == 200:
                    search_resp = SearchResponse(**response.json())
                    return search_resp.results

                return self._map_http_error(peer_id, response.status_code)
        except httpx.TimeoutException:
            return PeerFailure(
                nexus_id=peer_id,
                error_type=PeerErrorType.TIMEOUT,
                message="Peer timed out",
            )
        except Exception as e:
            logger.error(f"Broadcast failure to {peer_id}: {e}")
            return PeerFailure(
                nexus_id=peer_id,
                error_type=PeerErrorType.CONNECTION_REFUSED,
                message=str(e),
            )

    def _map_http_error(self, peer_id: str, status_code: int) -> PeerFailure:
        """Map HTTP codes to PeerErrorType. Rule #4."""
        err_type = PeerErrorType.SERVER_ERROR
        if status_code == 403:
            err_type = PeerErrorType.AUTHENTICATION_FAILED

        return PeerFailure(
            nexus_id=peer_id,
            error_type=err_type,
            status_code=status_code,
            message=f"Peer returned HTTP {status_code}",
        )

    def _aggregate_broadcast(self, outputs: List[Any]) -> SearchResponse:
        """Aggregate results and failures into a single response. Rule #4."""
        final_results: List[SearchResult] = []
        failures: List[PeerFailure] = []

        for item in outputs:
            if isinstance(item, list):
                # JPL Rule #2: Bound enrichment loop to prevent CPU exhaustion
                for res in item[:MAX_TOP_K]:
                    peer = self.registry.get_peer(res.nexus_id)
                    if peer:
                        res.nexus_name = peer.name
                final_results.extend(item[:MAX_TOP_K])
            elif isinstance(item, PeerFailure):
                failures.append(item)
            elif isinstance(item, Exception):
                # Handle task-level crash exceptions
                failures.append(
                    PeerFailure(nexus_id="unknown", message="Internal task crash")
                )

        return SearchResponse(
            results=final_results,
            peer_failures=failures,
            total_hits=len(final_results),
            nexus_count=len(outputs) + 1,  # +1 for local
        )
