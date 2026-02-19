"""
Nexus Peer Registry Storage.

Task 131: Peer registry storage & persistence.
JPL Rule #4: Small scope, manageable logic.
"""

import json
from pathlib import Path
from typing import List, Optional
from ingestforge.core.models.nexus import NexusPeer, NexusStatus, NexusRegistryStore
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_REGISTRY_PEERS = 50


class NexusRegistry:
    """
    Persistent storage for registered Nexus peers and global network state.
    """

    def __init__(self, data_path: Path):
        self._path = data_path / "nexus_peers.json"
        self._store = self._load()

    def _load(self) -> NexusRegistryStore:
        """Load store from disk. Rule #2: Bounded loop."""
        if not self._path.exists():
            return NexusRegistryStore()
        try:
            with open(self._path, "r") as f:
                data = json.load(f) or {}
                # Handle legacy format where root was a dict of peers
                if "peers" not in data and isinstance(data, dict):
                    data = {"peers": data}

                # JPL Rule #2: Enforce fixed upper bound on peer count
                if "peers" in data:
                    raw_peers = data["peers"]
                    items = list(raw_peers.items())[:MAX_REGISTRY_PEERS]
                    data["peers"] = dict(items)

                return NexusRegistryStore(**data)
        except Exception as e:
            logger.error(f"Failed to load nexus registry: {e}")
            return NexusRegistryStore()

    def save(self) -> bool:
        """Persist state to disk. Rule #7: Check returns."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._store.model_dump(mode="json"), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save nexus registry: {e}")
            return False

    def is_silenced(self) -> bool:
        """Check if global silence is active."""
        return self._store.global_silence

    def set_silence(self, enabled: bool) -> bool:
        """Enable or disable global network silence."""
        self._store.global_silence = enabled
        return self.save()

    def add_peer(self, peer: NexusPeer) -> None:
        """Add or update a peer."""
        self._store.peers[peer.id] = peer
        self.save()

    def get_peer(self, peer_id: str) -> Optional[NexusPeer]:
        """Retrieve peer by ID."""
        return self._store.peers.get(peer_id)

    def list_active_peers(self) -> List[NexusPeer]:
        """List peers that are not revoked."""
        return [
            p for p in self._store.peers.values() if p.status != NexusStatus.REVOKED
        ]

    def update_status(self, peer_id: str, status: NexusStatus) -> None:
        """Update peer health status."""
        if peer_id in self._store.peers:
            self._store.peers[peer_id].status = status
            self.save()
