"""
Nexus Blacklist Manager.

Task 123: High-performance revocation cache for emergency kill-switch.
JPL Power of Ten: Rule #2 (Fixed bounds), Rule #9 (Type hints).
"""

import logging
from typing import Set
from pathlib import Path

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed bounds
MAX_BLACKLIST_SIZE = 1000


class NexusBlacklist:
    """
    In-memory cache of revoked Peer IDs for sub-millisecond middleware checks.
    """

    def __init__(self, data_dir: Path):
        self._path = data_dir / "nexus_blacklist.json"
        self._revoked_ids: Set[str] = set()
        self._load()

    def is_revoked(self, peer_id: str) -> bool:
        """Check if peer is in the blacklist."""
        return peer_id in self._revoked_ids

    def add(self, peer_id: str) -> None:
        """Add peer to blacklist and persist."""
        if len(self._revoked_ids) < MAX_BLACKLIST_SIZE:
            self._revoked_ids.add(peer_id)
            self._save()
            logger.warning(f"Peer {peer_id} added to global blacklist.")

    def remove(self, peer_id: str) -> None:
        """Remove peer from blacklist (Manual restoration)."""
        if peer_id in self._revoked_ids:
            self._revoked_ids.remove(peer_id)
            self._save()

    def _load(self) -> None:
        """Load blacklist from disk. Rule #2: Bound input."""
        import json

        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                    # JPL Rule #2: Slice input to fixed bound
                    raw_list = data.get("revoked_ids", [])
                    self._revoked_ids = set(raw_list[:MAX_BLACKLIST_SIZE])
            except Exception as e:
                logger.error(f"Failed to load blacklist: {e}")

    def _save(self) -> None:
        """Persist blacklist to disk."""
        import json

        try:
            with open(self._path, "w") as f:
                json.dump({"revoked_ids": list(self._revoked_ids)}, f)
        except Exception as e:
            logger.error(f"Failed to save blacklist: {e}")
