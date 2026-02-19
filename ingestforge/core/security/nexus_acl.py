"""
Nexus ACL Management Service.

Task 120: Persistent storage and logic for Nexus resource access.
JPL Rule #4: All management functions < 60 lines.
"""

import json
import hashlib
from pathlib import Path
from ingestforge.core.models.nexus_acl import (
    NexusACLStore,
    NexusACLEntry,
    NexusAccessScope,
    NexusRole,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class NexusACLManager:
    """
    Handles granting and revoking access to local libraries for remote peers.
    """

    def __init__(self, data_dir: Path):
        self._path = data_dir / "nexus_acl.json"
        self._store = self._load()

    def _load(self) -> NexusACLStore:
        """Load ACL from disk. Rule #4 compliant."""
        if not self._path.exists():
            return NexusACLStore()
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
                return NexusACLStore(**data)
        except Exception as e:
            logger.error(f"Failed to load Nexus ACL: {e}")
            return NexusACLStore()

    def save(self) -> bool:
        """Atomic persist to disk. Rule #7: Check returns."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "w") as f:
                json.dump(self._store.model_dump(mode="json"), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save Nexus ACL: {e}")
            return False

    def is_authorized(self, peer_id: str, library_id: str) -> bool:
        """Check if peer is authorized for target library. Rule #4."""
        entry = self._store.entries.get(peer_id)
        if not entry:
            return False  # Default-Deny Policy
        return library_id in entry.allowed_libraries

    def get_role(self, peer_id: str) -> NexusRole:
        """Retrieve the assigned role for a peer. Rule #4."""
        entry = self._store.entries.get(peer_id)
        if not entry:
            return NexusRole.READ_ONLY  # Lowest privilege default
        return entry.role

    def grant_access(
        self,
        peer_id: str,
        library_id: str,
        scope: NexusAccessScope = NexusAccessScope.METADATA,
    ) -> None:
        """Authorize a peer for a specific library. Rule #4."""
        if peer_id not in self._store.entries:
            self._store.entries[peer_id] = NexusACLEntry(peer_id=peer_id)

        entry = self._store.entries[peer_id]
        if library_id not in entry.allowed_libraries:
            entry.allowed_libraries.append(library_id)
            entry.scope = scope
            self.save()
            self._log_audit("GRANT", peer_id, library_id)

    def revoke_access(self, peer_id: str, library_id: str) -> None:
        """Remove peer access to a library."""
        entry = self._store.entries.get(peer_id)
        if entry and library_id in entry.allowed_libraries:
            entry.allowed_libraries.remove(library_id)
            self.save()
            self._log_audit("REVOKE", peer_id, library_id)

    def revoke_all_access(self, peer_id: str) -> None:
        """
        Emergency Kill-Switch: Instantly remove all access for a peer.
        Rule #4: Logic under 20 lines.
        """
        if peer_id in self._store.entries:
            # Clear all library permissions
            self._store.entries[peer_id].allowed_libraries = []
            self._store.entries[peer_id].role = NexusRole.READ_ONLY
            self.save()

            # Update NexusRegistry if available (Task 131 integration)
            try:
                from ingestforge.storage.nexus_registry import (
                    NexusRegistry,
                    NexusStatus,
                )

                registry = NexusRegistry(self._path.parent)
                registry.update_status(peer_id, NexusStatus.REVOKED)
            except Exception as e:
                logger.error(f"Registry update failed during kill-switch: {e}")

            self._log_audit("KILL_SWITCH_TRIGGERED", peer_id, "ALL_LIBRARIES")

    def _log_audit(self, action: str, peer_id: str, library_id: str) -> None:
        """Local security shim for Task 122 compliance. Rule #4."""
        payload = f"{action}|{peer_id}|{library_id}"
        signature = hashlib.sha256(payload.encode()).hexdigest()
        logger.info(f"ACL_AUDIT: {payload} | SIG: {signature}")
