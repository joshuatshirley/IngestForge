"""
Nexus Audit Logger Service.

Task 122: Secure, non-blocking distributed search ledger.
JPL Power of Ten: Rule #2 (Bounded buffers), Rule #4 (Small functions).
"""

import asyncio
import logging
import hashlib
import json
import aiofiles
from pathlib import Path
from typing import List
from ingestforge.core.models.nexus_audit import (
    NexusAuditEntry,
    AuditDirection,
    AuditAction,
)
from ingestforge.core.config.nexus import NexusConfig

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed bounds
MAX_BUFFER_SIZE = 100
LOG_FLUSH_INTERVAL = 2.0


class NexusAuditLogger:
    """
    Async logger for signed Nexus activity.
    """

    def __init__(self, config: NexusConfig, data_dir: Path):
        self.config = config
        self._log_path = data_dir / "nexus_audit.log"
        self._buffer: List[NexusAuditEntry] = []
        self._lock = asyncio.Lock()

        # Ensure log dir exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    async def log_event(
        self,
        direction: AuditDirection,
        nexus_id: str,
        query_hash: str,
        action: AuditAction,
        resource_id: str = "*",
    ) -> None:
        """
        Create and buffer a signed audit entry.
        Rule #4: Logic < 40 lines.
        """
        entry = NexusAuditEntry(
            direction=direction,
            nexus_id=nexus_id,
            query_hash=query_hash,
            action=action,
            resource_id=resource_id,
        )

        # Cryptographic Integrity
        entry.signature = self._sign_entry(entry)

        async with self._lock:
            self._buffer.append(entry)
            if len(self._buffer) >= MAX_BUFFER_SIZE:
                await self._flush()

    def _sign_entry(self, entry: NexusAuditEntry) -> str:
        """Generate SHA-256 signature for the log entry."""
        # In production, use private key HMAC. For Task 122 MVP, use SHA-256 hash.
        payload = f"{entry.timestamp}|{entry.direction}|{entry.nexus_id}|{entry.query_hash}|{entry.action}"
        return hashlib.sha256(payload.encode()).hexdigest()

    async def _flush(self) -> None:
        """Write buffer to disk. Rule #4."""
        if not self._buffer:
            return

        try:
            # Atomic append using aiofiles
            async with aiofiles.open(self._log_path, mode="a") as f:
                for entry in self._buffer:
                    await f.write(json.dumps(entry.model_dump(mode="json")) + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Failed to flush Nexus audit log: {e}")

    async def force_flush(self) -> None:
        """Manually trigger a flush (e.g. on shutdown)."""
        async with self._lock:
            await self._flush()
