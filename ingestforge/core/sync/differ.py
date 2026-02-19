"""Sync Differ Engine.

Identifies discrepancies between local and remote document storage.
Follows NASA JPL Rule #1 (Simple Flow) and Rule #4 (Modular).
"""

from __future__ import annotations
from typing import List
from ingestforge.storage.base import ChunkRepository, ChunkRecord
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class SyncDiffer:
    """Logic for calculating synchronization deltas."""

    def calculate_push_delta(
        self, local_repo: ChunkRepository, remote_repo: ChunkRepository
    ) -> List[ChunkRecord]:
        """Determine which local chunks are missing from the remote repository.

        Rule #1: Flat logic with early returns.
        Rule #2: Fixed upper bounds would be applied during bulk transfer.
        """
        # 1. Get all IDs from remote (fast check)
        # Note: Implementation depends on storage backend having a fast list_ids method
        # for MVP we use get_all_chunks metadata
        try:
            # Simple ID-based difference
            local_chunks = local_repo.get_all_chunks()

            # Logic: We need a way to check remote presence efficiently
            # For MVP, we fetch all remote IDs if small, or use verify_chunk_exists
            delta = []
            for chunk in local_chunks:
                if not remote_repo.verify_chunk_exists(chunk.chunk_id):
                    delta.append(chunk)

            logger.info(f"Sync Delta Calculated: {len(delta)} chunks pending upload.")
            return delta

        except Exception as e:
            logger.error(f"Failed to calculate sync delta: {e}")
            return []

    def get_sync_stats(
        self, local_repo: ChunkRepository, remote_repo: ChunkRepository
    ) -> dict:
        """Get high-level counts for local vs remote."""
        local_count = local_repo.count()
        remote_count = remote_repo.count()
        return {
            "local": local_count,
            "remote": remote_count,
            "out_of_sync": local_count - remote_count,
        }
