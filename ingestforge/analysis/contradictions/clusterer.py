"""Fact Clustering for Contradiction Detection.

Groups semantically similar chunks across the corpus to identify potential conflicts.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from typing import List
import numpy as np

from ingestforge.storage.base import ChunkRecord
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class FactClusterer:
    """Logic for identifying related claims across multiple documents."""

    def __init__(self, similarity_threshold: float = 0.85):
        self.threshold = similarity_threshold

    def identify_clusters(self, chunks: List[ChunkRecord]) -> List[List[ChunkRecord]]:
        """Group chunks that speak about the same factual claim.

        Rule #1: Flat logic with early returns.
        Rule #2: Bounded cluster size.
        """
        if not chunks or len(chunks) < 2:
            return []

        clusters: List[List[ChunkRecord]] = []
        visited = set()
        MAX_CHUNKS = 500
        target_chunks = chunks[:MAX_CHUNKS]

        for i, chunk_a in enumerate(target_chunks):
            if chunk_a.chunk_id in visited:
                continue

            current_cluster = [chunk_a]
            visited.add(chunk_a.chunk_id)

            for j, chunk_b in enumerate(target_chunks[i + 1 :]):
                if chunk_b.chunk_id in visited:
                    continue

                if self._is_similar(chunk_a, chunk_b):
                    current_cluster.append(chunk_b)
                    visited.add(chunk_b.chunk_id)

            if len(current_cluster) > 1:
                clusters.append(current_cluster)

        logger.info(
            f"Identified {len(clusters)} factual clusters for contradiction check."
        )
        return clusters

    def _is_similar(self, a: ChunkRecord, b: ChunkRecord) -> bool:
        """Measure semantic similarity between two chunks."""
        if a.embedding is None or b.embedding is None:
            return False

        # Cosine similarity
        dot = np.dot(a.embedding, b.embedding)
        norm_a = np.linalg.norm(a.embedding)
        norm_b = np.linalg.norm(b.embedding)

        return (dot / (norm_a * norm_b)) >= self.threshold
