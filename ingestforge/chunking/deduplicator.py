"""
Chunk deduplication.

Remove near-duplicate chunks using similarity hashing.
"""

import hashlib
import re
from dataclasses import dataclass
from typing import Dict, List, Set

from ingestforge.chunking.semantic_chunker import ChunkRecord


@dataclass
class DeduplicationReport:
    """Report of deduplication operations."""

    original_count: int
    final_count: int
    duplicates_removed: int
    duplicate_groups: int


class Deduplicator:
    """
    Remove near-duplicate chunks.

    Uses MinHash-like approach for efficient similarity detection.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        min_shingle_size: int = 3,
    ):
        """
        Initialize deduplicator.

        Args:
            similarity_threshold: Minimum similarity to consider duplicate (0-1)
            min_shingle_size: Minimum word n-gram size for hashing
        """
        self.similarity_threshold = similarity_threshold
        self.min_shingle_size = min_shingle_size

    def deduplicate(
        self,
        chunks: List[ChunkRecord],
    ) -> tuple[List[ChunkRecord], DeduplicationReport]:
        """
        Remove near-duplicate chunks.

        Rule #4: Reduced from 62 → 32 lines

        Args:
            chunks: List of chunks to deduplicate

        Returns:
            Tuple of (deduplicated chunks, report)
        """
        if not chunks:
            return [], DeduplicationReport(0, 0, 0, 0)

        original_count = len(chunks)
        chunk_hashes = {
            c.chunk_id: self._compute_content_hash(c.content) for c in chunks
        }
        duplicate_groups = self._find_duplicate_groups(chunks, chunk_hashes)
        keep_ids, removed_count = self._select_chunks_to_keep(chunks, duplicate_groups)

        # Filter and reindex
        result = [c for c in chunks if c.chunk_id in keep_ids]
        self._reindex_chunks(result)

        report = DeduplicationReport(
            original_count=original_count,
            final_count=len(result),
            duplicates_removed=removed_count,
            duplicate_groups=len([g for g in duplicate_groups if len(g) > 1]),
        )

        return result, report

    def _select_chunks_to_keep(
        self, chunks: List[ChunkRecord], duplicate_groups: List[List[ChunkRecord]]
    ) -> tuple[Set[str], int]:
        """
        Select which chunks to keep from duplicate groups.

        Rule #4: Extracted selection logic (<60 lines)

        Returns:
            Tuple of (keep_ids, removed_count)
        """
        keep_ids: Set[str] = set()
        removed_count = 0

        # Keep best chunk from each group
        for group in duplicate_groups:
            if len(group) == 1:
                keep_ids.add(group[0].chunk_id)
            else:
                best = max(group, key=lambda c: c.word_count)
                keep_ids.add(best.chunk_id)
                removed_count += len(group) - 1

        # Add non-grouped chunks
        grouped_ids = {c.chunk_id for group in duplicate_groups for c in group}
        for chunk in chunks:
            if chunk.chunk_id not in grouped_ids:
                keep_ids.add(chunk.chunk_id)

        return keep_ids, removed_count

    def _reindex_chunks(self, chunks: List[ChunkRecord]) -> None:
        """
        Reindex chunks after deduplication.

        Rule #4: Extracted reindexing (<60 lines)
        """
        total = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total

    def _compute_content_hash(self, content: str) -> str:
        """Compute normalized content hash."""
        # Normalize content
        normalized = self._normalize_text(content)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _compute_shingles(self, content: str) -> Set[str]:
        """Compute word n-gram shingles."""
        normalized = self._normalize_text(content)
        words = normalized.split()

        if len(words) < self.min_shingle_size:
            return {normalized}

        shingles = set()
        for i in range(len(words) - self.min_shingle_size + 1):
            shingle = " ".join(words[i : i + self.min_shingle_size])
            shingles.add(shingle)

        return shingles

    def _compute_jaccard_similarity(
        self,
        shingles1: Set[str],
        shingles2: Set[str],
    ) -> float:
        """Compute Jaccard similarity between shingle sets."""
        if not shingles1 or not shingles2:
            return 0.0

        intersection = len(shingles1 & shingles2)
        union = len(shingles1 | shingles2)

        return intersection / union if union > 0 else 0.0

    def _find_duplicate_groups(
        self,
        chunks: List[ChunkRecord],
        chunk_hashes: Dict[str, str],
    ) -> List[List[ChunkRecord]]:
        """
        Find groups of similar chunks.

        Rule #4: Reduced from 64 → 24 lines
        """
        groups, unique_chunks = self._group_by_exact_hash(chunks, chunk_hashes)
        if len(unique_chunks) <= 1:
            return groups

        # Find near-duplicates among unique chunks
        chunk_shingles = {
            c.chunk_id: self._compute_shingles(c.content) for c in unique_chunks
        }
        near_duplicate_groups = self._find_similar_pairs(unique_chunks, chunk_shingles)
        groups.extend(near_duplicate_groups)

        return groups

    def _group_by_exact_hash(
        self, chunks: List[ChunkRecord], chunk_hashes: Dict[str, str]
    ) -> tuple[List[List[ChunkRecord]], List[ChunkRecord]]:
        """
        Group chunks by exact content hash.

        Rule #4: Extracted exact hash grouping (<60 lines)

        Returns:
            Tuple of (exact duplicate groups, unique chunks)
        """
        hash_groups: Dict[str, List[ChunkRecord]] = {}
        for chunk in chunks:
            h = chunk_hashes[chunk.chunk_id]
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(chunk)

        # Exact duplicates
        groups = [
            chunk_list for chunk_list in hash_groups.values() if len(chunk_list) > 1
        ]

        # Unique chunks (need near-duplicate check)
        unique = [c for c in chunks if len(hash_groups[chunk_hashes[c.chunk_id]]) == 1]

        return groups, unique

    def _find_similar_pairs(
        self, unique_chunks: List[ChunkRecord], chunk_shingles: Dict[str, Set[str]]
    ) -> List[List[ChunkRecord]]:
        """
        Find near-duplicate pairs using shingle similarity.

        Rule #4: Extracted similarity checking (<60 lines)
        """
        groups = []
        processed = set()

        for i, chunk1 in enumerate(unique_chunks):
            if chunk1.chunk_id in processed:
                continue

            similar_group = [chunk1]

            for chunk2 in unique_chunks[i + 1 :]:
                if chunk2.chunk_id in processed:
                    continue

                similarity = self._compute_jaccard_similarity(
                    chunk_shingles[chunk1.chunk_id],
                    chunk_shingles[chunk2.chunk_id],
                )

                if similarity >= self.similarity_threshold:
                    similar_group.append(chunk2)
                    processed.add(chunk2.chunk_id)

            if len(similar_group) > 1:
                groups.append(similar_group)
                processed.add(chunk1.chunk_id)

        return groups


# Alias for backwards compatibility
ChunkDeduplicator = Deduplicator
