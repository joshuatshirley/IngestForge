"""
Chunk size optimization.

Split overly large chunks and merge small chunks.
"""

from dataclasses import dataclass
from typing import Any, List, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config


@dataclass
class OptimizationReport:
    """Report of optimization operations."""

    original_count: int
    final_count: int
    chunks_split: int
    chunks_merged: int
    avg_size_before: float
    avg_size_after: float


class SizeOptimizer:
    """
    Optimize chunk sizes by splitting and merging.

    Ensures all chunks fall within configured size bounds.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.min_size = config.chunking.min_size
        self.max_size = config.chunking.max_size
        self.target_size = config.chunking.target_size

    def optimize(
        self,
        chunks: List[ChunkRecord],
    ) -> tuple[List[ChunkRecord], OptimizationReport]:
        """
        Optimize chunk sizes.

        Args:
            chunks: List of chunks to optimize

        Returns:
            Tuple of (optimized chunks, report)
        """
        if not chunks:
            return [], OptimizationReport(0, 0, 0, 0, 0.0, 0.0)

        original_count = len(chunks)
        avg_size_before = sum(c.word_count for c in chunks) / len(chunks)

        # Split large chunks
        split_chunks, splits = self._split_large_chunks(chunks)

        # Merge small chunks
        merged_chunks, merges = self._merge_small_chunks(split_chunks)

        # Reindex
        for i, chunk in enumerate(merged_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(merged_chunks)

        avg_size_after = (
            sum(c.word_count for c in merged_chunks) / len(merged_chunks)
            if merged_chunks
            else 0.0
        )

        report = OptimizationReport(
            original_count=original_count,
            final_count=len(merged_chunks),
            chunks_split=splits,
            chunks_merged=merges,
            avg_size_before=avg_size_before,
            avg_size_after=avg_size_after,
        )

        return merged_chunks, report

    def _split_large_chunks(
        self,
        chunks: List[ChunkRecord],
    ) -> tuple[List[ChunkRecord], int]:
        """Split chunks that exceed max size."""
        result: list[Any] = []
        split_count = 0

        for chunk in chunks:
            if chunk.word_count <= self.max_size:
                result.append(chunk)
            else:
                # Split into smaller chunks
                sub_chunks = self._split_chunk(chunk)
                result.extend(sub_chunks)
                split_count += 1

        return result, split_count

    def _split_chunk(self, chunk: ChunkRecord) -> List[ChunkRecord]:
        """Split a single large chunk."""
        content = chunk.content
        paragraphs = content.split("\n\n")

        sub_chunks = []
        current_content = []
        current_size = 0

        for para in paragraphs:
            para_words = len(para.split())

            if (
                current_size + para_words > self.target_size
                and current_size >= self.min_size
            ):
                # Create sub-chunk
                sub_content = "\n\n".join(current_content)
                sub_chunk = ChunkRecord(
                    chunk_id=f"{chunk.chunk_id}_p{len(sub_chunks)}",
                    document_id=chunk.document_id,
                    content=sub_content,
                    chunk_type=chunk.chunk_type,
                    section_title=chunk.section_title,
                    section_hierarchy=chunk.section_hierarchy,
                    source_file=chunk.source_file,
                    word_count=len(sub_content.split()),
                    char_count=len(sub_content),
                )
                sub_chunks.append(sub_chunk)
                current_content = []
                current_size = 0

            current_content.append(para)
            current_size += para_words

        # Last sub-chunk
        if current_content:
            sub_content = "\n\n".join(current_content)
            sub_chunk = ChunkRecord(
                chunk_id=f"{chunk.chunk_id}_p{len(sub_chunks)}",
                document_id=chunk.document_id,
                content=sub_content,
                chunk_type=chunk.chunk_type,
                section_title=chunk.section_title,
                section_hierarchy=chunk.section_hierarchy,
                source_file=chunk.source_file,
                word_count=len(sub_content.split()),
                char_count=len(sub_content),
            )
            sub_chunks.append(sub_chunk)

        return sub_chunks if sub_chunks else [chunk]

    def _merge_small_chunks(
        self,
        chunks: List[ChunkRecord],
    ) -> tuple[List[ChunkRecord], int]:
        """
        Merge chunks that are below min size.

        Rule #1: Reduced nesting with helper methods
        """
        if not chunks:
            return [], 0

        result: list[Any] = []
        merge_count = 0
        pending: Optional[ChunkRecord] = None

        for chunk in chunks:
            pending, added_count = self._process_chunk_for_merge(chunk, pending, result)
            merge_count += added_count
        final_merge_count = self._handle_final_pending(pending, result)
        merge_count += final_merge_count

        return result, merge_count

    def _process_chunk_for_merge(
        self,
        chunk: ChunkRecord,
        pending: Optional[ChunkRecord],
        result: List[ChunkRecord],
    ) -> tuple[Optional[ChunkRecord], int]:
        """
        Process a single chunk for merging.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines

        Returns:
            (new_pending, merge_count)
        """
        if pending is None:
            if chunk.word_count < self.min_size:
                return chunk, 0  # Becomes new pending
            result.append(chunk)
            return None, 0
        combined_size = pending.word_count + chunk.word_count

        if combined_size <= self.max_size:
            merged = self._merge_chunks(pending, chunk)
            if merged.word_count >= self.min_size:
                result.append(merged)
                return None, 1  # Merged and added
            return merged, 1  # Merged but still pending

        # Can't merge - add pending and handle current chunk
        result.append(pending)
        if chunk.word_count < self.min_size:
            return chunk, 0  # Current becomes new pending
        result.append(chunk)
        return None, 0

    def _handle_final_pending(
        self, pending: Optional[ChunkRecord], result: List[ChunkRecord]
    ) -> int:
        """
        Handle final pending chunk.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines

        Returns:
            merge_count (0 or 1)
        """
        if pending is None:
            return 0
        if not result or pending.word_count >= self.min_size:
            result.append(pending)
            return 0

        # Try to merge with last result
        last = result[-1]
        if last.word_count + pending.word_count <= self.max_size:
            result[-1] = self._merge_chunks(last, pending)
            return 1

        result.append(pending)
        return 0

    def _merge_chunks(
        self,
        chunk1: ChunkRecord,
        chunk2: ChunkRecord,
    ) -> ChunkRecord:
        """Merge two chunks into one."""
        combined_content = chunk1.content + "\n\n" + chunk2.content

        return ChunkRecord(
            chunk_id=f"{chunk1.chunk_id}_m{chunk2.chunk_index}",
            document_id=chunk1.document_id,
            content=combined_content,
            chunk_type=chunk1.chunk_type,  # Keep first chunk's type
            section_title=chunk1.section_title or chunk2.section_title,
            section_hierarchy=chunk1.section_hierarchy or chunk2.section_hierarchy,
            source_file=chunk1.source_file,
            page_start=chunk1.page_start,
            page_end=chunk2.page_end,
            word_count=len(combined_content.split()),
            char_count=len(combined_content),
        )


# Alias for backwards compatibility
ChunkSizeOptimizer = SizeOptimizer
