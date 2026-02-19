"""Topic-based chunking strategy.

Groups text by topic boundaries rather than fixed size."""

from __future__ import annotations

from typing import List, Dict, Any, Optional


class TopicChunker:
    """Chunk text by topic boundaries."""

    def __init__(self, max_chunk_size: int = 1000, min_chunk_size: int = 100) -> None:
        """Initialize topic chunker.

        Args:
            max_chunk_size: Maximum chunk size
            min_chunk_size: Minimum chunk size
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Chunk text by topic boundaries.

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of topic-based chunks
        """
        # Split into paragraphs
        paragraphs = text.split("\n\n")

        # Group by topic
        topic_groups = self._group_by_topic(paragraphs)

        # Create chunks
        chunks = []
        for group in topic_groups:
            chunk_text = "\n\n".join(group)

            if len(chunk_text) >= self.min_chunk_size:
                chunks.append(
                    {
                        "text": chunk_text,
                        "length": len(chunk_text),
                        "chunking_strategy": "topic",
                        "metadata": metadata or {},
                    }
                )

        return chunks

    def _group_by_topic(self, paragraphs: List[str]) -> List[List[str]]:
        """Group paragraphs by topic."""
        if not paragraphs:
            return []

        groups = []
        current_group = [paragraphs[0]]
        current_topic_words = set(paragraphs[0].lower().split()[:20])

        for para in paragraphs[1:]:
            para_words = set(para.lower().split()[:20])

            # Calculate topic overlap
            overlap = len(current_topic_words & para_words)
            union = current_topic_words | para_words
            similarity = overlap / len(union) if union else 0

            # Group if similar topic
            if similarity > 0.3:
                current_group.append(para)
                current_topic_words.update(para_words)
            else:
                groups.append(current_group)
                current_group = [para]
                current_topic_words = para_words

        if current_group:
            groups.append(current_group)

        return groups


def chunk_text(
    text: str, max_size: int = 1000, min_size: int = 100
) -> List[Dict[str, Any]]:
    """Chunk text by topics."""
    chunker = TopicChunker(max_size, min_size)
    return chunker.chunk(text)
