"""
Layout-Aware Chunker for Section-Respecting Text Splitting.

Chunks text while respecting section boundaries detected by
ChapterDetector, ensuring chunks never split mid-section."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.ingest.refinement import ChapterMarker
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_CHUNK_SIZE = 10000  # Characters
MAX_SECTIONS = 500
MIN_CHUNK_SIZE = 50


@dataclass
class LayoutSection:
    """A section of text bounded by structural markers.

    Attributes:
        title: Section title (from ChapterMarker)
        content: Section text content
        level: Hierarchy level (1=chapter, 2=section, etc.)
        start_pos: Start position in document
        end_pos: End position in document
    """

    title: str
    content: str
    level: int
    start_pos: int
    end_pos: int

    @property
    def word_count(self) -> int:
        """Get word count."""
        return len(self.content.split())

    @property
    def char_count(self) -> int:
        """Get character count."""
        return len(self.content)


class LayoutChunker:
    """Chunker that respects document section boundaries.

    This chunker uses ChapterMarker boundaries from the refinement
    stage to ensure chunks never split mid-section. Features:

    1. **by_title**: Split at title boundaries (level 1 markers)
    2. **respect_sections**: Keep sections intact when possible
    3. **combine_small**: Merge undersized sections

    Usage:
        >>> from ingestforge.ingest.refinement import TextRefinementPipeline
        >>> from ingestforge.ingest.refiners import ChapterDetector
        >>>
        >>> pipeline = TextRefinementPipeline([ChapterDetector()])
        >>> refined = pipeline.refine(text)
        >>>
        >>> chunker = LayoutChunker()
        >>> chunks = chunker.chunk_with_markers(
        ...     text, refined.chapter_markers, document_id="doc_123"
        ... )
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 100,
        combine_text_under_n_chars: int = 200,
        respect_section_boundaries: bool = True,
        chunk_by_title: bool = False,
    ) -> None:
        """Initialize layout chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            min_chunk_size: Minimum chunk size in characters
            combine_text_under_n_chars: Combine sections shorter than this
            respect_section_boundaries: Never split within sections
            chunk_by_title: Split only at title (level 1) boundaries
        """
        self.max_chunk_size = min(max_chunk_size, MAX_CHUNK_SIZE)
        self.min_chunk_size = max(min_chunk_size, MIN_CHUNK_SIZE)
        self.combine_text_under_n_chars = combine_text_under_n_chars
        self.respect_section_boundaries = respect_section_boundaries
        self.chunk_by_title = chunk_by_title

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkRecord]:
        """Chunk text without markers (falls back to paragraph splitting).

        Args:
            text: Text to chunk
            document_id: Document identifier
            source_file: Source file path
            metadata: Optional metadata

        Returns:
            List of ChunkRecord objects
        """
        # Without markers, split on paragraph boundaries
        sections = self._split_by_paragraphs(text)
        return self._sections_to_chunks(sections, document_id, source_file, metadata)

    def chunk_with_markers(
        self,
        text: str,
        markers: List[ChapterMarker],
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkRecord]:
        """Chunk text using ChapterMarker boundaries.

        Args:
            text: Text to chunk
            markers: Chapter markers from refinement
            document_id: Document identifier
            source_file: Source file path
            metadata: Optional metadata

        Returns:
            List of ChunkRecord objects
        """
        if not text:
            return []

        if not markers:
            return self.chunk(text, document_id, source_file, metadata)

        # Extract sections from markers
        sections = self._markers_to_sections(text, markers)

        # Apply combination and splitting rules
        sections = self._optimize_sections(sections)

        return self._sections_to_chunks(sections, document_id, source_file, metadata)

    def _markers_to_sections(
        self, text: str, markers: List[ChapterMarker]
    ) -> List[LayoutSection]:
        """Convert markers to sections.

        Args:
            text: Full text
            markers: Chapter markers

        Returns:
            List of LayoutSection
        """
        if not markers:
            return [
                LayoutSection(
                    title="",
                    content=text,
                    level=0,
                    start_pos=0,
                    end_pos=len(text),
                )
            ]

        # Sort markers by position
        sorted_markers = sorted(markers, key=lambda m: m.position)

        # Filter by level if chunk_by_title
        if self.chunk_by_title:
            sorted_markers = [m for m in sorted_markers if m.level == 1]

        sections: List[LayoutSection] = []

        # Handle text before first marker
        first_pos = sorted_markers[0].position if sorted_markers else len(text)
        if first_pos > 0:
            sections.append(
                LayoutSection(
                    title="",
                    content=text[:first_pos].strip(),
                    level=0,
                    start_pos=0,
                    end_pos=first_pos,
                )
            )

        # Create sections from markers
        for i, marker in enumerate(sorted_markers[:MAX_SECTIONS]):
            start = marker.position
            end = (
                sorted_markers[i + 1].position
                if i + 1 < len(sorted_markers)
                else len(text)
            )

            content = text[start:end].strip()
            sections.append(
                LayoutSection(
                    title=marker.title,
                    content=content,
                    level=marker.level,
                    start_pos=start,
                    end_pos=end,
                )
            )

        return sections

    def _optimize_sections(self, sections: List[LayoutSection]) -> List[LayoutSection]:
        """Optimize section sizes by combining small and splitting large.

        Args:
            sections: Input sections

        Returns:
            Optimized sections
        """
        # First combine small sections
        combined = self._combine_small_sections(sections)

        # Then split oversized sections
        result = self._split_large_sections(combined)

        return result

    def _combine_small_sections(
        self, sections: List[LayoutSection]
    ) -> List[LayoutSection]:
        """Combine undersized sections.

        Args:
            sections: Input sections

        Returns:
            Sections with small ones combined
        """
        if not sections:
            return []

        result: List[LayoutSection] = []
        current: Optional[LayoutSection] = None

        for section in sections:
            if current is None:
                current = section
                continue

            # Combine if current is small
            if current.char_count < self.combine_text_under_n_chars:
                combined_content = current.content + "\n\n" + section.content
                current = LayoutSection(
                    title=current.title or section.title,
                    content=combined_content,
                    level=min(current.level, section.level),
                    start_pos=current.start_pos,
                    end_pos=section.end_pos,
                )
            else:
                result.append(current)
                current = section

        if current is not None:
            result.append(current)

        return result

    def _split_large_sections(
        self, sections: List[LayoutSection]
    ) -> List[LayoutSection]:
        """Split oversized sections.

        Args:
            sections: Input sections

        Returns:
            Sections with large ones split
        """
        result: List[LayoutSection] = []

        for section in sections:
            if section.char_count <= self.max_chunk_size:
                result.append(section)
                continue

            # Split at paragraph boundaries
            split_sections = self._split_section_by_paragraphs(section)
            result.extend(split_sections)

        return result

    def _split_section_by_paragraphs(
        self, section: LayoutSection
    ) -> List[LayoutSection]:
        """Split a large section at paragraph boundaries.

        Args:
            section: Section to split

        Returns:
            List of smaller sections
        """
        paragraphs = section.content.split("\n\n")
        sections: List[LayoutSection] = []

        current_content = ""
        current_start = section.start_pos

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            potential_size = len(current_content) + len(para) + 2  # +2 for \n\n

            if potential_size > self.max_chunk_size and current_content:
                # Save current and start new
                sections.append(
                    LayoutSection(
                        title=section.title if not sections else "",
                        content=current_content,
                        level=section.level,
                        start_pos=current_start,
                        end_pos=current_start + len(current_content),
                    )
                )
                current_content = para
                current_start = current_start + len(current_content) + 2
            else:
                if current_content:
                    current_content += "\n\n" + para
                else:
                    current_content = para

        # Add remaining content
        if current_content:
            sections.append(
                LayoutSection(
                    title=section.title if not sections else "",
                    content=current_content,
                    level=section.level,
                    start_pos=current_start,
                    end_pos=section.end_pos,
                )
            )

        return sections

    def _split_by_paragraphs(self, text: str) -> List[LayoutSection]:
        """Split text by paragraphs (fallback when no markers).

        Args:
            text: Text to split

        Returns:
            List of LayoutSection
        """
        paragraphs = text.split("\n\n")
        sections: List[LayoutSection] = []
        pos = 0

        for para in paragraphs[:MAX_SECTIONS]:
            para = para.strip()
            if not para:
                pos += 2
                continue

            sections.append(
                LayoutSection(
                    title="",
                    content=para,
                    level=0,
                    start_pos=pos,
                    end_pos=pos + len(para),
                )
            )
            pos += len(para) + 2

        return sections

    def _sections_to_chunks(
        self,
        sections: List[LayoutSection],
        document_id: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Convert sections to ChunkRecord objects.

        Args:
            sections: Sections to convert
            document_id: Document ID
            source_file: Source file path
            metadata: Optional metadata

        Returns:
            List of ChunkRecord
        """
        chunks: List[ChunkRecord] = []
        total = len(sections)

        for idx, section in enumerate(sections):
            if not section.content:
                continue

            chunk_id = self._generate_chunk_id(document_id, idx, section.content)

            chunk = ChunkRecord(
                chunk_id=chunk_id,
                document_id=document_id,
                content=section.content,
                section_title=section.title,
                source_file=source_file,
                word_count=section.word_count,
                char_count=section.char_count,
                chunk_index=idx,
                total_chunks=total,
                metadata=metadata.copy() if metadata else {},
            )

            chunks.append(chunk)

        return chunks

    def _generate_chunk_id(self, document_id: str, index: int, content: str) -> str:
        """Generate unique chunk ID.

        Args:
            document_id: Document ID
            index: Chunk index
            content: Chunk content

        Returns:
            Unique ID string
        """
        hash_input = f"{document_id}_{index}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


def chunk_by_layout(
    text: str,
    markers: List[ChapterMarker],
    document_id: str = "doc",
    max_chunk_size: int = 2000,
) -> List[ChunkRecord]:
    """Convenience function to chunk text with layout awareness.

    Args:
        text: Text to chunk
        markers: Chapter markers from refinement
        document_id: Document identifier
        max_chunk_size: Maximum chunk size

    Returns:
        List of ChunkRecord
    """
    chunker = LayoutChunker(max_chunk_size=max_chunk_size)
    return chunker.chunk_with_markers(text, markers, document_id)


def chunk_by_title(
    text: str,
    markers: List[ChapterMarker],
    document_id: str = "doc",
) -> List[ChunkRecord]:
    """Convenience function to chunk at title boundaries only.

    Args:
        text: Text to chunk
        markers: Chapter markers
        document_id: Document identifier

    Returns:
        List of ChunkRecord
    """
    chunker = LayoutChunker(chunk_by_title=True)
    return chunker.chunk_with_markers(text, markers, document_id)
