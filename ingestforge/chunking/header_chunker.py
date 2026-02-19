"""Header-based chunking strategy.

Splits documents at section headers to keep related content together.
Ideal for structured documents like regulations, manuals, and technical docs.

This chunker identifies headers (Chapter, Section, numbered headings, markdown)
and creates chunks that preserve the natural document structure.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ingestforge.chunking.semantic_chunker import ChunkRecord


@dataclass
class HeaderMatch:
    """A detected header in the document."""

    text: str
    level: int  # 1 = Chapter, 2 = Section, 3 = Subsection, etc.
    position: int  # Character position in document
    header_type: str  # "chapter", "section", "numbered", "markdown"


class HeaderChunker:
    """Chunk documents by section headers.

    Detects structural headers and creates chunks that keep all content
    under a header together. This preserves document semantics better
    than arbitrary size-based chunking.

    Supported header formats:
    - Chapter X / CHAPTER X
    - Section X.X / SECTION X.X
    - Numbered headings (1. / 1.1 / a. / (1))
    - Markdown headers (# / ## / ###)
    - Appendix X
    - Article X
    """

    # Header patterns with priority levels
    HEADER_PATTERNS = [
        # Chapter headers (level 1) - trailing punct optional
        (r"^(?:CHAPTER|Chapter)\s+(\d+|[IVXLC]+)(?:[.:\s]|$)", 1, "chapter"),
        # Appendix headers (level 1)
        (r"^(?:APPENDIX|Appendix)\s+([A-Z])(?:[.:\s]|$)", 1, "appendix"),
        # Article headers (level 1)
        (r"^(?:ARTICLE|Article)\s+(\d+|[IVXLC]+)(?:[.:\s]|$)", 1, "article"),
        # Section headers (level 2)
        (r"^(?:SECTION|Section)\s+(\d+(?:\.\d+)?)(?:[.:\s]|$)", 2, "section"),
        # Numbered section (e.g., "4-1." or "4.1" or "4–1")
        (r"^(\d+[-–.]\d+)[.:\s]", 2, "numbered"),
        # Lettered subsection (e.g., "a." or "(a)")
        (r"^(?:\()?([a-z])\)?[.:\s]", 3, "lettered"),
        # Parenthetical numbered (e.g., "(1)")
        (r"^\((\d+)\)[.:\s]", 3, "parenthetical"),
        # Markdown H1
        (r"^#\s+(.+)$", 1, "markdown"),
        # Markdown H2
        (r"^##\s+(.+)$", 2, "markdown"),
        # Markdown H3
        (r"^###\s+(.+)$", 3, "markdown"),
        # ALL CAPS header (likely section title)
        (r"^([A-Z][A-Z\s]{10,50})$", 2, "caps"),
    ]

    def __init__(
        self,
        min_chunk_size: int = 100,
        max_chunk_size: int = 4000,
        split_level: int = 2,
        include_header_in_chunk: bool = True,
    ) -> None:
        """Initialize header chunker.

        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size (will split large sections)
            split_level: Header level to split at (1=chapter, 2=section, 3=subsection)
            include_header_in_chunk: Include the header text in the chunk content
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.split_level = split_level
        self.include_header_in_chunk = include_header_in_chunk

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkRecord]:
        """Chunk text by headers.

        Args:
            text: Text to chunk
            document_id: Unique document identifier
            source_file: Source file path
            metadata: Optional metadata

        Returns:
            List of ChunkRecord objects
        """
        if not text or len(text) < self.min_chunk_size:
            return self._create_single_chunk(text, document_id, source_file, metadata)

        # Find all headers
        headers = self._find_headers(text)

        if not headers:
            # No headers found - fall back to paragraph chunking
            return self._chunk_by_paragraphs(text, document_id, source_file, metadata)

        # Create chunks from header sections
        chunks = self._create_chunks_from_headers(
            text, headers, document_id, source_file, metadata
        )

        return chunks

    def _find_headers(self, text: str) -> List[HeaderMatch]:
        """Find all headers in the document."""
        headers = []
        lines = text.split("\n")
        current_pos = 0

        for line in lines:
            stripped = line.strip()

            # Try each pattern
            for pattern, level, header_type in self.HEADER_PATTERNS:
                match = re.match(pattern, stripped, re.MULTILINE)
                if match:
                    headers.append(
                        HeaderMatch(
                            text=stripped,
                            level=level,
                            position=current_pos,
                            header_type=header_type,
                        )
                    )
                    break  # Only match one pattern per line

            current_pos += len(line) + 1  # +1 for newline

        return headers

    def _create_chunks_from_headers(
        self,
        text: str,
        headers: List[HeaderMatch],
        document_id: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Create chunks based on header positions."""
        chunks = []

        # Filter to only headers at or above split level
        split_headers = [h for h in headers if h.level <= self.split_level]

        if not split_headers:
            # Use all headers if none match split level
            split_headers = headers

        # Add virtual end header
        split_headers.append(
            HeaderMatch(
                text="[END]",
                level=0,
                position=len(text),
                header_type="end",
            )
        )

        # Create chunks between headers
        for i, header in enumerate(split_headers[:-1]):
            next_header = split_headers[i + 1]

            # Extract section content
            start = header.position
            end = next_header.position

            if not self.include_header_in_chunk:
                # Skip past the header line
                newline_pos = text.find("\n", start)
                if newline_pos != -1 and newline_pos < end:
                    start = newline_pos + 1

            section_text = text[start:end].strip()

            if len(section_text) < self.min_chunk_size:
                continue  # Skip very small sections

            # Split if too large
            if len(section_text) > self.max_chunk_size:
                sub_chunks = self._split_large_section(
                    section_text,
                    header.text,
                    document_id,
                    source_file,
                    len(chunks),
                    metadata,
                )
                chunks.extend(sub_chunks)
            else:
                chunk = self._create_chunk_record(
                    section_text,
                    header.text,
                    document_id,
                    source_file,
                    len(chunks),
                    metadata,
                )
                chunks.append(chunk)

        # Update total_chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _split_large_section(
        self,
        text: str,
        section_title: str,
        document_id: str,
        source_file: str,
        start_index: int,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Split a large section into smaller chunks at paragraph boundaries."""
        chunks = []
        paragraphs = text.split("\n\n")

        current_text = ""
        current_index = start_index

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # Check if adding this paragraph exceeds max size
            test_text = current_text + "\n\n" + para if current_text else para

            if len(test_text) > self.max_chunk_size and current_text:
                # Save current chunk
                chunk = self._create_chunk_record(
                    current_text,
                    section_title,
                    document_id,
                    source_file,
                    current_index,
                    metadata,
                )
                chunks.append(chunk)
                current_index += 1
                current_text = para
            else:
                current_text = test_text

        # Don't forget the last chunk
        if current_text and len(current_text) >= self.min_chunk_size:
            chunk = self._create_chunk_record(
                current_text,
                section_title,
                document_id,
                source_file,
                current_index,
                metadata,
            )
            chunks.append(chunk)

        return chunks

    def _chunk_by_paragraphs(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Fallback: chunk by paragraphs when no headers found."""
        chunks = []
        paragraphs = text.split("\n\n")

        current_text = ""
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            test_text = current_text + "\n\n" + para if current_text else para

            if len(test_text) > self.max_chunk_size and current_text:
                chunk = self._create_chunk_record(
                    current_text, "", document_id, source_file, chunk_index, metadata
                )
                chunks.append(chunk)
                chunk_index += 1
                current_text = para
            else:
                current_text = test_text

        if current_text:
            chunk = self._create_chunk_record(
                current_text, "", document_id, source_file, chunk_index, metadata
            )
            chunks.append(chunk)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)

        return chunks

    def _create_single_chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Create a single chunk for short text."""
        return [
            self._create_chunk_record(
                text or "", "", document_id, source_file, 0, metadata
            )
        ]

    def _create_chunk_record(
        self,
        text: str,
        section_title: str,
        document_id: str,
        source_file: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]],
    ) -> ChunkRecord:
        """Create a ChunkRecord object."""
        chunk_id = hashlib.md5(
            f"{document_id}_{chunk_index}_{text[:50]}".encode()
        ).hexdigest()[:16]

        chunk_meta = metadata.copy() if metadata else {}
        chunk_meta["chunking_strategy"] = "header"

        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            content=text,
            section_title=section_title,
            source_file=source_file,
            word_count=len(text.split()),
            char_count=len(text),
            chunk_index=chunk_index,
            metadata=chunk_meta,
        )

    def get_strategy_name(self) -> str:
        """Return strategy name."""
        return "header"

    def get_config(self) -> Dict[str, Any]:
        """Return configuration."""
        return {
            "strategy": "header",
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "split_level": self.split_level,
        }
