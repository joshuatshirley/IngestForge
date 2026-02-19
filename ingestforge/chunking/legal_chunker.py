"""
Legal document chunking.

Splits legal documents by numbered sections/clauses (1.1, 1.2, Article 3, §4)
rather than semantic similarity, preserving document structure for citations.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config


@dataclass
class LegalSection:
    """A numbered section from a legal document."""

    number: str  # "1.1", "2.3.1", "Article 5"
    title: str  # Section title if present
    content: str  # Section content
    level: int  # Nesting level (1, 2, 3...)
    parent_number: Optional[str] = None  # Parent section number
    line_start: int = 0
    line_end: int = 0


class LegalChunker:
    """
    Chunker for legal documents that splits by numbered sections.

    Recognizes patterns like:
    - Numeric: "1.", "1.1", "1.1.1", "1.1.1.1"
    - Article: "Article 1", "Article I", "ARTICLE 1"
    - Section: "Section 1", "Section 1.1", "Sec. 1"
    - Paragraph: "§1", "§ 1.1", "¶1"
    - Roman: "I.", "II.", "III.", "IV."
    - Lettered: "(a)", "(b)", "(i)", "(ii)"
    """

    # Section header patterns (order matters - more specific first)
    SECTION_PATTERNS = [
        # Article headers
        (r"^(ARTICLE|Article)\s+(\d+|[IVXLCDM]+)\.?\s*[-–—:]?\s*(.*?)$", "article"),
        # Section headers
        (r"^(SECTION|Section|Sec\.?)\s+(\d+(?:\.\d+)*)\s*[-–—:]?\s*(.*?)$", "section"),
        # Paragraph symbol
        (r"^[§¶]\s*(\d+(?:\.\d+)*)\s*[-–—:]?\s*(.*?)$", "paragraph"),
        # Numbered sections (most common in contracts)
        (r"^(\d+(?:\.\d+)+)\s*[-–—:]?\s*(.*?)$", "numbered"),
        # Single number sections
        (r"^(\d+)\.\s+(.*?)$", "numbered_single"),
        # Roman numerals
        (r"^([IVXLCDM]+)\.\s*(.*?)$", "roman"),
        # Lettered subsections
        (r"^\(([a-z])\)\s*(.*?)$", "letter_lower"),
        (r"^\(([ivx]+)\)\s*(.*?)$", "roman_lower"),
        (r"^\((\d+)\)\s*(.*?)$", "number_paren"),
    ]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.min_section_size = 50  # Minimum chars to be a standalone chunk
        self.max_section_size = 2000  # Max chars before splitting

        # Compile patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.MULTILINE), ptype)
            for pattern, ptype in self.SECTION_PATTERNS
        ]

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict] = None,
    ) -> List[ChunkRecord]:
        """
        Chunk legal text by numbered sections.

        Rule #1: Reduced nesting from 4 → 2 levels
        Rule #4: Reduced from 66 → 28 lines

        Args:
            text: Legal document text
            document_id: Parent document ID
            source_file: Source file path
            metadata: Additional metadata

        Returns:
            List of ChunkRecord objects with section hierarchy preserved
        """
        if not text.strip():
            return []

        sections = self._extract_sections(text)
        if not sections:
            return self._fallback_chunk(text, document_id, source_file)
        records = self._create_section_chunks(sections, document_id, source_file)

        # Update total_chunks across all records
        total = len(records)
        for record in records:
            record.total_chunks = total

        return records

    def _create_section_chunks(
        self, sections: List, document_id: str, source_file: str
    ) -> List[ChunkRecord]:
        """
        Create chunk records from legal sections.

        Rule #1: Extracted to reduce nesting (max 2 levels)
        Rule #4: Function <60 lines
        """
        records = []
        for i, section in enumerate(sections):
            hierarchy = self._build_hierarchy(section, sections[:i])
            section_texts = self._split_if_needed(section.content)

            for j, chunk_text in enumerate(section_texts):
                record = self._create_single_section_chunk(
                    section,
                    chunk_text,
                    document_id,
                    source_file,
                    i,
                    j,
                    len(records),
                    hierarchy,
                )
                records.append(record)

        return records

    def _create_single_section_chunk(
        self,
        section: Any,
        chunk_text: str,
        document_id: str,
        source_file: str,
        section_idx: int,
        chunk_idx: int,
        record_idx: int,
        hierarchy: List[str],
    ) -> ChunkRecord:
        """
        Create a single chunk record for a section.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        chunk_id = self._generate_chunk_id(
            document_id, section_idx, chunk_idx, chunk_text
        )
        section_title = self._build_section_title(section)

        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            content=chunk_text,
            chunk_type="legal_clause",
            section_title=section_title,
            section_hierarchy=hierarchy,
            source_file=source_file,
            word_count=len(chunk_text.split()),
            char_count=len(chunk_text),
            chunk_index=record_idx,
            total_chunks=0,  # Updated by caller
        )

    def _build_section_title(self, section: Any) -> str:
        """
        Build section title from section data.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        if section.title:
            return f"{section.number} {section.title}"
        return section.number

    def _extract_sections(self, text: str) -> List[LegalSection]:
        """Extract numbered sections from text.

        Rule #1: Reduced nesting via helper extraction
        """
        sections = []
        lines = text.split("\n")

        current_section = None
        current_content = []
        current_start = 0

        for line_num, line in enumerate(lines):
            stripped = line.strip()

            # Check if line is a section header
            section_match = self._match_section_header(stripped)

            if section_match:
                # Save previous section
                self._save_section_if_valid(
                    current_section, current_content, line_num - 1, sections
                )

                # Start new section
                number, title, level, parent = section_match
                current_section = LegalSection(
                    number=number,
                    title=title,
                    content="",
                    level=level,
                    parent_number=parent,
                    line_start=line_num,
                )
                current_content = [stripped]
                current_start = line_num
            else:
                current_content.append(line)

        # Don't forget the last section
        self._save_section_if_valid(
            current_section, current_content, len(lines) - 1, sections
        )

        # Handle case where document has no section headers at start
        if not sections and current_content:
            # Create a single section for the whole document
            sections.append(
                LegalSection(
                    number="1",
                    title="",
                    content="\n".join(current_content).strip(),
                    level=1,
                    line_start=0,
                    line_end=len(lines) - 1,
                )
            )

        return sections

    def _save_section_if_valid(
        self,
        section: Optional[LegalSection],
        content_lines: List[str],
        end_line: int,
        sections_list: List[LegalSection],
    ) -> None:
        """Save section to list if it has valid content.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if not section:
            return

        section.content = "\n".join(content_lines).strip()
        section.line_end = end_line

        if section.content:
            sections_list.append(section)

    def _match_section_header(
        self, line: str
    ) -> Optional[Tuple[str, str, int, Optional[str]]]:
        """
        Match a section header pattern.

        Returns:
            Tuple of (number, title, level, parent_number) or None
        """
        if not line or len(line) > 200:  # Headers shouldn't be too long
            return None

        for pattern, ptype in self._compiled_patterns:
            match = pattern.match(line)
            if match:
                result = self._parse_section_type(ptype, match.groups())
                if result:
                    return result

        return None

    def _parse_section_type(
        self, ptype: str, groups: tuple
    ) -> Optional[Tuple[str, str, int, Optional[str]]]:
        """Parse matched groups based on section pattern type."""
        parsers = {
            "article": self._parse_article,
            "section": self._parse_section,
            "paragraph": self._parse_paragraph,
            "numbered": self._parse_numbered,
            "numbered_single": self._parse_numbered_single,
            "roman": self._parse_roman,
        }

        if ptype in parsers:
            return parsers[ptype](groups)
        elif ptype in ("letter_lower", "roman_lower", "number_paren"):
            return self._parse_subsection(groups)
        return None

    def _parse_article(self, groups: tuple) -> Tuple[str, str, int, Optional[str]]:
        """Parse Article pattern: Article 1 Title."""
        number = f"Article {groups[1]}"
        title = groups[2] if len(groups) > 2 else ""
        return (number, title.strip(), 1, None)

    def _parse_section(self, groups: tuple) -> Tuple[str, str, int, Optional[str]]:
        """Parse Section pattern: Section 1.1 Title."""
        number = groups[1]
        title = groups[2] if len(groups) > 2 else ""
        level = number.count(".") + 1
        parent = ".".join(number.split(".")[:-1]) if "." in number else None
        return (number, title.strip(), level, parent)

    def _parse_paragraph(self, groups: tuple) -> Tuple[str, str, int, Optional[str]]:
        """Parse Paragraph pattern: §1.1 Title."""
        number = f"§{groups[0]}"
        title = groups[1] if len(groups) > 1 else ""
        level = groups[0].count(".") + 1
        return (number, title.strip(), level, None)

    def _parse_numbered(self, groups: tuple) -> Tuple[str, str, int, Optional[str]]:
        """Parse numbered pattern: 1.1.1 Title."""
        number = groups[0]
        title = groups[1] if len(groups) > 1 else ""
        level = number.count(".") + 1
        parent = ".".join(number.split(".")[:-1]) if "." in number else None
        return (number, title.strip(), level, parent)

    def _parse_numbered_single(
        self, groups: tuple
    ) -> Tuple[str, str, int, Optional[str]]:
        """Parse single numbered pattern: 1. Title."""
        number = groups[0]
        title = groups[1] if len(groups) > 1 else ""
        return (number, title.strip(), 1, None)

    def _parse_roman(self, groups: tuple) -> Tuple[str, str, int, Optional[str]]:
        """Parse Roman numeral pattern: I. Title."""
        number = groups[0]
        title = groups[1] if len(groups) > 1 else ""
        return (number, title.strip(), 1, None)

    def _parse_subsection(self, groups: tuple) -> Tuple[str, str, int, Optional[str]]:
        """Parse subsection patterns: (a), (i), (1) Title."""
        number = f"({groups[0]})"
        title = groups[1] if len(groups) > 1 else ""
        return (number, title.strip(), 3, None)

    def _build_hierarchy(
        self, section: LegalSection, previous_sections: List[LegalSection]
    ) -> List[str]:
        """Build section hierarchy for citations."""
        hierarchy = []

        # Find parent sections
        if section.parent_number:
            for prev in reversed(previous_sections):
                if prev.number == section.parent_number:
                    # Recursively build parent's hierarchy
                    hierarchy = self._build_hierarchy(
                        prev, previous_sections[: previous_sections.index(prev)]
                    )
                    hierarchy.append(prev.number)
                    break

        return hierarchy

    def _split_if_needed(self, content: str) -> List[str]:
        """Split content if it exceeds max size."""
        if len(content) <= self.max_section_size:
            return [content]

        # Split by paragraphs first
        paragraphs = re.split(r"\n\s*\n", content)

        chunks = []
        current = []
        current_len = 0

        for para in paragraphs:
            para_len = len(para)

            # If single paragraph exceeds max size, split by sentences
            if para_len > self.max_section_size:
                # Flush current buffer first
                if current:
                    chunks.append("\n\n".join(current))
                    current = []
                    current_len = 0
                # Split large paragraph by sentences
                sentence_chunks = self._split_large_paragraph(para)
                chunks.extend(sentence_chunks)
                continue

            if current_len + para_len > self.max_section_size and current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0

            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks

    def _split_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph by sentences when it exceeds max size.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        # Split by sentence endings
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)

        chunks = []
        current = []
        current_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If single sentence exceeds max, include it anyway (avoid infinite recursion)
            if sentence_len > self.max_section_size and not current:
                chunks.append(sentence)
                continue

            if current_len + sentence_len + 1 > self.max_section_size and current:
                chunks.append(" ".join(current))
                current = []
                current_len = 0

            current.append(sentence)
            current_len += sentence_len + 1  # +1 for space

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _fallback_chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
    ) -> List[ChunkRecord]:
        """Fallback chunking when no sections are found."""
        from ingestforge.chunking.semantic_chunker import SemanticChunker

        chunker = SemanticChunker(self.config)
        return chunker.chunk(text, document_id, source_file)

    def _generate_chunk_id(
        self, document_id: str, section_idx: int, sub_idx: int, content: str
    ) -> str:
        """Generate unique chunk ID."""
        import hashlib

        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{document_id}_legal_{section_idx:04d}_{sub_idx:02d}_{content_hash}"
