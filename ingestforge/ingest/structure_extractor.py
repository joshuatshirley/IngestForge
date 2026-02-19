"""
Document structure extraction for IngestForge.

Extracts chapter, section, and subsection hierarchy from documents
to enable precise citation tracking (e.g., "Chapter 3, Section 2.1, p.47").
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum


class HeadingLevel(Enum):
    """Document heading levels."""

    PART = 0
    CHAPTER = 1
    SECTION = 2
    SUBSECTION = 3
    SUBSUBSECTION = 4


@dataclass
class DocumentHeading:
    """A heading within a document."""

    level: HeadingLevel
    title: str
    number: Optional[str] = None  # "3" or "3.2" or "3.2.1"
    page: Optional[int] = None
    char_position: Optional[int] = None  # Position in full text

    @property
    def full_title(self) -> str:
        """Get full title with number if available."""
        if self.number:
            return f"{self.number} {self.title}"
        return self.title

    @property
    def section_number(self) -> Optional[str]:
        """Get the section number (e.g., '3.2.1')."""
        return self.number

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.name,
            "title": self.title,
            "number": self.number,
            "page": self.page,
            "char_position": self.char_position,
        }


@dataclass
class DocumentSection:
    """A section within a document with content and subsections."""

    heading: DocumentHeading
    content: str = ""
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    subsections: List["DocumentSection"] = field(default_factory=list)

    @property
    def level(self) -> HeadingLevel:
        return self.heading.level

    @property
    def title(self) -> str:
        return self.heading.title

    @property
    def number(self) -> Optional[str]:
        return self.heading.number

    def get_all_content(self) -> str:
        """Get content including all subsections."""
        parts = [self.content]
        for sub in self.subsections:
            parts.append(sub.get_all_content())
        return "\n\n".join(parts)


@dataclass
class DocumentStructure:
    """Complete document structure."""

    title: str
    sections: List[DocumentSection]
    headings: List[DocumentHeading]
    page_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def find_section_for_page(self, page: int) -> Optional[DocumentSection]:
        """Find the section containing a given page."""
        return self._find_section_recursive(self.sections, page)

    def _find_section_recursive(
        self,
        sections: List[DocumentSection],
        page: int,
    ) -> Optional[DocumentSection]:
        """Recursively find section for page.

        Rule #1: Reduced nesting via helper extraction
        """
        for section in sections:
            match = self._check_section_for_page(section, page)
            if match:
                return match
        return None

    def _check_section_for_page(
        self, section: DocumentSection, page: int
    ) -> Optional[DocumentSection]:
        """Check if section contains page, including subsections.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        # Check if section has defined page range
        if section.page_start and section.page_end:
            if section.page_start <= page <= section.page_end:
                # Check subsections first for more specific match
                sub_match = self._find_section_recursive(section.subsections, page)
                return sub_match or section
            return None

        # Check if section starts at or before page
        if section.page_start and section.page_start <= page:
            sub_match = self._find_section_recursive(section.subsections, page)
            if sub_match:
                return sub_match
            return section

        return None

    def get_location_for_page(self, page: int) -> Dict[str, str]:
        """
        Get structured location for a page number.

        Returns dict with chapter, section, subsection as available.
        """
        result: dict[str, Any] = {}
        section = self.find_section_for_page(page)

        if not section:
            return result

        # Walk up the hierarchy
        current = section
        if current.level == HeadingLevel.SUBSUBSECTION:
            result["subsubsection"] = current.heading.full_title
            result["subsubsection_number"] = current.number
        if current.level == HeadingLevel.SUBSECTION:
            result["subsection"] = current.heading.full_title
            result["subsection_number"] = current.number
        if current.level == HeadingLevel.SECTION:
            result["section"] = current.heading.full_title
            result["section_number"] = current.number
        if current.level == HeadingLevel.CHAPTER:
            result["chapter"] = current.heading.full_title
            result["chapter_number"] = current.number

        return result


class StructureExtractor:
    """
    Extract document structure from PDF and text documents.

    Detects chapters, sections, and subsections using:
    - Font size analysis (for PDFs)
    - Numbering patterns (1.1, 1.2, etc.)
    - Keyword patterns (Chapter, Section, etc.)
    - Table of contents parsing

    Example:
        extractor = StructureExtractor()
        structure = extractor.extract_from_pdf(Path("document.pdf"))

        for heading in structure.headings:
            print(f"{heading.level.name}: {heading.full_title} (p.{heading.page})")
    """

    # Patterns for detecting headings
    CHAPTER_PATTERNS = [
        r"^Chapter\s+(\d+)[:\.\s]+(.+)$",
        r"^CHAPTER\s+(\d+)[:\.\s]+(.+)$",
        r"^(\d+)\.\s+(.+)$",  # "1. Introduction"
        r"^Part\s+(\d+|[IVX]+)[:\.\s]+(.+)$",
    ]

    SECTION_PATTERNS = [
        r"^(\d+\.\d+)\s+(.+)$",  # "1.1 Background"
        r"^Section\s+(\d+\.\d+)[:\.\s]+(.+)$",
        r"^§\s*(\d+\.\d+)\s+(.+)$",
    ]

    SUBSECTION_PATTERNS = [
        r"^(\d+\.\d+\.\d+)\s+(.+)$",  # "1.1.1 Details"
        r"^(\d+\.\d+\.\d+\.\d+)\s+(.+)$",  # "1.1.1.1 More details"
    ]

    def __init__(
        self,
        min_heading_words: int = 2,
        max_heading_words: int = 15,
    ):
        """
        Initialize structure extractor.

        Args:
            min_heading_words: Minimum words for a valid heading
            max_heading_words: Maximum words for a valid heading
        """
        self.min_heading_words = min_heading_words
        self.max_heading_words = max_heading_words

    def extract_from_pdf(self, file_path: Path) -> DocumentStructure:
        """
        Extract structure from a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            DocumentStructure with headings and sections
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))
            headings: list[DocumentHeading] = []
            page_count = len(doc)

            # Method 1: Try to extract from TOC (table of contents)
            toc = doc.get_toc()
            if toc:
                headings = self._parse_toc(toc)

            # Method 2: If no TOC, analyze text with font sizes
            if not headings:
                headings = self._extract_headings_from_text(doc)

            doc.close()

            # Build section hierarchy
            sections = self._build_section_hierarchy(headings)

            # Get document title
            title = self._extract_title(file_path, headings)

            return DocumentStructure(
                title=title,
                sections=sections,
                headings=headings,
                page_count=page_count,
            )

        except ImportError:
            raise ImportError(
                "PyMuPDF (fitz) is required for PDF structure extraction. "
                "Install with: pip install pymupdf"
            )

    def _determine_page_from_breaks(
        self, char_pos: int, page_breaks: List[int]
    ) -> Optional[int]:
        """
        Determine page number from character position.

        Rule #1: Simple loop with early return
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            char_pos: Character position in document
            page_breaks: List of character positions for page breaks

        Returns:
            Page number or None
        """
        for page, break_pos in enumerate(page_breaks):
            if char_pos < break_pos:
                return page + 1
        return None

    def _assign_page_to_heading(
        self, heading: DocumentHeading, char_pos: int, page_breaks: Optional[List[int]]
    ) -> None:
        """
        Assign page number to heading if page breaks provided.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            heading: Heading to update
            char_pos: Character position
            page_breaks: Optional page break positions
        """
        if not page_breaks:
            return

        page_num = self._determine_page_from_breaks(char_pos, page_breaks)
        if page_num:
            heading.page = page_num

    def _process_text_line(
        self,
        line: str,
        char_pos: int,
        page_breaks: Optional[List[int]],
        headings: List[DocumentHeading],
    ) -> int:
        """
        Process single text line for heading detection.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            line: Text line to process
            char_pos: Current character position
            page_breaks: Optional page break positions
            headings: List to append detected headings to

        Returns:
            Updated character position
        """
        stripped = line.strip()
        if not stripped:
            return char_pos + len(line) + 1

        heading = self._detect_heading(stripped, char_pos)
        if heading:
            self._assign_page_to_heading(heading, char_pos, page_breaks)
            headings.append(heading)

        return char_pos + len(line) + 1

    def extract_from_text(
        self,
        text: str,
        page_breaks: Optional[List[int]] = None,
    ) -> DocumentStructure:
        """
        Extract structure from plain text.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            text: Document text
            page_breaks: Character positions of page breaks

        Returns:
            DocumentStructure with detected headings
        """
        headings: list[DocumentHeading] = []
        lines = text.split("\n")
        char_pos = 0
        for line in lines:
            char_pos = self._process_text_line(line, char_pos, page_breaks, headings)

        sections = self._build_section_hierarchy(headings)

        return DocumentStructure(
            title=self._extract_title_from_headings(headings),
            sections=sections,
            headings=headings,
        )

    def _parse_toc(self, toc: List[Any]) -> List[DocumentHeading]:
        """Parse PDF table of contents into headings.

        Rule #1: Reduced nesting via dictionary dispatch
        """
        headings: List[DocumentHeading] = []

        for entry in toc:
            level, title, page = entry[0], entry[1], entry[2]

            # Map TOC level to HeadingLevel
            heading_level = self._map_toc_level_to_heading_level(level)

            # Try to extract number from title
            number, clean_title = self._extract_number(title)

            headings.append(
                DocumentHeading(
                    level=heading_level,
                    title=clean_title,
                    number=number,
                    page=page,
                )
            )

        return headings

    def _map_toc_level_to_heading_level(self, level: int) -> HeadingLevel:
        """Map TOC level number to HeadingLevel enum.

        Rule #1: Dictionary dispatch to eliminate if/elif chain
        Rule #4: Helper function <60 lines
        """
        level_map = {
            1: HeadingLevel.CHAPTER,
            2: HeadingLevel.SECTION,
            3: HeadingLevel.SUBSECTION,
        }
        return level_map.get(level, HeadingLevel.SUBSUBSECTION)

    def _extract_headings_from_text(self, doc: Any) -> List[DocumentHeading]:
        """Extract headings by analyzing text patterns."""
        headings: List[DocumentHeading] = []

        for page_num, page in enumerate(doc, 1):
            text = page.get_text()
            lines = text.split("\n")

            for line in lines:
                line = line.strip()
                heading = self._detect_heading(line)
                if heading:
                    heading.page = page_num
                    headings.append(heading)

        return headings

    def _detect_heading(
        self,
        text: str,
        char_position: Optional[int] = None,
    ) -> Optional[DocumentHeading]:
        """
        Detect if a line is a heading and determine its level.

        Rule #4: Reduced from 75 lines to <60 lines via helper extraction

        Args:
            text: Line of text to analyze
            char_position: Position in document

        Returns:
            DocumentHeading if detected, None otherwise
        """
        text = text.strip()
        word_count = len(text.split())
        if not self._is_valid_heading_length(text, word_count):
            return None
        heading = self._match_subsection_patterns(text, char_position)
        if heading:
            return heading

        heading = self._match_section_patterns(text, char_position)
        if heading:
            return heading

        heading = self._match_chapter_patterns(text, char_position)
        if heading:
            return heading

        # Check for all-caps headings (likely chapters)
        if text.isupper() and word_count <= 8:
            return DocumentHeading(
                level=HeadingLevel.CHAPTER,
                title=text.title(),
                char_position=char_position,
            )

        return None

    def _is_valid_heading_length(self, text: str, word_count: int) -> bool:
        """
        Check if text has valid word count for heading.

        Rule #4: Extracted to reduce function size
        """
        if word_count < self.min_heading_words or word_count > self.max_heading_words:
            # Exception for numbered headings
            if not re.match(r"^\d+\.", text):
                return False
        return True

    def _match_subsection_patterns(
        self, text: str, char_position: Optional[int]
    ) -> Optional[DocumentHeading]:
        """
        Match subsection patterns (most specific first).

        Rule #4: Extracted to reduce function size
        """
        for pattern in self.SUBSECTION_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                number = match.group(1)
                title = match.group(2).strip()
                # Determine depth by counting dots
                depth = number.count(".")
                if depth >= 3:
                    level = HeadingLevel.SUBSUBSECTION
                else:
                    level = HeadingLevel.SUBSECTION

                return DocumentHeading(
                    level=level,
                    title=title,
                    number=number,
                    char_position=char_position,
                )
        return None

    def _match_section_patterns(
        self, text: str, char_position: Optional[int]
    ) -> Optional[DocumentHeading]:
        """
        Match section patterns.

        Rule #4: Extracted to reduce function size
        """
        for pattern in self.SECTION_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return DocumentHeading(
                    level=HeadingLevel.SECTION,
                    title=match.group(2).strip(),
                    number=match.group(1),
                    char_position=char_position,
                )
        return None

    def _match_chapter_patterns(
        self, text: str, char_position: Optional[int]
    ) -> Optional[DocumentHeading]:
        """
        Match chapter patterns.

        Rule #4: Extracted to reduce function size
        """
        for pattern in self.CHAPTER_PATTERNS:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                return DocumentHeading(
                    level=HeadingLevel.CHAPTER,
                    title=match.group(2).strip(),
                    number=match.group(1),
                    char_position=char_position,
                )
        return None

    def _extract_number(self, title: str) -> Tuple[Optional[str], str]:
        """Extract number prefix from title."""
        patterns = [
            r"^(\d+(?:\.\d+)*)[:\.\s]+(.+)$",  # "1.2.3 Title" or "1.2.3: Title"
            r"^Chapter\s+(\d+)[:\.\s]+(.+)$",
            r"^Section\s+(\d+(?:\.\d+)*)[:\.\s]+(.+)$",
        ]

        for pattern in patterns:
            match = re.match(pattern, title, re.IGNORECASE)
            if match:
                return match.group(1), match.group(2).strip()

        return None, title

    def _build_section_hierarchy(
        self,
        headings: List[DocumentHeading],
    ) -> List[DocumentSection]:
        """Build hierarchical section structure from flat heading list."""
        if not headings:
            return []

        sections = []
        stack: List[DocumentSection] = []

        for i, heading in enumerate(headings):
            section = DocumentSection(
                heading=heading,
                page_start=heading.page,
            )

            # Calculate page_end from next heading
            if i + 1 < len(headings):
                next_page = headings[i + 1].page
                if next_page and heading.page:
                    section.page_end = (
                        next_page - 1 if next_page > heading.page else heading.page
                    )
            elif heading.page:
                section.page_end = heading.page  # Last section

            # Find parent based on level
            while stack and stack[-1].heading.level.value >= heading.level.value:
                stack.pop()

            if stack:
                stack[-1].subsections.append(section)
            else:
                sections.append(section)

            stack.append(section)

        return sections

    def _extract_title(
        self,
        file_path: Path,
        headings: List[DocumentHeading],
    ) -> str:
        """Extract document title."""
        # Try first heading
        if headings:
            first = headings[0]
            if first.level in [HeadingLevel.PART, HeadingLevel.CHAPTER]:
                return first.title

        # Fall back to filename
        return file_path.stem.replace("_", " ").replace("-", " ").title()

    def _extract_title_from_headings(self, headings: List[DocumentHeading]) -> str:
        """Extract title from heading list."""
        if headings:
            return headings[0].title
        return "Untitled Document"


def extract_pdf_structure(file_path: Path) -> DocumentStructure:
    """
    Convenience function to extract structure from PDF.

    Args:
        file_path: Path to PDF file

    Returns:
        DocumentStructure with headings and sections
    """
    extractor = StructureExtractor()
    return extractor.extract_from_pdf(file_path)


def extract_text_structure(
    text: str,
    page_breaks: Optional[List[int]] = None,
) -> DocumentStructure:
    """
    Convenience function to extract structure from text.

    Args:
        text: Document text
        page_breaks: Character positions of page breaks

    Returns:
        DocumentStructure with detected headings
    """
    extractor = StructureExtractor()
    return extractor.extract_from_text(text, page_breaks)
