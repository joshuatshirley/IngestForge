"""
Element Classifier for Unstructured-style Element Typing.

Classifies text elements by type (Title, NarrativeText, ListItem, Table,
Header, Footer, Code, Image) for better retrieval weighting and filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ingestforge.ingest.refinement import (
    ChapterMarker,
    DocumentElementType,
    IRefiner,
    RefinedText,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_ELEMENTS = 1000
MAX_LINE_LENGTH = 500
MIN_TITLE_LENGTH = 3
MAX_TITLE_LENGTH = 200


@dataclass
class ClassifiedElement:
    """A text element with its classification.

    Attributes:
        text: The element text content
        element_type: Classification type
        start_pos: Start position in document
        end_pos: End position in document
        confidence: Classification confidence (0.0-1.0)
    """

    text: str
    element_type: DocumentElementType
    start_pos: int
    end_pos: int
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "element_type": self.element_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "confidence": self.confidence,
        }


class ElementClassifier(IRefiner):
    """Classify text elements by semantic type.

    This refiner analyzes text to identify element types:

    1. **Title**: Chapter/section headings, document titles
    2. **NarrativeText**: Regular paragraph text
    3. **ListItem**: Bullet points, numbered lists
    4. **Table**: Tabular data (detected by structure)
    5. **Header**: Page headers (top of page, repeated)
    6. **Footer**: Page footers, page numbers
    7. **Code**: Code blocks, monospace sections
    8. **Image**: Image placeholders, figure references

    Uses pattern matching and spatial analysis for classification.

    Examples:
        >>> classifier = ElementClassifier()
        >>> result = classifier.refine("# Introduction\\n\\nThis is text.")
        >>> # Access classifications via chapter_markers
    """

    # List item patterns
    LIST_PATTERNS = [
        re.compile(r"^\s*[-•●○▪▸]\s+", re.MULTILINE),  # Bullet points
        re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE),  # Numbered lists
        re.compile(r"^\s*[a-z][.)]\s+", re.MULTILINE),  # Lettered lists
        re.compile(r"^\s*[ivxIVX]+[.)]\s+", re.MULTILINE),  # Roman numerals
    ]

    # Code block patterns
    CODE_PATTERNS = [
        re.compile(r"^```[\s\S]*?```$", re.MULTILINE),  # Fenced code
        re.compile(r"^    .+$", re.MULTILINE),  # Indented code
        re.compile(r"def \w+\s*\(|class \w+\s*[:(]|function \w+\s*\("),  # Function defs
    ]

    # Table patterns
    TABLE_PATTERNS = [
        re.compile(r"^\|.+\|$", re.MULTILINE),  # Markdown tables
        re.compile(r"^[\t|]+.+[\t|]+$", re.MULTILINE),  # Tab-separated
    ]

    # Image/figure patterns
    IMAGE_PATTERNS = [
        re.compile(r"!\[.*?\]\(.*?\)"),  # Markdown images
        re.compile(r"\[Figure\s+\d+\]", re.IGNORECASE),  # Figure references
        re.compile(r"\[Image\s*:?\s*.*?\]", re.IGNORECASE),  # Image placeholders
    ]

    def __init__(
        self,
        detect_titles: bool = True,
        detect_lists: bool = True,
        detect_code: bool = True,
        detect_tables: bool = True,
        use_chapter_markers: bool = True,
    ) -> None:
        """Initialize element classifier.

        Args:
            detect_titles: Detect title elements (default True)
            detect_lists: Detect list items (default True)
            detect_code: Detect code blocks (default True)
            detect_tables: Detect tables (default True)
            use_chapter_markers: Use existing chapter markers (default True)
        """
        self.detect_titles = detect_titles
        self.detect_lists = detect_lists
        self.detect_code = detect_code
        self.detect_tables = detect_tables
        self.use_chapter_markers = use_chapter_markers

    def is_available(self) -> bool:
        """Always available - uses only standard library."""
        return True

    def refine(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> RefinedText:
        """Classify elements in text.

        The classifications are stored in chapter_markers with the element_type
        encoded in the title field as a prefix: "[TYPE] actual title"

        Args:
            text: Text to classify
            metadata: Optional metadata

        Returns:
            RefinedText with element classifications in chapter_markers
        """
        if not text:
            return RefinedText(original=text, refined=text)

        elements = self._classify_all_elements(text)
        markers = self._elements_to_markers(elements)
        changes = self._build_changes(elements)

        return RefinedText(
            original=text,
            refined=text,  # Text not modified
            changes=changes,
            chapter_markers=markers,
        )

    def classify_text(self, text: str) -> List[ClassifiedElement]:
        """Classify all elements in text.

        Public method for direct element classification.

        Args:
            text: Text to classify

        Returns:
            List of ClassifiedElement objects
        """
        return self._classify_all_elements(text)

    def classify_line(self, line: str) -> DocumentElementType:
        """Classify a single line.

        Args:
            line: Line to classify

        Returns:
            DocumentElementType for the line
        """
        return self._classify_single_element(line)

    def _classify_all_elements(self, text: str) -> List[ClassifiedElement]:
        """Classify all elements in text.

        Args:
            text: Full text

        Returns:
            List of classified elements
        """
        elements: List[ClassifiedElement] = []

        # Split into paragraphs/blocks
        blocks = self._split_into_blocks(text)

        pos = 0
        for block in blocks[:MAX_ELEMENTS]:
            element_type = self._classify_single_element(block)
            confidence = self._calculate_confidence(block, element_type)

            elements.append(
                ClassifiedElement(
                    text=block,
                    element_type=element_type,
                    start_pos=pos,
                    end_pos=pos + len(block),
                    confidence=confidence,
                )
            )

            pos += len(block) + 1  # +1 for newline

        return elements

    def _split_into_blocks(self, text: str) -> List[str]:
        """Split text into logical blocks.

        Args:
            text: Text to split

        Returns:
            List of text blocks
        """
        # Split on paragraph breaks
        blocks = re.split(r"\n\n+", text)
        return [b.strip() for b in blocks if b.strip()]

    def _classify_single_element(self, text: str) -> DocumentElementType:
        """Classify a single text element.

        Args:
            text: Text to classify

        Returns:
            DocumentElementType
        """
        if not text:
            return DocumentElementType.UNCATEGORIZED

        # Check code first (highest specificity)
        if self.detect_code and self._is_code(text):
            return DocumentElementType.CODE

        # Check table
        if self.detect_tables and self._is_table(text):
            return DocumentElementType.TABLE

        # Check image/figure reference
        if self._is_image(text):
            return DocumentElementType.IMAGE

        # Check list item
        if self.detect_lists and self._is_list_item(text):
            return DocumentElementType.LIST_ITEM

        # Check header/footer BEFORE title (page numbers look like short titles)
        if self._is_header_footer(text):
            return DocumentElementType.HEADER

        # Check title
        if self.detect_titles and self._is_title(text):
            return DocumentElementType.TITLE

        # Default to narrative text
        return DocumentElementType.NARRATIVE_TEXT

    def _is_title(self, text: str) -> bool:
        """Check if text is a title.

        Args:
            text: Text to check

        Returns:
            True if text appears to be a title
        """
        stripped = text.strip()

        # Length constraints
        if len(stripped) < MIN_TITLE_LENGTH:
            return False
        if len(stripped) > MAX_TITLE_LENGTH:
            return False

        # Markdown header
        if stripped.startswith("#"):
            return True

        # All caps (likely chapter heading)
        if stripped.isupper() and len(stripped.split()) >= 2:
            return True

        # Numbered section pattern
        if re.match(r"^\d+(\.\d+)*\s+[A-Z]", stripped):
            return True

        # Chapter/Section keyword
        if re.match(r"^(Chapter|Section|Part)\s+", stripped, re.IGNORECASE):
            return True

        # Short line with no sentence-ending punctuation
        if len(stripped) < 80 and not stripped.endswith((".", "?", "!")):
            # And starts with capital
            if stripped[0].isupper():
                return True

        return False

    def _is_list_item(self, text: str) -> bool:
        """Check if text is a list item.

        Args:
            text: Text to check

        Returns:
            True if text is a list item
        """
        for pattern in self.LIST_PATTERNS:
            if pattern.match(text):
                return True
        return False

    def _is_code(self, text: str) -> bool:
        """Check if text is code.

        Args:
            text: Text to check

        Returns:
            True if text appears to be code
        """
        for pattern in self.CODE_PATTERNS:
            if pattern.search(text):
                return True

        # Check for code indicators
        code_indicators = ["def ", "class ", "function ", "import ", "return "]
        for indicator in code_indicators:
            if indicator in text:
                return True

        return False

    def _is_table(self, text: str) -> bool:
        """Check if text is a table.

        Args:
            text: Text to check

        Returns:
            True if text is a table
        """
        for pattern in self.TABLE_PATTERNS:
            if pattern.search(text):
                return True

        # Check for consistent column separators
        lines = text.split("\n")
        if len(lines) >= 2:
            pipe_counts = [line.count("|") for line in lines]
            if all(c >= 2 for c in pipe_counts) and len(set(pipe_counts)) == 1:
                return True

        return False

    def _is_image(self, text: str) -> bool:
        """Check if text references an image.

        Args:
            text: Text to check

        Returns:
            True if text is image reference
        """
        for pattern in self.IMAGE_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _is_header_footer(self, text: str) -> bool:
        """Check if text is a header or footer.

        Args:
            text: Text to check

        Returns:
            True if text appears to be header/footer
        """
        stripped = text.strip()

        # Page number patterns
        if re.match(r"^(Page\s+)?\d+(\s+of\s+\d+)?$", stripped, re.IGNORECASE):
            return True

        # Very short single line (likely header)
        if len(stripped) < 40 and "\n" not in stripped:
            # Contains date pattern
            if re.search(r"\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}", stripped):
                return True

        return False

    def _calculate_confidence(
        self, text: str, element_type: DocumentElementType
    ) -> float:
        """Calculate confidence for classification.

        Args:
            text: Classified text
            element_type: Assigned type

        Returns:
            Confidence score 0.0-1.0
        """
        # Base confidence by type
        if element_type == DocumentElementType.NARRATIVE_TEXT:
            return 0.9  # Default type has high confidence

        if element_type == DocumentElementType.CODE:
            # Higher confidence if multiple code indicators
            indicators = sum(1 for p in self.CODE_PATTERNS if p.search(text))
            return min(0.7 + (indicators * 0.1), 1.0)

        if element_type == DocumentElementType.TABLE:
            # Higher confidence for clear table structure
            lines = text.split("\n")
            if len(lines) >= 3:
                return 0.95
            return 0.8

        if element_type == DocumentElementType.TITLE:
            # Higher confidence for markdown headers
            if text.strip().startswith("#"):
                return 0.95
            return 0.8

        return 0.85  # Default confidence for other types

    def _elements_to_markers(
        self, elements: List[ClassifiedElement]
    ) -> List[ChapterMarker]:
        """Convert elements to chapter markers.

        Encodes element type in marker title as "[TYPE] text".

        Args:
            elements: Classified elements

        Returns:
            List of ChapterMarker
        """
        markers = []

        for element in elements:
            # Only create markers for non-narrative types
            if element.element_type == DocumentElementType.NARRATIVE_TEXT:
                continue

            # Truncate text for marker title
            text_preview = element.text[:50]
            if len(element.text) > 50:
                text_preview += "..."

            title = f"[{element.element_type.value}] {text_preview}"

            # Map element type to hierarchy level
            level = self._type_to_level(element.element_type)

            markers.append(
                ChapterMarker(
                    position=element.start_pos,
                    title=title,
                    level=level,
                )
            )

        return markers

    def _type_to_level(self, element_type: DocumentElementType) -> int:
        """Map element type to hierarchy level.

        Args:
            element_type: Element type

        Returns:
            Hierarchy level (1-4)
        """
        level_map = {
            DocumentElementType.TITLE: 1,
            DocumentElementType.TABLE: 2,
            DocumentElementType.CODE: 2,
            DocumentElementType.IMAGE: 2,
            DocumentElementType.LIST_ITEM: 3,
            DocumentElementType.HEADER: 4,
            DocumentElementType.FOOTER: 4,
        }
        return level_map.get(element_type, 3)

    def _build_changes(self, elements: List[ClassifiedElement]) -> List[str]:
        """Build change summary.

        Args:
            elements: Classified elements

        Returns:
            List of change descriptions
        """
        type_counts: Dict[DocumentElementType, int] = {}

        for element in elements:
            elem_type = element.element_type
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1

        changes = []
        for elem_type, count in sorted(type_counts.items(), key=lambda x: x[0].value):
            if elem_type != DocumentElementType.NARRATIVE_TEXT:
                changes.append(f"Detected {count} {elem_type.value} elements")

        return changes


def classify_elements(text: str) -> List[ClassifiedElement]:
    """Convenience function to classify text elements.

    Args:
        text: Text to classify

    Returns:
        List of ClassifiedElement
    """
    classifier = ElementClassifier()
    return classifier.classify_text(text)


def get_element_type(text: str) -> str:
    """Get element type for a single text block.

    Args:
        text: Text to classify

    Returns:
        Element type string
    """
    classifier = ElementClassifier()
    return classifier.classify_line(text).value
