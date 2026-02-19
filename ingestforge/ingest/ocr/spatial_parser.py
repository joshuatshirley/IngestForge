"""Spatial Coordinate Extractor for hOCR/ALTO XML.

Parses Tesseract hOCR output to extract bounding boxes and
spatial information for all text blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_ELEMENTS_PER_PAGE = 10000
MAX_PAGES = 500
MAX_NESTED_DEPTH = 10


class ElementType(str, Enum):
    """Types of OCR elements."""

    PAGE = "page"
    BLOCK = "block"
    PARAGRAPH = "para"
    LINE = "line"
    WORD = "word"


@dataclass
class BoundingBox:
    """Rectangular bounding box for an element."""

    x1: int  # Left
    y1: int  # Top
    x2: int  # Right
    y2: int  # Bottom

    @property
    def width(self) -> int:
        """Get box width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get box height."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        """Get box area."""
        return self.width * self.height

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if boxes overlap."""
        return not (
            self.x2 < other.x1
            or self.x1 > other.x2
            or self.y2 < other.y1
            or self.y1 > other.y2
        )

    def contains(self, other: "BoundingBox") -> bool:
        """Check if this box contains another."""
        return (
            self.x1 <= other.x1
            and self.y1 <= other.y1
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )


@dataclass
class OCRElement:
    """A single OCR element with spatial information."""

    element_type: ElementType
    bbox: BoundingBox
    text: str = ""
    confidence: float = 0.0
    element_id: str = ""
    children: List["OCRElement"] = field(default_factory=list)
    attributes: dict = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        """Check if element has no text content."""
        return not self.text.strip() and not self.children


@dataclass
class OCRPage:
    """A single page with all its elements."""

    page_number: int
    width: int
    height: int
    elements: List[OCRElement] = field(default_factory=list)

    def get_blocks(self) -> List[OCRElement]:
        """Get all block-level elements."""
        return [e for e in self.elements if e.element_type == ElementType.BLOCK]

    def get_lines(self) -> List[OCRElement]:
        """Get all line-level elements."""
        lines = []
        for elem in self.elements:
            if elem.element_type == ElementType.LINE:
                lines.append(elem)
            for child in elem.children:
                if child.element_type == ElementType.LINE:
                    lines.append(child)
        return lines[:MAX_ELEMENTS_PER_PAGE]


@dataclass
class OCRDocument:
    """Complete OCR document with all pages."""

    pages: List[OCRPage] = field(default_factory=list)
    source_file: str = ""

    def get_all_text(self) -> str:
        """Get all text content from document."""
        texts = []
        for page in self.pages[:MAX_PAGES]:
            for elem in page.elements[:MAX_ELEMENTS_PER_PAGE]:
                if elem.text:
                    texts.append(elem.text)
        return "\n".join(texts)


class HOCRParser:
    """Parser for Tesseract hOCR output format.

    hOCR is an HTML-based format that includes bounding box
    information in the title attribute of span elements.
    """

    # Regex patterns for hOCR bbox extraction
    BBOX_PATTERN = re.compile(r"bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)")
    CONF_PATTERN = re.compile(r"x_wconf\s+(\d+)")

    def parse_file(self, file_path: Path) -> OCRDocument:
        """Parse hOCR file.

        Args:
            file_path: Path to hOCR HTML file

        Returns:
            Parsed OCRDocument
        """
        if not file_path.exists():
            logger.warning(f"hOCR file not found: {file_path}")
            return OCRDocument(source_file=str(file_path))

        content = file_path.read_text(encoding="utf-8", errors="ignore")
        return self.parse_string(content, source=str(file_path))

    def parse_string(self, content: str, source: str = "") -> OCRDocument:
        """Parse hOCR content string.

        Args:
            content: hOCR HTML content
            source: Source file name

        Returns:
            Parsed OCRDocument
        """
        doc = OCRDocument(source_file=source)

        # Parse HTML
        try:
            # Clean up for XML parsing
            content = self._prepare_for_parsing(content)
            root = ET.fromstring(content)
        except ET.ParseError as e:
            logger.error(f"Failed to parse hOCR: {e}")
            return doc

        # Find pages
        pages = self._find_pages(root)
        for idx, page_elem in enumerate(pages[:MAX_PAGES]):
            page = self._parse_page(page_elem, idx + 1)
            if page:
                doc.pages.append(page)

        return doc

    def _prepare_for_parsing(self, content: str) -> str:
        """Prepare hOCR content for XML parsing.

        Args:
            content: Raw hOCR content

        Returns:
            Cleaned content
        """
        # Add XML wrapper if needed
        if not content.strip().startswith("<?xml"):
            content = f'<?xml version="1.0"?>\n<root>{content}</root>'

        # Fix common issues
        content = re.sub(r"&(?!amp;|lt;|gt;|quot;|apos;)", "&amp;", content)

        return content

    def _find_pages(self, root: ET.Element) -> List[ET.Element]:
        """Find all page elements in document.

        Args:
            root: Root XML element

        Returns:
            List of page elements
        """
        # hOCR uses class="ocr_page"
        pages = []
        for elem in root.iter():
            class_attr = elem.get("class", "")
            if "ocr_page" in class_attr:
                pages.append(elem)
        return pages

    def _parse_page(self, page_elem: ET.Element, page_num: int) -> Optional[OCRPage]:
        """Parse a single page element.

        Args:
            page_elem: Page XML element
            page_num: Page number

        Returns:
            Parsed OCRPage or None
        """
        # Get page bbox for dimensions
        title = page_elem.get("title", "")
        bbox = self._extract_bbox(title)
        if not bbox:
            return None

        page = OCRPage(
            page_number=page_num,
            width=bbox.x2,
            height=bbox.y2,
        )

        # Parse child elements
        elements = self._parse_children(page_elem, depth=0)
        page.elements = elements[:MAX_ELEMENTS_PER_PAGE]

        return page

    def _parse_children(self, parent: ET.Element, depth: int) -> List[OCRElement]:
        """Parse child elements recursively.

        Args:
            parent: Parent XML element
            depth: Current recursion depth

        Returns:
            List of parsed elements
        """
        if depth >= MAX_NESTED_DEPTH:
            return []

        elements = []

        for child in parent:
            class_attr = child.get("class", "")
            element = self._parse_element(child, class_attr, depth)
            if element:
                elements.append(element)

        return elements

    def _parse_element(
        self, elem: ET.Element, class_attr: str, depth: int
    ) -> Optional[OCRElement]:
        """Parse a single element.

        Args:
            elem: XML element
            class_attr: Element class attribute
            depth: Current depth

        Returns:
            Parsed OCRElement or None
        """
        # Determine element type
        elem_type = self._classify_element(class_attr)
        if not elem_type:
            return None

        title = elem.get("title", "")
        bbox = self._extract_bbox(title)
        if not bbox:
            return None

        # Extract text
        text = self._extract_text(elem)

        # Extract confidence
        confidence = self._extract_confidence(title)

        element = OCRElement(
            element_type=elem_type,
            bbox=bbox,
            text=text,
            confidence=confidence,
            element_id=elem.get("id", ""),
        )

        # Parse children
        element.children = self._parse_children(elem, depth + 1)

        return element

    def _classify_element(self, class_attr: str) -> Optional[ElementType]:
        """Classify element type from class attribute.

        Args:
            class_attr: HTML class attribute

        Returns:
            ElementType or None
        """
        if "ocr_page" in class_attr:
            return ElementType.PAGE
        if "ocr_carea" in class_attr or "ocr_block" in class_attr:
            return ElementType.BLOCK
        if "ocr_par" in class_attr:
            return ElementType.PARAGRAPH
        if "ocr_line" in class_attr or "ocr_textfloat" in class_attr:
            return ElementType.LINE
        if "ocrx_word" in class_attr:
            return ElementType.WORD
        return None

    def _extract_bbox(self, title: str) -> Optional[BoundingBox]:
        """Extract bounding box from title attribute.

        Args:
            title: Title attribute value

        Returns:
            BoundingBox or None
        """
        match = self.BBOX_PATTERN.search(title)
        if not match:
            return None

        return BoundingBox(
            x1=int(match.group(1)),
            y1=int(match.group(2)),
            x2=int(match.group(3)),
            y2=int(match.group(4)),
        )

    def _extract_confidence(self, title: str) -> float:
        """Extract confidence from title attribute.

        Args:
            title: Title attribute value

        Returns:
            Confidence value (0.0-1.0)
        """
        match = self.CONF_PATTERN.search(title)
        if match:
            return int(match.group(1)) / 100.0
        return 0.0

    def _extract_text(self, elem: ET.Element) -> str:
        """Extract text content from element.

        Args:
            elem: XML element

        Returns:
            Text content
        """
        # Get direct text
        text_parts = []
        if elem.text:
            text_parts.append(elem.text.strip())

        # Get tail text from children
        for child in elem:
            if child.tail:
                text_parts.append(child.tail.strip())

        return " ".join(text_parts)


class ALTOParser:
    """Parser for ALTO XML format.

    ALTO (Analyzed Layout and Text Object) is an XML schema
    for technical metadata describing the physical and logical
    structure of text content.
    """

    # ALTO namespace
    ALTO_NS = {"alto": "http://www.loc.gov/standards/alto/ns-v3#"}

    def parse_file(self, file_path: Path) -> OCRDocument:
        """Parse ALTO XML file.

        Args:
            file_path: Path to ALTO file

        Returns:
            Parsed OCRDocument
        """
        if not file_path.exists():
            logger.warning(f"ALTO file not found: {file_path}")
            return OCRDocument(source_file=str(file_path))

        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Failed to parse ALTO: {e}")
            return OCRDocument(source_file=str(file_path))

        return self._parse_root(root, str(file_path))

    def _parse_root(self, root: ET.Element, source: str) -> OCRDocument:
        """Parse ALTO root element.

        Args:
            root: Root XML element
            source: Source file name

        Returns:
            Parsed OCRDocument
        """
        doc = OCRDocument(source_file=source)

        # Find Layout section
        layout = root.find(".//Layout", self.ALTO_NS)
        if layout is None:
            # Try without namespace
            layout = root.find(".//Layout")

        if layout is None:
            return doc

        # Find pages
        pages = layout.findall("Page", self.ALTO_NS) or layout.findall("Page")

        for idx, page_elem in enumerate(pages[:MAX_PAGES]):
            page = self._parse_alto_page(page_elem, idx + 1)
            if page:
                doc.pages.append(page)

        return doc

    def _parse_alto_page(
        self, page_elem: ET.Element, page_num: int
    ) -> Optional[OCRPage]:
        """Parse ALTO page element.

        Args:
            page_elem: Page element
            page_num: Page number

        Returns:
            Parsed OCRPage
        """
        width = int(page_elem.get("WIDTH", 0))
        height = int(page_elem.get("HEIGHT", 0))

        if width == 0 or height == 0:
            return None

        page = OCRPage(page_number=page_num, width=width, height=height)

        # Find TextBlocks
        print_space = page_elem.find("PrintSpace", self.ALTO_NS) or page_elem.find(
            "PrintSpace"
        )
        if print_space is None:
            return page

        blocks = print_space.findall(
            ".//TextBlock", self.ALTO_NS
        ) or print_space.findall(".//TextBlock")

        for block in blocks[:MAX_ELEMENTS_PER_PAGE]:
            element = self._parse_alto_block(block)
            if element:
                page.elements.append(element)

        return page

    def _parse_alto_block(self, block: ET.Element) -> Optional[OCRElement]:
        """Parse ALTO TextBlock element.

        Args:
            block: TextBlock element

        Returns:
            Parsed OCRElement
        """
        bbox = self._extract_alto_bbox(block)
        if not bbox:
            return None

        element = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            element_id=block.get("ID", ""),
        )

        # Parse text lines
        lines = block.findall("TextLine", self.ALTO_NS) or block.findall("TextLine")
        for line in lines:
            line_elem = self._parse_alto_line(line)
            if line_elem:
                element.children.append(line_elem)

        # Combine text from children
        element.text = " ".join(child.text for child in element.children if child.text)

        return element

    def _parse_alto_line(self, line: ET.Element) -> Optional[OCRElement]:
        """Parse ALTO TextLine element.

        Args:
            line: TextLine element

        Returns:
            Parsed OCRElement
        """
        bbox = self._extract_alto_bbox(line)
        if not bbox:
            return None

        # Get words
        words = []
        strings = line.findall("String", self.ALTO_NS) or line.findall("String")
        for string in strings:
            content = string.get("CONTENT", "")
            if content:
                words.append(content)

        return OCRElement(
            element_type=ElementType.LINE,
            bbox=bbox,
            text=" ".join(words),
            element_id=line.get("ID", ""),
        )

    def _extract_alto_bbox(self, elem: ET.Element) -> Optional[BoundingBox]:
        """Extract bounding box from ALTO element.

        Args:
            elem: XML element

        Returns:
            BoundingBox or None
        """
        try:
            hpos = int(float(elem.get("HPOS", 0)))
            vpos = int(float(elem.get("VPOS", 0)))
            width = int(float(elem.get("WIDTH", 0)))
            height = int(float(elem.get("HEIGHT", 0)))

            if width == 0 or height == 0:
                return None

            return BoundingBox(
                x1=hpos,
                y1=vpos,
                x2=hpos + width,
                y2=vpos + height,
            )
        except (ValueError, TypeError):
            return None


def parse_ocr_file(file_path: Path) -> OCRDocument:
    """Parse OCR output file (auto-detect format).

    Args:
        file_path: Path to OCR file

    Returns:
        Parsed OCRDocument
    """
    suffix = file_path.suffix.lower()

    if suffix in (".hocr", ".html", ".htm"):
        parser = HOCRParser()
        return parser.parse_file(file_path)

    if suffix in (".alto", ".xml"):
        # Check content to determine format
        content = file_path.read_text(encoding="utf-8", errors="ignore")[:500]
        if "alto" in content.lower() or "ALTO" in content:
            parser = ALTOParser()
            return parser.parse_file(file_path)

        # Default to hOCR for other XML
        parser = HOCRParser()
        return parser.parse_file(file_path)

    logger.warning(f"Unknown OCR format: {suffix}")
    return OCRDocument(source_file=str(file_path))
