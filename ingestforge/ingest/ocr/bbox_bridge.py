"""
Bounding Box Bridge for OCR-to-Chunk Coordinate Flow.

Bridges BoundingBox coordinates from OCR extraction through to
ChunkMetadata and ChunkRecord, enabling precise source citations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.ingest.ocr.spatial_parser import BoundingBox, OCRElement, OCRPage
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_ELEMENTS_PER_CHUNK = 100


@dataclass
class ChunkBoundingBox:
    """Bounding box for a text chunk.

    Represents the combined bounding box of all OCR elements
    that make up a chunk. Includes page number for multi-page documents.

    Attributes:
        x1: Left edge coordinate
        y1: Top edge coordinate
        x2: Right edge coordinate
        y2: Bottom edge coordinate
        page_number: Page number in source document
    """

    x1: int
    y1: int
    x2: int
    y2: int
    page_number: int = 1

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary for metadata."""
        return {
            "bbox_x1": self.x1,
            "bbox_y1": self.y1,
            "bbox_x2": self.x2,
            "bbox_y2": self.y2,
            "page_number": self.page_number,
        }

    @classmethod
    def from_ocr_bbox(
        cls, bbox: BoundingBox, page_number: int = 1
    ) -> "ChunkBoundingBox":
        """Create from OCR BoundingBox.

        Args:
            bbox: OCR bounding box
            page_number: Page number in document

        Returns:
            ChunkBoundingBox instance
        """
        return cls(
            x1=bbox.x1,
            y1=bbox.y1,
            x2=bbox.x2,
            y2=bbox.y2,
            page_number=page_number,
        )

    @property
    def width(self) -> int:
        """Get width."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Get height."""
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        """Get center point."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def contains_point(self, x: int, y: int) -> bool:
        """Check if point is within bounding box."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class BoundingBoxBridge:
    """Bridges OCR bounding boxes to chunk metadata.

    This class provides utilities for:
    1. Extracting bounding boxes from OCR elements
    2. Combining multiple element boxes into chunk boxes
    3. Converting box coordinates to chunk metadata format
    4. Finding elements within a given text range

    Usage:
        >>> bridge = BoundingBoxBridge()
        >>> elements = get_ocr_elements()  # From OCR
        >>> chunk_bbox = bridge.combine_element_boxes(elements, page_num=1)
        >>> metadata = chunk_bbox.to_dict()
    """

    def __init__(self) -> None:
        """Initialize the bounding box bridge."""
        pass

    def combine_element_boxes(
        self, elements: List[OCRElement], page_number: int = 1
    ) -> Optional[ChunkBoundingBox]:
        """Combine bounding boxes of multiple elements.

        Creates a single bounding box that encompasses all given elements.

        Args:
            elements: List of OCR elements
            page_number: Page number for the combined box

        Returns:
            Combined ChunkBoundingBox or None if no valid boxes
        """
        if not elements:
            return None

        # Filter elements with valid bounding boxes
        valid_elements = [e for e in elements[:MAX_ELEMENTS_PER_CHUNK] if e.bbox]
        if not valid_elements:
            return None

        # Calculate encompassing box
        x1 = min(e.bbox.x1 for e in valid_elements)
        y1 = min(e.bbox.y1 for e in valid_elements)
        x2 = max(e.bbox.x2 for e in valid_elements)
        y2 = max(e.bbox.y2 for e in valid_elements)

        return ChunkBoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, page_number=page_number)

    def get_page_bounding_boxes(self, page: OCRPage) -> List[ChunkBoundingBox]:
        """Get bounding boxes for all elements on a page.

        Args:
            page: OCR page with elements

        Returns:
            List of ChunkBoundingBox for each element
        """
        boxes = []

        for element in page.elements[:MAX_ELEMENTS_PER_CHUNK]:
            if not element.bbox:
                continue

            box = ChunkBoundingBox.from_ocr_bbox(
                element.bbox, page_number=page.page_number
            )
            boxes.append(box)

        return boxes

    def find_elements_in_range(
        self,
        elements: List[OCRElement],
        text: str,
        start_char: int,
        end_char: int,
    ) -> List[OCRElement]:
        """Find OCR elements that correspond to a text range.

        Uses text matching to identify which elements fall within
        the given character range.

        Args:
            elements: All OCR elements
            text: Full text content
            start_char: Start character offset
            end_char: End character offset

        Returns:
            List of elements within the range
        """
        if not elements or not text:
            return []

        target_text = text[start_char:end_char]
        matching_elements = []
        current_pos = 0

        for element in elements[:MAX_ELEMENTS_PER_CHUNK]:
            if not element.text:
                continue

            # Find element position in full text
            elem_start = text.find(element.text, current_pos)
            if elem_start == -1:
                continue

            elem_end = elem_start + len(element.text)

            # Check if element overlaps with target range
            if elem_start < end_char and elem_end > start_char:
                matching_elements.append(element)

            current_pos = elem_start + 1

        return matching_elements

    def bbox_for_text_range(
        self,
        page: OCRPage,
        text: str,
        start_char: int,
        end_char: int,
    ) -> Optional[ChunkBoundingBox]:
        """Get bounding box for a specific text range.

        Args:
            page: OCR page
            text: Full text from the page
            start_char: Start character offset
            end_char: End character offset

        Returns:
            ChunkBoundingBox for the text range or None
        """
        # Flatten all elements from page
        all_elements = self._flatten_elements(page.elements)

        # Find elements in range
        range_elements = self.find_elements_in_range(
            all_elements, text, start_char, end_char
        )

        # Combine their bounding boxes
        return self.combine_element_boxes(range_elements, page.page_number)

    def _flatten_elements(self, elements: List[OCRElement]) -> List[OCRElement]:
        """Flatten nested OCR elements.

        Args:
            elements: Potentially nested elements

        Returns:
            Flat list of all elements
        """
        flat: List[OCRElement] = []

        for element in elements[:MAX_ELEMENTS_PER_CHUNK]:
            flat.append(element)
            if element.children:
                flat.extend(self._flatten_elements(element.children))

        return flat[:MAX_ELEMENTS_PER_CHUNK]

    def update_chunk_metadata(
        self,
        metadata: Dict[str, Any],
        bbox: Optional[ChunkBoundingBox],
    ) -> Dict[str, Any]:
        """Update chunk metadata with bounding box coordinates.

        Args:
            metadata: Existing chunk metadata
            bbox: Bounding box to add (or None)

        Returns:
            Updated metadata dictionary
        """
        if bbox is None:
            return metadata

        result = metadata.copy()
        result.update(bbox.to_dict())

        return result


def extract_bbox_from_elements(
    elements: List[OCRElement], page_number: int = 1
) -> Optional[Tuple[int, int, int, int]]:
    """Convenience function to get bbox tuple from elements.

    Args:
        elements: List of OCR elements
        page_number: Page number

    Returns:
        (x1, y1, x2, y2) tuple or None
    """
    bridge = BoundingBoxBridge()
    bbox = bridge.combine_element_boxes(elements, page_number)
    return bbox.to_tuple() if bbox else None


def bbox_to_metadata(
    bbox: Optional[Tuple[int, int, int, int]], page_number: int = 1
) -> Dict[str, int]:
    """Convert bbox tuple to metadata dictionary.

    Args:
        bbox: (x1, y1, x2, y2) tuple
        page_number: Page number

    Returns:
        Dictionary with bbox_x1, bbox_y1, bbox_x2, bbox_y2, page_number
    """
    if bbox is None:
        return {}

    return {
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "page_number": page_number,
    }
