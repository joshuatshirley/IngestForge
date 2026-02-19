"""Multi-Column Sequencer for OCR reading order.

Determines correct reading order for multi-column layouts
using spatial analysis of text blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from ingestforge.ingest.ocr.spatial_parser import (
    OCRDocument,
    OCRElement,
    OCRPage,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_COLUMNS = 10
MAX_BLOCKS_PER_PAGE = 1000
COLUMN_GAP_THRESHOLD = 50  # Minimum gap to consider column break


class LayoutType(str, Enum):
    """Types of page layouts."""

    SINGLE_COLUMN = "single"
    TWO_COLUMN = "two_column"
    THREE_COLUMN = "three_column"
    MULTI_COLUMN = "multi_column"
    MIXED = "mixed"


@dataclass
class Column:
    """A detected column region."""

    x_start: int
    x_end: int
    elements: List[OCRElement] = field(default_factory=list)

    @property
    def width(self) -> int:
        """Get column width."""
        return self.x_end - self.x_start

    @property
    def center_x(self) -> int:
        """Get column center X coordinate."""
        return (self.x_start + self.x_end) // 2


@dataclass
class ReadingOrder:
    """Ordered sequence of elements for reading."""

    elements: List[OCRElement] = field(default_factory=list)
    layout_type: LayoutType = LayoutType.SINGLE_COLUMN
    column_count: int = 1

    def get_text(self, separator: str = "\n") -> str:
        """Get text in reading order.

        Args:
            separator: Text separator

        Returns:
            Combined text
        """
        return separator.join(elem.text for elem in self.elements if elem.text)


@dataclass
class SequencerConfig:
    """Configuration for the sequencer."""

    column_gap_threshold: int = COLUMN_GAP_THRESHOLD
    prefer_top_to_bottom: bool = True
    merge_overlapping: bool = True
    min_block_height: int = 10


class MultiColumnSequencer:
    """Determines reading order for multi-column layouts.

    Analyzes spatial distribution of text blocks to detect
    columns and establish correct reading flow.
    """

    def __init__(self, config: Optional[SequencerConfig] = None) -> None:
        """Initialize sequencer.

        Args:
            config: Sequencer configuration
        """
        self.config = config or SequencerConfig()

    def sequence_document(self, doc: OCRDocument) -> List[ReadingOrder]:
        """Determine reading order for entire document.

        Args:
            doc: OCR document

        Returns:
            List of ReadingOrder per page
        """
        orders = []
        for page in doc.pages:
            order = self.sequence_page(page)
            orders.append(order)
        return orders

    def sequence_page(self, page: OCRPage) -> ReadingOrder:
        """Determine reading order for a single page.

        Args:
            page: OCR page

        Returns:
            ReadingOrder for the page
        """
        if not page.elements:
            return ReadingOrder()

        # Get blocks
        blocks = page.get_blocks()[:MAX_BLOCKS_PER_PAGE]
        if not blocks:
            # Fall back to all elements
            blocks = page.elements[:MAX_BLOCKS_PER_PAGE]

        # Detect layout
        layout_type, columns = self._detect_layout(blocks, page.width)

        # Sort elements by reading order
        ordered_elements = self._sort_by_reading_order(columns, layout_type)

        return ReadingOrder(
            elements=ordered_elements,
            layout_type=layout_type,
            column_count=len(columns),
        )

    def _detect_layout(
        self, blocks: List[OCRElement], page_width: int
    ) -> Tuple[LayoutType, List[Column]]:
        """Detect page layout and identify columns.

        Args:
            blocks: List of text blocks
            page_width: Page width

        Returns:
            Tuple of (LayoutType, list of Columns)
        """
        if not blocks:
            return LayoutType.SINGLE_COLUMN, []

        # Analyze x-coordinate distribution
        x_ranges = self._get_x_ranges(blocks)
        columns = self._find_columns(x_ranges, page_width)

        # Classify layout
        layout_type = self._classify_layout(len(columns))

        # Assign blocks to columns
        self._assign_blocks_to_columns(blocks, columns)

        return layout_type, columns

    def _get_x_ranges(
        self, blocks: List[OCRElement]
    ) -> List[Tuple[int, int, OCRElement]]:
        """Get X coordinate ranges for all blocks.

        Args:
            blocks: List of blocks

        Returns:
            List of (x_start, x_end, element) tuples
        """
        ranges = []
        for block in blocks:
            if block.bbox:
                ranges.append((block.bbox.x1, block.bbox.x2, block))
        return sorted(ranges, key=lambda r: r[0])

    def _find_columns(
        self, x_ranges: List[Tuple[int, int, OCRElement]], page_width: int
    ) -> List[Column]:
        """Find column boundaries from x ranges.

        Args:
            x_ranges: Sorted x coordinate ranges
            page_width: Page width

        Returns:
            List of detected columns
        """
        if not x_ranges:
            return [Column(x_start=0, x_end=page_width)]

        # Find gaps in x coverage
        gaps = self._find_gaps(x_ranges, page_width)

        # Create columns from gaps
        columns = self._gaps_to_columns(gaps, page_width)

        return columns[:MAX_COLUMNS]

    def _find_gaps(
        self, x_ranges: List[Tuple[int, int, OCRElement]], page_width: int
    ) -> List[Tuple[int, int]]:
        """Find gaps between text regions.

        Args:
            x_ranges: Sorted x coordinate ranges
            page_width: Page width

        Returns:
            List of (gap_start, gap_end) tuples
        """
        # Merge overlapping ranges
        merged = self._merge_ranges(x_ranges)

        gaps = []
        prev_end = 0

        for start, end in merged:
            if start - prev_end > self.config.column_gap_threshold:
                gaps.append((prev_end, start))
            prev_end = max(prev_end, end)

        return gaps

    def _merge_ranges(
        self, x_ranges: List[Tuple[int, int, OCRElement]]
    ) -> List[Tuple[int, int]]:
        """Merge overlapping x ranges.

        Args:
            x_ranges: List of (start, end, element) tuples

        Returns:
            List of merged (start, end) tuples
        """
        if not x_ranges:
            return []

        merged = []
        current_start, current_end = x_ranges[0][0], x_ranges[0][1]

        for start, end, _ in x_ranges[1:]:
            if start <= current_end + self.config.column_gap_threshold:
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))
        return merged

    def _gaps_to_columns(
        self, gaps: List[Tuple[int, int]], page_width: int
    ) -> List[Column]:
        """Convert gaps to column definitions.

        Args:
            gaps: List of gaps
            page_width: Page width

        Returns:
            List of columns
        """
        if not gaps:
            return [Column(x_start=0, x_end=page_width)]

        columns = []
        prev_end = 0

        for gap_start, gap_end in gaps:
            if gap_start > prev_end:
                columns.append(Column(x_start=prev_end, x_end=gap_start))
            prev_end = gap_end

        # Add final column
        if prev_end < page_width:
            columns.append(Column(x_start=prev_end, x_end=page_width))

        return columns

    def _classify_layout(self, column_count: int) -> LayoutType:
        """Classify layout type based on column count.

        Args:
            column_count: Number of columns

        Returns:
            LayoutType
        """
        if column_count <= 1:
            return LayoutType.SINGLE_COLUMN
        if column_count == 2:
            return LayoutType.TWO_COLUMN
        if column_count == 3:
            return LayoutType.THREE_COLUMN
        return LayoutType.MULTI_COLUMN

    def _assign_blocks_to_columns(
        self, blocks: List[OCRElement], columns: List[Column]
    ) -> None:
        """Assign blocks to their containing columns.

        Args:
            blocks: List of blocks
            columns: List of columns (modified in place)
        """
        for block in blocks:
            if not block.bbox:
                continue

            center_x = block.bbox.center[0]
            best_column = self._find_best_column(center_x, columns)
            if best_column:
                best_column.elements.append(block)

    def _find_best_column(self, x: int, columns: List[Column]) -> Optional[Column]:
        """Find the column containing an x coordinate.

        Args:
            x: X coordinate
            columns: List of columns

        Returns:
            Best matching column or None
        """
        for col in columns:
            if col.x_start <= x <= col.x_end:
                return col

        # Find closest column
        if columns:
            return min(columns, key=lambda c: abs(c.center_x - x))

        return None

    def _sort_by_reading_order(
        self, columns: List[Column], layout_type: LayoutType
    ) -> List[OCRElement]:
        """Sort elements by reading order.

        Args:
            columns: List of columns with assigned elements
            layout_type: Page layout type

        Returns:
            Sorted list of elements
        """
        ordered = []

        # Sort columns left to right
        sorted_columns = sorted(columns, key=lambda c: c.x_start)

        for column in sorted_columns:
            # Sort elements within column top to bottom
            sorted_elements = sorted(
                column.elements,
                key=lambda e: e.bbox.y1 if e.bbox else 0,
            )
            ordered.extend(sorted_elements)

        return ordered


def sequence_ocr_document(doc: OCRDocument) -> List[ReadingOrder]:
    """Convenience function to sequence a document.

    Args:
        doc: OCR document

    Returns:
        List of ReadingOrder per page
    """
    sequencer = MultiColumnSequencer()
    return sequencer.sequence_document(doc)


def detect_column_layout(page: OCRPage) -> Tuple[LayoutType, int]:
    """Detect column layout for a page.

    Args:
        page: OCR page

    Returns:
        Tuple of (LayoutType, column_count)
    """
    sequencer = MultiColumnSequencer()
    order = sequencer.sequence_page(page)
    return order.layout_type, order.column_count
