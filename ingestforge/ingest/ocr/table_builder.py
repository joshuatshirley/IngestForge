"""Table Reconstruction Engine for OCR output.

Detects and reconstructs tables from OCR spatial data
by analyzing element alignment and grid patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ingestforge.ingest.ocr.spatial_parser import (
    BoundingBox,
    OCRElement,
    OCRPage,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_ROWS = 500
MAX_COLS = 50
MAX_CELLS = 5000
ALIGNMENT_TOLERANCE = 15  # Pixels for alignment detection
MIN_TABLE_CELLS = 4  # Minimum cells to consider a table


class CellType(str, Enum):
    """Types of table cells."""

    HEADER = "header"
    DATA = "data"
    EMPTY = "empty"
    MERGED = "merged"


@dataclass
class TableCell:
    """A single table cell."""

    row: int
    col: int
    text: str
    bbox: BoundingBox
    cell_type: CellType = CellType.DATA
    row_span: int = 1
    col_span: int = 1

    @property
    def is_empty(self) -> bool:
        """Check if cell has no content."""
        return not self.text.strip()


@dataclass
class TableRow:
    """A row of table cells."""

    row_index: int
    cells: List[TableCell] = field(default_factory=list)
    y_start: int = 0
    y_end: int = 0

    @property
    def height(self) -> int:
        """Get row height."""
        return self.y_end - self.y_start


@dataclass
class Table:
    """A reconstructed table."""

    rows: List[TableRow] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    has_header: bool = False
    column_widths: List[int] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        """Get number of rows."""
        return len(self.rows)

    @property
    def col_count(self) -> int:
        """Get number of columns."""
        if not self.rows:
            return 0
        return max(len(row.cells) for row in self.rows)

    def get_cell(self, row: int, col: int) -> Optional[TableCell]:
        """Get cell at position."""
        if row < 0 or row >= len(self.rows):
            return None
        cells = self.rows[row].cells
        if col < 0 or col >= len(cells):
            return None
        return cells[col]

    def to_markdown(self) -> str:
        """Convert table to Markdown format."""
        if not self.rows:
            return ""

        lines = []
        col_count = self.col_count

        for idx, row in enumerate(self.rows):
            cells = [c.text.strip() for c in row.cells]
            # Pad row to match column count
            while len(cells) < col_count:
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")

            # Add header separator after first row if header
            if idx == 0 and self.has_header:
                lines.append("| " + " | ".join(["---"] * col_count) + " |")

        return "\n".join(lines)


@dataclass
class TableBuilderConfig:
    """Configuration for table detection."""

    alignment_tolerance: int = ALIGNMENT_TOLERANCE
    min_cells: int = MIN_TABLE_CELLS
    detect_headers: bool = True
    merge_aligned_cells: bool = True


class TableBuilder:
    """Reconstructs tables from OCR elements.

    Uses spatial analysis to detect grid patterns
    and reconstruct table structure.
    """

    def __init__(self, config: Optional[TableBuilderConfig] = None) -> None:
        """Initialize builder.

        Args:
            config: Builder configuration
        """
        self.config = config or TableBuilderConfig()

    def find_tables(self, page: OCRPage) -> List[Table]:
        """Find and reconstruct tables on a page.

        Args:
            page: OCR page with elements

        Returns:
            List of detected tables
        """
        if not page.elements:
            return []

        # Get candidate elements
        elements = self._get_table_candidates(page.elements)
        if len(elements) < self.config.min_cells:
            return []

        # Find grid patterns
        tables = self._detect_grid_patterns(elements)

        return tables

    def _get_table_candidates(self, elements: List[OCRElement]) -> List[OCRElement]:
        """Get elements that could be table cells.

        Args:
            elements: All page elements

        Returns:
            Filtered candidate elements
        """
        candidates = []
        for elem in elements:
            if not elem.bbox:
                continue
            # Skip very large elements (likely paragraphs)
            if elem.bbox.width > 500 and elem.bbox.height > 100:
                continue
            candidates.append(elem)

        return candidates[:MAX_CELLS]

    def _detect_grid_patterns(self, elements: List[OCRElement]) -> List[Table]:
        """Detect grid patterns in elements.

        Args:
            elements: Candidate elements

        Returns:
            List of detected tables
        """
        # Find horizontal alignments (rows)
        row_groups = self._group_by_y_alignment(elements)
        if len(row_groups) < 2:
            return []

        # Find vertical alignments (columns)
        col_positions = self._find_column_positions(elements)
        if len(col_positions) < 2:
            return []

        # Build table from grid
        table = self._build_table_from_grid(row_groups, col_positions)
        if not table or table.row_count < 2:
            return []

        # Detect header row
        if self.config.detect_headers:
            table.has_header = self._detect_header_row(table)

        return [table]

    def _group_by_y_alignment(
        self, elements: List[OCRElement]
    ) -> List[List[OCRElement]]:
        """Group elements by vertical alignment.

        Args:
            elements: Elements to group

        Returns:
            List of element groups (rows)
        """
        if not elements:
            return []

        # Sort by y coordinate
        sorted_elems = sorted(elements, key=lambda e: e.bbox.y1 if e.bbox else 0)

        groups: List[List[OCRElement]] = []
        current_group: List[OCRElement] = []
        current_y = -1000

        for elem in sorted_elems:
            if not elem.bbox:
                continue

            y = elem.bbox.y1
            if abs(y - current_y) <= self.config.alignment_tolerance:
                current_group.append(elem)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [elem]
                current_y = y

        if current_group:
            groups.append(current_group)

        return groups[:MAX_ROWS]

    def _find_column_positions(self, elements: List[OCRElement]) -> List[int]:
        """Find column x positions.

        Args:
            elements: Elements to analyze

        Returns:
            Sorted list of column x positions
        """
        x_positions: List[int] = []

        for elem in elements:
            if not elem.bbox:
                continue
            x = elem.bbox.x1
            # Check if near existing position
            found = False
            for existing in x_positions:
                if abs(x - existing) <= self.config.alignment_tolerance:
                    found = True
                    break
            if not found:
                x_positions.append(x)

        return sorted(x_positions)[:MAX_COLS]

    def _build_table_from_grid(
        self,
        row_groups: List[List[OCRElement]],
        col_positions: List[int],
    ) -> Optional[Table]:
        """Build table from detected grid.

        Args:
            row_groups: Elements grouped by row
            col_positions: Column x positions

        Returns:
            Reconstructed Table or None
        """
        if not row_groups or not col_positions:
            return None

        table = Table()
        table.column_widths = self._calculate_col_widths(col_positions)

        for row_idx, elements in enumerate(row_groups):
            table_row = self._build_row(row_idx, elements, col_positions)
            if table_row.cells:
                table.rows.append(table_row)

        # Calculate table bbox
        table.bbox = self._calculate_table_bbox(table)

        return table

    def _build_row(
        self,
        row_idx: int,
        elements: List[OCRElement],
        col_positions: List[int],
    ) -> TableRow:
        """Build a table row from elements.

        Args:
            row_idx: Row index
            elements: Elements in this row
            col_positions: Column x positions

        Returns:
            TableRow
        """
        row = TableRow(row_index=row_idx)

        # Sort elements by x position
        sorted_elems = sorted(elements, key=lambda e: e.bbox.x1 if e.bbox else 0)

        # Assign elements to columns
        for col_idx, col_x in enumerate(col_positions):
            cell = self._find_cell_at_column(sorted_elems, col_x, row_idx, col_idx)
            row.cells.append(cell)

        # Set row bounds
        if sorted_elems and sorted_elems[0].bbox:
            row.y_start = min(e.bbox.y1 for e in sorted_elems if e.bbox)
            row.y_end = max(e.bbox.y2 for e in sorted_elems if e.bbox)

        return row

    def _find_cell_at_column(
        self,
        elements: List[OCRElement],
        col_x: int,
        row_idx: int,
        col_idx: int,
    ) -> TableCell:
        """Find or create cell at column position.

        Args:
            elements: Row elements
            col_x: Column x position
            row_idx: Row index
            col_idx: Column index

        Returns:
            TableCell
        """
        # Find element at this column
        for elem in elements:
            if not elem.bbox:
                continue
            if abs(elem.bbox.x1 - col_x) <= self.config.alignment_tolerance:
                return TableCell(
                    row=row_idx,
                    col=col_idx,
                    text=elem.text,
                    bbox=elem.bbox,
                )

        # Create empty cell
        return TableCell(
            row=row_idx,
            col=col_idx,
            text="",
            bbox=BoundingBox(x1=col_x, y1=0, x2=col_x + 50, y2=20),
            cell_type=CellType.EMPTY,
        )

    def _calculate_col_widths(self, col_positions: List[int]) -> List[int]:
        """Calculate column widths from positions.

        Args:
            col_positions: Column x positions

        Returns:
            List of column widths
        """
        widths = []
        for i in range(len(col_positions) - 1):
            widths.append(col_positions[i + 1] - col_positions[i])
        # Last column gets default width
        if col_positions:
            widths.append(100)
        return widths

    def _calculate_table_bbox(self, table: Table) -> Optional[BoundingBox]:
        """Calculate bounding box for entire table.

        Args:
            table: Table to analyze

        Returns:
            BoundingBox or None
        """
        if not table.rows:
            return None

        x1 = y1 = 999999
        x2 = y2 = 0

        for row in table.rows:
            for cell in row.cells:
                if cell.bbox:
                    x1 = min(x1, cell.bbox.x1)
                    y1 = min(y1, cell.bbox.y1)
                    x2 = max(x2, cell.bbox.x2)
                    y2 = max(y2, cell.bbox.y2)

        if x1 == 999999:
            return None

        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def _detect_header_row(self, table: Table) -> bool:
        """Detect if first row is a header.

        Args:
            table: Table to analyze

        Returns:
            True if header detected
        """
        if not table.rows or len(table.rows) < 2:
            return False

        first_row = table.rows[0]
        second_row = table.rows[1]

        # Header if first row has no empty cells but data rows do
        first_empty = sum(1 for c in first_row.cells if c.is_empty)
        second_empty = sum(1 for c in second_row.cells if c.is_empty)

        if first_empty == 0 and second_empty > 0:
            return True

        # Header if first row is shorter (smaller font)
        if first_row.height < second_row.height * 0.9:
            return True

        return False


def find_tables_on_page(page: OCRPage) -> List[Table]:
    """Convenience function to find tables.

    Args:
        page: OCR page

    Returns:
        List of detected tables
    """
    builder = TableBuilder()
    return builder.find_tables(page)


def table_to_markdown(table: Table) -> str:
    """Convert table to Markdown.

    Args:
        table: Table to convert

    Returns:
        Markdown string
    """
    return table.to_markdown()
