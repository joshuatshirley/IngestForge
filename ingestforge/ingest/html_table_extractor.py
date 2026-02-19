"""
HTML table extraction and preservation.

Extracts tables from HTML documents and preserves their structure
for analysis, conversion, and export.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from html.parser import HTMLParser


class TableOutputFormat(Enum):
    """Output formats for extracted tables."""

    DICT = "dict"  # List of dictionaries (row-oriented)
    LIST = "list"  # List of lists (2D array)
    CSV = "csv"  # CSV string
    MARKDOWN = "markdown"  # Markdown table
    JSON = "json"  # JSON string
    TSV = "tsv"  # Tab-separated values
    HTML = "html"  # Clean HTML table


class CellType(Enum):
    """Type of table cell."""

    HEADER = "header"
    DATA = "data"


@dataclass
class TableCell:
    """A single cell in a table."""

    content: str
    cell_type: CellType = CellType.DATA
    rowspan: int = 1
    colspan: int = 1
    row_index: int = 0
    col_index: int = 0
    # Style/formatting hints
    is_numeric: bool = False
    is_empty: bool = False
    alignment: Optional[str] = None  # left, center, right
    raw_html: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "cell_type": self.cell_type.value,
            "rowspan": self.rowspan,
            "colspan": self.colspan,
            "row_index": self.row_index,
            "col_index": self.col_index,
            "is_numeric": self.is_numeric,
            "is_empty": self.is_empty,
            "alignment": self.alignment,
        }


@dataclass
class TableRow:
    """A row in a table."""

    cells: List[TableCell] = field(default_factory=list)
    is_header_row: bool = False
    row_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cells": [c.to_dict() for c in self.cells],
            "is_header_row": self.is_header_row,
            "row_index": self.row_index,
        }


@dataclass
class ExtractedTable:
    """A fully extracted table with metadata."""

    rows: List[TableRow] = field(default_factory=list)
    caption: Optional[str] = None
    # Normalized grid (handles rowspan/colspan)
    grid: List[List[str]] = field(default_factory=list)
    # Headers (first row or thead)
    headers: List[str] = field(default_factory=list)
    # Dimensions
    row_count: int = 0
    col_count: int = 0
    # Metadata
    table_index: int = 0
    id: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    summary: Optional[str] = None
    # Quality indicators
    has_header: bool = False
    is_regular: bool = True  # All rows same column count
    has_merged_cells: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": [r.to_dict() for r in self.rows],
            "caption": self.caption,
            "grid": self.grid,
            "headers": self.headers,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "table_index": self.table_index,
            "id": self.id,
            "classes": self.classes,
            "summary": self.summary,
            "has_header": self.has_header,
            "is_regular": self.is_regular,
            "has_merged_cells": self.has_merged_cells,
        }

    def to_markdown(self) -> str:
        """Convert to Markdown table."""
        if not self.grid and not self.headers:
            return ""

        lines = []

        # Headers
        if self.headers:
            headers = self.headers
        elif self.grid:
            headers = self.grid[0]
            self.grid = self.grid[1:]
        else:
            return ""

        # Calculate column widths
        widths = [len(h) for h in headers]
        for row in self.grid:
            for i, cell in enumerate(row):
                if i < len(widths):
                    widths[i] = max(widths[i], len(cell))

        # Header row
        header_cells = [
            h.ljust(widths[i]) if i < len(widths) else h for i, h in enumerate(headers)
        ]
        lines.append("| " + " | ".join(header_cells) + " |")

        # Separator
        separators = ["-" * w for w in widths]
        lines.append("| " + " | ".join(separators) + " |")

        # Data rows
        for row in self.grid:
            cells = []
            for i, cell in enumerate(row):
                if i < len(widths):
                    cells.append(cell.ljust(widths[i]))
                else:
                    cells.append(cell)
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)


class TableHTMLParser(HTMLParser):
    """HTML parser for extracting tables."""

    def __init__(self) -> None:
        super().__init__()
        self.tables: List[ExtractedTable] = []
        self.current_table: Optional[ExtractedTable] = None
        self.current_row: Optional[TableRow] = None
        self.current_cell: Optional[TableCell] = None
        self.in_table = False
        self.in_thead = False
        self.in_tbody = False
        self.in_caption = False
        self.cell_content: List[str] = []
        self.table_index = 0
        self.row_index = 0
        self.col_index = 0
        # Stack for nested table support
        self._table_stack: List[
            Tuple[
                Optional[ExtractedTable],
                Optional[TableRow],
                Optional[TableCell],
                bool,
                bool,
                bool,
                bool,
                List[str],
                int,
                int,
            ]
        ] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        """
        Handle opening HTML tags.

        Rule #1: Early returns eliminate nesting (no elif chains)
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        attrs_dict = dict(attrs)
        if tag == "table":
            self._handle_table_start(attrs_dict)
            return

        if tag == "caption" and self.in_table:
            self._handle_caption_start()
            return

        if tag == "thead":
            self.in_thead = True
            return

        if tag == "tbody":
            self.in_tbody = True
            return

        if tag == "tr" and self.in_table:
            self._handle_row_start()
            return

        if tag in ("th", "td") and self.current_row is not None:
            self._handle_cell_start(tag, attrs_dict)
            return

    def _handle_table_start(self, attrs_dict: Dict[str, Optional[str]]) -> None:
        """Handle table opening tag."""
        self._push_table_state_if_nested()

        class_attr = attrs_dict.get("class")
        self.current_table = ExtractedTable(
            table_index=self.table_index,
            id=attrs_dict.get("id"),
            classes=class_attr.split() if class_attr else [],
            summary=attrs_dict.get("summary"),
        )
        self.in_table = True
        self.row_index = 0
        self.table_index += 1

    def _push_table_state_if_nested(self) -> None:
        """Push current table state onto stack for nested table support."""
        if self.current_table is None:
            return

        self._table_stack.append(
            (
                self.current_table,
                self.current_row,
                self.current_cell,
                self.in_table,
                self.in_thead,
                self.in_tbody,
                self.in_caption,
                self.cell_content,
                self.row_index,
                self.col_index,
            )
        )
        self.current_row = None
        self.current_cell = None
        self.in_thead = False
        self.in_tbody = False
        self.in_caption = False
        self.cell_content = []

    def _handle_caption_start(self) -> None:
        """Handle caption opening tag."""
        self.in_caption = True
        self.cell_content = []

    def _handle_row_start(self) -> None:
        """Handle row opening tag."""
        self.current_row = TableRow(
            row_index=self.row_index,
            is_header_row=self.in_thead,
        )
        self.col_index = 0
        self.row_index += 1

    def _handle_cell_start(
        self, tag: str, attrs_dict: Dict[str, Optional[str]]
    ) -> None:
        """Handle cell (th/td) opening tag."""
        cell_type = CellType.HEADER if tag == "th" else CellType.DATA
        rowspan = int(attrs_dict.get("rowspan") or 1)
        colspan = int(attrs_dict.get("colspan") or 1)
        alignment = self._extract_cell_alignment(attrs_dict)

        if self.current_row is None:
            return

        self.current_cell = TableCell(
            content="",
            cell_type=cell_type,
            rowspan=rowspan,
            colspan=colspan,
            row_index=self.current_row.row_index,
            col_index=self.col_index,
            alignment=alignment,
        )
        self.cell_content = []

        self._mark_merged_cells_if_needed(rowspan, colspan)

    def _extract_cell_alignment(
        self, attrs_dict: Dict[str, Optional[str]]
    ) -> Optional[str]:
        """Extract text alignment from cell attributes."""
        if "align" in attrs_dict:
            return attrs_dict["align"]

        if "style" not in attrs_dict:
            return None

        style = attrs_dict["style"]
        if not style or "text-align:" not in style:
            return None

        match = re.search(r"text-align:\s*(left|center|right)", style)
        return match.group(1) if match else None

    def _mark_merged_cells_if_needed(self, rowspan: int, colspan: int) -> None:
        """Mark table as having merged cells if rowspan or colspan > 1."""
        if (rowspan > 1 or colspan > 1) and self.current_table:
            self.current_table.has_merged_cells = True

    def _handle_table_end(self) -> None:
        """
        Handle table closing tag.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if self.current_table is None:
            return

        self._finalize_table()
        self.tables.append(self.current_table)

        # Pop parent table state from stack if nested
        if self._table_stack:
            (
                self.current_table,
                self.current_row,
                self.current_cell,
                self.in_table,
                self.in_thead,
                self.in_tbody,
                self.in_caption,
                self.cell_content,
                self.row_index,
                self.col_index,
            ) = self._table_stack.pop()
            return

        # No nested table - clear state
        self.current_table = None
        self.in_table = False

    def _handle_caption_end(self) -> None:
        """
        Handle caption closing tag.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if self.in_caption and self.current_table:
            self.current_table.caption = "".join(self.cell_content).strip()

        self.in_caption = False

    def _handle_thead_end(self) -> None:
        """
        Handle thead closing tag.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.in_thead = False

    def _handle_tbody_end(self) -> None:
        """
        Handle tbody closing tag.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        self.in_tbody = False

    def _handle_row_end(self) -> None:
        """
        Handle row closing tag.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if self.current_row is None:
            return

        # Add row to table if available
        if self.current_table:
            self.current_table.rows.append(self.current_row)

        self.current_row = None

    def _handle_cell_end(self) -> None:
        """
        Handle cell closing tag (th or td).

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if self.current_cell is None:
            return

        # Extract and normalize content
        content = "".join(self.cell_content).strip()
        content = " ".join(content.split())  # Normalize whitespace

        # Update cell properties
        self.current_cell.content = content
        self.current_cell.is_empty = len(content) == 0
        self.current_cell.is_numeric = self._is_numeric(content)

        # Add cell to current row if available
        if self.current_row:
            self.current_row.cells.append(self.current_cell)

        # Advance column index
        self.col_index += self.current_cell.colspan
        self.current_cell = None

    def handle_endtag(self, tag: str) -> None:
        """
        Handle closing HTML tags.

        Rule #1: Early returns eliminate nesting (no elif chains)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Follows same pattern as handle_starttag for consistency.
        """
        if tag == "table":
            self._handle_table_end()
            return

        if tag == "caption":
            self._handle_caption_end()
            return

        if tag == "thead":
            self._handle_thead_end()
            return

        if tag == "tbody":
            self._handle_tbody_end()
            return

        if tag == "tr":
            self._handle_row_end()
            return

        if tag in ("th", "td"):
            self._handle_cell_end()
            return

    def handle_data(self, data: str) -> None:
        if self.in_caption or self.current_cell is not None:
            self.cell_content.append(data)

    def _is_numeric(self, text: str) -> bool:
        """Check if text represents a numeric value."""
        if not text:
            return False
        # Remove common numeric formatting
        cleaned = text.replace(",", "").replace("$", "").replace("%", "")
        cleaned = cleaned.replace("(", "-").replace(")", "")
        try:
            float(cleaned)
            return True
        except ValueError:
            return False

    def _fill_cell_span(
        self,
        grid: List[List[Optional[str]]],
        cell: TableCell,
        row_idx: int,
        col_idx: int,
        max_cols: int,
        row_count: int,
    ) -> None:
        """
        Fill grid cells for rowspan/colspan.

        Rule #1: Reduced nesting (max 3 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            grid: 2D grid to fill
            cell: Cell to place
            row_idx: Starting row index
            col_idx: Starting column index
            max_cols: Maximum columns
            row_count: Total row count
        """
        for r in range(cell.rowspan):
            for c in range(cell.colspan):
                target_row = row_idx + r
                target_col = col_idx + c
                if target_row < row_count and target_col < max_cols:
                    grid[target_row][target_col] = cell.content

    def _find_next_available_column(
        self,
        grid: List[List[Optional[str]]],
        row_idx: int,
        start_col: int,
        max_cols: int,
    ) -> int:
        """
        Find next available column in grid.

        Rule #1: Simple loop with early return
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            grid: 2D grid
            row_idx: Current row index
            start_col: Starting column to search from
            max_cols: Maximum columns

        Returns:
            Next available column index
        """
        col_idx = start_col
        while col_idx < max_cols and grid[row_idx][col_idx] is not None:
            col_idx += 1
        return col_idx

    def _build_table_grid(self, table: ExtractedTable) -> List[List[Optional[str]]]:
        """
        Build normalized grid handling rowspan/colspan.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            table: Table to build grid for

        Returns:
            2D grid with cell content
        """
        grid: List[List[Optional[str]]] = [
            [None] * table.col_count for _ in range(table.row_count)
        ]

        for row_idx, row in enumerate(table.rows):
            col_idx = 0
            for cell in row.cells:
                col_idx = self._find_next_available_column(
                    grid, row_idx, col_idx, table.col_count
                )
                if col_idx >= table.col_count:
                    break

                # Fill cells for rowspan/colspan
                self._fill_cell_span(
                    grid, cell, row_idx, col_idx, table.col_count, table.row_count
                )

                col_idx += cell.colspan

        return grid

    def _identify_headers(self, table: ExtractedTable) -> None:
        """
        Identify and extract table headers.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            table: Table to process
        """
        if not table.rows:
            return

        first_row = table.rows[0]
        if not (
            first_row.is_header_row
            or all(c.cell_type == CellType.HEADER for c in first_row.cells)
        ):
            return

        table.has_header = True
        if not table.grid:
            return

        table.headers = table.grid[0]
        table.grid = table.grid[1:]

    def _finalize_table(self) -> None:
        """
        Finalize table structure, build grid, identify headers.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints
        """
        if not self.current_table or not self.current_table.rows:
            return

        table = self.current_table

        # Determine column count
        max_cols = 0
        for row in table.rows:
            col_count = sum(cell.colspan for cell in row.cells)
            max_cols = max(max_cols, col_count)

        table.col_count = max_cols
        table.row_count = len(table.rows)

        # Build normalized grid
        grid = self._build_table_grid(table)

        # Convert None to empty string
        table.grid = [[c if c is not None else "" for c in row] for row in grid]

        # Check regularity
        row_lengths = set(len(row) for row in table.grid)
        table.is_regular = len(row_lengths) <= 1

        # Identify headers
        self._identify_headers(table)


class HTMLTableExtractor:
    """
    Extract and process tables from HTML documents.
    """

    def __init__(self) -> None:
        pass

    def extract(self, html: str) -> List[ExtractedTable]:
        """
        Extract all tables from HTML content.

        Args:
            html: HTML string

        Returns:
            List of ExtractedTable objects
        """
        parser = TableHTMLParser()
        parser.feed(html)
        return parser.tables
