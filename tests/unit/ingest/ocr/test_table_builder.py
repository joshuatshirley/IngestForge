"""Tests for table reconstruction from OCR output.

Tests table detection, cell assignment, and Markdown export."""

from __future__ import annotations


from ingestforge.ingest.ocr.spatial_parser import (
    BoundingBox,
    ElementType,
    OCRElement,
    OCRPage,
)
from ingestforge.ingest.ocr.table_builder import (
    CellType,
    Table,
    TableBuilder,
    TableBuilderConfig,
    TableCell,
    TableRow,
    find_tables_on_page,
    table_to_markdown,
)

# TableCell tests


class TestTableCell:
    """Tests for TableCell dataclass."""

    def test_cell_creation(self) -> None:
        """Test creating a cell."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        cell = TableCell(row=0, col=0, text="Data", bbox=bbox)

        assert cell.row == 0
        assert cell.col == 0
        assert cell.text == "Data"
        assert cell.cell_type == CellType.DATA

    def test_is_empty_with_content(self) -> None:
        """Test is_empty with content."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        cell = TableCell(row=0, col=0, text="value", bbox=bbox)

        assert cell.is_empty is False

    def test_is_empty_without_content(self) -> None:
        """Test is_empty without content."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        cell = TableCell(row=0, col=0, text="   ", bbox=bbox)

        assert cell.is_empty is True


# TableRow tests


class TestTableRow:
    """Tests for TableRow dataclass."""

    def test_row_creation(self) -> None:
        """Test creating a row."""
        row = TableRow(row_index=0, y_start=100, y_end=150)

        assert row.row_index == 0
        assert row.height == 50

    def test_row_with_cells(self) -> None:
        """Test row with cells."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        cells = [
            TableCell(row=0, col=0, text="A", bbox=bbox),
            TableCell(row=0, col=1, text="B", bbox=bbox),
        ]
        row = TableRow(row_index=0, cells=cells)

        assert len(row.cells) == 2


# Table tests


class TestTable:
    """Tests for Table dataclass."""

    def test_empty_table(self) -> None:
        """Test empty table."""
        table = Table()

        assert table.row_count == 0
        assert table.col_count == 0

    def test_table_dimensions(self) -> None:
        """Test table row and column count."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table()

        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row=0, col=0, text="A", bbox=bbox),
                TableCell(row=0, col=1, text="B", bbox=bbox),
            ],
        )
        row2 = TableRow(
            row_index=1,
            cells=[
                TableCell(row=1, col=0, text="C", bbox=bbox),
                TableCell(row=1, col=1, text="D", bbox=bbox),
            ],
        )
        table.rows = [row1, row2]

        assert table.row_count == 2
        assert table.col_count == 2

    def test_get_cell(self) -> None:
        """Test getting cell by position."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table()

        cell_a = TableCell(row=0, col=0, text="A", bbox=bbox)
        cell_b = TableCell(row=0, col=1, text="B", bbox=bbox)
        row = TableRow(row_index=0, cells=[cell_a, cell_b])
        table.rows = [row]

        assert table.get_cell(0, 0) == cell_a
        assert table.get_cell(0, 1) == cell_b
        assert table.get_cell(5, 5) is None

    def test_to_markdown_simple(self) -> None:
        """Test simple Markdown export."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table()

        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row=0, col=0, text="A", bbox=bbox),
                TableCell(row=0, col=1, text="B", bbox=bbox),
            ],
        )
        row2 = TableRow(
            row_index=1,
            cells=[
                TableCell(row=1, col=0, text="C", bbox=bbox),
                TableCell(row=1, col=1, text="D", bbox=bbox),
            ],
        )
        table.rows = [row1, row2]

        md = table.to_markdown()

        assert "| A | B |" in md
        assert "| C | D |" in md

    def test_to_markdown_with_header(self) -> None:
        """Test Markdown with header row."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table(has_header=True)

        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row=0, col=0, text="Name", bbox=bbox),
                TableCell(row=0, col=1, text="Value", bbox=bbox),
            ],
        )
        row2 = TableRow(
            row_index=1,
            cells=[
                TableCell(row=1, col=0, text="A", bbox=bbox),
                TableCell(row=1, col=1, text="1", bbox=bbox),
            ],
        )
        table.rows = [row1, row2]

        md = table.to_markdown()

        assert "| Name | Value |" in md
        assert "| --- | --- |" in md
        assert "| A | 1 |" in md


# TableBuilder tests


class TestTableBuilder:
    """Tests for TableBuilder."""

    def test_builder_creation(self) -> None:
        """Test creating builder."""
        builder = TableBuilder()
        assert builder.config is not None

    def test_builder_with_config(self) -> None:
        """Test builder with custom config."""
        config = TableBuilderConfig(min_cells=6)
        builder = TableBuilder(config=config)

        assert builder.config.min_cells == 6

    def test_find_tables_empty_page(self) -> None:
        """Test finding tables on empty page."""
        page = OCRPage(page_number=1, width=612, height=792)
        builder = TableBuilder()

        tables = builder.find_tables(page)

        assert len(tables) == 0

    def test_find_tables_not_enough_cells(self) -> None:
        """Test with too few cells."""
        page = OCRPage(page_number=1, width=612, height=792)
        bbox = BoundingBox(x1=50, y1=100, x2=150, y2=130)
        page.elements = [
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="Only one"),
        ]

        builder = TableBuilder()
        tables = builder.find_tables(page)

        assert len(tables) == 0

    def test_find_simple_table(self) -> None:
        """Test finding a simple table."""
        page = OCRPage(page_number=1, width=612, height=792)

        # Create 2x2 grid of elements
        elements = []
        for row in range(2):
            for col in range(2):
                bbox = BoundingBox(
                    x1=100 + col * 150,
                    y1=100 + row * 50,
                    x2=200 + col * 150,
                    y2=140 + row * 50,
                )
                elem = OCRElement(
                    element_type=ElementType.BLOCK,
                    bbox=bbox,
                    text=f"R{row}C{col}",
                )
                elements.append(elem)

        page.elements = elements
        builder = TableBuilder()
        tables = builder.find_tables(page)

        assert len(tables) == 1
        assert tables[0].row_count >= 2


class TestTableBuilderHelpers:
    """Tests for TableBuilder helper methods."""

    def test_group_by_y_alignment(self) -> None:
        """Test grouping elements by y position."""
        builder = TableBuilder()

        elements = []
        for y in [100, 100, 200, 200]:
            bbox = BoundingBox(x1=50, y1=y, x2=150, y2=y + 30)
            elements.append(
                OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="test")
            )

        groups = builder._group_by_y_alignment(elements)

        assert len(groups) == 2
        assert len(groups[0]) == 2
        assert len(groups[1]) == 2

    def test_find_column_positions(self) -> None:
        """Test finding column x positions."""
        builder = TableBuilder()

        elements = []
        for x in [100, 100, 300, 300]:
            bbox = BoundingBox(x1=x, y1=100, x2=x + 80, y2=140)
            elements.append(
                OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="test")
            )

        positions = builder._find_column_positions(elements)

        assert len(positions) == 2
        assert 100 in positions
        assert 300 in positions

    def test_calculate_col_widths(self) -> None:
        """Test column width calculation."""
        builder = TableBuilder()

        positions = [100, 250, 400]
        widths = builder._calculate_col_widths(positions)

        assert widths[0] == 150  # 250 - 100
        assert widths[1] == 150  # 400 - 250
        assert widths[2] == 100  # default for last column


class TestHeaderDetection:
    """Tests for table header detection."""

    def test_detect_header_no_rows(self) -> None:
        """Test header detection with no rows."""
        table = Table()
        builder = TableBuilder()

        assert builder._detect_header_row(table) is False

    def test_detect_header_single_row(self) -> None:
        """Test header detection with single row."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table()
        row = TableRow(
            row_index=0,
            cells=[
                TableCell(row=0, col=0, text="A", bbox=bbox),
            ],
        )
        table.rows = [row]

        builder = TableBuilder()
        assert builder._detect_header_row(table) is False

    def test_detect_header_by_empty_cells(self) -> None:
        """Test header detection by empty cell pattern."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table()

        # Header row - no empty cells
        row1 = TableRow(
            row_index=0,
            cells=[
                TableCell(row=0, col=0, text="Name", bbox=bbox),
                TableCell(row=0, col=1, text="Value", bbox=bbox),
            ],
            y_start=100,
            y_end=150,
        )

        # Data row - has empty cell
        row2 = TableRow(
            row_index=1,
            cells=[
                TableCell(row=1, col=0, text="A", bbox=bbox),
                TableCell(row=1, col=1, text="", bbox=bbox, cell_type=CellType.EMPTY),
            ],
            y_start=160,
            y_end=210,
        )

        table.rows = [row1, row2]

        builder = TableBuilder()
        assert builder._detect_header_row(table) is True


# Convenience function tests


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_find_tables_on_page(self) -> None:
        """Test find_tables_on_page function."""
        page = OCRPage(page_number=1, width=612, height=792)

        tables = find_tables_on_page(page)

        assert isinstance(tables, list)

    def test_table_to_markdown(self) -> None:
        """Test table_to_markdown function."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        table = Table()

        row = TableRow(
            row_index=0,
            cells=[
                TableCell(row=0, col=0, text="Test", bbox=bbox),
            ],
        )
        table.rows = [row]

        md = table_to_markdown(table)

        assert "| Test |" in md

    def test_table_to_markdown_empty(self) -> None:
        """Test table_to_markdown with empty table."""
        table = Table()

        md = table_to_markdown(table)

        assert md == ""
