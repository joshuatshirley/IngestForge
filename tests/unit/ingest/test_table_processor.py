"""Unit tests for TableProcessor."""


from ingestforge.ingest.ocr.spatial_parser import BoundingBox
from ingestforge.ingest.ocr.table_builder import Table, TableRow, TableCell
from ingestforge.ingest.html_table_extractor import (
    ExtractedTable,
)
from ingestforge.ingest.table_processor import (
    TableProcessor,
    ProcessedTable,
    process_table,
    table_to_chunk_metadata,
)


def create_ocr_table(rows_data: list, has_header: bool = True) -> Table:
    """Helper to create OCR Table for testing."""
    table = Table(has_header=has_header)

    for row_idx, cells_data in enumerate(rows_data):
        row = TableRow(row_index=row_idx)
        for col_idx, text in enumerate(cells_data):
            cell = TableCell(
                row=row_idx,
                col=col_idx,
                text=text,
                bbox=BoundingBox(
                    x1=col_idx * 100,
                    y1=row_idx * 30,
                    x2=(col_idx + 1) * 100,
                    y2=(row_idx + 1) * 30,
                ),
            )
            row.cells.append(cell)
        table.rows.append(row)

    return table


def create_html_table(
    headers: list, grid: list, caption: str = None, has_header: bool = True
) -> ExtractedTable:
    """Helper to create ExtractedTable for testing."""
    return ExtractedTable(
        headers=headers,
        grid=grid,
        caption=caption,
        has_header=has_header,
        row_count=len(grid) + (1 if headers else 0),
        col_count=len(headers) if headers else (len(grid[0]) if grid else 0),
    )


class TestProcessedTable:
    """Tests for ProcessedTable dataclass."""

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        processed = ProcessedTable(
            markdown="| A | B |\n|---|---|\n| 1 | 2 |",
            html="<table><tr><th>A</th><th>B</th></tr></table>",
            row_count=2,
            col_count=2,
            has_header=True,
            caption="Test table",
        )

        result = processed.to_dict()

        assert "table_markdown" in result
        assert "table_html" in result
        assert result["table_row_count"] == 2
        assert result["table_col_count"] == 2
        assert result["table_has_header"] is True
        assert result["table_caption"] == "Test table"


class TestTableProcessorOCR:
    """Tests for processing OCR tables."""

    def test_process_simple_table(self) -> None:
        """Test processing a simple OCR table."""
        processor = TableProcessor()
        table = create_ocr_table(
            [
                ["Name", "Age"],
                ["Alice", "30"],
                ["Bob", "25"],
            ]
        )

        result = processor.process_ocr_table(table)

        assert "Name" in result.markdown
        assert "Alice" in result.markdown
        assert "<table>" in result.html
        assert "<th>Name</th>" in result.html
        assert result.has_header is True

    def test_process_table_without_header(self) -> None:
        """Test processing table without header."""
        processor = TableProcessor()
        table = create_ocr_table(
            [
                ["A", "B"],
                ["C", "D"],
            ],
            has_header=False,
        )

        result = processor.process_ocr_table(table)

        assert result.has_header is False
        assert "<td>" in result.html  # No th tags

    def test_process_empty_table(self) -> None:
        """Test processing empty table."""
        processor = TableProcessor()
        table = Table()

        result = processor.process_ocr_table(table)

        assert result.markdown == ""
        assert result.html == ""

    def test_html_escaping(self) -> None:
        """Test that HTML special characters are escaped."""
        processor = TableProcessor()
        table = create_ocr_table(
            [
                ["<script>", "a & b"],
                ["x > y", '"quoted"'],
            ]
        )

        result = processor.process_ocr_table(table)

        assert "&lt;script&gt;" in result.html
        assert "&amp;" in result.html
        assert "&gt;" in result.html
        assert "&quot;" in result.html


class TestTableProcessorHTML:
    """Tests for processing HTML-extracted tables."""

    def test_process_html_table(self) -> None:
        """Test processing ExtractedTable."""
        processor = TableProcessor()
        table = create_html_table(
            headers=["Col1", "Col2"],
            grid=[["A", "B"], ["C", "D"]],
        )

        result = processor.process_html_table(table)

        assert "Col1" in result.markdown
        assert "<thead>" in result.html
        assert "<th>Col1</th>" in result.html

    def test_process_with_caption(self) -> None:
        """Test processing table with caption."""
        processor = TableProcessor(include_caption=True)
        table = create_html_table(
            headers=["X", "Y"],
            grid=[["1", "2"]],
            caption="Test Caption",
        )

        result = processor.process_html_table(table)

        assert result.caption == "Test Caption"
        assert "<caption>Test Caption</caption>" in result.html

    def test_caption_disabled(self) -> None:
        """Test that caption can be disabled."""
        processor = TableProcessor(include_caption=False)
        table = create_html_table(
            headers=["X", "Y"],
            grid=[["1", "2"]],
            caption="Test Caption",
        )

        result = processor.process_html_table(table)

        assert result.caption is None
        assert "<caption>" not in result.html


class TestConvenienceFunctions:
    """Tests for standalone convenience functions."""

    def test_process_table_function(self) -> None:
        """Test the process_table convenience function."""
        table = create_ocr_table(
            [
                ["A", "B"],
                ["1", "2"],
            ]
        )

        markdown, html = process_table(table)

        assert "A" in markdown
        assert "<table>" in html

    def test_table_to_chunk_metadata(self) -> None:
        """Test converting table to chunk metadata."""
        table = create_ocr_table(
            [
                ["Header"],
                ["Data"],
            ]
        )

        metadata = table_to_chunk_metadata(table, page_number=5)

        assert "table_markdown" in metadata
        assert "table_html" in metadata
        assert metadata["page_number"] == 5
        assert metadata["element_type"] == "Table"
