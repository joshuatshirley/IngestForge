"""
Tests for HTML table extraction.

This module tests extraction and formatting of tables from HTML documents.

Test Strategy
-------------
- Test table detection and extraction
- Test markdown output format
- Test edge cases (empty tables, merged cells, nested tables)

Organization
------------
- TestTableExtraction: Basic table extraction
- TestMarkdownFormat: Markdown output formatting
- TestEdgeCases: Edge cases and error handling
"""


from ingestforge.ingest.html_table_extractor import (
    HTMLTableExtractor,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestTableExtraction:
    """Tests for basic table extraction.

    Rule #4: Focused test class - tests extraction only
    """

    def test_extract_simple_table(self):
        """Test extracting a simple table with headers and data."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <thead>
                <tr><th>Name</th><th>Age</th></tr>
            </thead>
            <tbody>
                <tr><td>Alice</td><td>30</td></tr>
                <tr><td>Bob</td><td>25</td></tr>
            </tbody>
        </table>
        """

        tables = extractor.extract(html)

        assert len(tables) == 1
        table = tables[0]
        assert table.has_header is True
        assert table.headers == ["Name", "Age"]
        assert len(table.grid) == 2
        assert table.grid[0] == ["Alice", "30"]
        assert table.grid[1] == ["Bob", "25"]

    def test_extract_table_without_thead(self):
        """Test extracting a table without explicit thead."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><th>Product</th><th>Price</th></tr>
            <tr><td>Widget</td><td>$9.99</td></tr>
        </table>
        """

        tables = extractor.extract(html)

        assert len(tables) == 1
        table = tables[0]
        assert table.has_header is True
        assert table.headers == ["Product", "Price"]

    def test_extract_multiple_tables(self):
        """Test extracting multiple tables from one document."""
        extractor = HTMLTableExtractor()

        html = """
        <table id="table1">
            <tr><td>A</td></tr>
        </table>
        <table id="table2">
            <tr><td>B</td></tr>
        </table>
        """

        tables = extractor.extract(html)

        assert len(tables) == 2
        assert tables[0].id == "table1"
        assert tables[1].id == "table2"

    def test_extract_table_with_caption(self):
        """Test extracting table with caption."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <caption>Sales Data 2024</caption>
            <tr><td>Q1</td><td>$100</td></tr>
        </table>
        """

        tables = extractor.extract(html)

        assert len(tables) == 1
        assert tables[0].caption == "Sales Data 2024"


class TestMarkdownFormat:
    """Tests for markdown output formatting.

    Rule #4: Focused test class - tests markdown format only

    These tests verify the current markdown separator format.
    """

    def test_markdown_output_basic(self):
        """Test basic markdown table output format."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <thead>
                <tr><th>Name</th><th>Age</th></tr>
            </thead>
            <tbody>
                <tr><td>Alice</td><td>30</td></tr>
            </tbody>
        </table>
        """

        tables = extractor.extract(html)
        markdown = tables[0].to_markdown()

        # Verify markdown format
        lines = markdown.strip().split("\n")
        assert len(lines) == 3  # header, separator, data row

        # Header row
        assert "| Name" in lines[0]
        assert "| Age" in lines[0]

        # Separator row - verify current format uses dashes
        assert "| -" in lines[1]
        assert lines[1].count("|") >= 3  # At least 3 pipes for 2 columns

        # Data row
        assert "| Alice" in lines[2]
        assert "| 30" in lines[2]

    def test_markdown_separator_format(self):
        """Test that markdown separator uses correct dash format.

        The separator row should use dashes (-----) not colons.
        This test verifies the current format is maintained.
        """
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><th>Col1</th><th>Col2</th></tr>
            <tr><td>Data1</td><td>Data2</td></tr>
        </table>
        """

        tables = extractor.extract(html)
        markdown = tables[0].to_markdown()

        lines = markdown.strip().split("\n")
        separator = lines[1]

        # Separator should be: | ----- | ----- |
        # NOT: | :---: | :---: |
        assert ":" not in separator, "Separator should use dashes, not colons"
        assert (
            "---" in separator or "-----" in separator
        ), "Separator should contain dashes"

    def test_markdown_column_alignment(self):
        """Test that columns are properly aligned in markdown output."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><th>Short</th><th>Much Longer Header</th></tr>
            <tr><td>A</td><td>B</td></tr>
        </table>
        """

        tables = extractor.extract(html)
        markdown = tables[0].to_markdown()

        lines = markdown.strip().split("\n")

        # All lines should have the same structure
        assert all("|" in line for line in lines)
        # Header and separator should have matching column counts
        assert lines[0].count("|") == lines[1].count("|")


class TestCellProperties:
    """Tests for cell property detection.

    Rule #4: Focused test class - tests cell properties
    """

    def test_numeric_cell_detection(self):
        """Test detection of numeric cell values."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><td>123</td><td>$45.67</td><td>text</td></tr>
        </table>
        """

        tables = extractor.extract(html)
        row = tables[0].rows[0]

        assert row.cells[0].is_numeric is True  # 123
        assert row.cells[1].is_numeric is True  # $45.67
        assert row.cells[2].is_numeric is False  # text

    def test_empty_cell_detection(self):
        """Test detection of empty cells."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><td>value</td><td></td><td>   </td></tr>
        </table>
        """

        tables = extractor.extract(html)
        row = tables[0].rows[0]

        assert row.cells[0].is_empty is False
        assert row.cells[1].is_empty is True
        assert row.cells[2].is_empty is True  # Whitespace only


class TestEdgeCases:
    """Tests for edge cases and error handling.

    Rule #4: Focused test class - tests edge cases
    """

    def test_empty_table(self):
        """Test handling of empty table."""
        extractor = HTMLTableExtractor()

        html = "<table></table>"

        tables = extractor.extract(html)

        assert len(tables) == 1
        assert len(tables[0].rows) == 0

    def test_no_tables(self):
        """Test handling of HTML with no tables."""
        extractor = HTMLTableExtractor()

        html = "<div><p>No tables here</p></div>"

        tables = extractor.extract(html)

        assert len(tables) == 0

    def test_merged_cells_colspan(self):
        """Test handling of colspan in tables."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><td colspan="2">Merged</td></tr>
            <tr><td>A</td><td>B</td></tr>
        </table>
        """

        tables = extractor.extract(html)

        assert tables[0].has_merged_cells is True
        assert tables[0].grid[0][0] == "Merged"
        assert tables[0].grid[0][1] == "Merged"

    def test_merged_cells_rowspan(self):
        """Test handling of rowspan in tables."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr><td rowspan="2">Merged</td><td>A</td></tr>
            <tr><td>B</td></tr>
        </table>
        """

        tables = extractor.extract(html)

        assert tables[0].has_merged_cells is True
        assert tables[0].grid[0][0] == "Merged"
        assert tables[0].grid[1][0] == "Merged"

    def test_nested_tables(self):
        """Test handling of nested tables."""
        extractor = HTMLTableExtractor()

        html = """
        <table>
            <tr>
                <td>
                    <table>
                        <tr><td>Nested</td></tr>
                    </table>
                </td>
            </tr>
        </table>
        """

        tables = extractor.extract(html)

        # Should extract both tables
        assert len(tables) == 2


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Table extraction: 4 tests (simple, no thead, multiple, caption)
    - Markdown format: 3 tests (basic, separator format, column alignment)
    - Cell properties: 2 tests (numeric detection, empty detection)
    - Edge cases: 5 tests (empty, no tables, colspan, rowspan, nested)

    Total: 14 tests

Design Decisions:
    1. Focus on verifying current markdown separator format (dashes, not colons)
    2. Test cell property detection (numeric, empty)
    3. Test merged cell handling (colspan, rowspan)
    4. Test nested table extraction

Behaviors Tested:
    - Basic table extraction with headers
    - Tables without explicit thead
    - Multiple table extraction
    - Caption preservation
    - Markdown output format
    - Column alignment
    - Cell type detection
    - Empty cell handling
    - Merged cells
    - Nested tables
"""
