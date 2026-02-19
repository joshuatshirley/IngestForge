"""
Table Processor for Dual-Format Table Preservation.

Converts tables to both markdown and HTML formats, storing both
in chunk metadata for flexible downstream processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ingestforge.ingest.ocr.table_builder import Table
from ingestforge.ingest.html_table_extractor import ExtractedTable
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_TABLE_ROWS = 500
MAX_TABLE_COLS = 50
MAX_CELL_LENGTH = 1000


@dataclass
class ProcessedTable:
    """A table processed into multiple formats.

    Attributes:
        markdown: Markdown representation for embedding/search
        html: HTML representation for display
        row_count: Number of rows
        col_count: Number of columns
        has_header: Whether table has a header row
        caption: Optional table caption
    """

    markdown: str
    html: str
    row_count: int
    col_count: int
    has_header: bool
    caption: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata."""
        return {
            "table_markdown": self.markdown,
            "table_html": self.html,
            "table_row_count": self.row_count,
            "table_col_count": self.col_count,
            "table_has_header": self.has_header,
            "table_caption": self.caption,
        }


class TableProcessor:
    """Processor for converting tables to multiple formats.

    Converts tables from both OCR-extracted (Table) and HTML-extracted
    (ExtractedTable) sources into standardized markdown and HTML formats.

    Usage:
        >>> processor = TableProcessor()
        >>> result = processor.process_ocr_table(ocr_table)
        >>> print(result.markdown)
        >>> print(result.html)
    """

    def __init__(self, include_caption: bool = True) -> None:
        """Initialize table processor.

        Args:
            include_caption: Include table captions in output (default True)
        """
        self.include_caption = include_caption

    def process_ocr_table(self, table: Table) -> ProcessedTable:
        """Process an OCR-extracted table.

        Args:
            table: Table from OCR table_builder

        Returns:
            ProcessedTable with markdown and HTML
        """
        markdown = self._ocr_table_to_markdown(table)
        html = self._ocr_table_to_html(table)

        return ProcessedTable(
            markdown=markdown,
            html=html,
            row_count=table.row_count,
            col_count=table.col_count,
            has_header=table.has_header,
            caption=None,
        )

    def process_html_table(self, table: ExtractedTable) -> ProcessedTable:
        """Process an HTML-extracted table.

        Args:
            table: Table from html_table_extractor

        Returns:
            ProcessedTable with markdown and HTML
        """
        markdown = self._extracted_table_to_markdown(table)
        html = self._extracted_table_to_html(table)

        return ProcessedTable(
            markdown=markdown,
            html=html,
            row_count=table.row_count,
            col_count=table.col_count,
            has_header=table.has_header,
            caption=table.caption if self.include_caption else None,
        )

    def _ocr_table_to_markdown(self, table: Table) -> str:
        """Convert OCR table to markdown format.

        Args:
            table: OCR table

        Returns:
            Markdown string
        """
        if not table.rows:
            return ""

        return table.to_markdown()

    def _ocr_table_to_html(self, table: Table) -> str:
        """Convert OCR table to HTML format.

        Args:
            table: OCR table

        Returns:
            HTML string
        """
        if not table.rows:
            return ""

        lines = ["<table>"]

        for row_idx, row in enumerate(table.rows[:MAX_TABLE_ROWS]):
            lines.append("  <tr>")
            tag = "th" if row_idx == 0 and table.has_header else "td"

            for cell in row.cells[:MAX_TABLE_COLS]:
                content = self._escape_html(cell.text[:MAX_CELL_LENGTH])
                lines.append(f"    <{tag}>{content}</{tag}>")

            lines.append("  </tr>")

        lines.append("</table>")

        return "\n".join(lines)

    def _extracted_table_to_markdown(self, table: ExtractedTable) -> str:
        """Convert HTML-extracted table to markdown.

        Args:
            table: ExtractedTable from HTML

        Returns:
            Markdown string
        """
        return table.to_markdown()

    def _extracted_table_to_html(self, table: ExtractedTable) -> str:
        """Convert HTML-extracted table to clean HTML.

        Args:
            table: ExtractedTable from HTML

        Returns:
            HTML string
        """
        lines = ["<table>"]

        # Add caption if present
        if table.caption and self.include_caption:
            escaped_caption = self._escape_html(table.caption)
            lines.append(f"  <caption>{escaped_caption}</caption>")

        # Add header row
        if table.headers:
            lines.append("  <thead>")
            lines.append("    <tr>")
            for header in table.headers[:MAX_TABLE_COLS]:
                content = self._escape_html(header[:MAX_CELL_LENGTH])
                lines.append(f"      <th>{content}</th>")
            lines.append("    </tr>")
            lines.append("  </thead>")

        # Add body rows
        lines.append("  <tbody>")
        for row in table.grid[:MAX_TABLE_ROWS]:
            lines.append("    <tr>")
            for cell in row[:MAX_TABLE_COLS]:
                content = self._escape_html(cell[:MAX_CELL_LENGTH])
                lines.append(f"      <td>{content}</td>")
            lines.append("    </tr>")
        lines.append("  </tbody>")

        lines.append("</table>")

        return "\n".join(lines)

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )


def process_table(table: Table) -> Tuple[str, str]:
    """Convenience function to process OCR table.

    Args:
        table: OCR table

    Returns:
        Tuple of (markdown, html)
    """
    processor = TableProcessor()
    result = processor.process_ocr_table(table)
    return result.markdown, result.html


def table_to_chunk_metadata(table: Table, page_number: int = 1) -> Dict[str, Any]:
    """Convert table to metadata for chunk.

    Args:
        table: OCR table
        page_number: Page number

    Returns:
        Dictionary with table metadata
    """
    processor = TableProcessor()
    result = processor.process_ocr_table(table)

    metadata = result.to_dict()
    metadata["page_number"] = page_number
    metadata["element_type"] = "Table"

    return metadata
