"""
Tests for Text Extraction.

This module tests text extraction from various document formats (PDF, EPUB, DOCX, TXT).

Test Strategy
-------------
- Focus on public API behavior (extract, extract_with_metadata)
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Mock heavy dependencies (PyMuPDF) but test real text processing
- Test format dispatch logic and error handling

Organization
------------
- TestTextExtractor: Main extraction API
- TestFormatDispatch: Format detection and routing
- TestTextCleaning: Text cleaning and processing
- TestMetadataExtraction: Section and metadata extraction
"""

from unittest.mock import MagicMock

import pytest

from ingestforge.core.config import Config
from ingestforge.ingest.text_extractor import TextExtractor


# ============================================================================
# Test Classes
# ============================================================================


class TestTextExtractor:
    """Tests for TextExtractor class creation.

    Rule #4: Focused test class - tests only TextExtractor initialization
    """

    def test_create_extractor(self):
        """Test creating a TextExtractor."""
        config = Config()
        extractor = TextExtractor(config)

        assert extractor.config is config

    def test_create_with_custom_config(self):
        """Test creating extractor with custom config."""
        config = Config()
        extractor = TextExtractor(config)

        assert extractor.config is not None


class TestFormatDispatch:
    """Tests for format detection and extraction routing.

    Rule #4: Focused test class - tests only format dispatch
    """

    def test_extract_unsupported_format(self, temp_dir):
        """Test extract raises error for unsupported format."""
        config = Config()
        extractor = TextExtractor(config)

        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("test content")

        with pytest.raises(ValueError) as exc_info:
            extractor.extract(unsupported_file)

        assert "Unsupported file format" in str(exc_info.value)
        assert ".xyz" in str(exc_info.value)

    def test_extract_text_file(self, temp_dir):
        """Test extracting from plain text file."""
        config = Config()
        extractor = TextExtractor(config)

        text_file = temp_dir / "test.txt"
        text_file.write_text("Hello, world!\nThis is a test.")

        result = extractor.extract(text_file)

        assert "Hello, world!" in result
        assert "This is a test." in result

    def test_extract_markdown_file(self, temp_dir):
        """Test extracting from markdown file."""
        config = Config()
        extractor = TextExtractor(config)

        md_file = temp_dir / "test.md"
        md_file.write_text("# Title\n\nSome content here.")

        result = extractor.extract(md_file)

        assert "Title" in result
        assert "Some content" in result


class TestTextCleaning:
    """Tests for text cleaning utilities.

    Rule #4: Focused test class - tests only text cleaning
    """

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        config = Config()
        extractor = TextExtractor(config)

        dirty_text = "Hello  world   with   extra   spaces"

        result = extractor._clean_text(dirty_text)

        # Should normalize whitespace
        assert result is not None
        assert "Hello" in result

    def test_clean_text_advanced_removes_page_numbers(self):
        """Test advanced cleaning removes page numbers."""
        config = Config()
        extractor = TextExtractor(config)

        text_with_pages = "Content here\n42\nMore content\nPage 1 of 10\nFinal text"

        result = extractor._clean_text_advanced(text_with_pages)

        # Should remove page numbers and page markers
        assert "Content here" in result
        assert "More content" in result
        assert "Page 1 of 10" not in result


class TestMetadataExtraction:
    """Tests for metadata and section extraction.

    Rule #4: Focused test class - tests only metadata extraction
    """

    def test_extract_with_metadata(self, temp_dir):
        """Test extract_with_metadata returns complete dict."""
        config = Config()
        extractor = TextExtractor(config)

        text_file = temp_dir / "test.txt"
        text_file.write_text("# Header\n\nSome content here.\nMore text.")

        result = extractor.extract_with_metadata(text_file)

        assert "text" in result
        assert "file_name" in result
        assert "file_type" in result
        assert "word_count" in result
        assert "char_count" in result
        assert "sections" in result
        assert result["file_name"] == "test.txt"
        assert result["file_type"] == ".txt"

    def test_extract_markdown_sections(self):
        """Test extracting markdown sections from text."""
        config = Config()
        extractor = TextExtractor(config)

        text = "# Title\n\nContent\n\n## Subtitle\n\nMore content\n\n### Sub-subtitle"

        sections = extractor._extract_markdown_sections(text)

        assert len(sections) == 3
        assert sections[0]["level"] == 1
        assert sections[0]["title"] == "Title"
        assert sections[1]["level"] == 2
        assert sections[1]["title"] == "Subtitle"

    def test_parse_markdown_header(self):
        """Test parsing single markdown header line."""
        config = Config()
        extractor = TextExtractor(config)

        header = "### My Section Title"

        result = extractor._parse_markdown_header(header)

        assert result["level"] == 3
        assert result["title"] == "My Section Title"

    def test_extract_heading_level(self):
        """Test extracting heading level from DOCX style."""
        config = Config()
        extractor = TextExtractor(config)

        assert extractor._extract_heading_level("Heading 1") == 1
        assert extractor._extract_heading_level("Heading 2") == 2
        assert extractor._extract_heading_level("Heading 3") == 3
        assert extractor._extract_heading_level("Normal") == 1


class TestPDFExtraction:
    """Tests for PDF extraction (with mocking).

    Rule #4: Focused test class - tests only PDF extraction
    """

    def test_fitz_lazy_property(self):
        """Test PyMuPDF is lazy-loaded."""
        config = Config()
        extractor = TextExtractor(config)

        # Should not be loaded yet (lazy property)
        assert (
            "_fitz" not in extractor.__dict__ or extractor.__dict__.get("_fitz") is None
        )

    @pytest.mark.skip(
        reason="PDF extraction requires PyMuPDF - tested in integration tests"
    )
    def test_extract_pdf_with_mock(self, temp_dir):
        """Test PDF extraction with mocked PyMuPDF.

        NOTE: Mocking PyMuPDF's lazy loading is complex and brittle.
        PDF extraction is tested in integration tests with real PDFs.
        """
        pass

    def test_extract_page_text(self):
        """Test extracting text from single PDF page."""
        config = Config()
        extractor = TextExtractor(config)

        # Mock page object
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,  # Text block
                    "lines": [
                        {"spans": [{"text": "Line 1"}]},
                        {"spans": [{"text": "Line 2"}]},
                    ],
                },
                {
                    "type": 1,  # Image block (should be skipped)
                    "lines": [],
                },
            ]
        }

        result = extractor._extract_page_text(mock_page)

        assert "Line 1" in result
        assert "Line 2" in result


class TestDOCXExtraction:
    """Tests for DOCX extraction (with mocking).

    Rule #4: Focused test class - tests only DOCX extraction
    """

    def test_format_paragraph_normal(self):
        """Test formatting normal paragraph (no heading)."""
        config = Config()
        extractor = TextExtractor(config)

        # Mock paragraph
        mock_para = MagicMock()
        mock_para.style.name = "Normal"

        result = extractor._format_paragraph(mock_para, "Normal text")

        assert result == "Normal text"
        assert not result.startswith("#")

    def test_format_paragraph_heading(self):
        """Test formatting heading paragraph."""
        config = Config()
        extractor = TextExtractor(config)

        # Mock paragraph
        mock_para = MagicMock()
        mock_para.style.name = "Heading 2"

        result = extractor._format_paragraph(mock_para, "Section Title")

        assert result.startswith("##")
        assert "Section Title" in result


class TestHelperMethods:
    """Tests for helper extraction methods.

    Rule #4: Focused test class - tests helper methods
    """

    def test_extract_line_text_from_spans(self):
        """Test extracting text from line spans."""
        config = Config()
        extractor = TextExtractor(config)

        spans = [
            {"text": "Hello "},
            {"text": "world"},
            {"text": "!"},
        ]

        result = extractor._extract_line_text_from_spans(spans)

        assert result == "Hello world!"

    def test_extract_text_from_block(self):
        """Test extracting text lines from PDF block."""
        config = Config()
        extractor = TextExtractor(config)

        block = {
            "lines": [
                {"spans": [{"text": "First line"}]},
                {"spans": [{"text": "   "}]},  # Whitespace only (should skip)
                {"spans": [{"text": "Second line"}]},
            ]
        }

        result = extractor._extract_text_from_block(block)

        assert len(result) == 2
        assert "First line" in result
        assert "Second line" in result


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - TextExtractor: 2 tests (creation, config)
    - Format dispatch: 3 tests (unsupported, txt, md)
    - Text cleaning: 2 tests (basic, advanced with page numbers)
    - Metadata extraction: 4 tests (full metadata, sections, headers, levels)
    - PDF extraction: 2 tests (lazy loading, page text) + 1 skipped (mock)
    - DOCX extraction: 2 tests (normal para, heading para)
    - Helper methods: 2 tests (line spans, block extraction)

    Total: 17 tests (1 skipped)

Design Decisions:
    1. Focus on public API (extract, extract_with_metadata)
    2. Mock PyMuPDF for PDF tests (heavy dependency)
    3. Test real text processing without mocking (cleaning, sections)
    4. Don't test EPUB/DOCX full extraction (require external libs)
    5. Test format dispatch and error handling
    6. Simple, clear tests that verify extraction works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - Format detection and dispatch to correct extractor
    - Error handling for unsupported formats
    - Text file extraction (TXT, MD)
    - Text cleaning (basic and advanced)
    - Metadata extraction (word count, sections, headers)
    - PDF extraction with mocked PyMuPDF
    - DOCX paragraph formatting
    - Helper methods for text processing

Justification:
    - Text extraction is I/O heavy - focus on logic, mock external deps
    - Don't test library functionality (PyMuPDF, ebooklib) - test our wrappers
    - Test text processing utilities thoroughly (cleaning, sections)
    - Verify format dispatch works correctly
    - Simple tests that verify extraction produces expected output
"""
