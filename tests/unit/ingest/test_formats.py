"""
Tests for Format Handlers.

This module tests the format handler registry and format detection utilities.

Test Strategy
-------------
- Focus on format detection and handler retrieval
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test the registry, dataclass, and utility functions
- No mocking needed - pure data structure tests

Organization
------------
- TestFormatHandler: FormatHandler dataclass
- TestFormatRegistry: FORMAT_HANDLERS registry
- TestFormatDetection: get_format_handler function
- TestSupportedFormats: is_supported and get_supported_extensions
- TestMimeTypes: get_mime_type function
- TestSplittingRequirements: requires_splitting function
"""

from pathlib import Path


from ingestforge.ingest.formats import (
    FormatHandler,
    FORMAT_HANDLERS,
    get_format_handler,
    is_supported,
    get_supported_extensions,
    get_mime_type,
    requires_splitting,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestFormatHandler:
    """Tests for FormatHandler dataclass.

    Rule #4: Focused test class - tests only FormatHandler
    """

    def test_create_format_handler(self):
        """Test creating a FormatHandler."""
        handler = FormatHandler(
            extension=".pdf",
            name="PDF Document",
            mime_types=["application/pdf"],
            requires_split=True,
            extractor="pdf",
        )

        assert handler.extension == ".pdf"
        assert handler.name == "PDF Document"
        assert handler.mime_types == ["application/pdf"]
        assert handler.requires_split is True
        assert handler.extractor == "pdf"

    def test_format_handler_with_defaults(self):
        """Test FormatHandler with default extractor."""
        handler = FormatHandler(
            extension=".txt",
            name="Text",
            mime_types=["text/plain"],
            requires_split=False,
        )

        assert handler.extractor is None


class TestFormatRegistry:
    """Tests for FORMAT_HANDLERS registry.

    Rule #4: Focused test class - tests only FORMAT_HANDLERS
    """

    def test_registry_contains_pdf(self):
        """Test registry contains PDF handler."""
        assert ".pdf" in FORMAT_HANDLERS
        handler = FORMAT_HANDLERS[".pdf"]

        assert handler.extension == ".pdf"
        assert handler.name == "PDF Document"
        assert handler.requires_split is True

    def test_registry_contains_text_formats(self):
        """Test registry contains text/markdown handlers."""
        assert ".txt" in FORMAT_HANDLERS
        assert ".md" in FORMAT_HANDLERS

        txt_handler = FORMAT_HANDLERS[".txt"]
        md_handler = FORMAT_HANDLERS[".md"]

        assert txt_handler.extractor == "text"
        # Markdown now has a dedicated extractor
        assert md_handler.extractor == "markdown"

    def test_registry_contains_office_formats(self):
        """Test registry contains office document handlers."""
        assert ".docx" in FORMAT_HANDLERS
        assert ".epub" in FORMAT_HANDLERS

        docx_handler = FORMAT_HANDLERS[".docx"]
        epub_handler = FORMAT_HANDLERS[".epub"]

        assert docx_handler.requires_split is False
        assert epub_handler.requires_split is False


class TestFormatDetection:
    """Tests for format detection.

    Rule #4: Focused test class - tests format detection only
    """

    def test_get_format_handler_for_pdf(self):
        """Test getting handler for PDF file."""
        pdf_file = Path("document.pdf")
        handler = get_format_handler(pdf_file)

        assert handler is not None
        assert handler.extension == ".pdf"
        assert handler.requires_split is True

    def test_get_format_handler_for_text(self):
        """Test getting handler for text file."""
        txt_file = Path("notes.txt")
        handler = get_format_handler(txt_file)

        assert handler is not None
        assert handler.extension == ".txt"
        assert handler.extractor == "text"

    def test_get_format_handler_for_markdown(self):
        """Test getting handler for markdown file."""
        md_file = Path("README.md")
        handler = get_format_handler(md_file)

        assert handler is not None
        assert handler.extension == ".md"
        # Markdown now has a dedicated extractor
        assert handler.extractor == "markdown"

    def test_get_format_handler_for_unsupported(self):
        """Test getting handler for unsupported format returns None."""
        unknown_file = Path("data.xyz")
        handler = get_format_handler(unknown_file)

        assert handler is None

    def test_get_format_handler_case_insensitive(self):
        """Test format detection is case-insensitive."""
        pdf_upper = Path("DOCUMENT.PDF")
        pdf_mixed = Path("Document.Pdf")

        handler_upper = get_format_handler(pdf_upper)
        handler_mixed = get_format_handler(pdf_mixed)

        assert handler_upper is not None
        assert handler_mixed is not None
        assert handler_upper.extension == ".pdf"
        assert handler_mixed.extension == ".pdf"


class TestSupportedFormats:
    """Tests for supported format checking.

    Rule #4: Focused test class - tests is_supported only
    """

    def test_is_supported_pdf(self):
        """Test PDF is supported."""
        assert is_supported(Path("doc.pdf")) is True

    def test_is_supported_text(self):
        """Test text formats are supported."""
        assert is_supported(Path("notes.txt")) is True
        assert is_supported(Path("README.md")) is True

    def test_is_supported_office(self):
        """Test office formats are supported."""
        assert is_supported(Path("report.docx")) is True
        assert is_supported(Path("book.epub")) is True

    def test_is_supported_unsupported_format(self):
        """Test unsupported format returns False."""
        assert is_supported(Path("data.xyz")) is False
        assert is_supported(Path("archive.zip")) is False

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        extensions = get_supported_extensions()

        assert isinstance(extensions, list)
        assert ".pdf" in extensions
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".docx" in extensions
        assert ".epub" in extensions


class TestMimeTypes:
    """Tests for MIME type detection.

    Rule #4: Focused test class - tests MIME type detection only
    """

    def test_get_mime_type_for_pdf(self):
        """Test getting MIME type for PDF."""
        pdf_file = Path("document.pdf")
        mime_type = get_mime_type(pdf_file)

        assert mime_type == "application/pdf"

    def test_get_mime_type_for_text(self):
        """Test getting MIME type for text file."""
        txt_file = Path("notes.txt")
        mime_type = get_mime_type(txt_file)

        assert mime_type == "text/plain"

    def test_get_mime_type_for_markdown(self):
        """Test getting MIME type for markdown."""
        md_file = Path("README.md")
        mime_type = get_mime_type(md_file)

        # Should return first MIME type in list
        assert mime_type in ["text/markdown", "text/x-markdown"]

    def test_get_mime_type_for_unsupported(self):
        """Test getting MIME type for unsupported format returns None."""
        unknown_file = Path("data.xyz")
        mime_type = get_mime_type(unknown_file)

        assert mime_type is None


class TestSplittingRequirements:
    """Tests for splitting requirements.

    Rule #4: Focused test class - tests splitting requirements only
    """

    def test_requires_splitting_for_pdf(self):
        """Test PDF requires splitting."""
        pdf_file = Path("document.pdf")

        assert requires_splitting(pdf_file) is True

    def test_requires_splitting_for_text_formats(self):
        """Test text formats don't require splitting."""
        txt_file = Path("notes.txt")
        md_file = Path("README.md")

        assert requires_splitting(txt_file) is False
        assert requires_splitting(md_file) is False

    def test_requires_splitting_for_office_formats(self):
        """Test office formats don't require splitting."""
        docx_file = Path("report.docx")
        epub_file = Path("book.epub")

        assert requires_splitting(docx_file) is False
        assert requires_splitting(epub_file) is False

    def test_requires_splitting_for_unsupported(self):
        """Test unsupported format returns False."""
        unknown_file = Path("data.xyz")

        assert requires_splitting(unknown_file) is False


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - FormatHandler: 2 tests (creation, defaults)
    - FORMAT_HANDLERS: 3 tests (PDF, text formats, office formats)
    - Format detection: 5 tests (PDF, text, markdown, unsupported, case-insensitive)
    - Supported formats: 5 tests (PDF, text, office, unsupported, list)
    - MIME types: 4 tests (PDF, text, markdown, unsupported)
    - Splitting requirements: 4 tests (PDF, text, office, unsupported)

    Total: 23 tests

Design Decisions:
    1. Focus on data structure and utility functions
    2. No mocking needed - pure data structure tests
    3. Test all supported formats in registry
    4. Test edge cases (unsupported formats, case sensitivity)
    5. Simple, clear tests that verify format detection works
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - FormatHandler dataclass creation and defaults
    - FORMAT_HANDLERS registry contents
    - Format handler retrieval for different file types
    - Case-insensitive format detection
    - Supported format checking (is_supported)
    - List of supported extensions
    - MIME type detection
    - Splitting requirements for different formats

Justification:
    - Format detection is core ingestion infrastructure
    - Registry-based approach simplifies testing (no dependencies)
    - Test all public API functions
    - Verify format metadata is correct
    - Simple tests verify format system works correctly
"""
