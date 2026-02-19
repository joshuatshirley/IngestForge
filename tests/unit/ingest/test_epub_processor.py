"""
Tests for EPUB document processor.

This module tests extraction of text, metadata, and TOC from EPUB e-books.

Test Strategy
-------------
- Mock zipfile operations for EPUB structure
- Test metadata extraction from OPF
- Test content file extraction
- Test TOC parsing from NCX
- Test helper functions

Organization
------------
- TestBasicProcessing: Main process method
- TestOPFPathDetection: Finding content.opf
- TestMetadataExtraction: Metadata from OPF
- TestContentFileExtraction: Reading content files
- TestTextExtraction: HTML content extraction
- TestTOCExtraction: Table of contents from NCX
- TestHelperFunctions: extract_text and extract_with_metadata
- TestErrorHandling: Invalid EPUBs and edge cases
"""

from unittest.mock import MagicMock, patch
import pytest

from ingestforge.ingest.epub_processor import (
    EPUBProcessor,
    extract_text,
    extract_with_metadata,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicProcessing:
    """Tests for basic EPUB processing.

    Rule #4: Focused test class - tests process method
    """

    @patch("zipfile.is_zipfile")
    def test_process_rejects_non_zip(self, mock_is_zipfile, temp_dir):
        """Test process rejects non-ZIP files."""
        processor = EPUBProcessor()

        mock_is_zipfile.return_value = False

        epub_file = temp_dir / "test.epub"
        epub_file.touch()

        with pytest.raises(ValueError) as exc_info:
            processor.process(epub_file)

        assert "Not a valid EPUB" in str(exc_info.value)

    @patch("zipfile.ZipFile")
    @patch("zipfile.is_zipfile")
    def test_process_minimal_epub(self, mock_is_zipfile, mock_zipfile, temp_dir):
        """Test processing minimal EPUB."""
        processor = EPUBProcessor()

        mock_is_zipfile.return_value = True

        # Create mock ZIP
        mock_zip = MagicMock()
        mock_zip.read.return_value = b'<?xml version="1.0"?><package></package>'
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        epub_file = temp_dir / "test.epub"
        epub_file.touch()

        result = processor.process(epub_file)

        assert result["type"] == "epub"
        assert result["source"] == "test.epub"


class TestOPFPathDetection:
    """Tests for content.opf path detection.

    Rule #4: Focused test class - tests _get_opf_path
    """

    def test_get_opf_path_from_container(self):
        """Test getting OPF path from container.xml."""
        processor = EPUBProcessor()

        container_xml = b"""<?xml version="1.0"?>
<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
    <rootfiles>
        <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
    </rootfiles>
</container>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = container_xml

        result = processor._get_opf_path(mock_epub, "META-INF/container.xml")

        assert result == "OEBPS/content.opf"

    def test_get_opf_path_fallback(self):
        """Test OPF path falls back to default."""
        processor = EPUBProcessor()

        mock_epub = MagicMock()
        mock_epub.read.side_effect = Exception("File not found")

        result = processor._get_opf_path(mock_epub, "META-INF/container.xml")

        assert result == "content.opf"


class TestMetadataExtraction:
    """Tests for metadata extraction.

    Rule #4: Focused test class - tests _extract_metadata
    """

    def test_extract_metadata_basic(self):
        """Test extracting basic metadata from OPF."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Test Book</dc:title>
        <dc:creator>John Doe</dc:creator>
        <dc:publisher>Test Publisher</dc:publisher>
        <dc:date>2024</dc:date>
        <dc:language>en</dc:language>
        <dc:identifier>ISBN-123</dc:identifier>
    </metadata>
</package>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = opf_content

        result = processor._extract_metadata(mock_epub, "content.opf")

        assert result["title"] == "Test Book"
        assert result["author"] == "John Doe"
        assert result["publisher"] == "Test Publisher"
        assert result["date"] == "2024"
        assert result["language"] == "en"
        assert result["isbn"] == "ISBN-123"

    def test_extract_metadata_partial(self):
        """Test extracting partial metadata."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Partial Book</dc:title>
    </metadata>
</package>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = opf_content

        result = processor._extract_metadata(mock_epub, "content.opf")

        assert result["title"] == "Partial Book"
        assert "author" not in result  # Not present, should be filtered

    def test_extract_metadata_error(self):
        """Test metadata extraction handles errors."""
        processor = EPUBProcessor()

        mock_epub = MagicMock()
        mock_epub.read.side_effect = Exception("Read error")

        result = processor._extract_metadata(mock_epub, "content.opf")

        assert result == {}


class TestContentFileExtraction:
    """Tests for content file extraction.

    Rule #4: Focused test class - tests content file handling
    """

    def test_get_content_files_from_spine(self):
        """Test getting content files from spine order."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <manifest>
        <item id="chapter1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
        <item id="chapter2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
    </manifest>
    <spine>
        <itemref idref="chapter1"/>
        <itemref idref="chapter2"/>
    </spine>
</package>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = opf_content

        result = processor._get_content_files(mock_epub, "content.opf")

        assert len(result) == 2
        assert "ch1.xhtml" in result[0]
        assert "ch2.xhtml" in result[1]

    def test_get_content_files_with_base_dir(self):
        """Test content files with base directory."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <manifest>
        <item id="ch1" href="text/chapter1.html" media-type="application/xhtml+xml"/>
    </manifest>
    <spine>
        <itemref idref="ch1"/>
    </spine>
</package>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = opf_content

        result = processor._get_content_files(mock_epub, "OEBPS/content.opf")

        assert len(result) == 1
        assert "OEBPS/text/chapter1.html" in result[0]

    def test_get_content_files_no_spine(self):
        """Test handling missing spine."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <manifest>
        <item id="ch1" href="chapter1.html"/>
    </manifest>
</package>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = opf_content

        result = processor._get_content_files(mock_epub, "content.opf")

        assert result == []


class TestTextExtraction:
    """Tests for text extraction.

    Rule #4: Focused test class - tests _extract_text
    """

    def test_extract_text_from_file(self):
        """Test extracting text from content file."""
        processor = EPUBProcessor()

        html_content = b"""<html>
<head><title>Chapter</title></head>
<body>
<h1>Chapter 1</h1>
<p>This is the content.</p>
</body>
</html>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = html_content

        result = processor._extract_text_from_file(mock_epub, "chapter1.html")

        assert "Chapter 1" in result
        assert "This is the content" in result

    def test_extract_text_strips_scripts(self):
        """Test text extraction removes scripts."""
        processor = EPUBProcessor()

        html_content = b"""<html>
<body>
<script>alert('test');</script>
<p>Real content</p>
</body>
</html>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = html_content

        result = processor._extract_text_from_file(mock_epub, "page.html")

        assert "Real content" in result
        assert "alert" not in result

    def test_extract_text_strips_styles(self):
        """Test text extraction removes styles."""
        processor = EPUBProcessor()

        html_content = b"""<html>
<head><style>body { color: red; }</style></head>
<body><p>Content</p></body>
</html>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = html_content

        result = processor._extract_text_from_file(mock_epub, "page.html")

        assert "Content" in result
        assert "color: red" not in result

    def test_extract_text_error_handling(self):
        """Test text extraction handles errors."""
        processor = EPUBProcessor()

        mock_epub = MagicMock()
        mock_epub.read.side_effect = Exception("Read error")

        result = processor._extract_text_from_file(mock_epub, "missing.html")

        assert result == ""


class TestTOCExtraction:
    """Tests for table of contents extraction.

    Rule #4: Focused test class - tests _extract_toc
    """

    def test_extract_toc_from_ncx(self):
        """Test extracting TOC from NCX file."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <manifest>
        <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    </manifest>
</package>"""

        ncx_content = b"""<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/">
    <navMap>
        <navPoint>
            <navLabel><text>Chapter 1</text></navLabel>
            <content src="ch1.html"/>
        </navPoint>
        <navPoint>
            <navLabel><text>Chapter 2</text></navLabel>
            <content src="ch2.html"/>
        </navPoint>
    </navMap>
</ncx>"""

        mock_epub = MagicMock()

        def read_side_effect(path):
            if "content.opf" in path:
                return opf_content
            elif "toc.ncx" in path:
                return ncx_content
            return b""

        mock_epub.read.side_effect = read_side_effect

        result = processor._extract_toc(mock_epub, "content.opf")

        assert len(result) == 2
        assert result[0]["label"] == "Chapter 1"
        assert result[0]["href"] == "ch1.html"
        assert result[1]["label"] == "Chapter 2"
        assert result[1]["href"] == "ch2.html"

    def test_extract_toc_no_ncx(self):
        """Test TOC extraction when no NCX file."""
        processor = EPUBProcessor()

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <manifest>
        <item id="ch1" href="chapter1.html"/>
    </manifest>
</package>"""

        mock_epub = MagicMock()
        mock_epub.read.return_value = opf_content

        result = processor._extract_toc(mock_epub, "content.opf")

        assert result == []

    def test_parse_ncx_error_handling(self):
        """Test NCX parsing handles errors."""
        processor = EPUBProcessor()

        mock_epub = MagicMock()
        mock_epub.read.side_effect = Exception("Read error")

        result = processor._parse_ncx(mock_epub, "toc.ncx")

        assert result == []


class TestHTMLStripping:
    """Tests for HTML tag stripping.

    Rule #4: Focused test class - tests _strip_html_tags
    """

    def test_strip_basic_tags(self):
        """Test stripping basic HTML tags."""
        processor = EPUBProcessor()

        html = "<p>Paragraph</p><div>Division</div>"

        result = processor._strip_html_tags(html)

        assert "Paragraph" in result
        assert "Division" in result
        assert "<p>" not in result
        assert "<div>" not in result

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        processor = EPUBProcessor()

        html = "<p>Text    with     extra    spaces</p>"

        result = processor._strip_html_tags(html)

        assert "Text with extra spaces" in result


class TestHelperFunctions:
    """Tests for helper functions.

    Rule #4: Focused test class - tests module-level functions
    """

    @patch("zipfile.ZipFile")
    @patch("zipfile.is_zipfile")
    def test_extract_text_function(self, mock_is_zipfile, mock_zipfile, temp_dir):
        """Test extract_text helper function."""
        mock_is_zipfile.return_value = True

        opf_content = b'<?xml version="1.0"?><package></package>'

        mock_zip = MagicMock()
        mock_zip.read.return_value = opf_content
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        epub_file = temp_dir / "test.epub"
        epub_file.touch()

        result = extract_text(epub_file)

        assert isinstance(result, str)

    @patch("zipfile.ZipFile")
    @patch("zipfile.is_zipfile")
    def test_extract_with_metadata_function(
        self, mock_is_zipfile, mock_zipfile, temp_dir
    ):
        """Test extract_with_metadata helper function."""
        mock_is_zipfile.return_value = True

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Test</dc:title>
    </metadata>
</package>"""

        mock_zip = MagicMock()
        mock_zip.read.return_value = opf_content
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        epub_file = temp_dir / "test.epub"
        epub_file.touch()

        result = extract_with_metadata(epub_file)

        assert "text" in result
        assert "metadata" in result
        assert result["type"] == "epub"


class TestEdgeCases:
    """Tests for edge cases.

    Rule #4: Focused test class - tests edge cases
    """

    @patch("zipfile.ZipFile")
    @patch("zipfile.is_zipfile")
    def test_empty_epub(self, mock_is_zipfile, mock_zipfile, temp_dir):
        """Test processing empty EPUB."""
        mock_is_zipfile.return_value = True

        opf_content = b"""<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf">
    <manifest></manifest>
    <spine></spine>
</package>"""

        mock_zip = MagicMock()
        mock_zip.read.return_value = opf_content
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        epub_file = temp_dir / "empty.epub"
        epub_file.touch()

        result = EPUBProcessor().process(epub_file)

        assert result["text"] == ""

    def test_malformed_xml(self):
        """Test handling malformed XML."""
        processor = EPUBProcessor()

        mock_epub = MagicMock()
        mock_epub.read.return_value = b"<invalid xml"

        # Should not crash
        result = processor._extract_metadata(mock_epub, "content.opf")

        assert result == {}


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Basic processing: 2 tests (rejects non-zip, minimal epub)
    - OPF path: 2 tests (from container, fallback)
    - Metadata: 3 tests (basic, partial, error)
    - Content files: 3 tests (from spine, with base dir, no spine)
    - Text extraction: 4 tests (from file, strips scripts, strips styles, errors)
    - TOC: 3 tests (from NCX, no NCX, error handling)
    - HTML stripping: 2 tests (basic tags, whitespace)
    - Helper functions: 2 tests (extract_text, extract_with_metadata)
    - Edge cases: 2 tests (empty epub, malformed xml)

    Total: 23 tests

Design Decisions:
    1. Mock zipfile operations to avoid file dependencies
    2. Test XML parsing for OPF and NCX
    3. Cover metadata extraction paths
    4. Test content file ordering
    5. Verify HTML cleaning
    6. Test error handling throughout
    7. Include helper function tests

Behaviors Tested:
    - EPUB validation (must be ZIP)
    - OPF path detection from container.xml
    - Metadata extraction from OPF
    - Content file extraction from spine
    - HTML text extraction and cleaning
    - TOC parsing from NCX
    - Script and style removal
    - Error handling for missing/malformed files
    - Helper function interfaces
    - Edge cases and malformed data
"""
