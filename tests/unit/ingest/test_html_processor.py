"""
Tests for HTML processor.

This module tests extraction of content, metadata, and structure from HTML files.

Test Strategy
-------------
- Mock trafilatura and BeautifulSoup where appropriate
- Test content extraction (text, markdown, HTML cleaning)
- Test metadata extraction (title, author, date, etc.)
- Test structure extraction (headings, sections)
- Test fallback modes when libraries unavailable

Organization
------------
- TestHTMLSection: HTMLSection dataclass
- TestExtractedHTML: ExtractedHTML dataclass
- TestCanProcess: File type detection
- TestFileReading: Encoding detection
- TestContentExtraction: Main content extraction with trafilatura
- TestMetadataExtraction: Metadata from HTML meta tags
- TestStructureExtraction: Heading hierarchy
- TestTableExtraction: HTML table processing
- TestCitationMetadata: DOI, arXiv extraction
- TestFallbackMode: BeautifulSoup fallback
- TestErrorHandling: Invalid files and missing libraries
"""

from unittest.mock import MagicMock, patch

from ingestforge.ingest.html_processor import (
    HTMLSection,
    ExtractedHTML,
    HTMLProcessor,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestHTMLSection:
    """Tests for HTMLSection dataclass.

    Rule #4: Focused test class - tests only HTMLSection
    """

    def test_create_section(self):
        """Test creating an HTMLSection."""
        section = HTMLSection(
            level=1,
            title="Introduction",
            content="Section content here",
        )

        assert section.level == 1
        assert section.title == "Introduction"
        assert section.content == "Section content here"
        assert section.subsections == []

    def test_section_with_subsections(self):
        """Test section with nested subsections."""
        subsection = HTMLSection(
            level=2,
            title="Subsection",
            content="Subsection content",
        )

        section = HTMLSection(
            level=1,
            title="Main Section",
            content="Main content",
            subsections=[subsection],
        )

        assert len(section.subsections) == 1
        assert section.subsections[0].title == "Subsection"


class TestExtractedHTML:
    """Tests for ExtractedHTML dataclass.

    Rule #4: Focused test class - tests only ExtractedHTML
    """

    def test_create_extracted_html(self):
        """Test creating an ExtractedHTML result."""
        from ingestforge.core.provenance import SourceLocation, SourceType

        source_loc = SourceLocation(
            source_type=SourceType.WEBPAGE,
            title="Test Page",
        )

        result = ExtractedHTML(
            text="Plain text content",
            markdown="# Markdown content",
            html_clean="<p>Clean HTML</p>",
            title="Test Page",
            authors=["John Doe"],
            publication_date="2024-01-01",
            description="Page description",
            site_name="Example Site",
            url="https://example.com",
            language="en",
            sections=[],
            headings=[],
            source_location=source_loc,
        )

        assert result.text == "Plain text content"
        assert result.title == "Test Page"
        assert result.authors == ["John Doe"]
        assert result.publication_date == "2024-01-01"
        assert result.url == "https://example.com"


class TestCanProcess:
    """Tests for file type detection.

    Rule #4: Focused test class - tests can_process
    """

    def test_can_process_html(self, temp_dir):
        """Test can process .html files."""
        processor = HTMLProcessor()

        html_file = temp_dir / "page.html"
        html_file.touch()

        assert processor.can_process(html_file) is True

    def test_can_process_htm(self, temp_dir):
        """Test can process .htm files."""
        processor = HTMLProcessor()

        htm_file = temp_dir / "page.htm"
        htm_file.touch()

        assert processor.can_process(htm_file) is True

    def test_can_process_mhtml(self, temp_dir):
        """Test can process .mhtml files."""
        processor = HTMLProcessor()

        mhtml_file = temp_dir / "page.mhtml"
        mhtml_file.touch()

        assert processor.can_process(mhtml_file) is True

    def test_can_process_xhtml(self, temp_dir):
        """Test can process .xhtml files."""
        processor = HTMLProcessor()

        xhtml_file = temp_dir / "page.xhtml"
        xhtml_file.touch()

        assert processor.can_process(xhtml_file) is True

    def test_cannot_process_pdf(self, temp_dir):
        """Test cannot process non-HTML files."""
        processor = HTMLProcessor()

        pdf_file = temp_dir / "document.pdf"
        pdf_file.touch()

        assert processor.can_process(pdf_file) is False


class TestFileReading:
    """Tests for file reading with encoding detection.

    Rule #4: Focused test class - tests _read_file
    """

    def test_read_utf8_file(self, temp_dir):
        """Test reading UTF-8 encoded file."""
        processor = HTMLProcessor()

        html_file = temp_dir / "test.html"
        html_file.write_text("<html>Hello</html>", encoding="utf-8")

        content = processor._read_file(html_file)

        assert "<html>Hello</html>" in content

    def test_read_latin1_file(self, temp_dir):
        """Test reading Latin-1 encoded file."""
        processor = HTMLProcessor()

        html_file = temp_dir / "test.html"
        html_file.write_bytes(b"<html>Caf\xe9</html>")

        content = processor._read_file(html_file)

        assert "<html>" in content

    def test_read_with_bom(self, temp_dir):
        """Test reading file with BOM."""
        processor = HTMLProcessor()

        html_file = temp_dir / "test.html"
        html_file.write_text("<html>Test</html>", encoding="utf-8-sig")

        content = processor._read_file(html_file)

        assert "<html>Test</html>" in content


class TestContentExtraction:
    """Tests for content extraction.

    Rule #4: Focused test class - tests _extract_content
    """

    def test_extract_content_with_trafilatura(self):
        """Test content extraction using trafilatura."""
        processor = HTMLProcessor()

        with patch("trafilatura.extract") as mock_extract, patch(
            "trafilatura.settings.use_config"
        ) as mock_config:
            mock_config.return_value = MagicMock()
            mock_extract.side_effect = [
                "Plain text content",
                "# Markdown content",
                "<p>Clean HTML</p>",
            ]

            html_content = "<html><body><p>Test</p></body></html>"

            result = processor._extract_content(html_content)

            assert result["text"] == "Plain text content"
            assert result["markdown"] == "# Markdown content"
            assert result["html_clean"] == "<p>Clean HTML</p>"
            assert mock_extract.call_count == 3

    def test_extract_content_with_tables(self):
        """Test extraction with include_tables=True."""
        processor = HTMLProcessor(include_tables=True)

        with patch("trafilatura.extract") as mock_extract, patch(
            "trafilatura.settings.use_config"
        ) as mock_config:
            mock_config.return_value = MagicMock()
            mock_extract.return_value = "Content"

            html_content = "<html><body><table></table></body></html>"

            processor._extract_content(html_content)

            # Verify trafilatura called with include_tables
            calls = mock_extract.call_args_list
            assert any(call.kwargs.get("include_tables") for call in calls)

    def test_extract_content_with_links(self):
        """Test extraction with include_links=True."""
        processor = HTMLProcessor(include_links=True)

        with patch("trafilatura.extract") as mock_extract, patch(
            "trafilatura.settings.use_config"
        ) as mock_config:
            mock_config.return_value = MagicMock()
            mock_extract.return_value = "Content"

            html_content = "<html><body><a href='#'>Link</a></body></html>"

            processor._extract_content(html_content)

            # Verify trafilatura called with include_links
            calls = mock_extract.call_args_list
            assert any(call.kwargs.get("include_links") for call in calls)

    def test_extract_content_fallback(self):
        """Test fallback extraction with BeautifulSoup."""
        processor = HTMLProcessor()

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_soup.get_text.return_value = "Extracted text"
            mock_soup.find.return_value = mock_soup
            mock_bs.return_value = mock_soup

            html_content = "<html><body><p>Test</p></body></html>"

            result = processor._extract_content_fallback(html_content)

            assert result["text"] == "Extracted text"
            assert result["markdown"] == "Extracted text"


class TestMetadataExtraction:
    """Tests for metadata extraction.

    Rule #4: Focused test class - tests _extract_metadata
    """

    def test_extract_metadata_basic(self):
        """Test basic metadata extraction."""
        processor = HTMLProcessor()

        with patch("trafilatura.extract_metadata") as mock_extract_meta:
            mock_metadata = MagicMock()
            mock_metadata.title = "Test Title"
            mock_metadata.author = "John Doe"
            mock_metadata.date = "2024-01-01"
            mock_metadata.description = "Test description"
            mock_metadata.sitename = "Example Site"
            mock_metadata.url = "https://example.com"
            mock_metadata.language = "en"
            mock_metadata.categories = []
            mock_metadata.tags = []

            mock_extract_meta.return_value = mock_metadata

            html_content = "<html></html>"

            result = processor._extract_metadata(html_content)

            assert result["title"] == "Test Title"
            assert result["authors"] == ["John Doe"]
            assert result["date"] == "2024-01-01"

    def test_extract_metadata_multiple_authors(self):
        """Test parsing multiple authors."""
        processor = HTMLProcessor()

        with patch("trafilatura.extract_metadata") as mock_extract_meta:
            mock_metadata = MagicMock()
            mock_metadata.title = "Title"
            mock_metadata.author = "John Doe, Jane Smith"
            mock_metadata.date = None
            mock_metadata.description = None
            mock_metadata.sitename = None
            mock_metadata.url = None
            mock_metadata.language = None
            mock_metadata.categories = []
            mock_metadata.tags = []

            mock_extract_meta.return_value = mock_metadata

            result = processor._extract_metadata("<html></html>")

            assert "John Doe" in result["authors"]
            assert "Jane Smith" in result["authors"]

    def test_extract_metadata_fallback_title(self):
        """Test fallback title extraction with BeautifulSoup."""
        processor = HTMLProcessor()

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_title = MagicMock()
            mock_title.string = "Fallback Title"
            mock_soup.title = mock_title
            mock_soup.find.return_value = None
            mock_soup.html = None
            mock_bs.return_value = mock_soup

            result = processor._extract_metadata_fallback("<html></html>")

            assert result["title"] == "Fallback Title"

    def test_extract_metadata_fallback_og_tags(self):
        """Test Open Graph tag extraction in fallback mode."""
        processor = HTMLProcessor()

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_soup.title = None

            # Mock og:title
            mock_og_title = MagicMock()
            mock_og_title.get.return_value = "OG Title"

            def find_side_effect(tag, **kwargs):
                if kwargs.get("property") == "og:title":
                    return mock_og_title
                return None

            mock_soup.find.side_effect = find_side_effect
            mock_soup.html = None
            mock_bs.return_value = mock_soup

            result = processor._extract_metadata_fallback("<html></html>")

            assert result["title"] == "OG Title"


class TestStructureExtraction:
    """Tests for document structure extraction.

    Rule #4: Focused test class - tests _extract_structure
    """

    def test_extract_structure_basic(self):
        """Test extracting heading structure."""
        processor = HTMLProcessor()

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()

            # Create mock headings
            mock_h1 = MagicMock()
            mock_h1.get_text.return_value = "Main Title"
            mock_h1.get.return_value = "title-id"

            mock_h2 = MagicMock()
            mock_h2.get_text.return_value = "Subtitle"
            mock_h2.get.return_value = "subtitle-id"

            def find_all_side_effect(tag):
                if tag == "h1":
                    return [mock_h1]
                elif tag == "h2":
                    return [mock_h2]
                return []

            mock_soup.find_all.side_effect = find_all_side_effect
            mock_bs.return_value = mock_soup

            html_content = "<html></html>"

            sections, headings = processor._extract_structure(html_content)

            assert len(headings) == 2
            assert headings[0]["level"] == 1
            assert headings[0]["text"] == "Main Title"

    def test_extract_structure_hierarchy(self):
        """Test building hierarchical sections."""
        processor = HTMLProcessor()

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()

            # Create nested headings
            mock_h1 = MagicMock()
            mock_h1.get_text.return_value = "Chapter"
            mock_h1.get.return_value = None

            mock_h2 = MagicMock()
            mock_h2.get_text.return_value = "Section"
            mock_h2.get.return_value = None

            def find_all_side_effect(tag):
                if tag == "h1":
                    return [mock_h1]
                elif tag == "h2":
                    return [mock_h2]
                return []

            mock_soup.find_all.side_effect = find_all_side_effect
            mock_bs.return_value = mock_soup

            sections, headings = processor._extract_structure("<html></html>")

            # Should have 1 top-level section with 1 subsection
            assert len(sections) == 1
            assert len(sections[0].subsections) == 1
            assert sections[0].title == "Chapter"
            assert sections[0].subsections[0].title == "Section"

    def test_extract_structure_empty_headings(self):
        """Test extraction skips empty headings."""
        processor = HTMLProcessor()

        with patch("bs4.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()

            # Mock for empty heading - get_text(strip=True) returns ""
            mock_h1_empty = MagicMock()

            def get_text_empty(strip=False):
                if strip:
                    return ""  # After stripping, becomes empty
                return "   "

            mock_h1_empty.get_text.side_effect = get_text_empty

            # Mock for valid heading
            mock_h1_valid = MagicMock()
            mock_h1_valid.get_text.return_value = "Valid"
            mock_h1_valid.get.return_value = None

            # Only return items for h1, empty for others
            def find_all_side_effect(tag):
                if tag == "h1":
                    return [mock_h1_empty, mock_h1_valid]
                return []

            mock_soup.find_all.side_effect = find_all_side_effect
            mock_bs.return_value = mock_soup

            sections, headings = processor._extract_structure("<html></html>")

            # Should only extract valid heading (empty one skipped)
            assert len(headings) == 1
            assert headings[0]["text"] == "Valid"


class TestTableExtraction:
    """Tests for HTML table extraction.

    Rule #4: Focused test class - tests table processing
    """

    def test_table_extraction_enabled(self):
        """Test table extraction when enabled."""
        processor = HTMLProcessor(include_tables=True)

        with patch(
            "ingestforge.ingest.html_table_extractor.HTMLTableExtractor"
        ) as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_table = MagicMock()
            mock_table.caption = "Table 1"
            mock_table.to_markdown.return_value = "| Col1 | Col2 |"
            mock_extractor.extract.return_value = [mock_table]
            mock_extractor_class.return_value = mock_extractor

            extracted = {"text": "Content", "markdown": "# Content"}

            processor._append_table_content("<html></html>", extracted)

            assert "Table 1" in extracted["text"]
            assert "| Col1 | Col2 |" in extracted["text"]

    def test_table_extraction_multiple_tables(self):
        """Test extraction of multiple tables."""
        processor = HTMLProcessor(include_tables=True)

        with patch(
            "ingestforge.ingest.html_table_extractor.HTMLTableExtractor"
        ) as mock_extractor_class:
            mock_extractor = MagicMock()

            table1 = MagicMock()
            table1.caption = None
            table1.to_markdown.return_value = "| A |"

            table2 = MagicMock()
            table2.caption = "Table 2"
            table2.to_markdown.return_value = "| B |"

            mock_extractor.extract.return_value = [table1, table2]
            mock_extractor_class.return_value = mock_extractor

            extracted = {"text": "", "markdown": ""}

            processor._append_table_content("<html></html>", extracted)

            assert "| A |" in extracted["text"]
            assert "| B |" in extracted["text"]

    def test_table_extraction_no_tables(self):
        """Test extraction when no tables found."""
        processor = HTMLProcessor(include_tables=True)

        with patch(
            "ingestforge.ingest.html_table_extractor.HTMLTableExtractor"
        ) as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_extractor.extract.return_value = []
            mock_extractor_class.return_value = mock_extractor

            extracted = {"text": "Content"}

            processor._append_table_content("<html></html>", extracted)

            # Content should remain unchanged
            assert extracted["text"] == "Content"


class TestCitationMetadata:
    """Tests for citation metadata extraction.

    Rule #4: Focused test class - tests citation enrichment
    """

    def test_citation_metadata_extraction(self):
        """Test DOI and citation metadata extraction."""
        processor = HTMLProcessor()

        with patch(
            "ingestforge.ingest.citation_metadata_extractor.CitationMetadataExtractor"
        ) as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_citation = MagicMock()
            mock_citation.doi = "10.1234/example"
            mock_citation.arxiv_id = "2024.12345"
            mock_citation.isbn = None
            mock_citation.pmid = None
            mock_citation.journal = "Example Journal"
            mock_citation.abstract = "Abstract text"

            mock_extractor.extract_from_html.return_value = mock_citation
            mock_extractor_class.return_value = mock_extractor

            metadata = {}

            processor._enrich_citation_metadata("<html></html>", None, metadata)

            assert metadata["doi"] == "10.1234/example"
            assert metadata["arxiv_id"] == "2024.12345"
            assert metadata["journal"] == "Example Journal"
            assert metadata["abstract"] == "Abstract text"

    def test_citation_metadata_no_identifiers(self):
        """Test when no citation identifiers found."""
        processor = HTMLProcessor()

        with patch(
            "ingestforge.ingest.citation_metadata_extractor.CitationMetadataExtractor"
        ) as mock_extractor_class:
            mock_extractor = MagicMock()
            mock_citation = MagicMock()
            mock_citation.doi = None
            mock_citation.arxiv_id = None
            mock_citation.isbn = None
            mock_citation.pmid = None
            mock_citation.journal = None
            mock_citation.abstract = None

            mock_extractor.extract_from_html.return_value = mock_citation
            mock_extractor_class.return_value = mock_extractor

            metadata = {}

            processor._enrich_citation_metadata("<html></html>", None, metadata)

            # Metadata should remain empty
            assert "doi" not in metadata
            assert "arxiv_id" not in metadata


class TestParseAuthors:
    """Tests for author parsing.

    Rule #4: Focused test class - tests _parse_authors
    """

    def test_parse_single_author(self):
        """Test parsing single author."""
        processor = HTMLProcessor()

        result = processor._parse_authors("John Doe")

        assert result == ["John Doe"]

    def test_parse_multiple_authors_comma(self):
        """Test parsing comma-separated authors."""
        processor = HTMLProcessor()

        result = processor._parse_authors("John Doe, Jane Smith")

        assert "John Doe" in result
        assert "Jane Smith" in result

    def test_parse_multiple_authors_and(self):
        """Test parsing authors with 'and'."""
        processor = HTMLProcessor()

        result = processor._parse_authors("John Doe and Jane Smith")

        assert "John Doe" in result
        assert "Jane Smith" in result

    def test_parse_multiple_authors_ampersand(self):
        """Test parsing authors with '&'."""
        processor = HTMLProcessor()

        result = processor._parse_authors("John Doe & Jane Smith")

        assert "John Doe" in result
        assert "Jane Smith" in result

    def test_parse_authors_none(self):
        """Test parsing None returns empty list."""
        processor = HTMLProcessor()

        result = processor._parse_authors(None)

        assert result == []

    def test_parse_authors_empty(self):
        """Test parsing empty string returns empty list."""
        processor = HTMLProcessor()

        result = processor._parse_authors("")

        assert result == []


class TestProcessMethod:
    """Tests for main process method.

    Rule #4: Focused test class - tests process()
    """

    def test_process_basic(self, temp_dir):
        """Test basic HTML processing."""
        processor = HTMLProcessor()

        html_file = temp_dir / "test.html"
        html_file.write_text(
            "<html><head><title>Test</title></head><body>Content</body></html>"
        )

        with patch("trafilatura.extract") as mock_extract, patch(
            "trafilatura.extract_metadata"
        ) as mock_extract_meta, patch(
            "trafilatura.settings.use_config"
        ) as mock_config, patch("bs4.BeautifulSoup") as mock_bs:
            # Mock trafilatura
            mock_config.return_value = MagicMock()
            mock_extract.side_effect = ["Text", "# Markdown", "<p>HTML</p>"]

            mock_metadata = MagicMock()
            mock_metadata.title = "Test"
            mock_metadata.author = None
            mock_metadata.date = None
            mock_metadata.description = None
            mock_metadata.sitename = None
            mock_metadata.url = None
            mock_metadata.language = "en"
            mock_metadata.categories = []
            mock_metadata.tags = []
            mock_extract_meta.return_value = mock_metadata

            # Mock BeautifulSoup for structure
            mock_soup = MagicMock()
            mock_soup.find_all.return_value = []
            mock_bs.return_value = mock_soup

            result = processor.process(html_file)

            assert isinstance(result, ExtractedHTML)
            assert result.text == "Text"
            assert result.title == "Test"


class TestErrorHandling:
    """Tests for error handling.

    Rule #4: Focused test class - tests error conditions
    """

    def test_regex_fallback(self):
        """Test regex-based fallback when libraries unavailable."""
        processor = HTMLProcessor()

        # Simulate ImportError by patching BeautifulSoup to raise ImportError
        with patch("bs4.BeautifulSoup", side_effect=ImportError):
            html_content = "<html><head><script>alert('test');</script></head><body>Content</body></html>"

            result = processor._extract_content_fallback(html_content)

            # Should extract text and remove scripts
            assert "Content" in result["text"]
            assert "alert" not in result["text"]


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - HTMLSection: 2 tests (basic, with subsections)
    - ExtractedHTML: 1 test (creation)
    - File detection: 5 tests (html, htm, mhtml, xhtml, pdf)
    - File reading: 3 tests (utf8, latin1, bom)
    - Content extraction: 4 tests (trafilatura, tables, links, fallback)
    - Metadata: 4 tests (basic, multiple authors, fallback title, og tags)
    - Structure: 3 tests (basic, hierarchy, empty headings)
    - Tables: 3 tests (enabled, multiple, none)
    - Citation: 2 tests (with identifiers, without)
    - Parse authors: 6 tests (single, comma, and, ampersand, none, empty)
    - Process: 1 test (basic flow)
    - Error handling: 1 test (regex fallback)

    Total: 35 tests

Design Decisions:
    1. Mock trafilatura and BeautifulSoup to avoid dependencies
    2. Test all extraction modes (normal, fallback)
    3. Cover metadata sources (trafilatura, meta tags, Open Graph)
    4. Test author parsing variations
    5. Verify structure hierarchy building
    6. Test table integration
    7. Test citation metadata enrichment

Behaviors Tested:
    - File type detection for HTML variants
    - Encoding detection and handling
    - Content extraction with trafilatura
    - Fallback extraction with BeautifulSoup
    - Metadata extraction from multiple sources
    - Author name parsing
    - Document structure extraction
    - Heading hierarchy building
    - Table extraction and formatting
    - Citation metadata (DOI, arXiv, etc.)
    - Error handling and fallbacks
"""
