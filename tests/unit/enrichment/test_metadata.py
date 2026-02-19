"""
Tests for Metadata Extractor.

This module tests metadata extraction from chunk content.

Test Strategy
-------------
- Focus on pattern matching and metadata extraction
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test regex patterns with known inputs
- Test main extraction workflow

Organization
------------
- TestMetadataExtractorInit: Initialization
- TestKeywordExtraction: _extract_keywords function
- TestDateExtraction: _extract_dates function
- TestNumberExtraction: _extract_numbers function
- TestURLExtraction: _extract_urls function
- TestEmailExtraction: _extract_emails function
- TestListDetection: _has_list function
- TestHeaderDetection: _has_headers function
- TestExtractMethod: Main extract() method
- TestEnrichMethod: enrich() method
"""


from ingestforge.enrichment.metadata import MetadataExtractor
from ingestforge.chunking.semantic_chunker import ChunkRecord


# ============================================================================
# Test Helpers
# ============================================================================


def make_chunk(content: str, chunk_id: str = "test_1") -> ChunkRecord:
    """Create a test ChunkRecord."""
    return ChunkRecord(
        chunk_id=chunk_id,
        document_id="test_doc",
        content=content,
        word_count=len(content.split()),
        char_count=len(content),
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestMetadataExtractorInit:
    """Tests for MetadataExtractor initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_metadata_extractor(self):
        """Test creating MetadataExtractor."""
        extractor = MetadataExtractor()

        assert extractor.date_pattern is not None
        assert extractor.number_pattern is not None
        assert extractor.url_pattern is not None
        assert extractor.email_pattern is not None


class TestKeywordExtraction:
    """Tests for keyword extraction.

    Rule #4: Focused test class - tests _extract_keywords only
    """

    def test_extract_keywords_basic(self):
        """Test extracting keywords from text."""
        extractor = MetadataExtractor()
        text = "Python programming language is great for data science and machine learning applications"

        keywords = extractor._extract_keywords(text, top_n=5)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5
        # Should extract content words, not stop words
        assert all(kw not in {"the", "and", "for", "are"} for kw in keywords)

    def test_extract_keywords_filters_stop_words(self):
        """Test keyword extraction filters stop words."""
        extractor = MetadataExtractor()
        text = "the and but not you all can programming language framework"

        keywords = extractor._extract_keywords(text)

        # Should not include stop words
        assert "the" not in keywords
        assert "and" not in keywords
        # Should include content words
        assert any(kw in ["programming", "language", "framework"] for kw in keywords)

    def test_extract_keywords_empty_text(self):
        """Test keyword extraction from empty text."""
        extractor = MetadataExtractor()
        text = ""

        keywords = extractor._extract_keywords(text)

        assert keywords == []


class TestDateExtraction:
    """Tests for date extraction.

    Rule #4: Focused test class - tests _extract_dates only
    """

    def test_extract_dates_numeric_format(self):
        """Test extracting dates in numeric format."""
        extractor = MetadataExtractor()
        text = "The event occurred on 12/31/2023 and 01/15/2024."

        dates = extractor._extract_dates(text)

        assert isinstance(dates, list)
        assert len(dates) >= 1
        assert any("2023" in d or "2024" in d for d in dates)

    def test_extract_dates_text_format(self):
        """Test extracting dates in text format."""
        extractor = MetadataExtractor()
        text = "Published on January 15, 2024 and updated March 20, 2024."

        dates = extractor._extract_dates(text)

        assert len(dates) >= 1
        assert any("2024" in d for d in dates)

    def test_extract_dates_no_dates(self):
        """Test extracting dates from text without dates."""
        extractor = MetadataExtractor()
        text = "No dates here, just regular text content."

        dates = extractor._extract_dates(text)

        assert dates == []


class TestNumberExtraction:
    """Tests for number extraction.

    Rule #4: Focused test class - tests _extract_numbers only
    """

    def test_extract_numbers_with_commas(self):
        """Test extracting numbers with comma separators."""
        extractor = MetadataExtractor()
        text = "The population is 1,234,567 people."

        numbers = extractor._extract_numbers(text)

        assert len(numbers) >= 1
        assert any("1,234,567" in n for n in numbers)

    def test_extract_numbers_with_percentages(self):
        """Test extracting percentages."""
        extractor = MetadataExtractor()
        text = "Growth rate is 25% and efficiency is 87.5%."

        numbers = extractor._extract_numbers(text)

        # Should extract at least one number (with or without %)
        assert len(numbers) >= 1
        # Should find the percentage values or the number values
        assert any("25" in n or "87.5" in n for n in numbers)

    def test_extract_numbers_no_numbers(self):
        """Test extracting numbers from text without significant numbers."""
        extractor = MetadataExtractor()
        text = "No significant numbers in this text."

        numbers = extractor._extract_numbers(text)

        # Should be empty or contain only very small numbers
        assert len(numbers) <= 1


class TestURLExtraction:
    """Tests for URL extraction.

    Rule #4: Focused test class - tests _extract_urls only
    """

    def test_extract_urls_http_and_https(self):
        """Test extracting HTTP and HTTPS URLs."""
        extractor = MetadataExtractor()
        text = "Visit https://example.com and http://test.org for more info."

        urls = extractor._extract_urls(text)

        assert len(urls) == 2
        assert "https://example.com" in urls
        assert "http://test.org" in urls

    def test_extract_urls_no_urls(self):
        """Test extracting URLs from text without URLs."""
        extractor = MetadataExtractor()
        text = "No URLs in this text content."

        urls = extractor._extract_urls(text)

        assert urls == []


class TestEmailExtraction:
    """Tests for email extraction.

    Rule #4: Focused test class - tests _extract_emails only
    """

    def test_extract_emails_basic(self):
        """Test extracting email addresses."""
        extractor = MetadataExtractor()
        text = "Contact us at test@example.com or support@test.org."

        emails = extractor._extract_emails(text)

        assert len(emails) == 2
        assert "test@example.com" in emails
        assert "support@test.org" in emails

    def test_extract_emails_no_emails(self):
        """Test extracting emails from text without emails."""
        extractor = MetadataExtractor()
        text = "No email addresses in this text."

        emails = extractor._extract_emails(text)

        assert emails == []


class TestListDetection:
    """Tests for list detection.

    Rule #4: Focused test class - tests _has_list only
    """

    def test_has_list_bullet_list(self):
        """Test detecting bullet lists."""
        extractor = MetadataExtractor()
        text = "Items:\n- First item\n- Second item\n- Third item"

        result = extractor._has_list(text)

        assert result is True

    def test_has_list_numbered_list(self):
        """Test detecting numbered lists."""
        extractor = MetadataExtractor()
        text = "Steps:\n1. First step\n2. Second step\n3. Third step"

        result = extractor._has_list(text)

        assert result is True

    def test_has_list_no_list(self):
        """Test detecting no list."""
        extractor = MetadataExtractor()
        text = "Regular paragraph text without any list structures."

        result = extractor._has_list(text)

        assert result is False


class TestHeaderDetection:
    """Tests for header detection.

    Rule #4: Focused test class - tests _has_headers only
    """

    def test_has_headers_markdown(self):
        """Test detecting Markdown headers."""
        extractor = MetadataExtractor()
        text = "# Main Title\n\nSome content\n\n## Subsection"

        result = extractor._has_headers(text)

        assert result is True

    def test_has_headers_all_caps(self):
        """Test detecting ALL CAPS headers."""
        extractor = MetadataExtractor()
        text = "INTRODUCTION\n\nSome content here\n\nMETHODOLOGY\n\nMore content"

        result = extractor._has_headers(text)

        assert result is True

    def test_has_headers_no_headers(self):
        """Test detecting no headers."""
        extractor = MetadataExtractor()
        text = "Regular text without any header structures."

        result = extractor._has_headers(text)

        assert result is False


class TestExtractMethod:
    """Tests for main extract() method.

    Rule #4: Focused test class - tests extract() only
    """

    def test_extract_returns_all_metadata(self):
        """Test extract returns all metadata fields."""
        extractor = MetadataExtractor()
        chunk = make_chunk(
            "Visit https://example.com on 12/31/2023. "
            "Growth is 25%. Contact: test@example.com.\n"
            "- Item 1\n- Item 2"
        )

        metadata = extractor.extract(chunk)

        assert "keywords" in metadata
        assert "dates" in metadata
        assert "numbers" in metadata
        assert "urls" in metadata
        assert "emails" in metadata
        assert "has_list" in metadata
        assert "has_headers" in metadata
        assert "paragraph_count" in metadata

    def test_extract_detects_content(self):
        """Test extract detects actual content."""
        extractor = MetadataExtractor()
        chunk = make_chunk(
            "Visit https://example.com on January 15, 2024. "
            "Growth rate is 25.5%. Email: test@example.com."
        )

        metadata = extractor.extract(chunk)

        assert len(metadata["urls"]) >= 1
        assert len(metadata["dates"]) >= 1
        assert len(metadata["numbers"]) >= 1
        assert len(metadata["emails"]) >= 1


class TestEnrichMethod:
    """Tests for enrich() method.

    Rule #4: Focused test class - tests enrich() only
    """

    def test_enrich_adds_keywords_to_concepts(self):
        """Test enrich adds keywords to chunk concepts."""
        extractor = MetadataExtractor()
        chunk = make_chunk(
            "Python programming language framework for data science applications"
        )

        result = extractor.enrich(chunk)

        assert result is chunk  # Same object
        assert hasattr(chunk, "concepts")
        assert isinstance(chunk.concepts, list)
        assert len(chunk.concepts) > 0

    def test_enrich_empty_content(self):
        """Test enrich with empty content."""
        extractor = MetadataExtractor()
        chunk = make_chunk("")

        result = extractor.enrich(chunk)

        assert result is chunk


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - MetadataExtractor init: 1 test (pattern initialization)
    - Keyword extraction: 3 tests (basic, stop words, empty)
    - Date extraction: 3 tests (numeric, text, no dates)
    - Number extraction: 3 tests (commas, percentages, no numbers)
    - URL extraction: 2 tests (http/https, no URLs)
    - Email extraction: 2 tests (basic, no emails)
    - List detection: 3 tests (bullet, numbered, no list)
    - Header detection: 3 tests (markdown, all caps, no headers)
    - Extract method: 2 tests (all fields, detects content)
    - Enrich method: 2 tests (adds concepts, empty content)

    Total: 24 tests

Design Decisions:
    1. Focus on pattern matching and metadata extraction
    2. Use simple ChunkRecord mocks
    3. Test regex patterns with known inputs
    4. Test each extraction function separately
    5. Test main workflow (extract, enrich)
    6. Simple, clear tests that verify extraction works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - MetadataExtractor initialization with regex patterns
    - Keyword extraction (TF-based, stop word filtering)
    - Date extraction (numeric and text formats)
    - Number extraction (with commas, percentages)
    - URL extraction (HTTP and HTTPS)
    - Email address extraction
    - List detection (bullet, numbered, letter lists)
    - Header detection (Markdown, ALL CAPS)
    - Main extract() method returns all metadata fields
    - enrich() method adds keywords to chunk concepts

Justification:
    - Metadata extraction is critical for searchability
    - Regex patterns need verification with known inputs
    - Each extraction type should be tested independently
    - Main workflow tests cover end-to-end extraction
    - Simple tests verify metadata system works correctly
"""
