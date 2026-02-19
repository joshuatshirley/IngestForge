"""
Tests for enhanced arXiv client functionality.

This module tests the ArxivSearcher class including search,
paper lookup, PDF download, and BibTeX export.

Test Strategy
-------------
- Mock HTTP requests to avoid actual API calls
- Test rate limiting behavior
- Test Paper dataclass methods
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestPaperDataclass: Paper model tests
- TestArxivSearcher: Search functionality
- TestArxivDownload: PDF download
- TestRateLimiter: Rate limiting
- TestBibTeX: BibTeX export
- TestIDParsing: arXiv ID validation
"""

from datetime import datetime
from unittest.mock import MagicMock, patch


from ingestforge.discovery.arxiv_client import (
    ArxivSearcher,
    Paper,
    SortOrder,
    export_bibtex,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_arxiv_xml() -> bytes:
    """Create mock arXiv Atom XML response."""
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v1</id>
    <title>Attention Is All You Need</title>
    <summary>We propose a new simple network architecture, the Transformer.</summary>
    <published>2023-01-15T12:00:00Z</published>
    <updated>2023-01-20T12:00:00Z</updated>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <author><name>Niki Parmar</name></author>
    <link href="http://arxiv.org/abs/2301.12345v1" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2301.12345v1.pdf" rel="related" type="application/pdf"/>
    <arxiv:primary_category term="cs.CL"/>
    <category term="cs.CL"/>
    <category term="cs.LG"/>
    <arxiv:comment>15 pages, 5 figures</arxiv:comment>
  </entry>
</feed>
"""


def make_paper() -> Paper:
    """Create a test Paper object."""
    return Paper(
        arxiv_id="2301.12345",
        title="Test Paper Title",
        authors=["John Doe", "Jane Smith"],
        abstract="This is a test abstract.",
        published=datetime(2023, 1, 15),
        updated=datetime(2023, 1, 20),
        pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
        abs_url="https://arxiv.org/abs/2301.12345",
        categories=["cs.CL", "cs.LG"],
        primary_category="cs.CL",
        comment="15 pages",
        journal_ref=None,
        doi=None,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestPaperDataclass:
    """Tests for Paper dataclass.

    Rule #4: Focused test class
    """

    def test_paper_creation(self):
        """Test creating Paper instance."""
        paper = make_paper()

        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.primary_category == "cs.CL"

    def test_paper_bibtex_export(self):
        """Test BibTeX export from Paper."""
        paper = make_paper()
        bibtex = paper.to_bibtex()

        assert "@article{" in bibtex
        assert "doe2023" in bibtex.lower()
        assert "title = {Test Paper Title}" in bibtex
        assert "John Doe and Jane Smith" in bibtex
        assert "eprint = {2301.12345}" in bibtex
        assert "archivePrefix = {arXiv}" in bibtex

    def test_paper_bibtex_with_doi(self):
        """Test BibTeX export includes DOI when present."""
        paper = make_paper()
        paper.doi = "10.1234/example"
        bibtex = paper.to_bibtex()

        assert "doi = {10.1234/example}" in bibtex

    def test_paper_bibtex_with_journal_ref(self):
        """Test BibTeX export includes journal reference."""
        paper = make_paper()
        paper.journal_ref = "Nature 2023"
        bibtex = paper.to_bibtex()

        assert "journal = {Nature 2023}" in bibtex


class TestArxivSearcher:
    """Tests for ArxivSearcher class.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_returns_papers(self, mock_urlopen):
        """Test successful search returns Paper list."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_xml()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        papers = searcher.search("transformer attention", limit=10)

        assert len(papers) == 1
        assert papers[0].title == "Attention Is All You Need"
        assert papers[0].arxiv_id == "2301.12345"
        assert "Ashish Vaswani" in papers[0].authors

    @patch("urllib.request.urlopen")
    def test_search_empty_query_returns_empty(self, mock_urlopen):
        """Test empty query returns empty list."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        papers = searcher.search("", limit=10)

        assert papers == []
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_search_handles_network_error(self, mock_urlopen):
        """Test search handles network errors gracefully."""
        mock_urlopen.side_effect = Exception("Network error")

        searcher = ArxivSearcher(rate_limit_delay=0)
        papers = searcher.search("test query")

        assert papers == []

    @patch("urllib.request.urlopen")
    def test_search_limit_clamped(self, mock_urlopen):
        """Test search limit is clamped to valid range."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_xml()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)

        # Test upper limit
        searcher.search("test", limit=200)
        call_url = mock_urlopen.call_args[0][0].full_url
        assert "max_results=100" in call_url

    @patch("urllib.request.urlopen")
    def test_search_sort_order(self, mock_urlopen):
        """Test sort order parameter."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_xml()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        searcher.search("test", sort=SortOrder.SUBMITTED)

        call_url = mock_urlopen.call_args[0][0].full_url
        assert "sortBy=submittedDate" in call_url

    @patch("urllib.request.urlopen")
    def test_get_paper_by_id(self, mock_urlopen):
        """Test fetching paper by arXiv ID."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_xml()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        paper = searcher.get_paper("2301.12345")

        assert paper is not None
        assert paper.arxiv_id == "2301.12345"

    @patch("urllib.request.urlopen")
    def test_get_paper_invalid_id(self, mock_urlopen):
        """Test get_paper with invalid ID returns None."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        paper = searcher.get_paper("invalid")

        assert paper is None
        mock_urlopen.assert_not_called()


class TestArxivDownload:
    """Tests for arXiv PDF download.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_download_pdf_success(self, mock_urlopen, tmp_path):
        """Test successful PDF download."""
        # Mock PDF content
        mock_response = MagicMock()
        mock_response.read.side_effect = [b"PDF content", b""]
        mock_response.headers = {"Content-Type": "application/pdf"}
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        result = searcher.download_pdf("2301.12345", tmp_path)

        assert result.success is True
        assert result.arxiv_id == "2301.12345"
        assert result.file_path is not None

    def test_download_pdf_invalid_id(self, tmp_path):
        """Test download with invalid ID."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        result = searcher.download_pdf("invalid", tmp_path)

        assert result.success is False
        assert "Invalid" in result.error

    @patch("urllib.request.urlopen")
    def test_download_pdf_network_error(self, mock_urlopen, tmp_path):
        """Test download handles network error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "http://arxiv.org/pdf/2301.12345.pdf",
            404,
            "Not Found",
            {},
            None,
        )

        searcher = ArxivSearcher(rate_limit_delay=0)
        result = searcher.download_pdf("2301.12345", tmp_path)

        assert result.success is False
        assert "404" in result.error


class TestRateLimiter:
    """Tests for rate limiting.

    Rule #4: Focused test class
    """

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_rate_limiter_delays_requests(self, mock_urlopen, mock_sleep):
        """Test rate limiter sleeps between requests."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_xml()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=3.0)

        # First call
        searcher.search("query 1")
        # Second call - should trigger sleep
        searcher.search("query 2")

        # Verify sleep was called
        assert mock_sleep.called


class TestBibTeX:
    """Tests for BibTeX export.

    Rule #4: Focused test class
    """

    def test_export_bibtex_multiple_papers(self):
        """Test exporting multiple papers to BibTeX."""
        papers = [
            make_paper(),
            Paper(
                arxiv_id="2302.54321",
                title="Another Paper",
                authors=["Alice Brown"],
                abstract="Another abstract",
                published=datetime(2023, 2, 1),
                updated=datetime(2023, 2, 1),
                pdf_url="https://arxiv.org/pdf/2302.54321.pdf",
                abs_url="https://arxiv.org/abs/2302.54321",
                categories=["cs.AI"],
                primary_category="cs.AI",
            ),
        ]

        bibtex = export_bibtex(papers)

        assert "@article{doe2023" in bibtex.lower()
        assert "@article{brown2023" in bibtex.lower()
        assert bibtex.count("@article{") == 2

    def test_export_bibtex_empty_list(self):
        """Test exporting empty list returns empty string."""
        bibtex = export_bibtex([])
        assert bibtex == ""


class TestIDParsing:
    """Tests for arXiv ID parsing and validation.

    Rule #4: Focused test class
    """

    def test_clean_arxiv_id_simple(self):
        """Test cleaning simple arXiv ID."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        assert searcher._clean_arxiv_id("2301.12345") == "2301.12345"

    def test_clean_arxiv_id_with_prefix(self):
        """Test cleaning arXiv ID with arxiv: prefix."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        assert searcher._clean_arxiv_id("arxiv:2301.12345") == "2301.12345"

    def test_clean_arxiv_id_from_url(self):
        """Test extracting arXiv ID from URL."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        url = "https://arxiv.org/abs/2301.12345v2"
        assert searcher._clean_arxiv_id(url) == "2301.12345"

    def test_clean_arxiv_id_old_format(self):
        """Test cleaning old-format arXiv ID."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        assert searcher._clean_arxiv_id("cond-mat/0701234") == "cond-mat/0701234"

    def test_clean_arxiv_id_invalid(self):
        """Test invalid arXiv ID returns None."""
        searcher = ArxivSearcher(rate_limit_delay=0)
        assert searcher._clean_arxiv_id("invalid") is None
        assert searcher._clean_arxiv_id("") is None
        assert searcher._clean_arxiv_id("http://example.com/paper") is None


class TestXMLParsing:
    """Tests for XML response parsing.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_parse_entry_extracts_all_fields(self, mock_urlopen):
        """Test that entry parsing extracts all fields."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_xml()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        papers = searcher.search("test")

        assert len(papers) == 1
        paper = papers[0]

        assert paper.title == "Attention Is All You Need"
        assert len(paper.authors) == 3
        assert paper.primary_category == "cs.CL"
        assert "cs.CL" in paper.categories
        assert "cs.LG" in paper.categories
        assert paper.comment == "15 pages, 5 figures"

    @patch("urllib.request.urlopen")
    def test_parse_handles_missing_optional_fields(self, mock_urlopen):
        """Test parsing handles missing optional fields."""
        minimal_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001</id>
    <title>Minimal Paper</title>
    <summary>Abstract</summary>
    <published>2023-01-01T00:00:00Z</published>
    <updated>2023-01-01T00:00:00Z</updated>
  </entry>
</feed>
"""
        mock_response = MagicMock()
        mock_response.read.return_value = minimal_xml
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        papers = searcher.search("test")

        assert len(papers) == 1
        paper = papers[0]

        assert paper.title == "Minimal Paper"
        assert paper.authors == []
        assert paper.comment is None
        assert paper.doi is None

    @patch("urllib.request.urlopen")
    def test_parse_invalid_xml_returns_empty(self, mock_urlopen):
        """Test invalid XML returns empty list."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"<invalid xml"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        searcher = ArxivSearcher(rate_limit_delay=0)
        papers = searcher.search("test")

        assert papers == []


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Paper dataclass: 4 tests
    - ArxivSearcher search: 7 tests
    - PDF download: 3 tests
    - Rate limiting: 1 test
    - BibTeX export: 2 tests
    - ID parsing: 5 tests
    - XML parsing: 3 tests

    Total: 25 tests

Design Decisions:
    1. Mock all HTTP requests
    2. Test Paper dataclass separately
    3. Test error handling for all failure modes
    4. Verify rate limiting behavior
    5. Test BibTeX formatting
"""
