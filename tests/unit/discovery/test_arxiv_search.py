"""
Tests for arXiv Search functionality.

This module tests the arXiv API integration including search,
PDF URL generation, and rate limiting.

Test Strategy
-------------
- Mock HTTP requests to avoid actual API calls
- Test rate limiting behavior
- Test PDF URL conversion
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestArxivSearch: Basic search functionality
- TestRateLimiting: Rate limit enforcement
- TestPDFDownload: PDF URL generation and download
- TestErrorHandling: Error scenarios
"""

from unittest.mock import Mock, patch, MagicMock
import time


from ingestforge.discovery.academic_search import (
    search_arxiv,
    get_arxiv_pdf_url,
    download_arxiv_pdf,
    AcademicSource,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_arxiv_response() -> bytes:
    """Create mock arXiv Atom XML response."""
    return b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001v1</id>
    <title>Test Paper Title</title>
    <summary>This is a test abstract.</summary>
    <published>2023-01-01T00:00:00Z</published>
    <author><name>John Doe</name></author>
    <author><name>Jane Smith</name></author>
    <link href="http://arxiv.org/abs/2301.00001v1" rel="alternate" type="text/html"/>
  </entry>
</feed>
"""


def make_mock_academic_source(source_api: str = "arxiv") -> AcademicSource:
    """Create mock AcademicSource."""
    return AcademicSource(
        title="Test Paper",
        authors=["Author 1", "Author 2"],
        abstract="Test abstract",
        url="https://arxiv.org/abs/2301.00001",
        year="2023",
        citation_count=None,
        source_api=source_api,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestArxivSearch:
    """Tests for arXiv search functionality.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_arxiv_returns_results(self, mock_urlopen):
        """Test successful arXiv search."""
        # Setup mock response
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_response()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Execute search
        results = search_arxiv("test query", max_results=10)

        # Verify results
        assert len(results) == 1
        assert results[0].title == "Test Paper Title"
        assert results[0].source_api == "arxiv"
        assert results[0].year == "2023"
        assert len(results[0].authors) == 2

    @patch("urllib.request.urlopen")
    def test_search_arxiv_handles_empty_results(self, mock_urlopen):
        """Test arXiv search with no results."""
        # Setup mock response with empty feed
        empty_response = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>
"""
        mock_response = MagicMock()
        mock_response.read.return_value = empty_response
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Execute search
        results = search_arxiv("no results query")

        # Verify empty results
        assert len(results) == 0

    @patch("urllib.request.urlopen")
    def test_search_arxiv_handles_errors(self, mock_urlopen):
        """Test arXiv search error handling."""
        # Setup mock to raise exception
        mock_urlopen.side_effect = Exception("Network error")

        # Execute search - should not raise, returns empty list
        results = search_arxiv("test query")

        # Verify error handled gracefully
        assert results == []

    @patch("urllib.request.urlopen")
    def test_search_arxiv_formats_url_correctly(self, mock_urlopen):
        """Test that search URL is formatted correctly."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_response()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Execute search with special characters
        search_arxiv("machine learning", max_results=5)

        # Verify URL encoding
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        url = request_obj.full_url

        assert "search_query=all:machine+learning" in url
        assert "max_results=5" in url


class TestRateLimiting:
    """Tests for API rate limiting.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_rate_limiting_enforced(self, mock_urlopen):
        """Test that rate limiting delays API calls."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_response()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Make two rapid calls
        start_time = time.time()
        search_arxiv("query 1", max_results=1)
        search_arxiv("query 2", max_results=1)
        elapsed = time.time() - start_time

        # Should take at least 1 second due to rate limiting
        assert elapsed >= 1.0

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_rate_limiting_sleeps_correctly(self, mock_sleep, mock_urlopen):
        """Test that rate limiter calls sleep with correct duration."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_response()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # First call - no sleep needed
        search_arxiv("query 1")

        # Second call - should sleep
        search_arxiv("query 2")

        # Verify sleep was called
        assert mock_sleep.called


class TestPDFDownload:
    """Tests for PDF URL generation and download.

    Rule #4: Focused test class
    """

    def test_get_arxiv_pdf_url_converts_abs_to_pdf(self):
        """Test converting abstract URL to PDF URL."""
        abs_url = "https://arxiv.org/abs/2301.00001"
        pdf_url = get_arxiv_pdf_url(abs_url)

        assert pdf_url == "https://arxiv.org/pdf/2301.00001.pdf"

    def test_get_arxiv_pdf_url_handles_version(self):
        """Test PDF URL generation with version number."""
        abs_url = "https://arxiv.org/abs/2301.00001v2"
        pdf_url = get_arxiv_pdf_url(abs_url)

        assert pdf_url == "https://arxiv.org/pdf/2301.00001v2.pdf"

    def test_get_arxiv_pdf_url_rejects_non_arxiv(self):
        """Test that non-arXiv URLs return None."""
        non_arxiv_url = "https://example.com/paper.pdf"
        pdf_url = get_arxiv_pdf_url(non_arxiv_url)

        assert pdf_url is None

    def test_get_arxiv_pdf_url_handles_already_pdf(self):
        """Test URL that already has .pdf extension."""
        pdf_url_input = "https://arxiv.org/pdf/2301.00001.pdf"
        pdf_url = get_arxiv_pdf_url(pdf_url_input)

        # Should return same URL (idempotent)
        assert pdf_url == pdf_url_input

    @patch("ingestforge.ingest.pdf_downloader.download_pdf")
    def test_download_arxiv_pdf_success(self, mock_download, tmp_path):
        """Test successful PDF download."""
        # Setup mock
        mock_result = Mock()
        mock_result.success = True
        mock_result.file_path = tmp_path / "paper.pdf"
        mock_download.return_value = mock_result

        # Create paper source
        paper = make_mock_academic_source(source_api="arxiv")

        # Download
        result = download_arxiv_pdf(paper, tmp_path)

        # Verify
        assert result == mock_result.file_path
        mock_download.assert_called_once()

    @patch("ingestforge.ingest.pdf_downloader.download_pdf")
    def test_download_arxiv_pdf_failure(self, mock_download, tmp_path):
        """Test failed PDF download."""
        # Setup mock
        mock_result = Mock()
        mock_result.success = False
        mock_result.error = "Network error"
        mock_download.return_value = mock_result

        # Create paper source
        paper = make_mock_academic_source(source_api="arxiv")

        # Download - should return None on failure
        result = download_arxiv_pdf(paper, tmp_path)

        # Verify
        assert result is None

    def test_download_arxiv_pdf_rejects_non_arxiv(self, tmp_path):
        """Test that non-arXiv papers are rejected."""
        # Create non-arXiv paper
        paper = make_mock_academic_source(source_api="semantic_scholar")

        # Should return None for non-arXiv
        result = download_arxiv_pdf(paper, tmp_path)

        assert result is None


class TestErrorHandling:
    """Tests for error handling.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_handles_invalid_xml(self, mock_urlopen):
        """Test handling of malformed XML response."""
        # Setup mock with invalid XML
        mock_response = MagicMock()
        mock_response.read.return_value = b"<invalid>xml<"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Should not raise, returns empty list
        results = search_arxiv("test query")

        assert results == []

    @patch("urllib.request.urlopen")
    def test_handles_timeout(self, mock_urlopen):
        """Test handling of request timeout."""
        # Setup mock to raise timeout
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Timeout")

        # Should not raise, returns empty list
        results = search_arxiv("test query")

        assert results == []

    @patch("urllib.request.urlopen")
    def test_handles_missing_fields(self, mock_urlopen):
        """Test handling of XML with missing fields."""
        # Setup mock with minimal XML
        minimal_xml = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.00001</id>
    <title>Minimal Paper</title>
  </entry>
</feed>
"""
        mock_response = MagicMock()
        mock_response.read.return_value = minimal_xml
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Should not raise, returns results with missing fields as empty
        results = search_arxiv("test query")

        assert len(results) == 1
        assert results[0].title == "Minimal Paper"
        assert results[0].authors == []  # Missing authors
        assert results[0].abstract == ""  # Missing summary


class TestIntegration:
    """Integration tests for complete workflow.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_and_extract_pdf_url(self, mock_urlopen):
        """Test complete workflow: search -> extract PDF URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = make_mock_arxiv_response()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        # Search
        results = search_arxiv("test query", max_results=10)

        # Extract PDF URL from first result
        assert len(results) > 0
        pdf_url = get_arxiv_pdf_url(results[0].url)

        # Verify PDF URL format
        assert pdf_url is not None
        assert "arxiv.org/pdf/" in pdf_url
        assert pdf_url.endswith(".pdf")
