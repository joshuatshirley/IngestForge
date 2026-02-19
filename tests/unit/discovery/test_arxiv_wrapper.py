"""
Tests for ArxivDiscovery wrapper using the arxiv library.

This module tests the ArxivDiscovery class including search,
paper lookup, and PDF download functionality.

Test Strategy
-------------
- Mock the arxiv library to avoid actual API calls
- Test ArxivPaper dataclass methods
- Test search functionality with result limits
- Test paper structure validation
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestArxivPaperDataclass: ArxivPaper model tests
- TestArxivDiscoverySearch: Search functionality
- TestArxivDiscoveryGetPaper: Single paper lookup
- TestArxivDiscoveryDownload: PDF download
- TestArxivIdCleaning: ID validation and cleaning
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_arxiv_result():
    """Create a mock arxiv.Result object."""
    mock_result = MagicMock()
    mock_result.entry_id = "http://arxiv.org/abs/2301.12345v1"
    mock_result.title = "Attention Is All You Need"
    mock_result.summary = (
        "We propose a new simple network architecture, the Transformer."
    )
    mock_result.published = datetime(2023, 1, 15, 12, 0, 0)
    mock_result.updated = datetime(2023, 1, 20, 12, 0, 0)
    mock_result.pdf_url = "https://arxiv.org/pdf/2301.12345.pdf"
    mock_result.doi = "10.1234/example"
    mock_result.primary_category = "cs.CL"
    mock_result.categories = ["cs.CL", "cs.LG"]
    mock_result.comment = "15 pages, 5 figures"
    mock_result.journal_ref = "Nature 2023"

    # Mock authors
    author1 = MagicMock()
    author1.name = "Ashish Vaswani"
    author2 = MagicMock()
    author2.name = "Noam Shazeer"
    mock_result.authors = [author1, author2]

    return mock_result


def make_arxiv_paper():
    """Create a test ArxivPaper object."""
    from ingestforge.discovery.arxiv_wrapper import ArxivPaper

    return ArxivPaper(
        arxiv_id="2301.12345",
        title="Test Paper Title",
        authors=["John Doe", "Jane Smith"],
        abstract="This is a test abstract.",
        published_date=datetime(2023, 1, 15),
        updated_date=datetime(2023, 1, 20),
        categories=["cs.CL", "cs.LG"],
        pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
        doi="10.1234/example",
        primary_category="cs.CL",
        comment="15 pages",
        journal_ref=None,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestArxivPaperDataclass:
    """Tests for ArxivPaper dataclass.

    Rule #4: Focused test class
    """

    def test_paper_creation(self):
        """Test creating ArxivPaper instance."""
        paper = make_arxiv_paper()

        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.primary_category == "cs.CL"
        assert paper.doi == "10.1234/example"

    def test_paper_to_dict(self):
        """Test ArxivPaper to_dict conversion."""
        paper = make_arxiv_paper()
        paper_dict = paper.to_dict()

        assert paper_dict["arxiv_id"] == "2301.12345"
        assert paper_dict["title"] == "Test Paper Title"
        assert paper_dict["authors"] == ["John Doe", "Jane Smith"]
        assert "2023-01-15" in paper_dict["published_date"]
        assert paper_dict["doi"] == "10.1234/example"

    def test_paper_optional_fields_none(self):
        """Test ArxivPaper with None optional fields."""
        from ingestforge.discovery.arxiv_wrapper import ArxivPaper

        paper = ArxivPaper(
            arxiv_id="2301.00001",
            title="Minimal Paper",
            authors=["Author"],
            abstract="Abstract",
            published_date=datetime(2023, 1, 1),
            updated_date=datetime(2023, 1, 1),
            categories=["cs.AI"],
            pdf_url="https://arxiv.org/pdf/2301.00001.pdf",
            doi=None,
            primary_category=None,
            comment=None,
            journal_ref=None,
        )

        paper_dict = paper.to_dict()
        assert paper_dict["doi"] is None
        assert paper_dict["primary_category"] is None


class TestArxivDiscoverySearch:
    """Tests for ArxivDiscovery search functionality.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_search_returns_papers(self, mock_check):
        """Test successful search returns ArxivPaper list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            # Setup mock search results
            mock_result = make_mock_arxiv_result()
            mock_client = MagicMock()
            mock_client.results.return_value = [mock_result]
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            papers = discovery.search("transformer attention", max_results=10)

            assert len(papers) == 1
            assert papers[0].title == "Attention Is All You Need"
            assert papers[0].arxiv_id == "2301.12345"
            assert "Ashish Vaswani" in papers[0].authors

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_search_empty_query_returns_empty(self, mock_check):
        """Test empty query returns empty list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            papers = discovery.search("")

            assert papers == []

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_search_whitespace_query_returns_empty(self, mock_check):
        """Test whitespace-only query returns empty list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            papers = discovery.search("   ")

            assert papers == []

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_search_respects_max_results_limit(self, mock_check):
        """Test search max_results is clamped to valid range."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            mock_client = MagicMock()
            mock_client.results.return_value = []
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()

            # Test upper limit clamping
            discovery.search("test", max_results=500)
            call_args = mock_arxiv.Search.call_args
            assert call_args.kwargs["max_results"] == 300

            # Test lower limit clamping
            discovery.search("test", max_results=0)
            call_args = mock_arxiv.Search.call_args
            assert call_args.kwargs["max_results"] == 1

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_search_handles_exception(self, mock_check):
        """Test search handles exceptions gracefully."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            mock_client = MagicMock()
            mock_client.results.side_effect = Exception("Network error")
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            papers = discovery.search("test query")

            assert papers == []


class TestArxivDiscoveryGetPaper:
    """Tests for ArxivDiscovery get_paper functionality.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_get_paper_by_id(self, mock_check):
        """Test fetching paper by arXiv ID."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            mock_result = make_mock_arxiv_result()
            mock_client = MagicMock()
            mock_client.results.return_value = [mock_result]
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = discovery.get_paper("2301.12345")

            assert paper is not None
            assert paper.arxiv_id == "2301.12345"

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_get_paper_empty_id_returns_none(self, mock_check):
        """Test get_paper with empty ID returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = discovery.get_paper("")

            assert paper is None

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_get_paper_invalid_id_returns_none(self, mock_check):
        """Test get_paper with invalid ID format returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = discovery.get_paper("invalid-format")

            assert paper is None

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_get_paper_not_found_returns_none(self, mock_check):
        """Test get_paper returns None when paper not found."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            mock_client = MagicMock()
            mock_client.results.return_value = []  # No results
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = discovery.get_paper("2301.99999")

            assert paper is None


class TestArxivDiscoveryDownload:
    """Tests for ArxivDiscovery PDF download functionality.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_download_pdf_success(self, mock_check, tmp_path):
        """Test successful PDF download."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            # Setup mock download
            mock_result = MagicMock()
            expected_path = str(tmp_path / "2301.12345.pdf")
            mock_result.download_pdf.return_value = expected_path

            mock_client = MagicMock()
            mock_client.results.return_value = [mock_result]
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = make_arxiv_paper()
            result = discovery.download_pdf(paper, tmp_path)

            assert result is not None
            assert str(result) == expected_path

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_download_pdf_invalid_paper_returns_none(self, mock_check, tmp_path):
        """Test download with invalid paper returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            result = discovery.download_pdf(None, tmp_path)

            assert result is None

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_download_pdf_creates_output_dir(self, mock_check, tmp_path):
        """Test download creates output directory if needed."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            mock_result = MagicMock()
            new_dir = tmp_path / "new_subdir"
            expected_path = str(new_dir / "2301.12345.pdf")
            mock_result.download_pdf.return_value = expected_path

            mock_client = MagicMock()
            mock_client.results.return_value = [mock_result]
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = make_arxiv_paper()
            result = discovery.download_pdf(paper, new_dir)

            assert new_dir.exists()

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_download_pdf_handles_exception(self, mock_check, tmp_path):
        """Test download handles exceptions gracefully."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]

            mock_client = MagicMock()
            mock_client.results.side_effect = Exception("Download failed")
            mock_arxiv.Client.return_value = mock_client
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            paper = make_arxiv_paper()
            result = discovery.download_pdf(paper, tmp_path)

            assert result is None


class TestArxivIdCleaning:
    """Tests for arXiv ID cleaning and validation.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_clean_simple_id(self, mock_check):
        """Test cleaning simple arXiv ID."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            assert discovery._clean_arxiv_id("2301.12345") == "2301.12345"

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_clean_id_with_prefix(self, mock_check):
        """Test cleaning arXiv ID with arxiv: prefix."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            assert discovery._clean_arxiv_id("arxiv:2301.12345") == "2301.12345"

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_clean_id_from_url(self, mock_check):
        """Test extracting arXiv ID from URL."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            url = "https://arxiv.org/abs/2301.12345v2"
            assert discovery._clean_arxiv_id(url) == "2301.12345"

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_clean_old_format_id(self, mock_check):
        """Test cleaning old-format arXiv ID."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            assert discovery._clean_arxiv_id("cond-mat/0701234") == "cond-mat/0701234"

    @patch("ingestforge.discovery.arxiv_wrapper._check_arxiv_library")
    def test_clean_invalid_id_returns_none(self, mock_check):
        """Test invalid arXiv ID returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"arxiv": MagicMock()}):
            import sys

            mock_arxiv = sys.modules["arxiv"]
            mock_arxiv.Client.return_value = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            discovery = ArxivDiscovery()
            assert discovery._clean_arxiv_id("invalid") is None
            assert discovery._clean_arxiv_id("http://example.com") is None


class TestArxivDiscoveryInitialization:
    """Tests for ArxivDiscovery initialization.

    Rule #4: Focused test class
    """

    def test_init_raises_without_arxiv_library(self):
        """Test initialization raises ImportError without arxiv library."""
        with patch(
            "ingestforge.discovery.arxiv_wrapper._check_arxiv_library",
            return_value=False,
        ):
            from ingestforge.discovery.arxiv_wrapper import ArxivDiscovery

            with pytest.raises(ImportError) as exc_info:
                ArxivDiscovery()

            assert "arxiv library not installed" in str(exc_info.value)


class TestConvertResultToPaper:
    """Tests for result conversion helper.

    Rule #4: Focused test class
    """

    def test_convert_extracts_all_fields(self):
        """Test conversion extracts all metadata fields."""
        from ingestforge.discovery.arxiv_wrapper import _convert_result_to_paper

        mock_result = make_mock_arxiv_result()
        paper = _convert_result_to_paper(mock_result)

        assert paper.arxiv_id == "2301.12345"
        assert paper.title == "Attention Is All You Need"
        assert len(paper.authors) == 2
        assert "Ashish Vaswani" in paper.authors
        assert (
            paper.abstract
            == "We propose a new simple network architecture, the Transformer."
        )
        assert paper.doi == "10.1234/example"
        assert paper.primary_category == "cs.CL"
        assert "cs.CL" in paper.categories
        assert "cs.LG" in paper.categories
        assert paper.comment == "15 pages, 5 figures"
        assert paper.journal_ref == "Nature 2023"

    def test_convert_handles_missing_pdf_url(self):
        """Test conversion handles missing PDF URL."""
        from ingestforge.discovery.arxiv_wrapper import _convert_result_to_paper

        mock_result = make_mock_arxiv_result()
        mock_result.pdf_url = None  # Missing PDF URL
        paper = _convert_result_to_paper(mock_result)

        # Should generate PDF URL from arxiv_id
        assert "2301.12345" in paper.pdf_url
        assert paper.pdf_url.endswith(".pdf")

    def test_convert_strips_version_from_id(self):
        """Test conversion strips version suffix from ID."""
        from ingestforge.discovery.arxiv_wrapper import _convert_result_to_paper

        mock_result = make_mock_arxiv_result()
        mock_result.entry_id = "http://arxiv.org/abs/2301.12345v3"
        paper = _convert_result_to_paper(mock_result)

        # Version should be stripped
        assert paper.arxiv_id == "2301.12345"
        assert "v3" not in paper.arxiv_id


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - ArxivPaper dataclass: 3 tests
    - ArxivDiscovery search: 5 tests
    - ArxivDiscovery get_paper: 4 tests
    - PDF download: 4 tests
    - ID cleaning: 5 tests
    - Initialization: 1 test
    - Result conversion: 3 tests

    Total: 25 tests

Design Decisions:
    1. Mock arxiv library to avoid actual API calls
    2. Test ArxivPaper dataclass separately
    3. Test error handling for all failure modes
    4. Verify ID cleaning handles various formats
    5. Test conversion extracts all metadata
"""
