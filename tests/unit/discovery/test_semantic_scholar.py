"""
Tests for Semantic Scholar client functionality.

This module tests the SemanticScholarClient class including search,
paper lookup, citation traversal, and BibTeX export.

Test Strategy
-------------
- Mock HTTP requests to avoid actual API calls
- Test rate limiting behavior
- Test ScholarPaper dataclass methods
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestScholarPaperDataclass: Paper model tests
- TestSemanticScholarSearch: Search functionality
- TestCitations: Citation and reference retrieval
- TestRateLimiter: Rate limiting
- TestBibTeX: BibTeX export
"""

import json
from unittest.mock import MagicMock, patch


from ingestforge.discovery.semantic_scholar import (
    SemanticScholarClient,
    ScholarPaper,
    Author,
    CitationResult,
    export_bibtex,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_search_response() -> dict:
    """Create mock Semantic Scholar search response."""
    return {
        "total": 2,
        "data": [
            {
                "paperId": "abc123def456",
                "title": "Attention Is All You Need",
                "abstract": "We propose the Transformer architecture.",
                "year": 2017,
                "citationCount": 50000,
                "referenceCount": 37,
                "url": "https://www.semanticscholar.org/paper/abc123",
                "venue": "NeurIPS",
                "publicationDate": "2017-06-12",
                "externalIds": {"DOI": "10.1234/example", "ArXiv": "1706.03762"},
                "fieldsOfStudy": ["Computer Science"],
                "isOpenAccess": True,
                "openAccessPdf": {"url": "https://arxiv.org/pdf/1706.03762.pdf"},
                "authors": [
                    {
                        "authorId": "1234",
                        "name": "Ashish Vaswani",
                        "url": "https://...",
                    },
                    {"authorId": "5678", "name": "Noam Shazeer", "url": "https://..."},
                ],
            },
            {
                "paperId": "xyz789",
                "title": "BERT: Pre-training",
                "abstract": "We introduce BERT.",
                "year": 2018,
                "citationCount": 40000,
                "referenceCount": 55,
                "url": "https://www.semanticscholar.org/paper/xyz789",
                "venue": "NAACL",
                "authors": [
                    {"authorId": "9999", "name": "Jacob Devlin"},
                ],
            },
        ],
    }


def make_mock_paper_response() -> dict:
    """Create mock single paper response."""
    return {
        "paperId": "abc123def456",
        "title": "Attention Is All You Need",
        "abstract": "We propose the Transformer architecture.",
        "year": 2017,
        "citationCount": 50000,
        "referenceCount": 37,
        "url": "https://www.semanticscholar.org/paper/abc123",
        "venue": "NeurIPS",
        "authors": [
            {"authorId": "1234", "name": "Ashish Vaswani"},
        ],
        "fieldsOfStudy": ["Computer Science"],
    }


def make_mock_citations_response() -> dict:
    """Create mock citations response."""
    return {
        "data": [
            {
                "citingPaper": {
                    "paperId": "cite1",
                    "title": "Citing Paper 1",
                    "year": 2019,
                    "citationCount": 100,
                    "referenceCount": 20,
                    "url": "https://...",
                    "authors": [{"name": "Author 1"}],
                },
            },
            {
                "citingPaper": {
                    "paperId": "cite2",
                    "title": "Citing Paper 2",
                    "year": 2020,
                    "citationCount": 50,
                    "referenceCount": 15,
                    "url": "https://...",
                    "authors": [{"name": "Author 2"}],
                },
            },
        ],
    }


def make_scholar_paper() -> ScholarPaper:
    """Create a test ScholarPaper object."""
    return ScholarPaper(
        paper_id="abc123",
        title="Test Paper Title",
        authors=[
            Author(author_id="1234", name="John Doe"),
            Author(author_id="5678", name="Jane Smith"),
        ],
        abstract="This is a test abstract.",
        year=2023,
        citation_count=100,
        reference_count=30,
        url="https://www.semanticscholar.org/paper/abc123",
        venue="Test Conference",
        external_ids={"DOI": "10.1234/test", "ArXiv": "2301.12345"},
        fields_of_study=["Computer Science", "Machine Learning"],
        is_open_access=True,
        open_access_pdf_url="https://arxiv.org/pdf/2301.12345.pdf",
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestScholarPaperDataclass:
    """Tests for ScholarPaper dataclass.

    Rule #4: Focused test class
    """

    def test_paper_creation(self):
        """Test creating ScholarPaper instance."""
        paper = make_scholar_paper()

        assert paper.paper_id == "abc123"
        assert paper.title == "Test Paper Title"
        assert len(paper.authors) == 2
        assert paper.citation_count == 100

    def test_paper_doi_property(self):
        """Test DOI property access."""
        paper = make_scholar_paper()
        assert paper.doi == "10.1234/test"

    def test_paper_arxiv_id_property(self):
        """Test arXiv ID property access."""
        paper = make_scholar_paper()
        assert paper.arxiv_id == "2301.12345"

    def test_paper_bibtex_export(self):
        """Test BibTeX export from ScholarPaper."""
        paper = make_scholar_paper()
        bibtex = paper.to_bibtex()

        assert "@article{" in bibtex
        assert "doe2023" in bibtex.lower()
        assert "title = {Test Paper Title}" in bibtex
        assert "John Doe and Jane Smith" in bibtex

    def test_paper_bibtex_with_venue(self):
        """Test BibTeX includes venue as journal."""
        paper = make_scholar_paper()
        bibtex = paper.to_bibtex()

        assert "journal = {Test Conference}" in bibtex

    def test_author_full_name(self):
        """Test Author name formatting."""
        author = Author(author_id="123", name="John Doe", url="https://...")
        assert author.name == "John Doe"


class TestSemanticScholarSearch:
    """Tests for SemanticScholarClient search.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_returns_papers(self, mock_urlopen):
        """Test successful search returns paper list."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        papers = client.search("transformer attention", limit=10)

        assert len(papers) == 2
        assert papers[0].title == "Attention Is All You Need"
        assert papers[0].citation_count == 50000

    @patch("urllib.request.urlopen")
    def test_search_empty_query_returns_empty(self, mock_urlopen):
        """Test empty query returns empty list."""
        client = SemanticScholarClient()
        papers = client.search("", limit=10)

        assert papers == []
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_search_handles_network_error(self, mock_urlopen):
        """Test search handles network errors gracefully."""
        mock_urlopen.side_effect = Exception("Network error")

        client = SemanticScholarClient()
        papers = client.search("test query")

        assert papers == []

    @patch("urllib.request.urlopen")
    def test_search_with_fields_filter(self, mock_urlopen):
        """Test search with fields of study filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        client.search("test", fields_of_study=["Computer Science"])

        call_url = mock_urlopen.call_args[0][0].full_url
        assert "fieldsOfStudy=Computer+Science" in call_url

    @patch("urllib.request.urlopen")
    def test_search_with_year_range(self, mock_urlopen):
        """Test search with year range filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        client.search("test", year_range=(2020, 2023))

        call_url = mock_urlopen.call_args[0][0].full_url
        assert "year=2020-2023" in call_url


class TestPaperLookup:
    """Tests for single paper lookup.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_get_paper_success(self, mock_urlopen):
        """Test successful paper lookup."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_paper_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        paper = client.get_paper("abc123")

        assert paper is not None
        assert paper.paper_id == "abc123def456"
        assert paper.title == "Attention Is All You Need"

    @patch("urllib.request.urlopen")
    def test_get_paper_not_found(self, mock_urlopen):
        """Test paper not found returns None."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )

        client = SemanticScholarClient()
        paper = client.get_paper("nonexistent")

        assert paper is None

    def test_get_paper_empty_id(self):
        """Test empty paper ID returns None."""
        client = SemanticScholarClient()
        paper = client.get_paper("")

        assert paper is None


class TestCitations:
    """Tests for citation and reference retrieval.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_get_citations_success(self, mock_urlopen):
        """Test successful citation retrieval."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_citations_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        result = client.get_citations("abc123", limit=50, depth=1)

        assert isinstance(result, CitationResult)
        assert result.source_paper_id == "abc123"
        assert len(result.papers) == 2
        assert result.papers[0].title == "Citing Paper 1"

    @patch("urllib.request.urlopen")
    def test_get_citations_empty_id(self, mock_urlopen):
        """Test empty paper ID returns empty result."""
        client = SemanticScholarClient()
        result = client.get_citations("", limit=50)

        assert result.papers == []
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_get_references_success(self, mock_urlopen):
        """Test successful references retrieval."""
        mock_response = MagicMock()
        refs_response = {
            "data": [
                {
                    "citedPaper": {
                        "paperId": "ref1",
                        "title": "Referenced Paper",
                        "year": 2015,
                        "citationCount": 500,
                        "referenceCount": 10,
                        "url": "https://...",
                        "authors": [{"name": "Ref Author"}],
                    },
                },
            ],
        }
        mock_response.read.return_value = json.dumps(refs_response).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        refs = client.get_references("abc123", limit=50)

        assert len(refs) == 1
        assert refs[0].title == "Referenced Paper"


class TestRateLimiter:
    """Tests for rate limiting.

    Rule #4: Focused test class
    """

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_rate_limiter_tracks_requests(self, mock_urlopen, mock_sleep):
        """Test rate limiter tracks request times."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()

        # Make multiple requests
        for _ in range(5):
            client.search("test")

        # Verify urlopen was called
        assert mock_urlopen.call_count == 5


class TestBibTeX:
    """Tests for BibTeX export.

    Rule #4: Focused test class
    """

    def test_export_bibtex_multiple_papers(self):
        """Test exporting multiple papers to BibTeX."""
        papers = [
            make_scholar_paper(),
            ScholarPaper(
                paper_id="xyz789",
                title="Another Paper",
                authors=[Author(author_id="999", name="Alice Brown")],
                abstract="Another abstract",
                year=2022,
                citation_count=50,
                reference_count=20,
                url="https://...",
            ),
        ]

        bibtex = export_bibtex(papers)

        assert bibtex.count("@article{") == 2
        assert "doe2023" in bibtex.lower()
        assert "brown2022" in bibtex.lower()

    def test_export_bibtex_empty_list(self):
        """Test exporting empty list returns empty string."""
        bibtex = export_bibtex([])
        assert bibtex == ""

    def test_bibtex_includes_doi(self):
        """Test BibTeX includes DOI when present."""
        paper = make_scholar_paper()
        bibtex = paper.to_bibtex()

        assert "doi = {10.1234/test}" in bibtex


class TestAPIKey:
    """Tests for API key handling.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_api_key_added_to_headers(self, mock_urlopen):
        """Test API key is added to request headers."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient(api_key="test-api-key")
        client.search("test")

        # Check that request was made with API key header
        request = mock_urlopen.call_args[0][0]
        assert request.headers.get("X-api-key") == "test-api-key"


class TestErrorHandling:
    """Tests for error handling.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_handles_rate_limit_error(self, mock_urlopen):
        """Test handling of 429 rate limit error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, None
        )

        client = SemanticScholarClient()
        papers = client.search("test")

        assert papers == []

    @patch("urllib.request.urlopen")
    def test_handles_malformed_json(self, mock_urlopen):
        """Test handling of malformed JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not valid json"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = SemanticScholarClient()
        papers = client.search("test")

        assert papers == []


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - ScholarPaper dataclass: 6 tests
    - Search functionality: 5 tests
    - Paper lookup: 3 tests
    - Citations/references: 3 tests
    - Rate limiting: 1 test
    - BibTeX export: 3 tests
    - API key handling: 1 test
    - Error handling: 2 tests

    Total: 24 tests

Design Decisions:
    1. Mock all HTTP requests
    2. Test ScholarPaper dataclass separately
    3. Test error handling for all failure modes
    4. Verify API key is passed correctly
    5. Test BibTeX formatting
"""
