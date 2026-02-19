"""
Tests for CrossRef client functionality.

This module tests the CrossRefClient class including DOI lookup,
search, reference extraction, and BibTeX export.

Test Strategy
-------------
- Mock HTTP requests to avoid actual API calls
- Test DOI validation and cleaning
- Test Publication dataclass methods
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestPublicationDataclass: Publication model tests
- TestDOILookup: DOI lookup functionality
- TestSearch: Search functionality
- TestReferences: Reference retrieval
- TestBibTeX: BibTeX export
- TestDOICleaning: DOI validation
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch


from ingestforge.discovery.crossref import (
    CrossRefClient,
    Publication,
    Author,
    PublicationType,
    export_bibtex,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_work_response() -> dict:
    """Create mock CrossRef work response."""
    return {
        "status": "ok",
        "message": {
            "DOI": "10.1038/nature12373",
            "title": ["A Test Publication Title"],
            "author": [
                {"given": "John", "family": "Doe", "sequence": "first"},
                {"given": "Jane", "family": "Smith", "sequence": "additional"},
            ],
            "container-title": ["Nature"],
            "publisher": "Springer Nature",
            "published": {"date-parts": [[2023, 6, 15]]},
            "type": "journal-article",
            "abstract": "<p>This is an abstract with HTML.</p>",
            "ISSN": ["0028-0836", "1476-4687"],
            "volume": "598",
            "issue": "7880",
            "page": "123-128",
            "references-count": 45,
            "is-referenced-by-count": 500,
            "subject": ["General"],
            "license": [{"URL": "https://creativecommons.org/licenses/by/4.0/"}],
        },
    }


def make_mock_search_response() -> dict:
    """Create mock CrossRef search response."""
    return {
        "status": "ok",
        "message": {
            "total-results": 2,
            "items": [
                {
                    "DOI": "10.1038/nature12373",
                    "title": ["Publication One"],
                    "author": [{"given": "John", "family": "Doe"}],
                    "container-title": ["Nature"],
                    "publisher": "Springer Nature",
                    "published": {"date-parts": [[2023, 1, 1]]},
                    "type": "journal-article",
                    "references-count": 30,
                    "is-referenced-by-count": 100,
                },
                {
                    "DOI": "10.1234/example",
                    "title": ["Publication Two"],
                    "author": [{"given": "Jane", "family": "Smith"}],
                    "container-title": ["Science"],
                    "publisher": "AAAS",
                    "published": {"date-parts": [[2022, 6, 1]]},
                    "type": "journal-article",
                    "references-count": 25,
                    "is-referenced-by-count": 75,
                },
            ],
        },
    }


def make_publication() -> Publication:
    """Create a test Publication object."""
    return Publication(
        doi="10.1038/nature12373",
        title="Test Publication Title",
        authors=[
            Author(given="John", family="Doe", sequence="first"),
            Author(given="Jane", family="Smith", sequence="additional"),
        ],
        container_title="Nature",
        publisher="Springer Nature",
        published_date=datetime(2023, 6, 15),
        publication_type="journal-article",
        url="https://doi.org/10.1038/nature12373",
        abstract="This is a test abstract.",
        volume="598",
        issue="7880",
        pages="123-128",
        reference_count=45,
        is_referenced_by_count=500,
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestPublicationDataclass:
    """Tests for Publication dataclass.

    Rule #4: Focused test class
    """

    def test_publication_creation(self):
        """Test creating Publication instance."""
        pub = make_publication()

        assert pub.doi == "10.1038/nature12373"
        assert pub.title == "Test Publication Title"
        assert len(pub.authors) == 2
        assert pub.is_referenced_by_count == 500

    def test_publication_year_property(self):
        """Test year property extraction."""
        pub = make_publication()
        assert pub.year == 2023

    def test_publication_year_none_when_no_date(self):
        """Test year is None when no published date."""
        pub = Publication(
            doi="10.1234/test",
            title="Test",
            authors=[],
            container_title=None,
            publisher=None,
            published_date=None,
            publication_type="journal-article",
            url="https://doi.org/10.1234/test",
        )
        assert pub.year is None

    def test_author_full_name(self):
        """Test Author full_name property."""
        author = Author(given="John", family="Doe")
        assert author.full_name == "John Doe"

    def test_author_full_name_no_given(self):
        """Test Author full_name with only family name."""
        author = Author(given=None, family="Consortium")
        assert author.full_name == "Consortium"

    def test_publication_bibtex_export(self):
        """Test BibTeX export from Publication."""
        pub = make_publication()
        bibtex = pub.to_bibtex()

        assert "@article{" in bibtex
        assert "doe2023" in bibtex.lower()
        assert "title = {Test Publication Title}" in bibtex
        assert "doi = {10.1038/nature12373}" in bibtex
        assert "journal = {Nature}" in bibtex

    def test_publication_bibtex_includes_volume(self):
        """Test BibTeX includes volume and issue."""
        pub = make_publication()
        bibtex = pub.to_bibtex()

        assert "volume = {598}" in bibtex
        assert "number = {7880}" in bibtex
        assert "pages = {123-128}" in bibtex


class TestDOILookup:
    """Tests for DOI lookup functionality.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_lookup_doi_success(self, mock_urlopen):
        """Test successful DOI lookup."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(make_mock_work_response()).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        pub = client.lookup_doi("10.1038/nature12373")

        assert pub is not None
        assert pub.doi == "10.1038/nature12373"
        assert pub.title == "A Test Publication Title"
        assert pub.container_title == "Nature"

    @patch("urllib.request.urlopen")
    def test_lookup_doi_not_found(self, mock_urlopen):
        """Test DOI not found returns None."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )

        client = CrossRefClient()
        pub = client.lookup_doi("10.1234/nonexistent")

        assert pub is None

    def test_lookup_doi_invalid_format(self):
        """Test invalid DOI format returns None."""
        client = CrossRefClient()
        pub = client.lookup_doi("not-a-doi")

        assert pub is None

    @patch("urllib.request.urlopen")
    def test_lookup_doi_with_url_prefix(self, mock_urlopen):
        """Test DOI lookup with https://doi.org/ prefix."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(make_mock_work_response()).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        pub = client.lookup_doi("https://doi.org/10.1038/nature12373")

        assert pub is not None
        assert pub.doi == "10.1038/nature12373"


class TestSearch:
    """Tests for CrossRef search functionality.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_returns_publications(self, mock_urlopen):
        """Test successful search returns publications."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        pubs = client.search("neural networks", limit=20)

        assert len(pubs) == 2
        assert pubs[0].title == "Publication One"
        assert pubs[1].title == "Publication Two"

    @patch("urllib.request.urlopen")
    def test_search_empty_query_returns_empty(self, mock_urlopen):
        """Test empty query returns empty list."""
        client = CrossRefClient()
        pubs = client.search("", limit=20)

        assert pubs == []
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_search_with_type_filter(self, mock_urlopen):
        """Test search with publication type filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        client.search("test", publication_type=PublicationType.JOURNAL_ARTICLE)

        call_url = mock_urlopen.call_args[0][0].full_url
        # URL-encoded filter parameter
        assert "filter=" in call_url
        assert "journal-article" in call_url

    @patch("urllib.request.urlopen")
    def test_search_with_year_filter(self, mock_urlopen):
        """Test search with year range filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        client.search("test", from_year=2020, to_year=2023)

        call_url = mock_urlopen.call_args[0][0].full_url
        # URL-encoded filter contains year parameters
        assert "filter=" in call_url
        assert "2020" in call_url
        assert "2023" in call_url

    @patch("urllib.request.urlopen")
    def test_search_handles_network_error(self, mock_urlopen):
        """Test search handles network errors gracefully."""
        mock_urlopen.side_effect = Exception("Network error")

        client = CrossRefClient()
        pubs = client.search("test query")

        assert pubs == []


class TestReferences:
    """Tests for reference retrieval.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_get_references_success(self, mock_urlopen):
        """Test successful reference retrieval."""
        work_response = make_mock_work_response()
        work_response["message"]["reference"] = [
            {"DOI": "10.1234/ref1"},
            {"DOI": "10.1234/ref2"},
        ]

        ref_response = make_mock_work_response()
        ref_response["message"]["DOI"] = "10.1234/ref1"
        ref_response["message"]["title"] = ["Referenced Paper"]

        mock_response = MagicMock()
        mock_response.read.side_effect = [
            json.dumps(work_response).encode(),
            json.dumps(ref_response).encode(),
        ]
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        refs = client.get_references("10.1038/nature12373", limit=10)

        # At least one reference should be resolved
        assert len(refs) >= 1

    def test_get_references_invalid_doi(self):
        """Test invalid DOI returns empty list."""
        client = CrossRefClient()
        refs = client.get_references("invalid", limit=10)

        assert refs == []


class TestDOICleaning:
    """Tests for DOI cleaning and validation.

    Rule #4: Focused test class
    """

    def test_clean_doi_simple(self):
        """Test cleaning simple DOI."""
        client = CrossRefClient()
        assert client._clean_doi("10.1038/nature12373") == "10.1038/nature12373"

    def test_clean_doi_with_https_prefix(self):
        """Test cleaning DOI with https://doi.org/ prefix."""
        client = CrossRefClient()
        assert (
            client._clean_doi("https://doi.org/10.1038/nature12373")
            == "10.1038/nature12373"
        )

    def test_clean_doi_with_http_prefix(self):
        """Test cleaning DOI with http://doi.org/ prefix."""
        client = CrossRefClient()
        assert (
            client._clean_doi("http://doi.org/10.1038/nature12373")
            == "10.1038/nature12373"
        )

    def test_clean_doi_with_doi_prefix(self):
        """Test cleaning DOI with doi: prefix."""
        client = CrossRefClient()
        assert client._clean_doi("doi:10.1038/nature12373") == "10.1038/nature12373"

    def test_clean_doi_invalid_returns_none(self):
        """Test invalid DOI returns None."""
        client = CrossRefClient()
        assert client._clean_doi("not-a-doi") is None
        assert client._clean_doi("") is None
        assert client._clean_doi("http://example.com/paper") is None


class TestBibTeX:
    """Tests for BibTeX export.

    Rule #4: Focused test class
    """

    def test_export_bibtex_multiple_publications(self):
        """Test exporting multiple publications to BibTeX."""
        pubs = [
            make_publication(),
            Publication(
                doi="10.1234/example",
                title="Another Publication",
                authors=[Author(given="Alice", family="Brown")],
                container_title="Science",
                publisher="AAAS",
                published_date=datetime(2022, 3, 1),
                publication_type="journal-article",
                url="https://doi.org/10.1234/example",
            ),
        ]

        bibtex = export_bibtex(pubs)

        assert bibtex.count("@article{") == 2
        assert "doe2023" in bibtex.lower()
        assert "brown2022" in bibtex.lower()

    def test_export_bibtex_empty_list(self):
        """Test exporting empty list returns empty string."""
        bibtex = export_bibtex([])
        assert bibtex == ""

    def test_bibtex_type_mapping(self):
        """Test BibTeX entry type mapping."""
        # Journal article -> @article
        pub = make_publication()
        assert "@article{" in pub.to_bibtex()

        # Proceedings -> @inproceedings
        pub.publication_type = "proceedings-article"
        assert "@inproceedings{" in pub.to_bibtex()

        # Book chapter -> @incollection
        pub.publication_type = "book-chapter"
        assert "@incollection{" in pub.to_bibtex()


class TestMailto:
    """Tests for mailto parameter handling.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_mailto_added_to_url(self, mock_urlopen):
        """Test mailto parameter is added to URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(make_mock_work_response()).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient(mailto="test@example.com")
        client.lookup_doi("10.1038/nature12373")

        call_url = mock_urlopen.call_args[0][0].full_url
        assert (
            "mailto=test%40example.com" in call_url
            or "mailto=test@example.com" in call_url
        )


class TestHTMLCleaning:
    """Tests for HTML content cleaning.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_abstract_html_removed(self, mock_urlopen):
        """Test HTML is removed from abstract."""
        response = make_mock_work_response()
        response["message"]["abstract"] = "<p>Abstract with <b>bold</b> text.</p>"

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        pub = client.lookup_doi("10.1038/nature12373")

        assert pub is not None
        assert "<p>" not in pub.abstract
        assert "<b>" not in pub.abstract
        assert "Abstract with bold text." in pub.abstract


class TestErrorHandling:
    """Tests for error handling.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_handles_malformed_json(self, mock_urlopen):
        """Test handling of malformed JSON response."""
        mock_response = MagicMock()
        mock_response.read.return_value = b"not valid json"
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        pub = client.lookup_doi("10.1038/nature12373")

        assert pub is None

    @patch("urllib.request.urlopen")
    def test_handles_missing_required_fields(self, mock_urlopen):
        """Test handling of response with missing fields."""
        incomplete_response = {
            "status": "ok",
            "message": {
                "DOI": "10.1038/nature12373",
                # Missing title and other required fields
            },
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(incomplete_response).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CrossRefClient()
        pub = client.lookup_doi("10.1038/nature12373")

        # Should return None when required fields are missing
        assert pub is None


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Publication dataclass: 7 tests
    - DOI lookup: 4 tests
    - Search functionality: 5 tests
    - Reference retrieval: 2 tests
    - DOI cleaning: 5 tests
    - BibTeX export: 3 tests
    - Mailto handling: 1 test
    - HTML cleaning: 1 test
    - Error handling: 2 tests

    Total: 30 tests

Design Decisions:
    1. Mock all HTTP requests
    2. Test Publication dataclass separately
    3. Test DOI format validation
    4. Verify mailto parameter for polite pool
    5. Test BibTeX formatting with type mapping
"""
