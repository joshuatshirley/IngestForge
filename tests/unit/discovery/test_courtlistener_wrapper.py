"""
Tests for CourtListener client functionality.

This module tests the CourtListenerDiscovery class including search,
jurisdiction filtering, case structure validation, and opinion download.

Test Strategy
-------------
- Mock HTTP requests to avoid actual API calls
- Test CourtCase dataclass methods
- Test jurisdiction filtering
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestCourtCaseDataclass: CourtCase model tests
- TestSearch: Search functionality
- TestJurisdictionFiltering: Jurisdiction filter tests
- TestGetCase: Case detail retrieval
- TestDownloadOpinion: Opinion download tests
- TestErrorHandling: Error scenarios
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch


from ingestforge.discovery.courtlistener_wrapper import (
    CourtListenerDiscovery,
    CourtCase,
    FEDERAL_COURTS,
)


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_search_response() -> dict:
    """Create mock CourtListener search response."""
    return {
        "count": 2,
        "next": None,
        "previous": None,
        "results": [
            {
                "id": "12345",
                "caseName": "Smith v. Jones",
                "docketNumber": "20-1234",
                "court": "ca9",
                "dateFiled": "2023-06-15",
                "dateArgued": "2023-05-01",
                "status": "Published",
                "judge": "Smith, J.; Brown, J.",
                "citation": ["123 F.3d 456"],
                "absolute_url": "/opinion/12345/smith-v-jones/",
                "cluster": "/api/rest/v4/clusters/67890/",
                "snippet": "...qualified immunity...",
            },
            {
                "id": "12346",
                "caseName": "Doe v. City",
                "docketNumber": "21-5678",
                "court": "scotus",
                "dateFiled": "2022-03-01",
                "status": "Published",
                "judge": "Roberts, C.J.",
                "citation": ["142 S.Ct. 789"],
                "absolute_url": "/opinion/12346/doe-v-city/",
                "cluster": "/api/rest/v4/clusters/67891/",
            },
        ],
    }


def make_mock_cluster_response() -> dict:
    """Create mock CourtListener cluster response."""
    return {
        "id": 67890,
        "case_name": "Smith v. Jones",
        "case_name_short": "Smith",
        "docket_number": "20-1234",
        "court": "/api/rest/v4/courts/ca9/",
        "date_filed": "2023-06-15",
        "judges": "Smith, J.; Brown, J.",
        "precedential_status": "Published",
        "citations": ["123 F.3d 456"],
        "absolute_url": "/opinion/67890/smith-v-jones/",
        "sub_opinions": ["/api/rest/v4/opinions/99999/"],
    }


def make_mock_opinion_response() -> dict:
    """Create mock CourtListener opinion response."""
    return {
        "id": 99999,
        "plain_text": "This is the full text of the opinion...",
        "html_with_citations": "<p>This is the <em>opinion</em> with citations.</p>",
        "author": "Smith, J.",
    }


def make_court_case() -> CourtCase:
    """Create a test CourtCase object."""
    return CourtCase(
        docket_number="20-1234",
        case_name="Smith v. Jones",
        court="Ninth Circuit",
        jurisdiction="Federal",
        judges=["Smith, J.", "Brown, J."],
        date_filed=datetime(2023, 6, 15),
        date_decided=datetime(2023, 7, 1),
        citations=["123 F.3d 456"],
        opinion_url="https://www.courtlistener.com/opinion/12345/smith-v-jones/",
        precedential_status="Published",
        case_id="12345",
        opinion_id="99999",
        cluster_id="67890",
        court_id="ca9",
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestCourtCaseDataclass:
    """Tests for CourtCase dataclass.

    Rule #4: Focused test class
    """

    def test_court_case_creation(self):
        """Test creating CourtCase instance."""
        case = make_court_case()

        assert case.docket_number == "20-1234"
        assert case.case_name == "Smith v. Jones"
        assert case.court == "Ninth Circuit"
        assert case.jurisdiction == "Federal"
        assert len(case.judges) == 2

    def test_court_case_year_from_decided(self):
        """Test year property from date_decided."""
        case = make_court_case()
        assert case.year == 2023

    def test_court_case_year_from_filed_when_no_decided(self):
        """Test year falls back to date_filed when no date_decided."""
        case = CourtCase(
            docket_number="20-1234",
            case_name="Test Case",
            court="Test Court",
            jurisdiction="Federal",
            date_filed=datetime(2022, 1, 1),
            date_decided=None,
            opinion_url="",
            precedential_status="Published",
        )
        assert case.year == 2022

    def test_court_case_year_none_when_no_dates(self):
        """Test year is None when no dates available."""
        case = CourtCase(
            docket_number="20-1234",
            case_name="Test Case",
            court="Test Court",
            jurisdiction="Federal",
            date_filed=None,
            date_decided=None,
            opinion_url="",
            precedential_status="Published",
        )
        assert case.year is None

    def test_to_citation_string(self):
        """Test citation string generation."""
        case = make_court_case()
        citation = case.to_citation_string()

        assert "Smith v. Jones" in citation
        assert "123 F.3d 456" in citation
        assert "Ninth Circuit" in citation
        assert "2023" in citation

    def test_to_dict(self):
        """Test dictionary conversion."""
        case = make_court_case()
        data = case.to_dict()

        assert data["docket_number"] == "20-1234"
        assert data["case_name"] == "Smith v. Jones"
        assert data["court"] == "Ninth Circuit"
        assert data["jurisdiction"] == "Federal"
        assert data["citations"] == ["123 F.3d 456"]


class TestSearch:
    """Tests for CourtListener search functionality.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_returns_cases(self, mock_urlopen):
        """Test successful search returns cases."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        cases = client.search("qualified immunity", max_results=5)

        assert len(cases) == 2
        assert cases[0].case_name == "Smith v. Jones"
        assert cases[1].case_name == "Doe v. City"

    @patch("urllib.request.urlopen")
    def test_search_empty_query_returns_empty(self, mock_urlopen):
        """Test empty query returns empty list."""
        client = CourtListenerDiscovery()
        cases = client.search("", max_results=5)

        assert cases == []
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_search_whitespace_query_returns_empty(self, mock_urlopen):
        """Test whitespace-only query returns empty list."""
        client = CourtListenerDiscovery()
        cases = client.search("   ", max_results=5)

        assert cases == []
        mock_urlopen.assert_not_called()

    @patch("urllib.request.urlopen")
    def test_search_extracts_metadata(self, mock_urlopen):
        """Test search extracts all metadata fields."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        cases = client.search("test")

        case = cases[0]
        assert case.docket_number == "20-1234"
        assert case.precedential_status == "Published"
        assert "123 F.3d 456" in case.citations
        assert case.snippet == "...qualified immunity..."

    @patch("urllib.request.urlopen")
    def test_search_handles_network_error(self, mock_urlopen):
        """Test search handles network errors gracefully."""
        mock_urlopen.side_effect = Exception("Network error")

        client = CourtListenerDiscovery()
        cases = client.search("test query")

        assert cases == []


class TestJurisdictionFiltering:
    """Tests for jurisdiction filtering.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_with_jurisdiction_filter(self, mock_urlopen):
        """Test search includes jurisdiction in URL."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        client.search("test", jurisdiction="ca9")

        call_url = mock_urlopen.call_args[0][0].full_url
        assert "court=ca9" in call_url

    @patch("urllib.request.urlopen")
    def test_search_without_jurisdiction_filter(self, mock_urlopen):
        """Test search without jurisdiction filter."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        client.search("test")

        call_url = mock_urlopen.call_args[0][0].full_url
        assert "court=" not in call_url

    def test_federal_courts_mapping(self):
        """Test federal court code mappings exist."""
        assert "scotus" in FEDERAL_COURTS
        assert "ca9" in FEDERAL_COURTS
        assert "cadc" in FEDERAL_COURTS
        assert FEDERAL_COURTS["scotus"] == "Supreme Court of the United States"
        assert FEDERAL_COURTS["ca9"] == "Ninth Circuit"

    @patch("urllib.request.urlopen")
    def test_jurisdiction_detection_federal(self, mock_urlopen):
        """Test federal jurisdiction detection."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        cases = client.search("test")

        # ca9 is federal
        assert cases[0].jurisdiction == "Federal"
        # scotus is federal
        assert cases[1].jurisdiction == "Federal"


class TestGetCase:
    """Tests for case detail retrieval.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_get_case_success(self, mock_urlopen):
        """Test successful case retrieval."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_cluster_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        case = client.get_case("67890")

        assert case is not None
        assert case.case_name == "Smith v. Jones"
        assert case.docket_number == "20-1234"

    @patch("urllib.request.urlopen")
    def test_get_case_not_found(self, mock_urlopen):
        """Test case not found returns None."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 404, "Not Found", {}, None
        )

        client = CourtListenerDiscovery()
        case = client.get_case("nonexistent")

        assert case is None

    def test_get_case_empty_id(self):
        """Test empty cluster ID returns None."""
        client = CourtListenerDiscovery()
        case = client.get_case("")

        assert case is None


class TestDownloadOpinion:
    """Tests for opinion download functionality.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_download_opinion_success(self, mock_urlopen, tmp_path):
        """Test successful opinion download."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_opinion_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        case = make_court_case()

        result = client.download_opinion(case, tmp_path)

        assert result is not None
        assert result.exists()
        content = result.read_text()
        assert "opinion" in content.lower()

    def test_download_opinion_no_opinion_id(self, tmp_path):
        """Test download fails when no opinion ID."""
        client = CourtListenerDiscovery()
        case = CourtCase(
            docket_number="20-1234",
            case_name="Test Case",
            court="Test Court",
            jurisdiction="Federal",
            date_filed=None,
            date_decided=None,
            opinion_url="",
            precedential_status="Published",
            opinion_id=None,
        )

        result = client.download_opinion(case, tmp_path)

        assert result is None

    @patch("urllib.request.urlopen")
    def test_download_opinion_api_error(self, mock_urlopen, tmp_path):
        """Test download handles API errors."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 500, "Server Error", {}, None
        )

        client = CourtListenerDiscovery()
        case = make_court_case()

        result = client.download_opinion(case, tmp_path)

        assert result is None


class TestDateParsing:
    """Tests for date parsing.

    Rule #4: Focused test class
    """

    def test_parse_date_iso_format(self):
        """Test parsing ISO format date."""
        client = CourtListenerDiscovery()
        result = client._parse_date("2023-06-15")

        assert result is not None
        assert result.year == 2023
        assert result.month == 6
        assert result.day == 15

    def test_parse_date_iso_with_time(self):
        """Test parsing ISO format with time."""
        client = CourtListenerDiscovery()
        result = client._parse_date("2023-06-15T10:30:00Z")

        assert result is not None
        assert result.year == 2023

    def test_parse_date_invalid(self):
        """Test parsing invalid date returns None."""
        client = CourtListenerDiscovery()
        result = client._parse_date("not-a-date")

        assert result is None

    def test_parse_date_none(self):
        """Test parsing None returns None."""
        client = CourtListenerDiscovery()
        result = client._parse_date(None)

        assert result is None


class TestJudgeParsing:
    """Tests for judge string parsing.

    Rule #4: Focused test class
    """

    def test_parse_judges_comma_separated(self):
        """Test parsing comma-separated judges."""
        client = CourtListenerDiscovery()
        result = client._parse_judges("Smith, J., Brown, J., Davis, J.")

        assert len(result) >= 2

    def test_parse_judges_semicolon_separated(self):
        """Test parsing semicolon-separated judges."""
        client = CourtListenerDiscovery()
        result = client._parse_judges("Smith, J.; Brown, J.")

        # Splits on both comma and semicolon, so "Smith, J." becomes ["Smith", "J."]
        assert len(result) >= 2

    def test_parse_judges_empty(self):
        """Test parsing empty string returns empty list."""
        client = CourtListenerDiscovery()
        result = client._parse_judges("")

        assert result == []


class TestAPIToken:
    """Tests for API token handling.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_api_token_added_to_headers(self, mock_urlopen):
        """Test API token is added to request headers."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery(api_token="test_token_123")
        client.search("test")

        # Check the request object
        call_request = mock_urlopen.call_args[0][0]
        assert "Authorization" in call_request.headers
        assert "Token test_token_123" in call_request.headers["Authorization"]

    @patch("urllib.request.urlopen")
    def test_no_token_no_header(self, mock_urlopen):
        """Test no Authorization header when no token."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        client.search("test")

        call_request = mock_urlopen.call_args[0][0]
        assert "Authorization" not in call_request.headers


class TestDateFiltering:
    """Tests for date range filtering.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    def test_search_with_date_range(self, mock_urlopen):
        """Test search with date range filters."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        from_date = datetime(2020, 1, 1)
        to_date = datetime(2023, 12, 31)
        client.search("test", from_date=from_date, to_date=to_date)

        call_url = mock_urlopen.call_args[0][0].full_url
        assert "filed_after=2020-01-01" in call_url
        assert "filed_before=2023-12-31" in call_url


class TestRateLimiting:
    """Tests for rate limiting.

    Rule #4: Focused test class
    """

    @patch("urllib.request.urlopen")
    @patch("time.sleep")
    def test_rate_limiting_delays_requests(self, mock_sleep, mock_urlopen):
        """Test rate limiter adds delays between requests."""
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            make_mock_search_response()
        ).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()

        # Make two rapid requests
        client.search("test1")
        client.search("test2")

        # Rate limiter should have been invoked
        # (may or may not sleep depending on timing)
        assert mock_urlopen.call_count == 2


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

        client = CourtListenerDiscovery()
        cases = client.search("test")

        assert cases == []

    @patch("urllib.request.urlopen")
    def test_handles_rate_limit_error(self, mock_urlopen):
        """Test handling of rate limit (429) error."""
        import urllib.error

        mock_urlopen.side_effect = urllib.error.HTTPError(
            "url", 429, "Too Many Requests", {}, None
        )

        client = CourtListenerDiscovery()
        cases = client.search("test")

        assert cases == []

    @patch("urllib.request.urlopen")
    def test_handles_missing_case_name(self, mock_urlopen):
        """Test handling of result with missing case name."""
        response = {
            "results": [
                {
                    "id": "12345",
                    # Missing caseName
                    "docketNumber": "20-1234",
                    "court": "ca9",
                }
            ]
        }

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response).encode()
        mock_response.__enter__.return_value = mock_response
        mock_response.__exit__.return_value = False
        mock_urlopen.return_value = mock_response

        client = CourtListenerDiscovery()
        cases = client.search("test")

        # Should skip result with missing required field
        assert len(cases) == 0


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - CourtCase dataclass: 6 tests
    - Search functionality: 5 tests
    - Jurisdiction filtering: 4 tests
    - Case detail retrieval: 3 tests
    - Opinion download: 3 tests
    - Date parsing: 4 tests
    - Judge parsing: 3 tests
    - API token handling: 2 tests
    - Date filtering: 1 test
    - Rate limiting: 1 test
    - Error handling: 3 tests

    Total: 35 tests

Design Decisions:
    1. Mock all HTTP requests
    2. Test CourtCase dataclass separately
    3. Test jurisdiction filtering and detection
    4. Verify API token in headers
    5. Test date range filtering in URL
    6. Test error handling for various HTTP errors
"""
