"""
Tests for NVDDiscovery wrapper using the NIST NVD API.

This module tests the NVDDiscovery class including search,
CVE lookup, and CPE search functionality.

Test Strategy
-------------
- Mock the requests library to avoid actual API calls
- Test CVEEntry dataclass methods
- Test search functionality with severity filtering
- Test CVE structure validation
- Test CPE search
- Test error handling
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestCVEEntryDataclass: CVEEntry model tests
- TestNVDDiscoverySearch: Search functionality
- TestNVDDiscoveryGetCVE: Single CVE lookup
- TestNVDDiscoveryCPESearch: CPE-based search
- TestCVEIdCleaning: ID validation and cleaning
- TestRateLimiting: Rate limit handling
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_nvd_response(num_cves: int = 1):
    """Create a mock NVD API response."""
    vulnerabilities = []
    for i in range(num_cves):
        vulnerabilities.append(
            {
                "cve": {
                    "id": f"CVE-2024-{10000 + i}",
                    "descriptions": [
                        {"lang": "en", "value": f"Test vulnerability description {i}"}
                    ],
                    "published": "2024-01-15T12:00:00.000",
                    "lastModified": "2024-01-20T12:00:00.000",
                    "metrics": {
                        "cvssMetricV31": [
                            {
                                "cvssData": {
                                    "baseSeverity": "HIGH",
                                    "baseScore": 8.5,
                                    "vectorString": "CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
                                }
                            }
                        ]
                    },
                    "configurations": [
                        {
                            "nodes": [
                                {
                                    "cpeMatch": [
                                        {
                                            "criteria": "cpe:2.3:a:apache:tomcat:9.0.0:*:*:*:*:*:*:*"
                                        }
                                    ]
                                }
                            ]
                        }
                    ],
                    "references": [
                        {"url": "https://example.com/advisory"},
                        {"url": "https://example.com/patch"},
                    ],
                    "weaknesses": [
                        {"description": [{"value": "CWE-79"}]},
                        {"description": [{"value": "CWE-89"}]},
                    ],
                }
            }
        )

    return {
        "resultsPerPage": num_cves,
        "startIndex": 0,
        "totalResults": num_cves,
        "vulnerabilities": vulnerabilities,
    }


def make_cve_entry():
    """Create a test CVEEntry object."""
    from ingestforge.discovery.nvd_wrapper import CVEEntry

    return CVEEntry(
        cve_id="CVE-2024-12345",
        description="Test vulnerability affecting example software.",
        severity="HIGH",
        cvss_score=8.5,
        cvss_vector="CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:H/I:H/A:H",
        affected_products=["cpe:2.3:a:example:software:1.0:*:*:*:*:*:*:*"],
        published_date=datetime(2024, 1, 15),
        last_modified=datetime(2024, 1, 20),
        references=["https://example.com/advisory"],
        cwe_ids=["CWE-79"],
    )


# ============================================================================
# Test Classes
# ============================================================================


class TestCVEEntryDataclass:
    """Tests for CVEEntry dataclass.

    Rule #4: Focused test class
    """

    def test_cve_entry_creation(self):
        """Test creating CVEEntry instance."""
        cve = make_cve_entry()

        assert cve.cve_id == "CVE-2024-12345"
        assert cve.severity == "HIGH"
        assert cve.cvss_score == 8.5
        assert len(cve.affected_products) == 1
        assert len(cve.cwe_ids) == 1

    def test_cve_entry_to_dict(self):
        """Test CVEEntry to_dict conversion."""
        cve = make_cve_entry()
        cve_dict = cve.to_dict()

        assert cve_dict["cve_id"] == "CVE-2024-12345"
        assert cve_dict["severity"] == "HIGH"
        assert cve_dict["cvss_score"] == 8.5
        assert "2024-01-15" in cve_dict["published_date"]
        assert cve_dict["cwe_ids"] == ["CWE-79"]

    def test_cve_entry_optional_fields_none(self):
        """Test CVEEntry with None optional fields."""
        from ingestforge.discovery.nvd_wrapper import CVEEntry

        cve = CVEEntry(
            cve_id="CVE-2024-00001",
            description="Minimal CVE entry",
            severity=None,
            cvss_score=None,
            cvss_vector=None,
            affected_products=[],
            published_date=None,
            last_modified=None,
            references=[],
            cwe_ids=[],
        )

        cve_dict = cve.to_dict()
        assert cve_dict["severity"] is None
        assert cve_dict["cvss_score"] is None
        assert cve_dict["published_date"] is None


class TestNVDDiscoverySearch:
    """Tests for NVDDiscovery search functionality.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_returns_cves(self, mock_check):
        """Test successful search returns CVEEntry list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            # Setup mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = make_mock_nvd_response(2)
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search("apache tomcat", max_results=10)

            assert len(cves) == 2
            assert cves[0].cve_id == "CVE-2024-10000"
            assert cves[0].severity == "HIGH"
            assert cves[0].cvss_score == 8.5

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_empty_query_returns_empty(self, mock_check):
        """Test empty query returns empty list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search("")

            assert cves == []

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_whitespace_query_returns_empty(self, mock_check):
        """Test whitespace-only query returns empty list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search("   ")

            assert cves == []

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_with_severity_filter(self, mock_check):
        """Test search with severity filter adds parameter."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = make_mock_nvd_response(1)
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            discovery.search("test", severity="CRITICAL")

            # Verify severity was included in request
            call_args = mock_requests.get.call_args
            assert "cvssV3Severity=CRITICAL" in call_args[0][0]

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_respects_max_results_limit(self, mock_check):
        """Test search max_results is clamped to valid range."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = make_mock_nvd_response(0)
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()

            # Test upper limit clamping
            discovery.search("test", max_results=5000)
            call_args = mock_requests.get.call_args
            assert "resultsPerPage=2000" in call_args[0][0]

            # Test lower limit clamping
            discovery.search("test", max_results=0)
            call_args = mock_requests.get.call_args
            assert "resultsPerPage=1" in call_args[0][0]

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_handles_api_error(self, mock_check):
        """Test search handles API errors gracefully."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search("test")

            assert cves == []

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_handles_rate_limit(self, mock_check):
        """Test search handles rate limit (403) gracefully."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 403
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search("test")

            assert cves == []


class TestNVDDiscoveryGetCVE:
    """Tests for NVDDiscovery get_cve functionality.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_get_cve_by_id(self, mock_check):
        """Test fetching CVE by ID."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = make_mock_nvd_response(1)
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cve = discovery.get_cve("CVE-2024-10000")

            assert cve is not None
            assert cve.cve_id == "CVE-2024-10000"

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_get_cve_empty_id_returns_none(self, mock_check):
        """Test get_cve with empty ID returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cve = discovery.get_cve("")

            assert cve is None

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_get_cve_invalid_id_returns_none(self, mock_check):
        """Test get_cve with invalid ID format returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cve = discovery.get_cve("invalid-format")

            assert cve is None

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_get_cve_not_found_returns_none(self, mock_check):
        """Test get_cve returns None when CVE not found."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"vulnerabilities": []}
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cve = discovery.get_cve("CVE-2024-99999")

            assert cve is None


class TestNVDDiscoveryCPESearch:
    """Tests for NVDDiscovery CPE-based search.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_by_cpe(self, mock_check):
        """Test CPE-based search returns CVEs."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = make_mock_nvd_response(2)
            mock_requests.get.return_value = mock_response

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search_by_cpe("cpe:2.3:a:apache:tomcat:*:*:*:*:*:*:*:*")

            assert len(cves) == 2

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_by_cpe_empty_returns_empty(self, mock_check):
        """Test empty CPE returns empty list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search_by_cpe("")

            assert cves == []

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_search_by_cpe_invalid_format_returns_empty(self, mock_check):
        """Test invalid CPE format returns empty list."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            cves = discovery.search_by_cpe("invalid-cpe-format")

            assert cves == []


class TestCVEIdCleaning:
    """Tests for CVE ID cleaning and validation.

    Rule #4: Focused test class
    """

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_clean_standard_cve_id(self, mock_check):
        """Test cleaning standard CVE ID."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            assert discovery._clean_cve_id("CVE-2024-12345") == "CVE-2024-12345"

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_clean_lowercase_cve_id(self, mock_check):
        """Test cleaning lowercase CVE ID."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            assert discovery._clean_cve_id("cve-2024-12345") == "CVE-2024-12345"

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_clean_cve_id_without_prefix(self, mock_check):
        """Test cleaning CVE ID without CVE- prefix."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            assert discovery._clean_cve_id("2024-12345") == "CVE-2024-12345"

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_clean_invalid_cve_id_returns_none(self, mock_check):
        """Test invalid CVE ID returns None."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys

            mock_requests = sys.modules["requests"]
            mock_requests.get.return_value = MagicMock()

            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            discovery = NVDDiscovery()
            assert discovery._clean_cve_id("invalid") is None
            assert discovery._clean_cve_id("CVE-123") is None
            assert discovery._clean_cve_id("http://example.com") is None


class TestNVDDiscoveryInitialization:
    """Tests for NVDDiscovery initialization.

    Rule #4: Focused test class
    """

    def test_init_raises_without_requests_library(self):
        """Test initialization raises ImportError without requests library."""
        with patch(
            "ingestforge.discovery.nvd_wrapper._check_requests_library",
            return_value=False,
        ):
            from ingestforge.discovery.nvd_wrapper import NVDDiscovery

            with pytest.raises(ImportError) as exc_info:
                NVDDiscovery()

            assert "requests library not installed" in str(exc_info.value)

    @patch("ingestforge.discovery.nvd_wrapper._check_requests_library")
    def test_init_with_api_key(self, mock_check):
        """Test initialization with API key sets faster rate limit."""
        mock_check.return_value = True

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            from ingestforge.discovery.nvd_wrapper import (
                NVDDiscovery,
                API_KEY_RATE_LIMIT_SECONDS,
            )

            discovery = NVDDiscovery(api_key="test-api-key")
            assert discovery._api_key == "test-api-key"
            assert discovery._rate_limit == API_KEY_RATE_LIMIT_SECONDS


class TestCVEConversion:
    """Tests for NVD response conversion.

    Rule #4: Focused test class
    """

    def test_convert_extracts_all_fields(self):
        """Test conversion extracts all metadata fields."""
        from ingestforge.discovery.nvd_wrapper import _convert_nvd_to_cve_entry

        mock_response = make_mock_nvd_response(1)
        item = mock_response["vulnerabilities"][0]
        cve = _convert_nvd_to_cve_entry(item)

        assert cve.cve_id == "CVE-2024-10000"
        assert "Test vulnerability description" in cve.description
        assert cve.severity == "HIGH"
        assert cve.cvss_score == 8.5
        assert "CVSS:3.1" in cve.cvss_vector
        assert len(cve.affected_products) > 0
        assert len(cve.references) == 2
        assert "CWE-79" in cve.cwe_ids
        assert "CWE-89" in cve.cwe_ids

    def test_convert_handles_missing_cvss(self):
        """Test conversion handles missing CVSS data."""
        from ingestforge.discovery.nvd_wrapper import _convert_nvd_to_cve_entry

        item = {
            "cve": {
                "id": "CVE-2024-99999",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "published": "2024-01-15T12:00:00.000",
                "lastModified": "2024-01-20T12:00:00.000",
                "metrics": {},
                "configurations": [],
                "references": [],
                "weaknesses": [],
            }
        }

        cve = _convert_nvd_to_cve_entry(item)
        assert cve.severity is None
        assert cve.cvss_score is None
        assert cve.cvss_vector is None

    def test_convert_uses_cvss_v2_fallback(self):
        """Test conversion falls back to CVSS v2 if v3 unavailable."""
        from ingestforge.discovery.nvd_wrapper import _convert_nvd_to_cve_entry

        item = {
            "cve": {
                "id": "CVE-2024-88888",
                "descriptions": [{"lang": "en", "value": "Test"}],
                "published": "2024-01-15T12:00:00.000",
                "lastModified": "2024-01-20T12:00:00.000",
                "metrics": {
                    "cvssMetricV2": [
                        {
                            "cvssData": {
                                "baseScore": 7.5,
                                "vectorString": "AV:N/AC:L/Au:N/C:P/I:P/A:P",
                            }
                        }
                    ]
                },
                "configurations": [],
                "references": [],
                "weaknesses": [],
            }
        }

        cve = _convert_nvd_to_cve_entry(item)
        assert cve.cvss_score == 7.5
        assert cve.severity == "HIGH"  # Mapped from score


class TestHelperFunctions:
    """Tests for helper functions.

    Rule #4: Focused test class
    """

    def test_extract_description_english_preferred(self):
        """Test English description is preferred."""
        from ingestforge.discovery.nvd_wrapper import _extract_description

        cve_data = {
            "descriptions": [
                {"lang": "es", "value": "Spanish description"},
                {"lang": "en", "value": "English description"},
            ]
        }

        desc = _extract_description(cve_data)
        assert desc == "English description"

    def test_extract_description_fallback(self):
        """Test fallback to first description if no English."""
        from ingestforge.discovery.nvd_wrapper import _extract_description

        cve_data = {
            "descriptions": [
                {"lang": "es", "value": "Spanish description"},
            ]
        }

        desc = _extract_description(cve_data)
        assert desc == "Spanish description"

    def test_extract_description_empty(self):
        """Test empty descriptions returns default message."""
        from ingestforge.discovery.nvd_wrapper import _extract_description

        cve_data = {"descriptions": []}
        desc = _extract_description(cve_data)
        assert desc == "No description available"

    def test_map_v2_score_to_severity(self):
        """Test CVSS v2 score mapping to severity."""
        from ingestforge.discovery.nvd_wrapper import _map_v2_score_to_severity

        assert _map_v2_score_to_severity(10.0) == "CRITICAL"
        assert _map_v2_score_to_severity(9.0) == "CRITICAL"
        assert _map_v2_score_to_severity(8.5) == "HIGH"
        assert _map_v2_score_to_severity(7.0) == "HIGH"
        assert _map_v2_score_to_severity(5.0) == "MEDIUM"
        assert _map_v2_score_to_severity(4.0) == "MEDIUM"
        assert _map_v2_score_to_severity(3.0) == "LOW"
        assert _map_v2_score_to_severity(0.0) == "LOW"
        assert _map_v2_score_to_severity(None) is None


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - CVEEntry dataclass: 3 tests
    - NVDDiscovery search: 7 tests
    - NVDDiscovery get_cve: 4 tests
    - CPE search: 3 tests
    - CVE ID cleaning: 4 tests
    - Initialization: 2 tests
    - CVE conversion: 3 tests
    - Helper functions: 4 tests

    Total: 30 tests

Design Decisions:
    1. Mock requests library to avoid actual API calls
    2. Test CVEEntry dataclass separately
    3. Test error handling for all failure modes
    4. Verify CVE ID cleaning handles various formats
    5. Test CVSS v2 to v3 severity mapping
    6. Test rate limiting behavior
"""
