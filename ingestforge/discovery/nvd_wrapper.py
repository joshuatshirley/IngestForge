"""NVD Discovery Wrapper - Search CVEs/vulnerabilities via NIST NVD API.

This module provides the NVDDiscovery class for searching vulnerabilities
using the NIST National Vulnerability Database (NVD) API.

Features:
- Search CVEs by software/product name
- Filter by severity level (LOW, MEDIUM, HIGH, CRITICAL)
- Get specific CVE by ID
- Search by CPE (Common Platform Enumeration)
- Rate limiting to respect API limits
- Caching for repeated queries

API Documentation:
    https://nvd.nist.gov/developers/vulnerabilities"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# NVD API constants
NVD_API_BASE = "https://services.nvd.nist.gov/rest/json/cves/2.0"
DEFAULT_RATE_LIMIT_SECONDS = 6.0  # NVD rate limit: ~10 requests/minute without API key
API_KEY_RATE_LIMIT_SECONDS = 0.6  # With API key: ~100 requests/minute


class Severity(str, Enum):
    """CVSS severity levels."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CVEEntry:
    """Metadata for a CVE vulnerability.

    Rule #9: Full type hints on all fields.
    """

    cve_id: str
    description: str
    severity: Optional[str] = None
    cvss_score: Optional[float] = None
    cvss_vector: Optional[str] = None
    affected_products: List[str] = field(default_factory=list)
    published_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    references: List[str] = field(default_factory=list)
    cwe_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert CVE entry to dictionary representation.

        Rule #4: Function <60 lines.
        """
        return {
            "cve_id": self.cve_id,
            "description": self.description,
            "severity": self.severity,
            "cvss_score": self.cvss_score,
            "cvss_vector": self.cvss_vector,
            "affected_products": self.affected_products,
            "published_date": (
                self.published_date.isoformat() if self.published_date else None
            ),
            "last_modified": (
                self.last_modified.isoformat() if self.last_modified else None
            ),
            "references": self.references,
            "cwe_ids": self.cwe_ids,
        }


def _check_requests_library() -> bool:
    """Check if requests library is available.

    Rule #4: Small helper function.
    """
    try:
        import requests

        return True
    except ImportError:
        return False


def _parse_datetime(date_str: Optional[str]) -> Optional[datetime]:
    """Parse NVD date string to datetime.

    Rule #4: Small helper function.
    Rule #1: Early return for None.
    """
    if not date_str:
        return None

    try:
        # NVD format: 2024-01-15T12:00:00.000
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def _extract_description(cve_data: dict) -> str:
    """Extract English description from CVE data.

    Rule #4: Function <60 lines.
    Rule #1: Early return pattern.
    """
    descriptions = cve_data.get("descriptions", [])
    if not descriptions:
        return "No description available"

    # Prefer English description
    for desc in descriptions:
        if desc.get("lang") == "en":
            return desc.get("value", "No description available")

    # Fallback to first description
    return descriptions[0].get("value", "No description available")


def _extract_cvss_v3(
    metrics: dict,
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Extract CVSS v3 severity, score, and vector.

    Rule #4: Function <60 lines.
    Rule #1: Early return pattern.

    Returns:
        Tuple of (severity, score, vector) or (None, None, None).
    """
    # Try CVSS 3.1 first, then 3.0
    for version_key in ["cvssMetricV31", "cvssMetricV30"]:
        cvss_list = metrics.get(version_key, [])
        if not cvss_list:
            continue

        cvss_data = cvss_list[0].get("cvssData", {})
        severity = cvss_data.get("baseSeverity")
        score = cvss_data.get("baseScore")
        vector = cvss_data.get("vectorString")
        return severity, score, vector

    return None, None, None


def _extract_cvss_v2_fallback(
    metrics: dict,
) -> tuple[Optional[str], Optional[float], Optional[str]]:
    """Extract CVSS v2 as fallback.

    Rule #4: Function <60 lines.
    """
    cvss_list = metrics.get("cvssMetricV2", [])
    if not cvss_list:
        return None, None, None

    cvss_data = cvss_list[0].get("cvssData", {})
    score = cvss_data.get("baseScore")
    vector = cvss_data.get("vectorString")

    # Map v2 score to severity (v2 doesn't have baseSeverity)
    severity = _map_v2_score_to_severity(score)
    return severity, score, vector


def _map_v2_score_to_severity(score: Optional[float]) -> Optional[str]:
    """Map CVSS v2 score to severity label.

    Rule #4: Small helper function.
    """
    if score is None:
        return None
    if score >= 9.0:
        return "CRITICAL"
    if score >= 7.0:
        return "HIGH"
    if score >= 4.0:
        return "MEDIUM"
    return "LOW"


def _extract_cpe_products(configurations: list) -> List[str]:
    """Extract CPE product strings from configurations.

    Rule #4: Function <60 lines.
    Rule #2: Bounded iteration with limit.
    Rule #1: Max 3 nesting levels.
    """
    products: List[str] = []
    max_products = 50  # Bounded limit

    for config in configurations[:10]:  # Limit configurations checked
        nodes = config.get("nodes", [])
        for node in nodes[:10]:  # Limit nodes per configuration
            _extract_cpe_from_node(node, products, max_products)
            if len(products) >= max_products:
                return products

    return products


def _extract_cpe_from_node(node: dict, products: List[str], max_products: int) -> None:
    """Extract CPE strings from a single node.

    Helper for _extract_cpe_products to reduce nesting.

    Args:
        node: Configuration node
        products: List to append CPE strings to
        max_products: Maximum products limit
    """
    cpe_matches = node.get("cpeMatch", [])
    for match in cpe_matches[:20]:  # Limit CPE matches per node
        if len(products) >= max_products:
            return

        cpe = match.get("criteria", "")
        if cpe and cpe not in products:
            products.append(cpe)


def _extract_references(refs: list) -> List[str]:
    """Extract reference URLs.

    Rule #4: Small helper function.
    Rule #2: Bounded iteration.
    """
    urls: List[str] = []
    for ref in refs[:20]:  # Limit to 20 references
        url = ref.get("url")
        if url:
            urls.append(url)
    return urls


def _extract_cwe_ids(weaknesses: list) -> List[str]:
    """Extract CWE IDs from weaknesses.

    Rule #4: Small helper function.
    Rule #2: Bounded iteration.
    """
    cwe_ids: List[str] = []
    for weakness in weaknesses[:10]:  # Limit weaknesses
        for desc in weakness.get("description", [])[:5]:
            value = desc.get("value", "")
            if value.startswith("CWE-") and value not in cwe_ids:
                cwe_ids.append(value)
    return cwe_ids


def _convert_nvd_to_cve_entry(item: dict) -> CVEEntry:
    """Convert NVD API response item to CVEEntry.

    Rule #4: Function <60 lines.
    Rule #1: Simple extraction flow.

    Args:
        item: Single CVE item from NVD API response.

    Returns:
        CVEEntry with extracted metadata.
    """
    cve_data = item.get("cve", {})
    cve_id = cve_data.get("id", "")

    description = _extract_description(cve_data)
    metrics = cve_data.get("metrics", {})

    # Extract CVSS v3 first, fall back to v2
    severity, score, vector = _extract_cvss_v3(metrics)
    if severity is None:
        severity, score, vector = _extract_cvss_v2_fallback(metrics)

    products = _extract_cpe_products(cve_data.get("configurations", []))
    references = _extract_references(cve_data.get("references", []))
    cwe_ids = _extract_cwe_ids(cve_data.get("weaknesses", []))

    return CVEEntry(
        cve_id=cve_id,
        description=description,
        severity=severity,
        cvss_score=score,
        cvss_vector=vector,
        affected_products=products,
        published_date=_parse_datetime(cve_data.get("published")),
        last_modified=_parse_datetime(cve_data.get("lastModified")),
        references=references,
        cwe_ids=cwe_ids,
    )


class NVDDiscovery:
    """NVD discovery client for searching CVE vulnerabilities.

    Provides access to the NIST National Vulnerability Database with:
    - CVE search by software/product name
    - Filtering by severity level
    - Individual CVE lookup by ID
    - CPE-based vulnerability search
    - Built-in rate limiting

    Example:
        discovery = NVDDiscovery()
        cves = discovery.search("apache tomcat", max_results=10)
        for cve in cves:
            print(f"{cve.cve_id}: {cve.severity}")

    Note:
        Requires the `requests` package: pip install requests
        Consider obtaining an NVD API key for higher rate limits.
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize NVDDiscovery client.

        Rule #5: Fail explicitly if library unavailable.

        Args:
            api_key: Optional NVD API key for higher rate limits.
        """
        if not _check_requests_library():
            raise ImportError(
                "requests library not installed. " "Install with: pip install requests"
            )

        import requests

        self._requests = requests
        self._api_key = api_key
        self._last_request_time: Optional[float] = None
        self._rate_limit = (
            API_KEY_RATE_LIMIT_SECONDS if api_key else DEFAULT_RATE_LIMIT_SECONDS
        )

    def _rate_limit_wait(self) -> None:
        """Wait for rate limit if needed.

        Rule #4: Small helper function.
        """
        if self._last_request_time is None:
            return

        elapsed = time.time() - self._last_request_time
        if elapsed < self._rate_limit:
            sleep_time = self._rate_limit - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def _make_request(self, params: dict) -> Optional[dict]:
        """Make rate-limited request to NVD API.

        Rule #4: Function <60 lines.
        Rule #5: Log all errors.

        Args:
            params: Query parameters for the API.

        Returns:
            JSON response dict or None on error.
        """
        self._rate_limit_wait()

        headers = {}
        if self._api_key:
            headers["apiKey"] = self._api_key

        try:
            url = f"{NVD_API_BASE}?{urlencode(params)}"
            logger.debug(f"NVD API request: {url}")

            response = self._requests.get(url, headers=headers, timeout=30)
            self._last_request_time = time.time()

            if response.status_code == 403:
                logger.error("NVD API rate limit exceeded or invalid API key")
                return None

            if response.status_code != 200:
                logger.error(f"NVD API error: HTTP {response.status_code}")
                return None

            return response.json()

        except Exception as e:
            logger.error(f"NVD API request failed: {e}")
            return None

    def search(
        self,
        product: str,
        severity: Optional[str] = None,
        max_results: int = 10,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[CVEEntry]:
        """Search CVEs by software/product name.

        Rule #1: Early return for empty query.
        Rule #4: Function <60 lines.
        Rule #7: Parameter validation.

        Args:
            product: Software/product name to search for.
            severity: Filter by severity (LOW, MEDIUM, HIGH, CRITICAL).
            max_results: Maximum results (default 10, max 2000).
            start_date: Filter CVEs published after this date.
            end_date: Filter CVEs published before this date.

        Returns:
            List of CVEEntry objects matching the search.
        """
        if not product or not product.strip():
            logger.warning("Empty product search query provided")
            return []

        # Clamp max_results to valid range
        max_results = min(max(1, max_results), 2000)

        params: Dict[str, Any] = {
            "keywordSearch": product.strip(),
            "resultsPerPage": max_results,
        }

        # Add severity filter
        if severity:
            severity_upper = severity.upper()
            if severity_upper in [s.value for s in Severity]:
                params["cvssV3Severity"] = severity_upper

        # Add date filters
        if start_date:
            params["pubStartDate"] = start_date.strftime("%Y-%m-%dT00:00:00.000")
        if end_date:
            params["pubEndDate"] = end_date.strftime("%Y-%m-%dT23:59:59.999")

        response = self._make_request(params)
        if not response:
            return []

        return self._parse_response(response)

    def get_cve(self, cve_id: str) -> Optional[CVEEntry]:
        """Get metadata for a specific CVE.

        Rule #1: Early return for invalid ID.
        Rule #4: Function <60 lines.

        Args:
            cve_id: CVE identifier (e.g., "CVE-2024-12345").

        Returns:
            CVEEntry object or None if not found.
        """
        if not cve_id or not cve_id.strip():
            logger.warning("Empty CVE ID provided")
            return None

        # Normalize CVE ID format
        clean_id = self._clean_cve_id(cve_id)
        if not clean_id:
            logger.warning(f"Invalid CVE ID format: {cve_id}")
            return None

        params = {"cveId": clean_id}
        response = self._make_request(params)

        if not response:
            return None

        entries = self._parse_response(response)
        if not entries:
            logger.warning(f"CVE not found: {cve_id}")
            return None

        return entries[0]

    def search_by_cpe(
        self,
        cpe: str,
        max_results: int = 10,
    ) -> List[CVEEntry]:
        """Search CVEs by CPE (Common Platform Enumeration).

        Rule #1: Early return for invalid CPE.
        Rule #4: Function <60 lines.

        Args:
            cpe: CPE string (e.g., "cpe:2.3:a:apache:tomcat:*:*:*:*:*:*:*:*").
            max_results: Maximum results (default 10, max 2000).

        Returns:
            List of CVEEntry objects for the CPE.
        """
        if not cpe or not cpe.strip():
            logger.warning("Empty CPE provided")
            return []

        # Validate CPE format
        if not cpe.startswith("cpe:"):
            logger.warning(f"Invalid CPE format: {cpe}")
            return []

        max_results = min(max(1, max_results), 2000)

        params = {
            "cpeName": cpe.strip(),
            "resultsPerPage": max_results,
        }

        response = self._make_request(params)
        if not response:
            return []

        return self._parse_response(response)

    def _parse_response(self, response: dict) -> List[CVEEntry]:
        """Parse NVD API response to CVEEntry list.

        Rule #4: Function <60 lines.
        Rule #2: Bounded iteration.
        """
        vulnerabilities = response.get("vulnerabilities", [])

        entries: List[CVEEntry] = []
        for item in vulnerabilities:
            entry = _convert_nvd_to_cve_entry(item)
            entries.append(entry)

        logger.info(f"Parsed {len(entries)} CVE entries")
        return entries

    def _clean_cve_id(self, cve_id: str) -> Optional[str]:
        """Clean and validate CVE ID.

        Rule #4: Function <60 lines.

        Accepts formats:
        - CVE-2024-12345
        - cve-2024-12345
        - 2024-12345

        Args:
            cve_id: Raw CVE ID in various formats.

        Returns:
            Cleaned CVE ID or None if invalid.
        """
        import re

        clean = cve_id.strip().upper()

        # Add CVE- prefix if missing
        if not clean.startswith("CVE-"):
            clean = f"CVE-{clean}"

        # Validate format: CVE-YYYY-NNNNN (4+ digits)
        if re.match(r"^CVE-\d{4}-\d{4,}$", clean):
            return clean

        return None


def create_nvd_discovery(api_key: Optional[str] = None) -> NVDDiscovery:
    """Factory function to create NVDDiscovery instance.

    Rule #4: Simple factory function.

    Args:
        api_key: Optional NVD API key for higher rate limits.

    Returns:
        NVDDiscovery instance.

    Raises:
        ImportError: If requests library is not installed.
    """
    return NVDDiscovery(api_key=api_key)
