"""CourtListener API client - Court opinions and case law search.

This module provides a production-ready CourtListener API client with:
- Court opinion search by keyword
- Jurisdiction filtering (Federal/State circuits)
- Case metadata extraction
- Opinion download support
- Rate limiting (API compliance)"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class PrecedentialStatus(Enum):
    """Court opinion precedential status."""

    PUBLISHED = "Published"
    UNPUBLISHED = "Unpublished"
    ERRATA = "Errata"
    SEPARATE = "Separate"
    IN_CHAMBERS = "In-chambers"
    RELATING_TO = "Relating-to"
    UNKNOWN = "Unknown"


class JurisdictionType(Enum):
    """Federal vs state jurisdiction."""

    FEDERAL = "federal"
    STATE = "state"


@dataclass
class CourtCase:
    """Complete court case metadata.

    Rule #9: Full type hints on all fields.
    """

    docket_number: str
    case_name: str
    court: str
    jurisdiction: str
    date_filed: Optional[datetime]
    date_decided: Optional[datetime]
    opinion_url: str
    precedential_status: str
    judges: List[str] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)
    case_id: Optional[str] = None
    opinion_id: Optional[str] = None
    cluster_id: Optional[str] = None
    court_id: Optional[str] = None
    absolute_url: Optional[str] = None
    snippet: Optional[str] = None

    @property
    def year(self) -> Optional[int]:
        """Get decision year."""
        if self.date_decided:
            return self.date_decided.year
        if self.date_filed:
            return self.date_filed.year
        return None

    def to_citation_string(self) -> str:
        """Generate a basic citation string.

        Rule #4: Function <60 lines.
        """
        parts = [self.case_name]
        if self.citations:
            parts.append(self.citations[0])
        if self.court:
            parts.append(f"({self.court}")
            if self.year:
                parts[-1] += f" {self.year})"
            else:
                parts[-1] += ")"
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "docket_number": self.docket_number,
            "case_name": self.case_name,
            "court": self.court,
            "jurisdiction": self.jurisdiction,
            "judges": self.judges,
            "date_filed": self.date_filed.isoformat() if self.date_filed else None,
            "date_decided": self.date_decided.isoformat()
            if self.date_decided
            else None,
            "citations": self.citations,
            "opinion_url": self.opinion_url,
            "precedential_status": self.precedential_status,
            "case_id": self.case_id,
            "opinion_id": self.opinion_id,
            "cluster_id": self.cluster_id,
            "snippet": self.snippet,
        }


class _RateLimiter:
    """Rate limiter for CourtListener API.

    Rule #6: Encapsulates state at smallest scope.
    """

    def __init__(self, delay: float = 0.5) -> None:
        """Initialize rate limiter.

        Args:
            delay: Minimum seconds between requests (CourtListener recommends 0.5s).
        """
        self._delay = delay
        self._last_call: float = 0.0

    def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_call
        if elapsed < self._delay:
            sleep_time = self._delay - elapsed
            time.sleep(sleep_time)

    def mark_call(self) -> None:
        """Record that a call was made."""
        self._last_call = time.time()


# Federal circuit court mappings
FEDERAL_COURTS = {
    "scotus": "Supreme Court of the United States",
    "ca1": "First Circuit",
    "ca2": "Second Circuit",
    "ca3": "Third Circuit",
    "ca4": "Fourth Circuit",
    "ca5": "Fifth Circuit",
    "ca6": "Sixth Circuit",
    "ca7": "Seventh Circuit",
    "ca8": "Eighth Circuit",
    "ca9": "Ninth Circuit",
    "ca10": "Tenth Circuit",
    "ca11": "Eleventh Circuit",
    "cadc": "D.C. Circuit",
    "cafc": "Federal Circuit",
}


class CourtListenerDiscovery:
    """CourtListener API client for court opinion search.

    Provides access to the Free Law Project's CourtListener API with:
    - Opinion search by keyword
    - Jurisdiction filtering
    - Case metadata extraction
    - Opinion download

    Example:
        client = CourtListenerDiscovery()
        cases = client.search("qualified immunity", jurisdiction="ca9", max_results=5)
        for case in cases:
            print(f"{case.case_name} ({case.year})")
    """

    BASE_URL = "https://www.courtlistener.com/api/rest/v4"

    def __init__(self, api_token: Optional[str] = None) -> None:
        """Initialize CourtListener client.

        Args:
            api_token: CourtListener API token (optional, but recommended).
                      Without token, rate limits are more restrictive.
        """
        self._api_token = api_token
        self._limiter = _RateLimiter()

    def search(
        self,
        query: str,
        jurisdiction: Optional[str] = None,
        max_results: int = 5,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[CourtCase]:
        """Search court opinions by keyword.

        Rule #1: Early return for empty query.
        Rule #4: Function <60 lines.

        Args:
            query: Search query string.
            jurisdiction: Court ID filter (e.g., "ca9", "scotus", "calctapp").
            max_results: Maximum results to return (1-100).
            from_date: Filter opinions from this date.
            to_date: Filter opinions until this date.

        Returns:
            List of CourtCase objects with metadata.
        """
        if not query or not query.strip():
            logger.warning("Empty search query")
            return []

        max_results = min(max(1, max_results), 100)

        params = self._build_search_params(
            query, jurisdiction, max_results, from_date, to_date
        )
        url = f"{self.BASE_URL}/search/?{urlencode(params)}"
        data = self._fetch_api(url)

        if not data:
            return []

        return self._parse_search_results(data)

    def _build_search_params(
        self,
        query: str,
        jurisdiction: Optional[str],
        max_results: int,
        from_date: Optional[datetime],
        to_date: Optional[datetime],
    ) -> Dict[str, Any]:
        """Build search API parameters.

        Rule #4: Function <60 lines.
        """
        params: Dict[str, Any] = {
            "q": query,
            "type": "o",  # opinions
            "order_by": "score desc",
        }

        if jurisdiction:
            params["court"] = jurisdiction

        if from_date:
            params["filed_after"] = from_date.strftime("%Y-%m-%d")

        if to_date:
            params["filed_before"] = to_date.strftime("%Y-%m-%d")

        # Note: CourtListener pagination uses cursor-based approach
        # For simplicity, we use page_size for initial results
        params["page_size"] = max_results

        return params

    def get_case(self, cluster_id: str) -> Optional[CourtCase]:
        """Get case details by cluster ID.

        Rule #1: Early return for missing data.
        Rule #4: Function <60 lines.

        Args:
            cluster_id: CourtListener cluster ID.

        Returns:
            CourtCase or None if not found.
        """
        if not cluster_id:
            logger.warning("Empty cluster ID")
            return None

        url = f"{self.BASE_URL}/clusters/{cluster_id}/"
        data = self._fetch_api(url)

        if not data:
            return None

        return self._parse_cluster(data)

    def get_opinion(self, opinion_id: str) -> Optional[Dict[str, Any]]:
        """Get full opinion text by opinion ID.

        Args:
            opinion_id: CourtListener opinion ID.

        Returns:
            Dictionary with opinion text or None if not found.
        """
        if not opinion_id:
            logger.warning("Empty opinion ID")
            return None

        url = f"{self.BASE_URL}/opinions/{opinion_id}/"
        return self._fetch_api(url)

    def download_opinion(
        self,
        case: CourtCase,
        output_dir: Path,
        format_type: str = "txt",
    ) -> Optional[Path]:
        """Download opinion text to file.

        Rule #1: Early return for missing data.
        Rule #4: Function <60 lines.

        Args:
            case: CourtCase object with opinion_id.
            output_dir: Directory to save the opinion.
            format_type: Output format ("txt" or "html").

        Returns:
            Path to downloaded file or None on failure.
        """
        if not case.opinion_id:
            logger.warning(f"No opinion ID for case: {case.case_name}")
            return None

        opinion_data = self.get_opinion(case.opinion_id)
        if not opinion_data:
            return None

        # Get opinion text
        text = self._extract_opinion_text(opinion_data)
        if not text:
            logger.warning(f"No opinion text found for: {case.case_name}")
            return None

        # Create output file
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = self._sanitize_filename(case.case_name)
        filename = f"{safe_name}.{format_type}"
        output_path = output_dir / filename

        output_path.write_text(text, encoding="utf-8")
        logger.info(f"Downloaded opinion to: {output_path}")

        return output_path

    def _extract_opinion_text(self, opinion_data: Dict[str, Any]) -> Optional[str]:
        """Extract text from opinion API response."""
        # Try different text fields in order of preference
        for field in ["plain_text", "html_with_citations", "html", "html_lawbox"]:
            text = opinion_data.get(field)
            if text:
                if field.startswith("html"):
                    text = self._strip_html(text)
                return text
        return None

    def _strip_html(self, html: str) -> str:
        """Strip HTML tags from text."""
        import re

        clean = re.sub(r"<[^>]+>", "", html)
        return " ".join(clean.split())

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize case name for filename."""
        import re

        # Remove invalid characters
        safe = re.sub(r'[<>:"/\\|?*]', "", name)
        # Truncate and strip whitespace
        return safe[:100].strip()

    def _fetch_api(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch data from CourtListener API.

        Rule #1: Early return for errors.
        Rule #5: Log all errors.
        """
        import urllib.request
        import urllib.error

        self._limiter.wait_if_needed()

        headers = {
            "User-Agent": "IngestForge/1.0 (Legal Research Tool)",
            "Accept": "application/json",
        }

        if self._api_token:
            headers["Authorization"] = f"Token {self._api_token}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())

            self._limiter.mark_call()
            return data

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"Resource not found: {url}")
            elif e.code == 429:
                logger.warning("Rate limited by CourtListener API")
            else:
                logger.error(f"CourtListener API error: HTTP {e.code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching {url}: {e}")
            return None

    def _parse_search_results(self, data: Dict[str, Any]) -> List[CourtCase]:
        """Parse search results from API response.

        Rule #4: Function <60 lines.
        """
        cases = []
        results = data.get("results", [])

        for item in results:
            case = self._parse_search_result(item)
            if case:
                cases.append(case)

        return cases

    def _parse_search_result(self, item: Dict[str, Any]) -> Optional[CourtCase]:
        """Parse a single search result item.

        Rule #1: Early return for missing required fields.
        Rule #4: Function <60 lines.
        """
        case_name = item.get("caseName") or item.get("case_name")
        if not case_name:
            return None

        # Extract IDs from URLs or direct fields
        cluster_id = self._extract_id_from_url(item.get("cluster"))
        opinion_id = self._extract_id_from_url(item.get("id"))

        # Parse dates
        date_filed = self._parse_date(item.get("dateFiled") or item.get("date_filed"))
        date_decided = self._parse_date(
            item.get("dateArgued") or item.get("date_argued")
        )

        # Build opinion URL
        absolute_url = item.get("absolute_url", "")
        opinion_url = (
            f"https://www.courtlistener.com{absolute_url}" if absolute_url else ""
        )

        # Extract court info
        court = item.get("court") or ""
        court_id = item.get("court_id") or self._extract_id_from_url(court)

        return CourtCase(
            docket_number=item.get("docketNumber") or item.get("docket_number") or "",
            case_name=case_name,
            court=self._get_court_name(court_id) if court_id else court,
            jurisdiction=self._get_jurisdiction(court_id) if court_id else "Unknown",
            judges=self._parse_judges(item.get("judge", "")),
            date_filed=date_filed,
            date_decided=date_decided,
            citations=item.get("citation", [])
            if isinstance(item.get("citation"), list)
            else [],
            opinion_url=opinion_url,
            precedential_status=item.get("status") or "Unknown",
            case_id=item.get("id"),
            opinion_id=opinion_id,
            cluster_id=cluster_id,
            court_id=court_id,
            absolute_url=absolute_url,
            snippet=item.get("snippet"),
        )

    def _parse_cluster(self, data: Dict[str, Any]) -> Optional[CourtCase]:
        """Parse cluster data into CourtCase.

        Rule #4: Function <60 lines.
        """
        case_name = data.get("case_name") or data.get("case_name_short")
        if not case_name:
            return None

        # Extract court info
        court_url = data.get("court", "")
        court_id = self._extract_id_from_url(court_url)

        # Get opinion IDs if available
        opinions = data.get("sub_opinions", [])
        opinion_id = None
        if opinions:
            opinion_id = self._extract_id_from_url(opinions[0])

        return CourtCase(
            docket_number=data.get("docket_number", ""),
            case_name=case_name,
            court=self._get_court_name(court_id) if court_id else "",
            jurisdiction=self._get_jurisdiction(court_id) if court_id else "Unknown",
            judges=self._parse_judges(data.get("judges", "")),
            date_filed=self._parse_date(data.get("date_filed")),
            date_decided=self._parse_date(data.get("date_blocked")),
            citations=data.get("citations", []),
            opinion_url=f"https://www.courtlistener.com{data.get('absolute_url', '')}",
            precedential_status=data.get("precedential_status", "Unknown"),
            cluster_id=str(data.get("id", "")),
            opinion_id=opinion_id,
            court_id=court_id,
        )

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse date string to datetime."""
        if not date_str:
            return None

        try:
            # Handle ISO format
            if "T" in date_str:
                return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            # Handle simple date format
            return datetime.strptime(date_str, "%Y-%m-%d")
        except (ValueError, TypeError):
            return None

    def _parse_judges(self, judges_str: str) -> List[str]:
        """Parse judge string into list."""
        if not judges_str:
            return []
        # Split by common separators
        import re

        judges = re.split(r"[,;]", judges_str)
        return [j.strip() for j in judges if j.strip()]

    def _extract_id_from_url(self, url: Optional[str]) -> Optional[str]:
        """Extract ID from CourtListener API URL."""
        if not url:
            return None
        if isinstance(url, str):
            parts = url.rstrip("/").split("/")
            return parts[-1] if parts else None
        return str(url)

    def _get_court_name(self, court_id: Optional[str]) -> str:
        """Get human-readable court name."""
        if not court_id:
            return "Unknown Court"
        return FEDERAL_COURTS.get(court_id.lower(), court_id)

    def _get_jurisdiction(self, court_id: Optional[str]) -> str:
        """Determine jurisdiction type from court ID."""
        if not court_id:
            return "Unknown"
        court_lower = court_id.lower()
        if court_lower in FEDERAL_COURTS:
            return "Federal"
        if court_lower.startswith("ca"):
            return "Federal"
        return "State"
