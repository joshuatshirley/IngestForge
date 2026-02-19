"""CrossRef API client - DOI lookup and publication search.

This module provides a production-ready CrossRef API client with:
- DOI lookup for complete publication metadata
- Full-text search with filters
- Citation extraction
- Rate limiting (polite pool compliance)
- BibTeX export support"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class PublicationType(Enum):
    """CrossRef publication types."""

    JOURNAL_ARTICLE = "journal-article"
    PROCEEDINGS_ARTICLE = "proceedings-article"
    BOOK_CHAPTER = "book-chapter"
    BOOK = "book"
    POSTED_CONTENT = "posted-content"
    DISSERTATION = "dissertation"
    REPORT = "report"
    DATASET = "dataset"


@dataclass
class Author:
    """CrossRef author data.

    Rule #9: Full type hints.
    """

    given: Optional[str]
    family: str
    sequence: str = "additional"
    affiliation: List[str] = field(default_factory=list)
    orcid: Optional[str] = None

    @property
    def full_name(self) -> str:
        """Get full author name."""
        if self.given:
            return f"{self.given} {self.family}"
        return self.family


@dataclass
class Publication:
    """Complete CrossRef publication metadata.

    Rule #9: Full type hints on all fields.
    """

    doi: str
    title: str
    authors: List[Author]
    container_title: Optional[str]  # Journal/book name
    publisher: Optional[str]
    published_date: Optional[datetime]
    publication_type: str
    url: str
    abstract: Optional[str] = None
    issn: List[str] = field(default_factory=list)
    isbn: List[str] = field(default_factory=list)
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    reference_count: int = 0
    is_referenced_by_count: int = 0
    subject: List[str] = field(default_factory=list)
    license_url: Optional[str] = None

    @property
    def year(self) -> Optional[int]:
        """Get publication year."""
        return self.published_date.year if self.published_date else None

    def to_bibtex(self) -> str:
        """Convert publication to BibTeX format.

        Rule #4: Function <60 lines.
        """
        # Determine entry type
        entry_type = self._get_bibtex_type()

        # Create citation key
        first_author = self.authors[0].family if self.authors else "unknown"
        year = self.year or "unknown"
        key = f"{first_author.lower()}{year}"

        # Format authors
        authors_str = " and ".join(a.full_name for a in self.authors)

        lines = [
            f"@{entry_type}{{{key},",
            f"  title = {{{self.title}}},",
            f"  author = {{{authors_str}}},",
            f"  year = {{{year}}},",
            f"  doi = {{{self.doi}}},",
        ]

        if self.container_title:
            lines.append(f"  journal = {{{self.container_title}}},")
        if self.publisher:
            lines.append(f"  publisher = {{{self.publisher}}},")
        if self.volume:
            lines.append(f"  volume = {{{self.volume}}},")
        if self.issue:
            lines.append(f"  number = {{{self.issue}}},")
        if self.pages:
            lines.append(f"  pages = {{{self.pages}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")

        lines.append("}")
        return "\n".join(lines)

    def _get_bibtex_type(self) -> str:
        """Get appropriate BibTeX entry type."""
        type_map = {
            "journal-article": "article",
            "proceedings-article": "inproceedings",
            "book-chapter": "incollection",
            "book": "book",
            "dissertation": "phdthesis",
            "report": "techreport",
        }
        return type_map.get(self.publication_type, "misc")


class _RateLimiter:
    """Rate limiter for CrossRef API.

    Polite pool: send mailto parameter, respect rate limits.
    Rule #6: Encapsulates state at smallest scope.
    """

    def __init__(self, delay: float = 0.1) -> None:
        """Initialize rate limiter.

        Args:
            delay: Minimum seconds between requests.
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


class CrossRefClient:
    """CrossRef API client for DOI lookup and publication search.

    Provides access to the CrossRef API with:
    - DOI-based publication lookup
    - Full-text search with filters
    - Reference extraction
    - Rate limiting

    Example:
        client = CrossRefClient()
        pub = client.lookup_doi("10.1038/nature12373")
        print(f"{pub.title} by {pub.authors[0].full_name}")
    """

    BASE_URL = "https://api.crossref.org"

    def __init__(self, mailto: Optional[str] = None) -> None:
        """Initialize CrossRef client.

        Args:
            mailto: Email for polite pool (recommended for better rate limits).
        """
        self._mailto = mailto or "ingestforge@example.com"
        self._limiter = _RateLimiter()

    def lookup_doi(self, doi: str) -> Optional[Publication]:
        """Look up a publication by DOI.

        Rule #1: Early return for invalid DOI.
        Rule #4: Function <60 lines.

        Args:
            doi: DOI string (with or without "https://doi.org/" prefix).

        Returns:
            Publication or None if not found.
        """
        clean_doi = self._clean_doi(doi)
        if not clean_doi:
            logger.warning(f"Invalid DOI format: {doi}")
            return None

        url = f"{self.BASE_URL}/works/{quote(clean_doi, safe='')}"
        data = self._fetch_api(url)

        if not data:
            return None

        return self._parse_publication(data.get("message", {}))

    def search(
        self,
        query: str,
        limit: int = 20,
        publication_type: Optional[PublicationType] = None,
        from_year: Optional[int] = None,
        to_year: Optional[int] = None,
        sort: str = "relevance",
    ) -> List[Publication]:
        """Search for publications.

        Rule #1: Early return for empty query.
        Rule #4: Function <60 lines.

        Args:
            query: Search query string.
            limit: Maximum results (1-100).
            publication_type: Filter by publication type.
            from_year: Filter publications from this year.
            to_year: Filter publications up to this year.
            sort: Sort order ("relevance", "published", "cited").

        Returns:
            List of Publication objects.
        """
        if not query or not query.strip():
            logger.warning("Empty search query")
            return []

        limit = min(max(1, limit), 100)

        # Build parameters
        params = {
            "query": query,
            "rows": limit,
            "mailto": self._mailto,
        }

        if publication_type:
            params["filter"] = f"type:{publication_type.value}"
        if from_year:
            params["filter"] = params.get("filter", "") + f",from-pub-date:{from_year}"
        if to_year:
            params["filter"] = params.get("filter", "") + f",until-pub-date:{to_year}"

        if sort == "published":
            params["sort"] = "published"
            params["order"] = "desc"
        elif sort == "cited":
            params["sort"] = "is-referenced-by-count"
            params["order"] = "desc"

        url = f"{self.BASE_URL}/works?{urlencode(params)}"
        data = self._fetch_api(url)

        if not data:
            return []

        return self._parse_search_results(data)

    def get_references(self, doi: str, limit: int = 100) -> List[Publication]:
        """Get references cited by a publication.

        Note: Not all references have full CrossRef records.

        Args:
            doi: DOI of the citing publication.
            limit: Maximum references to return.

        Returns:
            List of publications that could be resolved.
        """
        clean_doi = self._clean_doi(doi)
        if not clean_doi:
            return []

        # Get the publication with its references
        url = f"{self.BASE_URL}/works/{quote(clean_doi, safe='')}"
        data = self._fetch_api(url)

        if not data:
            return []

        message = data.get("message", {})
        references = message.get("reference", [])

        # Try to resolve each reference with a DOI
        resolved = []
        for ref in references[:limit]:
            ref_doi = ref.get("DOI")
            if ref_doi and len(resolved) < limit:
                pub = self.lookup_doi(ref_doi)
                if pub:
                    resolved.append(pub)

        return resolved

    def _clean_doi(self, doi: str) -> Optional[str]:
        """Clean and validate DOI.

        Accepts formats:
        - 10.1234/example
        - https://doi.org/10.1234/example
        - doi:10.1234/example
        """
        if not doi:
            return None

        # Remove common prefixes
        clean = doi.strip()
        prefixes = ["https://doi.org/", "http://doi.org/", "doi:", "DOI:"]
        for prefix in prefixes:
            if clean.startswith(prefix):
                clean = clean[len(prefix) :]
                break

        # Validate DOI format (must start with 10.)
        if not clean.startswith("10."):
            return None

        return clean

    def _fetch_api(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch data from CrossRef API.

        Rule #1: Early return for errors.
        Rule #5: Log all errors.
        """
        import urllib.request
        import urllib.error

        self._limiter.wait_if_needed()

        # Add mailto for polite pool
        if "mailto=" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}mailto={self._mailto}"

        headers = {
            "User-Agent": f"IngestForge/1.0 (mailto:{self._mailto})",
            "Accept": "application/json",
        }

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())

            self._limiter.mark_call()
            return data

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"DOI not found: {url}")
            else:
                logger.error(f"CrossRef API error: HTTP {e.code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def _parse_search_results(self, data: Dict[str, Any]) -> List[Publication]:
        """Parse search results from API response."""
        publications = []
        message = data.get("message", {})

        for item in message.get("items", []):
            pub = self._parse_publication(item)
            if pub:
                publications.append(pub)

        return publications

    def _parse_publication(self, data: Dict[str, Any]) -> Optional[Publication]:
        """Parse publication data.

        Rule #1: Early return for missing data.
        Rule #4: Function <60 lines.
        """
        doi = data.get("DOI")
        titles = data.get("title", [])
        title = titles[0] if titles else None

        if not doi or not title:
            return None

        # Parse authors
        authors = self._parse_authors(data.get("author", []))

        # Parse container title (journal/book)
        container_titles = data.get("container-title", [])
        container_title = container_titles[0] if container_titles else None

        # Parse date
        published_date = self._parse_date(data)

        # Build URL
        url = f"https://doi.org/{doi}"

        # Parse abstract (may be HTML)
        abstract = data.get("abstract")
        if abstract:
            abstract = self._clean_html(abstract)

        return Publication(
            doi=doi,
            title=title,
            authors=authors,
            container_title=container_title,
            publisher=data.get("publisher"),
            published_date=published_date,
            publication_type=data.get("type", "unknown"),
            url=url,
            abstract=abstract,
            issn=data.get("ISSN", []),
            isbn=data.get("ISBN", []),
            volume=data.get("volume"),
            issue=data.get("issue"),
            pages=data.get("page"),
            reference_count=data.get("references-count", 0),
            is_referenced_by_count=data.get("is-referenced-by-count", 0),
            subject=data.get("subject", []),
            license_url=self._parse_license(data.get("license", [])),
        )

    def _parse_authors(self, authors_data: List[Dict[str, Any]]) -> List[Author]:
        """Parse author list from API response."""
        authors = []
        for item in authors_data:
            author = Author(
                given=item.get("given"),
                family=item.get("family", "Unknown"),
                sequence=item.get("sequence", "additional"),
                affiliation=[a.get("name", "") for a in item.get("affiliation", [])],
                orcid=item.get("ORCID"),
            )
            authors.append(author)
        return authors

    def _parse_date(self, data: Dict[str, Any]) -> Optional[datetime]:
        """Parse publication date from various date fields."""
        # Try different date fields in order of preference
        for field_name in [
            "published",
            "published-print",
            "published-online",
            "created",
        ]:
            date_parts = data.get(field_name, {}).get("date-parts", [[]])
            if date_parts and date_parts[0]:
                parts = date_parts[0]
                try:
                    year = parts[0]
                    month = parts[1] if len(parts) > 1 else 1
                    day = parts[2] if len(parts) > 2 else 1
                    return datetime(year, month, day)
                except (ValueError, IndexError):
                    continue
        return None

    def _parse_license(self, licenses: List[Dict[str, Any]]) -> Optional[str]:
        """Extract license URL from license data."""
        for lic in licenses:
            url = lic.get("URL")
            if url:
                return url
        return None

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        # Simple HTML tag removal
        clean = re.sub(r"<[^>]+>", "", text)
        # Normalize whitespace
        return " ".join(clean.split())


def export_bibtex(publications: List[Publication]) -> str:
    """Export list of publications to BibTeX format.

    Rule #4: Function <60 lines.
    Rule #9: Full type hints.

    Args:
        publications: List of Publication objects.

    Returns:
        BibTeX formatted string.
    """
    entries = [pub.to_bibtex() for pub in publications]
    return "\n\n".join(entries)
