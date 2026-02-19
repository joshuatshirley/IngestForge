"""Semantic Scholar API client - Search papers with citation data.

This module provides a production-ready Semantic Scholar API client with:
- Paper search with citation counts
- Individual paper lookup by ID
- Citation and reference traversal
- Rate limiting (100 requests/5 minutes for free tier)
- BibTeX export support"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Author:
    """Semantic Scholar author data.

    Rule #9: Full type hints.
    """

    author_id: Optional[str]
    name: str
    url: Optional[str] = None


@dataclass
class ScholarPaper:
    """Complete Semantic Scholar paper metadata.

    Rule #9: Full type hints on all fields.
    """

    paper_id: str
    title: str
    authors: List[Author]
    abstract: Optional[str]
    year: Optional[int]
    citation_count: int
    reference_count: int
    url: str
    venue: Optional[str] = None
    publication_date: Optional[str] = None
    external_ids: Dict[str, str] = field(default_factory=dict)
    fields_of_study: List[str] = field(default_factory=list)
    is_open_access: bool = False
    open_access_pdf_url: Optional[str] = None

    @property
    def doi(self) -> Optional[str]:
        """Get DOI if available."""
        return self.external_ids.get("DOI")

    @property
    def arxiv_id(self) -> Optional[str]:
        """Get arXiv ID if available."""
        return self.external_ids.get("ArXiv")

    def to_bibtex(self) -> str:
        """Convert paper to BibTeX format.

        Rule #4: Function <60 lines.
        """
        # Create citation key
        first_author = self.authors[0].name.split()[-1] if self.authors else "unknown"
        year = self.year or "unknown"
        key = f"{first_author.lower()}{year}"

        # Format authors
        authors_str = " and ".join(a.name for a in self.authors)

        lines = [
            f"@article{{{key},",
            f"  title = {{{self.title}}},",
            f"  author = {{{authors_str}}},",
            f"  year = {{{year}}},",
        ]

        if self.venue:
            lines.append(f"  journal = {{{self.venue}}},")
        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.url:
            lines.append(f"  url = {{{self.url}}},")

        lines.append("}")
        return "\n".join(lines)


@dataclass
class CitationResult:
    """Result of citation/reference traversal.

    Rule #9: Full type hints.
    """

    source_paper_id: str
    papers: List[ScholarPaper]
    depth: int
    total_found: int


class _RateLimiter:
    """Rate limiter for Semantic Scholar API.

    Free tier: 100 requests per 5 minutes.
    Rule #6: Encapsulates state at smallest scope.
    """

    def __init__(
        self, requests_per_window: int = 100, window_seconds: int = 300
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_window: Max requests per time window.
            window_seconds: Time window in seconds.
        """
        self._requests_per_window = requests_per_window
        self._window_seconds = window_seconds
        self._request_times: List[float] = []

    def wait_if_needed(self) -> None:
        """Wait if rate limit would be exceeded."""
        now = time.time()
        window_start = now - self._window_seconds

        # Remove old requests outside window
        self._request_times = [t for t in self._request_times if t > window_start]

        # Check if at limit
        if len(self._request_times) >= self._requests_per_window:
            oldest = self._request_times[0]
            sleep_time = oldest + self._window_seconds - now + 0.1
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self._request_times = []

    def mark_call(self) -> None:
        """Record that a call was made."""
        self._request_times.append(time.time())


class SemanticScholarClient:
    """Semantic Scholar API client for paper search and citation data.

    Provides access to the Semantic Scholar API with:
    - Paper search with full metadata
    - Individual paper lookup
    - Citation and reference traversal
    - Rate limiting for free tier

    Example:
        client = SemanticScholarClient()
        papers = client.search("attention is all you need", limit=10)
        for paper in papers:
            print(f"{paper.title} ({paper.citation_count} citations)")
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    PAPER_FIELDS = [
        "paperId",
        "title",
        "abstract",
        "year",
        "citationCount",
        "referenceCount",
        "url",
        "venue",
        "publicationDate",
        "externalIds",
        "fieldsOfStudy",
        "isOpenAccess",
        "openAccessPdf",
        "authors.authorId",
        "authors.name",
        "authors.url",
    ]

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Initialize Semantic Scholar client.

        Args:
            api_key: Optional API key for higher rate limits.
        """
        self._api_key = api_key
        self._limiter = _RateLimiter()

    def search(
        self,
        query: str,
        limit: int = 10,
        fields_of_study: Optional[List[str]] = None,
        year_range: Optional[tuple[int, int]] = None,
    ) -> List[ScholarPaper]:
        """Search for papers matching query.

        Rule #1: Early return for empty query.
        Rule #4: Function <60 lines.

        Args:
            query: Search query string.
            limit: Maximum results (1-100).
            fields_of_study: Filter by fields (e.g., ["Computer Science"]).
            year_range: Filter by year range (start, end).

        Returns:
            List of ScholarPaper objects.
        """
        if not query or not query.strip():
            logger.warning("Empty search query")
            return []

        limit = min(max(1, limit), 100)

        # Build URL
        params = {
            "query": query,
            "limit": limit,
            "fields": ",".join(self.PAPER_FIELDS),
        }

        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        url = f"{self.BASE_URL}/paper/search?{urlencode(params)}"
        data = self._fetch_api(url)

        if not data:
            return []

        return self._parse_search_results(data)

    def get_paper(self, paper_id: str) -> Optional[ScholarPaper]:
        """Get metadata for a specific paper.

        Rule #1: Early return for invalid ID.

        Args:
            paper_id: Semantic Scholar paper ID, DOI, or arXiv ID.
                Formats: "649def34f8be52c8b66281af98ae884c09aef38b",
                        "DOI:10.1234/example", "ARXIV:2301.12345"

        Returns:
            ScholarPaper or None if not found.
        """
        if not paper_id:
            logger.warning("Empty paper ID")
            return None

        fields = ",".join(self.PAPER_FIELDS)
        url = f"{self.BASE_URL}/paper/{paper_id}?fields={fields}"
        data = self._fetch_api(url)

        if not data:
            return None

        return self._parse_paper(data)

    def get_citations(
        self,
        paper_id: str,
        limit: int = 100,
        depth: int = 1,
    ) -> CitationResult:
        """Get papers that cite the given paper.

        Rule #1: Early return for invalid parameters, max 3 nesting levels.
        Rule #4: Function <60 lines.

        Args:
            paper_id: Paper ID to get citations for.
            limit: Maximum citations per level.
            depth: How many levels deep to traverse (1-3).

        Returns:
            CitationResult with citing papers.
        """
        if not paper_id:
            return CitationResult(paper_id, [], 0, 0)

        depth = min(max(1, depth), 3)
        limit = min(max(1, limit), 1000)

        all_papers = []
        current_ids = [paper_id]
        seen_ids = {paper_id}

        for level in range(depth):
            next_ids = []
            for pid in current_ids:
                papers = self._fetch_citations(pid, limit)
                next_ids.extend(
                    self._process_citation_papers(papers, seen_ids, all_papers)
                )

            current_ids = next_ids[:limit]
            if not current_ids:
                break

        return CitationResult(
            source_paper_id=paper_id,
            papers=all_papers,
            depth=depth,
            total_found=len(all_papers),
        )

    def _process_citation_papers(
        self,
        papers: List[ScholarPaper],
        seen_ids: set[str],
        all_papers: List[ScholarPaper],
    ) -> List[str]:
        """Process citation papers and update tracking sets.

        Rule #1: Extracted to reduce nesting in get_citations.

        Args:
            papers: Papers to process
            seen_ids: Set of already seen paper IDs
            all_papers: List to append new papers to

        Returns:
            List of new paper IDs
        """
        new_ids = []
        for paper in papers:
            if paper.paper_id not in seen_ids:
                seen_ids.add(paper.paper_id)
                all_papers.append(paper)
                new_ids.append(paper.paper_id)
        return new_ids

    def get_references(
        self,
        paper_id: str,
        limit: int = 100,
    ) -> List[ScholarPaper]:
        """Get papers referenced by the given paper.

        Rule #1: Early return for invalid ID.

        Args:
            paper_id: Paper ID to get references for.
            limit: Maximum references to return.

        Returns:
            List of referenced papers.
        """
        if not paper_id:
            return []

        limit = min(max(1, limit), 1000)
        fields = ",".join(self.PAPER_FIELDS)
        url = (
            f"{self.BASE_URL}/paper/{paper_id}/references?fields={fields}&limit={limit}"
        )
        data = self._fetch_api(url)

        if not data:
            return []

        return self._parse_citation_list(data)

    def _fetch_citations(self, paper_id: str, limit: int) -> List[ScholarPaper]:
        """Fetch citing papers for a paper ID."""
        fields = ",".join(self.PAPER_FIELDS)
        url = (
            f"{self.BASE_URL}/paper/{paper_id}/citations?fields={fields}&limit={limit}"
        )
        data = self._fetch_api(url)

        if not data:
            return []

        return self._parse_citation_list(data)

    def _fetch_api(self, url: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Semantic Scholar API.

        Rule #1: Early return for errors.
        Rule #5: Log all errors.
        """
        import urllib.request
        import urllib.error

        self._limiter.wait_if_needed()

        headers = {"User-Agent": "IngestForge/1.0 (Academic Research Tool)"}
        if self._api_key:
            headers["x-api-key"] = self._api_key

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read())

            self._limiter.mark_call()
            return data

        except urllib.error.HTTPError as e:
            if e.code == 404:
                logger.debug(f"Paper not found: {url}")
            elif e.code == 429:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)
            else:
                logger.error(f"Semantic Scholar API error: HTTP {e.code}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return None

    def _parse_search_results(self, data: Dict[str, Any]) -> List[ScholarPaper]:
        """Parse search results from API response."""
        papers = []
        for item in data.get("data", []):
            paper = self._parse_paper(item)
            if paper:
                papers.append(paper)
        return papers

    def _parse_citation_list(self, data: Dict[str, Any]) -> List[ScholarPaper]:
        """Parse citations/references list from API response."""
        papers = []
        for item in data.get("data", []):
            # Citations come wrapped in "citingPaper" or "citedPaper"
            paper_data = item.get("citingPaper") or item.get("citedPaper") or item
            paper = self._parse_paper(paper_data)
            if paper:
                papers.append(paper)
        return papers

    def _parse_paper(self, data: Dict[str, Any]) -> Optional[ScholarPaper]:
        """Parse paper data to ScholarPaper.

        Rule #1: Early return for missing data.
        Rule #4: Function <60 lines.
        """
        paper_id = data.get("paperId")
        title = data.get("title")

        if not paper_id or not title:
            return None

        # Parse authors
        authors = []
        for author_data in data.get("authors", []):
            authors.append(
                Author(
                    author_id=author_data.get("authorId"),
                    name=author_data.get("name", "Unknown"),
                    url=author_data.get("url"),
                )
            )

        # Parse external IDs
        external_ids = data.get("externalIds") or {}

        # Parse open access PDF
        oa_pdf = data.get("openAccessPdf")
        oa_pdf_url = oa_pdf.get("url") if isinstance(oa_pdf, dict) else None

        return ScholarPaper(
            paper_id=paper_id,
            title=title,
            authors=authors,
            abstract=data.get("abstract"),
            year=data.get("year"),
            citation_count=data.get("citationCount", 0),
            reference_count=data.get("referenceCount", 0),
            url=data.get("url", ""),
            venue=data.get("venue"),
            publication_date=data.get("publicationDate"),
            external_ids=external_ids,
            fields_of_study=data.get("fieldsOfStudy") or [],
            is_open_access=data.get("isOpenAccess", False),
            open_access_pdf_url=oa_pdf_url,
        )


def export_bibtex(papers: List[ScholarPaper]) -> str:
    """Export list of papers to BibTeX format.

    Rule #4: Function <60 lines.
    Rule #9: Full type hints.

    Args:
        papers: List of ScholarPaper objects.

    Returns:
        BibTeX formatted string.
    """
    entries = [paper.to_bibtex() for paper in papers]
    return "\n\n".join(entries)
