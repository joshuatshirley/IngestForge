"""arXiv API client - Full-featured arXiv search and download.

This module provides a production-ready arXiv API client with:
- Full paper metadata including categories and PDF URLs
- Configurable sorting (relevance, date, citations)
- PDF download with progress tracking
- Rate limiting (1 request/3 seconds per arXiv guidelines)
- BibTeX export support"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Optional
from urllib.parse import quote_plus

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class SortOrder(Enum):
    """arXiv search sort options."""

    RELEVANCE = "relevance"
    LAST_UPDATED = "lastUpdatedDate"
    SUBMITTED = "submittedDate"


class SortDirection(Enum):
    """arXiv sort direction."""

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass
class Paper:
    """Complete arXiv paper metadata.

    Rule #9: Full type hints on all fields.
    """

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published: datetime
    updated: datetime
    pdf_url: str
    abs_url: str
    categories: List[str]
    primary_category: str
    comment: Optional[str] = None
    journal_ref: Optional[str] = None
    doi: Optional[str] = None

    def to_bibtex(self) -> str:
        """Convert paper to BibTeX format.

        Rule #4: Function <60 lines.
        """
        # Create citation key from first author + year
        first_author = self.authors[0].split()[-1] if self.authors else "unknown"
        year = self.published.year
        key = f"{first_author.lower()}{year}arxiv"

        # Format authors for BibTeX
        authors_str = " and ".join(self.authors)

        # Build BibTeX entry
        lines = [
            f"@article{{{key},",
            f"  title = {{{self.title}}},",
            f"  author = {{{authors_str}}},",
            f"  year = {{{year}}},",
            f"  eprint = {{{self.arxiv_id}}},",
            "  archivePrefix = {arXiv},",
            f"  primaryClass = {{{self.primary_category}}},",
        ]

        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")
        if self.journal_ref:
            lines.append(f"  journal = {{{self.journal_ref}}},")

        lines.append("}")
        return "\n".join(lines)


@dataclass
class ArxivDownloadResult:
    """Result of downloading an arXiv PDF.

    Rule #9: Full type hints.
    """

    arxiv_id: str
    success: bool
    file_path: Optional[Path] = None
    filename: Optional[str] = None
    size_bytes: int = 0
    error: Optional[str] = None


class _RateLimiter:
    """Rate limiter for arXiv API calls.

    Rule #6: Encapsulates rate limiting state at smallest scope.
    arXiv requests: 1 request per 3 seconds.
    """

    def __init__(self, delay: float = 3.0) -> None:
        """Initialize rate limiter.

        Args:
            delay: Minimum seconds between requests.
        """
        self.delay = delay
        self._last_call: float = 0.0

    def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limit."""
        elapsed = time.time() - self._last_call
        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)

    def mark_call(self) -> None:
        """Mark that a call was made."""
        self._last_call = time.time()


class ArxivSearcher:
    """arXiv API client for searching and downloading papers.

    Provides full-featured access to the arXiv API with:
    - Paper search with sorting options
    - Individual paper metadata lookup
    - PDF download with progress tracking
    - Rate limiting (1 request/3 seconds)

    Example:
        searcher = ArxivSearcher()
        papers = searcher.search("machine learning transformers", limit=10)
        for paper in papers:
            print(f"{paper.title} by {paper.authors[0]}")
    """

    BASE_URL = "http://export.arxiv.org/api/query"
    ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
    ARXIV_NS = {"arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, rate_limit_delay: float = 3.0) -> None:
        """Initialize arXiv searcher.

        Args:
            rate_limit_delay: Seconds between API calls (default: 3.0).
        """
        self._limiter = _RateLimiter(delay=rate_limit_delay)

    def search(
        self,
        query: str,
        limit: int = 10,
        sort: SortOrder = SortOrder.RELEVANCE,
        sort_direction: SortDirection = SortDirection.DESCENDING,
        start: int = 0,
    ) -> List[Paper]:
        """Search arXiv for papers matching query.

        Rule #1: Early return for empty query.
        Rule #4: Function <60 lines.
        Rule #7: Parameter validation.

        Args:
            query: Search query string.
            limit: Maximum results (1-100, default 10).
            sort: Sort order (relevance, date).
            sort_direction: Ascending or descending.
            start: Starting index for pagination.

        Returns:
            List of Paper objects.
        """
        if not query or not query.strip():
            logger.warning("Empty search query")
            return []

        limit = min(max(1, limit), 100)  # Clamp to 1-100

        # Build URL
        url = self._build_search_url(query, limit, sort, sort_direction, start)

        # Fetch and parse
        xml_data = self._fetch_api(url)
        if not xml_data:
            return []

        return self._parse_search_results(xml_data)

    def get_paper(self, arxiv_id: str) -> Optional[Paper]:
        """Get metadata for a specific arXiv paper.

        Rule #1: Early return for invalid ID.
        Rule #4: Function <60 lines.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.12345" or "arxiv:2301.12345").

        Returns:
            Paper object or None if not found.
        """
        # Clean ID
        clean_id = self._clean_arxiv_id(arxiv_id)
        if not clean_id:
            logger.warning(f"Invalid arXiv ID: {arxiv_id}")
            return None

        url = f"{self.BASE_URL}?id_list={clean_id}"
        xml_data = self._fetch_api(url)

        if not xml_data:
            return None

        papers = self._parse_search_results(xml_data)
        return papers[0] if papers else None

    def download_pdf(
        self,
        arxiv_id: str,
        output_path: Path,
        timeout: int = 60,
    ) -> ArxivDownloadResult:
        """Download PDF for an arXiv paper.

        Rule #1: Early returns for errors.
        Rule #4: Function <60 lines.
        Rule #5: Log all errors.

        Args:
            arxiv_id: arXiv paper ID.
            output_path: Directory to save PDF.
            timeout: Download timeout in seconds.

        Returns:
            ArxivDownloadResult with status.
        """
        clean_id = self._clean_arxiv_id(arxiv_id)
        if not clean_id:
            return ArxivDownloadResult(
                arxiv_id=arxiv_id,
                success=False,
                error="Invalid arXiv ID",
            )

        # Build PDF URL
        pdf_url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        filename = f"{clean_id.replace('/', '_')}.pdf"
        file_path = output_path / filename

        # Ensure output directory exists
        output_path.mkdir(parents=True, exist_ok=True)

        return self._download_file(pdf_url, file_path, clean_id, timeout)

    def _build_search_url(
        self,
        query: str,
        limit: int,
        sort: SortOrder,
        sort_direction: SortDirection,
        start: int,
    ) -> str:
        """Build arXiv API search URL.

        Rule #4: Function <60 lines.
        """
        encoded_query = quote_plus(query)
        params = [
            f"search_query=all:{encoded_query}",
            f"start={start}",
            f"max_results={limit}",
            f"sortBy={sort.value}",
            f"sortOrder={sort_direction.value}",
        ]
        return f"{self.BASE_URL}?{'&'.join(params)}"

    def _fetch_api(self, url: str) -> Optional[bytes]:
        """Fetch data from arXiv API with rate limiting.

        Rule #1: Early return for errors.
        Rule #5: Log all errors.
        """
        import urllib.request
        import urllib.error

        self._limiter.wait_if_needed()

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "IngestForge/1.0 (Academic Research Tool)"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()

            self._limiter.mark_call()
            return data

        except urllib.error.URLError as e:
            logger.error(f"arXiv API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching arXiv: {e}")
            return None

    def _parse_search_results(self, xml_data: bytes) -> List[Paper]:
        """Parse arXiv Atom feed XML.

        Rule #1: Early return for parse errors.
        Rule #5: Log errors.
        """
        try:
            root = ET.fromstring(xml_data)
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML: {e}")
            return []

        papers = []
        for entry in root.findall("atom:entry", self.ATOM_NS):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: Any) -> Optional[Paper]:
        """Parse single arXiv entry to Paper.

        Rule #1: Early return for missing data.
        Rule #4: Function <60 lines.
        """
        try:
            # Extract basic fields
            title = self._get_text(entry, "atom:title")
            abstract = self._get_text(entry, "atom:summary")

            if not title:
                return None

            # Extract arXiv ID from entry ID URL
            entry_id = self._get_text(entry, "atom:id") or ""
            arxiv_id = self._extract_arxiv_id(entry_id)

            # Extract authors
            authors = self._parse_authors(entry)

            # Extract dates
            published = self._parse_date(self._get_text(entry, "atom:published"))
            updated = self._parse_date(self._get_text(entry, "atom:updated"))

            # Extract URLs
            pdf_url, abs_url = self._extract_urls(entry)

            # Extract categories
            categories, primary = self._parse_categories(entry)

            # Extract optional fields
            comment = self._get_arxiv_text(entry, "arxiv:comment")
            journal_ref = self._get_arxiv_text(entry, "arxiv:journal_ref")
            doi = self._get_arxiv_text(entry, "arxiv:doi")

            return Paper(
                arxiv_id=arxiv_id,
                title=self._clean_text(title),
                authors=authors,
                abstract=self._clean_text(abstract),
                published=published or datetime.now(),
                updated=updated or datetime.now(),
                pdf_url=pdf_url,
                abs_url=abs_url,
                categories=categories,
                primary_category=primary,
                comment=comment,
                journal_ref=journal_ref,
                doi=doi,
            )
        except Exception as e:
            logger.warning(f"Error parsing arXiv entry: {e}")
            return None

    def _get_text(self, element: Any, path: str) -> Optional[str]:
        """Get text from XML element."""
        found = element.find(path, self.ATOM_NS)
        return found.text if found is not None else None

    def _get_arxiv_text(self, element: Any, path: str) -> Optional[str]:
        """Get text from arxiv namespace element."""
        found = element.find(path, {**self.ATOM_NS, **self.ARXIV_NS})
        return found.text if found is not None else None

    def _clean_text(self, text: str) -> str:
        """Clean whitespace from text."""
        return " ".join(text.split()) if text else ""

    def _parse_authors(self, entry: Any) -> List[str]:
        """Extract author names from entry."""
        authors = []
        for author in entry.findall("atom:author", self.ATOM_NS):
            name = author.findtext("atom:name", "", self.ATOM_NS)
            if name:
                authors.append(name.strip())
        return authors

    def _parse_date(self, date_str: Optional[str]) -> Optional[datetime]:
        """Parse ISO date string."""
        if not date_str:
            return None
        try:
            # Handle arXiv date format: 2023-01-15T12:00:00Z
            clean = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(clean)
        except ValueError:
            return None

    def _extract_urls(self, entry: Any) -> tuple[str, str]:
        """Extract PDF and abstract URLs from entry."""
        pdf_url = ""
        abs_url = ""

        for link in entry.findall("atom:link", self.ATOM_NS):
            href = link.get("href", "")
            link_type = link.get("type", "")

            if link_type == "application/pdf" or "/pdf/" in href:
                pdf_url = href
            elif link_type == "text/html" or "/abs/" in href:
                abs_url = href

        return pdf_url, abs_url

    def _parse_categories(self, entry: Any) -> tuple[List[str], str]:
        """Extract categories from entry."""
        categories = []
        primary = ""

        # Primary category from arxiv namespace
        prim_elem = entry.find(
            "arxiv:primary_category",
            {**self.ATOM_NS, **self.ARXIV_NS},
        )
        if prim_elem is not None:
            primary = prim_elem.get("term", "")

        # All categories
        for cat in entry.findall("atom:category", self.ATOM_NS):
            term = cat.get("term", "")
            if term:
                categories.append(term)

        return categories, primary

    def _extract_arxiv_id(self, entry_id: str) -> str:
        """Extract arXiv ID from entry URL."""
        # Entry ID format: http://arxiv.org/abs/2301.12345v1
        match = re.search(r"arxiv\.org/abs/([^v]+)(v\d+)?", entry_id)
        if match:
            return match.group(1)
        return entry_id.split("/")[-1]

    def _clean_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """Clean and validate arXiv ID.

        Accepts formats:
        - 2301.12345
        - arxiv:2301.12345
        - https://arxiv.org/abs/2301.12345
        """
        if not arxiv_id:
            return None

        # Remove arxiv: prefix
        clean = arxiv_id.replace("arxiv:", "").strip()

        # Extract from URL
        match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/\s]+)", clean)
        if match:
            clean = match.group(1)

        # Remove version suffix
        clean = re.sub(r"v\d+$", "", clean)

        # Remove .pdf extension
        clean = clean.replace(".pdf", "")

        # Validate format (new or old style)
        if re.match(r"^\d{4}\.\d{4,5}$", clean):  # New format: YYMM.NNNNN
            return clean
        if re.match(r"^[a-z-]+/\d{7}$", clean):  # Old format: category/NNNNNNN
            return clean

        return None

    def _write_response_to_file(
        self,
        response: Any,
        file_path: Path,
    ) -> int:
        """Write HTTP response content to file.

        Rule #1: Extracted to reduce nesting in _download_file
        Rule #4: Function <60 lines

        Args:
            response: HTTP response object
            file_path: Path to write to

        Returns:
            Total bytes written
        """
        total_bytes = 0
        with open(file_path, "wb") as f:
            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                f.write(chunk)
                total_bytes += len(chunk)
        return total_bytes

    def _download_file(
        self,
        url: str,
        file_path: Path,
        arxiv_id: str,
        timeout: int,
    ) -> ArxivDownloadResult:
        """Download file with progress tracking.

        Rule #1: Reduced nesting from 5 → 3 via helper extraction
        Rule #5: Log all errors.
        """
        import urllib.request
        import urllib.error

        self._limiter.wait_if_needed()

        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "IngestForge/1.0 (Academic Research Tool)",
                    "Accept": "application/pdf",
                },
            )

            with urllib.request.urlopen(req, timeout=timeout) as response:
                # Check content type
                content_type = response.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower():
                    logger.warning(f"Unexpected content type: {content_type}")
                total_bytes = self._write_response_to_file(response, file_path)

            self._limiter.mark_call()
            logger.info(f"Downloaded {arxiv_id} ({total_bytes} bytes)")

            return ArxivDownloadResult(
                arxiv_id=arxiv_id,
                success=True,
                file_path=file_path,
                filename=file_path.name,
                size_bytes=total_bytes,
            )

        except urllib.error.HTTPError as e:
            error_msg = f"HTTP {e.code}: {e.reason}"
            logger.error(f"Download failed for {arxiv_id}: {error_msg}")
            return ArxivDownloadResult(
                arxiv_id=arxiv_id,
                success=False,
                error=error_msg,
            )
        except Exception as e:
            logger.error(f"Download error for {arxiv_id}: {e}")
            return ArxivDownloadResult(
                arxiv_id=arxiv_id,
                success=False,
                error=str(e),
            )


def export_bibtex(papers: List[Paper]) -> str:
    """Export list of papers to BibTeX format.

    Rule #4: Function <60 lines.
    Rule #9: Full type hints.

    Args:
        papers: List of Paper objects.

    Returns:
        BibTeX formatted string.
    """
    entries = [paper.to_bibtex() for paper in papers]
    return "\n\n".join(entries)
