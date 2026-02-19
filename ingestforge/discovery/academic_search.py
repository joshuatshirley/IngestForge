"""Academic search - Find papers via arXiv and Semantic Scholar APIs."""

import xml.etree.ElementTree as ET
import time
from pathlib import Path
from typing import Any, List, Optional
from dataclasses import dataclass
from urllib.parse import quote_plus

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Rate limiting: minimum seconds between API calls
_RATE_LIMIT_DELAY = 1.0


class _RateLimiter:
    """Rate limiter for API calls.

    Rule #6: Encapsulates rate limiting state at smallest scope.
    """

    def __init__(self, delay: float = _RATE_LIMIT_DELAY) -> None:
        self.delay = delay
        self.last_call = 0.0

    def wait_if_needed(self) -> None:
        """Wait if needed to respect rate limit."""
        time_since_last = time.time() - self.last_call
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)

    def mark_call(self) -> None:
        """Mark that a call was made."""
        self.last_call = time.time()


_arxiv_limiter = _RateLimiter()
_scholar_limiter = _RateLimiter()


@dataclass
class AcademicSource:
    """A source found via academic search."""

    title: str
    authors: List[str]
    abstract: str
    url: str
    year: Optional[str]
    citation_count: Optional[int]
    source_api: str  # "arxiv" or "semantic_scholar"


def search_arxiv(
    query: str,
    max_results: int = 10,
) -> List[AcademicSource]:
    """
    Search arXiv for academic papers with rate limiting.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of AcademicSource from arXiv

    Note:
        Implements 1-second rate limiting between calls
    """
    import urllib.request

    _arxiv_limiter.wait_if_needed()

    encoded_query = quote_plus(query)
    url = (
        f"http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}"
        f"&start=0&max_results={max_results}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IngestForge/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            xml_data = response.read()

        _arxiv_limiter.mark_call()
        return _parse_arxiv_response(xml_data)

    except Exception as e:
        logger.warning(f"arXiv search failed: {e}")
        return []


def _parse_arxiv_response(xml_data: bytes) -> List[AcademicSource]:
    """
    Parse arXiv Atom XML response.

    Rule #1: Reduced nesting with helper methods
    """
    try:
        root = ET.fromstring(xml_data)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        results = [
            _parse_arxiv_entry(entry, ns) for entry in root.findall("atom:entry", ns)
        ]
        return results

    except ET.ParseError as e:
        logger.warning(f"Failed to parse arXiv response: {e}")
        return []


def _parse_arxiv_entry(entry: Any, ns: dict[str, str]) -> AcademicSource:
    """
    Parse single arXiv entry.

    Rule #1: Extracted to reduce nesting
    Rule #4: Function <60 lines
    """
    title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
    abstract = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
    authors = _parse_arxiv_authors(entry, ns)
    url = _extract_arxiv_url(entry, ns)

    # Extract year from published date
    published = entry.findtext("atom:published", "", ns)
    year = published[:4] if published else None

    return AcademicSource(
        title=title,
        authors=authors,
        abstract=abstract[:500],
        url=url,
        year=year,
        citation_count=None,
        source_api="arxiv",
    )


def _parse_arxiv_authors(entry: Any, ns: dict[str, str]) -> List[str]:
    """
    Parse authors from arXiv entry.

    Rule #1: Extracted to reduce nesting
    Rule #4: Function <60 lines
    """
    authors = []
    for author in entry.findall("atom:author", ns):
        name = author.findtext("atom:name", "", ns)
        if name:
            authors.append(name)
    return authors


def _extract_arxiv_url(entry: Any, ns: dict[str, str]) -> str:
    """
    Extract URL from arXiv entry.

    Rule #1: Extracted to reduce nesting
    Rule #4: Function <60 lines
    """
    # Try to find HTML link first
    for link in entry.findall("atom:link", ns):
        if link.get("type") == "text/html":
            return link.get("href", "")

    # Fallback to entry ID
    return entry.findtext("atom:id", "", ns) or ""


def search_semantic_scholar(
    query: str,
    max_results: int = 10,
) -> List[AcademicSource]:
    """
    Search Semantic Scholar for academic papers with rate limiting.

    Args:
        query: Search query
        max_results: Maximum results to return

    Returns:
        List of AcademicSource from Semantic Scholar

    Note:
        Implements 1-second rate limiting between calls
    """
    import urllib.request
    import json

    _scholar_limiter.wait_if_needed()

    encoded_query = quote_plus(query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={encoded_query}"
        f"&limit={max_results}"
        f"&fields=title,authors,abstract,url,year,citationCount"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "IngestForge/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = json.loads(response.read())

        _scholar_limiter.mark_call()
        return _parse_semantic_scholar_response(data)

    except Exception as e:
        logger.warning(f"Semantic Scholar search failed: {e}")
        return []


def _parse_semantic_scholar_response(data: dict[str, Any]) -> List[AcademicSource]:
    """Parse Semantic Scholar JSON response."""
    results = []

    for paper in data.get("data", []):
        title = paper.get("title", "")
        authors = [a.get("name", "") for a in paper.get("authors", []) if a.get("name")]
        abstract = (paper.get("abstract") or "")[:500]
        url = paper.get("url", "")
        year = str(paper.get("year")) if paper.get("year") else None
        citations = paper.get("citationCount")

        results.append(
            AcademicSource(
                title=title,
                authors=authors,
                abstract=abstract,
                url=url,
                year=year,
                citation_count=citations,
                source_api="semantic_scholar",
            )
        )

    return results


def search_academic(
    query: str,
    max_results: int = 10,
) -> List[AcademicSource]:
    """
    Search both arXiv and Semantic Scholar.

    Args:
        query: Search query
        max_results: Maximum results per source

    Returns:
        Combined list of AcademicSource results
    """
    results = []

    # Try both APIs
    arxiv_results = search_arxiv(query, max_results=max_results)
    results.extend(arxiv_results)

    scholar_results = search_semantic_scholar(query, max_results=max_results)
    results.extend(scholar_results)

    # Sort by citation count (papers with citations first)
    results.sort(
        key=lambda x: (x.citation_count or 0),
        reverse=True,
    )

    return results


def get_arxiv_pdf_url(arxiv_url: str) -> Optional[str]:
    """
    Convert arXiv abstract URL to PDF URL.

    Args:
        arxiv_url: arXiv abstract URL (e.g., https://arxiv.org/abs/2301.00001)

    Returns:
        PDF URL or None if not an arXiv URL

    Example:
        >>> get_arxiv_pdf_url("https://arxiv.org/abs/2301.00001")
        'https://arxiv.org/pdf/2301.00001.pdf'
    """
    if "arxiv.org" not in arxiv_url:
        return None

    # If already a PDF URL, return as-is (idempotent)
    if "/pdf/" in arxiv_url and arxiv_url.endswith(".pdf"):
        return arxiv_url

    # Convert /abs/ to /pdf/ and add .pdf extension
    if "/abs/" in arxiv_url:
        pdf_url = arxiv_url.replace("/abs/", "/pdf/")
        if not pdf_url.endswith(".pdf"):
            pdf_url += ".pdf"
        return pdf_url

    return None


def download_arxiv_pdf(
    paper: AcademicSource,
    output_dir: Path,
) -> Optional[Path]:
    """
    Download PDF for an arXiv paper.

    Args:
        paper: Academic source from arXiv
        output_dir: Directory to save PDF

    Returns:
        Path to downloaded PDF or None if failed

    Note:
        Uses pdf_downloader module for actual download
    """
    if paper.source_api != "arxiv":
        logger.warning(f"Not an arXiv paper: {paper.source_api}")
        return None

    pdf_url = get_arxiv_pdf_url(paper.url)
    if not pdf_url:
        logger.warning(f"Could not determine PDF URL: {paper.url}")
        return None

    try:
        from ingestforge.ingest.pdf_downloader import download_pdf

        result = download_pdf(
            url=pdf_url,
            output_dir=output_dir,
            filename=None,  # Auto-generate from URL
            max_size_mb=50,  # Reasonable limit for papers
        )

        if result.success:
            logger.info(f"Downloaded: {result.filename}")
            return result.file_path

        logger.warning(f"Download failed: {result.error}")
        return None

    except Exception as e:
        logger.warning(f"PDF download error: {e}")
        return None


def _format_authors_string(authors: List[str]) -> str:
    """
    Format author list with et al. for long lists.

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        authors: List of author names

    Returns:
        Formatted author string
    """
    authors_str = ", ".join(authors[:3])
    if len(authors) > 3:
        authors_str += " et al."
    return authors_str


def _add_pdf_link_if_arxiv(source: AcademicSource, lines: List[str]) -> None:
    """
    Add PDF link for arXiv papers.

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        source: Academic source to check
        lines: List to append PDF link to
    """
    if source.source_api != "arxiv":
        return

    pdf_url = get_arxiv_pdf_url(source.url)
    if pdf_url:
        lines.append(f"PDF: {pdf_url}")


def _format_single_result(source: AcademicSource, index: int, lines: List[str]) -> None:
    """
    Format a single academic search result.

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        source: Academic source to format
        index: Result number (1-based)
        lines: List to append formatted lines to
    """
    assert source is not None, "Source cannot be None"
    assert lines is not None, "Lines list cannot be None"
    assert index > 0, "Index must be positive"

    # Format authors
    authors_str = _format_authors_string(source.authors)

    # Add title and authors
    lines.append(f"## {index}. {source.title}")
    lines.append(f"*{authors_str}*")

    # Add optional metadata
    if source.year:
        lines.append(f"Year: {source.year}")
    if source.citation_count:
        lines.append(f"Citations: {source.citation_count}")

    lines.append(f"Source: {source.source_api}")

    if source.abstract:
        lines.append(f"\n{source.abstract[:200]}...")

    # Add URL and PDF link
    if source.url:
        lines.append(f"\n{source.url}")
        _add_pdf_link_if_arxiv(source, lines)

    lines.append("")


def format_academic_results(results: List[AcademicSource]) -> str:
    """
    Format academic search results as markdown.

    Rule #1: Zero nesting - all logic extracted to helpers
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        results: List of academic sources to format

    Returns:
        Formatted markdown string
    """
    if not results:
        return "No academic papers found."

    lines = ["# Academic Sources Found", ""]
    for i, source in enumerate(results, 1):
        _format_single_result(source, i, lines)

    return "\n".join(lines)
