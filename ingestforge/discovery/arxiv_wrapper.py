"""arXiv Discovery Wrapper - Search and retrieve papers using the arxiv library.

This module provides the ArxivDiscovery class for searching arXiv papers
using the official `arxiv` Python library. It offers a cleaner API than
raw HTTP requests and handles rate limiting automatically.

Features:
- Search papers by topic/keyword
- Get individual paper metadata by arXiv ID
- Download PDFs to local directory
- Full metadata extraction (authors, abstract, categories, DOI)"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ArxivPaper:
    """Metadata for an arXiv paper.

    Rule #9: Full type hints on all fields.
    """

    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: datetime
    updated_date: datetime
    categories: List[str]
    pdf_url: str
    doi: Optional[str] = None
    primary_category: Optional[str] = None
    comment: Optional[str] = None
    journal_ref: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert paper to dictionary representation.

        Rule #4: Function <60 lines.
        """
        return {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "published_date": self.published_date.isoformat(),
            "updated_date": self.updated_date.isoformat(),
            "categories": self.categories,
            "pdf_url": self.pdf_url,
            "doi": self.doi,
            "primary_category": self.primary_category,
            "comment": self.comment,
            "journal_ref": self.journal_ref,
        }


def _check_arxiv_library() -> bool:
    """Check if arxiv library is available.

    Rule #4: Small helper function.
    """
    try:
        import arxiv

        return True
    except ImportError:
        return False


def _convert_result_to_paper(result) -> ArxivPaper:
    """Convert arxiv.Result to ArxivPaper.

    Rule #1: Early return pattern not needed - simple mapping.
    Rule #4: Function <60 lines.

    Args:
        result: arxiv.Result object from the library.

    Returns:
        ArxivPaper with extracted metadata.
    """
    # Extract arXiv ID from entry_id URL
    arxiv_id = result.entry_id.split("/")[-1]
    # Remove version suffix for consistent IDs
    if "v" in arxiv_id:
        arxiv_id = arxiv_id.rsplit("v", 1)[0]

    return ArxivPaper(
        arxiv_id=arxiv_id,
        title=result.title,
        authors=[author.name for author in result.authors],
        abstract=result.summary,
        published_date=result.published,
        updated_date=result.updated,
        categories=list(result.categories),
        pdf_url=result.pdf_url or f"https://arxiv.org/pdf/{arxiv_id}.pdf",
        doi=result.doi,
        primary_category=result.primary_category,
        comment=result.comment,
        journal_ref=result.journal_ref,
    )


class ArxivDiscovery:
    """arXiv discovery client using the official arxiv Python library.

    Provides full-featured access to arXiv with:
    - Paper search with keyword queries
    - Individual paper lookup by ID
    - PDF download with proper naming
    - Automatic rate limiting (handled by library)

    Example:
        discovery = ArxivDiscovery()
        papers = discovery.search("quantum computing", max_results=5)
        for paper in papers:
            print(f"{paper.title} by {paper.authors[0]}")

    Note:
        Requires the `arxiv` package: pip install arxiv
    """

    def __init__(self) -> None:
        """Initialize ArxivDiscovery client.

        Rule #5: Fail explicitly if library unavailable.
        """
        if not _check_arxiv_library():
            raise ImportError(
                "arxiv library not installed. " "Install with: pip install arxiv"
            )

        import arxiv

        self._arxiv = arxiv
        self._client = arxiv.Client()

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[ArxivPaper]:
        """Search arXiv for papers matching query.

        Rule #1: Early return for empty query.
        Rule #4: Function <60 lines.
        Rule #7: Parameter validation.

        Args:
            query: Search query string (keywords, title, authors).
            max_results: Maximum number of results (default 5, max 300).

        Returns:
            List of ArxivPaper objects matching the query.
        """
        if not query or not query.strip():
            logger.warning("Empty search query provided")
            return []

        # Clamp max_results to reasonable bounds
        max_results = min(max(1, max_results), 300)

        try:
            search = self._arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=self._arxiv.SortCriterion.Relevance,
            )

            papers = []
            for result in self._client.results(search):
                paper = _convert_result_to_paper(result)
                papers.append(paper)

            logger.info(f"Found {len(papers)} papers for query: {query}")
            return papers

        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    def get_paper(self, arxiv_id: str) -> Optional[ArxivPaper]:
        """Get metadata for a specific arXiv paper.

        Rule #1: Early return for invalid ID.
        Rule #4: Function <60 lines.

        Args:
            arxiv_id: arXiv paper ID (e.g., "2301.12345").

        Returns:
            ArxivPaper object or None if not found.
        """
        if not arxiv_id or not arxiv_id.strip():
            logger.warning("Empty arXiv ID provided")
            return None

        # Clean the ID (remove arxiv: prefix, version suffix)
        clean_id = self._clean_arxiv_id(arxiv_id)
        if not clean_id:
            logger.warning(f"Invalid arXiv ID format: {arxiv_id}")
            return None

        try:
            search = self._arxiv.Search(id_list=[clean_id])
            results = list(self._client.results(search))

            if not results:
                logger.warning(f"Paper not found: {arxiv_id}")
                return None

            return _convert_result_to_paper(results[0])

        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
            return None

    def download_pdf(
        self,
        paper: ArxivPaper,
        output_dir: Path,
    ) -> Optional[Path]:
        """Download PDF for an arXiv paper.

        Rule #1: Early return for invalid parameters.
        Rule #4: Function <60 lines.
        Rule #5: Log all errors.

        Args:
            paper: ArxivPaper object with arxiv_id.
            output_dir: Directory to save the PDF.

        Returns:
            Path to downloaded PDF, or None on failure.
        """
        if not paper or not paper.arxiv_id:
            logger.error("Invalid paper object provided")
            return None

        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate safe filename
        safe_id = paper.arxiv_id.replace("/", "_").replace(":", "_")
        filename = f"{safe_id}.pdf"
        output_path = output_dir / filename

        try:
            # Use arxiv library's download method
            search = self._arxiv.Search(id_list=[paper.arxiv_id])
            results = list(self._client.results(search))

            if not results:
                logger.error(f"Paper not found for download: {paper.arxiv_id}")
                return None

            # Download to output directory
            downloaded_path = results[0].download_pdf(
                dirpath=str(output_dir),
                filename=filename,
            )

            logger.info(f"Downloaded {paper.arxiv_id} to {downloaded_path}")
            return Path(downloaded_path)

        except Exception as e:
            logger.error(f"Failed to download PDF for {paper.arxiv_id}: {e}")
            return None

    def _clean_arxiv_id(self, arxiv_id: str) -> Optional[str]:
        """Clean and validate arXiv ID.

        Rule #4: Function <60 lines.

        Accepts formats:
        - 2301.12345
        - arxiv:2301.12345
        - https://arxiv.org/abs/2301.12345

        Args:
            arxiv_id: Raw arXiv ID in various formats.

        Returns:
            Cleaned arXiv ID or None if invalid.
        """
        import re

        clean = arxiv_id.strip()

        # Remove arxiv: prefix
        if clean.lower().startswith("arxiv:"):
            clean = clean[6:]

        # Extract from URL
        url_match = re.search(r"arxiv\.org/(?:abs|pdf)/([^/\s]+)", clean)
        if url_match:
            clean = url_match.group(1)

        # Remove version suffix (vN)
        clean = re.sub(r"v\d+$", "", clean)

        # Remove .pdf extension
        clean = clean.replace(".pdf", "")

        # Validate format
        if re.match(r"^\d{4}\.\d{4,5}$", clean):
            return clean
        if re.match(r"^[a-z-]+/\d{7}$", clean):
            return clean

        return None


def create_arxiv_discovery() -> ArxivDiscovery:
    """Factory function to create ArxivDiscovery instance.

    Rule #4: Simple factory function.

    Returns:
        ArxivDiscovery instance.

    Raises:
        ImportError: If arxiv library is not installed.
    """
    return ArxivDiscovery()
