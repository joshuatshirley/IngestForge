"""Reference scraper for gathering literary context from the web.

Scrapes Wikipedia, SparkNotes, CliffsNotes, Shmoop, Fandom, and
Project Gutenberg pages about literary works and authors.
"""

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

try:
    import trafilatura
    from trafilatura import extract_metadata
except ImportError:
    trafilatura = None  # type: ignore
    extract_metadata = None  # type: ignore

from ingestforge.core.logging import get_logger
from ingestforge.ingest.anti_detection import (
    AntiDetectionConfig,
    AntiDetectionManager,
    DelayConfig,
)

logger = get_logger(__name__)


@dataclass
class ScrapedPage:
    """A single scraped web page."""

    url: str
    title: str
    text: str
    html: str
    category: str  # work, author, character, reference, etc.
    source: str  # wikipedia, fandom, sparknotes, etc.
    fetched_at: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class ScrapeManifest:
    """Manifest tracking a scraping session."""

    work_title: str
    author: str
    source: str
    depth: str
    pages: List[dict]
    started_at: str
    completed_at: str
    total_pages: int
    successful_pages: int
    failed_pages: int


# Source detection patterns: (domain substring, source name)
_SOURCE_PATTERNS = [
    ("wikipedia.org", "wikipedia"),
    ("fandom.com", "fandom"),
    ("sparknotes.com", "sparknotes"),
    ("cliffsnotes.com", "cliffsnotes"),
    ("shmoop.com", "shmoop"),
    ("gutenberg.org", "gutenberg"),
]


class ReferenceScraper:
    """Scrapes literary reference material from the web.

    Supports Wikipedia, SparkNotes, CliffsNotes, Shmoop, Fandom,
    and Project Gutenberg. Uses anti-detection measures and respects
    rate limits.
    """

    def __init__(
        self,
        min_delay: float = 1.0,
        max_delay: float = 3.0,
        max_pages: int = 10,
        respect_robots: bool = True,
    ):
        self.max_pages = max_pages
        self.respect_robots = respect_robots
        self._pages_fetched = 0

        delay_config = DelayConfig(
            min_delay=min_delay,
            max_delay=max_delay,
        )
        config = AntiDetectionConfig(delay_config=delay_config)
        self._anti_detection = AntiDetectionManager(config)

    def gather_wikipedia(
        self,
        work_title: str,
        author: Optional[str] = None,
        depth: str = "shallow",
    ) -> List[ScrapedPage]:
        """Gather Wikipedia pages about a literary work.

        Args:
            work_title: Title of the literary work.
            author: Author name (optional).
            depth: "shallow" (3 pages) or "deep" (10+ pages).

        Returns:
            List of ScrapedPage objects.
        """
        urls = self._build_wikipedia_urls(work_title, author, depth)
        pages = []

        for url, category in urls:
            if self._pages_fetched >= self.max_pages:
                break
            page = self._fetch_page(url, category, "wikipedia")
            pages.append(page)

        return pages

    def gather_from_urls(
        self,
        urls: List[str],
        tag: str = "reference",
    ) -> List[ScrapedPage]:
        """Scrape arbitrary URLs.

        Args:
            urls: List of URLs to scrape.
            tag: Category tag for all pages.

        Returns:
            List of ScrapedPage objects.
        """
        pages = []

        for url in urls:
            if self._pages_fetched >= self.max_pages:
                break
            source = self._detect_source(url)
            page = self._fetch_page(url, tag, source)
            pages.append(page)

        return pages

    def save_manifest(
        self,
        pages: List[ScrapedPage],
        work_title: str,
        author: Optional[str],
        source: str,
        depth: str,
        output_path: Path,
    ) -> Path:
        """Save a scraping manifest to disk.

        Args:
            pages: List of scraped pages.
            work_title: Title of the work.
            author: Author name.
            source: Primary source type.
            depth: Scrape depth.
            output_path: Directory to save manifest in.

        Returns:
            Path to the saved manifest file.
        """
        output_path.mkdir(parents=True, exist_ok=True)

        successful = [p for p in pages if p.success]
        failed = [p for p in pages if not p.success]

        pages_data = []
        for page in pages:
            pages_data.append(
                {
                    "url": page.url,
                    "title": page.title,
                    "category": page.category,
                    "source": page.source,
                    "fetched_at": page.fetched_at,
                    "success": page.success,
                    "error": page.error,
                    "text_length": len(page.text),
                }
            )

        manifest = ScrapeManifest(
            work_title=work_title,
            author=author or "",
            source=source,
            depth=depth,
            pages=pages_data,
            started_at=pages[0].fetched_at if pages else datetime.now().isoformat(),
            completed_at=pages[-1].fetched_at if pages else datetime.now().isoformat(),
            total_pages=len(pages),
            successful_pages=len(successful),
            failed_pages=len(failed),
        )

        manifest_path = output_path / "scrape_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)

        return manifest_path

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _build_wikipedia_urls(
        self,
        work_title: str,
        author: Optional[str],
        depth: str,
    ) -> List[Tuple[str, str]]:
        """Build Wikipedia URLs for a literary work.

        Args:
            work_title: Title of the work.
            author: Author name (optional).
            depth: "shallow" or "deep".

        Returns:
            List of (url, category) tuples.
        """
        base = "https://en.wikipedia.org/wiki/"
        title_slug = work_title.replace(" ", "_")

        urls = [
            (f"{base}{title_slug}", "work"),
            (f"{base}{title_slug}_(novel)", "work"),
        ]

        if author:
            author_slug = author.replace(" ", "_")
            urls.append((f"{base}{author_slug}", "author"))

        if depth == "deep":
            urls.append((f"{base}Characters_in_{title_slug}", "character"))
            urls.append((f"{base}{title_slug}_(adaptations)", "adaptations"))
            if author:
                author_slug = author.replace(" ", "_")
                urls.append((f"{base}{author_slug}_bibliography", "bibliography"))

        return urls

    def _detect_source(self, url: str) -> str:
        """Detect the source type from a URL.

        Args:
            url: The URL to classify.

        Returns:
            Source type string (wikipedia, sparknotes, etc.).
        """
        url_lower = url.lower()
        for pattern, source_name in _SOURCE_PATTERNS:
            if pattern in url_lower:
                return source_name
        return "web"

    def _fetch_page(
        self,
        url: str,
        category: str,
        source: str,
    ) -> ScrapedPage:
        """
        Fetch and extract content from a single URL.

        Rule #4: Reduced from 78 lines to <60 lines via helper extraction

        Args:
            url: URL to fetch.
            category: Page category.
            source: Source type.

        Returns:
            ScrapedPage with extracted content or error info.
        """
        self._pages_fetched += 1
        now = datetime.now().isoformat()

        try:
            html, title, text = self._fetch_and_extract_content(url)
            return self._build_success_page(
                url, title, text, html, category, source, now
            )

        except Exception as e:
            self._update_anti_detection_on_error(url)
            return self._build_error_page(url, category, source, now, str(e))

    def _fetch_and_extract_content(self, url: str) -> tuple:
        """
        Fetch and extract content from URL.

        Rule #4: Extracted to reduce function size

        Returns:
            (html, title, text) tuple

        Raises:
            Exception: If fetch fails or returns no content
        """
        # Anti-detection measures
        self._anti_detection.wait_for_request(url)
        request_info = self._anti_detection.prepare_request(url)

        # Fetch via trafilatura
        html = trafilatura.fetch_url(url)

        if html is None:
            self._anti_detection.update_after_request(url, success=False)
            raise ValueError("No content returned")

        # Extract text and title
        text = trafilatura.extract(html) or ""
        title = self._extract_title(html)

        self._anti_detection.update_after_request(url, success=True)

        return html, title, text

    def _build_success_page(
        self,
        url: str,
        title: str,
        text: str,
        html: str,
        category: str,
        source: str,
        fetched_at: str,
    ) -> ScrapedPage:
        """
        Build successful ScrapedPage.

        Rule #4: Extracted to reduce function size
        """
        return ScrapedPage(
            url=url,
            title=title,
            text=text,
            html=html,
            category=category,
            source=source,
            fetched_at=fetched_at,
            success=True,
        )

    def _build_error_page(
        self,
        url: str,
        category: str,
        source: str,
        fetched_at: str,
        error: str,
    ) -> ScrapedPage:
        """
        Build error ScrapedPage.

        Rule #4: Extracted to reduce function size
        """
        return ScrapedPage(
            url=url,
            title="",
            text="",
            html="",
            category=category,
            source=source,
            fetched_at=fetched_at,
            success=False,
            error=error,
        )

    def _update_anti_detection_on_error(self, url: str) -> None:
        """
        Update anti-detection state after error.

        Rule #4: Extracted to reduce function size
        """
        try:
            self._anti_detection.update_after_request(url, success=False)
        except Exception as err:
            logger.debug(f"Failed to update anti-detection state after error: {err}")

    def _extract_title(self, html: str) -> str:
        """Extract page title from HTML.

        Tries trafilatura metadata first, falls back to regex on <title> tag.

        Args:
            html: Raw HTML string.

        Returns:
            Extracted title, or empty string if not found.
        """
        # Try trafilatura metadata
        try:
            if extract_metadata:
                metadata = extract_metadata(html)
                if metadata and hasattr(metadata, "title") and metadata.title:
                    return metadata.title
        except Exception as e:
            logger.debug(f"Failed to extract title with trafilatura: {e}")

        # Fallback: regex on <title> tag
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = m.group(1).strip()
            # Remove common suffixes
            for suffix in [" - Wikipedia", " — Wikipedia", " | Wikipedia"]:
                if suffix in title:
                    title = title.split(suffix)[0].strip()
                    break
            return title

        return ""
