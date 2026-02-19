"""BFS Crawler Engine for recursive web scraping.

Implements breadth-first crawling with depth limits and
visited-set tracking to prevent infinite loops."""

from __future__ import annotations

import asyncio
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Deque, List, Optional, Set
from urllib.parse import urljoin, urlparse

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_PAGES = 100
MAX_DEPTH = 5
MAX_QUEUE_SIZE = 1000
MAX_CONCURRENT_REQUESTS = 5
DEFAULT_DELAY_MS = 500
MAX_URL_LENGTH = 2048


class CrawlStatus(str, Enum):
    """Status of a crawl operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class CrawlPageStatus(str, Enum):
    """Status of a single page fetch during crawling."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


@dataclass
class CrawlConfig:
    """Configuration for the crawler."""

    max_pages: int = MAX_PAGES
    max_depth: int = MAX_DEPTH
    delay_ms: int = DEFAULT_DELAY_MS
    concurrent_requests: int = MAX_CONCURRENT_REQUESTS
    follow_robots_txt: bool = True
    include_subdomains: bool = False
    url_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        self.max_pages = min(self.max_pages, MAX_PAGES)
        self.max_depth = min(self.max_depth, MAX_DEPTH)


@dataclass
class CrawlPage:
    """A page discovered during crawling."""

    url: str
    depth: int
    parent_url: str = ""
    status: CrawlPageStatus = CrawlPageStatus.PENDING
    content: str = ""
    title: str = ""
    links_found: int = 0
    error: str = ""
    crawled_at: Optional[datetime] = None

    @property
    def domain(self) -> str:
        """Get page domain."""
        parsed = urlparse(self.url)
        return parsed.netloc


@dataclass
class CrawlResult:
    """Result of a crawl operation."""

    start_url: str
    status: CrawlStatus
    pages: List[CrawlPage] = field(default_factory=list)
    total_discovered: int = 0
    total_crawled: int = 0
    total_failed: int = 0
    duration_seconds: float = 0.0
    error: str = ""

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_crawled == 0:
            return 0.0
        return (self.total_crawled - self.total_failed) / self.total_crawled


@dataclass
class QueueItem:
    """Item in the crawl queue."""

    url: str
    depth: int
    parent_url: str = ""


class BFSCrawler:
    """Breadth-first search crawler with depth limiting.

    Uses BFS to discover and crawl pages while respecting
    depth limits and domain restrictions.
    """

    def __init__(
        self,
        config: Optional[CrawlConfig] = None,
        fetch_func: Optional[Callable] = None,
    ) -> None:
        """Initialize crawler.

        Args:
            config: Crawler configuration
            fetch_func: Custom fetch function for pages
        """
        self.config = config or CrawlConfig()
        self._fetch_func = fetch_func
        self._queue: Deque[QueueItem] = deque()
        self._visited: Set[str] = set()
        self._pages: List[CrawlPage] = []
        self._status = CrawlStatus.PENDING
        self._stop_requested = False
        self._start_domain = ""

    def reset(self) -> None:
        """Reset crawler state for new crawl."""
        self._queue.clear()
        self._visited.clear()
        self._pages.clear()
        self._status = CrawlStatus.PENDING
        self._stop_requested = False
        self._start_domain = ""

    def stop(self) -> None:
        """Request crawler to stop."""
        self._stop_requested = True

    async def crawl(self, start_url: str) -> CrawlResult:
        """Start crawling from a URL.

        Args:
            start_url: URL to start crawling from

        Returns:
            CrawlResult with all discovered pages
        """
        self.reset()

        # Validate start URL
        if not self._is_valid_url(start_url):
            return CrawlResult(
                start_url=start_url,
                status=CrawlStatus.ERROR,
                error="Invalid start URL",
            )

        # Initialize
        self._start_domain = urlparse(start_url).netloc
        self._status = CrawlStatus.IN_PROGRESS
        start_time = datetime.now()

        # Add start URL to queue
        self._queue.append(QueueItem(url=start_url, depth=0))
        self._visited.add(self._normalize_url(start_url))

        # BFS loop
        await self._crawl_loop()

        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()

        # Determine final status
        if self._stop_requested:
            self._status = CrawlStatus.STOPPED
        elif len(self._pages) > 0:
            self._status = CrawlStatus.COMPLETED
        else:
            self._status = CrawlStatus.ERROR

        return CrawlResult(
            start_url=start_url,
            status=self._status,
            pages=self._pages,
            total_discovered=len(self._visited),
            total_crawled=len(self._pages),
            total_failed=sum(
                1 for p in self._pages if p.status == CrawlPageStatus.FAILED
            ),
            duration_seconds=duration,
        )

    async def _crawl_loop(self) -> None:
        """Main BFS crawl loop."""
        while self._queue and not self._stop_requested:
            # Check limits
            if len(self._pages) >= self.config.max_pages:
                logger.info(f"Reached max pages limit: {self.config.max_pages}")
                break

            # Get next URL
            item = self._queue.popleft()

            # Skip if too deep
            if item.depth > self.config.max_depth:
                continue

            # Crawl page
            page = await self._crawl_page(item)
            self._pages.append(page)

            # Extract and queue links
            if page.status == CrawlPageStatus.SUCCESS:
                await self._extract_and_queue_links(page, item.depth)

            # Delay between requests
            if self.config.delay_ms > 0:
                await asyncio.sleep(self.config.delay_ms / 1000)

    async def _crawl_page(self, item: QueueItem) -> CrawlPage:
        """Crawl a single page.

        Args:
            item: Queue item with URL info

        Returns:
            CrawlPage with results
        """
        page = CrawlPage(
            url=item.url,
            depth=item.depth,
            parent_url=item.parent_url,
        )

        try:
            if self._fetch_func:
                result = await self._fetch_func(item.url)
                page.content = result.get("content", "")
                page.title = result.get("title", "")
                page.status = CrawlPageStatus.SUCCESS
            else:
                # No fetch function, mark as success with no content
                page.status = CrawlPageStatus.SUCCESS

            page.crawled_at = datetime.now()

        except Exception as e:
            page.status = CrawlPageStatus.FAILED
            page.error = str(e)
            logger.exception(f"Failed to crawl {item.url}: {e}")

        return page

    async def _extract_and_queue_links(
        self, page: CrawlPage, current_depth: int
    ) -> None:
        """Extract links from page and add to queue.

        Args:
            page: Crawled page
            current_depth: Current depth level
        """
        links = self._extract_links(page.content, page.url)
        page.links_found = len(links)

        for link in links:
            # Check queue size limit
            if len(self._queue) >= MAX_QUEUE_SIZE:
                break

            # Normalize and check if visited
            normalized = self._normalize_url(link)
            if normalized in self._visited:
                continue

            # Validate link
            if not self._should_crawl(link):
                continue

            # Add to queue and visited
            self._visited.add(normalized)
            self._queue.append(
                QueueItem(
                    url=link,
                    depth=current_depth + 1,
                    parent_url=page.url,
                )
            )

    def _extract_links(self, content: str, base_url: str) -> List[str]:
        """Extract links from page content.

        Args:
            content: Page HTML content
            base_url: Base URL for relative links

        Returns:
            List of absolute URLs
        """
        if not content:
            return []

        links: List[str] = []

        # Simple href extraction
        href_pattern = re.compile(r'href=["\']([^"\']+)["\']', re.IGNORECASE)
        matches = href_pattern.findall(content)

        for href in matches:
            # Skip anchors, javascript, mailto
            if href.startswith(("#", "javascript:", "mailto:", "tel:")):
                continue

            # Convert to absolute URL
            absolute = urljoin(base_url, href)

            # Validate length
            if len(absolute) > MAX_URL_LENGTH:
                continue

            links.append(absolute)

        return links

    def _should_crawl(self, url: str) -> bool:
        """Check if URL should be crawled.

        Args:
            url: URL to check

        Returns:
            True if should crawl
        """
        if not self._is_valid_url(url):
            return False

        # Check domain
        parsed = urlparse(url)
        if not self._is_allowed_domain(parsed.netloc):
            return False

        # Check include patterns
        if self.config.url_patterns:
            if not any(re.search(p, url) for p in self.config.url_patterns):
                return False

        # Check exclude patterns
        if self.config.exclude_patterns:
            if any(re.search(p, url) for p in self.config.exclude_patterns):
                return False

        return True

    def _is_allowed_domain(self, domain: str) -> bool:
        """Check if domain is allowed.

        Args:
            domain: Domain to check

        Returns:
            True if allowed
        """
        if domain == self._start_domain:
            return True

        if self.config.include_subdomains:
            return domain.endswith(f".{self._start_domain}")

        return False

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid.

        Args:
            url: URL to validate

        Returns:
            True if valid
        """
        if not url:
            return False

        if len(url) > MAX_URL_LENGTH:
            return False

        try:
            parsed = urlparse(url)
            return parsed.scheme in ("http", "https") and bool(parsed.netloc)
        except Exception:
            logger.exception(f"Error parsing URL {url}: Treating as invalid")
            return False

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        parsed = urlparse(url)

        # Remove fragment
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

        # Remove trailing slash
        if normalized.endswith("/"):
            normalized = normalized[:-1]

        return normalized.lower()


def create_crawler(
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
    include_subdomains: bool = False,
) -> BFSCrawler:
    """Factory function to create crawler.

    Args:
        max_pages: Maximum pages to crawl
        max_depth: Maximum crawl depth
        include_subdomains: Include subdomains

    Returns:
        Configured BFSCrawler
    """
    config = CrawlConfig(
        max_pages=max_pages,
        max_depth=max_depth,
        include_subdomains=include_subdomains,
    )
    return BFSCrawler(config=config)


async def crawl_site(
    start_url: str,
    max_pages: int = MAX_PAGES,
    max_depth: int = MAX_DEPTH,
) -> CrawlResult:
    """Convenience function to crawl a site.

    Args:
        start_url: URL to start from
        max_pages: Maximum pages
        max_depth: Maximum depth

    Returns:
        CrawlResult
    """
    crawler = create_crawler(max_pages=max_pages, max_depth=max_depth)
    return await crawler.crawl(start_url)
