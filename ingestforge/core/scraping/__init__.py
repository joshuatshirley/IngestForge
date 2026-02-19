"""Web scraping utilities with dynamic content support.

This package provides scraping functionality:
- browser: Playwright-based headless browser (SCRAPE-002.1)
- wait_strategy: Async selector waiting (SCRAPE-002.2)
- crawler: BFS recursive crawler (SCRAPE-003.1)
- guard: Domain security guard (SCRAPE-003.2)
"""

from ingestforge.core.scraping.browser import (
    BrowserConfig,
    BrowserContext,
    BrowserManager,
    PageResult,
    create_browser_manager,
    fetch_with_js,
)
from ingestforge.core.scraping.wait_strategy import (
    WaitCondition,
    WaitConfig,
    WaitResult,
    WaitStrategy,
    wait_for_content,
    wait_for_selector,
)

from ingestforge.core.scraping.crawler import (
    BFSCrawler,
    CrawlConfig,
    CrawlPage,
    CrawlPageStatus,
    CrawlResult,
    CrawlStatus,
    QueueItem,
    crawl_site,
    create_crawler,
)

from ingestforge.core.scraping.guard import (
    DomainGuard,
    GuardConfig,
    RobotsRules,
    ValidationReport,
    ValidationResult,
    create_guard,
    validate_url,
)

__all__ = [
    # Browser (SCRAPE-002.1)
    "BrowserConfig",
    "BrowserContext",
    "BrowserManager",
    "PageResult",
    "create_browser_manager",
    "fetch_with_js",
    # Wait strategy (SCRAPE-002.2)
    "WaitCondition",
    "WaitConfig",
    "WaitResult",
    "WaitStrategy",
    "wait_for_content",
    "wait_for_selector",
    # Crawler (SCRAPE-003.1)
    "BFSCrawler",
    "CrawlConfig",
    "CrawlPage",
    "CrawlPageStatus",
    "CrawlResult",
    "CrawlStatus",
    "QueueItem",
    "crawl_site",
    "create_crawler",
    # Domain guard (SCRAPE-003.2)
    "DomainGuard",
    "GuardConfig",
    "RobotsRules",
    "ValidationReport",
    "ValidationResult",
    "create_guard",
    "validate_url",
]
