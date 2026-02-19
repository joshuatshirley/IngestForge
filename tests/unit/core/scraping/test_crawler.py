"""Tests for BFS crawler engine.

Tests crawling, depth limiting, and link extraction."""

from __future__ import annotations

import pytest

from ingestforge.core.scraping.crawler import (
    BFSCrawler,
    CrawlConfig,
    CrawlPage,
    CrawlPageStatus,
    CrawlResult,
    CrawlStatus,
    QueueItem,
    create_crawler,
    MAX_PAGES,
    MAX_DEPTH,
)

# CrawlStatus tests


class TestCrawlStatus:
    """Tests for CrawlStatus enum."""

    def test_statuses_defined(self) -> None:
        """Test all statuses are defined."""
        statuses = [s.value for s in CrawlStatus]

        assert "pending" in statuses
        assert "in_progress" in statuses
        assert "completed" in statuses
        assert "stopped" in statuses
        assert "error" in statuses


# CrawlPageStatus tests


class TestCrawlPageStatus:
    """Tests for CrawlPageStatus enum."""

    def test_statuses_defined(self) -> None:
        """Test all page statuses are defined."""
        statuses = [s.value for s in CrawlPageStatus]

        assert "pending" in statuses
        assert "success" in statuses
        assert "failed" in statuses
        assert "skipped" in statuses
        assert "blocked" in statuses


# CrawlConfig tests


class TestCrawlConfig:
    """Tests for CrawlConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = CrawlConfig()

        assert config.max_pages == MAX_PAGES
        assert config.max_depth == MAX_DEPTH
        assert config.follow_robots_txt is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CrawlConfig(
            max_pages=50,
            max_depth=3,
            delay_ms=1000,
        )

        assert config.max_pages == 50
        assert config.max_depth == 3
        assert config.delay_ms == 1000

    def test_config_bounds_enforced(self) -> None:
        """Test that bounds are enforced."""
        config = CrawlConfig(
            max_pages=1000,  # Above MAX_PAGES
            max_depth=20,  # Above MAX_DEPTH
        )

        assert config.max_pages == MAX_PAGES
        assert config.max_depth == MAX_DEPTH


# CrawlPage tests


class TestCrawlPage:
    """Tests for CrawlPage dataclass."""

    def test_page_creation(self) -> None:
        """Test creating a page."""
        page = CrawlPage(
            url="https://example.com/page",
            depth=1,
            parent_url="https://example.com",
        )

        assert page.url == "https://example.com/page"
        assert page.depth == 1
        assert page.status == CrawlPageStatus.PENDING

    def test_page_domain(self) -> None:
        """Test domain extraction."""
        page = CrawlPage(
            url="https://www.example.com/path",
            depth=0,
        )

        assert page.domain == "www.example.com"


# CrawlResult tests


class TestCrawlResult:
    """Tests for CrawlResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a result."""
        result = CrawlResult(
            start_url="https://example.com",
            status=CrawlStatus.COMPLETED,
            total_crawled=10,
            total_failed=2,
        )

        assert result.start_url == "https://example.com"
        assert result.status == CrawlStatus.COMPLETED

    def test_success_rate(self) -> None:
        """Test success rate calculation."""
        result = CrawlResult(
            start_url="https://example.com",
            status=CrawlStatus.COMPLETED,
            total_crawled=10,
            total_failed=2,
        )

        assert result.success_rate == 0.8

    def test_success_rate_zero_crawled(self) -> None:
        """Test success rate with zero crawled."""
        result = CrawlResult(
            start_url="https://example.com",
            status=CrawlStatus.ERROR,
            total_crawled=0,
        )

        assert result.success_rate == 0.0


# QueueItem tests


class TestQueueItem:
    """Tests for QueueItem dataclass."""

    def test_item_creation(self) -> None:
        """Test creating queue item."""
        item = QueueItem(
            url="https://example.com/page",
            depth=2,
            parent_url="https://example.com",
        )

        assert item.url == "https://example.com/page"
        assert item.depth == 2


# BFSCrawler tests


class TestBFSCrawler:
    """Tests for BFSCrawler."""

    def test_crawler_creation(self) -> None:
        """Test creating crawler."""
        crawler = BFSCrawler()

        assert crawler.config is not None

    def test_crawler_with_config(self) -> None:
        """Test crawler with custom config."""
        config = CrawlConfig(max_pages=50)
        crawler = BFSCrawler(config=config)

        assert crawler.config.max_pages == 50

    def test_crawler_reset(self) -> None:
        """Test crawler reset."""
        crawler = BFSCrawler()
        crawler._visited.add("https://example.com")
        crawler._status = CrawlStatus.COMPLETED

        crawler.reset()

        assert len(crawler._visited) == 0
        assert crawler._status == CrawlStatus.PENDING

    def test_crawler_stop(self) -> None:
        """Test crawler stop."""
        crawler = BFSCrawler()
        crawler.stop()

        assert crawler._stop_requested is True

    @pytest.mark.asyncio
    async def test_crawl_invalid_url(self) -> None:
        """Test crawling invalid URL."""
        crawler = BFSCrawler()

        result = await crawler.crawl("")

        assert result.status == CrawlStatus.ERROR
        assert "Invalid" in result.error

    @pytest.mark.asyncio
    async def test_crawl_valid_url(self) -> None:
        """Test crawling valid URL (no fetch function)."""
        crawler = BFSCrawler()

        result = await crawler.crawl("https://example.com")

        # Without fetch function, should still work
        assert result.status in (CrawlStatus.COMPLETED, CrawlStatus.ERROR)
        assert result.start_url == "https://example.com"


class TestLinkExtraction:
    """Tests for link extraction."""

    def test_extract_links_empty(self) -> None:
        """Test extracting from empty content."""
        crawler = BFSCrawler()

        links = crawler._extract_links("", "https://example.com")

        assert len(links) == 0

    def test_extract_links_absolute(self) -> None:
        """Test extracting absolute links."""
        crawler = BFSCrawler()
        content = '<a href="https://example.com/page1">Link</a>'

        links = crawler._extract_links(content, "https://example.com")

        assert "https://example.com/page1" in links

    def test_extract_links_relative(self) -> None:
        """Test extracting relative links."""
        crawler = BFSCrawler()
        content = '<a href="/page2">Link</a>'

        links = crawler._extract_links(content, "https://example.com")

        assert "https://example.com/page2" in links

    def test_extract_links_skip_anchors(self) -> None:
        """Test that anchors are skipped."""
        crawler = BFSCrawler()
        content = '<a href="#section">Anchor</a>'

        links = crawler._extract_links(content, "https://example.com")

        assert len(links) == 0

    def test_extract_links_skip_javascript(self) -> None:
        """Test that javascript links are skipped."""
        crawler = BFSCrawler()
        content = '<a href="javascript:void(0)">JS</a>'

        links = crawler._extract_links(content, "https://example.com")

        assert len(links) == 0


class TestDomainRestriction:
    """Tests for domain restriction."""

    def test_allowed_same_domain(self) -> None:
        """Test same domain is allowed."""
        crawler = BFSCrawler()
        crawler._start_domain = "example.com"

        assert crawler._is_allowed_domain("example.com") is True

    def test_blocked_different_domain(self) -> None:
        """Test different domain is blocked."""
        crawler = BFSCrawler()
        crawler._start_domain = "example.com"

        assert crawler._is_allowed_domain("other.com") is False

    def test_allowed_subdomain_when_enabled(self) -> None:
        """Test subdomain allowed when enabled."""
        config = CrawlConfig(include_subdomains=True)
        crawler = BFSCrawler(config=config)
        crawler._start_domain = "example.com"

        assert crawler._is_allowed_domain("www.example.com") is True
        assert crawler._is_allowed_domain("blog.example.com") is True

    def test_blocked_subdomain_when_disabled(self) -> None:
        """Test subdomain blocked when disabled."""
        config = CrawlConfig(include_subdomains=False)
        crawler = BFSCrawler(config=config)
        crawler._start_domain = "example.com"

        assert crawler._is_allowed_domain("www.example.com") is False


class TestURLValidation:
    """Tests for URL validation."""

    def test_valid_http_url(self) -> None:
        """Test valid HTTP URL."""
        crawler = BFSCrawler()

        assert crawler._is_valid_url("http://example.com") is True

    def test_valid_https_url(self) -> None:
        """Test valid HTTPS URL."""
        crawler = BFSCrawler()

        assert crawler._is_valid_url("https://example.com") is True

    def test_invalid_empty_url(self) -> None:
        """Test invalid empty URL."""
        crawler = BFSCrawler()

        assert crawler._is_valid_url("") is False

    def test_invalid_no_scheme(self) -> None:
        """Test invalid URL without scheme."""
        crawler = BFSCrawler()

        assert crawler._is_valid_url("example.com") is False

    def test_invalid_ftp_scheme(self) -> None:
        """Test invalid FTP scheme."""
        crawler = BFSCrawler()

        assert crawler._is_valid_url("ftp://example.com") is False


class TestURLNormalization:
    """Tests for URL normalization."""

    def test_normalize_removes_fragment(self) -> None:
        """Test that fragments are removed."""
        crawler = BFSCrawler()

        normalized = crawler._normalize_url("https://example.com/page#section")

        assert "#" not in normalized

    def test_normalize_removes_trailing_slash(self) -> None:
        """Test that trailing slash is removed."""
        crawler = BFSCrawler()

        normalized = crawler._normalize_url("https://example.com/path/")

        assert not normalized.endswith("/")

    def test_normalize_lowercase(self) -> None:
        """Test that URL is lowercased."""
        crawler = BFSCrawler()

        normalized = crawler._normalize_url("https://EXAMPLE.COM/Path")

        assert normalized == normalized.lower()


# Factory function tests


class TestCreateCrawler:
    """Tests for create_crawler factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        crawler = create_crawler()

        assert crawler.config.max_pages == MAX_PAGES
        assert crawler.config.max_depth == MAX_DEPTH

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        crawler = create_crawler(
            max_pages=50,
            max_depth=3,
            include_subdomains=True,
        )

        assert crawler.config.max_pages == 50
        assert crawler.config.max_depth == 3
        assert crawler.config.include_subdomains is True
