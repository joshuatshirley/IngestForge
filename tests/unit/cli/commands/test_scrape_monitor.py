"""Tests for crawler TUI monitor.

Tests monitor display and statistics tracking."""

from __future__ import annotations

from datetime import datetime


from ingestforge.core.scraping.crawler import (
    CrawlPage,
    CrawlPageStatus,
    CrawlResult,
    CrawlStatus,
)
from ingestforge.cli.commands.scrape_monitor import (
    CrawlMonitor,
    MonitorConfig,
    MonitorStats,
    create_monitor,
    create_pages_table,
    MAX_DISPLAY_PAGES,
)

# MonitorStats tests


class TestMonitorStats:
    """Tests for MonitorStats dataclass."""

    def test_stats_defaults(self) -> None:
        """Test default stats values."""
        stats = MonitorStats()

        assert stats.pages_crawled == 0
        assert stats.pages_pending == 0
        assert stats.pages_failed == 0

    def test_elapsed_seconds_no_start(self) -> None:
        """Test elapsed with no start time."""
        stats = MonitorStats()

        assert stats.elapsed_seconds == 0.0

    def test_elapsed_seconds_with_start(self) -> None:
        """Test elapsed with start time."""
        stats = MonitorStats(start_time=datetime.now())

        # Should be very small but > 0
        assert stats.elapsed_seconds >= 0.0

    def test_pages_per_second_zero(self) -> None:
        """Test pages per second with zero elapsed."""
        stats = MonitorStats()

        assert stats.pages_per_second == 0.0

    def test_pages_per_second_calculation(self) -> None:
        """Test pages per second calculation."""
        stats = MonitorStats(
            pages_crawled=10,
            start_time=datetime.now(),
        )

        # Rate should be calculable
        rate = stats.pages_per_second
        assert isinstance(rate, float)


# MonitorConfig tests


class TestMonitorConfig:
    """Tests for MonitorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = MonitorConfig()

        assert config.show_urls is True
        assert config.show_stats is True
        assert config.show_progress is True
        assert config.max_display_pages == MAX_DISPLAY_PAGES

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = MonitorConfig(
            show_urls=False,
            compact_mode=True,
        )

        assert config.show_urls is False
        assert config.compact_mode is True


# CrawlMonitor tests


class TestCrawlMonitor:
    """Tests for CrawlMonitor."""

    def test_monitor_creation(self) -> None:
        """Test creating monitor."""
        monitor = CrawlMonitor()

        assert monitor.config is not None
        assert monitor.console is not None

    def test_monitor_with_config(self) -> None:
        """Test monitor with custom config."""
        config = MonitorConfig(show_urls=False)
        monitor = CrawlMonitor(config=config)

        assert monitor.config.show_urls is False

    def test_monitor_start(self) -> None:
        """Test starting monitor."""
        monitor = CrawlMonitor()
        monitor.start()

        assert monitor._is_running is True
        assert monitor.stats.start_time is not None

    def test_monitor_stop(self) -> None:
        """Test stopping monitor."""
        monitor = CrawlMonitor()
        monitor.start()
        monitor.stop()

        assert monitor._is_running is False


class TestMonitorUpdate:
    """Tests for monitor updates."""

    def test_update_with_page(self) -> None:
        """Test updating with a page."""
        monitor = CrawlMonitor()
        monitor.start()

        page = CrawlPage(
            url="https://example.com",
            depth=0,
            status=CrawlPageStatus.SUCCESS,
            links_found=5,
        )

        monitor.update(page=page, pending=10)

        assert monitor.stats.pages_crawled == 1
        assert monitor.stats.links_discovered == 5
        assert monitor.stats.pages_pending == 10

    def test_update_failed_page(self) -> None:
        """Test updating with failed page."""
        monitor = CrawlMonitor()
        monitor.start()

        page = CrawlPage(
            url="https://example.com",
            depth=0,
            status=CrawlPageStatus.FAILED,
            error="Connection error",
        )

        monitor.update(page=page)

        assert monitor.stats.pages_failed == 1

    def test_recent_pages_limited(self) -> None:
        """Test that recent pages are limited."""
        config = MonitorConfig(max_display_pages=5)
        monitor = CrawlMonitor(config=config)
        monitor.start()

        # Add more pages than limit
        for i in range(10):
            page = CrawlPage(
                url=f"https://example.com/page{i}",
                depth=0,
                status=CrawlPageStatus.SUCCESS,
            )
            monitor.update(page=page)

        assert len(monitor._recent_pages) == 5


class TestMonitorRender:
    """Tests for monitor rendering."""

    def test_render_returns_panel(self) -> None:
        """Test that render returns a panel."""
        monitor = CrawlMonitor()
        monitor.start()

        panel = monitor.render()

        # Should return a Rich Panel
        assert panel is not None

    def test_render_stats(self) -> None:
        """Test rendering stats section."""
        monitor = CrawlMonitor()
        monitor.start()
        monitor.stats.pages_crawled = 5

        stats = monitor._render_stats()

        assert "5" in stats
        assert "Pages Crawled" in stats

    def test_render_progress(self) -> None:
        """Test rendering progress section."""
        monitor = CrawlMonitor()
        monitor.start()
        monitor.stats.current_url = "https://example.com/page"

        progress = monitor._render_progress()

        assert "example.com" in progress

    def test_render_pages_empty(self) -> None:
        """Test rendering empty pages list."""
        monitor = CrawlMonitor()
        monitor.start()

        pages = monitor._render_pages()

        assert "No pages crawled" in pages


class TestSummaryPrinting:
    """Tests for summary printing."""

    def test_print_summary(self, capsys) -> None:
        """Test printing summary."""
        monitor = CrawlMonitor()

        result = CrawlResult(
            start_url="https://example.com",
            status=CrawlStatus.COMPLETED,
            total_crawled=10,
            total_failed=2,
            duration_seconds=5.5,
        )

        monitor.print_summary(result)

        # Should not raise any errors
        assert True

    def test_print_summary_with_error(self, capsys) -> None:
        """Test printing summary with error."""
        monitor = CrawlMonitor()

        result = CrawlResult(
            start_url="https://example.com",
            status=CrawlStatus.ERROR,
            error="Connection failed",
        )

        monitor.print_summary(result)

        # Should not raise any errors
        assert True


# Factory function tests


class TestCreateMonitor:
    """Tests for create_monitor factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        monitor = create_monitor()

        assert monitor.config.show_urls is True

    def test_create_compact(self) -> None:
        """Test creating compact monitor."""
        monitor = create_monitor(compact=True)

        assert monitor.config.compact_mode is True


class TestCreatePagesTable:
    """Tests for create_pages_table function."""

    def test_table_creation(self) -> None:
        """Test creating pages table."""
        pages = [
            CrawlPage(
                url="https://example.com",
                depth=0,
                status=CrawlPageStatus.SUCCESS,
                links_found=5,
            ),
            CrawlPage(
                url="https://example.com/page",
                depth=1,
                status=CrawlPageStatus.FAILED,
                error="Error",
            ),
        ]

        table = create_pages_table(pages)

        assert table is not None
        assert table.title == "Crawled Pages"

    def test_table_empty_pages(self) -> None:
        """Test table with no pages."""
        table = create_pages_table([])

        assert table is not None

    def test_table_truncates_url(self) -> None:
        """Test that long URLs are truncated."""
        long_url = "https://example.com/" + "a" * 100
        pages = [
            CrawlPage(
                url=long_url,
                depth=0,
                status=CrawlPageStatus.SUCCESS,
            ),
        ]

        table = create_pages_table(pages)

        # Should not raise any errors
        assert table is not None
