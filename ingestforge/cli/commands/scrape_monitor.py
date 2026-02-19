"""Crawler TUI Monitor for real-time progress visualization.

Provides Rich-based terminal UI for monitoring crawler progress,
displaying discovered pages and crawl statistics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from ingestforge.core.scraping.crawler import (
    CrawlPage,
    CrawlPageStatus,
    CrawlResult,
    CrawlStatus,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_DISPLAY_PAGES = 20
MAX_URL_DISPLAY_LENGTH = 60
REFRESH_RATE_HZ = 4


@dataclass
class MonitorStats:
    """Statistics for the monitor display."""

    pages_crawled: int = 0
    pages_pending: int = 0
    pages_failed: int = 0
    links_discovered: int = 0
    current_depth: int = 0
    start_time: Optional[datetime] = None
    current_url: str = ""

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def pages_per_second(self) -> float:
        """Calculate crawl rate."""
        elapsed = self.elapsed_seconds
        if elapsed == 0:
            return 0.0
        return self.pages_crawled / elapsed


@dataclass
class MonitorConfig:
    """Configuration for the monitor."""

    show_urls: bool = True
    show_stats: bool = True
    show_progress: bool = True
    max_display_pages: int = MAX_DISPLAY_PAGES
    compact_mode: bool = False


class CrawlMonitor:
    """Real-time TUI monitor for crawler progress.

    Uses Rich library to display live crawl statistics,
    discovered pages, and progress information.
    """

    def __init__(self, config: Optional[MonitorConfig] = None) -> None:
        """Initialize monitor.

        Args:
            config: Monitor configuration
        """
        self.config = config or MonitorConfig()
        self.console = Console()
        self.stats = MonitorStats()
        self._recent_pages: List[CrawlPage] = []
        self._is_running = False

    def start(self) -> None:
        """Start the monitor session."""
        self.stats = MonitorStats(start_time=datetime.now())
        self._recent_pages = []
        self._is_running = True

    def stop(self) -> None:
        """Stop the monitor session."""
        self._is_running = False

    def update(
        self,
        page: Optional[CrawlPage] = None,
        pending: int = 0,
        current_depth: int = 0,
    ) -> None:
        """Update monitor with new data.

        Args:
            page: Newly crawled page
            pending: Number of pending pages
            current_depth: Current crawl depth
        """
        self.stats.pages_pending = pending
        self.stats.current_depth = current_depth

        if page:
            self._recent_pages.append(page)
            self._recent_pages = self._recent_pages[-self.config.max_display_pages :]

            if page.status == CrawlPageStatus.SUCCESS:
                self.stats.pages_crawled += 1
                self.stats.links_discovered += page.links_found
            elif page.status == CrawlPageStatus.FAILED:
                self.stats.pages_failed += 1

            self.stats.current_url = page.url

    def render(self) -> Panel:
        """Render the monitor display.

        Returns:
            Rich Panel with monitor content
        """
        layout = Layout()

        # Build sections
        sections = []

        if self.config.show_stats:
            sections.append(self._render_stats())

        if self.config.show_progress:
            sections.append(self._render_progress())

        if self.config.show_urls:
            sections.append(self._render_pages())

        # Combine into layout
        content = "\n".join(sections)

        return Panel(
            content,
            title="[bold blue]Crawler Monitor[/bold blue]",
            border_style="blue",
        )

    def _render_stats(self) -> str:
        """Render statistics section.

        Returns:
            Formatted stats string
        """
        elapsed = self.stats.elapsed_seconds
        rate = self.stats.pages_per_second

        lines = [
            f"[cyan]Pages Crawled:[/cyan] {self.stats.pages_crawled}",
            f"[cyan]Pages Pending:[/cyan] {self.stats.pages_pending}",
            f"[cyan]Pages Failed:[/cyan] [red]{self.stats.pages_failed}[/red]",
            f"[cyan]Links Found:[/cyan] {self.stats.links_discovered}",
            f"[cyan]Current Depth:[/cyan] {self.stats.current_depth}",
            f"[cyan]Elapsed:[/cyan] {elapsed:.1f}s",
            f"[cyan]Rate:[/cyan] {rate:.2f} pages/sec",
        ]

        return "\n".join(lines)

    def _render_progress(self) -> str:
        """Render progress section.

        Returns:
            Formatted progress string
        """
        url = self.stats.current_url
        if len(url) > MAX_URL_DISPLAY_LENGTH:
            url = url[: MAX_URL_DISPLAY_LENGTH - 3] + "..."

        return f"\n[yellow]Current:[/yellow] {url}"

    def _render_pages(self) -> str:
        """Render recent pages section.

        Returns:
            Formatted pages string
        """
        if not self._recent_pages:
            return "\n[dim]No pages crawled yet...[/dim]"

        lines = ["\n[bold]Recent Pages:[/bold]"]

        for page in self._recent_pages[-10:]:
            status_color = "green" if page.status == CrawlPageStatus.SUCCESS else "red"
            url = page.url
            if len(url) > MAX_URL_DISPLAY_LENGTH:
                url = url[: MAX_URL_DISPLAY_LENGTH - 3] + "..."

            lines.append(f"  [{status_color}]●[/{status_color}] {url}")

        return "\n".join(lines)

    def print_summary(self, result: CrawlResult) -> None:
        """Print final crawl summary.

        Args:
            result: Crawl result
        """
        # Status panel
        status_color = "green" if result.status == CrawlStatus.COMPLETED else "red"
        status_text = f"[{status_color}]{result.status.value.upper()}[/{status_color}]"

        self.console.print()
        self.console.print(
            Panel(
                f"Crawl {status_text}",
                title="[bold]Summary[/bold]",
                border_style=status_color,
            )
        )

        # Stats table
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Start URL", result.start_url)
        table.add_row("Total Discovered", str(result.total_discovered))
        table.add_row("Total Crawled", str(result.total_crawled))
        table.add_row("Total Failed", str(result.total_failed))
        table.add_row("Duration", f"{result.duration_seconds:.2f}s")
        table.add_row("Success Rate", f"{result.success_rate:.1%}")

        self.console.print(table)

        # Error if any
        if result.error:
            self.console.print(f"\n[red]Error:[/red] {result.error}")


def create_monitor(
    show_urls: bool = True,
    compact: bool = False,
) -> CrawlMonitor:
    """Factory function to create monitor.

    Args:
        show_urls: Show crawled URLs
        compact: Use compact display

    Returns:
        Configured CrawlMonitor
    """
    config = MonitorConfig(
        show_urls=show_urls,
        compact_mode=compact,
    )
    return CrawlMonitor(config=config)


def print_crawl_summary(result: CrawlResult) -> None:
    """Convenience function to print crawl summary.

    Args:
        result: Crawl result
    """
    monitor = CrawlMonitor()
    monitor.print_summary(result)


def create_pages_table(pages: List[CrawlPage]) -> Table:
    """Create Rich table for pages.

    Args:
        pages: List of crawled pages

    Returns:
        Rich Table
    """
    table = Table(title="Crawled Pages")
    table.add_column("Status", style="bold")
    table.add_column("Depth", justify="center")
    table.add_column("Links", justify="right")
    table.add_column("URL")

    for page in pages[:MAX_DISPLAY_PAGES]:
        status_icon = "✓" if page.status == CrawlPageStatus.SUCCESS else "✗"
        status_color = "green" if page.status == CrawlPageStatus.SUCCESS else "red"

        url = page.url
        if len(url) > MAX_URL_DISPLAY_LENGTH:
            url = url[: MAX_URL_DISPLAY_LENGTH - 3] + "..."

        table.add_row(
            f"[{status_color}]{status_icon}[/{status_color}]",
            str(page.depth),
            str(page.links_found),
            url,
        )

    return table
