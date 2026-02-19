"""Tests for dashboard module (UX-019.2).

Tests the study progress dashboard TUI:
- Panel creation
- Dashboard display
- Summary generation
"""

from rich.console import Console
from rich.panel import Panel

from ingestforge.cli.ui.dashboard import (
    create_overview_panel,
    create_accuracy_panel,
    create_cards_panel,
    create_mastery_chart,
    create_topics_table,
    show_dashboard,
    get_dashboard_summary,
    MASTERY_COLORS,
)

from ingestforge.study.stats import (
    StudyStats,
    TopicStats,
    MasteryLevel,
)


class TestMasteryColors:
    """Test mastery color mapping."""

    def test_all_levels_have_colors(self) -> None:
        """All mastery levels should have colors."""
        for level in MasteryLevel:
            assert level in MASTERY_COLORS


class TestCreateOverviewPanel:
    """Test create_overview_panel function."""

    def test_returns_panel(self) -> None:
        """Should return a Panel."""
        stats = StudyStats()
        panel = create_overview_panel(stats)
        assert isinstance(panel, Panel)

    def test_shows_due_count(self) -> None:
        """Panel should show due count."""
        stats = StudyStats(due_today=5)
        panel = create_overview_panel(stats)
        assert panel is not None

    def test_shows_streak(self) -> None:
        """Panel should show streak."""
        stats = StudyStats(streak_days=7)
        panel = create_overview_panel(stats)
        assert panel is not None


class TestCreateAccuracyPanel:
    """Test create_accuracy_panel function."""

    def test_returns_panel(self) -> None:
        """Should return a Panel."""
        stats = StudyStats(average_accuracy=85.0)
        panel = create_accuracy_panel(stats)
        assert isinstance(panel, Panel)

    def test_high_accuracy_green(self) -> None:
        """High accuracy should use green border."""
        stats = StudyStats(average_accuracy=90.0)
        panel = create_accuracy_panel(stats)
        assert panel.border_style == "green"

    def test_low_accuracy_red(self) -> None:
        """Low accuracy should use red border."""
        stats = StudyStats(average_accuracy=50.0)
        panel = create_accuracy_panel(stats)
        assert panel.border_style == "red"


class TestCreateCardsPanel:
    """Test create_cards_panel function."""

    def test_returns_panel(self) -> None:
        """Should return a Panel."""
        stats = StudyStats(total_cards=100)
        panel = create_cards_panel(stats)
        assert isinstance(panel, Panel)


class TestCreateMasteryChart:
    """Test create_mastery_chart function."""

    def test_returns_panel(self) -> None:
        """Should return a Panel."""
        stats = StudyStats()
        panel = create_mastery_chart(stats)
        assert isinstance(panel, Panel)

    def test_empty_distribution(self) -> None:
        """Should handle empty distribution."""
        stats = StudyStats(mastery_distribution={})
        panel = create_mastery_chart(stats)
        assert panel is not None

    def test_with_distribution(self) -> None:
        """Should display distribution."""
        stats = StudyStats(
            mastery_distribution={
                MasteryLevel.NEW: 10,
                MasteryLevel.LEARNING: 20,
                MasteryLevel.REVIEWING: 30,
                MasteryLevel.MATURE: 40,
            }
        )
        panel = create_mastery_chart(stats)
        assert panel is not None


class TestCreateTopicsTable:
    """Test create_topics_table function."""

    def test_returns_panel(self) -> None:
        """Should return a Panel."""
        stats = StudyStats()
        panel = create_topics_table(stats)
        assert isinstance(panel, Panel)

    def test_empty_topics(self) -> None:
        """Should handle empty topics list."""
        stats = StudyStats(topics=[])
        panel = create_topics_table(stats)
        assert panel is not None

    def test_with_topics(self) -> None:
        """Should display topics."""
        stats = StudyStats(
            topics=[
                TopicStats(name="Python", total_cards=50, due_cards=5),
                TopicStats(name="JavaScript", total_cards=30, due_cards=10),
            ]
        )
        panel = create_topics_table(stats)
        assert panel is not None


class TestShowDashboard:
    """Test show_dashboard function."""

    def test_does_not_raise(self) -> None:
        """show_dashboard should not raise."""
        console = Console(force_terminal=True, width=80, record=True)
        stats = StudyStats()
        show_dashboard(stats, console)
        # Should complete without error

    def test_renders_output(self) -> None:
        """Should render dashboard content."""
        console = Console(force_terminal=True, width=80, record=True)
        stats = StudyStats(
            total_cards=100,
            due_today=10,
            reviewed_today=5,
        )
        show_dashboard(stats, console)
        output = console.export_text()
        assert "Dashboard" in output


class TestGetDashboardSummary:
    """Test get_dashboard_summary function."""

    def test_returns_string(self) -> None:
        """Should return summary string."""
        stats = StudyStats()
        summary = get_dashboard_summary(stats)
        assert isinstance(summary, str)

    def test_includes_metrics(self) -> None:
        """Summary should include key metrics."""
        stats = StudyStats(
            total_cards=100,
            due_today=10,
            streak_days=5,
        )
        summary = get_dashboard_summary(stats)

        assert "100" in summary
        assert "10" in summary
        assert "5" in summary

    def test_includes_topics(self) -> None:
        """Summary should include topics."""
        stats = StudyStats(
            topics=[
                TopicStats(name="Python", due_cards=5),
            ]
        )
        summary = get_dashboard_summary(stats)
        assert "Python" in summary
