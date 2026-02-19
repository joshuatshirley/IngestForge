"""Verification Result UI for fact-checker.

Rich-based display of verification results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingestforge.agent.fact_checker import (
    VerificationResult,
    VerificationStatus,
    DebateRole,
)

MAX_DISPLAY_ARGUMENTS = 10
MAX_EVIDENCE_DISPLAY = 5


@dataclass
class DisplayConfig:
    """Configuration for result display."""

    show_arguments: bool = True
    show_evidence: bool = True
    show_scores: bool = True
    max_arguments: int = MAX_DISPLAY_ARGUMENTS
    compact: bool = False


class VerificationDisplay:
    """Displays verification results using Rich."""

    def __init__(
        self,
        console: Optional[Console] = None,
        config: Optional[DisplayConfig] = None,
    ) -> None:
        """Initialize display.

        Args:
            console: Rich console
            config: Display configuration
        """
        self.console = console or Console()
        self.config = config or DisplayConfig()

    def display(self, result: VerificationResult) -> None:
        """Display verification result.

        Args:
            result: Result to display
        """
        # Status panel
        self._display_status(result)

        # Score display
        if self.config.show_scores:
            self._display_scores(result)

        # Arguments
        if self.config.show_arguments and result.arguments:
            self._display_arguments(result)

        # Summary
        self._display_summary(result)

    def _display_status(self, result: VerificationResult) -> None:
        """Display status panel.

        Args:
            result: Verification result
        """
        status_colors = {
            VerificationStatus.VERIFIED: "green",
            VerificationStatus.REFUTED: "red",
            VerificationStatus.UNCERTAIN: "yellow",
            VerificationStatus.CONTESTED: "orange1",
        }

        status_icons = {
            VerificationStatus.VERIFIED: "[bold green]✓[/]",
            VerificationStatus.REFUTED: "[bold red]✗[/]",
            VerificationStatus.UNCERTAIN: "[bold yellow]?[/]",
            VerificationStatus.CONTESTED: "[bold orange1]⚔[/]",
        }

        color = status_colors.get(result.status, "white")
        icon = status_icons.get(result.status, "")

        status_text = f"{icon} {result.status.value.upper()}"
        confidence_text = f"Confidence: {result.confidence:.0%}"

        content = f"{status_text}\n{confidence_text}"

        panel = Panel(
            content,
            title="[bold]Claim Verification[/bold]",
            subtitle=result.claim.content[:50] + "...",
            border_style=color,
        )
        self.console.print(panel)

    def _display_scores(self, result: VerificationResult) -> None:
        """Display debate scores.

        Args:
            result: Verification result
        """
        if self.config.compact:
            self._display_scores_compact(result)
            return

        table = Table(title="Debate Scores", show_header=True)
        table.add_column("Role", style="cyan")
        table.add_column("Score", style="green")
        table.add_column("Bar", style="blue")

        # Proponent row
        prop_bar = self._make_bar(result.proponent_score)
        table.add_row(
            "Proponent",
            f"{result.proponent_score:.2f}",
            prop_bar,
        )

        # Critic row
        critic_bar = self._make_bar(result.critic_score)
        table.add_row(
            "Critic",
            f"{result.critic_score:.2f}",
            critic_bar,
        )

        self.console.print(table)

    def _display_scores_compact(self, result: VerificationResult) -> None:
        """Display scores in compact format.

        Args:
            result: Verification result
        """
        self.console.print(
            f"[cyan]Proponent:[/cyan] {result.proponent_score:.2f} | "
            f"[magenta]Critic:[/magenta] {result.critic_score:.2f}"
        )

    def _make_bar(self, score: float) -> str:
        """Make a text-based progress bar.

        Args:
            score: Score value (0-1)

        Returns:
            Bar string
        """
        filled = int(score * 20)
        empty = 20 - filled
        return "█" * filled + "░" * empty

    def _display_arguments(self, result: VerificationResult) -> None:
        """Display debate arguments.

        Args:
            result: Verification result
        """
        table = Table(title="Debate Arguments", show_header=True)
        table.add_column("Round", style="cyan", width=6)
        table.add_column("Role", style="green", width=12)
        table.add_column("Position", style="white")

        args_to_show = result.arguments[: self.config.max_arguments]

        for arg in args_to_show:
            role_style = (
                "bold green" if arg.role == DebateRole.PROPONENT else "bold magenta"
            )
            position = (
                arg.position[:60] + "..." if len(arg.position) > 60 else arg.position
            )

            table.add_row(
                str(arg.round_number),
                f"[{role_style}]{arg.role.value}[/]",
                position,
            )

        self.console.print(table)

        if len(result.arguments) > self.config.max_arguments:
            remaining = len(result.arguments) - self.config.max_arguments
            self.console.print(f"[dim]... and {remaining} more arguments[/dim]")

    def _display_summary(self, result: VerificationResult) -> None:
        """Display result summary.

        Args:
            result: Verification result
        """
        if not result.summary:
            return

        panel = Panel(
            result.summary,
            title="[bold]Summary[/bold]",
            border_style="blue",
        )
        self.console.print(panel)


class BatchVerificationDisplay:
    """Displays multiple verification results."""

    def __init__(
        self,
        console: Optional[Console] = None,
    ) -> None:
        """Initialize batch display.

        Args:
            console: Rich console
        """
        self.console = console or Console()
        self._single_display = VerificationDisplay(
            console=self.console,
            config=DisplayConfig(compact=True),
        )

    def display_batch(self, results: list[VerificationResult]) -> None:
        """Display multiple results.

        Args:
            results: Results to display
        """
        if not results:
            self.console.print("[yellow]No verification results[/yellow]")
            return

        # Summary table
        self._display_summary_table(results)

        # Statistics
        self._display_statistics(results)

    def _display_summary_table(
        self,
        results: list[VerificationResult],
    ) -> None:
        """Display summary table.

        Args:
            results: Results to summarize
        """
        table = Table(title="Verification Results", show_header=True)
        table.add_column("#", style="dim", width=4)
        table.add_column("Claim", style="white")
        table.add_column("Status", style="cyan")
        table.add_column("Confidence", style="green")

        for i, result in enumerate(results[:20], 1):
            claim = result.claim.content[:40] + "..."
            status = self._format_status(result.status)
            confidence = f"{result.confidence:.0%}"

            table.add_row(str(i), claim, status, confidence)

        self.console.print(table)

    def _format_status(self, status: VerificationStatus) -> str:
        """Format status for display.

        Args:
            status: Verification status

        Returns:
            Formatted string
        """
        formats = {
            VerificationStatus.VERIFIED: "[green]✓ Verified[/]",
            VerificationStatus.REFUTED: "[red]✗ Refuted[/]",
            VerificationStatus.UNCERTAIN: "[yellow]? Uncertain[/]",
            VerificationStatus.CONTESTED: "[orange1]⚔ Contested[/]",
        }
        return formats.get(status, status.value)

    def _display_statistics(
        self,
        results: list[VerificationResult],
    ) -> None:
        """Display verification statistics.

        Args:
            results: Results for statistics
        """
        total = len(results)
        verified = sum(1 for r in results if r.is_verified)
        refuted = sum(1 for r in results if r.status == VerificationStatus.REFUTED)
        uncertain = total - verified - refuted

        avg_confidence = sum(r.confidence for r in results) / max(total, 1)

        stats = f"""
[bold]Statistics[/bold]
Total claims: {total}
[green]Verified: {verified}[/green] | [red]Refuted: {refuted}[/red] | [yellow]Other: {uncertain}[/yellow]
Average confidence: {avg_confidence:.0%}
"""
        self.console.print(stats)


def display_result(
    result: VerificationResult,
    console: Optional[Console] = None,
) -> None:
    """Convenience function to display a result.

    Args:
        result: Result to display
        console: Optional console
    """
    display = VerificationDisplay(console=console)
    display.display(result)


def display_batch(
    results: list[VerificationResult],
    console: Optional[Console] = None,
) -> None:
    """Convenience function to display batch results.

    Args:
        results: Results to display
        console: Optional console
    """
    display = BatchVerificationDisplay(console=console)
    display.display_batch(results)
