"""Tests for verification result UI.

Tests Rich-based display components."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from ingestforge.agent.fact_checker import (
    VerificationStatus,
    DebateRole,
    DebateArgument,
    Claim,
    VerificationResult,
)
from ingestforge.agent.verification_ui import (
    DisplayConfig,
    VerificationDisplay,
    BatchVerificationDisplay,
    display_result,
    display_batch,
    MAX_DISPLAY_ARGUMENTS,
)

# Test fixtures


def make_result(
    status: VerificationStatus = VerificationStatus.VERIFIED,
    confidence: float = 0.9,
) -> VerificationResult:
    """Create a test verification result."""
    claim = Claim(content="Test claim for verification")
    args = [
        DebateArgument(
            role=DebateRole.PROPONENT,
            position="Supporting position",
            round_number=0,
        ),
        DebateArgument(
            role=DebateRole.CRITIC,
            position="Critique position",
            round_number=0,
        ),
    ]

    return VerificationResult(
        claim=claim,
        status=status,
        confidence=confidence,
        proponent_score=0.8,
        critic_score=0.4,
        arguments=args,
        summary="Test summary text",
    )


def capture_console() -> tuple[Console, StringIO]:
    """Create console that captures output."""
    output = StringIO()
    console = Console(file=output, force_terminal=True, width=100)
    return console, output


# DisplayConfig tests


class TestDisplayConfig:
    """Tests for DisplayConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = DisplayConfig()

        assert config.show_arguments is True
        assert config.show_evidence is True
        assert config.show_scores is True
        assert config.compact is False

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = DisplayConfig(
            show_arguments=False,
            compact=True,
            max_arguments=5,
        )

        assert config.show_arguments is False
        assert config.compact is True
        assert config.max_arguments == 5


# VerificationDisplay tests


class TestVerificationDisplay:
    """Tests for VerificationDisplay class."""

    def test_display_creation(self) -> None:
        """Test creating display."""
        display = VerificationDisplay()

        assert display is not None
        assert display.console is not None

    def test_display_with_config(self) -> None:
        """Test display with custom config."""
        config = DisplayConfig(compact=True)
        display = VerificationDisplay(config=config)

        assert display.config.compact is True

    def test_display_verified(self) -> None:
        """Test displaying verified result."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result(VerificationStatus.VERIFIED)

        display.display(result)

        text = output.getvalue()
        assert "VERIFIED" in text

    def test_display_refuted(self) -> None:
        """Test displaying refuted result."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result(VerificationStatus.REFUTED)

        display.display(result)

        text = output.getvalue()
        assert "REFUTED" in text

    def test_display_uncertain(self) -> None:
        """Test displaying uncertain result."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result(VerificationStatus.UNCERTAIN)

        display.display(result)

        text = output.getvalue()
        assert "UNCERTAIN" in text

    def test_display_contested(self) -> None:
        """Test displaying contested result."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result(VerificationStatus.CONTESTED)

        display.display(result)

        text = output.getvalue()
        assert "CONTESTED" in text


class TestScoreDisplay:
    """Tests for score display."""

    def test_shows_proponent_score(self) -> None:
        """Test proponent score is shown."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result()

        display.display(result)

        text = output.getvalue()
        assert "Proponent" in text or "0.80" in text

    def test_shows_critic_score(self) -> None:
        """Test critic score is shown."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result()

        display.display(result)

        text = output.getvalue()
        assert "Critic" in text or "0.40" in text

    def test_compact_scores(self) -> None:
        """Test compact score display."""
        console, output = capture_console()
        config = DisplayConfig(compact=True)
        display = VerificationDisplay(console=console, config=config)
        result = make_result()

        display.display(result)

        text = output.getvalue()
        # Compact mode uses different format
        assert "Proponent" in text or "0.80" in text


class TestArgumentDisplay:
    """Tests for argument display."""

    def test_shows_arguments(self) -> None:
        """Test arguments are shown."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result()

        display.display(result)

        text = output.getvalue()
        assert "proponent" in text.lower() or "critic" in text.lower()

    def test_hides_arguments_when_disabled(self) -> None:
        """Test arguments hidden when disabled."""
        console, output = capture_console()
        config = DisplayConfig(show_arguments=False)
        display = VerificationDisplay(console=console, config=config)
        result = make_result()

        display.display(result)

        text = output.getvalue()
        # Arguments table should not appear
        assert "Debate Arguments" not in text


class TestSummaryDisplay:
    """Tests for summary display."""

    def test_shows_summary(self) -> None:
        """Test summary is shown."""
        console, output = capture_console()
        display = VerificationDisplay(console=console)
        result = make_result()

        display.display(result)

        text = output.getvalue()
        assert "Summary" in text or "Test summary" in text


# BatchVerificationDisplay tests


class TestBatchVerificationDisplay:
    """Tests for BatchVerificationDisplay class."""

    def test_batch_display_creation(self) -> None:
        """Test creating batch display."""
        display = BatchVerificationDisplay()

        assert display is not None

    def test_display_empty_batch(self) -> None:
        """Test displaying empty batch."""
        console, output = capture_console()
        display = BatchVerificationDisplay(console=console)

        display.display_batch([])

        text = output.getvalue()
        assert "No verification" in text

    def test_display_single_result(self) -> None:
        """Test displaying single result in batch."""
        console, output = capture_console()
        display = BatchVerificationDisplay(console=console)
        results = [make_result()]

        display.display_batch(results)

        text = output.getvalue()
        assert "Results" in text or "Verified" in text

    def test_display_multiple_results(self) -> None:
        """Test displaying multiple results."""
        console, output = capture_console()
        display = BatchVerificationDisplay(console=console)
        results = [
            make_result(VerificationStatus.VERIFIED),
            make_result(VerificationStatus.REFUTED),
            make_result(VerificationStatus.UNCERTAIN),
        ]

        display.display_batch(results)

        text = output.getvalue()
        assert "Total" in text or "Statistics" in text


class TestBatchStatistics:
    """Tests for batch statistics."""

    def test_shows_total_count(self) -> None:
        """Test total count is shown."""
        console, output = capture_console()
        display = BatchVerificationDisplay(console=console)
        results = [make_result() for _ in range(5)]

        display.display_batch(results)

        text = output.getvalue()
        assert "5" in text or "Total" in text

    def test_shows_verified_count(self) -> None:
        """Test verified count is shown."""
        console, output = capture_console()
        display = BatchVerificationDisplay(console=console)
        results = [
            make_result(VerificationStatus.VERIFIED),
            make_result(VerificationStatus.VERIFIED),
            make_result(VerificationStatus.REFUTED),
        ]

        display.display_batch(results)

        text = output.getvalue()
        # Should show verified count
        assert "Verified" in text


# Convenience function tests


class TestDisplayResult:
    """Tests for display_result function."""

    def test_display_result(self) -> None:
        """Test convenience display function."""
        console, output = capture_console()
        result = make_result()

        display_result(result, console=console)

        text = output.getvalue()
        assert len(text) > 0


class TestDisplayBatch:
    """Tests for display_batch function."""

    def test_display_batch(self) -> None:
        """Test convenience batch display function."""
        console, output = capture_console()
        results = [make_result(), make_result()]

        display_batch(results, console=console)

        text = output.getvalue()
        assert len(text) > 0


# Constant tests


class TestConstants:
    """Tests for module constants."""

    def test_max_display_arguments(self) -> None:
        """Test MAX_DISPLAY_ARGUMENTS is reasonable."""
        assert MAX_DISPLAY_ARGUMENTS > 0
        assert MAX_DISPLAY_ARGUMENTS == 10
