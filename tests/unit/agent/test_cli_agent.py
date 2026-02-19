"""Tests for agent CLI commands.

Tests CLI integration for agent functionality."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from ingestforge.cli.commands.agent import (
    agent_group,
)


@pytest.fixture
def runner() -> CliRunner:
    """Create CLI runner."""
    return CliRunner()


# Agent group tests


class TestAgentGroup:
    """Tests for agent command group."""

    def test_group_exists(self, runner: CliRunner) -> None:
        """Test agent group exists."""
        result = runner.invoke(agent_group, ["--help"])

        assert result.exit_code == 0
        assert "agent" in result.output.lower()

    def test_has_run_command(self, runner: CliRunner) -> None:
        """Test run command is available."""
        result = runner.invoke(agent_group, ["--help"])

        assert "run" in result.output

    def test_has_tools_command(self, runner: CliRunner) -> None:
        """Test tools command is available."""
        result = runner.invoke(agent_group, ["--help"])

        assert "tools" in result.output

    def test_has_status_command(self, runner: CliRunner) -> None:
        """Test status command is available."""
        result = runner.invoke(agent_group, ["--help"])

        assert "status" in result.output


# Run command tests


class TestRunCommand:
    """Tests for run command."""

    @pytest.mark.skip(reason="Requires LLM model - run manually with local LLM")
    @pytest.mark.integration
    def test_run_simple_task(self, runner: CliRunner) -> None:
        """Test running a simple task (requires LLM)."""
        result = runner.invoke(agent_group, ["run", "Test task", "-q"])

        assert result.exit_code == 0
        assert "Result" in result.output or "Answer" in result.output

    @pytest.mark.skip(reason="Requires LLM model - run manually with local LLM")
    @pytest.mark.integration
    def test_run_with_iterations(self, runner: CliRunner) -> None:
        """Test running with custom iterations (requires LLM)."""
        result = runner.invoke(
            agent_group,
            ["run", "Research topic", "-m", "5", "-q"],
        )

        assert result.exit_code == 0

    def test_run_empty_task(self, runner: CliRunner) -> None:
        """Test running empty task."""
        result = runner.invoke(agent_group, ["run", ""])

        assert result.exit_code == 1
        assert "Empty task" in result.output

    @pytest.mark.skip(reason="Requires LLM model - run manually with local LLM")
    @pytest.mark.integration
    def test_run_with_output(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test running with output file (requires LLM)."""
        output = tmp_path / "report.md"

        result = runner.invoke(
            agent_group,
            ["run", "Generate report", "-q", "-o", str(output)],
        )

        assert result.exit_code == 0
        assert output.exists()

    @pytest.mark.skip(reason="Requires LLM model - run manually with local LLM")
    @pytest.mark.integration
    def test_run_with_json_format(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test running with JSON output (requires LLM)."""
        output = tmp_path / "report.json"

        result = runner.invoke(
            agent_group,
            ["run", "Task", "-q", "-o", str(output), "-f", "json"],
        )

        assert result.exit_code == 0
        content = output.read_text()
        assert '"title"' in content

    @pytest.mark.skip(reason="Requires LLM model - run manually with local LLM")
    @pytest.mark.integration
    def test_run_with_html_format(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test running with HTML output (requires LLM)."""
        output = tmp_path / "report.html"

        result = runner.invoke(
            agent_group,
            ["run", "Task", "-q", "-o", str(output), "-f", "html"],
        )

        assert result.exit_code == 0
        content = output.read_text()
        assert "<html>" in content


# Tools command tests


class TestToolsCommand:
    """Tests for tools command."""

    def test_list_tools(self, runner: CliRunner) -> None:
        """Test listing tools."""
        result = runner.invoke(agent_group, ["tools"])

        assert result.exit_code == 0
        assert "echo" in result.output
        assert "format" in result.output

    def test_tools_shows_category(self, runner: CliRunner) -> None:
        """Test that category is shown."""
        result = runner.invoke(agent_group, ["tools"])

        assert (
            "utility" in result.output.lower() or "transform" in result.output.lower()
        )


# Status command tests


class TestStatusCommand:
    """Tests for status command."""

    def test_show_status(self, runner: CliRunner) -> None:
        """Test showing status."""
        result = runner.invoke(agent_group, ["status"])

        assert result.exit_code == 0
        assert "Status" in result.output

    def test_status_shows_tools(self, runner: CliRunner) -> None:
        """Test status shows tool count."""
        result = runner.invoke(agent_group, ["status"])

        assert "Tools" in result.output or "registered" in result.output

    def test_status_shows_iterations(self, runner: CliRunner) -> None:
        """Test status shows max steps."""
        result = runner.invoke(agent_group, ["status"])

        assert "steps" in result.output.lower()  # Fixed: was "iterations"


# Help tests


class TestHelpOutput:
    """Tests for help output."""

    def test_run_help(self, runner: CliRunner) -> None:
        """Test run command help."""
        result = runner.invoke(agent_group, ["run", "--help"])

        assert result.exit_code == 0
        assert "--max-steps" in result.output  # Fixed: was --max-iterations
        assert "--output" in result.output
        assert "--format" in result.output
        assert "--domain-aware" in result.output  # STORY-30

    def test_tools_help(self, runner: CliRunner) -> None:
        """Test tools command help."""
        result = runner.invoke(agent_group, ["tools", "--help"])

        assert result.exit_code == 0

    def test_status_help(self, runner: CliRunner) -> None:
        """Test status command help."""
        result = runner.invoke(agent_group, ["status", "--help"])

        assert result.exit_code == 0
