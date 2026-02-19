"""Story Arc command - Analyze narrative structure.

Provides story arc analysis with multiple structure types:
- analyze: Analyze story structure (three-act, hero-journey, etc.)
- visualize: Export arc as diagram or ASCII art"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.literary.base import LiteraryCommand
from ingestforge.cli.literary.models import (
    PlotPoint,
    StoryArc,
    STRUCTURE_TYPES,
)
from ingestforge.cli.core import ProgressManager
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Structure Templates
# ============================================================================

THREE_ACT_TEMPLATE = [
    {"name": "Exposition", "type": "exposition", "position": 0.0},
    {"name": "Inciting Incident", "type": "inciting_incident", "position": 0.1},
    {"name": "Rising Action", "type": "rising_action", "position": 0.25},
    {"name": "Midpoint", "type": "midpoint", "position": 0.5},
    {"name": "Crisis", "type": "crisis", "position": 0.75},
    {"name": "Climax", "type": "climax", "position": 0.85},
    {"name": "Falling Action", "type": "falling_action", "position": 0.9},
    {"name": "Resolution", "type": "resolution", "position": 1.0},
]

HERO_JOURNEY_TEMPLATE = [
    {"name": "Ordinary World", "type": "ordinary_world", "position": 0.0},
    {"name": "Call to Adventure", "type": "call", "position": 0.08},
    {"name": "Refusal of the Call", "type": "refusal", "position": 0.12},
    {"name": "Meeting the Mentor", "type": "mentor", "position": 0.17},
    {"name": "Crossing the Threshold", "type": "threshold", "position": 0.25},
    {"name": "Tests, Allies, Enemies", "type": "tests", "position": 0.35},
    {"name": "Approach to Innermost Cave", "type": "approach", "position": 0.45},
    {"name": "Ordeal", "type": "ordeal", "position": 0.5},
    {"name": "Reward", "type": "reward", "position": 0.6},
    {"name": "The Road Back", "type": "road_back", "position": 0.75},
    {"name": "Resurrection", "type": "resurrection", "position": 0.85},
    {"name": "Return with Elixir", "type": "return", "position": 1.0},
]

FIVE_ACT_TEMPLATE = [
    {"name": "Exposition", "type": "exposition", "position": 0.0},
    {"name": "Rising Action", "type": "rising_action", "position": 0.2},
    {"name": "Climax", "type": "climax", "position": 0.4},
    {"name": "Falling Action", "type": "falling_action", "position": 0.6},
    {"name": "Denouement", "type": "denouement", "position": 0.8},
    {"name": "Resolution", "type": "resolution", "position": 1.0},
]

FREYTAG_TEMPLATE = [
    {"name": "Introduction", "type": "exposition", "position": 0.0},
    {"name": "Rising Action", "type": "rising_action", "position": 0.2},
    {"name": "Climax", "type": "climax", "position": 0.5},
    {"name": "Falling Action", "type": "falling_action", "position": 0.7},
    {"name": "Catastrophe", "type": "catastrophe", "position": 0.9},
    {"name": "Resolution", "type": "resolution", "position": 1.0},
]

STRUCTURE_TEMPLATES = {
    "three-act": THREE_ACT_TEMPLATE,
    "hero-journey": HERO_JOURNEY_TEMPLATE,
    "five-act": FIVE_ACT_TEMPLATE,
    "freytag": FREYTAG_TEMPLATE,
}

# ============================================================================
# Story Arc Analyzer Class
# ============================================================================


class StoryArcAnalyzer:
    """Analyze story structure and narrative arc.

    Identifies plot points and tension curves.
    Follows Rule #4: All methods <60 lines.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """Initialize story arc analyzer.

        Args:
            llm_client: Optional LLM for enhanced analysis
        """
        self.llm_client = llm_client

        # Keywords for plot point detection
        self._tension_keywords = {
            "high": [
                "danger",
                "fight",
                "battle",
                "death",
                "kill",
                "war",
                "attack",
                "escape",
                "crisis",
                "desperate",
            ],
            "rising": [
                "challenge",
                "conflict",
                "problem",
                "trouble",
                "threat",
                "discover",
                "reveal",
                "secret",
                "journey",
                "quest",
            ],
            "low": [
                "peace",
                "calm",
                "rest",
                "home",
                "safe",
                "happy",
                "resolution",
                "end",
                "finally",
                "together",
            ],
        }

    def analyze_structure(
        self, chunks: List[Any], structure: str = "three-act"
    ) -> StoryArc:
        """Analyze narrative structure.

        Args:
            chunks: Ordered text chunks
            structure: Structure type (three-act, hero-journey, etc.)

        Returns:
            StoryArc with detected plot points

        Rule #1: Early return for invalid structure
        """
        if structure not in STRUCTURE_TEMPLATES:
            logger.warning(f"Unknown structure: {structure}, using three-act")
            structure = "three-act"

        template = STRUCTURE_TEMPLATES[structure]

        # Calculate tension curve
        tension_curve = self.calculate_tension_curve(chunks)

        # Identify plot points
        plot_points = self.identify_plot_points(chunks, template)

        # Build acts
        acts = self._build_acts(template, chunks)

        arc = StoryArc(
            structure_type=structure,
            plot_points=plot_points,
            tension_curve=tension_curve,
            acts=acts,
        )

        # Use LLM for enhanced summary if available
        if self.llm_client:
            arc.summary = self._generate_summary(arc, chunks)
        else:
            arc.summary = f"Story follows {structure} structure with {len(plot_points)} plot points."

        return arc

    def identify_plot_points(
        self, chunks: List[Any], template: List[Dict[str, Any]]
    ) -> List[PlotPoint]:
        """Identify plot points based on template.

        Args:
            chunks: Text chunks
            template: Structure template with positions

        Returns:
            List of PlotPoint objects
        """
        plot_points: List[PlotPoint] = []
        total_chunks = len(chunks)

        for point_template in template:
            position = point_template["position"]
            chunk_index = int(position * (total_chunks - 1))

            # Get description from chunk
            chunk = chunks[chunk_index] if chunk_index < total_chunks else chunks[-1]
            content = getattr(chunk, "content", str(chunk))
            description = content[:200] + "..." if len(content) > 200 else content

            plot_point = PlotPoint(
                name=point_template["name"],
                type=point_template["type"],
                position=position,
                description=description,
                chunk_index=chunk_index,
            )
            plot_points.append(plot_point)

        return plot_points

    def calculate_tension_curve(self, chunks: List[Any]) -> List[float]:
        """Calculate tension/intensity curve across narrative.

        Args:
            chunks: Text chunks

        Returns:
            List of tension values (0.0 to 1.0)
        """
        tension_values: List[float] = []

        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk)).lower()
            tension = self._calculate_chunk_tension(content)
            tension_values.append(tension)

        # Smooth the curve
        return self._smooth_curve(tension_values)

    def _calculate_chunk_tension(self, content: str) -> float:
        """Calculate tension for a single chunk.

        Args:
            content: Chunk content

        Returns:
            Tension value (0.0 to 1.0)
        """
        high_count = sum(content.count(kw) for kw in self._tension_keywords["high"])
        rising_count = sum(content.count(kw) for kw in self._tension_keywords["rising"])
        low_count = sum(content.count(kw) for kw in self._tension_keywords["low"])

        # Calculate weighted tension
        tension = (high_count * 3 + rising_count * 2 - low_count) / 10

        return max(0.0, min(1.0, 0.3 + tension))

    def _smooth_curve(self, values: List[float], window: int = 3) -> List[float]:
        """Smooth curve with moving average.

        Args:
            values: Raw values
            window: Smoothing window size

        Returns:
            Smoothed values
        """
        if len(values) < window:
            return values

        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            avg = sum(values[start:end]) / (end - start)
            smoothed.append(round(avg, 3))

        return smoothed

    def _build_acts(
        self, template: List[Dict[str, Any]], chunks: List[Any]
    ) -> List[Dict[str, Any]]:
        """Build act/section divisions.

        Args:
            template: Structure template
            chunks: Text chunks

        Returns:
            List of act dictionaries
        """
        # Group template points into acts
        act_boundaries = [0.0, 0.33, 0.67, 1.0]  # Default three-act

        acts = []
        for i in range(len(act_boundaries) - 1):
            start = act_boundaries[i]
            end = act_boundaries[i + 1]

            points_in_act = [
                p
                for p in template
                if start <= p["position"] < end or (end == 1.0 and p["position"] == 1.0)
            ]

            act = {
                "number": i + 1,
                "name": f"Act {i + 1}",
                "start_position": start,
                "end_position": end,
                "plot_points": [p["name"] for p in points_in_act],
            }
            acts.append(act)

        return acts

    def _generate_summary(self, arc: StoryArc, chunks: List[Any]) -> str:
        """Generate arc summary using LLM.

        Args:
            arc: Story arc to summarize
            chunks: Text chunks

        Returns:
            Summary string
        """
        # Build context from beginning, middle, and end
        sample_indices = [0, len(chunks) // 2, -1]
        context_parts = []

        for idx in sample_indices:
            chunk = chunks[idx]
            content = getattr(chunk, "content", str(chunk))[:300]
            context_parts.append(content)

        context = "\n---\n".join(context_parts)

        prompt = (
            f"Analyze this narrative using {arc.structure_type} structure:\n\n"
            f"{context}\n\n"
            "Provide a 2-3 sentence summary of the story arc."
        )

        try:
            return self.llm_client.generate(prompt)
        except Exception as e:
            logger.warning(f"Failed to generate summary: {e}")
            return f"Story follows {arc.structure_type} structure."

    def export_visualization(self, arc: StoryArc, format: str = "mermaid") -> str:
        """Export arc visualization.

        Args:
            arc: Story arc to export
            format: Output format (mermaid, ascii)

        Returns:
            Visualization string
        """
        if format == "ascii":
            return arc.to_ascii_art()
        return arc.to_mermaid()


# ============================================================================
# Arc Command Class
# ============================================================================


class ArcCommand(LiteraryCommand):
    """Analyze story structure and narrative arc."""

    def execute(
        self,
        work: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        structure: str = "three-act",
        visualize: bool = False,
    ) -> int:
        """Analyze story arc in a literary work.

        Args:
            work: Name of the literary work
            project: Project directory
            output: Output file (optional)
            structure: Structure type
            visualize: Include visualization

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_work_name(work)
            self._validate_structure(structure)

            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self._search_for_structure(ctx["storage"], work)

            if not chunks:
                self._handle_no_context(work)
                return 0

            analyzer = StoryArcAnalyzer(llm_client)

            arc = ProgressManager.run_with_spinner(
                lambda: analyzer.analyze_structure(chunks, structure),
                "Analyzing story structure...",
                "Analysis complete",
            )

            self._display_arc(work, arc, visualize)

            if output:
                self._save_arc(output, work, arc)

            return 0

        except Exception as e:
            return self.handle_error(e, "Arc analysis failed")

    def _validate_structure(self, structure: str) -> None:
        """Validate structure type.

        Args:
            structure: Structure type to validate

        Raises:
            typer.BadParameter: If invalid
        """
        if structure not in STRUCTURE_TYPES:
            raise typer.BadParameter(
                f"Invalid structure: {structure}. "
                f"Valid options: {', '.join(STRUCTURE_TYPES)}"
            )

    def _search_for_structure(self, storage: Any, work: str) -> List[Any]:
        """Search for structural context."""
        return ProgressManager.run_with_spinner(
            lambda: storage.search(f"{work} plot structure events chapters", k=30),
            f"Searching for '{work}'...",
            "Context retrieved",
        )

    def _handle_no_context(self, work: str) -> None:
        """Handle case where no context found."""
        self.print_warning(f"No context found for '{work}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {work}\n"
            "  2. Using 'lit gather' to fetch Wikipedia pages\n"
            "  3. Checking the work name spelling"
        )

    def _display_arc(self, work: str, arc: StoryArc, visualize: bool) -> None:
        """Display story arc analysis."""
        self.console.print()

        # Plot points table
        table = Table(title=f"Story Arc: {work} ({arc.structure_type})")
        table.add_column("Position", style="dim")
        table.add_column("Plot Point", style="cyan")
        table.add_column("Type", style="magenta")

        for point in arc.plot_points:
            table.add_row(
                f"{point.position:.0%}",
                point.name,
                point.type,
            )

        self.console.print(table)

        # Summary
        if arc.summary:
            self.console.print()
            panel = Panel(
                arc.summary,
                title="[bold]Summary[/bold]",
                border_style="blue",
            )
            self.console.print(panel)

        # Visualization
        if visualize:
            self.console.print()
            self.console.print("[bold]Tension Curve:[/bold]")
            self.console.print(arc.to_ascii_art())

            self.console.print()
            self.console.print("[bold]Mermaid Diagram:[/bold]")
            self.console.print("```mermaid")
            self.console.print(arc.to_mermaid())
            self.console.print("```")

    def _save_arc(self, output: Path, work: str, arc: StoryArc) -> None:
        """Save arc analysis to file.

        Rule #1: Max 3 nesting levels via helper extraction.
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                data = {
                    "work": work,
                    "generated": timestamp,
                    "arc": arc.to_dict(),
                }
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")

            elif output.suffix == ".png":
                # Save ASCII art (would need proper PNG export for real implementation)
                self.print_warning("PNG export not implemented, saving as text")
                output = output.with_suffix(".txt")
                output.write_text(arc.to_ascii_art(), encoding="utf-8")

            else:
                content = self._format_arc_markdown(work, timestamp, arc)
                output.write_text(content, encoding="utf-8")

            self.print_success(f"Arc saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save: {e}")

    def _format_arc_markdown(self, work: str, timestamp: str, arc: StoryArc) -> str:
        """Format arc as markdown.

        Rule #1: Extracted to reduce nesting in _save_arc.

        Args:
            work: Work name
            timestamp: Generation timestamp
            arc: Story arc to format

        Returns:
            Markdown formatted string
        """
        lines = [
            f"# Story Arc: {work}",
            "",
            f"Generated: {timestamp}",
            f"Structure: {arc.structure_type}",
            "",
            "---",
            "",
            "## Summary",
            arc.summary,
            "",
            "## Plot Points",
            "",
        ]

        for point in arc.plot_points:
            lines.append(f"### {point.name} ({point.position:.0%})")
            lines.append(f"Type: {point.type}")
            lines.append(f"{point.description}")
            lines.append("")

        lines.extend(
            [
                "## Tension Curve",
                "```",
                arc.to_ascii_art(),
                "```",
                "",
                "## Mermaid Diagram",
                "```mermaid",
                arc.to_mermaid(),
                "```",
            ]
        )

        return "\n".join(lines)


# ============================================================================
# Typer Command Wrapper
# ============================================================================


def command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    structure: str = typer.Option(
        "three-act",
        "--structure",
        "-s",
        help=f"Structure type ({', '.join(STRUCTURE_TYPES)})",
    ),
    visualize: bool = typer.Option(
        False, "--visualize", "-v", help="Include visualizations"
    ),
) -> None:
    """Analyze story structure and narrative arc.

    Identifies plot points and tension curve using various structural models.

    Structure types:
    - three-act: Classic three-act structure (default)
    - hero-journey: Joseph Campbell's Hero's Journey
    - five-act: Shakespearean five-act structure
    - freytag: Freytag's Pyramid

    Requires documents about the work to be ingested first.

    Examples:
        # Analyze with default structure
        ingestforge lit arc "The Odyssey"

        # Use hero's journey
        ingestforge lit arc "Star Wars" --structure hero-journey

        # Include visualizations
        ingestforge lit arc "Hamlet" --visualize

        # Save to file
        ingestforge lit arc "1984" -o arc.md
    """
    cmd = ArcCommand()
    exit_code = cmd.execute(work, project, output, structure, visualize)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for visualization
def visualize_command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for visualization"
    ),
    format: str = typer.Option(
        "ascii", "--format", "-f", help="Output format (ascii, mermaid)"
    ),
    structure: str = typer.Option(
        "three-act", "--structure", "-s", help="Structure type"
    ),
    project: Optional[Path] = typer.Option(None, "--project", "-p"),
) -> None:
    """Generate story arc visualization.

    Examples:
        # ASCII art visualization
        ingestforge lit arc visualize "Hamlet"

        # Mermaid diagram
        ingestforge lit arc visualize "Hamlet" --format mermaid

        # Save to file
        ingestforge lit arc visualize "1984" -o arc.png
    """
    cmd = ArcCommand()
    exit_code = cmd.execute(work, project, output, structure, visualize=True)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for structure analysis
def analyze_command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    structure: str = typer.Option(
        "three-act", "--structure", "-s", help="Structure type"
    ),
    project: Optional[Path] = typer.Option(None, "--project", "-p"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """Analyze story structure in detail.

    Examples:
        # Analyze with hero's journey
        ingestforge lit arc analyze "Lord of the Rings" --structure hero-journey
    """
    cmd = ArcCommand()
    exit_code = cmd.execute(work, project, output, structure, visualize=False)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
