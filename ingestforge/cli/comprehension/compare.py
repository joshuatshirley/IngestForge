"""Compare command - Side-by-side comparison of concepts.

Analyzes similarities and differences between multiple concepts.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from ingestforge.cli.comprehension.base import ComprehensionCommand


class CompareCommand(ComprehensionCommand):
    """Compare concepts side-by-side."""

    def execute(
        self,
        concepts: List[str],
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """
        Compare multiple concepts from the knowledge base.

        Rule #4: Function under 60 lines
        """
        try:
            # Validate inputs (Commandment #7)
            self.validate_concepts(concepts)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Search for context about each concept
            all_chunks = self._gather_concept_contexts(ctx["storage"], concepts)

            if not any(all_chunks.values()):
                self._handle_no_context(concepts)
                return 0

            # Generate comparison
            comparison_data = self._generate_comparison(
                llm_client, concepts, all_chunks
            )

            if not comparison_data:
                self.print_error("Failed to generate comparison")
                return 1

            # Display comparison
            self._display_comparison(comparison_data, concepts)

            # Save to file if requested
            if output:
                self.save_json_output(
                    output, comparison_data, f"Comparison saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Comparison failed")

    def validate_concepts(self, concepts: List[str]) -> None:
        """Validate concept list.

        Args:
            concepts: List of concepts to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if len(concepts) < 2:
            raise typer.BadParameter("Must provide at least 2 concepts to compare")

        if len(concepts) > 5:
            raise typer.BadParameter("Cannot compare more than 5 concepts at once")

        for concept in concepts:
            self.validate_concept(concept)

    def _gather_concept_contexts(
        self, storage: Any, concepts: List[str]
    ) -> Dict[str, list]:
        """Gather context for each concept.

        Args:
            storage: ChunkRepository instance
            concepts: List of concepts

        Returns:
            Dict mapping concept to chunks
        """
        all_chunks = {}

        for concept in concepts:
            chunks = self.search_concept_context(storage, concept, k=15)
            all_chunks[concept] = chunks

        return all_chunks

    def _handle_no_context(self, concepts: List[str]) -> None:
        """Handle case where no context found.

        Args:
            concepts: Concepts that were searched
        """
        self.print_warning(f"No context found for concepts: {', '.join(concepts)}")
        self.print_info(
            "Try:\n"
            "  1. Ingesting documents about these topics\n"
            "  2. Using broader search terms\n"
            "  3. Checking spelling"
        )

    def _generate_comparison(
        self,
        llm_client: Any,
        concepts: List[str],
        all_chunks: Dict[str, list],
    ) -> Optional[Dict[str, Any]]:
        """Generate comparison using LLM.

        Args:
            llm_client: LLM provider instance
            concepts: List of concepts to compare
            all_chunks: Context chunks for each concept

        Returns:
            Comparison data dict or None if failed
        """
        # Build combined context
        context = self._build_combined_context(concepts, all_chunks)

        # Build prompt
        prompt = self._build_comparison_prompt(concepts, context)

        # Generate comparison
        response = self.generate_with_llm(
            llm_client, prompt, f"comparison of {len(concepts)} concepts"
        )

        # Parse JSON response
        comparison_data = self.parse_json_response(response)

        if not comparison_data:
            # Fallback: simple structure
            comparison_data = {
                "concepts": concepts,
                "similarities": [],
                "differences": [],
                "summary": response[:500],
            }

        return comparison_data

    def _build_combined_context(
        self, concepts: List[str], all_chunks: Dict[str, list]
    ) -> str:
        """Build combined context from all concepts.

        Args:
            concepts: List of concepts
            all_chunks: Chunks for each concept

        Returns:
            Combined context string
        """
        context_parts = []

        for concept in concepts:
            chunks = all_chunks.get(concept, [])
            if chunks:
                concept_context = self.format_context_for_prompt(
                    chunks, max_length=1500
                )
                context_parts.append(
                    f"--- Context for '{concept}' ---\n{concept_context}\n"
                )

        return "\n".join(context_parts)

    def _build_comparison_prompt(self, concepts: List[str], context: str) -> str:
        """Build prompt for comparison generation.

        Args:
            concepts: List of concepts to compare
            context: Combined context

        Returns:
            Formatted prompt
        """
        concepts_str = ", ".join(f'"{c}"' for c in concepts)

        prompt_parts = [
            f"Compare and contrast these concepts: {concepts_str}\n",
            "\nContext from knowledge base:\n",
            context,
            "\n\nGenerate comparison in this JSON format:\n",
            "{\n",
            f'  "concepts": [{concepts_str}],\n',
            '  "similarities": [\n',
            '    {"aspect": "aspect name", "description": "how they are similar"},\n',
            "    ...\n",
            "  ],\n",
            '  "differences": [\n',
            '    {"aspect": "aspect name", "concept1": "...", "concept2": "...", ...},\n',
            "    ...\n",
            "  ],\n",
            '  "use_cases": {\n',
            '    "concept1": ["use case 1", ...],\n',
            "    ...\n",
            "  },\n",
            '  "strengths_weaknesses": {\n',
            '    "concept1": {"strengths": [...], "weaknesses": [...]},\n',
            "    ...\n",
            "  },\n",
            '  "summary": "brief comparison summary"\n',
            "}\n",
            "\nRequirements:\n",
            "- Base comparison on provided context\n",
            "- Identify 3-5 key similarities\n",
            "- Identify 3-5 key differences\n",
            "- List use cases for each concept\n",
            "- Note strengths and weaknesses\n",
            "- Provide clear summary\n",
            "\nReturn ONLY valid JSON, no additional text.",
        ]

        return "".join(prompt_parts)

    def _display_comparison(
        self, comparison_data: Dict[str, Any], concepts: List[str]
    ) -> None:
        """Display comparison.

        Args:
            comparison_data: Comparison data dict
            concepts: List of concepts
        """
        self.console.print()

        # Title
        title = f"[bold cyan]Comparison: {' vs '.join(concepts)}[/bold cyan]"
        self.console.print(Panel(title, border_style="cyan"))

        self.console.print()

        # Summary
        summary = comparison_data.get("summary", "")
        if summary:
            self.console.print("[bold]Summary:[/bold]")
            self.console.print(Markdown(summary))
            self.console.print()

        # Similarities
        similarities = comparison_data.get("similarities", [])
        if similarities:
            self.console.print("[bold green]Similarities:[/bold green]")
            for sim in similarities:
                aspect = sim.get("aspect", "")
                desc = sim.get("description", "")
                self.console.print(f"  • [yellow]{aspect}:[/yellow] {desc}")
            self.console.print()

        # Differences
        differences = comparison_data.get("differences", [])
        if differences:
            self._display_differences_table(differences, concepts)

        # Use cases
        use_cases = comparison_data.get("use_cases", {})
        if use_cases:
            self._display_use_cases(use_cases)

        # Strengths and weaknesses
        strengths_weaknesses = comparison_data.get("strengths_weaknesses", {})
        if strengths_weaknesses:
            self._display_strengths_weaknesses(strengths_weaknesses)

    def _display_differences_table(
        self, differences: List[Dict], concepts: List[str]
    ) -> None:
        """Display differences in table format.

        Args:
            differences: List of difference dicts
            concepts: List of concepts
        """
        self.console.print("[bold red]Differences:[/bold red]")

        table = Table(show_header=True, show_lines=True)
        table.add_column("Aspect", style="yellow", width=20)

        for concept in concepts:
            table.add_column(concept, style="cyan", width=30)

        for diff in differences:
            aspect = diff.get("aspect", "")
            row = [aspect]

            for idx, concept in enumerate(concepts, 1):
                key = f"concept{idx}" if len(concepts) == 2 else concept
                value = diff.get(key, "-")
                row.append(value)

            table.add_row(*row)

        self.console.print(table)
        self.console.print()

    def _display_use_cases(self, use_cases: Dict[str, List[str]]) -> None:
        """Display use cases for each concept.

        Args:
            use_cases: Dict mapping concept to use cases
        """
        self.console.print("[bold blue]Use Cases:[/bold blue]")

        for concept, cases in use_cases.items():
            self.console.print(f"\n  [yellow]{concept}:[/yellow]")
            for case in cases:
                self.console.print(f"    • {case}")

        self.console.print()

    def _display_strengths_weaknesses(
        self, strengths_weaknesses: Dict[str, Dict[str, List[str]]]
    ) -> None:
        """Display strengths and weaknesses.

        Args:
            strengths_weaknesses: Dict with strengths/weaknesses per concept
        """
        self.console.print("[bold magenta]Strengths & Weaknesses:[/bold magenta]")

        for concept, sw in strengths_weaknesses.items():
            self.console.print(f"\n  [yellow]{concept}:[/yellow]")

            strengths = sw.get("strengths", [])
            if strengths:
                self.console.print("    [green]Strengths:[/green]")
                for strength in strengths:
                    self.console.print(f"      ✓ {strength}")

            weaknesses = sw.get("weaknesses", [])
            if weaknesses:
                self.console.print("    [red]Weaknesses:[/red]")
                for weakness in weaknesses:
                    self.console.print(f"      ✗ {weakness}")

        self.console.print()


# Typer command wrapper
def command(
    concepts: List[str] = typer.Argument(
        ..., help="Concepts to compare (2-5 concepts)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON format)"
    ),
) -> None:
    """Compare concepts side-by-side.

    Analyzes similarities, differences, use cases, and trade-offs
    between 2-5 concepts from your knowledge base.

    Examples:
        # Compare two concepts
        ingestforge comprehension compare "Python" "JavaScript"

        # Compare three concepts
        ingestforge comprehension compare "REST" "GraphQL" "gRPC"

        # Save to file
        ingestforge comprehension compare "SQL" "NoSQL" -o comparison.json

        # Specific project
        ingestforge comprehension compare "TCP" "UDP" -p /path/to/project
    """
    cmd = CompareCommand()
    exit_code = cmd.execute(concepts, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
