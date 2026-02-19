"""Explain command - Provide clear explanations of concepts.

Generates both simple (ELI5) and detailed expert explanations.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.markdown import Markdown

from ingestforge.cli.comprehension.base import ComprehensionCommand


class ExplainCommand(ComprehensionCommand):
    """Provide clear explanations of concepts."""

    def execute(
        self,
        concept: str,
        mode: str = "detailed",
        project: Optional[Path] = None,
        output: Optional[Path] = None,
    ) -> int:
        """
        Explain a concept from the knowledge base.

        Rule #4: Function under 60 lines
        """
        try:
            # Validate inputs (Commandment #7)
            self.validate_concept(concept)
            self.validate_mode(mode)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Search for context about the concept
            chunks = self.search_concept_context(ctx["storage"], concept, k=30)

            if not chunks:
                self._handle_no_context(concept)
                return 0

            # Generate explanation
            explanation_data = self._generate_explanation(
                llm_client, concept, chunks, mode
            )

            if not explanation_data:
                self.print_error("Failed to generate explanation")
                return 1

            # Display explanation
            self._display_explanation(explanation_data, concept, mode)

            # Save to file if requested
            if output:
                self.save_json_output(
                    output, explanation_data, f"Explanation saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Explanation failed")

    def validate_mode(self, mode: str) -> None:
        """Validate explanation mode.

        Args:
            mode: Mode string to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        valid_modes = ["simple", "eli5", "detailed", "expert"]

        if mode.lower() not in valid_modes:
            raise typer.BadParameter(
                f"Invalid mode '{mode}'. " f"Must be one of: {', '.join(valid_modes)}"
            )

    def _handle_no_context(self, concept: str) -> None:
        """Handle case where no context found.

        Args:
            concept: Concept that was searched
        """
        self.print_warning(f"No context found for concept: '{concept}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {concept}\n"
            "  2. Using a broader search term\n"
            "  3. Checking spelling"
        )

    def _generate_explanation(
        self,
        llm_client: Any,
        concept: str,
        chunks: list,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate explanation using LLM.

        Args:
            llm_client: LLM provider instance
            concept: Concept to explain
            chunks: Context chunks
            mode: Explanation mode

        Returns:
            Explanation data dict or None if failed
        """
        # Build context and prompt
        context = self.format_context_for_prompt(chunks)
        prompt = self._build_explanation_prompt(concept, mode, context)

        # Generate explanation
        response = self.generate_with_llm(llm_client, prompt, f"{mode} explanation")

        # Parse JSON response
        explanation_data = self.parse_json_response(response)

        if not explanation_data:
            # Fallback: use raw response
            explanation_data = {
                "concept": concept,
                "mode": mode,
                "explanation": response,
                "key_points": [],
                "examples": [],
            }

        return explanation_data

    def _build_explanation_prompt(self, concept: str, mode: str, context: str) -> str:
        """Build prompt for explanation generation.

        Args:
            concept: Concept to explain
            mode: Explanation mode
            context: Context from knowledge base

        Returns:
            Formatted prompt
        """
        mode_instructions = self._get_mode_instructions(mode)

        prompt_parts = [
            f"Explain the concept: {concept}\n",
            f"\nMode: {mode}\n",
            f"{mode_instructions}\n",
            "\nContext from knowledge base:\n",
            context,
            "\n\nGenerate explanation in this JSON format:\n",
            "{\n",
            '  "concept": "concept name",\n',
            '  "mode": "explanation mode",\n',
            '  "explanation": "clear explanation",\n',
            '  "key_points": ["point 1", "point 2", ...],\n',
            '  "examples": ["example 1", "example 2", ...],\n',
            '  "related_concepts": ["concept 1", "concept 2", ...],\n',
            '  "common_misconceptions": ["misconception 1", ...]\n',
            "}\n",
            "\nRequirements:\n",
            "- Base explanation on provided context\n",
            f"- Follow {mode} style strictly\n",
            "- Include 3-5 key points\n",
            "- Provide 2-3 concrete examples\n",
            "- List related concepts\n",
            "- Address common misconceptions\n",
            "\nReturn ONLY valid JSON, no additional text.",
        ]

        return "".join(prompt_parts)

    def _get_mode_instructions(self, mode: str) -> str:
        """Get instructions for specific explanation mode.

        Args:
            mode: Explanation mode

        Returns:
            Instructions string
        """
        instructions = {
            "simple": "Use simple language, short sentences, avoid jargon",
            "eli5": "Explain like I'm 5 years old - use analogies, simple words, friendly tone",
            "detailed": "Provide comprehensive explanation with technical details, assume intermediate knowledge",
            "expert": "Technical expert-level explanation with precise terminology, assumptions, and nuances",
        }

        return instructions.get(mode, instructions["detailed"])

    def _display_explanation(
        self, explanation_data: Dict[str, Any], concept: str, mode: str
    ) -> None:
        """Display explanation.

        Args:
            explanation_data: Explanation data dict
            concept: Concept name
            mode: Explanation mode
        """
        self.console.print()

        # Title
        title = f"[bold cyan]Explanation: {concept}[/bold cyan] ({mode} mode)"
        self.console.print(Panel(title, border_style="cyan"))

        self.console.print()

        # Main explanation
        explanation = explanation_data.get("explanation", "No explanation available")
        self.console.print(Markdown(explanation))

        self.console.print()

        # Key points
        key_points = explanation_data.get("key_points", [])
        if key_points:
            self.console.print("[bold yellow]Key Points:[/bold yellow]")
            for point in key_points:
                self.console.print(f"  • {point}")
            self.console.print()

        # Examples
        examples = explanation_data.get("examples", [])
        if examples:
            self.console.print("[bold green]Examples:[/bold green]")
            for idx, example in enumerate(examples, 1):
                self.console.print(f"  {idx}. {example}")
            self.console.print()

        # Related concepts
        related = explanation_data.get("related_concepts", [])
        if related:
            self.console.print("[bold blue]Related Concepts:[/bold blue]")
            self.console.print(f"  {', '.join(related)}")
            self.console.print()

        # Common misconceptions
        misconceptions = explanation_data.get("common_misconceptions", [])
        if misconceptions:
            self.console.print("[bold red]Common Misconceptions:[/bold red]")
            for misconception in misconceptions:
                self.console.print(f"  ⚠ {misconception}")
            self.console.print()


# Typer command wrapper
def command(
    concept: str = typer.Argument(..., help="Concept to explain"),
    mode: str = typer.Option(
        "detailed",
        "--mode",
        "-m",
        help="Explanation mode (simple/eli5/detailed/expert)",
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON format)"
    ),
) -> None:
    """Explain a concept from your knowledge base.

    Generates clear, context-aware explanations in different styles.

    Explanation modes:
    - simple: Plain language, no jargon
    - eli5: Explain like I'm 5 (analogies, simple words)
    - detailed: Comprehensive with technical details (default)
    - expert: Technical expert-level explanation

    Examples:
        # Detailed explanation
        ingestforge comprehension explain "machine learning"

        # Simple explanation
        ingestforge comprehension explain "quantum computing" --mode simple

        # ELI5 style
        ingestforge comprehension explain "blockchain" --mode eli5

        # Save to file
        ingestforge comprehension explain "neural networks" -o explanation.json

        # Specific project
        ingestforge comprehension explain "photosynthesis" -p /path/to/project
    """
    cmd = ExplainCommand()
    exit_code = cmd.execute(concept, mode, project, output)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
