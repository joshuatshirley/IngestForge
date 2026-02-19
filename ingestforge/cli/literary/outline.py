"""Outline command - Generate structural outline of literary work.

Analyzes narrative structure and generates a comprehensive outline
showing plot progression, key events, and structural elements.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import typer
from rich.panel import Panel
from rich.markdown import Markdown

from ingestforge.cli.literary.base import LiteraryCommand


class OutlineCommand(LiteraryCommand):
    """Generate structural outline of literary work."""

    def execute(
        self,
        work: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        detailed: bool = False,
    ) -> int:
        """Generate structural outline of a literary work.

        Args:
            work: Name of the literary work
            project: Project directory
            output: Output file for outline (optional)
            detailed: Include detailed scene-by-scene breakdown

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_work_name(work)

            # Initialize with storage
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Search for structural context
            chunks = self._search_for_structure(ctx["storage"], work)

            if not chunks:
                self._handle_no_context(work)
                return 0

            # Generate outline
            outline = self._generate_outline(llm_client, work, chunks, detailed)

            # Display results
            self._display_outline(work, outline)

            # Save to file if requested
            if output:
                self._save_to_file(output, work, outline)

            return 0

        except Exception as e:
            return self.handle_error(e, "Outline generation failed")

    def _search_for_structure(self, storage: Any, work: str) -> list[Any]:
        """Search for structural context.

        Args:
            storage: ChunkRepository instance
            work: Name of literary work

        Returns:
            List of relevant chunks
        """
        from ingestforge.cli.core import ProgressManager

        query = f"{work} plot structure events chapters"

        return ProgressManager.run_with_spinner(
            lambda: storage.search(query, k=25),
            f"Searching for structural information about '{work}'...",
            "Context retrieved",
        )

    def _handle_no_context(self, work: str) -> None:
        """Handle case where no context found.

        Args:
            work: Name of literary work
        """
        self.print_warning(f"No context found for '{work}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {work}\n"
            "  2. Using 'lit gather' to fetch Wikipedia pages\n"
            "  3. Checking the work name spelling"
        )

    def _generate_outline(
        self, llm_client: Any, work: str, chunks: list, detailed: bool
    ) -> str:
        """Generate structural outline using LLM.

        Args:
            llm_client: LLM provider instance
            work: Name of literary work
            chunks: Context chunks
            detailed: Whether to include detailed breakdown

        Returns:
            Generated outline text
        """
        # Build context and prompt
        context = self.format_context_for_prompt(chunks)
        prompt = self._build_outline_prompt(work, context, detailed)

        # Generate outline
        task_desc = "detailed outline" if detailed else "outline"
        return self.generate_analysis(llm_client, prompt, task_desc)

    def _build_outline_prompt(self, work: str, context: str, detailed: bool) -> str:
        """Build prompt for outline generation.

        Args:
            work: Name of literary work
            context: Context from knowledge base
            detailed: Whether to include detailed breakdown

        Returns:
            Formatted prompt
        """
        if detailed:
            additional_instructions = (
                "Generate a detailed structural outline including:\n"
                "1. Major divisions (acts, parts, books)\n"
                "2. Chapter or section summaries\n"
                "3. Key events and plot points\n"
                "4. Character introductions\n"
                "5. Narrative structure elements (exposition, rising action, etc.)\n"
                "6. Important transitions and turning points\n\n"
                "Use hierarchical structure with clear headings."
            )
        else:
            additional_instructions = (
                "Generate a high-level structural outline including:\n"
                "1. Major divisions (acts, parts, sections)\n"
                "2. Key plot developments\n"
                "3. Narrative arc (exposition, climax, resolution)\n"
                "4. Major turning points\n\n"
                "Keep it concise and hierarchical."
            )

        return self.build_literary_prompt(
            "structural outline", work, context, additional_instructions
        )

    def _display_outline(self, work: str, outline: str) -> None:
        """Display structural outline.

        Args:
            work: Name of literary work
            outline: Generated outline text
        """
        self.console.print()

        panel = Panel(
            Markdown(outline),
            title=f"[bold blue]Structural Outline: {work}[/bold blue]",
            border_style="blue",
        )

        self.console.print(panel)

    def _save_to_file(self, output: Path, work: str, outline: str) -> None:
        """Save outline to file.

        Args:
            output: Output file path
            work: Name of literary work
            outline: Outline text
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            content = (
                f"# Structural Outline: {work}\n\n"
                f"Generated: {timestamp}\n"
                f"Tool: IngestForge Literary Analysis\n\n"
                f"---\n\n"
                f"{outline}\n"
            )

            output.write_text(content, encoding="utf-8")
            self.print_success(f"Outline saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save to file: {e}")


# Typer command wrapper
def command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for outline"
    ),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Include detailed scene-by-scene breakdown"
    ),
) -> None:
    """Generate a structural outline of a literary work.

    Analyzes narrative structure and creates a hierarchical outline
    showing plot progression, key events, and structural elements.

    Requires documents about the work to be ingested first.

    Examples:
        # Generate basic outline
        ingestforge lit outline "Pride and Prejudice"

        # Generate detailed outline
        ingestforge lit outline "The Odyssey" --detailed

        # Save to file
        ingestforge lit outline "1984" --output 1984_outline.md

        # Specific project
        ingestforge lit outline "Hamlet" -p /path/to/project
    """
    cmd = OutlineCommand()
    exit_code = cmd.execute(work, project, output, detailed)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
