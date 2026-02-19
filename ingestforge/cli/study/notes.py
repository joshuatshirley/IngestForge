"""Notes command - Generate organized study notes.

Creates comprehensive study notes organized by topic and concept.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.panel import Panel
from rich.markdown import Markdown

from ingestforge.cli.study.base import StudyCommand


class NotesCommand(StudyCommand):
    """Generate study notes from knowledge base."""

    def execute(
        self,
        topic: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        format_type: str = "detailed",
    ) -> int:
        """
        Generate study notes about a topic.

        Rule #4: Function under 60 lines
        """
        try:
            # Validate inputs
            self.validate_topic(topic)
            self.validate_format(format_type)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Search for context
            chunks = self.search_topic_context(ctx["storage"], topic, k=40)

            if not chunks:
                self._handle_no_context(topic)
                return 0

            # Generate notes
            notes_data = self._generate_notes(llm_client, topic, chunks, format_type)

            if not notes_data:
                self.print_error("Failed to generate notes")
                return 1

            # Display notes
            self._display_notes(notes_data, topic)

            # Save to file if requested
            if output:
                self._save_notes(output, notes_data)

            return 0

        except Exception as e:
            return self.handle_error(e, "Notes generation failed")

    def validate_format(self, format_type: str) -> None:
        """Validate note format type.

        Args:
            format_type: Format string

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        valid_formats = ["outline", "detailed", "summary", "cornell"]

        if format_type.lower() not in valid_formats:
            raise typer.BadParameter(
                f"Invalid format '{format_type}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )

    def _handle_no_context(self, topic: str) -> None:
        """Handle case where no context found.

        Args:
            topic: Topic that was searched
        """
        self.print_warning(f"No context found for topic: '{topic}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {topic}\n"
            "  2. Using a broader search term\n"
            "  3. Checking spelling"
        )

    def _generate_notes(
        self,
        llm_client: Any,
        topic: str,
        chunks: list,
        format_type: str,
    ) -> Optional[Dict[str, Any]]:
        """Generate notes using LLM.

        Args:
            llm_client: LLM provider instance
            topic: Topic for notes
            chunks: Context chunks
            format_type: Note format type

        Returns:
            Notes data dict or None
        """
        # Build context
        context = self.format_context_for_prompt(chunks, max_length=6000)

        # Build prompt
        prompt = self._build_notes_prompt(topic, format_type, context)

        # Generate notes
        response = self.generate_with_llm(llm_client, prompt, f"{format_type} notes")

        # Parse JSON response
        notes_data = self.parse_json_response(response)

        if not notes_data:
            # Fallback: use raw response as notes
            notes_data = {
                "topic": topic,
                "format": format_type,
                "content": response,
                "sections": [],
                "key_points": [],
            }

        return notes_data

    def _build_notes_prompt(self, topic: str, format_type: str, context: str) -> str:
        """Build prompt for notes generation.

        Args:
            topic: Topic for notes
            format_type: Note format
            context: Context from chunks

        Returns:
            Formatted prompt
        """
        format_instructions = self._get_format_instructions(format_type)

        prompt_parts = [
            f"Generate {format_type} study notes about: {topic}\n",
            "\nContext from knowledge base:\n",
            context,
            f"\n\nFormat: {format_instructions}\n",
            "\nGenerate notes in this JSON format:\n",
            "{\n",
            f'  "topic": "{topic}",\n',
            f'  "format": "{format_type}",\n',
            '  "content": "main notes content in markdown",\n',
            '  "sections": [\n',
            "    {\n",
            '      "title": "section title",\n',
            '      "content": "section content",\n',
            '      "key_points": ["point 1", "point 2"]\n',
            "    },\n",
            "    ...\n",
            "  ],\n",
            '  "key_points": ["overall key point 1", ...],\n',
            '  "summary": "brief summary",\n',
            '  "next_steps": ["suggested next step 1", ...]\n',
            "}\n",
            "\nRequirements:\n",
            "- Base all content on provided context\n",
            f"- Follow {format_type} format strictly\n",
            "- Include 3-7 main sections\n",
            "- Extract 5-10 key points per section\n",
            "- Provide actionable next steps\n",
            "- Use clear, student-friendly language\n",
            "\nReturn ONLY valid JSON, no additional text.",
        ]

        return "".join(prompt_parts)

    def _get_format_instructions(self, format_type: str) -> str:
        """Get instructions for note format.

        Args:
            format_type: Note format

        Returns:
            Format instructions
        """
        instructions = {
            "outline": "Hierarchical bullet points with main topics and subtopics",
            "detailed": "Comprehensive notes with full explanations and examples",
            "summary": "Concise summary highlighting only the most important points",
            "cornell": "Cornell method with cues, notes, and summary sections",
        }

        return instructions.get(format_type, instructions["detailed"])

    def _display_notes(self, notes_data: Dict[str, Any], topic: str) -> None:
        """Display notes.

        Args:
            notes_data: Notes data dict
            topic: Topic name
        """
        self.console.print()

        # Title
        format_type = notes_data.get("format", "detailed")
        title = f"[bold cyan]{format_type.title()} Notes: {topic}[/bold cyan]"
        self.console.print(Panel(title, border_style="cyan"))

        self.console.print()

        # Main content
        content = notes_data.get("content", "")
        if content:
            self.console.print(Markdown(content))
            self.console.print()

        # Sections
        sections = notes_data.get("sections", [])
        if sections:
            self._display_sections(sections)

        # Key points
        key_points = notes_data.get("key_points", [])
        if key_points:
            self.console.print("[bold yellow]Key Points:[/bold yellow]")
            for point in key_points:
                self.console.print(f"  • {point}")
            self.console.print()

        # Summary
        summary = notes_data.get("summary", "")
        if summary:
            self.console.print("[bold green]Summary:[/bold green]")
            self.console.print(f"  {summary}")
            self.console.print()

        # Next steps
        next_steps = notes_data.get("next_steps", [])
        if next_steps:
            self.console.print("[bold blue]Next Steps:[/bold blue]")
            for idx, step in enumerate(next_steps, 1):
                self.console.print(f"  {idx}. {step}")
            self.console.print()

    def _display_sections(self, sections: List[Dict]) -> None:
        """Display note sections.

        Args:
            sections: List of section dicts
        """
        for section in sections:
            title = section.get("title", "")
            section_content = section.get("content", "")
            section_points = section.get("key_points", [])

            if title:
                self.console.print(f"\n[bold cyan]{title}[/bold cyan]")

            if section_content:
                self.console.print(section_content)

            if section_points:
                self.console.print("\n[dim]Key points:[/dim]")
                for point in section_points:
                    self.console.print(f"  • [dim]{point}[/dim]")

            self.console.print()

    def _save_notes(self, output: Path, notes_data: Dict[str, Any]) -> None:
        """Save notes to file.

        Args:
            output: Output path
            notes_data: Notes data
        """
        # Determine format from extension
        if output.suffix.lower() == ".md":
            self._save_markdown_notes(output, notes_data)
        else:
            # Default to JSON
            self.save_json_output(output, notes_data, f"Notes saved to: {output}")

    def _save_markdown_notes(self, output: Path, notes_data: Dict[str, Any]) -> None:
        """Save notes as markdown.

        Args:
            output: Output path
            notes_data: Notes data
        """
        topic = notes_data.get("topic", "Topic")
        format_type = notes_data.get("format", "detailed")
        content = notes_data.get("content", "")
        sections = notes_data.get("sections", [])
        key_points = notes_data.get("key_points", [])
        summary = notes_data.get("summary", "")
        next_steps = notes_data.get("next_steps", [])

        md_parts = [
            f"# {topic} - Study Notes\n",
            f"\n**Format**: {format_type}\n",
            "\n---\n\n",
        ]

        # Main content
        if content:
            md_parts.append(f"{content}\n\n")

        # Sections
        if sections:
            md_parts.append("## Detailed Notes\n\n")
            for section in sections:
                self._append_section_markdown(md_parts, section)

        # Key points
        if key_points:
            md_parts.append("## Key Points\n\n")
            for point in key_points:
                md_parts.append(f"- {point}\n")
            md_parts.append("\n")

        # Summary
        if summary:
            md_parts.append(f"## Summary\n\n{summary}\n\n")

        # Next steps
        if next_steps:
            md_parts.append("## Next Steps\n\n")
            for idx, step in enumerate(next_steps, 1):
                md_parts.append(f"{idx}. {step}\n")

        markdown_content = "".join(md_parts)
        output.write_text(markdown_content, encoding="utf-8")
        self.print_success(f"Notes saved to: {output}")

    def _append_section_markdown(
        self, md_parts: List[str], section: Dict[str, Any]
    ) -> None:
        """Append section markdown to parts list.

        Args:
            md_parts: List to append to
            section: Section dictionary
        """
        section_title = section.get("title", "")
        section_content = section.get("content", "")
        section_points = section.get("key_points", [])

        if section_title:
            md_parts.append(f"### {section_title}\n\n")

        if section_content:
            md_parts.append(f"{section_content}\n\n")

        if section_points:
            md_parts.append("**Key Points:**\n")
            for point in section_points:
                md_parts.append(f"- {point}\n")
            md_parts.append("\n")


# Typer command wrapper
def command(
    topic: str = typer.Argument(..., help="Topic to generate notes for"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (.md or .json)"
    ),
    format_type: str = typer.Option(
        "detailed",
        "--format",
        "-f",
        help="Note format (outline/detailed/summary/cornell)",
    ),
) -> None:
    """Generate organized study notes for a topic.

    Creates comprehensive notes from your knowledge base in
    various study-friendly formats.

    Note formats:
    - outline: Hierarchical bullet points
    - detailed: Full explanations with examples (default)
    - summary: Concise key points only
    - cornell: Cornell note-taking method

    Examples:
        # Generate detailed notes
        ingestforge study notes "Machine Learning"

        # Outline format
        ingestforge study notes "Biology" --format outline

        # Save to file
        ingestforge study notes "History" -o history_notes.md

        # Cornell method
        ingestforge study notes "Physics" -f cornell

        # Specific project
        ingestforge study notes "Chemistry" -p /path/to/project
    """
    cmd = NotesCommand()
    exit_code = cmd.execute(topic, project, output, format_type)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
