"""Flashcards command - Generate flashcard sets for memorization.

Creates front/back flashcards for active recall and spaced repetition.
Supports multiple export formats: Anki CSV, Quizlet, and Markdown.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
#1 (Simple Control Flow), and #9 (Type Hints).
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Optional, Dict, Any, List
import typer
from rich.panel import Panel
from rich.table import Table

from ingestforge.cli.study.base import StudyCommand


# Valid formats for flashcard export
VALID_FORMATS = ["anki", "quizlet", "markdown", "json"]
VALID_CARD_TYPES = ["definition", "concept", "fact", "process"]


class FlashcardsCommand(StudyCommand):
    """Generate flashcard sets for memorization."""

    def execute(
        self,
        topic: str,
        count: int = 10,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        card_type: str = "definition",
        output_format: str = "anki",
        include_sources: bool = True,
    ) -> int:
        """
        Generate flashcards about a topic.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            topic: Topic to generate flashcards for
            count: Number of flashcards
            project: Project directory
            output: Output file path
            card_type: Type of flashcards
            output_format: Export format (anki, quizlet, markdown, json)
            include_sources: Include source citations
        """
        try:
            self._validate_inputs(topic, count, card_type, output_format)
            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self.search_topic_context(ctx["storage"], topic, k=30)
            if not chunks:
                self._handle_no_context(topic)
                return 0

            flashcard_data = self._generate_flashcards(
                llm_client, topic, chunks, count, card_type, include_sources
            )

            if not flashcard_data:
                self.print_error("Failed to generate flashcards")
                return 1

            self._display_flashcards(flashcard_data, topic)

            if output:
                self._save_flashcards(output, flashcard_data, output_format)

            return 0

        except Exception as e:
            return self.handle_error(e, "Flashcard generation failed")

    def _validate_inputs(
        self, topic: str, count: int, card_type: str, output_format: str
    ) -> None:
        """Validate all inputs.

        Rule #4: Function <60 lines
        """
        self.validate_topic(topic)
        self.validate_count(count, min_val=1, max_val=100)
        self._validate_card_type(card_type)
        self._validate_format(output_format)

    def _validate_card_type(self, card_type: str) -> None:
        """Validate card type.

        Args:
            card_type: Card type string to validate

        Raises:
            typer.BadParameter: If invalid
        """
        if card_type.lower() not in VALID_CARD_TYPES:
            raise typer.BadParameter(
                f"Invalid card type '{card_type}'. "
                f"Must be one of: {', '.join(VALID_CARD_TYPES)}"
            )

    def _validate_format(self, output_format: str) -> None:
        """Validate output format.

        Args:
            output_format: Format string to validate

        Raises:
            typer.BadParameter: If invalid
        """
        if output_format.lower() not in VALID_FORMATS:
            raise typer.BadParameter(
                f"Invalid format '{output_format}'. "
                f"Must be one of: {', '.join(VALID_FORMATS)}"
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
            "  2. Using a broader topic term\n"
            "  3. Checking spelling"
        )

    def _generate_flashcards(
        self,
        llm_client: Any,
        topic: str,
        chunks: list,
        count: int,
        card_type: str,
        include_sources: bool,
    ) -> Optional[Dict[str, Any]]:
        """Generate flashcards using LLM.

        Rule #4: Function <60 lines
        """
        context = self.format_context_for_prompt(chunks)
        source_info = self._extract_sources(chunks) if include_sources else []

        prompt = self._build_flashcard_prompt(
            topic, count, card_type, context, include_sources
        )

        response = self.generate_with_llm(llm_client, prompt, f"{count} flashcards")
        flashcard_data = self.parse_json_response(response)

        if not flashcard_data:
            flashcard_data = self._create_fallback_flashcards(topic, card_type, count)

        # Add source metadata
        if include_sources and source_info:
            flashcard_data["sources"] = source_info

        return flashcard_data

    def _extract_sources(self, chunks: list) -> List[Dict[str, str]]:
        """Extract source information from chunks.

        Args:
            chunks: List of chunks

        Returns:
            List of source dictionaries
        """
        sources = {}
        for chunk in chunks:
            metadata = (
                self.extract_chunk_metadata(chunk)
                if hasattr(self, "extract_chunk_metadata")
                else {}
            )
            source = metadata.get("source", "Unknown")
            if source not in sources:
                sources[source] = {
                    "source": source,
                    "page": metadata.get("page", ""),
                }
        return list(sources.values())

    def extract_chunk_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Extract metadata from chunk.

        Args:
            chunk: Chunk object or dict

        Returns:
            Metadata dictionary
        """
        if isinstance(chunk, dict):
            return chunk.get("metadata", {})
        if hasattr(chunk, "metadata"):
            metadata = chunk.metadata
            if isinstance(metadata, dict):
                return metadata
            return vars(metadata) if metadata else {}
        return {}

    def _build_flashcard_prompt(
        self,
        topic: str,
        count: int,
        card_type: str,
        context: str,
        include_sources: bool,
    ) -> str:
        """Build prompt for flashcard generation.

        Rule #4: Function <60 lines
        """
        card_type_instructions = self._get_card_type_instructions(card_type)

        source_instruction = ""
        if include_sources:
            source_instruction = '      "source": "source document name",\n'

        return f"""Generate {count} {card_type} flashcards about: {topic}

Context from knowledge base:
{context}

Generate flashcards following this JSON format:
{{
  "topic": "{topic}",
  "card_type": "{card_type}",
  "cards": [
    {{
      "front": "question or term",
      "back": "answer or definition",
      "hint": "optional hint",
      "tags": ["tag1", "tag2"],
{source_instruction}    }}
  ]
}}

Requirements:
- Generate exactly {count} flashcards
- Base all content on the provided context
- Follow {card_type} card style: {card_type_instructions}
- Keep front side concise (one question/term)
- Make back side clear and complete
- Add relevant tags for organization
- Include hints where helpful

Return ONLY valid JSON, no additional text."""

    def _get_card_type_instructions(self, card_type: str) -> str:
        """Get instructions for specific card type."""
        instructions = {
            "definition": "Front: term, Back: definition with example",
            "concept": "Front: concept name, Back: explanation and significance",
            "fact": "Front: question, Back: factual answer with context",
            "process": "Front: process name, Back: step-by-step explanation",
        }
        return instructions.get(card_type, "Front: question, Back: answer")

    def _create_fallback_flashcards(
        self, topic: str, card_type: str, count: int
    ) -> Dict[str, Any]:
        """Create fallback flashcard structure."""
        return {
            "topic": topic,
            "card_type": card_type,
            "cards": [
                {
                    "front": f"Sample question {i+1} about {topic}",
                    "back": "Answer would be generated from context",
                    "hint": "",
                    "tags": [topic],
                    "source": "",
                }
                for i in range(min(count, 3))
            ],
        }

    def _save_flashcards(
        self, output: Path, flashcard_data: Dict[str, Any], output_format: str
    ) -> None:
        """Save flashcards in specified format.

        Rule #1: Dictionary dispatch for format selection
        """
        format_handlers = {
            "anki": self._save_anki_format,
            "quizlet": self._save_quizlet_format,
            "markdown": self._save_markdown_format,
            "json": self._save_json_format,
        }

        handler = format_handlers.get(output_format.lower(), self._save_json_format)
        handler(output, flashcard_data)

    def _save_anki_format(self, output: Path, data: Dict[str, Any]) -> None:
        """Save as Anki-compatible CSV.

        Format: Front;Back;Tags
        Uses semicolon as Anki default separator.
        """
        # Ensure .csv extension
        if output.suffix.lower() != ".csv":
            output = output.with_suffix(".csv")

        cards = data.get("cards", [])
        output_buffer = io.StringIO()
        writer = csv.writer(output_buffer, delimiter=";", quoting=csv.QUOTE_ALL)

        # Anki doesn't need headers but we add them for clarity
        writer.writerow(["Front", "Back", "Tags", "Source"])

        for card in cards:
            front = card.get("front", "")
            back = card.get("back", "")
            tags = " ".join(card.get("tags", []))
            source = card.get("source", "")

            # Include hint in back if available
            hint = card.get("hint", "")
            if hint:
                back = f"{back}\n\nHint: {hint}"

            writer.writerow([front, back, tags, source])

        output.write_text(output_buffer.getvalue(), encoding="utf-8")
        self.print_success(f"Anki flashcards saved to: {output}")
        self.print_info("Import into Anki: File > Import, select CSV")

    def _save_quizlet_format(self, output: Path, data: Dict[str, Any]) -> None:
        """Save as Quizlet-compatible format.

        Format: Term TAB Definition (one per line)
        """
        if output.suffix.lower() != ".txt":
            output = output.with_suffix(".txt")

        cards = data.get("cards", [])
        lines = []

        for card in cards:
            front = card.get("front", "").replace("\t", " ").replace("\n", " ")
            back = card.get("back", "").replace("\t", " ").replace("\n", " ")
            lines.append(f"{front}\t{back}")

        output.write_text("\n".join(lines), encoding="utf-8")
        self.print_success(f"Quizlet flashcards saved to: {output}")
        self.print_info("Import into Quizlet: Create > Import")

    def _save_markdown_format(self, output: Path, data: Dict[str, Any]) -> None:
        """Save as Markdown flashcard document."""
        if output.suffix.lower() != ".md":
            output = output.with_suffix(".md")

        topic = data.get("topic", "Flashcards")
        cards = data.get("cards", [])
        sources = data.get("sources", [])

        lines = [
            f"# {topic} - Flashcards",
            "",
            f"**Total Cards:** {len(cards)}",
            f"**Card Type:** {data.get('card_type', 'mixed')}",
            "",
            "---",
            "",
        ]

        for idx, card in enumerate(cards, 1):
            lines.extend(self._format_markdown_card(idx, card))

        # Add sources section
        if sources:
            lines.extend(self._format_sources_section(sources))

        lines.append("\n*Generated by IngestForge*")
        output.write_text("\n".join(lines), encoding="utf-8")
        self.print_success(f"Markdown flashcards saved to: {output}")

    def _format_markdown_card(self, idx: int, card: Dict[str, Any]) -> List[str]:
        """Format a single card as markdown."""
        front = card.get("front", "")
        back = card.get("back", "")
        hint = card.get("hint", "")
        tags = card.get("tags", [])
        source = card.get("source", "")

        lines = [
            f"## Card {idx}",
            "",
            f"**Q:** {front}",
            "",
            f"**A:** {back}",
            "",
        ]

        if hint:
            lines.append(f"*Hint: {hint}*")
            lines.append("")

        if tags:
            lines.append(f"Tags: {', '.join(tags)}")

        if source:
            lines.append(f"Source: {source}")

        lines.extend(["", "---", ""])
        return lines

    def _format_sources_section(self, sources: List[Dict[str, str]]) -> List[str]:
        """Format sources section for markdown."""
        lines = ["", "## Sources", ""]
        for idx, src in enumerate(sources, 1):
            source_name = src.get("source", "Unknown")
            page = src.get("page", "")
            page_info = f" (p. {page})" if page else ""
            lines.append(f"{idx}. {source_name}{page_info}")
        return lines

    def _save_json_format(self, output: Path, data: Dict[str, Any]) -> None:
        """Save as JSON format."""
        self.save_json_output(output, data, f"JSON flashcards saved to: {output}")

    def _display_flashcards(self, flashcard_data: Dict[str, Any], topic: str) -> None:
        """Display flashcards in table format."""
        self.console.print()

        cards = flashcard_data.get("cards", [])
        self.print_info(f"Generated {len(cards)} flashcards for: {topic}")
        self.console.print()

        table = Table(title=f"Flashcard Set: {topic}", show_lines=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Front", style="yellow", width=35)
        table.add_column("Back", style="green", width=45)
        table.add_column("Tags", style="blue", width=15)

        for idx, card in enumerate(cards, 1):
            front = card.get("front", "")[:80]
            back = card.get("back", "")[:120]
            tags = ", ".join(card.get("tags", [])[:3])
            table.add_row(str(idx), front, back, tags)

        self.console.print(table)
        self._display_study_tips()

    def _display_study_tips(self) -> None:
        """Display study tips for flashcards."""
        self.console.print()
        tips = Panel(
            "**Study Tips:**\n\n"
            "1. Review cards regularly (spaced repetition)\n"
            "2. Try to recall answer before flipping\n"
            "3. Focus on cards you find difficult\n"
            "4. Use tags to organize by topic\n"
            "5. Export to Anki or Quizlet for SRS",
            title="[bold blue]How to Use These Flashcards[/bold blue]",
            border_style="blue",
        )
        self.console.print(tips)


# Create subcommand group for flashcards
flashcards_app = typer.Typer(
    name="flashcards",
    help="Flashcard generation commands",
    add_completion=False,
)


@flashcards_app.command("generate")
def generate_command(
    topic: str = typer.Argument(..., help="Topic to create flashcards for"),
    count: int = typer.Option(
        50, "--count", "-n", help="Number of flashcards to generate"
    ),
    output_format: str = typer.Option(
        "anki", "--format", "-f", help="Output format (anki/quizlet/markdown/json)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for flashcards"
    ),
    card_type: str = typer.Option(
        "definition", "--type", "-t", help="Card type (definition/concept/fact/process)"
    ),
    include_sources: bool = typer.Option(
        True, "--sources/--no-sources", help="Include source citations"
    ),
) -> None:
    """Generate flashcards from knowledge base.

    Creates flashcards optimized for spaced repetition learning.
    Supports multiple export formats for Anki, Quizlet, or plain Markdown.

    Examples:
        # Generate 50 Anki flashcards
        ingestforge study flashcards generate "machine learning" --format anki

        # Generate Quizlet-compatible cards
        ingestforge study flashcards generate "biology" -f quizlet -o cards.txt

        # Markdown format with sources
        ingestforge study flashcards generate "history" -f markdown --sources

        # JSON format for custom processing
        ingestforge study flashcards generate "physics" -f json -o cards.json
    """
    cmd = FlashcardsCommand()
    exit_code = cmd.execute(
        topic, count, project, output, card_type, output_format, include_sources
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Legacy command wrapper for backwards compatibility
def command(
    topic: str = typer.Argument(..., help="Topic to create flashcards for"),
    count: int = typer.Option(
        10, "--count", "-n", help="Number of flashcards to generate"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for flashcards"
    ),
    card_type: str = typer.Option(
        "definition", "--type", "-t", help="Card type (definition/concept/fact/process)"
    ),
    output_format: str = typer.Option(
        "anki", "--format", "-f", help="Output format (anki/quizlet/markdown/json)"
    ),
) -> None:
    """Generate flashcards for memorization and study.

    Creates front/back flashcards optimized for active recall
    and spaced repetition learning.

    Supported formats:
    - anki: CSV format for Anki import
    - quizlet: Tab-separated for Quizlet import
    - markdown: Human-readable markdown document
    - json: JSON format for custom processing

    Card types:
    - definition: Term and definition pairs
    - concept: Conceptual understanding
    - fact: Factual recall questions
    - process: Step-by-step procedures

    Examples:
        # Generate Anki flashcards
        ingestforge study flashcards "Biology" --format anki -o biology.csv

        # Generate Quizlet flashcards
        ingestforge study flashcards "History" --format quizlet -o history.txt

        # Generate Markdown flashcards
        ingestforge study flashcards "Math" --format markdown -o math.md
    """
    cmd = FlashcardsCommand()
    exit_code = cmd.execute(
        topic, count, project, output, card_type, output_format, True
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
