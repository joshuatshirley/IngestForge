"""Character command - Analyze literary characters.

Provides character extraction, tracking, and relationship analysis:
- extract: Extract characters from text using NER
- analyze: Analyze specific character's role and development
- relationships: Generate character relationship graph"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

from ingestforge.cli.literary.base import LiteraryCommand
from ingestforge.cli.literary.models import (
    Appearance,
    Character,
    CharacterProfile,
    Relationship,
    RelationshipGraph,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Character Extractor Class
# ============================================================================


class CharacterExtractor:
    """Extract and track characters from literary text.

    Uses NER to identify PERSON entities and tracks them across chunks.
    Follows Rule #4: All methods <60 lines.
    """

    def __init__(self, min_mentions: int = 2) -> None:
        """Initialize character extractor.

        Args:
            min_mentions: Minimum mentions to be considered a character
        """
        self.min_mentions = min_mentions
        self._entity_extractor: Optional[Any] = None

    def _get_entity_extractor(self) -> Any:
        """Lazy-load entity extractor.

        Returns:
            EntityExtractor instance
        """
        if self._entity_extractor is not None:
            return self._entity_extractor

        try:
            from ingestforge.enrichment.entities import EntityExtractor

            self._entity_extractor = EntityExtractor(use_spacy=True)
        except ImportError:
            logger.warning("EntityExtractor not available, using basic extraction")
            self._entity_extractor = None

        return self._entity_extractor

    def extract_characters(self, text: str) -> List[Character]:
        """Extract characters from text using NER.

        Args:
            text: Text to analyze

        Returns:
            List of Character objects

        Rule #1: Early return for empty text
        Rule #4: Function <60 lines
        """
        if not text.strip():
            return []

        extractor = self._get_entity_extractor()
        if extractor is None:
            return self._extract_basic(text)

        return self._extract_with_ner(text, extractor)

    def _extract_with_ner(self, text: str, extractor: Any) -> List[Character]:
        """Extract characters using NER.

        Args:
            text: Text to analyze
            extractor: EntityExtractor instance

        Returns:
            List of Character objects
        """
        entities = extractor.extract(text)
        persons = entities.get("person", [])

        # Count mentions and deduplicate
        mention_counts: Dict[str, int] = defaultdict(int)
        for person in persons:
            normalized = person.strip().title()
            mention_counts[normalized] += 1

        # Create Character objects for frequent mentions
        characters = []
        for name, count in mention_counts.items():
            if count < self.min_mentions:
                continue
            char = Character(name=name, mention_count=count)
            characters.append(char)

        return sorted(characters, key=lambda c: -c.mention_count)

    def _extract_basic(self, text: str) -> List[Character]:
        """Basic character extraction using capitalization patterns.

        Args:
            text: Text to analyze

        Returns:
            List of Character objects
        """
        import re

        # Match capitalized names (First Last pattern)
        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b"
        matches = re.findall(pattern, text)

        # Count and filter
        mention_counts: Dict[str, int] = defaultdict(int)
        for match in matches:
            mention_counts[match] += 1

        characters = []
        for name, count in mention_counts.items():
            if count < self.min_mentions:
                continue
            char = Character(name=name, mention_count=count)
            characters.append(char)

        return sorted(characters, key=lambda c: -c.mention_count)

    def track_appearances(self, character: str, chunks: List[Any]) -> List[Appearance]:
        """Track character appearances across chunks.

        Args:
            character: Character name to track
            chunks: List of ChunkRecord objects

        Returns:
            List of Appearance objects
        """
        appearances = []
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            content = getattr(chunk, "content", str(chunk))

            if character.lower() not in content.lower():
                continue

            # Calculate position in narrative
            position = i / max(total_chunks - 1, 1)

            # Extract context around mention
            context = self._extract_context(content, character)

            appearance = Appearance(
                chunk_index=i,
                chunk_id=getattr(chunk, "chunk_id", f"chunk_{i}"),
                context=context,
                position=position,
            )
            appearances.append(appearance)

        return appearances

    def _extract_context(self, text: str, character: str, window: int = 100) -> str:
        """Extract context around character mention.

        Args:
            text: Full text
            character: Character name
            window: Context window size

        Returns:
            Context string
        """
        lower_text = text.lower()
        lower_char = character.lower()

        idx = lower_text.find(lower_char)
        if idx == -1:
            return text[:200] if len(text) > 200 else text

        start = max(0, idx - window)
        end = min(len(text), idx + len(character) + window)

        context = text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(text):
            context = context + "..."

        return context

    def analyze_relationships(
        self, characters: List[Character], chunks: List[Any]
    ) -> RelationshipGraph:
        """Analyze relationships between characters.

        Args:
            characters: List of characters to analyze
            chunks: Text chunks for context

        Returns:
            RelationshipGraph with detected relationships
        """
        graph = RelationshipGraph(characters=characters)

        # Build co-occurrence matrix
        char_names = [c.name.lower() for c in characters]
        cooccurrence = self._build_cooccurrence(char_names, chunks)

        # Detect relationships from co-occurrence
        for i, char1 in enumerate(characters):
            for j, char2 in enumerate(characters):
                if i >= j:
                    continue

                count = cooccurrence.get((i, j), 0)
                if count < 2:
                    continue

                rel_type = self._infer_relationship_type(char1, char2, chunks)
                relationship = Relationship(
                    target=char2.name,
                    relationship_type=rel_type,
                    confidence=min(0.9, 0.5 + count * 0.1),
                )
                graph.edges.append(relationship)
                char1.relationships[char2.name] = rel_type

        return graph

    def _build_cooccurrence(
        self, char_names: List[str], chunks: List[Any]
    ) -> Dict[Tuple[int, int], int]:
        """Build character co-occurrence matrix.

        Args:
            char_names: Lowercase character names
            chunks: Text chunks

        Returns:
            Co-occurrence counts by character index pair
        """
        cooccurrence: Dict[Tuple[int, int], int] = defaultdict(int)

        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk)).lower()

            # Find which characters appear in this chunk
            present = []
            for i, name in enumerate(char_names):
                if name in content:
                    present.append(i)

            # Update co-occurrence for all pairs
            for i, idx1 in enumerate(present):
                for idx2 in present[i + 1 :]:
                    key = (min(idx1, idx2), max(idx1, idx2))
                    cooccurrence[key] += 1

        return cooccurrence

    def _infer_relationship_type(
        self, char1: Character, char2: Character, chunks: List[Any]
    ) -> str:
        """Infer relationship type from context.

        Rule #1: Max 3 nesting levels via early continue.

        Args:
            char1: First character
            char2: Second character
            chunks: Text chunks for context

        Returns:
            Relationship type string
        """
        # Keywords for relationship types
        relationship_keywords = {
            "family": [
                "mother",
                "father",
                "son",
                "daughter",
                "brother",
                "sister",
                "parent",
                "child",
                "wife",
                "husband",
                "married",
            ],
            "romantic": ["love", "loves", "beloved", "kiss", "kissed", "heart"],
            "enemy": ["enemy", "hate", "hates", "rival", "fought", "against"],
            "friend": ["friend", "friends", "companion", "ally", "together"],
            "mentor": ["teacher", "taught", "student", "mentor", "guide", "learn"],
        }

        # Check chunks for relationship indicators
        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk)).lower()

            # Skip if both characters not present
            if char1.name.lower() not in content:
                continue
            if char2.name.lower() not in content:
                continue
            detected_type = self._check_relationship_keywords(
                content, relationship_keywords
            )
            if detected_type:
                return detected_type

        return "associated"

    def _check_relationship_keywords(
        self, content: str, relationship_keywords: Dict[str, List[str]]
    ) -> Optional[str]:
        """Check content for relationship keywords.

        Rule #1: Extracted to reduce nesting in _infer_relationship_type.

        Args:
            content: Text content to check
            relationship_keywords: Dict of relationship types to keywords

        Returns:
            Relationship type or None
        """
        for rel_type, keywords in relationship_keywords.items():
            for keyword in keywords:
                if keyword in content:
                    return rel_type
        return None

    def generate_profile(
        self, character: Character, chunks: List[Any], llm_client: Any = None
    ) -> CharacterProfile:
        """Generate comprehensive character profile.

        Args:
            character: Character to profile
            chunks: Text chunks for context
            llm_client: Optional LLM for enhanced analysis

        Returns:
            CharacterProfile with analysis
        """
        appearances = self.track_appearances(character.name, chunks)

        profile = CharacterProfile(
            character=character,
            key_moments=[app.context for app in appearances[:5]],
        )

        if llm_client is None:
            profile.description = f"Character appearing {character.mention_count} times"
            return profile

        # Use LLM for enhanced profile
        profile = self._enhance_profile_with_llm(profile, chunks, llm_client)
        return profile

    def _enhance_profile_with_llm(
        self, profile: CharacterProfile, chunks: List[Any], llm_client: Any
    ) -> CharacterProfile:
        """Enhance profile using LLM analysis.

        Args:
            profile: Base profile to enhance
            chunks: Text chunks
            llm_client: LLM client

        Returns:
            Enhanced CharacterProfile
        """
        # Build context from relevant chunks
        char_name = profile.character.name
        relevant_chunks = []

        for chunk in chunks:
            content = getattr(chunk, "content", str(chunk))
            if char_name.lower() in content.lower():
                relevant_chunks.append(content[:300])
            if len(relevant_chunks) >= 5:
                break

        context = "\n---\n".join(relevant_chunks)

        prompt = (
            f"Analyze the character '{char_name}' based on these excerpts:\n\n"
            f"{context}\n\n"
            "Provide:\n"
            "1. Brief description (1-2 sentences)\n"
            "2. Character arc summary (2-3 sentences)\n"
            "3. Key traits (3-5 traits)\n"
            "4. Motivations (1-3 motivations)\n\n"
            "Format as JSON with keys: description, arc_summary, traits, motivations"
        )

        try:
            response = llm_client.generate(prompt)
            data = json.loads(response)

            profile.description = data.get("description", profile.description)
            profile.arc_summary = data.get("arc_summary", "")
            profile.traits = data.get("traits", [])
            profile.motivations = data.get("motivations", [])

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse LLM response: {e}")

        return profile


# ============================================================================
# Character Command Class
# ============================================================================


class CharacterCommand(LiteraryCommand):
    """Analyze literary characters.

    Provides subcommands for character extraction and analysis.
    """

    def __init__(self) -> None:
        """Initialize character command."""
        super().__init__()
        self.extractor = CharacterExtractor()

    def execute(
        self,
        work: str,
        character: Optional[str] = None,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        relationships: bool = False,
    ) -> int:
        """Analyze characters in a literary work.

        Args:
            work: Name of the literary work
            character: Specific character to analyze (optional)
            project: Project directory
            output: Output file for analysis (optional)
            relationships: Include relationship analysis

        Returns:
            0 on success, 1 on error
        """
        try:
            self.validate_work_name(work)

            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            search_query = self._build_search_query(work, character)
            chunks = self._search_for_context(ctx["storage"], search_query)

            if not chunks:
                self._handle_no_context(work, character)
                return 0

            if character:
                return self._analyze_single_character(
                    work, character, chunks, llm_client, output, relationships
                )

            return self._analyze_all_characters(
                work, chunks, llm_client, output, relationships
            )

        except Exception as e:
            return self.handle_error(e, "Character analysis failed")

    def _analyze_single_character(
        self,
        work: str,
        character: str,
        chunks: List[Any],
        llm_client: Any,
        output: Optional[Path],
        relationships: bool,
    ) -> int:
        """Analyze a single character.

        Args:
            work: Literary work name
            character: Character name
            chunks: Context chunks
            llm_client: LLM client
            output: Output path
            relationships: Include relationships

        Returns:
            Exit code
        """
        char_obj = Character(name=character)
        profile = self.extractor.generate_profile(char_obj, chunks, llm_client)

        if relationships:
            all_chars = self.extractor.extract_characters(
                " ".join(getattr(c, "content", str(c)) for c in chunks)
            )
            graph = self.extractor.analyze_relationships(all_chars, chunks)
            profile.relationships = [
                e for e in graph.edges if e.target.lower() != character.lower()
            ]

        self._display_profile(work, profile)

        if output:
            self._save_profile(output, work, profile)

        return 0

    def _analyze_all_characters(
        self,
        work: str,
        chunks: List[Any],
        llm_client: Any,
        output: Optional[Path],
        relationships: bool,
    ) -> int:
        """Analyze all characters in a work.

        Args:
            work: Literary work name
            chunks: Context chunks
            llm_client: LLM client
            output: Output path
            relationships: Include relationships

        Returns:
            Exit code
        """
        combined_text = " ".join(getattr(c, "content", str(c)) for c in chunks)
        characters = self.extractor.extract_characters(combined_text)

        if not characters:
            self.print_warning("No characters found in the text")
            return 0

        if relationships:
            graph = self.extractor.analyze_relationships(characters, chunks)
            self._display_relationship_graph(work, graph)
        else:
            self._display_character_list(work, characters)

        if output:
            self._save_characters(output, work, characters, relationships)

        return 0

    def _build_search_query(self, work: str, character: Optional[str]) -> str:
        """Build search query for character context."""
        if character:
            return f"{work} character {character}"
        return f"{work} characters"

    def _search_for_context(self, storage: Any, query: str) -> List[Any]:
        """Search for character context."""
        from ingestforge.cli.core import ProgressManager

        return ProgressManager.run_with_spinner(
            lambda: storage.search(query, k=20),
            f"Searching for context: '{query}'...",
            "Context retrieved",
        )

    def _handle_no_context(self, work: str, character: Optional[str]) -> None:
        """Handle case where no context found."""
        subject = f"'{character}' in '{work}'" if character else f"'{work}'"
        self.print_warning(f"No context found for {subject}")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {work}\n"
            "  2. Using 'lit gather' to fetch Wikipedia pages\n"
            "  3. Checking spelling of work/character names"
        )

    def _display_profile(self, work: str, profile: CharacterProfile) -> None:
        """Display character profile."""
        self.console.print()

        content_lines = [
            f"**Name:** {profile.character.name}",
            f"**Mentions:** {profile.character.mention_count}",
            "",
        ]

        if profile.description:
            content_lines.extend(["**Description:**", profile.description, ""])

        if profile.arc_summary:
            content_lines.extend(["**Character Arc:**", profile.arc_summary, ""])

        if profile.traits:
            content_lines.extend(["**Traits:**", ", ".join(profile.traits), ""])

        if profile.motivations:
            content_lines.extend(
                ["**Motivations:**", ", ".join(profile.motivations), ""]
            )

        if profile.relationships:
            content_lines.append("**Relationships:**")
            for rel in profile.relationships:
                content_lines.append(f"- {rel.target}: {rel.relationship_type}")

        panel = Panel(
            Markdown("\n".join(content_lines)),
            title=f"[bold green]Character Profile: {profile.character.name} in {work}[/bold green]",
            border_style="green",
        )
        self.console.print(panel)

    def _display_character_list(self, work: str, characters: List[Character]) -> None:
        """Display list of characters."""
        self.console.print()

        table = Table(title=f"Characters in {work}")
        table.add_column("Name", style="cyan")
        table.add_column("Mentions", style="magenta")
        table.add_column("Aliases", style="dim")

        for char in characters[:20]:
            aliases = ", ".join(char.aliases) if char.aliases else "-"
            table.add_row(char.name, str(char.mention_count), aliases)

        self.console.print(table)

    def _display_relationship_graph(self, work: str, graph: RelationshipGraph) -> None:
        """Display character relationship graph."""
        self.console.print()

        # Display as table
        table = Table(title=f"Character Relationships in {work}")
        table.add_column("Character", style="cyan")
        table.add_column("Related To", style="green")
        table.add_column("Relationship", style="magenta")

        for char in graph.characters:
            for target, rel_type in char.relationships.items():
                table.add_row(char.name, target, rel_type)

        self.console.print(table)

        # Show Mermaid diagram
        self.console.print()
        self.console.print("[bold]Mermaid Diagram:[/bold]")
        self.console.print("```mermaid")
        self.console.print(graph.to_mermaid())
        self.console.print("```")

    def _save_profile(self, output: Path, work: str, profile: CharacterProfile) -> None:
        """Save character profile to file."""
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                data = {
                    "work": work,
                    "generated": timestamp,
                    "profile": profile.to_dict(),
                }
                output.write_text(json.dumps(data, indent=2), encoding="utf-8")
            else:
                content = self._format_profile_markdown(work, profile, timestamp)
                output.write_text(content, encoding="utf-8")

            self.print_success(f"Profile saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save profile: {e}")

    def _format_profile_markdown(
        self, work: str, profile: CharacterProfile, timestamp: str
    ) -> str:
        """Format profile as markdown."""
        lines = [
            f"# Character Profile: {profile.character.name}",
            f"## {work}",
            "",
            f"Generated: {timestamp}",
            "",
            "---",
            "",
        ]

        if profile.description:
            lines.extend(["## Description", profile.description, ""])

        if profile.arc_summary:
            lines.extend(["## Character Arc", profile.arc_summary, ""])

        if profile.traits:
            lines.extend(["## Traits", "- " + "\n- ".join(profile.traits), ""])

        if profile.motivations:
            lines.extend(
                ["## Motivations", "- " + "\n- ".join(profile.motivations), ""]
            )

        if profile.relationships:
            lines.append("## Relationships")
            for rel in profile.relationships:
                lines.append(f"- **{rel.target}**: {rel.relationship_type}")
            lines.append("")

        return "\n".join(lines)

    def _format_character_markdown(self, char: Character) -> List[str]:
        """Format a single character as markdown lines.

        Rule #1: Extracted to reduce nesting in _save_characters
        Rule #4: Function <60 lines

        Args:
            char: Character to format

        Returns:
            List of markdown lines
        """
        lines = [
            f"## {char.name}",
            f"- Mentions: {char.mention_count}",
        ]
        if char.aliases:
            lines.append(f"- Aliases: {', '.join(char.aliases)}")
        if char.relationships:
            lines.append("- Relationships:")
            for target, rel in char.relationships.items():
                lines.append(f"  - {target}: {rel}")
        lines.append("")
        return lines

    def _save_characters_json(
        self,
        output: Path,
        work: str,
        characters: List[Character],
        timestamp: str,
    ) -> None:
        """Save characters to JSON file.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        data = {
            "work": work,
            "generated": timestamp,
            "characters": [c.to_dict() for c in characters],
        }
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _save_characters_markdown(
        self,
        output: Path,
        work: str,
        characters: List[Character],
        timestamp: str,
    ) -> None:
        """Save characters to markdown file.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        lines = [
            f"# Characters in {work}",
            "",
            f"Generated: {timestamp}",
            "",
            "---",
            "",
        ]
        for char in characters:
            lines.extend(self._format_character_markdown(char))

        output.write_text("\n".join(lines), encoding="utf-8")

    def _save_characters(
        self,
        output: Path,
        work: str,
        characters: List[Character],
        relationships: bool,
    ) -> None:
        """Save character list to file.

        Rule #1: Reduced nesting from 5 → 2 via helper extraction
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if output.suffix == ".json":
                self._save_characters_json(output, work, characters, timestamp)
            else:
                self._save_characters_markdown(output, work, characters, timestamp)

            self.print_success(f"Characters saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save characters: {e}")


# ============================================================================
# Typer Command Wrappers
# ============================================================================


def command(
    work: str = typer.Argument(..., help="Name of the literary work"),
    character: Optional[str] = typer.Option(
        None, "--character", "-c", help="Specific character to analyze"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for analysis"
    ),
    relationships: bool = typer.Option(
        False, "--relationships", "-r", help="Include relationship analysis"
    ),
) -> None:
    """Analyze characters in a literary work.

    Analyzes character development, relationships, and significance.
    Can analyze all characters or focus on a specific character.

    Requires documents about the work to be ingested first.

    Examples:
        # Analyze all characters
        ingestforge lit character "Romeo and Juliet"

        # Analyze specific character
        ingestforge lit character "Hamlet" --character "Ophelia"

        # Include relationships
        ingestforge lit character "1984" -r

        # Save to file
        ingestforge lit character "Macbeth" -o characters.json
    """
    cmd = CharacterCommand()
    exit_code = cmd.execute(work, character, project, output, relationships)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


# Subcommand for extraction
def extract_command(
    file: Path = typer.Argument(..., help="Text file to analyze"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output JSON file"
    ),
    min_mentions: int = typer.Option(
        2, "--min-mentions", "-m", help="Minimum mentions to include"
    ),
) -> None:
    """Extract characters from a text file.

    Uses NER to identify character names and track mentions.

    Examples:
        # Extract from file
        ingestforge lit character extract novel.txt

        # Save to JSON
        ingestforge lit character extract novel.txt -o characters.json

        # Adjust minimum mentions
        ingestforge lit character extract novel.txt -m 3
    """
    from ingestforge.cli.core import ProgressManager

    if not file.exists():
        ProgressManager.print_error(f"File not found: {file}")
        raise typer.Exit(code=1)

    try:
        text = file.read_text(encoding="utf-8")
    except Exception as e:
        ProgressManager.print_error(f"Failed to read file: {e}")
        raise typer.Exit(code=1)

    extractor = CharacterExtractor(min_mentions=min_mentions)

    characters = ProgressManager.run_with_spinner(
        lambda: extractor.extract_characters(text),
        "Extracting characters...",
        f"Found {len(characters) if 'characters' in dir() else 0} characters",
    )

    if not characters:
        ProgressManager.print_warning("No characters found matching criteria")
        raise typer.Exit(code=0)

    # Display results
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title=f"Characters in {file.name}")
    table.add_column("Name", style="cyan")
    table.add_column("Mentions", style="magenta")

    for char in characters:
        table.add_row(char.name, str(char.mention_count))

    console.print(table)

    if output:
        data = {"file": str(file), "characters": [c.to_dict() for c in characters]}
        output.write_text(json.dumps(data, indent=2), encoding="utf-8")
        ProgressManager.print_success(f"Saved to: {output}")
