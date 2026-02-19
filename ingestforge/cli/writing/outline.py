"""Outline command - Create structured outlines from source material.

This module provides outline generation functionality with:
- Hierarchical structure building
- Theme-based organization
- Multiple output formats (numbered, bulleted, markdown)"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.panel import Panel
from rich.tree import Tree

from ingestforge.cli.writing.base import WritingCommand

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes (Rule #9: Full type hints)
# ============================================================================


@dataclass
class OutlineSection:
    """Represents a section in an outline.

    Attributes:
        title: Section title
        level: Nesting level (1, 2, 3, etc.)
        notes: Notes for this section
        sources: Source references
        subsections: Child sections
    """

    title: str
    level: int = 1
    notes: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    subsections: List["OutlineSection"] = field(default_factory=list)

    def add_subsection(self, section: "OutlineSection") -> None:
        """Add a subsection.

        Args:
            section: Section to add
        """
        section.level = self.level + 1
        self.subsections.append(section)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Section as dictionary
        """
        return {
            "title": self.title,
            "level": self.level,
            "notes": self.notes,
            "sources": self.sources,
            "subsections": [s.to_dict() for s in self.subsections],
        }

    def get_all_sections_flat(self) -> List["OutlineSection"]:
        """Get all sections as flat list.

        Returns:
            List of all sections including nested
        """
        result = [self]
        for sub in self.subsections:
            result.extend(sub.get_all_sections_flat())
        return result


@dataclass
class Outline:
    """Represents a complete outline document.

    Attributes:
        title: Outline title
        sections: Top-level sections
        depth: Maximum nesting depth
        format: Output format
        sources_used: All sources referenced
    """

    title: str
    sections: List[OutlineSection] = field(default_factory=list)
    depth: int = 3
    format: str = "numbered"
    sources_used: List[str] = field(default_factory=list)

    def add_section(self, section: OutlineSection) -> None:
        """Add a top-level section.

        Args:
            section: Section to add
        """
        section.level = 1
        self.sections.append(section)

    def get_section_count(self) -> int:
        """Get total section count including subsections.

        Returns:
            Total number of sections
        """
        count = 0
        for section in self.sections:
            count += len(section.get_all_sections_flat())
        return count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Outline as dictionary
        """
        return {
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "depth": self.depth,
            "format": self.format,
            "sources_used": self.sources_used,
            "section_count": self.get_section_count(),
        }


# ============================================================================
# Outline Builder (Rule #4: Small focused classes)
# ============================================================================


class OutlineBuilder:
    """Build outlines from source material.

    This class handles:
    - Building hierarchical outlines
    - Suggesting sections based on content
    - Organizing chunks by theme

    Attributes:
        llm_client: LLM client for generation
    """

    STRUCTURE_PROMPT = """Analyze the topic and source material to create a detailed outline.

Topic: {topic}

Source Material:
{context}

Requirements:
- Create {depth} levels of hierarchy (main sections, subsections, sub-subsections)
- Each section should have a clear purpose
- Include notes about what to cover in each section
- Reference relevant source material for each section

Return JSON:
{{
    "title": "Outline Title",
    "sections": [
        {{
            "title": "Section Title",
            "notes": ["Note 1", "Note 2"],
            "sources": ["source_id_1"],
            "subsections": [
                {{
                    "title": "Subsection Title",
                    "notes": ["Note"],
                    "sources": [],
                    "subsections": []
                }}
            ]
        }}
    ]
}}"""

    def __init__(self, llm_client: Any) -> None:
        """Initialize outline builder.

        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client

    def build(
        self,
        topic: str,
        sources: List[Any],
        depth: int = 3,
    ) -> Outline:
        """Build outline from topic and sources.

        Rule #1: Max 3 nesting levels

        Args:
            topic: Topic for the outline
            sources: Source chunks
            depth: Maximum hierarchy depth

        Returns:
            Generated Outline object
        """
        # Prepare context
        context = self._format_sources(sources)
        prompt = self.STRUCTURE_PROMPT.format(
            topic=topic,
            context=context,
            depth=depth,
        )

        # Generate with LLM
        response = self.llm_client.generate(prompt)
        data = self._parse_outline_response(response)

        # Build outline from data
        return self._build_outline_from_data(topic, data, depth, sources)

    def _format_sources(self, sources: List[Any]) -> str:
        """Format sources for context.

        Args:
            sources: Source chunks

        Returns:
            Formatted source text
        """
        if not sources:
            return "[No source material provided]"

        parts: List[str] = []
        for idx, chunk in enumerate(sources[:15], 1):
            text = self._get_chunk_text(chunk)
            source_id = self._get_source_id(chunk)
            parts.append(f"[{source_id}]: {text[:300]}...")

        return "\n\n".join(parts)

    def _get_chunk_text(self, chunk: Any) -> str:
        """Get text from chunk.

        Args:
            chunk: Chunk object

        Returns:
            Text content
        """
        if hasattr(chunk, "text"):
            return chunk.text
        if hasattr(chunk, "content"):
            return chunk.content
        return str(chunk)

    def _get_source_id(self, chunk: Any) -> str:
        """Get source ID from chunk.

        Args:
            chunk: Chunk object

        Returns:
            Source identifier
        """
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            return chunk.metadata.get("source", f"source_{id(chunk)}")
        if hasattr(chunk, "source_file"):
            return chunk.source_file
        return f"source_{id(chunk)}"

    def _parse_outline_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into outline data.

        Rule #5: Log errors

        Args:
            response: LLM response string

        Returns:
            Parsed outline data
        """
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {e}, attempting extraction")
            return self._extract_json_from_text(response)

    def _extract_json_from_text(self, text: str) -> Dict[str, Any]:
        """Extract JSON from surrounding text.

        Args:
            text: Text containing JSON

        Returns:
            Extracted data or default
        """
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error(f"JSON extraction failed: {e}")

        return {"title": "Outline", "sections": []}

    def _build_outline_from_data(
        self,
        topic: str,
        data: Dict[str, Any],
        depth: int,
        sources: List[Any],
    ) -> Outline:
        """Build Outline object from parsed data.

        Args:
            topic: Outline topic
            data: Parsed outline data
            depth: Maximum depth
            sources: Source chunks

        Returns:
            Outline object
        """
        outline = Outline(
            title=data.get("title", topic),
            depth=depth,
            sources_used=self._collect_source_ids(sources),
        )

        for section_data in data.get("sections", []):
            section = self._build_section(section_data, level=1, max_depth=depth)
            outline.add_section(section)

        return outline

    def _build_section(
        self,
        data: Dict[str, Any],
        level: int,
        max_depth: int,
    ) -> OutlineSection:
        """Build section from data recursively.

        Rule #1: Limit recursion depth

        Args:
            data: Section data
            level: Current level
            max_depth: Maximum depth

        Returns:
            OutlineSection object
        """
        section = OutlineSection(
            title=data.get("title", "Untitled Section"),
            level=level,
            notes=data.get("notes", []),
            sources=data.get("sources", []),
        )

        # Only add subsections if within depth limit
        if level < max_depth:
            for sub_data in data.get("subsections", []):
                subsection = self._build_section(sub_data, level + 1, max_depth)
                section.subsections.append(subsection)

        return section

    def _collect_source_ids(self, sources: List[Any]) -> List[str]:
        """Collect all source IDs.

        Args:
            sources: Source chunks

        Returns:
            List of source IDs
        """
        ids: List[str] = []
        for chunk in sources:
            source_id = self._get_source_id(chunk)
            if source_id not in ids:
                ids.append(source_id)
        return ids

    def suggest_sections(self, topic: str, context: str) -> List[OutlineSection]:
        """Suggest sections for a topic.

        Args:
            topic: Topic for suggestions
            context: Source context

        Returns:
            List of suggested sections
        """
        prompt = f"""Suggest 5-8 main sections for an outline about: "{topic}"

Context:
{context}

Return JSON:
{{
    "sections": [
        {{"title": "Section Title", "notes": ["Key point to cover"]}}
    ]
}}"""

        response = self.llm_client.generate(prompt)
        data = self._parse_outline_response(response)

        sections: List[OutlineSection] = []
        for item in data.get("sections", []):
            sections.append(
                OutlineSection(
                    title=item.get("title", ""),
                    notes=item.get("notes", []),
                )
            )

        return sections

    def organize_by_theme(
        self,
        chunks: List[Any],
    ) -> Dict[str, List[Any]]:
        """Organize chunks by theme.

        Args:
            chunks: Chunks to organize

        Returns:
            Dictionary mapping theme to chunks
        """
        if not chunks:
            return {}

        # Get text sample for theme extraction
        sample_text = "\n".join(self._get_chunk_text(c)[:200] for c in chunks[:10])

        prompt = f"""Identify 3-5 main themes in this content:

{sample_text}

Return JSON:
{{
    "themes": ["Theme 1", "Theme 2", "Theme 3"]
}}"""

        response = self.llm_client.generate(prompt)
        data = self._parse_outline_response(response)
        themes = data.get("themes", ["General"])

        # Assign chunks to themes based on content matching
        return self._assign_chunks_to_themes(chunks, themes)

    def _assign_chunks_to_themes(
        self,
        chunks: List[Any],
        themes: List[str],
    ) -> Dict[str, List[Any]]:
        """Assign chunks to themes based on content.

        Args:
            chunks: Chunks to assign
            themes: Available themes

        Returns:
            Theme to chunks mapping
        """
        result: Dict[str, List[Any]] = {theme: [] for theme in themes}
        result["Other"] = []

        for chunk in chunks:
            text = self._get_chunk_text(chunk).lower()
            assigned = False

            for theme in themes:
                if theme.lower() in text:
                    result[theme].append(chunk)
                    assigned = True
                    break

            if not assigned:
                result["Other"].append(chunk)

        # Remove empty themes
        return {k: v for k, v in result.items() if v}


# ============================================================================
# Outline Formatter (Rule #4: Focused class)
# ============================================================================


class OutlineFormatter:
    """Format outlines for different output types.

    Supports numbered, bulleted, and markdown formats.
    """

    def format(self, outline: Outline, format_type: str) -> str:
        """Format outline for output.

        Rule #1: Dictionary dispatch

        Args:
            outline: Outline to format
            format_type: Format type (numbered, bulleted, markdown)

        Returns:
            Formatted string
        """
        formatters = {
            "numbered": self._format_numbered,
            "bulleted": self._format_bulleted,
            "markdown": self._format_markdown,
        }

        formatter = formatters.get(format_type.lower(), self._format_numbered)
        return formatter(outline)

    def _format_numbered(self, outline: Outline) -> str:
        """Format as numbered outline.

        Args:
            outline: Outline to format

        Returns:
            Numbered outline string
        """
        lines = [f"# {outline.title}\n"]

        for idx, section in enumerate(outline.sections, 1):
            lines.extend(self._format_section_numbered(section, str(idx)))

        return "\n".join(lines)

    def _format_section_numbered(
        self,
        section: OutlineSection,
        prefix: str,
    ) -> List[str]:
        """Format section with numbered prefix.

        Args:
            section: Section to format
            prefix: Number prefix (e.g., "1", "1.1")

        Returns:
            List of formatted lines
        """
        indent = "  " * (section.level - 1)
        lines = [f"{indent}{prefix}. {section.title}"]

        for note in section.notes:
            lines.append(f"{indent}   - {note}")

        for idx, sub in enumerate(section.subsections, 1):
            sub_prefix = f"{prefix}.{idx}"
            lines.extend(self._format_section_numbered(sub, sub_prefix))

        return lines

    def _format_bulleted(self, outline: Outline) -> str:
        """Format as bulleted outline.

        Args:
            outline: Outline to format

        Returns:
            Bulleted outline string
        """
        lines = [f"# {outline.title}\n"]

        for section in outline.sections:
            lines.extend(self._format_section_bulleted(section))

        return "\n".join(lines)

    def _format_section_bulleted(self, section: OutlineSection) -> List[str]:
        """Format section with bullets.

        Args:
            section: Section to format

        Returns:
            List of formatted lines
        """
        bullets = ["*", "-", "+", "o"]
        bullet = bullets[min(section.level - 1, len(bullets) - 1)]
        indent = "  " * (section.level - 1)

        lines = [f"{indent}{bullet} {section.title}"]

        for note in section.notes:
            lines.append(f"{indent}  - {note}")

        for sub in section.subsections:
            lines.extend(self._format_section_bulleted(sub))

        return lines

    def _format_markdown(self, outline: Outline) -> str:
        """Format as markdown.

        Args:
            outline: Outline to format

        Returns:
            Markdown outline string
        """
        lines = [f"# {outline.title}\n"]

        for section in outline.sections:
            lines.extend(self._format_section_markdown(section))

        if outline.sources_used:
            lines.append("\n---\n## Sources\n")
            for source in outline.sources_used:
                lines.append(f"- {source}")

        return "\n".join(lines)

    def _format_section_markdown(self, section: OutlineSection) -> List[str]:
        """Format section as markdown.

        Args:
            section: Section to format

        Returns:
            List of markdown lines
        """
        heading = "#" * (section.level + 1)
        lines = [f"{heading} {section.title}"]

        if section.notes:
            lines.append("")
            for note in section.notes:
                lines.append(f"- {note}")

        if section.sources:
            lines.append(f"\n*Sources: {', '.join(section.sources)}*")

        lines.append("")

        for sub in section.subsections:
            lines.extend(self._format_section_markdown(sub))

        return lines

    def to_tree(self, outline: Outline) -> Tree:
        """Convert outline to Rich Tree for display.

        Args:
            outline: Outline to convert

        Returns:
            Rich Tree object
        """
        tree = Tree(f"[bold blue]{outline.title}[/bold blue]")

        for section in outline.sections:
            self._add_section_to_tree(tree, section)

        return tree

    def _add_section_to_tree(self, tree: Tree, section: OutlineSection) -> None:
        """Add section to tree recursively.

        Args:
            tree: Parent tree/branch
            section: Section to add
        """
        branch = tree.add(f"[green]{section.title}[/green]")

        for note in section.notes:
            branch.add(f"[dim]{note}[/dim]")

        for sub in section.subsections:
            self._add_section_to_tree(branch, sub)


# ============================================================================
# Command Implementation
# ============================================================================


class OutlineCommand(WritingCommand):
    """Create structured outlines CLI command."""

    def execute(
        self,
        topic: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        depth: int = 3,
        format_type: str = "numbered",
    ) -> int:
        """Execute outline creation command.

        Args:
            topic: Topic for outline
            project: Project directory
            output: Output file path
            depth: Maximum depth
            format_type: Output format

        Returns:
            Exit code
        """
        try:
            return self._execute_outline_creation(
                topic, project, output, depth, format_type
            )
        except Exception as e:
            logger.error(f"Outline creation failed: {e}")
            return self.handle_error(e, "Outline creation failed")

    def _execute_outline_creation(
        self,
        topic: str,
        project: Optional[Path],
        output: Optional[Path],
        depth: int,
        format_type: str,
    ) -> int:
        """Internal outline creation logic.

        Args:
            topic: Topic for outline
            project: Project directory
            output: Output file
            depth: Maximum depth
            format_type: Output format

        Returns:
            Exit code
        """
        ctx = self.initialize_context(project, require_storage=True)
        llm_client = self.get_llm_client(ctx)

        if not llm_client:
            self.print_error("No LLM client available")
            return 1

        chunks = self.search_context(ctx["storage"], topic, k=30)
        if not chunks:
            self.print_warning(f"No context found for: '{topic}'")
            return 0

        # Build outline
        builder = OutlineBuilder(llm_client)
        outline = self._build_outline(builder, topic, chunks, depth)

        # Format and display
        formatter = OutlineFormatter()
        self._display_outline(outline, formatter, format_type)

        if output:
            self._save_outline(output, outline, formatter, format_type)

        return 0

    def _build_outline(
        self,
        builder: OutlineBuilder,
        topic: str,
        chunks: List[Any],
        depth: int,
    ) -> Outline:
        """Build outline with progress.

        Args:
            builder: OutlineBuilder instance
            topic: Topic
            chunks: Source chunks
            depth: Maximum depth

        Returns:
            Generated Outline
        """
        from ingestforge.cli.core import ProgressManager

        return ProgressManager.run_with_spinner(
            lambda: builder.build(topic, chunks, depth),
            "Building outline...",
            "Outline created",
        )

    def _display_outline(
        self,
        outline: Outline,
        formatter: OutlineFormatter,
        format_type: str,
    ) -> None:
        """Display outline to console.

        Args:
            outline: Outline to display
            formatter: Formatter instance
            format_type: Format type
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Sections:[/bold] {outline.get_section_count()} | "
                f"[bold]Depth:[/bold] {outline.depth} | "
                f"[bold]Sources:[/bold] {len(outline.sources_used)}",
                title="Outline Summary",
                border_style="cyan",
            )
        )
        self.console.print()

        # Display as tree for visual appeal
        tree = formatter.to_tree(outline)
        self.console.print(tree)

    def _save_outline(
        self,
        output: Path,
        outline: Outline,
        formatter: OutlineFormatter,
        format_type: str,
    ) -> None:
        """Save outline to file.

        Args:
            output: Output path
            outline: Outline to save
            formatter: Formatter
            format_type: Format type
        """
        formatted = formatter.format(outline, format_type)
        output.write_text(formatted, encoding="utf-8")
        self.print_success(f"Outline saved to: {output}")

        # Save JSON metadata
        meta_path = output.with_suffix(".json")
        meta_path.write_text(json.dumps(outline.to_dict(), indent=2), encoding="utf-8")


# ============================================================================
# CLI Command Function
# ============================================================================


def command(
    topic: str = typer.Argument(..., help="Topic for the outline"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file"),
    depth: int = typer.Option(3, "-d", "--depth", help="Maximum outline depth (1-5)"),
    format: str = typer.Option(
        "numbered",
        "-f",
        "--format",
        help="Output format (numbered, bulleted, markdown)",
    ),
) -> None:
    """Create a structured outline from source material.

    Examples:
        # Create numbered outline
        ingestforge writing outline "Machine Learning Thesis"

        # Create deep outline with 5 levels
        ingestforge writing outline "Research Topic" -d 5

        # Create markdown outline
        ingestforge writing outline "Project Plan" -f markdown -o outline.md
    """
    # Validate depth
    if depth < 1 or depth > 5:
        raise typer.BadParameter("Depth must be between 1 and 5")

    exit_code = OutlineCommand().execute(topic, project, output, depth, format)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
