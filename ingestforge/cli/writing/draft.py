"""Draft command - Generate drafts with citations.

This module provides draft generation functionality with:
- Multiple writing styles (academic, blog, technical, casual)
- Inline citation support [Author, Year] format
- Multiple output formats (markdown, plain, latex)"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from rich.markdown import Markdown
from rich.panel import Panel

from ingestforge.cli.writing.base import WritingCommand

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes (Rule #9: Full type hints)
# ============================================================================


@dataclass
class Citation:
    """Represents a single citation reference.

    Attributes:
        author: Author name(s)
        year: Publication year
        title: Source title
        source_id: Internal source identifier
        page: Page number (optional)
        url: Source URL (optional)
    """

    author: str
    year: str
    title: str
    source_id: str
    page: Optional[str] = None
    url: Optional[str] = None

    def to_inline(self) -> str:
        """Format as inline citation [Author, Year].

        Returns:
            Inline citation string
        """
        return f"[{self.author}, {self.year}]"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Citation as dictionary
        """
        return {
            "author": self.author,
            "year": self.year,
            "title": self.title,
            "source_id": self.source_id,
            "page": self.page,
            "url": self.url,
        }


@dataclass
class Draft:
    """Represents a generated draft document.

    Attributes:
        content: Main draft text content
        citations: List of citations used
        word_count: Total word count
        style: Writing style used
        sources_used: List of source identifiers
        format: Output format (markdown, plain, latex)
    """

    content: str
    citations: List[Citation] = field(default_factory=list)
    word_count: int = 0
    style: str = "academic"
    sources_used: List[str] = field(default_factory=list)
    format: str = "markdown"

    def __post_init__(self) -> None:
        """Calculate word count if not provided."""
        if self.word_count == 0 and self.content:
            self.word_count = len(self.content.split())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Draft as dictionary
        """
        return {
            "content": self.content,
            "citations": [c.to_dict() for c in self.citations],
            "word_count": self.word_count,
            "style": self.style,
            "sources_used": self.sources_used,
            "format": self.format,
        }


# ============================================================================
# Draft Generator (Rule #4: Small focused classes)
# ============================================================================


class DraftGenerator:
    """Generate drafts with citations from source material.

    This class handles:
    - Generating drafts in multiple styles
    - Adding inline citations
    - Formatting output in various formats

    Attributes:
        llm_client: LLM client for generation
        style_prompts: Style-specific prompt templates
    """

    # Style prompt templates (Rule #6: Class-level constants)
    STYLE_PROMPTS: Dict[str, str] = {
        "academic": (
            "Write in a formal academic style with precise terminology, "
            "balanced arguments, and objective analysis. Use passive voice "
            "where appropriate and maintain scholarly tone."
        ),
        "blog": (
            "Write in an engaging blog style with a conversational tone. "
            "Use first person, include rhetorical questions, and make "
            "complex topics accessible to general readers."
        ),
        "technical": (
            "Write in a clear technical style with precise definitions, "
            "step-by-step explanations, and concrete examples. "
            "Use active voice and avoid unnecessary jargon."
        ),
        "casual": (
            "Write in a relaxed, conversational style. Use simple language, "
            "short sentences, and relatable examples. Be friendly and direct."
        ),
    }

    LENGTH_GUIDES: Dict[str, str] = {
        "short": "approximately 300-500 words",
        "medium": "approximately 800-1200 words",
        "long": "approximately 1800-2500 words",
    }

    def __init__(self, llm_client: Any) -> None:
        """Initialize draft generator.

        Args:
            llm_client: LLM client for text generation
        """
        self.llm_client = llm_client

    def generate(
        self,
        topic: str,
        style: str,
        length: int,
        context: List[Any],
        include_citations: bool = True,
    ) -> Draft:
        """Generate a draft on the given topic.

        Rule #1: Max 3 nesting levels
        Rule #4: <60 lines

        Args:
            topic: Topic to write about
            style: Writing style (academic, blog, technical, casual)
            length: Target word count
            context: List of source chunks for context
            include_citations: Whether to include inline citations

        Returns:
            Generated Draft object
        """
        style = style.lower() if style else "academic"
        if style not in self.STYLE_PROMPTS:
            logger.warning(f"Unknown style '{style}', defaulting to academic")
            style = "academic"

        # Build prompt
        prompt = self._build_generation_prompt(
            topic, style, length, context, include_citations
        )

        # Generate with LLM
        response = self.llm_client.generate(prompt)

        # Extract citations from context
        citations = self._extract_citations(context) if include_citations else []
        sources = self._get_source_ids(context)

        return Draft(
            content=response,
            citations=citations,
            word_count=len(response.split()),
            style=style,
            sources_used=sources,
            format="markdown",
        )

    def _build_generation_prompt(
        self,
        topic: str,
        style: str,
        length: int,
        context: List[Any],
        include_citations: bool,
    ) -> str:
        """Build the generation prompt.

        Rule #4: <60 lines

        Args:
            topic: Topic to write about
            style: Writing style
            length: Target word count
            context: Source chunks
            include_citations: Whether to include citations

        Returns:
            Complete prompt string
        """
        style_instruction = self.STYLE_PROMPTS.get(
            style, self.STYLE_PROMPTS["academic"]
        )
        length_guide = self._get_length_guide(length)

        context_text = self._format_context_with_sources(context)

        citation_instruction = ""
        if include_citations:
            citation_instruction = (
                "\nInclude inline citations in [Author, Year] format where "
                "you reference the source material. Cite specific claims and data."
            )

        return f"""Generate a {length_guide} draft about: "{topic}"

Style Instructions:
{style_instruction}

Source Material:
{context_text}

Requirements:
- Create a well-structured draft with introduction, body, and conclusion
- Use the source material to support your points
- Maintain logical flow between sections{citation_instruction}
- Format the output as clean markdown

Generate the draft now:"""

    def _get_length_guide(self, length: int) -> str:
        """Get length guide text.

        Args:
            length: Target word count

        Returns:
            Length guide string
        """
        if length <= 500:
            return self.LENGTH_GUIDES["short"]
        if length <= 1200:
            return self.LENGTH_GUIDES["medium"]
        return self.LENGTH_GUIDES["long"]

    def _format_context_with_sources(self, chunks: List[Any]) -> str:
        """Format chunks with source attribution.

        Rule #1: Early return for empty

        Args:
            chunks: Source chunks

        Returns:
            Formatted context string
        """
        if not chunks:
            return "[No source material provided]"

        parts: List[str] = []
        for idx, chunk in enumerate(chunks[:10], 1):  # Limit to 10 sources
            text = getattr(chunk, "text", getattr(chunk, "content", str(chunk)))
            source = self._get_chunk_source(chunk)
            parts.append(f"[Source {idx}: {source}]\n{text[:500]}...")

        return "\n\n".join(parts)

    def _get_chunk_source(self, chunk: Any) -> str:
        """Extract source identifier from chunk.

        Args:
            chunk: Chunk object

        Returns:
            Source identifier string
        """
        # Try different attributes
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            return chunk.metadata.get("source", "Unknown")
        if hasattr(chunk, "source_file"):
            return chunk.source_file
        return "Unknown Source"

    def _extract_citations(self, chunks: List[Any]) -> List[Citation]:
        """Extract citations from chunks.

        Args:
            chunks: Source chunks

        Returns:
            List of Citation objects
        """
        citations: List[Citation] = []
        seen_sources: set[str] = set()

        for chunk in chunks:
            citation = self._extract_single_citation(chunk)
            if citation and citation.source_id not in seen_sources:
                seen_sources.add(citation.source_id)
                citations.append(citation)

        return citations

    def _extract_single_citation(self, chunk: Any) -> Optional[Citation]:
        """Extract a single citation from a chunk.

        Rule #1: Early returns

        Args:
            chunk: Source chunk

        Returns:
            Citation or None
        """
        metadata = self._get_metadata(chunk)
        if not metadata:
            return None

        # Extract author, year, title
        author = metadata.get("author", "Unknown Author")
        year = (
            metadata.get("year", metadata.get("date", "n.d."))[:4]
            if metadata.get("year") or metadata.get("date")
            else "n.d."
        )
        title = metadata.get("title", metadata.get("source", "Untitled"))
        source_id = metadata.get("source", metadata.get("document_id", "unknown"))

        return Citation(
            author=author,
            year=year,
            title=title,
            source_id=source_id,
            page=metadata.get("page"),
            url=metadata.get("url"),
        )

    def _get_metadata(self, chunk: Any) -> Dict[str, Any]:
        """Get metadata from chunk.

        Args:
            chunk: Chunk object

        Returns:
            Metadata dictionary
        """
        if isinstance(chunk, dict):
            return chunk.get("metadata", chunk)
        if hasattr(chunk, "metadata"):
            meta = chunk.metadata
            return meta if isinstance(meta, dict) else {}
        return {}

    def _get_source_ids(self, chunks: List[Any]) -> List[str]:
        """Get unique source IDs from chunks.

        Args:
            chunks: Source chunks

        Returns:
            List of source IDs
        """
        sources: set[str] = set()
        for chunk in chunks:
            source = self._get_chunk_source(chunk)
            if source != "Unknown Source":
                sources.add(source)
        return list(sources)

    def add_citations(self, draft: str, sources: List[Any]) -> str:
        """Add inline citations to draft text.

        Args:
            draft: Draft text
            sources: Source chunks

        Returns:
            Draft with citations added
        """
        # Already handled in generation
        return draft

    def format_output(self, draft: Draft, output_format: str) -> str:
        """Format draft for output.

        Rule #1: Dictionary dispatch

        Args:
            draft: Draft object
            output_format: Output format (markdown, plain, latex)

        Returns:
            Formatted draft string
        """
        formatters = {
            "markdown": self._format_markdown,
            "plain": self._format_plain,
            "latex": self._format_latex,
        }

        formatter = formatters.get(output_format.lower(), self._format_markdown)
        return formatter(draft)

    def _format_markdown(self, draft: Draft) -> str:
        """Format as markdown.

        Args:
            draft: Draft object

        Returns:
            Markdown formatted string
        """
        output = [draft.content]

        if draft.citations:
            output.append("\n\n---\n## References\n")
            for citation in draft.citations:
                output.append(
                    f"- {citation.author} ({citation.year}). *{citation.title}*"
                )

        return "\n".join(output)

    def _format_plain(self, draft: Draft) -> str:
        """Format as plain text.

        Args:
            draft: Draft object

        Returns:
            Plain text string
        """
        # Remove markdown formatting
        text = re.sub(r"[#*_`]", "", draft.content)
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        return text

    def _format_latex(self, draft: Draft) -> str:
        """Format as LaTeX.

        Args:
            draft: Draft object

        Returns:
            LaTeX formatted string
        """
        # Convert markdown to LaTeX basics
        content = draft.content
        content = re.sub(r"^# (.+)$", r"\\section{\1}", content, flags=re.MULTILINE)
        content = re.sub(r"^## (.+)$", r"\\subsection{\1}", content, flags=re.MULTILINE)
        content = re.sub(
            r"^### (.+)$", r"\\subsubsection{\1}", content, flags=re.MULTILINE
        )
        content = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", content)
        content = re.sub(r"\*(.+?)\*", r"\\textit{\1}", content)
        content = re.sub(r"\[([^\]]+), (\d{4})\]", r"\\cite{\1\2}", content)

        output = [
            "\\documentclass{article}",
            "\\begin{document}",
            content,
            "\\end{document}",
        ]

        return "\n".join(output)


# ============================================================================
# Command Implementation (Rule #4: Small functions)
# ============================================================================


class DraftCommand(WritingCommand):
    """Generate drafts with citations CLI command."""

    def execute(
        self,
        topic: str,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        style: str = "academic",
        length: int = 1000,
        include_citations: bool = True,
        output_format: str = "markdown",
    ) -> int:
        """Execute draft generation command.

        Rule #1: Max 3 nesting, early returns
        Rule #5: Log all errors

        Args:
            topic: Topic to write about
            project: Project directory
            output: Output file path
            style: Writing style
            length: Target word count
            include_citations: Include inline citations
            output_format: Output format

        Returns:
            Exit code (0=success, 1=error)
        """
        try:
            return self._execute_draft_generation(
                topic, project, output, style, length, include_citations, output_format
            )
        except Exception as e:
            logger.error(f"Draft generation failed: {e}")
            return self.handle_error(e, "Draft generation failed")

    def _execute_draft_generation(
        self,
        topic: str,
        project: Optional[Path],
        output: Optional[Path],
        style: str,
        length: int,
        include_citations: bool,
        output_format: str,
    ) -> int:
        """Internal draft generation logic.

        Args:
            topic: Topic to write about
            project: Project directory
            output: Output file path
            style: Writing style
            length: Target word count
            include_citations: Include citations
            output_format: Output format

        Returns:
            Exit code
        """
        # Initialize context
        ctx = self.initialize_context(project, require_storage=True)
        llm_client = self.get_llm_client(ctx)

        if not llm_client:
            self.print_error("No LLM client available")
            return 1

        # Search for context
        chunks = self.search_context(ctx["storage"], topic, k=40)
        if not chunks:
            self.print_warning(f"No context found for: '{topic}'")
            return 0

        # Generate draft
        generator = DraftGenerator(llm_client)
        draft = self._generate_draft(
            generator, topic, style, length, chunks, include_citations
        )

        # Format and output
        formatted = generator.format_output(draft, output_format)
        self._display_draft(formatted, draft, output_format)

        if output:
            self._save_draft(output, formatted, draft)

        return 0

    def _generate_draft(
        self,
        generator: DraftGenerator,
        topic: str,
        style: str,
        length: int,
        chunks: List[Any],
        include_citations: bool,
    ) -> Draft:
        """Generate the draft with progress indicator.

        Args:
            generator: DraftGenerator instance
            topic: Topic to write about
            style: Writing style
            length: Target word count
            chunks: Source chunks
            include_citations: Include citations

        Returns:
            Generated Draft
        """
        from ingestforge.cli.core import ProgressManager

        return ProgressManager.run_with_spinner(
            lambda: generator.generate(topic, style, length, chunks, include_citations),
            f"Generating {style} draft...",
            "Draft generated",
        )

    def _display_draft(self, formatted: str, draft: Draft, output_format: str) -> None:
        """Display draft to console.

        Args:
            formatted: Formatted draft text
            draft: Draft object
            output_format: Output format
        """
        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Style:[/bold] {draft.style} | "
                f"[bold]Words:[/bold] {draft.word_count} | "
                f"[bold]Sources:[/bold] {len(draft.sources_used)}",
                title="Draft Summary",
                border_style="green",
            )
        )
        self.console.print()

        if output_format == "markdown":
            self.console.print(Markdown(formatted))
        else:
            self.console.print(formatted)

    def _save_draft(self, output: Path, formatted: str, draft: Draft) -> None:
        """Save draft to file.

        Args:
            output: Output file path
            formatted: Formatted draft text
            draft: Draft object
        """
        output.write_text(formatted, encoding="utf-8")
        self.print_success(f"Draft saved to: {output}")

        # Also save metadata
        meta_path = output.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(draft.to_dict(), indent=2), encoding="utf-8")


# ============================================================================
# CLI Command Function
# ============================================================================


def command(
    topic: str = typer.Argument(..., help="Topic to write about"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output file (.md)"
    ),
    style: str = typer.Option(
        "academic",
        "-s",
        "--style",
        help="Writing style (academic, blog, technical, casual)",
    ),
    length: int = typer.Option(
        1000,
        "-l",
        "--length",
        help="Target word count",
    ),
    citations: bool = typer.Option(
        True,
        "--citations/--no-citations",
        help="Include inline citations",
    ),
    format: str = typer.Option(
        "markdown",
        "-f",
        "--format",
        help="Output format (markdown, plain, latex)",
    ),
) -> None:
    """Generate a draft with citations from source material.

    Examples:
        # Generate academic draft
        ingestforge writing draft "Introduction to neural networks"

        # Generate blog post style
        ingestforge writing draft "AI in healthcare" --style blog

        # Specify length and output
        ingestforge writing draft "Machine learning" -l 2000 -o draft.md

        # LaTeX output without citations
        ingestforge writing draft "Research topic" -f latex --no-citations
    """
    exit_code = DraftCommand().execute(
        topic, project, output, style, length, citations, format
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
