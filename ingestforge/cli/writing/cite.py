"""Cite command - Citation management for writing.

This module provides citation management functionality with:
- Citation insertion in multiple styles (APA, MLA, Chicago, IEEE, Harvard)
- Bibliography generation
- Citation verification"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import typer
from rich.panel import Panel

from ingestforge.cli.writing.base import WritingCommand

logger = logging.getLogger(__name__)

# ============================================================================
# Data Classes (Rule #9: Full type hints)
# ============================================================================


@dataclass
class Source:
    """Represents a citation source.

    Attributes:
        id: Unique source identifier
        author: Author name(s)
        title: Source title
        year: Publication year
        source_type: Type (article, book, website, etc.)
        journal: Journal name (for articles)
        publisher: Publisher name
        url: URL (for websites)
        page: Page numbers
        volume: Volume number
        issue: Issue number
        doi: Digital Object Identifier
    """

    id: str
    author: str
    title: str
    year: str
    source_type: str = "article"
    journal: Optional[str] = None
    publisher: Optional[str] = None
    url: Optional[str] = None
    page: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    doi: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Source as dictionary
        """
        return {
            "id": self.id,
            "author": self.author,
            "title": self.title,
            "year": self.year,
            "source_type": self.source_type,
            "journal": self.journal,
            "publisher": self.publisher,
            "url": self.url,
            "page": self.page,
            "volume": self.volume,
            "issue": self.issue,
            "doi": self.doi,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Source":
        """Create from dictionary.

        Args:
            data: Source data

        Returns:
            Source instance
        """
        return cls(
            id=data.get("id", "unknown"),
            author=data.get("author", "Unknown"),
            title=data.get("title", "Untitled"),
            year=data.get("year", "n.d."),
            source_type=data.get("source_type", "article"),
            journal=data.get("journal"),
            publisher=data.get("publisher"),
            url=data.get("url"),
            page=data.get("page"),
            volume=data.get("volume"),
            issue=data.get("issue"),
            doi=data.get("doi"),
        )


@dataclass
class VerificationResult:
    """Result of citation verification.

    Attributes:
        is_valid: Whether all citations are valid
        total_citations: Total citation count
        valid_citations: Number of valid citations
        missing_sources: Citations without sources
        broken_citations: Malformed citations
        warnings: Warning messages
    """

    is_valid: bool = True
    total_citations: int = 0
    valid_citations: int = 0
    missing_sources: List[str] = field(default_factory=list)
    broken_citations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Result as dictionary
        """
        return {
            "is_valid": self.is_valid,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "missing_sources": self.missing_sources,
            "broken_citations": self.broken_citations,
            "warnings": self.warnings,
        }


# ============================================================================
# Citation Manager (Rule #4: Small focused classes)
# ============================================================================


class CitationManager:
    """Manage citations in documents.

    Supports:
    - APA, MLA, Chicago, IEEE, Harvard citation styles
    - Inline citation insertion
    - Bibliography generation
    - Citation verification
    """

    # Citation style patterns (Rule #6: Class-level constants)
    CITATION_PATTERNS: Dict[str, str] = {
        "apa": r"\(([^)]+),\s*(\d{4})\)",  # (Author, 2024)
        "mla": r"\(([^)]+)\s+\d+\)",  # (Author 123)
        "chicago": r"\(([^)]+)\s+(\d{4})\)",  # (Author 2024)
        "ieee": r"\[(\d+)\]",  # [1]
        "harvard": r"\(([^)]+),\s*(\d{4})\)",  # (Author, 2024)
    }

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        """Initialize citation manager.

        Args:
            llm_client: Optional LLM client for smart citation
        """
        self.llm_client = llm_client

    def insert_citations(
        self,
        text: str,
        sources: List[Source],
        style: str = "apa",
    ) -> str:
        """Insert inline citations into text.

        Rule #1: Early return for empty

        Args:
            text: Text to add citations to
            sources: Available sources
            style: Citation style

        Returns:
            Text with citations added
        """
        if not text or not sources:
            return text

        style = self._validate_style(style)

        # Find citation markers and replace with proper format
        result = text
        for source in sources:
            formatted = self._format_inline_citation(source, style)
            # Replace source references with formatted citations
            patterns = [
                rf"\[{re.escape(source.id)}\]",
                rf"\[{re.escape(source.author)}\]",
            ]
            for pattern in patterns:
                result = re.sub(pattern, formatted, result)

        return result

    def format_bibliography(
        self,
        sources: List[Source],
        style: str = "apa",
    ) -> str:
        """Format sources as bibliography.

        Args:
            sources: Sources to format
            style: Citation style

        Returns:
            Formatted bibliography string
        """
        if not sources:
            return ""

        style = self._validate_style(style)
        formatter = self._get_bibliography_formatter(style)

        entries: List[str] = []
        for source in sorted(sources, key=lambda s: s.author.lower()):
            entries.append(formatter(source))

        return "\n\n".join(entries)

    def verify_citations(
        self,
        text: str,
        sources: List[Source],
    ) -> VerificationResult:
        """Verify citations in text against sources.

        Args:
            text: Text with citations
            sources: Available sources

        Returns:
            VerificationResult with verification details
        """
        result = VerificationResult()

        # Find all citations in text
        citations = self._extract_citations_from_text(text)
        result.total_citations = len(citations)

        # Build source lookup
        source_lookup = self._build_source_lookup(sources)

        # Verify each citation
        for citation in citations:
            is_valid, message = self._verify_single_citation(citation, source_lookup)
            if is_valid:
                result.valid_citations += 1
            else:
                result.missing_sources.append(citation)
                result.warnings.append(message)

        result.is_valid = result.valid_citations == result.total_citations
        return result

    def extract_sources_from_chunks(
        self,
        chunks: List[Any],
    ) -> List[Source]:
        """Extract source information from chunks.

        Args:
            chunks: Chunks with metadata

        Returns:
            List of Source objects
        """
        sources: List[Source] = []
        seen_ids: set[str] = set()

        for chunk in chunks:
            source = self._extract_source_from_chunk(chunk)
            if source and source.id not in seen_ids:
                seen_ids.add(source.id)
                sources.append(source)

        return sources

    def parse_bibtex(self, bibtex: str) -> List[Source]:
        """Parse BibTeX format into sources.

        Args:
            bibtex: BibTeX string

        Returns:
            List of Source objects
        """
        sources: List[Source] = []
        entries = re.findall(r"@(\w+)\{([^,]+),([^@]+)\}", bibtex, re.DOTALL)

        for entry_type, entry_id, content in entries:
            source = self._parse_bibtex_entry(entry_type, entry_id, content)
            if source:
                sources.append(source)

        return sources

    def to_bibtex(self, sources: List[Source]) -> str:
        """Convert sources to BibTeX format.

        Args:
            sources: Sources to convert

        Returns:
            BibTeX string
        """
        entries: List[str] = []

        for source in sources:
            entries.append(self._source_to_bibtex(source))

        return "\n\n".join(entries)

    # ========================================================================
    # Private Helper Methods
    # ========================================================================

    def _validate_style(self, style: str) -> str:
        """Validate citation style.

        Args:
            style: Input style

        Returns:
            Validated style
        """
        style = style.lower()
        valid_styles = ["apa", "mla", "chicago", "ieee", "harvard"]
        if style not in valid_styles:
            logger.warning(f"Unknown style '{style}', using 'apa'")
            return "apa"
        return style

    def _format_inline_citation(self, source: Source, style: str) -> str:
        """Format inline citation for style.

        Rule #1: Dictionary dispatch

        Args:
            source: Source to cite
            style: Citation style

        Returns:
            Formatted citation string
        """
        formatters = {
            "apa": self._format_apa_inline,
            "mla": self._format_mla_inline,
            "chicago": self._format_chicago_inline,
            "ieee": self._format_ieee_inline,
            "harvard": self._format_harvard_inline,
        }

        formatter = formatters.get(style, self._format_apa_inline)
        return formatter(source)

    def _format_apa_inline(self, source: Source) -> str:
        """Format APA inline citation.

        Args:
            source: Source to cite

        Returns:
            APA inline citation
        """
        return f"({source.author}, {source.year})"

    def _format_mla_inline(self, source: Source) -> str:
        """Format MLA inline citation.

        Args:
            source: Source to cite

        Returns:
            MLA inline citation
        """
        page = f" {source.page}" if source.page else ""
        return f"({source.author}{page})"

    def _format_chicago_inline(self, source: Source) -> str:
        """Format Chicago inline citation.

        Args:
            source: Source to cite

        Returns:
            Chicago inline citation
        """
        return f"({source.author} {source.year})"

    def _format_ieee_inline(self, source: Source) -> str:
        """Format IEEE inline citation.

        Args:
            source: Source to cite

        Returns:
            IEEE inline citation
        """
        return f"[{source.id}]"

    def _format_harvard_inline(self, source: Source) -> str:
        """Format Harvard inline citation.

        Args:
            source: Source to cite

        Returns:
            Harvard inline citation
        """
        return f"({source.author}, {source.year})"

    def _get_bibliography_formatter(self, style: str):
        """Get bibliography formatter for style.

        Args:
            style: Citation style

        Returns:
            Formatter function
        """
        formatters = {
            "apa": self._format_apa_bibliography,
            "mla": self._format_mla_bibliography,
            "chicago": self._format_chicago_bibliography,
            "ieee": self._format_ieee_bibliography,
            "harvard": self._format_harvard_bibliography,
        }
        return formatters.get(style, self._format_apa_bibliography)

    def _format_apa_bibliography(self, source: Source) -> str:
        """Format APA bibliography entry.

        Args:
            source: Source to format

        Returns:
            APA bibliography entry
        """
        parts = [f"{source.author} ({source.year})."]
        parts.append(f"*{source.title}*.")

        if source.journal:
            parts.append(f"*{source.journal}*")
            if source.volume:
                parts.append(f", {source.volume}")
            if source.issue:
                parts.append(f"({source.issue})")
            if source.page:
                parts.append(f", {source.page}")
            parts.append(".")

        if source.publisher:
            parts.append(f"{source.publisher}.")

        if source.doi:
            parts.append(f"https://doi.org/{source.doi}")

        return " ".join(parts)

    def _format_mla_bibliography(self, source: Source) -> str:
        """Format MLA bibliography entry.

        Args:
            source: Source to format

        Returns:
            MLA bibliography entry
        """
        parts = [f"{source.author}."]
        parts.append(f'"{source.title}."')

        if source.journal:
            parts.append(f"*{source.journal}*,")
            if source.volume:
                parts.append(f"vol. {source.volume},")
            if source.issue:
                parts.append(f"no. {source.issue},")
            parts.append(f"{source.year},")
            if source.page:
                parts.append(f"pp. {source.page}.")

        return " ".join(parts)

    def _format_chicago_bibliography(self, source: Source) -> str:
        """Format Chicago bibliography entry.

        Args:
            source: Source to format

        Returns:
            Chicago bibliography entry
        """
        parts = [f"{source.author}."]
        parts.append(f'"{source.title}."')

        if source.journal:
            parts.append(f"*{source.journal}*")
            if source.volume:
                parts.append(f" {source.volume}")
            if source.issue:
                parts.append(f", no. {source.issue}")
            parts.append(f" ({source.year})")
            if source.page:
                parts.append(f": {source.page}")
            parts.append(".")

        return " ".join(parts)

    def _format_ieee_bibliography(self, source: Source) -> str:
        """Format IEEE bibliography entry.

        Args:
            source: Source to format

        Returns:
            IEEE bibliography entry
        """
        parts = [f"[{source.id}]"]
        parts.append(f'{source.author}, "{source.title},"')

        if source.journal:
            parts.append(f"*{source.journal}*,")
            if source.volume:
                parts.append(f"vol. {source.volume},")
            if source.page:
                parts.append(f"pp. {source.page},")
            parts.append(f"{source.year}.")

        return " ".join(parts)

    def _format_harvard_bibliography(self, source: Source) -> str:
        """Format Harvard bibliography entry.

        Args:
            source: Source to format

        Returns:
            Harvard bibliography entry
        """
        parts = [f"{source.author} ({source.year})"]
        parts.append(f"'{source.title}',")

        if source.journal:
            parts.append(f"*{source.journal}*,")
            if source.volume:
                parts.append(f"{source.volume}")
            if source.issue:
                parts.append(f"({source.issue}),")
            if source.page:
                parts.append(f"pp. {source.page}.")

        return " ".join(parts)

    def _extract_citations_from_text(self, text: str) -> List[str]:
        """Extract citations from text.

        Args:
            text: Text to search

        Returns:
            List of citation strings
        """
        citations: List[str] = []

        # Common citation patterns
        patterns = [
            r"\(([^)]+),\s*\d{4}\)",  # (Author, Year)
            r"\(([^)]+)\s+\d+\)",  # (Author Page)
            r"\[(\d+)\]",  # [Number]
            r"\[[^\]]+\]",  # [Any reference]
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)

        return list(set(citations))

    def _build_source_lookup(
        self,
        sources: List[Source],
    ) -> Dict[str, Source]:
        """Build lookup dictionary for sources.

        Args:
            sources: Sources to index

        Returns:
            Dictionary mapping keys to sources
        """
        lookup: Dict[str, Source] = {}

        for source in sources:
            lookup[source.id.lower()] = source
            lookup[source.author.lower()] = source
            lookup[f"{source.author.lower()}, {source.year}"] = source

        return lookup

    def _verify_single_citation(
        self,
        citation: str,
        source_lookup: Dict[str, Source],
    ) -> Tuple[bool, str]:
        """Verify single citation.

        Args:
            citation: Citation string
            source_lookup: Source lookup dictionary

        Returns:
            Tuple of (is_valid, message)
        """
        citation_lower = citation.lower()

        # Check various forms
        if citation_lower in source_lookup:
            return True, ""

        # Check if any source matches
        for key in source_lookup:
            if key in citation_lower or citation_lower in key:
                return True, ""

        return False, f"No source found for: {citation}"

    def _extract_source_from_chunk(self, chunk: Any) -> Optional[Source]:
        """Extract source from chunk metadata.

        Args:
            chunk: Chunk object

        Returns:
            Source or None
        """
        metadata = self._get_metadata(chunk)
        if not metadata:
            return None

        author = metadata.get("author", "Unknown")
        title = metadata.get("title", metadata.get("source", "Untitled"))

        # Skip if no meaningful data
        if author == "Unknown" and title == "Untitled":
            return None

        return Source(
            id=metadata.get("source", metadata.get("document_id", f"src_{id(chunk)}")),
            author=author,
            title=title,
            year=self._extract_year(metadata),
            source_type=metadata.get("type", "article"),
            journal=metadata.get("journal"),
            publisher=metadata.get("publisher"),
            url=metadata.get("url"),
            page=metadata.get("page"),
            volume=metadata.get("volume"),
            issue=metadata.get("issue"),
            doi=metadata.get("doi"),
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

    def _extract_year(self, metadata: Dict[str, Any]) -> str:
        """Extract year from metadata.

        Args:
            metadata: Metadata dictionary

        Returns:
            Year string
        """
        year = metadata.get("year", metadata.get("date", "n.d."))
        if isinstance(year, str) and len(year) >= 4:
            return year[:4]
        return str(year) if year else "n.d."

    def _parse_bibtex_entry(
        self,
        entry_type: str,
        entry_id: str,
        content: str,
    ) -> Optional[Source]:
        """Parse single BibTeX entry.

        Args:
            entry_type: Entry type (article, book, etc.)
            entry_id: Entry identifier
            content: Entry content

        Returns:
            Source or None
        """
        fields: Dict[str, str] = {}

        for match in re.finditer(r"(\w+)\s*=\s*\{([^}]+)\}", content):
            fields[match.group(1).lower()] = match.group(2)

        if not fields:
            return None

        return Source(
            id=entry_id,
            author=fields.get("author", "Unknown"),
            title=fields.get("title", "Untitled"),
            year=fields.get("year", "n.d."),
            source_type=entry_type,
            journal=fields.get("journal"),
            publisher=fields.get("publisher"),
            url=fields.get("url"),
            page=fields.get("pages"),
            volume=fields.get("volume"),
            issue=fields.get("number"),
            doi=fields.get("doi"),
        )

    def _source_to_bibtex(self, source: Source) -> str:
        """Convert source to BibTeX entry.

        Args:
            source: Source to convert

        Returns:
            BibTeX entry string
        """
        entry_type = source.source_type if source.source_type else "article"
        lines = [f"@{entry_type}{{{source.id},"]
        lines.append(f"  author = {{{source.author}}},")
        lines.append(f"  title = {{{source.title}}},")
        lines.append(f"  year = {{{source.year}}},")

        if source.journal:
            lines.append(f"  journal = {{{source.journal}}},")
        if source.publisher:
            lines.append(f"  publisher = {{{source.publisher}}},")
        if source.volume:
            lines.append(f"  volume = {{{source.volume}}},")
        if source.issue:
            lines.append(f"  number = {{{source.issue}}},")
        if source.page:
            lines.append(f"  pages = {{{source.page}}},")
        if source.doi:
            lines.append(f"  doi = {{{source.doi}}},")
        if source.url:
            lines.append(f"  url = {{{source.url}}},")

        lines.append("}")
        return "\n".join(lines)


# ============================================================================
# Command Implementation
# ============================================================================


class CiteInsertCommand(WritingCommand):
    """Insert citations into document."""

    def execute(
        self,
        file_path: Path,
        output: Optional[Path] = None,
        project: Optional[Path] = None,
        style: str = "apa",
    ) -> int:
        """Execute citation insertion."""
        try:
            return self._execute_insert(file_path, output, project, style)
        except Exception as e:
            logger.error(f"Citation insertion failed: {e}")
            return self.handle_error(e, "Citation insertion failed")

    def _execute_insert(
        self,
        file_path: Path,
        output: Optional[Path],
        project: Optional[Path],
        style: str,
    ) -> int:
        """Internal insertion logic."""
        if not file_path.exists():
            self.print_error(f"File not found: {file_path}")
            return 1

        ctx = self.initialize_context(project, require_storage=True)
        chunks = (
            ctx["storage"].get_all_chunks()
            if hasattr(ctx["storage"], "get_all_chunks")
            else []
        )

        manager = CitationManager()
        sources = manager.extract_sources_from_chunks(chunks)

        text = file_path.read_text(encoding="utf-8")
        result = manager.insert_citations(text, sources, style)

        output_path = output or file_path.with_suffix(f".cited{file_path.suffix}")
        output_path.write_text(result, encoding="utf-8")

        self.print_success(f"Citations inserted: {output_path}")
        self.print_info(f"Style: {style}, Sources: {len(sources)}")

        return 0


class CiteFormatCommand(WritingCommand):
    """Format bibliography in specified style."""

    def execute(
        self,
        file_path: Path,
        output: Optional[Path] = None,
        project: Optional[Path] = None,
        style: str = "apa",
    ) -> int:
        """Execute bibliography formatting."""
        try:
            return self._execute_format(file_path, output, project, style)
        except Exception as e:
            logger.error(f"Bibliography formatting failed: {e}")
            return self.handle_error(e, "Bibliography formatting failed")

    def _execute_format(
        self,
        file_path: Path,
        output: Optional[Path],
        project: Optional[Path],
        style: str,
    ) -> int:
        """Internal formatting logic."""
        manager = CitationManager()

        # Check if it's a BibTeX file
        if file_path.suffix.lower() == ".bib":
            if not file_path.exists():
                self.print_error(f"File not found: {file_path}")
                return 1
            bibtex = file_path.read_text(encoding="utf-8")
            sources = manager.parse_bibtex(bibtex)
        else:
            # Use sources from storage
            ctx = self.initialize_context(project, require_storage=True)
            chunks = (
                ctx["storage"].get_all_chunks()
                if hasattr(ctx["storage"], "get_all_chunks")
                else []
            )
            sources = manager.extract_sources_from_chunks(chunks)

        if not sources:
            self.print_warning("No sources found")
            return 0

        bibliography = manager.format_bibliography(sources, style)

        if output:
            output.write_text(bibliography, encoding="utf-8")
            self.print_success(f"Bibliography saved to: {output}")
        else:
            self.console.print()
            self.console.print(
                Panel(
                    bibliography,
                    title=f"Bibliography ({style.upper()})",
                    border_style="green",
                )
            )

        self.print_info(f"Formatted {len(sources)} sources in {style.upper()} style")
        return 0


class CiteCheckCommand(WritingCommand):
    """Verify citations in document."""

    def execute(
        self,
        file_path: Path,
        project: Optional[Path] = None,
    ) -> int:
        """Execute citation verification."""
        try:
            return self._execute_check(file_path, project)
        except Exception as e:
            logger.error(f"Citation verification failed: {e}")
            return self.handle_error(e, "Citation verification failed")

    def _execute_check(
        self,
        file_path: Path,
        project: Optional[Path],
    ) -> int:
        """Internal check logic."""
        if not file_path.exists():
            self.print_error(f"File not found: {file_path}")
            return 1

        ctx = self.initialize_context(project, require_storage=True)
        chunks = (
            ctx["storage"].get_all_chunks()
            if hasattr(ctx["storage"], "get_all_chunks")
            else []
        )

        manager = CitationManager()
        sources = manager.extract_sources_from_chunks(chunks)
        text = file_path.read_text(encoding="utf-8")

        result = manager.verify_citations(text, sources)

        self._display_verification_result(result)

        return 0 if result.is_valid else 1

    def _display_verification_result(self, result: VerificationResult) -> None:
        """Display verification result."""
        self.console.print()

        status = "[green]VALID[/green]" if result.is_valid else "[red]INVALID[/red]"
        self.console.print(
            Panel(
                f"Status: {status}\n"
                f"Total Citations: {result.total_citations}\n"
                f"Valid: {result.valid_citations}\n"
                f"Missing: {len(result.missing_sources)}",
                title="Citation Verification",
                border_style="cyan",
            )
        )

        if result.missing_sources:
            self.console.print("\n[yellow]Missing Sources:[/yellow]")
            for source in result.missing_sources:
                self.console.print(f"  - {source}")

        if result.warnings:
            self.console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result.warnings:
                self.console.print(f"  - {warning}")


# ============================================================================
# CLI Command Functions
# ============================================================================


def insert_command(
    file_path: Path = typer.Argument(..., help="Document to add citations to"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    style: str = typer.Option(
        "apa",
        "-s",
        "--style",
        help="Citation style (apa, mla, chicago, ieee, harvard)",
    ),
) -> None:
    """Insert citations into document.

    Examples:
        # Insert APA citations
        ingestforge writing cite insert draft.md

        # Use MLA style
        ingestforge writing cite insert draft.md --style mla

        # Save to new file
        ingestforge writing cite insert draft.md -o cited_draft.md
    """
    exit_code = CiteInsertCommand().execute(file_path, output, project, style)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def format_command(
    file_path: Path = typer.Argument(..., help="BibTeX file or use storage sources"),
    output: Optional[Path] = typer.Option(None, "-o", "--output", help="Output file"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
    style: str = typer.Option(
        "apa",
        "-s",
        "--style",
        help="Citation style (apa, mla, chicago, ieee, harvard)",
    ),
) -> None:
    """Format bibliography in specified style.

    Examples:
        # Format BibTeX in APA
        ingestforge writing cite format references.bib

        # Format in MLA and save
        ingestforge writing cite format refs.bib -s mla -o bibliography.md
    """
    exit_code = CiteFormatCommand().execute(file_path, output, project, style)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


def check_command(
    file_path: Path = typer.Argument(..., help="Document to verify"),
    project: Optional[Path] = typer.Option(
        None, "-p", "--project", help="Project directory"
    ),
) -> None:
    """Verify citations in document.

    Examples:
        # Check citations in draft
        ingestforge writing cite check draft.md

        # Check with specific project
        ingestforge writing cite check paper.md -p ./my_project
    """
    exit_code = CiteCheckCommand().execute(file_path, project)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
