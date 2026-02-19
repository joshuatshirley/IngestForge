"""CSL Style Engine for bibliography generation.

Generates formatted citations and bibliographies using
Citation Style Language (CSL) specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_REFERENCES = 500
MAX_TITLE_LENGTH = 500
MAX_AUTHORS = 50
MAX_STYLE_NAME_LENGTH = 100


class CitationType(Enum):
    """Types of citations."""

    ARTICLE = "article-journal"
    BOOK = "book"
    CHAPTER = "chapter"
    CONFERENCE = "paper-conference"
    THESIS = "thesis"
    REPORT = "report"
    WEBPAGE = "webpage"
    DATASET = "dataset"


class OutputFormat(Enum):
    """Output formats for bibliography."""

    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"
    RTF = "rtf"


@dataclass
class Author:
    """Author information."""

    family: str
    given: str = ""
    suffix: str = ""

    def to_csl(self) -> dict[str, str]:
        """Convert to CSL JSON format."""
        result = {"family": self.family}

        if self.given:
            result["given"] = self.given
        if self.suffix:
            result["suffix"] = self.suffix

        return result

    def __str__(self) -> str:
        """Format as string."""
        if self.given:
            return f"{self.family}, {self.given}"
        return self.family


@dataclass
class DateParts:
    """Date components for citations."""

    year: int
    month: Optional[int] = None
    day: Optional[int] = None

    def to_csl(self) -> dict[str, Any]:
        """Convert to CSL JSON format."""
        parts = [self.year]

        if self.month:
            parts.append(self.month)
        if self.day:
            parts.append(self.day)

        return {"date-parts": [parts]}


@dataclass
class Reference:
    """A bibliographic reference."""

    id: str
    title: str
    authors: list[Author] = field(default_factory=list)
    citation_type: CitationType = CitationType.ARTICLE
    issued: Optional[DateParts] = None
    container_title: str = ""
    volume: str = ""
    issue: str = ""
    page: str = ""
    doi: str = ""
    url: str = ""
    publisher: str = ""
    abstract: str = ""

    def __post_init__(self) -> None:
        """Validate reference on creation."""
        self.title = self.title[:MAX_TITLE_LENGTH]
        self.authors = self.authors[:MAX_AUTHORS]

    def to_csl(self) -> dict[str, Any]:
        """Convert to CSL JSON format."""
        csl: dict[str, Any] = {
            "id": self.id,
            "type": self.citation_type.value,
            "title": self.title,
        }

        if self.authors:
            csl["author"] = [a.to_csl() for a in self.authors]

        if self.issued:
            csl["issued"] = self.issued.to_csl()

        if self.container_title:
            csl["container-title"] = self.container_title

        if self.volume:
            csl["volume"] = self.volume

        if self.issue:
            csl["issue"] = self.issue

        if self.page:
            csl["page"] = self.page

        if self.doi:
            csl["DOI"] = self.doi

        if self.url:
            csl["URL"] = self.url

        if self.publisher:
            csl["publisher"] = self.publisher

        return csl


@dataclass
class Citation:
    """A single citation instance."""

    reference_ids: list[str]
    prefix: str = ""
    suffix: str = ""
    locator: str = ""
    locator_type: str = "page"

    def to_csl(self) -> list[dict[str, Any]]:
        """Convert to CSL citation format."""
        items: list[dict[str, Any]] = []

        for ref_id in self.reference_ids:
            item: dict[str, Any] = {"id": ref_id}

            if self.prefix:
                item["prefix"] = self.prefix

            if self.suffix:
                item["suffix"] = self.suffix

            if self.locator:
                item["locator"] = self.locator
                item["label"] = self.locator_type

            items.append(item)

        return items


@dataclass
class Bibliography:
    """Generated bibliography output."""

    entries: list[str]
    format: OutputFormat
    style: str
    reference_count: int

    def to_string(self) -> str:
        """Convert to single string."""
        return "\n\n".join(self.entries)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entries": self.entries,
            "format": self.format.value,
            "style": self.style,
            "count": self.reference_count,
        }


class CSLEngine:
    """Citation Style Language engine.

    Generates formatted citations and bibliographies
    from reference data using CSL styles.
    """

    # Built-in style templates
    STYLES = {
        "apa": "American Psychological Association 7th edition",
        "mla": "Modern Language Association 9th edition",
        "chicago": "Chicago Manual of Style 17th edition",
        "ieee": "IEEE",
        "harvard": "Harvard Reference format 1",
        "vancouver": "Vancouver",
    }

    def __init__(
        self,
        style: str = "apa",
        locale: str = "en-US",
    ) -> None:
        """Initialize the engine.

        Args:
            style: Citation style name
            locale: Locale for formatting
        """
        self.style = style[:MAX_STYLE_NAME_LENGTH]
        self.locale = locale
        self._references: dict[str, Reference] = {}

    @property
    def reference_count(self) -> int:
        """Number of stored references."""
        return len(self._references)

    @property
    def reference_ids(self) -> list[str]:
        """List of reference IDs."""
        return list(self._references.keys())

    def add_reference(self, reference: Reference) -> bool:
        """Add a reference to the engine.

        Args:
            reference: Reference to add

        Returns:
            True if added
        """
        if len(self._references) >= MAX_REFERENCES:
            logger.warning(f"Max references ({MAX_REFERENCES}) reached")
            return False

        self._references[reference.id] = reference
        return True

    def remove_reference(self, ref_id: str) -> bool:
        """Remove a reference.

        Args:
            ref_id: Reference ID

        Returns:
            True if removed
        """
        if ref_id not in self._references:
            return False

        del self._references[ref_id]
        return True

    def get_reference(self, ref_id: str) -> Optional[Reference]:
        """Get a reference by ID.

        Args:
            ref_id: Reference ID

        Returns:
            Reference or None
        """
        return self._references.get(ref_id)

    def format_citation(
        self,
        citation: Citation,
        format: OutputFormat = OutputFormat.TEXT,
    ) -> str:
        """Format a single citation.

        Args:
            citation: Citation to format
            format: Output format

        Returns:
            Formatted citation string
        """
        if not citation.reference_ids:
            return ""

        # Get references
        refs = [
            self._references[rid]
            for rid in citation.reference_ids
            if rid in self._references
        ]

        if not refs:
            return "[?]"

        # Format based on style
        return self._format_inline_citation(refs, format)

    def generate_bibliography(
        self,
        ref_ids: Optional[list[str]] = None,
        format: OutputFormat = OutputFormat.TEXT,
    ) -> Bibliography:
        """Generate a bibliography.

        Args:
            ref_ids: Specific references (or all if None)
            format: Output format

        Returns:
            Generated bibliography
        """
        # Get references
        if ref_ids:
            refs = [self._references[rid] for rid in ref_ids if rid in self._references]
        else:
            refs = list(self._references.values())

        # Sort by author/year
        refs = self._sort_references(refs)

        # Format entries
        entries = [self._format_entry(ref, format) for ref in refs]

        return Bibliography(
            entries=entries,
            format=format,
            style=self.style,
            reference_count=len(entries),
        )

    def _format_inline_citation(
        self,
        refs: list[Reference],
        format: OutputFormat,
    ) -> str:
        """Format inline citation.

        Args:
            refs: References to cite
            format: Output format

        Returns:
            Formatted string
        """
        if self.style in ("apa", "harvard"):
            return self._format_author_year(refs, format)

        if self.style == "ieee":
            return self._format_numeric(refs, format)

        return self._format_author_year(refs, format)

    def _format_author_year(
        self,
        refs: list[Reference],
        format: OutputFormat,
    ) -> str:
        """Format as author-year citation.

        Args:
            refs: References
            format: Output format

        Returns:
            Formatted string
        """
        parts: list[str] = []

        for ref in refs:
            author = ref.authors[0].family if ref.authors else "Unknown"
            year = ref.issued.year if ref.issued else "n.d."
            parts.append(f"{author}, {year}")

        citation = "; ".join(parts)
        return f"({citation})"

    def _format_numeric(
        self,
        refs: list[Reference],
        format: OutputFormat,
    ) -> str:
        """Format as numeric citation.

        Args:
            refs: References
            format: Output format

        Returns:
            Formatted string
        """
        # Get indices
        ref_list = list(self._references.keys())
        indices = []

        for ref in refs:
            if ref.id in ref_list:
                idx = ref_list.index(ref.id) + 1
                indices.append(str(idx))

        return f"[{', '.join(indices)}]"

    def _format_entry(
        self,
        ref: Reference,
        format: OutputFormat,
    ) -> str:
        """Format a bibliography entry.

        Args:
            ref: Reference to format
            format: Output format

        Returns:
            Formatted entry
        """
        if self.style == "apa":
            return self._format_apa_entry(ref, format)

        if self.style == "mla":
            return self._format_mla_entry(ref, format)

        if self.style == "ieee":
            return self._format_ieee_entry(ref, format)

        return self._format_apa_entry(ref, format)

    def _format_apa_entry(
        self,
        ref: Reference,
        format: OutputFormat,
    ) -> str:
        """Format entry in APA style.

        Args:
            ref: Reference
            format: Output format

        Returns:
            Formatted entry
        """
        parts: list[str] = []

        # Authors
        if ref.authors:
            author_strs = [str(a) for a in ref.authors[:3]]
            if len(ref.authors) > 3:
                author_strs.append("et al.")
            parts.append(", ".join(author_strs))

        # Year
        if ref.issued:
            parts.append(f"({ref.issued.year})")
        else:
            parts.append("(n.d.)")

        # Title
        if format == OutputFormat.HTML:
            parts.append(f"<i>{ref.title}</i>.")
        elif format == OutputFormat.MARKDOWN:
            parts.append(f"*{ref.title}*.")
        else:
            parts.append(f"{ref.title}.")

        # Journal/Container
        if ref.container_title:
            if format == OutputFormat.HTML:
                parts.append(f"<i>{ref.container_title}</i>")
            elif format == OutputFormat.MARKDOWN:
                parts.append(f"*{ref.container_title}*")
            else:
                parts.append(ref.container_title)

        # Volume/Issue/Pages
        loc_parts: list[str] = []
        if ref.volume:
            loc_parts.append(ref.volume)
        if ref.issue:
            loc_parts.append(f"({ref.issue})")
        if ref.page:
            loc_parts.append(ref.page)
        if loc_parts:
            parts.append(", ".join(loc_parts) + ".")

        # DOI
        if ref.doi:
            parts.append(f"https://doi.org/{ref.doi}")

        return " ".join(parts)

    def _format_mla_entry(
        self,
        ref: Reference,
        format: OutputFormat,
    ) -> str:
        """Format entry in MLA style.

        Args:
            ref: Reference
            format: Output format

        Returns:
            Formatted entry
        """
        parts: list[str] = []

        # Authors (Last, First format)
        if ref.authors:
            parts.append(str(ref.authors[0]) + ".")

        # Title in quotes for articles
        if ref.citation_type == CitationType.ARTICLE:
            parts.append(f'"{ref.title}."')
        else:
            if format == OutputFormat.HTML:
                parts.append(f"<i>{ref.title}</i>.")
            elif format == OutputFormat.MARKDOWN:
                parts.append(f"*{ref.title}*.")
            else:
                parts.append(f"{ref.title}.")

        # Container
        if ref.container_title:
            if format == OutputFormat.HTML:
                parts.append(f"<i>{ref.container_title}</i>,")
            elif format == OutputFormat.MARKDOWN:
                parts.append(f"*{ref.container_title}*,")
            else:
                parts.append(f"{ref.container_title},")

        # Volume/Issue
        if ref.volume:
            parts.append(f"vol. {ref.volume},")
        if ref.issue:
            parts.append(f"no. {ref.issue},")

        # Year
        if ref.issued:
            parts.append(f"{ref.issued.year},")

        # Pages
        if ref.page:
            parts.append(f"pp. {ref.page}.")

        return " ".join(parts)

    def _format_ieee_entry(
        self,
        ref: Reference,
        format: OutputFormat,
    ) -> str:
        """Format entry in IEEE style.

        Args:
            ref: Reference
            format: Output format

        Returns:
            Formatted entry
        """
        parts: list[str] = []

        # Authors (initials first)
        if ref.authors:
            author_strs: list[str] = []
            for author in ref.authors[:3]:
                initials = author.given[0] + "." if author.given else ""
                author_strs.append(f"{initials} {author.family}")
            parts.append(", ".join(author_strs) + ",")

        # Title in quotes
        parts.append(f'"{ref.title},"')

        # Container italic
        if ref.container_title:
            if format == OutputFormat.HTML:
                parts.append(f"<i>{ref.container_title}</i>,")
            elif format == OutputFormat.MARKDOWN:
                parts.append(f"*{ref.container_title}*,")
            else:
                parts.append(f"{ref.container_title},")

        # Volume/Pages
        if ref.volume:
            parts.append(f"vol. {ref.volume},")
        if ref.page:
            parts.append(f"pp. {ref.page},")

        # Year
        if ref.issued:
            parts.append(f"{ref.issued.year}.")

        return " ".join(parts)

    def _sort_references(
        self,
        refs: list[Reference],
    ) -> list[Reference]:
        """Sort references by author/year.

        Args:
            refs: References to sort

        Returns:
            Sorted list
        """

        def sort_key(ref: Reference) -> tuple[str, int]:
            author = ref.authors[0].family if ref.authors else "ZZZ"
            year = ref.issued.year if ref.issued else 9999
            return (author.lower(), year)

        return sorted(refs, key=sort_key)

    def clear(self) -> None:
        """Remove all references."""
        self._references.clear()


def create_engine(
    style: str = "apa",
    locale: str = "en-US",
) -> CSLEngine:
    """Factory function to create CSL engine.

    Args:
        style: Citation style
        locale: Locale

    Returns:
        Configured engine
    """
    return CSLEngine(style=style, locale=locale)


def format_references(
    references: list[Reference],
    style: str = "apa",
    format: OutputFormat = OutputFormat.TEXT,
) -> str:
    """Convenience function to format references.

    Args:
        references: References to format
        style: Citation style
        format: Output format

    Returns:
        Formatted bibliography string
    """
    engine = create_engine(style=style)

    for ref in references:
        engine.add_reference(ref)

    bib = engine.generate_bibliography(format=format)
    return bib.to_string()
