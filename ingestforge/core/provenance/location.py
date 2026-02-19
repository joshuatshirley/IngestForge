"""
SourceLocation class for tracking content provenance.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional, List, TYPE_CHECKING
import hashlib
import re

from ingestforge.core.logging import get_logger
from ingestforge.core.provenance.models import SourceType, CitationStyle
from ingestforge.core.provenance.author import Author

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ingestforge.core.source_linker import SourceLink


@dataclass
class SourceLocation:
    """
    Precise location within a source for citations.

    Tracks the exact position of content down to chapter, section,
    page, and paragraph level for accurate citations.

    Examples:
        # Simple web article
        loc = SourceLocation(
            source_type=SourceType.WEBPAGE,
            title="Introduction to Quantum Computing",
            url="https://example.com/quantum",
            authors=[Author("Jane Smith")],
        )

        # Precise PDF citation
        loc = SourceLocation(
            source_type=SourceType.PDF,
            title="Quantum Computing: An Applied Approach",
            authors=[Author("Jack Hidary")],
            publication_date="2019",
            chapter="Chapter 3: Quantum Gates",
            section="3.2 Single-Qubit Gates",
            page_start=47,
            page_end=48,
            paragraph_number=3,
        )
        print(loc.to_short_cite())  # [Hidary 2019, Ch.3 §3.2, p.47]
    """

    # Source identification
    source_id: str = ""
    source_type: SourceType = SourceType.UNKNOWN

    # Bibliographic information
    title: str = ""
    authors: List[Author] = field(default_factory=list)
    publication_date: Optional[str] = None  # Year or full date
    publisher: Optional[str] = None
    url: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None

    # Access information
    accessed_date: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d")
    )
    file_path: Optional[str] = None

    # Structural location (hierarchy)
    chapter: Optional[str] = None  # "Chapter 3: Quantum Gates"
    chapter_number: Optional[int] = None
    section: Optional[str] = None  # "3.2 Single-Qubit Gates"
    section_number: Optional[str] = None  # "3.2"
    subsection: Optional[str] = None  # "3.2.1 The Hadamard Gate"

    # Precise location
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    paragraph_number: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    # For video/audio sources
    timestamp_start: Optional[str] = None  # "14:32"
    timestamp_end: Optional[str] = None

    # Content hash for deduplication
    content_hash: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate source_id if not provided."""
        if not self.source_id:
            self.source_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique ID for this source."""
        components = [
            self.title,
            self.url or self.file_path or "",
            ",".join(a.name for a in self.authors),
            self.publication_date or "",
        ]
        content = "|".join(components)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    @property
    def page_range(self) -> Optional[str]:
        """Get page range as string."""
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return str(self.page_start)
        return f"{self.page_start}-{self.page_end}"

    @property
    def primary_author(self) -> Optional[Author]:
        """Get the first author."""
        return self.authors[0] if self.authors else None

    @property
    def year(self) -> Optional[str]:
        """Extract year from publication_date."""
        if not self.publication_date:
            return None
        # Try to extract 4-digit year
        match = re.search(r"\b(19|20)\d{2}\b", self.publication_date)
        if match:
            return match.group()
        return self.publication_date

    def to_short_cite(self) -> str:
        """
        Generate short inline citation.

        Examples:
            [Smith 2023, p.47]
            [Smith & Jones 2023, Ch.3, p.12-15]
            [Wikipedia: Quantum Computing, §2.1]
        """
        author_year = self._format_author_for_cite()
        location_parts = self._format_location_for_cite()

        if location_parts:
            return f"[{author_year}, {', '.join(location_parts)}]"
        return f"[{author_year}]"

    def _format_author_for_cite(self) -> str:
        """Format author and year for short citation."""
        parts = []

        if self.authors:
            if len(self.authors) == 1:
                parts.append(self.authors[0].last_name or self.authors[0].name)
            elif len(self.authors) == 2:
                parts.append(
                    f"{self.authors[0].last_name} & {self.authors[1].last_name}"
                )
            else:
                parts.append(f"{self.authors[0].last_name} et al.")
        elif self.source_type == SourceType.WEBPAGE:
            parts.append(self._extract_site_name() or self.title[:30])
        else:
            parts.append(self.title[:30] if self.title else "Unknown")

        if self.year:
            parts.append(self.year)

        return " ".join(parts)

    def _format_location_for_cite(self) -> List[str]:
        """Format location details for short citation."""
        location_parts = []

        if self.chapter_number:
            location_parts.append(f"Ch.{self.chapter_number}")
        elif self.chapter:
            match = re.search(r"Chapter\s+(\d+)", self.chapter, re.IGNORECASE)
            if match:
                location_parts.append(f"Ch.{match.group(1)}")

        if self.section_number:
            location_parts.append(f"§{self.section_number}")
        elif self.section:
            match = re.search(r"^(\d+(?:\.\d+)*)", self.section)
            if match:
                location_parts.append(f"§{match.group(1)}")

        if self.page_range:
            location_parts.append(f"p.{self.page_range}")

        if self.paragraph_number:
            location_parts.append(f"¶{self.paragraph_number}")

        if self.timestamp_start:
            location_parts.append(self.timestamp_start)

        return location_parts

    def _extract_site_name(self) -> Optional[str]:
        """Extract site name from URL."""
        if not self.url:
            return None
        # Simple extraction - get domain without www
        match = re.search(r"://(?:www\.)?([^/]+)", self.url)
        if match:
            domain = match.group(1)
            # Special cases
            if "wikipedia" in domain:
                return "Wikipedia"
            if "arxiv" in domain:
                return "arXiv"
            # Return capitalized domain name
            return domain.split(".")[0].title()
        return None

    def to_citation(self, style: CitationStyle = CitationStyle.APA) -> str:
        """
        Generate full citation in specified style.

        Rule #1: Dictionary dispatch eliminates nesting

        Args:
            style: Citation style (apa, mla, chicago, harvard, ieee, bibtex)

        Returns:
            Formatted citation string
        """
        formatters = {
            CitationStyle.APA: self._format_apa,
            CitationStyle.MLA: self._format_mla,
            CitationStyle.CHICAGO: self._format_chicago,
            CitationStyle.BIBTEX: self._format_bibtex,
        }

        formatter = formatters.get(style, self._format_apa)
        return formatter()

    def _format_apa(self) -> str:
        """Format citation in APA 7th edition style."""
        parts = []

        # Authors
        if self.authors:
            author_strs = [a.format_apa() for a in self.authors]
            if len(author_strs) == 1:
                parts.append(author_strs[0])
            elif len(author_strs) == 2:
                parts.append(f"{author_strs[0]} & {author_strs[1]}")
            else:
                parts.append(f"{', '.join(author_strs[:-1])}, & {author_strs[-1]}")

        # Year
        if self.year:
            parts.append(f"({self.year}).")
        else:
            parts.append("(n.d.).")

        # Title
        if self.title:
            if self.source_type in [SourceType.WEBPAGE, SourceType.PDF]:
                parts.append(f"*{self.title}*.")
            else:
                parts.append(f"{self.title}.")

        # Publisher
        if self.publisher:
            parts.append(f"{self.publisher}.")

        # URL
        if self.url:
            parts.append(self.url)

        return " ".join(parts)

    def _format_mla(self) -> str:
        """Format citation in MLA 9th edition style."""
        parts = []

        # Authors (Last, First format)
        if self.authors:
            if len(self.authors) == 1:
                a = self.authors[0]
                if a.first_name and a.last_name:
                    parts.append(f"{a.last_name}, {a.first_name}.")
                else:
                    parts.append(f"{a.name}.")
            else:
                # First author inverted, rest normal
                first = self.authors[0]
                if first.first_name and first.last_name:
                    author_str = f"{first.last_name}, {first.first_name}"
                else:
                    author_str = first.name
                for a in self.authors[1:]:
                    author_str += f", and {a.format_full()}"
                parts.append(f"{author_str}.")

        # Title in quotes for articles, italics for books
        if self.title:
            if self.source_type == SourceType.WEBPAGE:
                parts.append(f'"{self.title}."')
            else:
                parts.append(f"*{self.title}*.")

        # Publisher
        if self.publisher:
            parts.append(f"{self.publisher},")

        # Year
        if self.year:
            parts.append(f"{self.year}.")

        # URL
        if self.url:
            parts.append(self.url)

        return " ".join(parts)

    def _format_chicago(self) -> str:
        """Format citation in Chicago style (notes-bibliography)."""
        parts = []

        # Authors (First Last format)
        if self.authors:
            author_strs = [a.format_full() for a in self.authors]
            if len(author_strs) == 1:
                parts.append(f"{author_strs[0]}.")
            elif len(author_strs) == 2:
                parts.append(f"{author_strs[0]} and {author_strs[1]}.")
            else:
                parts.append(f"{', '.join(author_strs[:-1])}, and {author_strs[-1]}.")

        # Title
        if self.title:
            parts.append(f"*{self.title}*.")

        # Publisher and year
        if self.publisher and self.year:
            parts.append(f"{self.publisher}, {self.year}.")
        elif self.year:
            parts.append(f"{self.year}.")

        # URL with access date
        if self.url:
            parts.append(f"Accessed {self.accessed_date}. {self.url}.")

        return " ".join(parts)

    def _format_bibtex(self) -> str:
        """Format citation as BibTeX entry."""
        # Determine entry type
        if self.source_type == SourceType.WEBPAGE:
            entry_type = "misc"
        elif self.source_type == SourceType.PDF:
            entry_type = "article" if not self.isbn else "book"
        else:
            entry_type = "misc"

        # Generate cite key
        author_key = ""
        if self.authors:
            author_key = self.authors[0].last_name or "unknown"
            author_key = re.sub(r"[^a-zA-Z]", "", author_key).lower()
        year_key = self.year or "nd"
        title_word = ""
        if self.title:
            words = re.findall(r"[a-zA-Z]+", self.title)
            title_word = words[0].lower() if words else ""
        cite_key = f"{author_key}{year_key}{title_word}"

        # Build entry
        lines = [f"@{entry_type}{{{cite_key},"]

        if self.authors:
            author_str = " and ".join(a.format_full() for a in self.authors)
            lines.append(f"  author = {{{author_str}}},")

        if self.title:
            lines.append(f"  title = {{{self.title}}},")

        if self.year:
            lines.append(f"  year = {{{self.year}}},")

        if self.publisher:
            lines.append(f"  publisher = {{{self.publisher}}},")

        if self.url:
            lines.append(f"  url = {{{self.url}}},")

        if self.doi:
            lines.append(f"  doi = {{{self.doi}}},")

        if self.isbn:
            lines.append(f"  isbn = {{{self.isbn}}},")

        lines.append("}")

        return "\n".join(lines)

    def to_link(self, prefer_vscode: bool = False) -> "SourceLink":
        """
        Generate a clickable source link for this location.

        Creates a file:// URI (or vscode:// if preferred) that can be clicked
        to open the source document at the relevant location.

        Args:
            prefer_vscode: If True, generate vscode:// URIs instead of file://

        Returns:
            SourceLink with URI, file existence status, and navigation hint

        Example:
            loc = SourceLocation(
                file_path="/docs/thesis.pdf",
                page_start=47,
                chapter="Chapter 3",
            )
            link = loc.to_link()
            print(link.uri)  # file:///docs/thesis.pdf#page=47
            print(link.navigation_hint)  # Chapter 3 -> Page 47
            print(link.file_exists)  # True/False
        """
        from ingestforge.core.source_linker import SourceLinker

        linker = SourceLinker(prefer_vscode=prefer_vscode)
        return linker.generate_link(
            file_path=self.file_path,
            page_start=self.page_start,
            line_start=self.line_start,
            chapter=self.chapter,
            chapter_number=self.chapter_number,
            section=self.section,
            section_number=self.section_number,
            subsection=self.subsection,
            paragraph_number=self.paragraph_number,
            url=self.url,
            source_type=self.source_type.value if self.source_type else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "title": self.title,
            "authors": [
                {"name": a.name, "first_name": a.first_name, "last_name": a.last_name}
                for a in self.authors
            ],
            "publication_date": self.publication_date,
            "publisher": self.publisher,
            "url": self.url,
            "doi": self.doi,
            "isbn": self.isbn,
            "accessed_date": self.accessed_date,
            "file_path": self.file_path,
            "chapter": self.chapter,
            "chapter_number": self.chapter_number,
            "section": self.section,
            "section_number": self.section_number,
            "subsection": self.subsection,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "paragraph_number": self.paragraph_number,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceLocation":
        """Create from dictionary."""
        authors = cls._parse_authors_from_dict(data)
        source_type = cls._parse_source_type_from_dict(data)

        return cls(
            source_id=data.get("source_id", ""),
            source_type=source_type,
            title=data.get("title", ""),
            authors=authors,
            publication_date=data.get("publication_date"),
            publisher=data.get("publisher"),
            url=data.get("url"),
            doi=data.get("doi"),
            isbn=data.get("isbn"),
            accessed_date=data.get(
                "accessed_date", datetime.now().strftime("%Y-%m-%d")
            ),
            file_path=data.get("file_path"),
            chapter=data.get("chapter"),
            chapter_number=data.get("chapter_number"),
            section=data.get("section"),
            section_number=data.get("section_number"),
            subsection=data.get("subsection"),
            page_start=data.get("page_start"),
            page_end=data.get("page_end"),
            paragraph_number=data.get("paragraph_number"),
            line_start=data.get("line_start"),
            line_end=data.get("line_end"),
            timestamp_start=data.get("timestamp_start"),
            timestamp_end=data.get("timestamp_end"),
            content_hash=data.get("content_hash"),
        )

    @classmethod
    def _parse_author_none(cls) -> Author:
        """
        Parse None author entry.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        logger.warning("Found None author entry, using placeholder")
        return Author(name="Unknown")

    @classmethod
    def _parse_author_object(cls, author: Author) -> Author:
        """
        Parse Author object (pass through).

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        return author

    @classmethod
    def _parse_author_dict(cls, author_dict: dict[str, Any]) -> Author:
        """
        Parse author from dictionary.

        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints
        """
        return Author(
            name=author_dict.get("name", "Unknown"),
            first_name=author_dict.get("first_name"),
            last_name=author_dict.get("last_name"),
            affiliation=author_dict.get("affiliation"),
        )

    @classmethod
    def _parse_author_str(cls, author_str: str) -> Author:
        """
        Parse author from string.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        return Author(name=author_str)

    @classmethod
    def _parse_author_unknown(cls, author: Any) -> Author:
        """
        Parse unknown author type with fallback.

        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        logger.warning(f"Unknown author type {type(author)}, using placeholder")
        return Author(name="Unknown")

    @classmethod
    def _get_author_parser(cls, author: Any) -> Callable[[Any], Author]:
        """
        Get parser function for author based on type.

        Rule #1: Dictionary dispatch eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            author: Author data to parse

        Returns:
            Parser function for this author type
        """
        if author is None:
            return lambda _: cls._parse_author_none()
        if isinstance(author, Author):
            return cls._parse_author_object
        if isinstance(author, dict):
            return cls._parse_author_dict
        if isinstance(author, str):
            return cls._parse_author_str

        # Default for unknown types
        return cls._parse_author_unknown

    @classmethod
    def _parse_single_author(cls, author: Any) -> Author:
        """
        Parse single author with defensive error handling.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation via try/except
        Rule #9: Full type hints

        Args:
            author: Author data in any supported format

        Returns:
            Parsed Author object
        """
        try:
            parser = cls._get_author_parser(author)
            return parser(author)
        except Exception as e:
            logger.warning(f"Failed to parse author {author}: {e}")
            return Author(name="Unknown")

    @classmethod
    def _parse_authors_from_dict(cls, data: dict[str, Any]) -> List[Author]:
        """
        Parse authors list from dict with defensive error handling.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            data: Dictionary containing authors list

        Returns:
            List of parsed Author objects
        """
        authors = []
        for author in data.get("authors", []):
            parsed_author = cls._parse_single_author(author)
            authors.append(parsed_author)

        return authors

    @classmethod
    def _parse_source_type_from_dict(cls, data: dict[str, Any]) -> SourceType:
        """Parse source_type from dict with fallback to UNKNOWN."""
        try:
            return SourceType(data.get("source_type", "unknown"))
        except ValueError:
            logger.warning(
                f"Unknown source_type: {data.get('source_type')}, defaulting to UNKNOWN"
            )
            return SourceType.UNKNOWN
