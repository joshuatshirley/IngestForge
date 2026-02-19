"""Citation export formats - BibTeX and RIS formatters.

CITE-002.1: Technical citation export to standard bibliography formats.

Supports:
- BibTeX (.bib) for LaTeX/academic use
- RIS (.ris) for reference managers (Zotero, Mendeley, EndNote)"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class CitationType(Enum):
    """Types of citations supported."""

    ARTICLE = "article"
    BOOK = "book"
    INPROCEEDINGS = "inproceedings"
    INCOLLECTION = "incollection"
    TECHREPORT = "techreport"
    THESIS = "thesis"
    MISC = "misc"
    WEBPAGE = "webpage"


# BibTeX type mapping
BIBTEX_TYPES: Dict[CitationType, str] = {
    CitationType.ARTICLE: "article",
    CitationType.BOOK: "book",
    CitationType.INPROCEEDINGS: "inproceedings",
    CitationType.INCOLLECTION: "incollection",
    CitationType.TECHREPORT: "techreport",
    CitationType.THESIS: "phdthesis",
    CitationType.MISC: "misc",
    CitationType.WEBPAGE: "misc",
}

# RIS type mapping
RIS_TYPES: Dict[CitationType, str] = {
    CitationType.ARTICLE: "JOUR",
    CitationType.BOOK: "BOOK",
    CitationType.INPROCEEDINGS: "CONF",
    CitationType.INCOLLECTION: "CHAP",
    CitationType.TECHREPORT: "RPRT",
    CitationType.THESIS: "THES",
    CitationType.MISC: "GEN",
    CitationType.WEBPAGE: "ELEC",
}


@dataclass
class Citation:
    """Represents a citation with metadata.

    Attributes:
        title: Title of the work
        authors: List of author names
        year: Publication year
        citation_type: Type of citation
        journal: Journal name (for articles)
        volume: Volume number
        issue: Issue number
        pages: Page range (e.g., "1-10")
        publisher: Publisher name
        doi: Digital Object Identifier
        url: Web URL
        abstract: Abstract text
        keywords: List of keywords
        source_file: Original source file
        chunk_id: Associated chunk ID
    """

    title: str
    authors: List[str] = field(default_factory=list)
    year: Optional[int] = None
    citation_type: CitationType = CitationType.MISC
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    publisher: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    source_file: Optional[str] = None
    chunk_id: Optional[str] = None

    def generate_key(self) -> str:
        """Generate a unique citation key.

        Returns:
            Citation key like "smith2024" or "unknown2024"
        """
        # Use first author's last name
        if self.authors:
            first_author = self.authors[0]
            # Extract last name (simple heuristic)
            parts = first_author.strip().split()
            last_name = parts[-1] if parts else "unknown"
            # Clean for use as key
            last_name = re.sub(r"[^a-zA-Z]", "", last_name).lower()
        else:
            last_name = "unknown"

        year = self.year or datetime.now().year

        return f"{last_name}{year}"


class BibTeXFormatter:
    """Formats citations as BibTeX entries."""

    def format(self, citation: Citation) -> str:
        """Format a single citation as BibTeX.

        Args:
            citation: Citation to format

        Returns:
            BibTeX entry string
        """
        entry_type = BIBTEX_TYPES.get(citation.citation_type, "misc")
        key = citation.generate_key()

        lines = [f"@{entry_type}{{{key},"]

        # Required fields
        lines.append(f"  title = {{{self._escape(citation.title)}}},")

        # Authors
        if citation.authors:
            author_str = " and ".join(citation.authors)
            lines.append(f"  author = {{{self._escape(author_str)}}},")

        # Year
        if citation.year:
            lines.append(f"  year = {{{citation.year}}},")

        # Optional fields
        if citation.journal:
            lines.append(f"  journal = {{{self._escape(citation.journal)}}},")

        if citation.volume:
            lines.append(f"  volume = {{{citation.volume}}},")

        if citation.issue:
            lines.append(f"  number = {{{citation.issue}}},")

        if citation.pages:
            lines.append(f"  pages = {{{citation.pages}}},")

        if citation.publisher:
            lines.append(f"  publisher = {{{self._escape(citation.publisher)}}},")

        if citation.doi:
            lines.append(f"  doi = {{{citation.doi}}},")

        if citation.url:
            lines.append(f"  url = {{{citation.url}}},")

        if citation.abstract:
            # Truncate long abstracts
            abstract = citation.abstract[:500]
            lines.append(f"  abstract = {{{self._escape(abstract)}}},")

        if citation.keywords:
            keywords_str = ", ".join(citation.keywords)
            lines.append(f"  keywords = {{{self._escape(keywords_str)}}},")

        lines.append("}")

        return "\n".join(lines)

    def format_many(self, citations: List[Citation]) -> str:
        """Format multiple citations as BibTeX.

        Args:
            citations: List of citations

        Returns:
            Complete BibTeX file content
        """
        entries = [self.format(c) for c in citations]
        return "\n\n".join(entries)

    def _escape(self, text: str) -> str:
        """Escape special BibTeX characters.

        Args:
            text: Text to escape

        Returns:
            Escaped text
        """
        # Replace special characters
        replacements = [
            ("&", r"\&"),
            ("%", r"\%"),
            ("_", r"\_"),
            ("#", r"\#"),
            ("$", r"\$"),
        ]

        result = text
        for old, new in replacements:
            result = result.replace(old, new)

        return result


class RISFormatter:
    """Formats citations as RIS entries."""

    def format(self, citation: Citation) -> str:
        """Format a single citation as RIS.

        Args:
            citation: Citation to format

        Returns:
            RIS entry string
        """
        entry_type = RIS_TYPES.get(citation.citation_type, "GEN")

        lines = [f"TY  - {entry_type}"]

        # Title
        lines.append(f"TI  - {citation.title}")

        # Authors (one per line)
        for author in citation.authors:
            lines.append(f"AU  - {author}")

        # Year
        if citation.year:
            lines.append(f"PY  - {citation.year}")

        # Journal
        if citation.journal:
            lines.append(f"JO  - {citation.journal}")

        # Volume
        if citation.volume:
            lines.append(f"VL  - {citation.volume}")

        # Issue
        if citation.issue:
            lines.append(f"IS  - {citation.issue}")

        # Pages
        if citation.pages:
            # RIS uses SP (start page) and EP (end page)
            if "-" in citation.pages:
                start, end = citation.pages.split("-", 1)
                lines.append(f"SP  - {start.strip()}")
                lines.append(f"EP  - {end.strip()}")
            else:
                lines.append(f"SP  - {citation.pages}")

        # Publisher
        if citation.publisher:
            lines.append(f"PB  - {citation.publisher}")

        # DOI
        if citation.doi:
            lines.append(f"DO  - {citation.doi}")

        # URL
        if citation.url:
            lines.append(f"UR  - {citation.url}")

        # Abstract
        if citation.abstract:
            lines.append(f"AB  - {citation.abstract}")

        # Keywords
        for keyword in citation.keywords:
            lines.append(f"KW  - {keyword}")

        # End of record
        lines.append("ER  - ")

        return "\n".join(lines)

    def format_many(self, citations: List[Citation]) -> str:
        """Format multiple citations as RIS.

        Args:
            citations: List of citations

        Returns:
            Complete RIS file content
        """
        entries = [self.format(c) for c in citations]
        return "\n\n".join(entries)


def format_bibtex(citation: Citation) -> str:
    """Convenience function to format a single citation as BibTeX.

    Args:
        citation: Citation to format

    Returns:
        BibTeX entry string
    """
    return BibTeXFormatter().format(citation)


def format_ris(citation: Citation) -> str:
    """Convenience function to format a single citation as RIS.

    Args:
        citation: Citation to format

    Returns:
        RIS entry string
    """
    return RISFormatter().format(citation)


def export_bibliography(
    citations: List[Citation],
    output_path: Path,
    format: str = "bibtex",
) -> None:
    """Export citations to a bibliography file.

    Args:
        citations: List of citations to export
        output_path: Output file path
        format: Export format ("bibtex" or "ris")

    Raises:
        ValueError: If format is not supported
    """
    if format.lower() == "bibtex":
        formatter = BibTeXFormatter()
        content = formatter.format_many(citations)
    elif format.lower() == "ris":
        formatter = RISFormatter()
        content = formatter.format_many(citations)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'bibtex' or 'ris'.")

    output_path.write_text(content, encoding="utf-8")


def citation_from_metadata(metadata: Dict[str, Any]) -> Citation:
    """Create a Citation from chunk metadata.

    Args:
        metadata: Metadata dictionary from chunk

    Returns:
        Citation object
    """
    # Extract authors
    authors = _extract_authors(metadata)

    # Determine citation type
    citation_type = _determine_citation_type(metadata.get("source", ""))

    # Extract year
    year = _extract_year(metadata)

    return Citation(
        title=metadata.get("title", metadata.get("source", "Untitled")),
        authors=authors,
        year=year,
        citation_type=citation_type,
        journal=metadata.get("journal"),
        volume=metadata.get("volume"),
        issue=metadata.get("issue"),
        pages=metadata.get("pages"),
        publisher=metadata.get("publisher"),
        doi=metadata.get("doi"),
        url=metadata.get("url"),
        abstract=metadata.get("abstract"),
        keywords=metadata.get("keywords", []),
        source_file=metadata.get("source"),
        chunk_id=metadata.get("chunk_id"),
    )


def _extract_authors(metadata: Dict[str, Any]) -> List[str]:
    """Extract and parse author information from metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        List of author names
    """
    if "author" not in metadata:
        return []

    author_val = metadata["author"]

    # Already a list
    if isinstance(author_val, list):
        return author_val

    # Parse string
    if not isinstance(author_val, str):
        return []

    # Split on semicolon
    if ";" in author_val:
        return [a.strip() for a in author_val.split(";")]

    # Split on 'and'
    if " and " in author_val.lower():
        return [a.strip() for a in re.split(r"\s+and\s+", author_val, flags=re.I)]

    # Single author
    return [author_val]


def _determine_citation_type(source: str) -> CitationType:
    """Determine citation type from source.

    Args:
        source: Source filename or URL

    Returns:
        CitationType enum value
    """
    if source.endswith(".pdf"):
        return CitationType.ARTICLE

    if "http" in source:
        return CitationType.WEBPAGE

    return CitationType.MISC


def _extract_year(metadata: Dict[str, Any]) -> Optional[int]:
    """Extract year from metadata.

    Args:
        metadata: Metadata dictionary

    Returns:
        Year as integer, or None
    """
    # Try explicit year field
    if "year" in metadata:
        try:
            return int(metadata["year"])
        except (ValueError, TypeError) as e:
            logger.debug(f"Invalid year value in metadata: {e}")
            return None

    # Try to extract from date field
    if "date" not in metadata:
        return None

    date_str = str(metadata["date"])
    year_match = re.search(r"\b(19|20)\d{2}\b", date_str)
    if year_match:
        return int(year_match.group())

    return None
