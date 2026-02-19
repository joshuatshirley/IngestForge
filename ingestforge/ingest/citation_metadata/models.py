#!/usr/bin/env python3
"""Data models for citation metadata extraction.

Defines enums and dataclasses for source types, authors, and citation metadata.
"""

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class SourceType(Enum):
    """Type of source document."""

    WEBPAGE = "webpage"
    JOURNAL_ARTICLE = "journal_article"
    BOOK = "book"
    BOOK_CHAPTER = "book_chapter"
    CONFERENCE_PAPER = "conference_paper"
    THESIS = "thesis"
    REPORT = "report"
    PREPRINT = "preprint"
    NEWS_ARTICLE = "news_article"
    BLOG_POST = "blog_post"
    DATASET = "dataset"
    SOFTWARE = "software"
    VIDEO = "video"
    COURT_OPINION = "court_opinion"
    UNKNOWN = "unknown"


@dataclass
class Author:
    """Author information."""

    name: str
    given_name: str = ""
    family_name: str = ""
    orcid: str = ""
    affiliation: str = ""
    email: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_string(cls, name: str) -> "Author":
        """Parse author name into components."""
        name = name.strip()
        parts = name.split()

        if not parts:
            return cls(name="")
        elif len(parts) == 1:
            return cls(name=name, family_name=name)
        else:
            # Assume "Given Family" or "Family, Given"
            if "," in name:
                family, given = name.split(",", 1)
                return cls(
                    name=f"{given.strip()} {family.strip()}",
                    given_name=given.strip(),
                    family_name=family.strip(),
                )
            else:
                return cls(
                    name=name,
                    given_name=" ".join(parts[:-1]),
                    family_name=parts[-1],
                )


@dataclass
class CitationMetadata:
    """Complete citation metadata for a source."""

    title: str = ""
    authors: list[Any] = field(default_factory=list)  # List of Author dicts
    date_published: str = ""
    date_accessed: str = ""
    year: Optional[int] = None
    source_type: SourceType = SourceType.UNKNOWN

    # Publication info
    journal: str = ""
    volume: str = ""
    issue: str = ""
    pages: str = ""
    publisher: str = ""
    edition: str = ""

    # Identifiers
    doi: str = ""
    isbn: str = ""
    issn: str = ""
    arxiv_id: str = ""
    pmid: str = ""
    pmcid: str = ""
    url: str = ""

    # Additional
    abstract: str = ""
    keywords: list[Any] = field(default_factory=list)
    language: str = ""
    license: str = ""

    # Source tracking
    extraction_source: str = ""  # Where metadata came from
    confidence: float = 0.0
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d["source_type"] = self.source_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CitationMetadata":
        """Create from dictionary."""
        if "source_type" in data:
            if isinstance(data["source_type"], str):
                data["source_type"] = SourceType(data["source_type"])
        return cls(**data)

    @property
    def has_required_fields(self) -> bool:
        """Check if minimum required fields are present."""
        return bool(self.title) and (bool(self.authors) or bool(self.url))

    @property
    def citation_key(self) -> str:
        """Generate a citation key."""
        if self.authors and self.year:
            first_author = self.authors[0]
            if isinstance(first_author, dict):
                family = first_author.get(
                    "family_name", first_author.get("name", "unknown")
                )
            else:
                family = first_author.family_name or first_author.name
            # Clean family name
            family = re.sub(r"[^a-zA-Z]", "", family)
            return f"{family.lower()}{self.year}"
        elif self.title and self.year:
            # Use first word of title
            word = (
                re.sub(r"[^a-zA-Z]", "", self.title.split()[0])
                if self.title.split()
                else "unknown"
            )
            return f"{word.lower()}{self.year}"
        return "unknown"


@dataclass
class LegalMetadata(CitationMetadata):
    """Legal-specific citation metadata."""

    docket_number: str = ""
    jurisdiction: str = ""
    judge: str = ""
    court: str = ""
    reporter: str = ""
    citation: str = ""
    legal_role: str = ""  # Fact, Opinion, Dissent, etc.

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "docket_number": self.docket_number,
                "jurisdiction": self.jurisdiction,
                "judge": self.judge,
                "court": self.court,
                "reporter": self.reporter,
                "citation": self.citation,
                "legal_role": self.legal_role,
            }
        )
        return d
