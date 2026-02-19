"""Bluebook Citation Parser for legal documents.

Identifies and parses legal citation patterns in text following
the Bluebook citation format used in US legal documents.

Supported Citation Formats:
- Federal reporters: 347 U.S. 483, 123 F.3d 456, 456 F. Supp. 2d 789
- State reporters: 123 Cal. 3d 456, 789 N.Y.2d 123
- Supreme Court: Brown v. Board of Education, 347 U.S. 483 (1954)
- Circuit courts: Smith v. Jones, 123 F.3d 456 (9th Cir. 1999)
- Pin cites: 347 U.S. at 490"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Pattern, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class LegalCitation:
    """Structured legal citation with parsed components.

    Attributes:
        raw_text: Original citation text as found in document
        volume: Reporter volume number
        reporter: Reporter abbreviation (e.g., U.S., F.3d, Cal.)
        page: Starting page number
        pin_cite: Specific page reference (e.g., "at 490")
        court: Court identifier (e.g., 9th Cir., S.D.N.Y.)
        year: Decision year
        case_name: Party names if present (e.g., "Brown v. Board of Education")
        start: Character offset where citation starts
        end: Character offset where citation ends
    """

    raw_text: str
    volume: int
    reporter: str
    page: int
    pin_cite: Optional[int] = None
    court: Optional[str] = None
    year: Optional[int] = None
    case_name: Optional[str] = None
    start: int = 0
    end: int = 0

    def to_dict(self) -> Dict:
        """Convert citation to dictionary format.

        Returns:
            Dictionary with all citation fields
        """
        return {
            "raw_text": self.raw_text,
            "volume": self.volume,
            "reporter": self.reporter,
            "page": self.page,
            "pin_cite": self.pin_cite,
            "court": self.court,
            "year": self.year,
            "case_name": self.case_name,
            "start": self.start,
            "end": self.end,
        }

    def __hash__(self) -> int:
        """Enable hashing for deduplication."""
        return hash((self.volume, self.reporter, self.page))

    def __eq__(self, other: object) -> bool:
        """Enable equality comparison."""
        if not isinstance(other, LegalCitation):
            return False
        return (
            self.volume == other.volume
            and self.reporter == other.reporter
            and self.page == other.page
        )


# =============================================================================
# Reporter Definitions
# =============================================================================

# Federal Reporter abbreviations with normalized forms
FEDERAL_REPORTERS: Dict[str, str] = {
    # Supreme Court
    "U.S.": "U.S.",
    "U. S.": "U.S.",
    "S. Ct.": "S. Ct.",
    "S.Ct.": "S. Ct.",
    "L. Ed.": "L. Ed.",
    "L.Ed.": "L. Ed.",
    "L. Ed. 2d": "L. Ed. 2d",
    "L.Ed.2d": "L. Ed. 2d",
    # Circuit Courts
    "F.": "F.",
    "F.2d": "F.2d",
    "F. 2d": "F.2d",
    "F.3d": "F.3d",
    "F. 3d": "F.3d",
    "F.4th": "F.4th",
    "F. 4th": "F.4th",
    # District Courts
    "F. Supp.": "F. Supp.",
    "F.Supp.": "F. Supp.",
    "F. Supp. 2d": "F. Supp. 2d",
    "F.Supp.2d": "F. Supp. 2d",
    "F. Supp. 3d": "F. Supp. 3d",
    "F.Supp.3d": "F. Supp. 3d",
    # Federal Rules Decisions
    "F.R.D.": "F.R.D.",
    "F. R. D.": "F.R.D.",
    # Bankruptcy
    "B.R.": "B.R.",
    "B. R.": "B.R.",
}

# State Reporter abbreviations with normalized forms
STATE_REPORTERS: Dict[str, str] = {
    # California
    "Cal.": "Cal.",
    "Cal. 2d": "Cal. 2d",
    "Cal.2d": "Cal. 2d",
    "Cal. 3d": "Cal. 3d",
    "Cal.3d": "Cal. 3d",
    "Cal. 4th": "Cal. 4th",
    "Cal.4th": "Cal. 4th",
    "Cal. 5th": "Cal. 5th",
    "Cal.5th": "Cal. 5th",
    "Cal. App.": "Cal. App.",
    "Cal.App.": "Cal. App.",
    "Cal. App. 2d": "Cal. App. 2d",
    "Cal.App.2d": "Cal. App. 2d",
    "Cal. App. 3d": "Cal. App. 3d",
    "Cal.App.3d": "Cal. App. 3d",
    "Cal. App. 4th": "Cal. App. 4th",
    "Cal.App.4th": "Cal. App. 4th",
    "Cal. App. 5th": "Cal. App. 5th",
    "Cal.App.5th": "Cal. App. 5th",
    "Cal. Rptr.": "Cal. Rptr.",
    "Cal.Rptr.": "Cal. Rptr.",
    "Cal. Rptr. 2d": "Cal. Rptr. 2d",
    "Cal.Rptr.2d": "Cal. Rptr. 2d",
    "Cal. Rptr. 3d": "Cal. Rptr. 3d",
    "Cal.Rptr.3d": "Cal. Rptr. 3d",
    # New York
    "N.Y.": "N.Y.",
    "N. Y.": "N.Y.",
    "N.Y.2d": "N.Y.2d",
    "N. Y. 2d": "N.Y.2d",
    "N.Y.3d": "N.Y.3d",
    "N. Y. 3d": "N.Y.3d",
    "A.D.": "A.D.",
    "A. D.": "A.D.",
    "A.D.2d": "A.D.2d",
    "A. D. 2d": "A.D.2d",
    "A.D.3d": "A.D.3d",
    "A. D. 3d": "A.D.3d",
    "N.Y.S.": "N.Y.S.",
    "N. Y. S.": "N.Y.S.",
    "N.Y.S.2d": "N.Y.S.2d",
    "N. Y. S. 2d": "N.Y.S.2d",
    "N.Y.S.3d": "N.Y.S.3d",
    "N. Y. S. 3d": "N.Y.S.3d",
    # Texas
    "Tex.": "Tex.",
    "S.W.": "S.W.",
    "S. W.": "S.W.",
    "S.W.2d": "S.W.2d",
    "S. W. 2d": "S.W.2d",
    "S.W.3d": "S.W.3d",
    "S. W. 3d": "S.W.3d",
    # Regional reporters
    "A.": "A.",
    "A.2d": "A.2d",
    "A. 2d": "A.2d",
    "A.3d": "A.3d",
    "A. 3d": "A.3d",
    "N.E.": "N.E.",
    "N. E.": "N.E.",
    "N.E.2d": "N.E.2d",
    "N. E. 2d": "N.E.2d",
    "N.E.3d": "N.E.3d",
    "N. E. 3d": "N.E.3d",
    "N.W.": "N.W.",
    "N. W.": "N.W.",
    "N.W.2d": "N.W.2d",
    "N. W. 2d": "N.W.2d",
    "P.": "P.",
    "P.2d": "P.2d",
    "P. 2d": "P.2d",
    "P.3d": "P.3d",
    "P. 3d": "P.3d",
    "So.": "So.",
    "So. 2d": "So. 2d",
    "So.2d": "So. 2d",
    "So. 3d": "So. 3d",
    "So.3d": "So. 3d",
    "S.E.": "S.E.",
    "S. E.": "S.E.",
    "S.E.2d": "S.E.2d",
    "S. E. 2d": "S.E.2d",
}

# Court abbreviations
COURT_ABBREVIATIONS: Dict[str, str] = {
    # Federal Circuit Courts
    "1st Cir.": "1st Cir.",
    "2d Cir.": "2d Cir.",
    "2nd Cir.": "2d Cir.",
    "3d Cir.": "3d Cir.",
    "3rd Cir.": "3d Cir.",
    "4th Cir.": "4th Cir.",
    "5th Cir.": "5th Cir.",
    "6th Cir.": "6th Cir.",
    "7th Cir.": "7th Cir.",
    "8th Cir.": "8th Cir.",
    "9th Cir.": "9th Cir.",
    "10th Cir.": "10th Cir.",
    "11th Cir.": "11th Cir.",
    "D.C. Cir.": "D.C. Cir.",
    "Fed. Cir.": "Fed. Cir.",
    # District Courts (common examples)
    "S.D.N.Y.": "S.D.N.Y.",
    "E.D.N.Y.": "E.D.N.Y.",
    "N.D. Cal.": "N.D. Cal.",
    "C.D. Cal.": "C.D. Cal.",
    "S.D. Cal.": "S.D. Cal.",
    "N.D. Ill.": "N.D. Ill.",
    "D. Mass.": "D. Mass.",
    "E.D. Pa.": "E.D. Pa.",
    "W.D. Pa.": "W.D. Pa.",
    "D. Del.": "D. Del.",
    "E.D. Tex.": "E.D. Tex.",
    "W.D. Tex.": "W.D. Tex.",
    "S.D. Tex.": "S.D. Tex.",
    "N.D. Tex.": "N.D. Tex.",
}

# =============================================================================
# BluebookParser Class
# =============================================================================


class BluebookParser:
    """Parser for Bluebook-format legal citations.

    Identifies and parses legal citations in text, supporting
    federal courts, state courts, and various reporter formats.

    Example:
        >>> parser = BluebookParser()
        >>> citations = parser.extract_citations(
        ...     "The Court held in Brown v. Board of Education, "
        ...     "347 U.S. 483 (1954), that separate but equal is unconstitutional."
        ... )
        >>> for c in citations:
        ...     print(f"{c.case_name}: {c.volume} {c.reporter} {c.page}")
    """

    def __init__(self) -> None:
        """Initialize Bluebook parser with compiled patterns.

        Rule #4: Simple initialization
        """
        self._patterns = self._compile_patterns()
        self._all_reporters = {**FEDERAL_REPORTERS, **STATE_REPORTERS}

    def _compile_patterns(self) -> Dict[str, Pattern]:
        """Compile regex patterns for citation extraction.

        Rule #4: Function <60 lines
        Rule #6: Variables at smallest scope

        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {}

        # Build reporter alternatives (sorted by length to prefer longer matches)
        all_reporters = {**FEDERAL_REPORTERS, **STATE_REPORTERS}
        reporter_alts = "|".join(
            re.escape(r) for r in sorted(all_reporters.keys(), key=len, reverse=True)
        )

        # Full citation: Case Name, Volume Reporter Page (Court Year)
        patterns["full_citation"] = re.compile(
            r"([A-Z][a-zA-Z\'\-\.]+(?:\s+[A-Za-z\'\-\.]+)*)"  # Case name
            r"\s+v\.\s+"  # v.
            r"([A-Z][a-zA-Z\'\-\.]+(?:\s+[A-Za-z\'\-\.]+)*)"  # Defendant
            r",?\s*"
            r"(\d+)\s+"  # Volume
            rf"({reporter_alts})\s+"  # Reporter
            r"(\d+)"  # Page
            r"(?:\s*,\s*(\d+))?"  # Optional pin cite
            r"(?:\s*\(([^)]+)\))?"  # Optional parenthetical (court year)
        )

        # Short citation: Volume Reporter Page
        patterns["short_citation"] = re.compile(
            r"(\d+)\s+"  # Volume
            rf"({reporter_alts})\s+"  # Reporter
            r"(\d+)"  # Page
            r"(?:\s*,\s*(\d+))?"  # Optional pin cite
            r"(?:\s*\(([^)]+)\))?"  # Optional parenthetical
        )

        # Pin cite: Volume Reporter at Page
        patterns["pin_cite"] = re.compile(
            r"(\d+)\s+"  # Volume
            rf"({reporter_alts})\s+"  # Reporter
            r"at\s+"  # "at"
            r"(\d+)"  # Pin cite page
        )

        # Id. citation
        patterns["id_citation"] = re.compile(
            r"\bId\.\s+" r"at\s+" r"(\d+)"  # Pin cite
        )

        return patterns

    def extract_citations(self, text: str) -> List[LegalCitation]:
        """Extract all legal citations from text.

        Args:
            text: Text to search for citations

        Returns:
            List of LegalCitation objects sorted by position

        Rule #1: Early return for empty text
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        if not text or not text.strip():
            return []

        citations: List[LegalCitation] = []
        seen_positions: set = set()

        # Extract full citations first (most specific)
        full_citations = self._extract_full_citations(text)
        for cite in full_citations:
            citations.append(cite)
            seen_positions.add((cite.start, cite.end))

        # Extract short citations (avoid duplicates)
        short_citations = self._extract_short_citations(text)
        for cite in short_citations:
            if not self._overlaps_existing(cite, seen_positions):
                citations.append(cite)
                seen_positions.add((cite.start, cite.end))

        # Extract pin cites (avoid duplicates)
        pin_citations = self._extract_pin_cites(text)
        for cite in pin_citations:
            if not self._overlaps_existing(cite, seen_positions):
                citations.append(cite)
                seen_positions.add((cite.start, cite.end))

        return sorted(citations, key=lambda c: c.start)

    def _overlaps_existing(self, citation: LegalCitation, positions: set) -> bool:
        """Check if citation overlaps with existing citations.

        Args:
            citation: Citation to check
            positions: Set of (start, end) tuples

        Returns:
            True if overlaps with existing citation
        """
        for start, end in positions:
            if not (citation.end <= start or citation.start >= end):
                return True
        return False

    def _extract_full_citations(self, text: str) -> List[LegalCitation]:
        """Extract full citations with case names.

        Args:
            text: Text to search

        Returns:
            List of LegalCitation objects

        Rule #4: Function <60 lines
        """
        citations: List[LegalCitation] = []
        pattern = self._patterns["full_citation"]

        for match in pattern.finditer(text):
            try:
                plaintiff = match.group(1).strip()
                defendant = match.group(2).strip()
                volume = int(match.group(3))
                reporter = match.group(4)
                page = int(match.group(5))
                pin_cite_str = match.group(6)
                paren = match.group(7)

                citation = self._create_citation_from_match(
                    match=match,
                    volume=volume,
                    reporter=reporter,
                    page=page,
                    pin_cite_str=pin_cite_str,
                    paren=paren,
                    case_name=f"{plaintiff} v. {defendant}",
                )
                citations.append(citation)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse full citation: {match.group(0)}: {e}")

        return citations

    def _extract_short_citations(self, text: str) -> List[LegalCitation]:
        """Extract short citations (volume reporter page).

        Args:
            text: Text to search

        Returns:
            List of LegalCitation objects

        Rule #4: Function <60 lines
        """
        citations: List[LegalCitation] = []
        pattern = self._patterns["short_citation"]

        for match in pattern.finditer(text):
            try:
                volume = int(match.group(1))
                reporter = match.group(2)
                page = int(match.group(3))
                pin_cite_str = match.group(4)
                paren = match.group(5)

                citation = self._create_citation_from_match(
                    match=match,
                    volume=volume,
                    reporter=reporter,
                    page=page,
                    pin_cite_str=pin_cite_str,
                    paren=paren,
                    case_name=None,
                )
                citations.append(citation)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse short citation: {match.group(0)}: {e}")

        return citations

    def _extract_pin_cites(self, text: str) -> List[LegalCitation]:
        """Extract pin cites (volume reporter at page).

        Args:
            text: Text to search

        Returns:
            List of LegalCitation objects

        Rule #4: Function <60 lines
        """
        citations: List[LegalCitation] = []
        pattern = self._patterns["pin_cite"]

        for match in pattern.finditer(text):
            try:
                volume = int(match.group(1))
                reporter = match.group(2)
                pin_page = int(match.group(3))

                citation = LegalCitation(
                    raw_text=match.group(0),
                    volume=volume,
                    reporter=self.normalize_reporter(reporter),
                    page=0,  # Unknown starting page for pin cite only
                    pin_cite=pin_page,
                    start=match.start(),
                    end=match.end(),
                )
                citations.append(citation)
            except (ValueError, IndexError) as e:
                logger.debug(f"Failed to parse pin cite: {match.group(0)}: {e}")

        return citations

    def _create_citation_from_match(
        self,
        match: re.Match,
        volume: int,
        reporter: str,
        page: int,
        pin_cite_str: Optional[str],
        paren: Optional[str],
        case_name: Optional[str],
    ) -> LegalCitation:
        """Create LegalCitation from regex match components.

        Args:
            match: Regex match object
            volume: Volume number
            reporter: Reporter abbreviation
            page: Starting page
            pin_cite_str: Optional pin cite string
            paren: Optional parenthetical text
            case_name: Optional case name

        Returns:
            LegalCitation object

        Rule #4: Function <60 lines
        """
        pin_cite = int(pin_cite_str) if pin_cite_str else None
        court, year = self._parse_parenthetical(paren)

        return LegalCitation(
            raw_text=match.group(0),
            volume=volume,
            reporter=self.normalize_reporter(reporter),
            page=page,
            pin_cite=pin_cite,
            court=court,
            year=year,
            case_name=case_name,
            start=match.start(),
            end=match.end(),
        )

    def _parse_parenthetical(
        self, paren: Optional[str]
    ) -> Tuple[Optional[str], Optional[int]]:
        """Parse parenthetical for court and year.

        Args:
            paren: Parenthetical text (e.g., "9th Cir. 1999")

        Returns:
            Tuple of (court, year)

        Rule #4: Function <60 lines
        """
        if not paren:
            return None, None

        court = None
        year = None

        # Extract year (4-digit number)
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", paren)
        if year_match:
            year = int(year_match.group(1))

        # Check for court abbreviations
        for abbrev, normalized in COURT_ABBREVIATIONS.items():
            if abbrev in paren:
                court = normalized
                break

        return court, year

    def parse_citation(self, citation_text: str) -> Optional[LegalCitation]:
        """Parse a single citation string.

        Args:
            citation_text: Citation text to parse

        Returns:
            LegalCitation object or None if parsing fails

        Rule #1: Early return for empty input
        Rule #4: Function <60 lines
        """
        if not citation_text or not citation_text.strip():
            return None

        citations = self.extract_citations(citation_text)
        return citations[0] if citations else None

    def normalize_reporter(self, abbrev: str) -> str:
        """Normalize reporter abbreviation to standard form.

        Args:
            abbrev: Reporter abbreviation (may have spacing variations)

        Returns:
            Normalized reporter abbreviation

        Rule #4: Function <60 lines
        """
        # Check federal reporters
        if abbrev in FEDERAL_REPORTERS:
            return FEDERAL_REPORTERS[abbrev]

        # Check state reporters
        if abbrev in STATE_REPORTERS:
            return STATE_REPORTERS[abbrev]

        # Return as-is if not found
        return abbrev

    def to_bluebook_format(self, citation: LegalCitation) -> str:
        """Format citation in proper Bluebook style.

        Args:
            citation: LegalCitation object to format

        Returns:
            Properly formatted Bluebook citation string

        Rule #4: Function <60 lines
        """
        parts = []

        # Case name (italicized placeholder)
        if citation.case_name:
            parts.append(citation.case_name)
            parts.append(",")

        # Volume, reporter, page
        parts.append(f" {citation.volume} {citation.reporter} {citation.page}")

        # Pin cite
        if citation.pin_cite:
            parts.append(f", {citation.pin_cite}")

        # Parenthetical
        paren_parts = []
        if citation.court:
            paren_parts.append(citation.court)
        if citation.year:
            paren_parts.append(str(citation.year))

        if paren_parts:
            parts.append(f" ({' '.join(paren_parts)})")

        return "".join(parts).strip()

    def enrich(self, chunk: Dict) -> Dict:
        """Enrich chunk with legal citation metadata.

        Compatible with IngestForge enrichment pipeline.

        Args:
            chunk: Chunk dictionary with 'text' or 'content' field

        Returns:
            Enriched chunk with 'legal_citations' field

        Rule #4: Function <60 lines
        """
        text = chunk.get("text") or chunk.get("content", "")

        if not text:
            chunk["legal_citations"] = []
            chunk["citation_count"] = 0
            return chunk

        citations = self.extract_citations(text)

        chunk["legal_citations"] = [c.to_dict() for c in citations]
        chunk["citation_count"] = len(citations)

        # Add reporter statistics
        reporter_counts: Dict[str, int] = {}
        for cite in citations:
            reporter_counts[cite.reporter] = reporter_counts.get(cite.reporter, 0) + 1
        chunk["reporter_counts"] = reporter_counts

        return chunk


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_citations(text: str) -> List[LegalCitation]:
    """Extract legal citations from text.

    Convenience function for direct citation extraction.

    Args:
        text: Text to search for citations

    Returns:
        List of LegalCitation objects
    """
    parser = BluebookParser()
    return parser.extract_citations(text)


def parse_citation(citation_text: str) -> Optional[LegalCitation]:
    """Parse a single citation string.

    Convenience function for parsing individual citations.

    Args:
        citation_text: Citation text to parse

    Returns:
        LegalCitation object or None if parsing fails
    """
    parser = BluebookParser()
    return parser.parse_citation(citation_text)


def enrich_with_citations(chunk: Dict) -> Dict:
    """Enrich chunk with legal citation metadata.

    Convenience function for pipeline integration.

    Args:
        chunk: Chunk dictionary

    Returns:
        Enriched chunk with legal citations
    """
    parser = BluebookParser()
    return parser.enrich(chunk)
