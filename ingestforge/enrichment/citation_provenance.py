"""Citation Provenance Engine for the Research Vertical (RES-003).

Extracts citations from text, detects their style, and links them to
source chunks in the knowledge base for traceability.

Supported Citation Styles:
- APA: (Smith, 2020), (Smith & Jones, 2020), (Smith et al., 2020)
- MLA: (Smith 42), (Smith and Jones 42-45)
- Chicago: Smith (2020), Smith and Jones (2020, 42)
- Numbered: [1], [1-3], [1,3,5]"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ingestforge.storage.base import ChunkRepository

# Max bounds (Rule #2)
MAX_CITATIONS_PER_DOCUMENT = 1000
MAX_CHUNKS_TO_SEARCH = 500
MAX_TEXT_LENGTH = 500000

# Confidence thresholds
HIGH_CONFIDENCE = 0.9
MEDIUM_CONFIDENCE = 0.7
LOW_CONFIDENCE = 0.5


class CitationStyle(str, Enum):
    """Detected citation style."""

    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    NUMBERED = "numbered"
    UNKNOWN = "unknown"


@dataclass
class CitationProvenance:
    """Provenance data for a single citation.

    Attributes:
        raw_text: Original citation text as found in document.
        author: Primary author name (last name).
        year: Publication year if found.
        page_ref: Page reference if found.
        citation_style: Detected citation style.
        matched_chunks: IDs of matching source chunks.
        confidence: Match confidence (0.0-1.0).
    """

    raw_text: str
    author: str
    year: Optional[int]
    page_ref: Optional[str]
    citation_style: str
    matched_chunks: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "raw_text": self.raw_text,
            "author": self.author,
            "year": self.year,
            "page_ref": self.page_ref,
            "citation_style": self.citation_style,
            "matched_chunks": self.matched_chunks,
            "confidence": self.confidence,
        }


class _Logger:
    """Lazy logger holder."""

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


# =============================================================================
# Citation Pattern Definitions
# =============================================================================

# APA patterns: (Smith, 2020), (Smith & Jones, 2020), (Smith et al., 2020)
APA_SINGLE = re.compile(r"\(([A-Z][a-z]+),\s*((?:19|20)\d{2})\)", re.UNICODE)
APA_TWO_AUTHORS = re.compile(
    r"\(([A-Z][a-z]+)\s*[&]\s*([A-Z][a-z]+),\s*((?:19|20)\d{2})\)", re.UNICODE
)
APA_ET_AL = re.compile(
    r"\(([A-Z][a-z]+)\s+et\s+al\.?,\s*((?:19|20)\d{2})\)", re.UNICODE
)
APA_WITH_PAGE = re.compile(
    r"\(([A-Z][a-z]+)(?:\s+et\s+al\.?)?(?:\s*[&]\s*[A-Z][a-z]+)?,\s*((?:19|20)\d{2}),\s*p{1,2}\.\s*(\d+(?:-\d+)?)\)",
    re.UNICODE,
)

# MLA patterns: (Smith 42), (Smith and Jones 42-45)
MLA_SINGLE = re.compile(r"\(([A-Z][a-z]+)\s+(\d+(?:-\d+)?)\)", re.UNICODE)
MLA_TWO_AUTHORS = re.compile(
    r"\(([A-Z][a-z]+)\s+and\s+([A-Z][a-z]+)\s+(\d+(?:-\d+)?)\)", re.UNICODE
)

# Chicago author-date: Smith (2020), Smith and Jones (2020, 42), Smith et al. (2020)
CHICAGO_AUTHOR = re.compile(
    r"([A-Z][a-z]+(?:\s+(?:and\s+[A-Z][a-z]+|et\s+al\.?))?)\s*\(((?:19|20)\d{2})(?:,\s*(\d+(?:-\d+)?))?\)",
    re.UNICODE,
)

# Numbered citations: [1], [1-3], [1,3,5]
NUMBERED_SINGLE = re.compile(r"\[(\d+)\]")
NUMBERED_RANGE = re.compile(r"\[(\d+)-(\d+)\]")
NUMBERED_LIST = re.compile(r"\[(\d+(?:,\s*\d+)+)\]")

# =============================================================================
# Citation Provenance Engine
# =============================================================================


class CitationProvenanceEngine:
    """Engine for extracting citations and linking to source chunks.

    Supports multiple citation styles and provides chunk linking
    for research traceability.
    """

    def __init__(self, strict_matching: bool = False) -> None:
        """Initialize citation provenance engine.

        Args:
            strict_matching: If True, require exact author match for linking.
        """
        self.strict_matching = strict_matching

    def extract_citations(self, text: str) -> List[CitationProvenance]:
        """Extract all citations from text.

        Args:
            text: Input text to extract citations from.

        Returns:
            List of CitationProvenance objects.

        Raises:
            ValueError: If text exceeds maximum length.
        """
        # Validate input (Rule #7)
        if not text:
            return []
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Text too long: {len(text)} chars (max {MAX_TEXT_LENGTH})"
            )

        citations: List[CitationProvenance] = []

        # Extract by style in priority order
        citations.extend(self._extract_apa(text))
        citations.extend(self._extract_mla(text))
        citations.extend(self._extract_chicago(text))
        citations.extend(self._extract_numbered(text))

        # Deduplicate by raw_text
        citations = self._deduplicate_citations(citations)

        # Apply upper bound (Rule #2)
        if len(citations) > MAX_CITATIONS_PER_DOCUMENT:
            _Logger.get().warning(
                f"Citation count exceeds limit: {len(citations)} > {MAX_CITATIONS_PER_DOCUMENT}"
            )
            citations = citations[:MAX_CITATIONS_PER_DOCUMENT]

        return citations

    def detect_style(self, citation_text: str) -> str:
        """Detect the citation style of a given citation text.

        Args:
            citation_text: Citation text to analyze.

        Returns:
            Citation style as string (apa, mla, chicago, numbered, unknown).
        """
        if not citation_text:
            return CitationStyle.UNKNOWN.value

        # Check patterns in order of specificity
        if self._is_apa_style(citation_text):
            return CitationStyle.APA.value
        if self._is_mla_style(citation_text):
            return CitationStyle.MLA.value
        if self._is_chicago_style(citation_text):
            return CitationStyle.CHICAGO.value
        if self._is_numbered_style(citation_text):
            return CitationStyle.NUMBERED.value

        return CitationStyle.UNKNOWN.value

    def link_to_chunks(
        self,
        citation: CitationProvenance,
        storage: ChunkRepository,
        library_filter: Optional[str] = None,
    ) -> List[str]:
        """Link a citation to matching chunks in storage.

        Args:
            citation: Citation to link.
            storage: Storage backend to search.
            library_filter: Optional library filter.

        Returns:
            List of matching chunk IDs.
        """
        if not citation.author:
            return []

        # Build search query from citation data
        query = self._build_search_query(citation)

        # Search storage
        results = storage.search(
            query=query,
            top_k=min(20, MAX_CHUNKS_TO_SEARCH),
            library_filter=library_filter,
        )

        # Filter results by author/year matching
        matched_ids = self._filter_matching_chunks(citation, results)

        # Update citation
        citation.matched_chunks = matched_ids
        citation.confidence = self._calculate_link_confidence(
            citation, len(matched_ids)
        )

        return matched_ids

    def build_citation_graph(
        self,
        document_id: str,
        text: str,
        storage: Optional[ChunkRepository] = None,
    ) -> Dict[str, List[str]]:
        """Build a citation graph for a document.

        Maps each citation to its source chunks for visualization
        and traceability.

        Args:
            document_id: ID of the document being analyzed.
            text: Document text to extract citations from.
            storage: Optional storage for chunk linking.

        Returns:
            Dict mapping citation raw_text to list of chunk IDs.
        """
        citations = self.extract_citations(text)
        graph: Dict[str, List[str]] = {}

        # Fixed upper bound (Rule #2)
        num_citations = min(len(citations), MAX_CITATIONS_PER_DOCUMENT)

        for i in range(num_citations):
            citation = citations[i]

            if storage:
                chunk_ids = self.link_to_chunks(citation, storage)
            else:
                chunk_ids = []

            graph[citation.raw_text] = chunk_ids

        _Logger.get().info(
            f"Built citation graph for {document_id}: " f"{len(graph)} citations found"
        )

        return graph

    # =========================================================================
    # Private Extraction Methods
    # =========================================================================

    def _extract_apa(self, text: str) -> List[CitationProvenance]:
        """Extract APA-style citations."""
        citations: List[CitationProvenance] = []

        # APA with page reference (most specific)
        for match in APA_WITH_PAGE.finditer(text):
            citations.append(
                CitationProvenance(
                    raw_text=match.group(0),
                    author=match.group(1),
                    year=int(match.group(2)),
                    page_ref=match.group(3),
                    citation_style=CitationStyle.APA.value,
                    confidence=HIGH_CONFIDENCE,
                )
            )

        # APA et al.
        for match in APA_ET_AL.finditer(text):
            if not self._is_already_extracted(match.group(0), citations):
                citations.append(
                    CitationProvenance(
                        raw_text=match.group(0),
                        author=match.group(1),
                        year=int(match.group(2)),
                        page_ref=None,
                        citation_style=CitationStyle.APA.value,
                        confidence=HIGH_CONFIDENCE,
                    )
                )

        # APA two authors
        for match in APA_TWO_AUTHORS.finditer(text):
            if not self._is_already_extracted(match.group(0), citations):
                citations.append(
                    CitationProvenance(
                        raw_text=match.group(0),
                        author=match.group(1),
                        year=int(match.group(3)),
                        page_ref=None,
                        citation_style=CitationStyle.APA.value,
                        confidence=HIGH_CONFIDENCE,
                    )
                )

        # APA single author
        for match in APA_SINGLE.finditer(text):
            if not self._is_already_extracted(match.group(0), citations):
                citations.append(
                    CitationProvenance(
                        raw_text=match.group(0),
                        author=match.group(1),
                        year=int(match.group(2)),
                        page_ref=None,
                        citation_style=CitationStyle.APA.value,
                        confidence=HIGH_CONFIDENCE,
                    )
                )

        return citations

    def _extract_mla(self, text: str) -> List[CitationProvenance]:
        """Extract MLA-style citations."""
        citations: List[CitationProvenance] = []

        # MLA two authors
        for match in MLA_TWO_AUTHORS.finditer(text):
            citations.append(
                CitationProvenance(
                    raw_text=match.group(0),
                    author=match.group(1),
                    year=None,
                    page_ref=match.group(3),
                    citation_style=CitationStyle.MLA.value,
                    confidence=HIGH_CONFIDENCE,
                )
            )

        # MLA single author
        for match in MLA_SINGLE.finditer(text):
            if not self._is_already_extracted(match.group(0), citations):
                citations.append(
                    CitationProvenance(
                        raw_text=match.group(0),
                        author=match.group(1),
                        year=None,
                        page_ref=match.group(2),
                        citation_style=CitationStyle.MLA.value,
                        confidence=MEDIUM_CONFIDENCE,
                    )
                )

        return citations

    def _extract_chicago(self, text: str) -> List[CitationProvenance]:
        """Extract Chicago-style citations."""
        citations: List[CitationProvenance] = []

        for match in CHICAGO_AUTHOR.finditer(text):
            # Parse author (may contain "and")
            author_str = match.group(1)
            primary_author = author_str.split(" and ")[0].strip()

            citations.append(
                CitationProvenance(
                    raw_text=match.group(0),
                    author=primary_author,
                    year=int(match.group(2)),
                    page_ref=match.group(3) if match.group(3) else None,
                    citation_style=CitationStyle.CHICAGO.value,
                    confidence=HIGH_CONFIDENCE,
                )
            )

        return citations

    def _extract_numbered(self, text: str) -> List[CitationProvenance]:
        """Extract numbered citations."""
        citations: List[CitationProvenance] = []

        # Numbered range [1-3]
        for match in NUMBERED_RANGE.finditer(text):
            citations.append(
                CitationProvenance(
                    raw_text=match.group(0),
                    author=f"ref_{match.group(1)}-{match.group(2)}",
                    year=None,
                    page_ref=None,
                    citation_style=CitationStyle.NUMBERED.value,
                    confidence=HIGH_CONFIDENCE,
                )
            )

        # Numbered list [1,3,5]
        for match in NUMBERED_LIST.finditer(text):
            if not self._is_already_extracted(match.group(0), citations):
                citations.append(
                    CitationProvenance(
                        raw_text=match.group(0),
                        author=f"refs_{match.group(1).replace(' ', '')}",
                        year=None,
                        page_ref=None,
                        citation_style=CitationStyle.NUMBERED.value,
                        confidence=HIGH_CONFIDENCE,
                    )
                )

        # Numbered single [1]
        for match in NUMBERED_SINGLE.finditer(text):
            if not self._is_already_extracted(match.group(0), citations):
                citations.append(
                    CitationProvenance(
                        raw_text=match.group(0),
                        author=f"ref_{match.group(1)}",
                        year=None,
                        page_ref=None,
                        citation_style=CitationStyle.NUMBERED.value,
                        confidence=HIGH_CONFIDENCE,
                    )
                )

        return citations

    # =========================================================================
    # Private Helper Methods
    # =========================================================================

    def _is_apa_style(self, text: str) -> bool:
        """Check if text matches APA citation style."""
        return bool(
            APA_SINGLE.search(text)
            or APA_TWO_AUTHORS.search(text)
            or APA_ET_AL.search(text)
            or APA_WITH_PAGE.search(text)
        )

    def _is_mla_style(self, text: str) -> bool:
        """Check if text matches MLA citation style."""
        return bool(MLA_SINGLE.search(text) or MLA_TWO_AUTHORS.search(text))

    def _is_chicago_style(self, text: str) -> bool:
        """Check if text matches Chicago citation style."""
        return bool(CHICAGO_AUTHOR.search(text))

    def _is_numbered_style(self, text: str) -> bool:
        """Check if text matches numbered citation style."""
        return bool(
            NUMBERED_SINGLE.search(text)
            or NUMBERED_RANGE.search(text)
            or NUMBERED_LIST.search(text)
        )

    def _is_already_extracted(
        self,
        raw_text: str,
        citations: List[CitationProvenance],
    ) -> bool:
        """Check if citation was already extracted."""
        for citation in citations:
            if citation.raw_text == raw_text:
                return True
        return False

    def _deduplicate_citations(
        self,
        citations: List[CitationProvenance],
    ) -> List[CitationProvenance]:
        """Remove duplicate citations by raw_text."""
        seen: set = set()
        unique: List[CitationProvenance] = []

        for citation in citations:
            if citation.raw_text not in seen:
                seen.add(citation.raw_text)
                unique.append(citation)

        return unique

    def _build_search_query(self, citation: CitationProvenance) -> str:
        """Build search query from citation data."""
        parts = [citation.author]

        if citation.year:
            parts.append(str(citation.year))

        return " ".join(parts)

    def _filter_matching_chunks(
        self,
        citation: CitationProvenance,
        results: List[Any],
    ) -> List[str]:
        """Filter search results to truly matching chunks."""
        matched: List[str] = []
        author_lower = citation.author.lower()

        # Fixed upper bound (Rule #2)
        num_results = min(len(results), MAX_CHUNKS_TO_SEARCH)

        for i in range(num_results):
            result = results[i]
            content_lower = result.content.lower()

            # Check author match
            if author_lower not in content_lower:
                continue

            # Check year match if available
            if citation.year:
                if str(citation.year) not in result.content:
                    continue

            matched.append(result.chunk_id)

        return matched

    def _calculate_link_confidence(
        self,
        citation: CitationProvenance,
        match_count: int,
    ) -> float:
        """Calculate confidence score for chunk linking."""
        if match_count == 0:
            return 0.0

        base_confidence = citation.confidence

        # More matches = higher confidence (up to a point)
        if match_count >= 3:
            return min(1.0, base_confidence * 1.1)
        if match_count >= 1:
            return base_confidence

        return LOW_CONFIDENCE


# =============================================================================
# Convenience Functions
# =============================================================================


def extract_academic_citations(text: str) -> List[CitationProvenance]:
    """Extract all academic citations from text.

    Convenience function for citation extraction.

    Args:
        text: Input text.

    Returns:
        List of CitationProvenance objects.
    """
    engine = CitationProvenanceEngine()
    return engine.extract_citations(text)


def detect_citation_style(citation_text: str) -> str:
    """Detect the citation style of a single citation.

    Args:
        citation_text: Citation text.

    Returns:
        Citation style string.
    """
    engine = CitationProvenanceEngine()
    return engine.detect_style(citation_text)


def link_citations_to_chunks(
    text: str,
    storage: ChunkRepository,
    library_filter: Optional[str] = None,
) -> List[CitationProvenance]:
    """Extract citations and link to chunks in storage.

    Args:
        text: Input text.
        storage: Storage backend.
        library_filter: Optional library filter.

    Returns:
        List of CitationProvenance with linked chunks.
    """
    engine = CitationProvenanceEngine()
    citations = engine.extract_citations(text)

    for citation in citations:
        engine.link_to_chunks(citation, storage, library_filter)

    return citations
