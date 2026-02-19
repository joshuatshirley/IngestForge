"""Internal reference extractor for citation graph (CITE-001.1).

Extracts and maps internal citations between corpus documents,
building a link map from source chunks to referenced chunks.

NASA JPL Commandments compliance:
- Rule #1: Simple control flow, no deep nesting
- Rule #2: Fixed upper bounds on reference counts
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.core.citation.link_extractor import (
        LinkExtractor,
        InternalReference,
    )

    extractor = LinkExtractor(storage)
    references = extractor.extract_from_chunk(chunk)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Set, Tuple, TYPE_CHECKING

from ingestforge.core.logging import get_logger
from ingestforge.core.citation.constants import LEGAL_REPORTER_PATTERN

if TYPE_CHECKING:
    from ingestforge.storage.base import BaseStorage

logger = get_logger(__name__)
MAX_REFERENCES_PER_CHUNK = 100
MAX_SEARCH_RESULTS = 50


class ReferenceType(Enum):
    """Types of internal references."""

    EXPLICIT_CITATION = "explicit"  # [Author, Year] or (Author Year)
    SECTION_REF = "section"  # "see Section 3.2"
    FIGURE_REF = "figure"  # "Figure 5", "Fig. 2"
    TABLE_REF = "table"  # "Table 1"
    EQUATION_REF = "equation"  # "Equation (3)"
    CHAPTER_REF = "chapter"  # "Chapter 4"
    PAGE_REF = "page"  # "page 42"
    FOOTNOTE_REF = "footnote"  # "[1]", "footnote 3"
    DOCUMENT_REF = "document"  # "see document X"
    LEGAL_CITATION = "legal"  # "347 U.S. 483"
    WIKILINK = "wikilink"  # "[[Page Name]]"


@dataclass
class InternalReference:
    """A reference from one chunk to another.

    Attributes:
        source_chunk_id: ID of the chunk containing the reference
        target_chunk_id: ID of the referenced chunk (if resolved)
        reference_type: Type of reference
        reference_text: The actual text of the reference
        context: Surrounding text for context
        confidence: Confidence score (0-1) for the match
        resolved: Whether the reference was resolved to a target
    """

    source_chunk_id: str
    target_chunk_id: Optional[str] = None
    reference_type: ReferenceType = ReferenceType.EXPLICIT_CITATION
    reference_text: str = ""
    context: str = ""
    confidence: float = 0.0
    resolved: bool = False

    @property
    def is_resolved(self) -> bool:
        """Check if reference is resolved."""
        return self.resolved and self.target_chunk_id is not None


@dataclass
class ReferencePattern:
    """Pattern for detecting references."""

    pattern: re.Pattern[str]
    ref_type: ReferenceType
    priority: int = 1


# Reference patterns ordered by priority
REFERENCE_PATTERNS: List[ReferencePattern] = [
    # Explicit citations: [Author, Year] or (Author Year)
    ReferencePattern(
        re.compile(r"\[([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?),?\s*(\d{4})\]"),
        ReferenceType.EXPLICIT_CITATION,
        priority=10,
    ),
    ReferencePattern(
        re.compile(r"\(([A-Z][a-zA-Z]+(?:\s+et\s+al\.?)?),?\s*(\d{4})\)"),
        ReferenceType.EXPLICIT_CITATION,
        priority=10,
    ),
    # Section references
    ReferencePattern(
        re.compile(r"(?:see\s+)?[Ss]ection\s+(\d+(?:\.\d+)*)"),
        ReferenceType.SECTION_REF,
        priority=5,
    ),
    # Figure references
    ReferencePattern(
        re.compile(r"(?:see\s+)?[Ff]ig(?:ure)?\.?\s*(\d+(?:\.\d+)?)"),
        ReferenceType.FIGURE_REF,
        priority=5,
    ),
    # Table references
    ReferencePattern(
        re.compile(r"(?:see\s+)?[Tt]able\s+(\d+(?:\.\d+)?)"),
        ReferenceType.TABLE_REF,
        priority=5,
    ),
    # Chapter references
    ReferencePattern(
        re.compile(r"(?:see\s+)?[Cc]hapter\s+(\d+)"),
        ReferenceType.CHAPTER_REF,
        priority=4,
    ),
    # Equation references
    ReferencePattern(
        re.compile(r"[Ee]quation\s*[\(\[]?(\d+)[\)\]]?"),
        ReferenceType.EQUATION_REF,
        priority=3,
    ),
    # Page references
    ReferencePattern(
        re.compile(r"(?:see\s+)?page\s+(\d+)"),
        ReferenceType.PAGE_REF,
        priority=2,
    ),
    # Footnote references
    ReferencePattern(
        re.compile(r"\[(\d{1,3})\]"),
        ReferenceType.FOOTNOTE_REF,
        priority=1,
    ),
    # Legal citations: 347 U.S. 483
    ReferencePattern(
        LEGAL_REPORTER_PATTERN,
        ReferenceType.LEGAL_CITATION,
        priority=8,
    ),
    # Wikilinks: [[Page Name]] or [[Page Name|Alias]]
    ReferencePattern(
        re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"),
        ReferenceType.WIKILINK,
        priority=9,
    ),
]


@dataclass
class LinkMap:
    """Map of links between chunks.

    Attributes:
        links: Dict mapping source_id -> list of target_ids
        reverse_links: Dict mapping target_id -> list of source_ids
        references: List of all extracted references
    """

    links: Dict[str, List[str]] = field(default_factory=dict)
    reverse_links: Dict[str, List[str]] = field(default_factory=dict)
    references: List[InternalReference] = field(default_factory=list)

    def add_reference(self, ref: InternalReference) -> None:
        """Add a reference to the link map."""
        if not ref.is_resolved:
            return

        source = ref.source_chunk_id
        target = ref.target_chunk_id

        # Add forward link
        if source not in self.links:
            self.links[source] = []
        if target not in self.links[source]:
            self.links[source].append(target)

        # Add reverse link
        if target not in self.reverse_links:
            self.reverse_links[target] = []
        if source not in self.reverse_links[target]:
            self.reverse_links[target].append(source)

        self.references.append(ref)

    def get_outgoing(self, chunk_id: str) -> List[str]:
        """Get chunks referenced by this chunk."""
        return self.links.get(chunk_id, [])

    def get_incoming(self, chunk_id: str) -> List[str]:
        """Get chunks that reference this chunk."""
        return self.reverse_links.get(chunk_id, [])

    @property
    def node_count(self) -> int:
        """Count unique nodes in the graph."""
        nodes: Set[str] = set()
        nodes.update(self.links.keys())
        for targets in self.links.values():
            nodes.update(targets)
        return len(nodes)

    @property
    def edge_count(self) -> int:
        """Count total edges in the graph."""
        return sum(len(targets) for targets in self.links.values())


class LinkExtractor:
    """Extracts internal references from chunks.

    Args:
        storage: Storage backend for resolving references
        min_confidence: Minimum confidence for accepting matches
    """

    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        min_confidence: float = 0.5,
    ) -> None:
        """Initialize the link extractor."""
        self.storage = storage
        self.min_confidence = min_confidence
        self._patterns = REFERENCE_PATTERNS

    def extract_references(
        self,
        text: str,
        source_chunk_id: str,
    ) -> List[InternalReference]:
        """Extract all references from text.

        Args:
            text: Text to extract references from
            source_chunk_id: ID of the source chunk

        Returns:
            List of extracted references
        """
        if not text or not text.strip():
            return []

        references: List[InternalReference] = []
        seen_texts: Set[str] = set()

        for pattern_def in self._patterns:
            matches = pattern_def.pattern.finditer(text)

            for match in matches:
                if len(references) >= MAX_REFERENCES_PER_CHUNK:
                    logger.warning(
                        f"Hit max references ({MAX_REFERENCES_PER_CHUNK}) "
                        f"for chunk {source_chunk_id}"
                    )
                    break

                ref_text = match.group(0)

                # Skip duplicates
                if ref_text in seen_texts:
                    continue
                seen_texts.add(ref_text)

                # Get context (50 chars before and after)
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]

                ref = InternalReference(
                    source_chunk_id=source_chunk_id,
                    reference_type=pattern_def.ref_type,
                    reference_text=ref_text,
                    context=context,
                )
                references.append(ref)

        return references

    def resolve_references(
        self,
        references: List[InternalReference],
        source_document: Optional[str] = None,
    ) -> List[InternalReference]:
        """Attempt to resolve references to target chunk IDs.

        Args:
            references: References to resolve
            source_document: Source document for context

        Returns:
            List of references with resolved targets
        """
        if not self.storage:
            return references

        for ref in references:
            resolved = self._resolve_single(ref, source_document)
            if resolved:
                ref.target_chunk_id = resolved[0]
                ref.confidence = resolved[1]
                ref.resolved = True

        return references

    def _resolve_single(
        self,
        ref: InternalReference,
        source_document: Optional[str],
    ) -> Optional[Tuple[str, float]]:
        """Resolve a single reference.

        Args:
            ref: Reference to resolve
            source_document: Source document context

        Returns:
            Tuple of (target_chunk_id, confidence) or None
        """
        if not self.storage:
            return None

        # Build search query based on reference type
        query = self._build_query(ref)
        if not query:
            return None

        try:
            results = self.storage.search(
                query=query,
                limit=MAX_SEARCH_RESULTS,
            )

            if not results:
                return None

            # Find best match
            best_match = self._find_best_match(ref, results, source_document)
            if best_match and best_match[1] >= self.min_confidence:
                return best_match

        except Exception as e:
            logger.warning(f"Failed to resolve reference: {e}")

        return None

    def _build_query(self, ref: InternalReference) -> str:
        """Build search query for a reference."""
        ref_type = ref.reference_type

        if ref_type == ReferenceType.EXPLICIT_CITATION:
            # Extract author and year
            return ref.reference_text

        if ref_type == ReferenceType.SECTION_REF:
            return f"section {ref.reference_text}"

        if ref_type == ReferenceType.FIGURE_REF:
            return f"figure {ref.reference_text}"

        if ref_type == ReferenceType.TABLE_REF:
            return f"table {ref.reference_text}"

        if ref_type == ReferenceType.CHAPTER_REF:
            return f"chapter {ref.reference_text}"

        if ref_type == ReferenceType.LEGAL_CITATION:
            return ref.reference_text

        if ref_type == ReferenceType.WIKILINK:
            # Remove brackets and alias
            text = ref.reference_text.strip("[]")
            if "|" in text:
                text = text.split("|")[0]
            return text

        # Default: use the reference text
        return ref.reference_text

    def _find_best_match(
        self,
        ref: InternalReference,
        results: List[Dict],
        source_document: Optional[str],
    ) -> Optional[Tuple[str, float]]:
        """Find the best matching chunk for a reference."""
        if not results:
            return None

        best_id: Optional[str] = None
        best_score: float = 0.0

        for result in results:
            chunk_id = result.get("id")
            if not chunk_id:
                continue

            # Skip self-references
            if chunk_id == ref.source_chunk_id:
                continue

            # Calculate match score
            score = self._calculate_match_score(ref, result, source_document)

            if score > best_score:
                best_score = score
                best_id = chunk_id

        if best_id:
            return (best_id, best_score)

        return None

    def _calculate_match_score(
        self,
        ref: InternalReference,
        result: Dict,
        source_document: Optional[str],
    ) -> float:
        """Calculate match score between reference and result."""
        score = 0.0
        content = result.get("content", "")
        metadata = result.get("metadata", {})

        # Check if reference text appears in content
        if ref.reference_text.lower() in content.lower():
            score += 0.3

        # Boost for same document
        doc_source = metadata.get("source_file", "")
        if source_document and doc_source == source_document:
            score += 0.2

        # Base retrieval score
        retrieval_score = result.get("score", 0.5)
        score += retrieval_score * 0.5

        return min(score, 1.0)

    def build_link_map(
        self,
        chunks: List[Dict],
    ) -> LinkMap:
        """Build a complete link map from chunks.

        Args:
            chunks: List of chunks with 'id', 'content', 'metadata'

        Returns:
            LinkMap with all extracted references
        """
        link_map = LinkMap()

        for chunk in chunks:
            chunk_id = chunk.get("id", "")
            content = chunk.get("content", "")
            metadata = chunk.get("metadata", {})
            source_doc = metadata.get("source_file")

            if not chunk_id or not content:
                continue

            # Extract references
            refs = self.extract_references(content, chunk_id)

            # Resolve references
            resolved = self.resolve_references(refs, source_doc)

            # Add to link map
            for ref in resolved:
                if ref.is_resolved:
                    link_map.add_reference(ref)

        logger.info(
            f"Built link map: {link_map.node_count} nodes, "
            f"{link_map.edge_count} edges"
        )

        return link_map


def extract_internal_references(
    text: str,
    source_chunk_id: str = "",
) -> List[InternalReference]:
    """Convenience function to extract references.

    Args:
        text: Text to extract from
        source_chunk_id: Source chunk ID

    Returns:
        List of extracted references
    """
    extractor = LinkExtractor()
    return extractor.extract_references(text, source_chunk_id)
