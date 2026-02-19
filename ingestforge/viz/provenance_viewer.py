"""One-Click Source Provenance Viewer.

One-Click Source Provenance Viewer.
Follows NASA JPL Power of Ten rules.

Enables users to click any extracted field and see the highlighted
source text in the original document.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_LINEAGE_DEPTH = 50
MAX_CONTEXT_LINES = 20
MAX_CONTEXT_CHARS = 5000
MAX_SOURCES_PER_CELL = 20
MAX_HIGHLIGHT_LENGTH = 10000
MAX_DOCUMENT_ID_LENGTH = 256
MAX_TRANSFORMATION_STEPS = 100


@dataclass
class ProvenanceReference:
    """Link from a table cell to its source artifact.

    GWT-1: Provenance reference resolution.
    Rule #9: Complete type hints.
    """

    cell_id: str
    artifact_id: str
    source_document_id: str
    char_offset_start: Optional[int] = None
    char_offset_end: Optional[int] = None
    page_number: Optional[int] = None
    section_title: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate reference.

        Rule #5: Assert preconditions.
        """
        assert len(self.cell_id) > 0, "cell_id cannot be empty"
        assert len(self.artifact_id) > 0, "artifact_id cannot be empty"
        assert (
            len(self.source_document_id) <= MAX_DOCUMENT_ID_LENGTH
        ), f"source_document_id exceeds {MAX_DOCUMENT_ID_LENGTH} characters"

    @property
    def has_offsets(self) -> bool:
        """Check if character offsets are available."""
        return self.char_offset_start is not None and self.char_offset_end is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cell_id": self.cell_id,
            "artifact_id": self.artifact_id,
            "source_document_id": self.source_document_id,
            "char_offset_start": self.char_offset_start,
            "char_offset_end": self.char_offset_end,
            "page_number": self.page_number,
            "section_title": self.section_title,
        }


@dataclass
class LineageChain:
    """Full provenance trace from cell to source document.

    GWT-1: Provenance reference resolution.
    Rule #9: Complete type hints.
    """

    references: List[ProvenanceReference] = field(default_factory=list)
    root_artifact_id: str = ""
    depth: int = 0
    transformation_steps: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate lineage chain.

        Rule #5: Assert preconditions.
        """
        assert (
            self.depth <= MAX_LINEAGE_DEPTH
        ), f"lineage depth {self.depth} exceeds maximum {MAX_LINEAGE_DEPTH}"
        assert (
            len(self.transformation_steps) <= MAX_TRANSFORMATION_STEPS
        ), f"transformation steps exceed {MAX_TRANSFORMATION_STEPS}"

    @property
    def is_valid(self) -> bool:
        """Check if lineage chain is valid."""
        return len(self.references) > 0 and len(self.root_artifact_id) > 0

    @property
    def source_count(self) -> int:
        """Number of source references."""
        return len(self.references)

    def add_reference(self, ref: ProvenanceReference) -> bool:
        """Add a reference to the chain.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.references) >= MAX_SOURCES_PER_CELL:
            return False
        self.references.append(ref)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "references": [r.to_dict() for r in self.references],
            "root_artifact_id": self.root_artifact_id,
            "depth": self.depth,
            "transformation_steps": self.transformation_steps[
                :MAX_TRANSFORMATION_STEPS
            ],
            "source_count": self.source_count,
        }


@dataclass
class SourceTextSpan:
    """Extracted source text with context and highlight markers.

    GWT-2: Source text extraction.
    GWT-3: Context window.
    GWT-4: Highlight mapping.
    Rule #9: Complete type hints.
    """

    content: str
    highlight_start: int
    highlight_end: int
    source_location: str
    artifact_id: str = ""
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    context_lines_before: int = 0
    context_lines_after: int = 0

    def __post_init__(self) -> None:
        """Validate span.

        Rule #5: Assert preconditions.
        """
        assert self.highlight_start >= 0, "highlight_start cannot be negative"
        assert (
            self.highlight_end >= self.highlight_start
        ), "highlight_end must be >= highlight_start"
        assert self.highlight_end <= len(
            self.content
        ), "highlight_end exceeds content length"

    @property
    def highlighted_text(self) -> str:
        """Get the highlighted portion of the content."""
        return self.content[self.highlight_start : self.highlight_end]

    @property
    def context_before(self) -> str:
        """Get text before the highlight."""
        return self.content[: self.highlight_start]

    @property
    def context_after(self) -> str:
        """Get text after the highlight."""
        return self.content[self.highlight_end :]

    @property
    def has_context(self) -> bool:
        """Check if span includes context beyond highlight."""
        return self.highlight_start > 0 or self.highlight_end < len(self.content)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content[:MAX_CONTEXT_CHARS],
            "highlighted_text": self.highlighted_text[:MAX_HIGHLIGHT_LENGTH],
            "highlight_start": self.highlight_start,
            "highlight_end": self.highlight_end,
            "source_location": self.source_location,
            "artifact_id": self.artifact_id,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "has_context": self.has_context,
        }

    def format_with_markers(
        self, start_marker: str = ">>>", end_marker: str = "<<<"
    ) -> str:
        """Format content with highlight markers.

        GWT-4: Highlight mapping.

        Args:
            start_marker: Marker before highlighted text.
            end_marker: Marker after highlighted text.

        Returns:
            Formatted string with markers around highlight.
        """
        return (
            self.content[: self.highlight_start]
            + start_marker
            + self.highlighted_text
            + end_marker
            + self.content[self.highlight_end :]
        )


# ---------------------------------------------------------------------------
# Provenance Resolver
# ---------------------------------------------------------------------------


class ProvenanceResolver:
    """Resolves artifact lineage chains.

    GWT-1: Provenance reference resolution.
    Rule #4: Methods < 60 lines.
    """

    def __init__(
        self,
        artifact_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        max_depth: int = MAX_LINEAGE_DEPTH,
    ) -> None:
        """Initialize resolver.

        Rule #5: Assert preconditions.

        Args:
            artifact_lookup: Function to look up artifact by ID.
            max_depth: Maximum lineage traversal depth.
        """
        assert (
            0 < max_depth <= MAX_LINEAGE_DEPTH
        ), f"max_depth must be between 1 and {MAX_LINEAGE_DEPTH}"
        self._artifact_lookup = artifact_lookup or self._default_lookup
        self._max_depth = max_depth

    @staticmethod
    def _default_lookup(artifact_id: str) -> Optional[Dict[str, Any]]:
        """Default lookup returns None (no storage configured)."""
        return None

    def resolve(self, artifact_id: str) -> LineageChain:
        """Resolve full lineage chain for an artifact.

        GWT-1: Provenance reference resolution.
        Rule #1: No recursion - uses iterative loop.

        Args:
            artifact_id: ID of artifact to resolve.

        Returns:
            LineageChain with all provenance references.
        """
        assert artifact_id is not None, "artifact_id cannot be None"
        assert len(artifact_id) > 0, "artifact_id cannot be empty"

        chain = LineageChain(root_artifact_id=artifact_id)
        current_id = artifact_id
        visited: set = set()
        depth = 0

        # Iterative lineage traversal (no recursion - Rule #1)
        while current_id and depth < self._max_depth:
            if current_id in visited:
                break  # Cycle detection
            visited.add(current_id)

            artifact = self._artifact_lookup(current_id)
            if artifact is None:
                break

            ref = self._create_reference_from_artifact(artifact, depth)
            if ref:
                chain.add_reference(ref)

            # Extract transformation step
            provenance = artifact.get("provenance", [])
            if (
                provenance
                and len(chain.transformation_steps) < MAX_TRANSFORMATION_STEPS
            ):
                for step in provenance[-3:]:  # Last 3 steps
                    if step not in chain.transformation_steps:
                        chain.transformation_steps.append(step)

            # Update root if found
            root_id = artifact.get("root_artifact_id")
            if root_id:
                chain.root_artifact_id = root_id

            # Move to parent
            current_id = artifact.get("parent_id")
            depth += 1

        chain.depth = depth
        return chain

    def _create_reference_from_artifact(
        self, artifact: Dict[str, Any], depth: int
    ) -> Optional[ProvenanceReference]:
        """Create ProvenanceReference from artifact data."""
        artifact_id = artifact.get("artifact_id", "")
        if not artifact_id:
            return None

        metadata = artifact.get("metadata", {})
        return ProvenanceReference(
            cell_id=f"depth_{depth}_{artifact_id[:8]}",
            artifact_id=artifact_id,
            source_document_id=artifact.get("document_id", artifact_id),
            char_offset_start=metadata.get("char_offset_start"),
            char_offset_end=metadata.get("char_offset_end"),
            page_number=metadata.get("page_number"),
            section_title=metadata.get("section_title"),
        )

    def get_root_document(self, artifact_id: str) -> Optional[str]:
        """Get the root document ID for an artifact.

        Args:
            artifact_id: ID of artifact to trace.

        Returns:
            Root document ID or None if not found.
        """
        chain = self.resolve(artifact_id)
        if chain.references:
            return chain.references[-1].source_document_id
        return None

    def trace_transformations(self, artifact_id: str) -> List[str]:
        """Get list of transformations applied to create this artifact.

        Args:
            artifact_id: ID of artifact to trace.

        Returns:
            List of transformation step names.
        """
        chain = self.resolve(artifact_id)
        return chain.transformation_steps


# ---------------------------------------------------------------------------
# Source Text Extractor
# ---------------------------------------------------------------------------


class SourceTextExtractor:
    """Extracts source text with context window.

    GWT-2: Source text extraction.
    GWT-3: Context window.
    Rule #4: Methods < 60 lines.
    """

    def __init__(
        self,
        document_reader: Optional[Callable[[str], Optional[str]]] = None,
        default_context_lines: int = 3,
    ) -> None:
        """Initialize extractor.

        Rule #5: Assert preconditions.

        Args:
            document_reader: Function to read document content by ID.
            default_context_lines: Default context lines around highlight.
        """
        assert (
            0 <= default_context_lines <= MAX_CONTEXT_LINES
        ), f"default_context_lines must be between 0 and {MAX_CONTEXT_LINES}"
        self._document_reader = document_reader or self._default_reader
        self._default_context_lines = default_context_lines

    @staticmethod
    def _default_reader(document_id: str) -> Optional[str]:
        """Default reader returns None (no storage configured)."""
        return None

    def extract(
        self,
        reference: ProvenanceReference,
        context_lines: Optional[int] = None,
    ) -> Optional[SourceTextSpan]:
        """Extract source text with context window.

        GWT-2: Source text extraction.
        GWT-3: Context window.

        Args:
            reference: Provenance reference with offsets.
            context_lines: Number of context lines (default: 3).

        Returns:
            SourceTextSpan with highlighted text or None if not found.
        """
        assert reference is not None, "reference cannot be None"

        ctx_lines = (
            context_lines if context_lines is not None else self._default_context_lines
        )
        ctx_lines = min(ctx_lines, MAX_CONTEXT_LINES)

        # Read document content
        content = self._document_reader(reference.source_document_id)
        if content is None:
            return None

        # If no offsets, return chunk-level reference
        if not reference.has_offsets:
            return self._create_chunk_span(reference, content)

        # Extract with context
        return self._extract_with_context(reference, content, ctx_lines)

    def _create_chunk_span(
        self, reference: ProvenanceReference, content: str
    ) -> SourceTextSpan:
        """Create span for chunk without specific offsets."""
        truncated = content[:MAX_CONTEXT_CHARS]
        return SourceTextSpan(
            content=truncated,
            highlight_start=0,
            highlight_end=len(truncated),
            source_location=reference.source_document_id,
            artifact_id=reference.artifact_id,
            page_number=reference.page_number,
            section_title=reference.section_title,
        )

    def _extract_with_context(
        self, reference: ProvenanceReference, content: str, context_lines: int
    ) -> Optional[SourceTextSpan]:
        """Extract text span with context lines."""
        start = reference.char_offset_start or 0
        end = reference.char_offset_end or len(content)

        # Validate offsets
        if start >= len(content) or end > len(content):
            return None

        # Find context boundaries
        context_start = self._find_context_start(content, start, context_lines)
        context_end = self._find_context_end(content, end, context_lines)

        # Adjust highlight offsets relative to context
        highlight_start = start - context_start
        highlight_end = end - context_start

        context_content = content[context_start:context_end]
        if len(context_content) > MAX_CONTEXT_CHARS:
            # Truncate but keep highlight visible
            context_content = context_content[:MAX_CONTEXT_CHARS]
            if highlight_end > MAX_CONTEXT_CHARS:
                highlight_end = MAX_CONTEXT_CHARS

        return SourceTextSpan(
            content=context_content,
            highlight_start=highlight_start,
            highlight_end=min(highlight_end, len(context_content)),
            source_location=reference.source_document_id,
            artifact_id=reference.artifact_id,
            page_number=reference.page_number,
            section_title=reference.section_title,
            context_lines_before=context_lines,
            context_lines_after=context_lines,
        )

    def _find_context_start(
        self, content: str, position: int, context_lines: int
    ) -> int:
        """Find start position including context lines before."""
        if context_lines <= 0:
            return position

        # Count newlines backwards
        lines_found = 0
        pos = position
        while pos > 0 and lines_found < context_lines:
            pos -= 1
            if content[pos] == "\n":
                lines_found += 1

        # Move past the newline if we stopped on one
        if pos > 0 and content[pos] == "\n":
            pos += 1

        return pos

    def _find_context_end(self, content: str, position: int, context_lines: int) -> int:
        """Find end position including context lines after."""
        if context_lines <= 0:
            return position

        # Count newlines forwards
        lines_found = 0
        pos = position
        while pos < len(content) and lines_found < context_lines:
            if content[pos] == "\n":
                lines_found += 1
            pos += 1

        return pos

    def get_page_context(self, document_id: str, page_number: int) -> Optional[str]:
        """Get full page content for context.

        Args:
            document_id: ID of the document.
            page_number: Page number (1-indexed).

        Returns:
            Page content or None if not found.
        """
        # This would typically integrate with document storage
        # For now, return None (not implemented)
        return None


# ---------------------------------------------------------------------------
# Provenance Viewer (Integration Layer)
# ---------------------------------------------------------------------------


class ProvenanceViewer:
    """Integration layer for viewing provenance from table cells.

    GWT-5: Multi-source aggregation.
    Rule #4: Methods < 60 lines.
    """

    def __init__(
        self,
        resolver: Optional[ProvenanceResolver] = None,
        extractor: Optional[SourceTextExtractor] = None,
    ) -> None:
        """Initialize viewer.

        Args:
            resolver: Provenance resolver instance.
            extractor: Source text extractor instance.
        """
        self._resolver = resolver or ProvenanceResolver()
        self._extractor = extractor or SourceTextExtractor()

    def from_artifact_id(
        self,
        artifact_id: str,
        context_lines: int = 3,
    ) -> List[SourceTextSpan]:
        """Get provenance spans for an artifact.

        GWT-5: Multi-source aggregation.

        Args:
            artifact_id: ID of artifact to view.
            context_lines: Context lines around highlight.

        Returns:
            List of source text spans.
        """
        assert artifact_id is not None, "artifact_id cannot be None"
        assert len(artifact_id) > 0, "artifact_id cannot be empty"
        assert (
            0 <= context_lines <= MAX_CONTEXT_LINES
        ), f"context_lines must be between 0 and {MAX_CONTEXT_LINES}"

        chain = self._resolver.resolve(artifact_id)
        spans: List[SourceTextSpan] = []

        for ref in chain.references[:MAX_SOURCES_PER_CELL]:
            span = self._extractor.extract(ref, context_lines)
            if span:
                spans.append(span)

        return spans

    def from_table_cell(
        self,
        data_source: Any,
        row_index: int,
        column: str,
        context_lines: int = 3,
    ) -> List[SourceTextSpan]:
        """Get provenance spans for a table cell.

        GWT-5: Multi-source aggregation.

        Args:
            data_source: TableDataSource instance.
            row_index: Row index in data source.
            column: Column field name.
            context_lines: Context lines around highlight.

        Returns:
            List of source text spans.
        """
        assert data_source is not None, "data_source cannot be None"
        assert row_index >= 0, "row_index cannot be negative"
        assert column is not None, "column cannot be None"

        # Extract artifact_id from row metadata
        rows = getattr(data_source, "rows", [])
        if row_index >= len(rows):
            return []

        row = rows[row_index]
        metadata = row.get("_metadata", row.get("metadata", {}))

        # Try column-specific provenance first
        column_provenance = metadata.get(f"_provenance_{column}", {})
        artifact_id = column_provenance.get("artifact_id")

        # Fall back to row-level provenance
        if not artifact_id:
            artifact_id = metadata.get("_artifact_id", metadata.get("artifact_id"))

        if not artifact_id:
            return []

        return self.from_artifact_id(artifact_id, context_lines)

    def format_for_display(
        self,
        spans: List[SourceTextSpan],
        include_location: bool = True,
    ) -> str:
        """Format spans for text display.

        Args:
            spans: List of source text spans.
            include_location: Include source location header.

        Returns:
            Formatted display string.
        """
        if not spans:
            return "No provenance information available."

        parts: List[str] = []
        for i, span in enumerate(spans[:MAX_SOURCES_PER_CELL]):
            if include_location:
                header = f"[Source {i + 1}: {span.source_location}"
                if span.page_number:
                    header += f", p.{span.page_number}"
                header += "]"
                parts.append(header)

            parts.append(span.format_with_markers())
            parts.append("")  # Blank line between sources

        return "\n".join(parts).rstrip()

    def to_dict(self, spans: List[SourceTextSpan]) -> Dict[str, Any]:
        """Convert spans to dictionary format.

        Args:
            spans: List of source text spans.

        Returns:
            Dictionary with provenance data.
        """
        return {
            "source_count": len(spans),
            "spans": [s.to_dict() for s in spans[:MAX_SOURCES_PER_CELL]],
        }


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_provenance_reference(
    artifact_id: str,
    source_document_id: str,
    char_start: Optional[int] = None,
    char_end: Optional[int] = None,
) -> ProvenanceReference:
    """Convenience function to create a ProvenanceReference.

    Args:
        artifact_id: ID of the artifact.
        source_document_id: ID of the source document.
        char_start: Start character offset.
        char_end: End character offset.

    Returns:
        Configured ProvenanceReference.
    """
    return ProvenanceReference(
        cell_id=f"ref_{artifact_id[:8]}",
        artifact_id=artifact_id,
        source_document_id=source_document_id,
        char_offset_start=char_start,
        char_offset_end=char_end,
    )


def create_provenance_viewer(
    artifact_lookup: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    document_reader: Optional[Callable[[str], Optional[str]]] = None,
) -> ProvenanceViewer:
    """Convenience function to create a configured ProvenanceViewer.

    Args:
        artifact_lookup: Function to look up artifacts by ID.
        document_reader: Function to read document content.

    Returns:
        Configured ProvenanceViewer.
    """
    resolver = ProvenanceResolver(artifact_lookup=artifact_lookup)
    extractor = SourceTextExtractor(document_reader=document_reader)
    return ProvenanceViewer(resolver=resolver, extractor=extractor)


def view_cell_provenance(
    artifact_id: str,
    artifact_lookup: Callable[[str], Optional[Dict[str, Any]]],
    document_reader: Callable[[str], Optional[str]],
    context_lines: int = 3,
) -> List[SourceTextSpan]:
    """Convenience function to view provenance for an artifact.

    Args:
        artifact_id: ID of the artifact.
        artifact_lookup: Function to look up artifacts.
        document_reader: Function to read documents.
        context_lines: Context lines around highlight.

    Returns:
        List of source text spans.
    """
    viewer = create_provenance_viewer(artifact_lookup, document_reader)
    return viewer.from_artifact_id(artifact_id, context_lines)
