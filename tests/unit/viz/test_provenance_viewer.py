"""Tests for One-Click Source Provenance Viewer.

Comprehensive GWT test suite following NASA JPL Power of Ten rules.
"""

import pytest
from typing import Any, Dict, Optional

from ingestforge.viz.provenance_viewer import (
    # Constants
    MAX_LINEAGE_DEPTH,
    MAX_CONTEXT_LINES,
    MAX_CONTEXT_CHARS,
    MAX_SOURCES_PER_CELL,
    MAX_DOCUMENT_ID_LENGTH,
    MAX_TRANSFORMATION_STEPS,
    # Classes
    ProvenanceReference,
    LineageChain,
    SourceTextSpan,
    ProvenanceResolver,
    SourceTextExtractor,
    ProvenanceViewer,
    # Convenience functions
    create_provenance_reference,
    create_provenance_viewer,
    view_cell_provenance,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_artifact_data() -> Dict[str, Dict[str, Any]]:
    """Sample artifact data for testing."""
    return {
        "chunk-001": {
            "artifact_id": "chunk-001",
            "document_id": "doc-001",
            "parent_id": "text-001",
            "root_artifact_id": "file-001",
            "metadata": {
                "char_offset_start": 100,
                "char_offset_end": 150,
                "page_number": 3,
                "section_title": "Introduction",
            },
            "provenance": ["from-file", "split-text", "chunked"],
        },
        "text-001": {
            "artifact_id": "text-001",
            "document_id": "doc-001",
            "parent_id": "file-001",
            "root_artifact_id": "file-001",
            "metadata": {},
            "provenance": ["from-file", "extracted-text"],
        },
        "file-001": {
            "artifact_id": "file-001",
            "document_id": "doc-001",
            "parent_id": None,
            "root_artifact_id": None,
            "metadata": {
                "source_location": "/path/to/document.pdf",
            },
            "provenance": ["ingested"],
        },
    }


@pytest.fixture
def sample_document_content() -> Dict[str, str]:
    """Sample document content for testing."""
    return {
        "doc-001": """This is the beginning of the document.
Line 2 of the document content.
Line 3 has some important information.
This is line 4 with the highlighted text here in this section.
Line 5 continues the narrative.
Line 6 provides more context.
Line 7 is the final line of context.""",
    }


@pytest.fixture
def artifact_lookup(sample_artifact_data):
    """Create artifact lookup function."""

    def lookup(artifact_id: str) -> Optional[Dict[str, Any]]:
        return sample_artifact_data.get(artifact_id)

    return lookup


@pytest.fixture
def document_reader(sample_document_content):
    """Create document reader function."""

    def reader(document_id: str) -> Optional[str]:
        return sample_document_content.get(document_id)

    return reader


# ---------------------------------------------------------------------------
# TestProvenanceReference
# ---------------------------------------------------------------------------


class TestProvenanceReference:
    """Tests for ProvenanceReference dataclass."""

    def test_create_basic_reference(self):
        """Test creating a basic provenance reference."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )
        assert ref.cell_id == "cell-001"
        assert ref.artifact_id == "artifact-001"
        assert ref.source_document_id == "doc-001"

    def test_create_reference_with_offsets(self):
        """Test creating reference with character offsets."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=100,
            char_offset_end=200,
        )
        assert ref.char_offset_start == 100
        assert ref.char_offset_end == 200
        assert ref.has_offsets is True

    def test_has_offsets_false_when_missing(self):
        """Test has_offsets is False when offsets missing."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )
        assert ref.has_offsets is False

    def test_has_offsets_false_when_partial(self):
        """Test has_offsets is False when only one offset provided."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=100,
        )
        assert ref.has_offsets is False

    def test_to_dict(self):
        """Test converting reference to dictionary."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=100,
            char_offset_end=200,
            page_number=5,
        )
        result = ref.to_dict()
        assert result["cell_id"] == "cell-001"
        assert result["artifact_id"] == "artifact-001"
        assert result["page_number"] == 5

    def test_empty_cell_id_raises(self):
        """Test that empty cell_id raises assertion error."""
        with pytest.raises(AssertionError, match="cell_id cannot be empty"):
            ProvenanceReference(
                cell_id="",
                artifact_id="artifact-001",
                source_document_id="doc-001",
            )

    def test_empty_artifact_id_raises(self):
        """Test that empty artifact_id raises assertion error."""
        with pytest.raises(AssertionError, match="artifact_id cannot be empty"):
            ProvenanceReference(
                cell_id="cell-001",
                artifact_id="",
                source_document_id="doc-001",
            )

    def test_long_document_id_raises(self):
        """Test that overly long document_id raises assertion error."""
        with pytest.raises(AssertionError, match="source_document_id exceeds"):
            ProvenanceReference(
                cell_id="cell-001",
                artifact_id="artifact-001",
                source_document_id="x" * (MAX_DOCUMENT_ID_LENGTH + 1),
            )


# ---------------------------------------------------------------------------
# TestLineageChain
# ---------------------------------------------------------------------------


class TestLineageChain:
    """Tests for LineageChain dataclass."""

    def test_create_empty_chain(self):
        """Test creating empty lineage chain."""
        chain = LineageChain()
        assert chain.references == []
        assert chain.root_artifact_id == ""
        assert chain.depth == 0
        assert chain.is_valid is False

    def test_create_valid_chain(self):
        """Test creating valid lineage chain."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )
        chain = LineageChain(
            references=[ref],
            root_artifact_id="root-001",
            depth=2,
        )
        assert chain.is_valid is True
        assert chain.source_count == 1

    def test_add_reference(self):
        """Test adding reference to chain."""
        chain = LineageChain(root_artifact_id="root-001")
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )
        result = chain.add_reference(ref)
        assert result is True
        assert chain.source_count == 1

    def test_add_reference_at_capacity(self):
        """Test adding reference when at capacity."""
        refs = [
            ProvenanceReference(
                cell_id=f"cell-{i}",
                artifact_id=f"artifact-{i}",
                source_document_id=f"doc-{i}",
            )
            for i in range(MAX_SOURCES_PER_CELL)
        ]
        chain = LineageChain(references=refs, root_artifact_id="root-001")

        new_ref = ProvenanceReference(
            cell_id="cell-new",
            artifact_id="artifact-new",
            source_document_id="doc-new",
        )
        result = chain.add_reference(new_ref)
        assert result is False
        assert chain.source_count == MAX_SOURCES_PER_CELL

    def test_depth_exceeds_maximum_raises(self):
        """Test that excessive depth raises assertion error."""
        with pytest.raises(AssertionError, match="lineage depth"):
            LineageChain(depth=MAX_LINEAGE_DEPTH + 1)

    def test_transformation_steps_exceed_maximum_raises(self):
        """Test that too many transformation steps raises assertion."""
        with pytest.raises(AssertionError, match="transformation steps"):
            LineageChain(transformation_steps=["step"] * (MAX_TRANSFORMATION_STEPS + 1))

    def test_to_dict(self):
        """Test converting chain to dictionary."""
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )
        chain = LineageChain(
            references=[ref],
            root_artifact_id="root-001",
            depth=3,
            transformation_steps=["step1", "step2"],
        )
        result = chain.to_dict()
        assert result["root_artifact_id"] == "root-001"
        assert result["depth"] == 3
        assert result["source_count"] == 1
        assert len(result["transformation_steps"]) == 2


# ---------------------------------------------------------------------------
# TestSourceTextSpan
# ---------------------------------------------------------------------------


class TestSourceTextSpan:
    """Tests for SourceTextSpan dataclass."""

    def test_create_basic_span(self):
        """Test creating basic source text span."""
        span = SourceTextSpan(
            content="Hello world",
            highlight_start=6,
            highlight_end=11,
            source_location="/path/to/doc.pdf",
        )
        assert span.highlighted_text == "world"
        assert span.context_before == "Hello "
        assert span.context_after == ""

    def test_highlighted_text_extraction(self):
        """Test extracting highlighted portion."""
        content = "Before [HIGHLIGHTED] After"
        span = SourceTextSpan(
            content=content,
            highlight_start=8,
            highlight_end=19,
            source_location="doc.pdf",
        )
        assert span.highlighted_text == "HIGHLIGHTED"

    def test_context_before_and_after(self):
        """Test extracting context before and after highlight."""
        content = "Line1\nLine2\nHIGHLIGHT\nLine4\nLine5"
        span = SourceTextSpan(
            content=content,
            highlight_start=12,
            highlight_end=21,
            source_location="doc.pdf",
        )
        assert span.context_before == "Line1\nLine2\n"
        assert span.context_after == "\nLine4\nLine5"
        assert span.has_context is True

    def test_has_context_false_when_full_highlight(self):
        """Test has_context is False when entire content is highlighted."""
        span = SourceTextSpan(
            content="All highlighted",
            highlight_start=0,
            highlight_end=15,
            source_location="doc.pdf",
        )
        assert span.has_context is False

    def test_format_with_markers(self):
        """Test formatting with highlight markers."""
        span = SourceTextSpan(
            content="Before HIGHLIGHT After",
            highlight_start=7,
            highlight_end=16,
            source_location="doc.pdf",
        )
        result = span.format_with_markers(">>>", "<<<")
        assert result == "Before >>>HIGHLIGHT<<< After"

    def test_format_with_custom_markers(self):
        """Test formatting with custom markers."""
        span = SourceTextSpan(
            content="abc",
            highlight_start=1,
            highlight_end=2,
            source_location="doc.pdf",
        )
        result = span.format_with_markers("[", "]")
        assert result == "a[b]c"

    def test_negative_highlight_start_raises(self):
        """Test that negative highlight_start raises assertion error."""
        with pytest.raises(AssertionError, match="highlight_start cannot be negative"):
            SourceTextSpan(
                content="test",
                highlight_start=-1,
                highlight_end=2,
                source_location="doc.pdf",
            )

    def test_highlight_end_before_start_raises(self):
        """Test that highlight_end < start raises assertion error."""
        with pytest.raises(
            AssertionError, match="highlight_end must be >= highlight_start"
        ):
            SourceTextSpan(
                content="test",
                highlight_start=5,
                highlight_end=2,
                source_location="doc.pdf",
            )

    def test_highlight_end_exceeds_content_raises(self):
        """Test that highlight_end > content length raises assertion."""
        with pytest.raises(
            AssertionError, match="highlight_end exceeds content length"
        ):
            SourceTextSpan(
                content="test",
                highlight_start=0,
                highlight_end=10,
                source_location="doc.pdf",
            )

    def test_to_dict(self):
        """Test converting span to dictionary."""
        span = SourceTextSpan(
            content="Test content",
            highlight_start=5,
            highlight_end=12,
            source_location="doc.pdf",
            page_number=3,
        )
        result = span.to_dict()
        assert result["highlighted_text"] == "content"
        assert result["page_number"] == 3
        assert result["has_context"] is True


# ---------------------------------------------------------------------------
# TestGWT1ProvenanceReferenceResolution
# ---------------------------------------------------------------------------


class TestGWT1ProvenanceReferenceResolution:
    """GWT-1: Provenance Reference Resolution tests."""

    def test_resolve_single_artifact(self, artifact_lookup):
        """Given artifact_id, When resolved, Then lineage chain returned."""
        resolver = ProvenanceResolver(artifact_lookup=artifact_lookup)
        chain = resolver.resolve("chunk-001")

        assert chain.is_valid
        assert chain.source_count >= 1

    def test_resolve_traces_to_root(self, artifact_lookup):
        """Given chunk artifact, When resolved, Then traces to root file."""
        resolver = ProvenanceResolver(artifact_lookup=artifact_lookup)
        chain = resolver.resolve("chunk-001")

        assert chain.root_artifact_id == "file-001"

    def test_resolve_captures_transformation_steps(self, artifact_lookup):
        """Given artifact with provenance, When resolved, Then steps captured."""
        resolver = ProvenanceResolver(artifact_lookup=artifact_lookup)
        chain = resolver.resolve("chunk-001")

        assert len(chain.transformation_steps) > 0
        assert "chunked" in chain.transformation_steps

    def test_resolve_unknown_artifact_returns_empty_chain(self):
        """Given unknown artifact_id, When resolved, Then empty chain returned."""
        resolver = ProvenanceResolver()
        chain = resolver.resolve("unknown-artifact")

        assert chain.is_valid is False
        assert chain.source_count == 0

    def test_resolve_handles_cycle_detection(self):
        """Given circular lineage, When resolved, Then cycle detected."""
        cyclic_data = {
            "a": {"artifact_id": "a", "parent_id": "b"},
            "b": {"artifact_id": "b", "parent_id": "a"},
        }

        def lookup(aid):
            return cyclic_data.get(aid)

        resolver = ProvenanceResolver(artifact_lookup=lookup)
        chain = resolver.resolve("a")

        # Should not hang, and depth should be limited
        assert chain.depth <= 2

    def test_resolve_respects_max_depth(self):
        """Given deep lineage, When resolved, Then max depth respected."""
        # Create a chain deeper than allowed
        deep_data = {}
        for i in range(MAX_LINEAGE_DEPTH + 10):
            deep_data[f"artifact-{i}"] = {
                "artifact_id": f"artifact-{i}",
                "parent_id": f"artifact-{i+1}" if i < MAX_LINEAGE_DEPTH + 9 else None,
            }

        def lookup(aid):
            return deep_data.get(aid)

        resolver = ProvenanceResolver(artifact_lookup=lookup, max_depth=10)
        chain = resolver.resolve("artifact-0")

        assert chain.depth <= 10

    def test_get_root_document(self, artifact_lookup):
        """Given artifact, When get_root_document called, Then root ID returned."""
        resolver = ProvenanceResolver(artifact_lookup=artifact_lookup)
        root = resolver.get_root_document("chunk-001")

        assert root is not None

    def test_trace_transformations(self, artifact_lookup):
        """Given artifact, When trace_transformations called, Then steps returned."""
        resolver = ProvenanceResolver(artifact_lookup=artifact_lookup)
        steps = resolver.trace_transformations("chunk-001")

        assert isinstance(steps, list)


# ---------------------------------------------------------------------------
# TestGWT2SourceTextExtraction
# ---------------------------------------------------------------------------


class TestGWT2SourceTextExtraction:
    """GWT-2: Source Text Extraction tests."""

    def test_extract_with_offsets(self, document_reader):
        """Given reference with offsets, When extracted, Then exact span returned."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=142,
            char_offset_end=158,
        )

        span = extractor.extract(ref, context_lines=0)

        assert span is not None
        assert len(span.highlighted_text) == 16

    def test_extract_without_offsets_returns_full_chunk(self, document_reader):
        """Given reference without offsets, When extracted, Then full content."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )

        span = extractor.extract(ref)

        assert span is not None
        assert span.highlight_start == 0
        assert span.highlight_end == len(span.content)

    def test_extract_unknown_document_returns_none(self):
        """Given unknown document, When extracted, Then None returned."""
        extractor = SourceTextExtractor()
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="unknown-doc",
            char_offset_start=0,
            char_offset_end=10,
        )

        span = extractor.extract(ref)

        assert span is None

    def test_extract_invalid_offsets_returns_none(self, document_reader):
        """Given invalid offsets, When extracted, Then None returned."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=99999,
            char_offset_end=99999 + 10,
        )

        span = extractor.extract(ref)

        assert span is None

    def test_extract_preserves_artifact_metadata(self, document_reader):
        """Given reference with metadata, When extracted, Then metadata preserved."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            page_number=5,
            section_title="Introduction",
        )

        span = extractor.extract(ref)

        assert span.page_number == 5
        assert span.section_title == "Introduction"


# ---------------------------------------------------------------------------
# TestGWT3ContextWindow
# ---------------------------------------------------------------------------


class TestGWT3ContextWindow:
    """GWT-3: Context Window tests."""

    def test_extract_with_context_lines(self, document_reader):
        """Given context_lines, When extracted, Then context included."""
        extractor = SourceTextExtractor(
            document_reader=document_reader,
            default_context_lines=2,
        )
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=75,
            char_offset_end=90,
        )

        span = extractor.extract(ref, context_lines=2)

        assert span is not None
        assert span.has_context is True
        assert span.highlight_start > 0  # Has context before

    def test_extract_with_zero_context(self, document_reader):
        """Given zero context_lines, When extracted, Then no context."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=75,
            char_offset_end=90,
        )

        span = extractor.extract(ref, context_lines=0)

        assert span is not None
        assert span.highlight_start == 0  # No context before

    def test_context_lines_capped_at_maximum(self, document_reader):
        """Given excessive context_lines, When extracted, Then capped."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=75,
            char_offset_end=90,
        )

        span = extractor.extract(ref, context_lines=MAX_CONTEXT_LINES + 100)

        assert span is not None

    def test_context_at_document_start(self, document_reader):
        """Given highlight at start, When extracted, Then handles boundary."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=0,
            char_offset_end=10,
        )

        span = extractor.extract(ref, context_lines=3)

        assert span is not None
        assert span.highlight_start == 0

    def test_context_at_document_end(self, document_reader):
        """Given highlight at end, When extracted, Then handles boundary."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        content = document_reader("doc-001")
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=len(content) - 20,
            char_offset_end=len(content),
        )

        span = extractor.extract(ref, context_lines=3)

        assert span is not None


# ---------------------------------------------------------------------------
# TestGWT4HighlightMapping
# ---------------------------------------------------------------------------


class TestGWT4HighlightMapping:
    """GWT-4: Highlight Mapping tests."""

    def test_highlight_markers_correct_position(self, document_reader):
        """Given span, When formatted, Then markers at correct position."""
        extractor = SourceTextExtractor(document_reader=document_reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=37,
            char_offset_end=47,
        )

        span = extractor.extract(ref, context_lines=0)
        formatted = span.format_with_markers(">>>", "<<<")

        assert ">>>" in formatted
        assert "<<<" in formatted
        # Markers should wrap the exact highlight
        assert ">>>" + span.highlighted_text + "<<<" in formatted

    def test_highlight_text_property_accurate(self):
        """Given span, When highlighted_text accessed, Then accurate."""
        span = SourceTextSpan(
            content="abc EXACT def",
            highlight_start=4,
            highlight_end=9,
            source_location="doc.pdf",
        )

        assert span.highlighted_text == "EXACT"

    def test_context_before_property_accurate(self):
        """Given span, When context_before accessed, Then accurate."""
        span = SourceTextSpan(
            content="abc EXACT def",
            highlight_start=4,
            highlight_end=9,
            source_location="doc.pdf",
        )

        assert span.context_before == "abc "

    def test_context_after_property_accurate(self):
        """Given span, When context_after accessed, Then accurate."""
        span = SourceTextSpan(
            content="abc EXACT def",
            highlight_start=4,
            highlight_end=9,
            source_location="doc.pdf",
        )

        assert span.context_after == " def"


# ---------------------------------------------------------------------------
# TestGWT5MultiSourceAggregation
# ---------------------------------------------------------------------------


class TestGWT5MultiSourceAggregation:
    """GWT-5: Multi-Source Aggregation tests."""

    def test_viewer_aggregates_multiple_sources(self, artifact_lookup, document_reader):
        """Given multi-source cell, When viewed, Then all sources listed."""
        viewer = create_provenance_viewer(artifact_lookup, document_reader)
        spans = viewer.from_artifact_id("chunk-001", context_lines=2)

        assert isinstance(spans, list)

    def test_viewer_from_table_cell_extracts_artifact_id(self):
        """Given table cell with metadata, When viewed, Then artifact found."""

        # Mock data source
        class MockDataSource:
            rows = [
                {
                    "name": "Test",
                    "_metadata": {"_artifact_id": "test-artifact"},
                }
            ]

        viewer = ProvenanceViewer()
        spans = viewer.from_table_cell(MockDataSource(), 0, "name")

        # Will return empty since no storage configured
        assert isinstance(spans, list)

    def test_viewer_from_table_cell_handles_missing_metadata(self):
        """Given cell without metadata, When viewed, Then empty list."""

        class MockDataSource:
            rows = [{"name": "Test"}]

        viewer = ProvenanceViewer()
        spans = viewer.from_table_cell(MockDataSource(), 0, "name")

        assert spans == []

    def test_viewer_from_table_cell_invalid_row_index(self):
        """Given invalid row index, When viewed, Then empty list."""

        class MockDataSource:
            rows = []

        viewer = ProvenanceViewer()
        spans = viewer.from_table_cell(MockDataSource(), 99, "name")

        assert spans == []

    def test_format_for_display_empty_spans(self):
        """Given empty spans, When formatted, Then appropriate message."""
        viewer = ProvenanceViewer()
        result = viewer.format_for_display([])

        assert "No provenance" in result

    def test_format_for_display_single_span(self):
        """Given single span, When formatted, Then includes location."""
        viewer = ProvenanceViewer()
        span = SourceTextSpan(
            content="Context before HIGHLIGHT context after",
            highlight_start=15,
            highlight_end=24,
            source_location="/path/to/doc.pdf",
            page_number=3,
        )

        result = viewer.format_for_display([span])

        assert "[Source 1:" in result
        assert "/path/to/doc.pdf" in result
        assert "p.3" in result
        assert ">>>" in result  # Default markers

    def test_format_for_display_multiple_spans(self):
        """Given multiple spans, When formatted, Then all included."""
        viewer = ProvenanceViewer()
        spans = [
            SourceTextSpan(
                content="Source 1 text",
                highlight_start=0,
                highlight_end=8,
                source_location="doc1.pdf",
            ),
            SourceTextSpan(
                content="Source 2 text",
                highlight_start=0,
                highlight_end=8,
                source_location="doc2.pdf",
            ),
        ]

        result = viewer.format_for_display(spans)

        assert "[Source 1:" in result
        assert "[Source 2:" in result

    def test_to_dict_output(self):
        """Given spans, When to_dict called, Then correct structure."""
        viewer = ProvenanceViewer()
        span = SourceTextSpan(
            content="test",
            highlight_start=0,
            highlight_end=4,
            source_location="doc.pdf",
        )

        result = viewer.to_dict([span])

        assert result["source_count"] == 1
        assert len(result["spans"]) == 1


# ---------------------------------------------------------------------------
# TestProvenanceResolver
# ---------------------------------------------------------------------------


class TestProvenanceResolver:
    """Additional tests for ProvenanceResolver."""

    def test_init_default_max_depth(self):
        """Test default max depth initialization."""
        resolver = ProvenanceResolver()
        assert resolver._max_depth == MAX_LINEAGE_DEPTH

    def test_init_custom_max_depth(self):
        """Test custom max depth initialization."""
        resolver = ProvenanceResolver(max_depth=10)
        assert resolver._max_depth == 10

    def test_init_invalid_max_depth_raises(self):
        """Test invalid max depth raises assertion."""
        with pytest.raises(AssertionError):
            ProvenanceResolver(max_depth=0)

        with pytest.raises(AssertionError):
            ProvenanceResolver(max_depth=MAX_LINEAGE_DEPTH + 1)

    def test_resolve_none_artifact_id_raises(self):
        """Test None artifact_id raises assertion."""
        resolver = ProvenanceResolver()
        with pytest.raises(AssertionError, match="artifact_id cannot be None"):
            resolver.resolve(None)

    def test_resolve_empty_artifact_id_raises(self):
        """Test empty artifact_id raises assertion."""
        resolver = ProvenanceResolver()
        with pytest.raises(AssertionError, match="artifact_id cannot be empty"):
            resolver.resolve("")


# ---------------------------------------------------------------------------
# TestSourceTextExtractor
# ---------------------------------------------------------------------------


class TestSourceTextExtractor:
    """Additional tests for SourceTextExtractor."""

    def test_init_default_context_lines(self):
        """Test default context lines initialization."""
        extractor = SourceTextExtractor()
        assert extractor._default_context_lines == 3

    def test_init_custom_context_lines(self):
        """Test custom context lines initialization."""
        extractor = SourceTextExtractor(default_context_lines=5)
        assert extractor._default_context_lines == 5

    def test_init_invalid_context_lines_raises(self):
        """Test invalid context lines raises assertion."""
        with pytest.raises(AssertionError):
            SourceTextExtractor(default_context_lines=-1)

        with pytest.raises(AssertionError):
            SourceTextExtractor(default_context_lines=MAX_CONTEXT_LINES + 1)

    def test_extract_none_reference_raises(self):
        """Test None reference raises assertion."""
        extractor = SourceTextExtractor()
        with pytest.raises(AssertionError, match="reference cannot be None"):
            extractor.extract(None)


# ---------------------------------------------------------------------------
# TestConvenienceFunctions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_provenance_reference_basic(self):
        """Test creating provenance reference via convenience function."""
        ref = create_provenance_reference(
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )

        assert ref.artifact_id == "artifact-001"
        assert ref.source_document_id == "doc-001"
        assert ref.has_offsets is False

    def test_create_provenance_reference_with_offsets(self):
        """Test creating reference with offsets."""
        ref = create_provenance_reference(
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_start=100,
            char_end=200,
        )

        assert ref.char_offset_start == 100
        assert ref.char_offset_end == 200
        assert ref.has_offsets is True

    def test_create_provenance_viewer_default(self):
        """Test creating viewer with defaults."""
        viewer = create_provenance_viewer()

        assert viewer is not None
        assert isinstance(viewer, ProvenanceViewer)

    def test_create_provenance_viewer_with_callbacks(
        self, artifact_lookup, document_reader
    ):
        """Test creating viewer with custom callbacks."""
        viewer = create_provenance_viewer(
            artifact_lookup=artifact_lookup,
            document_reader=document_reader,
        )

        assert viewer is not None

    def test_view_cell_provenance_integration(self, artifact_lookup, document_reader):
        """Test view_cell_provenance convenience function."""
        spans = view_cell_provenance(
            artifact_id="chunk-001",
            artifact_lookup=artifact_lookup,
            document_reader=document_reader,
            context_lines=2,
        )

        assert isinstance(spans, list)


# ---------------------------------------------------------------------------
# TestJPLCompliance
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for NASA JPL Power of Ten compliance."""

    def test_rule2_max_lineage_depth_constant(self):
        """Rule #2: MAX_LINEAGE_DEPTH is defined and reasonable."""
        assert MAX_LINEAGE_DEPTH == 50
        assert MAX_LINEAGE_DEPTH > 0

    def test_rule2_max_context_lines_constant(self):
        """Rule #2: MAX_CONTEXT_LINES is defined and reasonable."""
        assert MAX_CONTEXT_LINES == 20
        assert MAX_CONTEXT_LINES > 0

    def test_rule2_max_sources_per_cell_constant(self):
        """Rule #2: MAX_SOURCES_PER_CELL is defined and reasonable."""
        assert MAX_SOURCES_PER_CELL == 20
        assert MAX_SOURCES_PER_CELL > 0

    def test_rule5_preconditions_enforced(self):
        """Rule #5: Preconditions enforced via assertions."""
        # Empty cell_id
        with pytest.raises(AssertionError):
            ProvenanceReference(
                cell_id="",
                artifact_id="a",
                source_document_id="d",
            )

        # Invalid highlight range
        with pytest.raises(AssertionError):
            SourceTextSpan(
                content="test",
                highlight_start=5,
                highlight_end=3,
                source_location="doc.pdf",
            )

    def test_rule7_return_values_checked(self):
        """Rule #7: Return values properly typed and checked."""
        resolver = ProvenanceResolver()
        chain = resolver.resolve("unknown")

        # Should return valid object, not None
        assert chain is not None
        assert isinstance(chain, LineageChain)
        assert chain.is_valid is False

    def test_rule9_type_hints_present(self):
        """Rule #9: Type hints present on all public methods."""
        # Check that type hints are present (via __annotations__)
        assert "artifact_id" in ProvenanceReference.__annotations__
        assert "content" in SourceTextSpan.__annotations__
        assert "references" in LineageChain.__annotations__


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_document_content(self):
        """Test handling of empty document content."""

        def reader(doc_id):
            return ""

        extractor = SourceTextExtractor(document_reader=reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )

        span = extractor.extract(ref)

        assert span is not None
        assert span.content == ""

    def test_single_character_document(self):
        """Test handling of single character document."""

        def reader(doc_id):
            return "X"

        extractor = SourceTextExtractor(document_reader=reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=0,
            char_offset_end=1,
        )

        span = extractor.extract(ref, context_lines=0)

        assert span is not None
        assert span.highlighted_text == "X"

    def test_very_long_content_truncated(self):
        """Test that very long content is truncated."""
        long_content = "X" * (MAX_CONTEXT_CHARS + 1000)

        def reader(doc_id):
            return long_content

        extractor = SourceTextExtractor(document_reader=reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
        )

        span = extractor.extract(ref)

        assert len(span.content) <= MAX_CONTEXT_CHARS

    def test_unicode_content_handling(self):
        """Test handling of unicode content."""
        unicode_content = "Hello \u4e16\u754c \U0001F600 World"

        def reader(doc_id):
            return unicode_content

        extractor = SourceTextExtractor(document_reader=reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=6,
            char_offset_end=8,
        )

        span = extractor.extract(ref, context_lines=0)

        assert span is not None

    def test_newlines_only_content(self):
        """Test handling of content with only newlines."""

        def reader(doc_id):
            return "\n\n\n\n\n"

        extractor = SourceTextExtractor(document_reader=reader)
        ref = ProvenanceReference(
            cell_id="cell-001",
            artifact_id="artifact-001",
            source_document_id="doc-001",
            char_offset_start=1,
            char_offset_end=2,
        )

        span = extractor.extract(ref, context_lines=1)

        assert span is not None

    def test_viewer_handles_missing_rows_attribute(self):
        """Test viewer handles data source without rows attribute."""

        class BadDataSource:
            pass

        viewer = ProvenanceViewer()
        spans = viewer.from_table_cell(BadDataSource(), 0, "name")

        assert spans == []

    def test_negative_row_index_raises(self):
        """Test negative row index raises assertion."""

        class MockDataSource:
            rows = []

        viewer = ProvenanceViewer()
        with pytest.raises(AssertionError, match="row_index cannot be negative"):
            viewer.from_table_cell(MockDataSource(), -1, "name")
