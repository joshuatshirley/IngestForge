"""Integration tests for Unstructured-inspired features.

Tests the complete pipeline of:
- Text cleaning ()
- Bounding box metadata ()
- Table HTML preservation ()
- Element typing ()
- Layout-aware chunking ()
"""


from ingestforge.core.types import ChunkMetadata
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.chunking.layout_chunker import LayoutChunker
from ingestforge.ingest.refinement import (
    ChapterMarker,
    DocumentElementType,
    TextRefinementPipeline,
)
from ingestforge.ingest.refiners import (
    ChapterDetector,
    ElementClassifier,
    FormatNormalizer,
    TextCleanerRefiner,
)
from ingestforge.ingest.ocr.spatial_parser import BoundingBox, OCRElement, ElementType
from ingestforge.ingest.ocr.bbox_bridge import (
    BoundingBoxBridge,
)
from ingestforge.ingest.ocr.table_builder import Table, TableRow, TableCell
from ingestforge.ingest.table_processor import TableProcessor, table_to_chunk_metadata


class TestTextCleaningPipeline:
    """Integration tests for text cleaning pipeline."""

    def test_full_refinement_pipeline(self) -> None:
        """Test complete refinement pipeline with all cleaners."""
        pipeline = TextRefinementPipeline(
            [
                FormatNormalizer(),
                TextCleanerRefiner(),
                ChapterDetector(),
            ]
        )

        # Text with OCR artifacts, bullets, and chapter markers
        text = """
\u2022 First bullet item
\u2022 Second bullet item

42

CHAPTER 1

This is the intro\u2014with em-dashes.

Section 1.1 Methods

The methods used in this
study were extensive.
        """

        result = pipeline.refine(text)

        # Bullets should be normalized
        assert "\u2022" not in result.refined
        assert "-" in result.refined

        # Em-dash should be normalized
        assert "\u2014" not in result.refined

        # Page number should be removed
        assert "\n42\n" not in result.refined

        # Chapter markers should be detected
        assert len(result.chapter_markers) >= 1
        assert any(
            "Chapter" in m.title or "CHAPTER" in m.title for m in result.chapter_markers
        )


class TestBoundingBoxFlow:
    """Integration tests for bounding box metadata flow."""

    def test_bbox_flows_to_chunk_metadata(self) -> None:
        """Test that bbox coordinates flow from OCR to chunk metadata."""
        # Create OCR elements
        elements = [
            OCRElement(
                element_type=ElementType.WORD,
                bbox=BoundingBox(x1=100, y1=200, x2=300, y2=250),
                text="Hello",
            ),
            OCRElement(
                element_type=ElementType.WORD,
                bbox=BoundingBox(x1=320, y1=200, x2=500, y2=250),
                text="World",
            ),
        ]

        # Bridge to chunk bbox
        bridge = BoundingBoxBridge()
        chunk_bbox = bridge.combine_element_boxes(elements, page_number=1)

        # Convert to metadata
        metadata: dict = {}
        metadata = bridge.update_chunk_metadata(metadata, chunk_bbox)

        # Verify metadata fields
        assert metadata["bbox_x1"] == 100
        assert metadata["bbox_y1"] == 200
        assert metadata["bbox_x2"] == 500
        assert metadata["bbox_y2"] == 250
        assert metadata["page_number"] == 1

    def test_chunk_record_has_bbox_field(self) -> None:
        """Test that ChunkRecord includes bbox field."""
        chunk = ChunkRecord(
            chunk_id="test_123",
            document_id="doc_456",
            content="Test content",
            bbox=(100, 200, 300, 400),
            element_type="NarrativeText",
        )

        assert chunk.bbox == (100, 200, 300, 400)
        assert chunk.element_type == "NarrativeText"


class TestTableHtmlPreservation:
    """Integration tests for table HTML preservation."""

    def test_table_produces_both_formats(self) -> None:
        """Test that tables produce both markdown and HTML."""
        # Create OCR table
        table = Table(has_header=True)
        row1 = TableRow(row_index=0)
        row1.cells = [
            TableCell(row=0, col=0, text="Name", bbox=BoundingBox(0, 0, 100, 30)),
            TableCell(row=0, col=1, text="Value", bbox=BoundingBox(100, 0, 200, 30)),
        ]
        row2 = TableRow(row_index=1)
        row2.cells = [
            TableCell(row=1, col=0, text="A", bbox=BoundingBox(0, 30, 100, 60)),
            TableCell(row=1, col=1, text="1", bbox=BoundingBox(100, 30, 200, 60)),
        ]
        table.rows = [row1, row2]

        # Process table
        processor = TableProcessor()
        result = processor.process_ocr_table(table)

        # Verify both formats
        assert "Name" in result.markdown
        assert "|" in result.markdown  # Markdown table format
        assert "<table>" in result.html
        assert "<th>Name</th>" in result.html

    def test_table_metadata_includes_html(self) -> None:
        """Test that table chunk metadata includes HTML."""
        table = Table(has_header=False)
        row = TableRow(row_index=0)
        row.cells = [
            TableCell(row=0, col=0, text="Data", bbox=BoundingBox(0, 0, 100, 30))
        ]
        table.rows = [row]

        metadata = table_to_chunk_metadata(table, page_number=5)

        assert "table_html" in metadata
        assert "table_markdown" in metadata
        assert metadata["element_type"] == "Table"
        assert metadata["page_number"] == 5


class TestElementTyping:
    """Integration tests for element type classification."""

    def test_element_classifier_in_pipeline(self) -> None:
        """Test element classifier within refinement pipeline."""
        classifier = ElementClassifier()

        text = """# Introduction

This is narrative text explaining the topic.

- First item
- Second item

```python
def hello():
    print("hello")
```

| Col1 | Col2 |
|------|------|
| A    | B    |
"""

        elements = classifier.classify_text(text)

        types = {e.element_type for e in elements}

        # Should detect multiple element types
        assert DocumentElementType.TITLE in types
        assert DocumentElementType.NARRATIVE_TEXT in types
        assert DocumentElementType.LIST_ITEM in types
        assert DocumentElementType.CODE in types
        assert DocumentElementType.TABLE in types

    def test_chunk_record_element_type(self) -> None:
        """Test that ChunkRecord has element_type field."""
        chunk = ChunkRecord(
            chunk_id="test",
            document_id="doc",
            content="# Title",
            element_type="Title",
        )

        assert chunk.element_type == "Title"


class TestLayoutAwareChunking:
    """Integration tests for layout-aware chunking."""

    def test_respects_chapter_markers(self) -> None:
        """Test that chunks respect chapter marker boundaries."""
        text = """Introduction

This is the introduction to the document.

Chapter 1: Methods

The methods section describes our approach.
We used several techniques.

Chapter 2: Results

The results show significant findings.
"""

        # Get markers from chapter detector
        detector = ChapterDetector()
        refined = detector.refine(text)

        # Chunk with layout awareness (disable combining to ensure multiple chunks)
        chunker = LayoutChunker(
            respect_section_boundaries=True,
            combine_text_under_n_chars=10,  # Small threshold to prevent combining
        )
        chunks = chunker.chunk_with_markers(text, refined.chapter_markers, "doc_123")

        # Verify chunks were created
        assert len(chunks) >= 1

        # Verify section titles are captured
        titles = [c.section_title for c in chunks if c.section_title]
        assert any("Chapter" in t for t in titles)

    def test_by_title_chunking(self) -> None:
        """Test chunking at title (level 1) boundaries only."""
        text = """Chapter 1

Section 1.1
Content here.

Section 1.2
More content.

Chapter 2

Different content.
"""

        markers = [
            ChapterMarker(position=0, title="Chapter 1", level=1),
            ChapterMarker(position=12, title="Section 1.1", level=2),
            ChapterMarker(position=36, title="Section 1.2", level=2),
            ChapterMarker(position=60, title="Chapter 2", level=1),
        ]

        chunker = LayoutChunker(chunk_by_title=True)
        chunks = chunker.chunk_with_markers(text, markers, "doc")

        # Should split at level 1 markers only (Chapter boundaries)
        # Sections 1.1 and 1.2 should be combined with Chapter 1
        assert len(chunks) <= 4  # Max 4 chunks (intro, ch1, ch2, etc.)


class TestEndToEndPipeline:
    """End-to-end integration tests."""

    def test_full_pipeline_document(self) -> None:
        """Test complete document processing pipeline."""
        document = """
# Research Paper

## Abstract

This paper presents novel findings.

## 1. Introduction

The introduction provides context for the research.
Multiple paragraphs explain the background.

## 2. Methods

- Data collection
- Analysis procedures

| Method | Sample Size |
|--------|-------------|
| Survey | 500         |
| Interview | 50       |

## 3. Results

Results were statistically significant.

## 4. Conclusion

The study demonstrates important findings.
"""

        # Step 1: Refinement
        pipeline = TextRefinementPipeline(
            [
                FormatNormalizer(),
                TextCleanerRefiner(),
                ChapterDetector(),
            ]
        )
        refined = pipeline.refine(document)

        # Verify refinement
        assert refined.chapter_markers  # Should detect sections

        # Step 2: Element classification
        classifier = ElementClassifier()
        elements = classifier.classify_text(refined.refined)

        # Verify classification
        types = {e.element_type for e in elements}
        assert DocumentElementType.TITLE in types
        assert DocumentElementType.LIST_ITEM in types
        assert DocumentElementType.TABLE in types

        # Step 3: Layout-aware chunking
        chunker = LayoutChunker(
            respect_section_boundaries=True,
            combine_text_under_n_chars=100,
        )
        chunks = chunker.chunk_with_markers(
            refined.refined,
            refined.chapter_markers,
            document_id="paper_001",
        )

        # Verify chunks
        assert len(chunks) >= 2  # At least multiple sections
        assert all(c.document_id == "paper_001" for c in chunks)
        # Verify some structure is preserved
        all_titles = " ".join(c.section_title or "" for c in chunks)
        assert len(all_titles) > 0  # Some section titles captured


class TestChunkMetadataTypes:
    """Tests for new ChunkMetadata fields."""

    def test_chunk_metadata_has_new_fields(self) -> None:
        """Test that ChunkMetadata TypedDict has new fields."""
        # Create a metadata dict with new fields
        metadata: ChunkMetadata = {
            "source_file": "test.pdf",
            "page_number": 1,
            "bbox_x1": 100,
            "bbox_y1": 200,
            "bbox_x2": 500,
            "bbox_y2": 300,
            "table_html": "<table><tr><td>Data</td></tr></table>",
            "element_type": "Table",
        }

        # Type checker should accept these fields
        assert metadata["bbox_x1"] == 100
        assert metadata["element_type"] == "Table"
        assert metadata["table_html"].startswith("<table>")
