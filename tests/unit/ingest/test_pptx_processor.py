"""
Tests for PPTX (PowerPoint) processor.

This module tests extraction of text, speaker notes, and tables from
PowerPoint presentations.

Test Strategy
-------------
- Focus on public API (process_pptx, can_process)
- Mock python-pptx library to avoid file dependencies
- Test slide extraction, notes, tables
- Verify metadata extraction from presentation properties

Organization
------------
- TestCanProcess: File type detection
- TestSlideContent: SlideContent dataclass
- TestPptxContent: PptxContent dataclass and properties
- TestBasicExtraction: Main extraction flow
- TestTitleExtraction: Slide title extraction
- TestBodyTextExtraction: Body text from shapes
- TestNotesExtraction: Speaker notes extraction
- TestTableExtraction: Table data extraction
- TestMetadataExtraction: Core properties extraction
- TestPptxProcessor: Class-based wrapper
- TestErrorHandling: Missing library and invalid files
"""

from unittest.mock import MagicMock, patch
import pytest

from ingestforge.ingest.pptx_processor import (
    SlideContent,
    PptxContent,
    can_process,
    process_pptx,
    PptxProcessor,
    _extract_title,
    _extract_body_text,
    _extract_notes,
    _extract_tables,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestCanProcess:
    """Tests for file type detection.

    Rule #4: Focused test class - tests only can_process
    """

    def test_can_process_pptx_file(self, temp_dir):
        """Test can_process returns True for .pptx files."""
        pptx_file = temp_dir / "presentation.pptx"
        pptx_file.touch()

        assert can_process(pptx_file) is True

    def test_can_process_uppercase_extension(self, temp_dir):
        """Test can_process handles uppercase extension."""
        pptx_file = temp_dir / "presentation.PPTX"
        pptx_file.touch()

        assert can_process(pptx_file) is True

    def test_cannot_process_pdf(self, temp_dir):
        """Test can_process returns False for non-PPTX files."""
        pdf_file = temp_dir / "document.pdf"
        pdf_file.touch()

        assert can_process(pdf_file) is False

    def test_cannot_process_txt(self, temp_dir):
        """Test can_process returns False for text files."""
        txt_file = temp_dir / "notes.txt"
        txt_file.touch()

        assert can_process(txt_file) is False


class TestSlideContent:
    """Tests for SlideContent dataclass.

    Rule #4: Focused test class - tests only SlideContent
    """

    def test_create_slide_content(self):
        """Test creating a SlideContent."""
        slide = SlideContent(
            slide_number=1,
            title="Introduction",
            body_text="Welcome to the presentation",
            speaker_notes="Remember to smile",
        )

        assert slide.slide_number == 1
        assert slide.title == "Introduction"
        assert slide.body_text == "Welcome to the presentation"
        assert slide.speaker_notes == "Remember to smile"
        assert slide.table_text == ""

    def test_slide_content_with_table(self):
        """Test SlideContent with table text."""
        slide = SlideContent(
            slide_number=2,
            title="Data",
            body_text="See table below",
            speaker_notes="",
            table_text="Header1 | Header2\nData1 | Data2",
        )

        assert slide.table_text == "Header1 | Header2\nData1 | Data2"

    def test_slide_word_count(self):
        """Test word count calculation."""
        slide = SlideContent(
            slide_number=1,
            title="Title",
            body_text="This is body text",
            speaker_notes="These are notes",
        )

        # "Title" + "This is body text" (4 words) + "These are notes" (3 words) = 8
        assert slide.word_count == 8

    def test_slide_word_count_empty(self):
        """Test word count with empty slide."""
        slide = SlideContent(
            slide_number=1,
            title="",
            body_text="",
            speaker_notes="",
        )

        assert slide.word_count == 0


class TestPptxContent:
    """Tests for PptxContent dataclass.

    Rule #4: Focused test class - tests only PptxContent
    """

    def test_create_pptx_content(self):
        """Test creating a PptxContent."""
        slides = [
            SlideContent(1, "Slide 1", "Content 1", "Notes 1"),
            SlideContent(2, "Slide 2", "Content 2", "Notes 2"),
        ]

        pptx = PptxContent(
            title="My Presentation",
            slides=slides,
            total_slides=2,
        )

        assert pptx.title == "My Presentation"
        assert len(pptx.slides) == 2
        assert pptx.total_slides == 2
        assert pptx.metadata == {}

    def test_pptx_word_count(self):
        """Test total word count calculation."""
        slides = [
            SlideContent(1, "Title One", "Body one two", ""),
            SlideContent(2, "Title Two", "Body three four five", ""),
        ]

        pptx = PptxContent(
            title="Test",
            slides=slides,
            total_slides=2,
        )

        # Slide 1: 5 words, Slide 2: 6 words = 11 total
        assert pptx.word_count == 11

    def test_full_text_with_titles(self):
        """Test full_text concatenation with titles."""
        slides = [
            SlideContent(1, "Introduction", "Welcome", ""),
        ]

        pptx = PptxContent(
            title="Test",
            slides=slides,
            total_slides=1,
        )

        full_text = pptx.full_text

        assert "## Slide 1: Introduction" in full_text
        assert "Welcome" in full_text

    def test_full_text_without_titles(self):
        """Test full_text with slides without titles."""
        slides = [
            SlideContent(1, "", "Content without title", ""),
        ]

        pptx = PptxContent(
            title="Test",
            slides=slides,
            total_slides=1,
        )

        full_text = pptx.full_text

        assert "## Slide 1" in full_text
        assert "Content without title" in full_text

    def test_full_text_with_notes(self):
        """Test full_text includes speaker notes."""
        slides = [
            SlideContent(1, "Slide", "Body", "Important notes"),
        ]

        pptx = PptxContent(
            title="Test",
            slides=slides,
            total_slides=1,
        )

        full_text = pptx.full_text

        assert "[Notes: Important notes]" in full_text

    def test_full_text_with_tables(self):
        """Test full_text includes tables."""
        slides = [
            SlideContent(1, "Data", "Body", "", "Col1 | Col2"),
        ]

        pptx = PptxContent(
            title="Test",
            slides=slides,
            total_slides=1,
        )

        full_text = pptx.full_text

        assert "Col1 | Col2" in full_text


class TestBasicExtraction:
    """Tests for main extraction flow.

    Rule #4: Focused test class - tests process_pptx
    """

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_process_pptx_basic(self, mock_presentation_class, temp_dir):
        """Test basic PPTX processing."""
        # Create mock presentation
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs

        # Mock slides
        mock_slide = MagicMock()
        mock_slide.shapes.title = MagicMock()
        mock_slide.shapes.title.text = "Test Slide"
        mock_slide.shapes = [mock_slide.shapes.title]
        mock_slide.has_notes_slide = False

        mock_prs.slides = [mock_slide]
        mock_prs.core_properties = None

        # Create test file
        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        result = process_pptx(pptx_file)

        assert isinstance(result, PptxContent)
        assert result.total_slides == 1
        assert len(result.slides) == 1

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_process_multiple_slides(self, mock_presentation_class, temp_dir):
        """Test processing multiple slides."""
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs

        # Create 3 mock slides
        mock_slides = []
        for i in range(3):
            mock_slide = MagicMock()
            mock_slide.shapes.title = None
            mock_slide.shapes = []
            mock_slide.has_notes_slide = False
            mock_slides.append(mock_slide)

        mock_prs.slides = mock_slides
        mock_prs.core_properties = None

        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        result = process_pptx(pptx_file)

        assert result.total_slides == 3
        assert len(result.slides) == 3

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_presentation_title_from_first_slide(
        self, mock_presentation_class, temp_dir
    ):
        """Test presentation title extracted from first slide."""
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs

        mock_slide = MagicMock()
        mock_slide.shapes.title = MagicMock()
        mock_slide.shapes.title.text = "Main Title"
        mock_slide.shapes = [mock_slide.shapes.title]
        mock_slide.has_notes_slide = False

        mock_prs.slides = [mock_slide]
        mock_prs.core_properties = None

        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        result = process_pptx(pptx_file)

        assert result.title == "Main Title"

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_presentation_title_fallback_to_filename(
        self, mock_presentation_class, temp_dir
    ):
        """Test presentation title falls back to filename."""
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs

        mock_slide = MagicMock()
        mock_slide.shapes.title = None
        mock_slide.shapes = []
        mock_slide.has_notes_slide = False

        mock_prs.slides = [mock_slide]
        mock_prs.core_properties = None

        pptx_file = temp_dir / "my_presentation.pptx"
        pptx_file.touch()

        result = process_pptx(pptx_file)

        assert result.title == "my_presentation"


class TestTitleExtraction:
    """Tests for slide title extraction.

    Rule #4: Focused test class - tests _extract_title
    """

    def test_extract_title_present(self):
        """Test extracting title when present."""
        mock_slide = MagicMock()
        mock_slide.shapes.title = MagicMock()
        mock_slide.shapes.title.text = "Slide Title"

        result = _extract_title(mock_slide)

        assert result == "Slide Title"

    def test_extract_title_with_whitespace(self):
        """Test title extraction strips whitespace."""
        mock_slide = MagicMock()
        mock_slide.shapes.title = MagicMock()
        mock_slide.shapes.title.text = "  Title with spaces  "

        result = _extract_title(mock_slide)

        assert result == "Title with spaces"

    def test_extract_title_none(self):
        """Test extracting title when None."""
        mock_slide = MagicMock()
        mock_slide.shapes.title = None

        result = _extract_title(mock_slide)

        assert result == ""


class TestBodyTextExtraction:
    """Tests for body text extraction.

    Rule #4: Focused test class - tests _extract_body_text
    """

    def test_extract_body_text_basic(self):
        """Test extracting body text from shapes."""
        mock_slide = MagicMock()
        mock_slide.shapes.title = None

        # Create text shape
        mock_shape = MagicMock()
        mock_shape.has_text_frame = True
        mock_para = MagicMock()
        mock_para.text = "Body paragraph"
        mock_shape.text_frame.paragraphs = [mock_para]

        mock_slide.shapes = [mock_shape]

        result = _extract_body_text(mock_slide)

        assert "Body paragraph" in result

    def test_extract_body_text_multiple_shapes(self):
        """Test extracting from multiple shapes."""
        mock_slide = MagicMock()
        mock_slide.shapes.title = None

        # Create two text shapes
        shapes = []
        for text in ["First shape", "Second shape"]:
            mock_shape = MagicMock()
            mock_shape.has_text_frame = True
            mock_para = MagicMock()
            mock_para.text = text
            mock_shape.text_frame.paragraphs = [mock_para]
            shapes.append(mock_shape)

        mock_slide.shapes = shapes

        result = _extract_body_text(mock_slide)

        assert "First shape" in result
        assert "Second shape" in result

    def test_extract_body_text_skips_title(self):
        """Test body extraction skips title shape."""
        mock_slide = MagicMock()

        # Create title shape
        mock_title = MagicMock()
        mock_title.has_text_frame = True
        mock_para = MagicMock()
        mock_para.text = "Title text"
        mock_title.text_frame.paragraphs = [mock_para]

        mock_slide.shapes.title = mock_title
        mock_slide.shapes = [mock_title]

        result = _extract_body_text(mock_slide)

        # Title should not be in body text
        assert result == ""

    def test_extract_body_text_skips_empty_paragraphs(self):
        """Test extraction skips empty paragraphs."""
        mock_slide = MagicMock()
        mock_slide.shapes.title = None

        mock_shape = MagicMock()
        mock_shape.has_text_frame = True

        # Mix of empty and non-empty paragraphs
        para1 = MagicMock()
        para1.text = "Real content"
        para2 = MagicMock()
        para2.text = "   "  # Whitespace only
        para3 = MagicMock()
        para3.text = ""  # Empty

        mock_shape.text_frame.paragraphs = [para1, para2, para3]
        mock_slide.shapes = [mock_shape]

        result = _extract_body_text(mock_slide)

        assert "Real content" in result
        # Should only have one line
        assert result.count("\n") == 0


class TestNotesExtraction:
    """Tests for speaker notes extraction.

    Rule #4: Focused test class - tests _extract_notes
    """

    def test_extract_notes_present(self):
        """Test extracting speaker notes when present."""
        mock_slide = MagicMock()
        mock_slide.has_notes_slide = True
        mock_slide.notes_slide.notes_text_frame.text = "Important notes"

        result = _extract_notes(mock_slide)

        assert result == "Important notes"

    def test_extract_notes_with_whitespace(self):
        """Test notes extraction strips whitespace."""
        mock_slide = MagicMock()
        mock_slide.has_notes_slide = True
        mock_slide.notes_slide.notes_text_frame.text = "  Notes  "

        result = _extract_notes(mock_slide)

        assert result == "Notes"

    def test_extract_notes_absent(self):
        """Test extracting notes when not present."""
        mock_slide = MagicMock()
        mock_slide.has_notes_slide = False

        result = _extract_notes(mock_slide)

        assert result == ""


class TestTableExtraction:
    """Tests for table extraction.

    Rule #4: Focused test class - tests _extract_tables
    """

    def test_extract_table_basic(self):
        """Test extracting a simple table."""
        mock_slide = MagicMock()

        # Create table shape
        mock_shape = MagicMock()
        mock_shape.has_table = True

        # Create table with 2 rows
        mock_row1 = MagicMock()
        cell1_1 = MagicMock()
        cell1_1.text = "Header1"
        cell1_2 = MagicMock()
        cell1_2.text = "Header2"
        mock_row1.cells = [cell1_1, cell1_2]

        mock_row2 = MagicMock()
        cell2_1 = MagicMock()
        cell2_1.text = "Data1"
        cell2_2 = MagicMock()
        cell2_2.text = "Data2"
        mock_row2.cells = [cell2_1, cell2_2]

        mock_shape.table.rows = [mock_row1, mock_row2]
        mock_slide.shapes = [mock_shape]

        result = _extract_tables(mock_slide)

        assert "Header1 | Header2" in result
        assert "Data1 | Data2" in result

    def test_extract_multiple_tables(self):
        """Test extracting multiple tables."""
        mock_slide = MagicMock()

        # Create two table shapes
        tables = []
        for i in range(2):
            mock_shape = MagicMock()
            mock_shape.has_table = True

            mock_row = MagicMock()
            cell = MagicMock()
            cell.text = f"Table{i+1}"
            mock_row.cells = [cell]

            mock_shape.table.rows = [mock_row]
            tables.append(mock_shape)

        mock_slide.shapes = tables

        result = _extract_tables(mock_slide)

        assert "Table1" in result
        assert "Table2" in result

    def test_extract_table_strips_whitespace(self):
        """Test table extraction strips cell whitespace."""
        mock_slide = MagicMock()

        mock_shape = MagicMock()
        mock_shape.has_table = True

        mock_row = MagicMock()
        cell = MagicMock()
        cell.text = "  Data  "
        mock_row.cells = [cell]

        mock_shape.table.rows = [mock_row]
        mock_slide.shapes = [mock_shape]

        result = _extract_tables(mock_slide)

        assert "Data" in result
        assert "  Data  " not in result

    def test_extract_no_tables(self):
        """Test extraction when no tables present."""
        mock_slide = MagicMock()

        mock_shape = MagicMock()
        mock_shape.has_table = False

        mock_slide.shapes = [mock_shape]

        result = _extract_tables(mock_slide)

        assert result == ""


class TestMetadataExtraction:
    """Tests for metadata extraction.

    Rule #4: Focused test class - tests metadata extraction
    """

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_extract_metadata_with_properties(self, mock_presentation_class, temp_dir):
        """Test extracting metadata from core properties."""
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs

        # Mock core properties
        mock_props = MagicMock()
        mock_props.author = "John Doe"
        mock_props.title = "My Presentation"
        mock_props.created = "2024-01-01"
        mock_props.modified = "2024-01-02"

        mock_prs.core_properties = mock_props
        mock_prs.slides = []

        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        result = process_pptx(pptx_file)

        assert result.metadata["author"] == "John Doe"
        assert result.metadata["title"] == "My Presentation"
        assert result.metadata["created"] == "2024-01-01"
        assert result.metadata["modified"] == "2024-01-02"
        # Title from metadata overrides slide title
        assert result.title == "My Presentation"

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_extract_metadata_no_properties(self, mock_presentation_class, temp_dir):
        """Test extraction when core properties is None."""
        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs

        mock_prs.core_properties = None
        mock_prs.slides = []

        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        result = process_pptx(pptx_file)

        assert result.metadata == {}


class TestPptxProcessor:
    """Tests for class-based wrapper.

    Rule #4: Focused test class - tests PptxProcessor class
    """

    def test_processor_can_process(self, temp_dir):
        """Test PptxProcessor.can_process."""
        processor = PptxProcessor()

        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        assert processor.can_process(pptx_file) is True

    @patch("ingestforge.ingest.pptx_processor.Presentation")
    def test_processor_process(self, mock_presentation_class, temp_dir):
        """Test PptxProcessor.process."""
        processor = PptxProcessor()

        mock_prs = MagicMock()
        mock_presentation_class.return_value = mock_prs
        mock_prs.slides = []
        mock_prs.core_properties = None

        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        result = processor.process(pptx_file)

        assert isinstance(result, PptxContent)


class TestErrorHandling:
    """Tests for error handling.

    Rule #4: Focused test class - tests error conditions
    """

    @patch("ingestforge.ingest.pptx_processor.pptx", None)
    def test_missing_pptx_library(self, temp_dir):
        """Test error when python-pptx not installed."""
        pptx_file = temp_dir / "test.pptx"
        pptx_file.touch()

        with pytest.raises(ImportError) as exc_info:
            process_pptx(pptx_file)

        assert "python-pptx" in str(exc_info.value)


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - File detection: 4 tests (pptx, uppercase, pdf, txt)
    - SlideContent: 4 tests (creation, table, word count, empty)
    - PptxContent: 7 tests (creation, word count, full_text variations)
    - Basic extraction: 5 tests (basic, multiple, title variations)
    - Title extraction: 3 tests (present, whitespace, none)
    - Body text: 4 tests (basic, multiple, skip title, skip empty)
    - Notes: 3 tests (present, whitespace, absent)
    - Tables: 4 tests (basic, multiple, whitespace, none)
    - Metadata: 2 tests (with properties, without)
    - Processor class: 2 tests (can_process, process)
    - Error handling: 1 test (missing library)

    Total: 39 tests

Design Decisions:
    1. Mock python-pptx to avoid file dependencies
    2. Test public API and helper functions
    3. Cover all extraction paths (title, body, notes, tables)
    4. Verify metadata handling
    5. Test edge cases (empty, whitespace, None values)

Behaviors Tested:
    - File type detection via extension
    - SlideContent and PptxContent dataclasses
    - Word count calculations
    - Full text concatenation with formatting
    - Title, body, notes, and table extraction
    - Metadata extraction from core properties
    - Title fallback to filename
    - Whitespace trimming
    - Empty content handling
    - Class-based wrapper interface
    - ImportError for missing library
"""
