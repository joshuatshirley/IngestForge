"""Tests for multi-column reading order sequencer.

Tests column detection and element ordering."""

from __future__ import annotations


from ingestforge.ingest.ocr.spatial_parser import (
    BoundingBox,
    ElementType,
    OCRDocument,
    OCRElement,
    OCRPage,
)
from ingestforge.ingest.ocr.sequencer import (
    Column,
    LayoutType,
    MultiColumnSequencer,
    ReadingOrder,
    SequencerConfig,
    detect_column_layout,
    sequence_ocr_document,
)

# Column tests


class TestColumn:
    """Tests for Column dataclass."""

    def test_column_properties(self) -> None:
        """Test column width and center."""
        col = Column(x_start=100, x_end=300)

        assert col.width == 200
        assert col.center_x == 200


# ReadingOrder tests


class TestReadingOrder:
    """Tests for ReadingOrder dataclass."""

    def test_reading_order_defaults(self) -> None:
        """Test default values."""
        order = ReadingOrder()

        assert len(order.elements) == 0
        assert order.layout_type == LayoutType.SINGLE_COLUMN
        assert order.column_count == 1

    def test_get_text(self) -> None:
        """Test text extraction."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem1 = OCRElement(element_type=ElementType.LINE, bbox=bbox, text="Hello")
        elem2 = OCRElement(element_type=ElementType.LINE, bbox=bbox, text="World")

        order = ReadingOrder(elements=[elem1, elem2])
        text = order.get_text()

        assert text == "Hello\nWorld"

    def test_get_text_custom_separator(self) -> None:
        """Test text with custom separator."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem1 = OCRElement(element_type=ElementType.LINE, bbox=bbox, text="A")
        elem2 = OCRElement(element_type=ElementType.LINE, bbox=bbox, text="B")

        order = ReadingOrder(elements=[elem1, elem2])
        text = order.get_text(separator=" | ")

        assert text == "A | B"


# MultiColumnSequencer tests


class TestMultiColumnSequencer:
    """Tests for MultiColumnSequencer."""

    def test_sequencer_creation(self) -> None:
        """Test creating sequencer."""
        sequencer = MultiColumnSequencer()
        assert sequencer.config is not None

    def test_sequencer_with_config(self) -> None:
        """Test sequencer with custom config."""
        config = SequencerConfig(column_gap_threshold=100)
        sequencer = MultiColumnSequencer(config=config)

        assert sequencer.config.column_gap_threshold == 100

    def test_sequence_empty_page(self) -> None:
        """Test sequencing empty page."""
        page = OCRPage(page_number=1, width=612, height=792)
        sequencer = MultiColumnSequencer()

        order = sequencer.sequence_page(page)

        assert len(order.elements) == 0
        assert order.layout_type == LayoutType.SINGLE_COLUMN

    def test_sequence_single_column(self) -> None:
        """Test single column layout detection."""
        page = OCRPage(page_number=1, width=612, height=792)

        # Create vertically stacked elements
        for i in range(3):
            bbox = BoundingBox(x1=50, y1=100 + i * 50, x2=500, y2=140 + i * 50)
            elem = OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text=f"Line {i}",
            )
            page.elements.append(elem)

        sequencer = MultiColumnSequencer()
        order = sequencer.sequence_page(page)

        assert order.layout_type == LayoutType.SINGLE_COLUMN
        assert order.column_count == 1

    def test_sequence_two_column(self) -> None:
        """Test two column layout detection."""
        page = OCRPage(page_number=1, width=612, height=792)

        # Left column elements
        for i in range(3):
            bbox = BoundingBox(x1=50, y1=100 + i * 50, x2=250, y2=140 + i * 50)
            elem = OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text=f"Left {i}",
            )
            page.elements.append(elem)

        # Right column elements (with clear gap)
        for i in range(3):
            bbox = BoundingBox(x1=350, y1=100 + i * 50, x2=550, y2=140 + i * 50)
            elem = OCRElement(
                element_type=ElementType.BLOCK,
                bbox=bbox,
                text=f"Right {i}",
            )
            page.elements.append(elem)

        sequencer = MultiColumnSequencer()
        order = sequencer.sequence_page(page)

        assert order.layout_type == LayoutType.TWO_COLUMN
        assert order.column_count == 2

    def test_reading_order_top_to_bottom(self) -> None:
        """Test elements sorted top to bottom within column."""
        page = OCRPage(page_number=1, width=612, height=792)

        # Add elements out of order
        bbox3 = BoundingBox(x1=50, y1=200, x2=200, y2=220)
        bbox1 = BoundingBox(x1=50, y1=100, x2=200, y2=120)
        bbox2 = BoundingBox(x1=50, y1=150, x2=200, y2=170)

        page.elements = [
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox3, text="Third"),
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox1, text="First"),
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox2, text="Second"),
        ]

        sequencer = MultiColumnSequencer()
        order = sequencer.sequence_page(page)

        texts = [e.text for e in order.elements]
        assert texts == ["First", "Second", "Third"]


class TestLayoutClassification:
    """Tests for layout type classification."""

    def test_classify_single_column(self) -> None:
        """Test single column classification."""
        sequencer = MultiColumnSequencer()

        assert sequencer._classify_layout(0) == LayoutType.SINGLE_COLUMN
        assert sequencer._classify_layout(1) == LayoutType.SINGLE_COLUMN

    def test_classify_two_column(self) -> None:
        """Test two column classification."""
        sequencer = MultiColumnSequencer()
        assert sequencer._classify_layout(2) == LayoutType.TWO_COLUMN

    def test_classify_three_column(self) -> None:
        """Test three column classification."""
        sequencer = MultiColumnSequencer()
        assert sequencer._classify_layout(3) == LayoutType.THREE_COLUMN

    def test_classify_multi_column(self) -> None:
        """Test multi column classification."""
        sequencer = MultiColumnSequencer()
        assert sequencer._classify_layout(4) == LayoutType.MULTI_COLUMN
        assert sequencer._classify_layout(5) == LayoutType.MULTI_COLUMN


class TestColumnDetection:
    """Tests for column boundary detection."""

    def test_find_best_column_exact_match(self) -> None:
        """Test finding column by exact x position."""
        sequencer = MultiColumnSequencer()
        columns = [
            Column(x_start=0, x_end=200),
            Column(x_start=250, x_end=450),
        ]

        result = sequencer._find_best_column(100, columns)
        assert result == columns[0]

        result = sequencer._find_best_column(350, columns)
        assert result == columns[1]

    def test_find_best_column_closest(self) -> None:
        """Test finding closest column when outside."""
        sequencer = MultiColumnSequencer()
        columns = [
            Column(x_start=100, x_end=200),
            Column(x_start=300, x_end=400),
        ]

        # Left of all columns - should find leftmost
        result = sequencer._find_best_column(50, columns)
        assert result == columns[0]

        # Right of all columns - should find rightmost
        result = sequencer._find_best_column(500, columns)
        assert result == columns[1]


# Convenience function tests


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_sequence_ocr_document(self) -> None:
        """Test document sequencing function."""
        doc = OCRDocument()
        page1 = OCRPage(page_number=1, width=612, height=792)
        page2 = OCRPage(page_number=2, width=612, height=792)

        bbox = BoundingBox(x1=50, y1=100, x2=200, y2=150)
        page1.elements.append(
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="Page 1")
        )
        page2.elements.append(
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="Page 2")
        )

        doc.pages = [page1, page2]

        orders = sequence_ocr_document(doc)

        assert len(orders) == 2
        assert orders[0].elements[0].text == "Page 1"
        assert orders[1].elements[0].text == "Page 2"

    def test_detect_column_layout(self) -> None:
        """Test column layout detection function."""
        page = OCRPage(page_number=1, width=612, height=792)

        bbox = BoundingBox(x1=50, y1=100, x2=500, y2=150)
        page.elements.append(
            OCRElement(element_type=ElementType.BLOCK, bbox=bbox, text="Text")
        )

        layout_type, col_count = detect_column_layout(page)

        assert layout_type == LayoutType.SINGLE_COLUMN
        assert col_count == 1
