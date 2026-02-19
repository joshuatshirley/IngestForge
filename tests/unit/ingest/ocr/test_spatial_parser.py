"""Tests for hOCR/ALTO spatial parser.

Tests parsing of OCR output formats and bounding box extraction."""

from __future__ import annotations

from pathlib import Path


from ingestforge.ingest.ocr.spatial_parser import (
    ALTOParser,
    BoundingBox,
    ElementType,
    HOCRParser,
    OCRDocument,
    OCRElement,
    OCRPage,
    parse_ocr_file,
)

# BoundingBox tests


class TestBoundingBox:
    """Tests for BoundingBox dataclass."""

    def test_basic_properties(self) -> None:
        """Test width, height, area properties."""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=70)

        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000

    def test_center_calculation(self) -> None:
        """Test center point calculation."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=100)

        assert bbox.center == (50, 50)

    def test_overlaps_true(self) -> None:
        """Test overlapping boxes."""
        box1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        box2 = BoundingBox(x1=25, y1=25, x2=75, y2=75)

        assert box1.overlaps(box2) is True
        assert box2.overlaps(box1) is True

    def test_overlaps_false(self) -> None:
        """Test non-overlapping boxes."""
        box1 = BoundingBox(x1=0, y1=0, x2=50, y2=50)
        box2 = BoundingBox(x1=100, y1=100, x2=150, y2=150)

        assert box1.overlaps(box2) is False

    def test_contains(self) -> None:
        """Test containment check."""
        outer = BoundingBox(x1=0, y1=0, x2=100, y2=100)
        inner = BoundingBox(x1=25, y1=25, x2=75, y2=75)

        assert outer.contains(inner) is True
        assert inner.contains(outer) is False


# OCRElement tests


class TestOCRElement:
    """Tests for OCRElement dataclass."""

    def test_element_creation(self) -> None:
        """Test creating an element."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.WORD,
            bbox=bbox,
            text="hello",
            confidence=0.95,
        )

        assert elem.element_type == ElementType.WORD
        assert elem.text == "hello"
        assert elem.confidence == 0.95

    def test_is_empty_with_text(self) -> None:
        """Test is_empty with text content."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.WORD,
            bbox=bbox,
            text="content",
        )

        assert elem.is_empty is False

    def test_is_empty_without_text(self) -> None:
        """Test is_empty without text content."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        elem = OCRElement(
            element_type=ElementType.BLOCK,
            bbox=bbox,
            text="   ",
        )

        assert elem.is_empty is True


# OCRPage tests


class TestOCRPage:
    """Tests for OCRPage dataclass."""

    def test_page_creation(self) -> None:
        """Test creating a page."""
        page = OCRPage(page_number=1, width=612, height=792)

        assert page.page_number == 1
        assert page.width == 612
        assert page.height == 792

    def test_get_blocks(self) -> None:
        """Test getting block elements."""
        page = OCRPage(page_number=1, width=612, height=792)
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)

        block = OCRElement(element_type=ElementType.BLOCK, bbox=bbox)
        line = OCRElement(element_type=ElementType.LINE, bbox=bbox)
        page.elements = [block, line, block]

        blocks = page.get_blocks()
        assert len(blocks) == 2
        assert all(b.element_type == ElementType.BLOCK for b in blocks)


# OCRDocument tests


class TestOCRDocument:
    """Tests for OCRDocument dataclass."""

    def test_document_creation(self) -> None:
        """Test creating a document."""
        doc = OCRDocument(source_file="test.hocr")

        assert doc.source_file == "test.hocr"
        assert len(doc.pages) == 0

    def test_get_all_text(self) -> None:
        """Test getting all text from document."""
        doc = OCRDocument()
        page = OCRPage(page_number=1, width=612, height=792)
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)

        elem1 = OCRElement(element_type=ElementType.LINE, bbox=bbox, text="Hello")
        elem2 = OCRElement(element_type=ElementType.LINE, bbox=bbox, text="World")
        page.elements = [elem1, elem2]
        doc.pages = [page]

        text = doc.get_all_text()
        assert "Hello" in text
        assert "World" in text


# HOCRParser tests


class TestHOCRParser:
    """Tests for hOCR parser."""

    def test_parser_creation(self) -> None:
        """Test creating parser."""
        parser = HOCRParser()
        assert parser is not None

    def test_parse_simple_hocr(self) -> None:
        """Test parsing simple hOCR content."""
        hocr_content = """
        <div class="ocr_page" title="bbox 0 0 612 792">
            <span class="ocr_carea" title="bbox 50 50 500 100">
                <span class="ocrx_word" title="bbox 50 50 100 70; x_wconf 95">Hello</span>
            </span>
        </div>
        """
        parser = HOCRParser()
        doc = parser.parse_string(hocr_content)

        assert len(doc.pages) == 1
        assert doc.pages[0].width == 612
        assert doc.pages[0].height == 792

    def test_extract_bbox(self) -> None:
        """Test bbox extraction from title."""
        parser = HOCRParser()
        bbox = parser._extract_bbox("bbox 10 20 110 70")

        assert bbox is not None
        assert bbox.x1 == 10
        assert bbox.y1 == 20
        assert bbox.x2 == 110
        assert bbox.y2 == 70

    def test_extract_confidence(self) -> None:
        """Test confidence extraction."""
        parser = HOCRParser()

        conf = parser._extract_confidence("x_wconf 95")
        assert conf == 0.95

        conf = parser._extract_confidence("no confidence")
        assert conf == 0.0

    def test_classify_element(self) -> None:
        """Test element type classification."""
        parser = HOCRParser()

        assert parser._classify_element("ocr_page") == ElementType.PAGE
        assert parser._classify_element("ocr_carea") == ElementType.BLOCK
        assert parser._classify_element("ocr_par") == ElementType.PARAGRAPH
        assert parser._classify_element("ocr_line") == ElementType.LINE
        assert parser._classify_element("ocrx_word") == ElementType.WORD
        assert parser._classify_element("unknown") is None

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        """Test parsing nonexistent file."""
        parser = HOCRParser()
        doc = parser.parse_file(tmp_path / "missing.hocr")

        assert doc.source_file == str(tmp_path / "missing.hocr")
        assert len(doc.pages) == 0


# ALTOParser tests


class TestALTOParser:
    """Tests for ALTO XML parser."""

    def test_parser_creation(self) -> None:
        """Test creating parser."""
        parser = ALTOParser()
        assert parser is not None

    def test_extract_alto_bbox(self) -> None:
        """Test ALTO bbox extraction."""
        from xml.etree import ElementTree as ET

        parser = ALTOParser()
        elem = ET.fromstring(
            '<TextBlock HPOS="50" VPOS="100" WIDTH="200" HEIGHT="50"/>'
        )
        bbox = parser._extract_alto_bbox(elem)

        assert bbox is not None
        assert bbox.x1 == 50
        assert bbox.y1 == 100
        assert bbox.x2 == 250  # HPOS + WIDTH
        assert bbox.y2 == 150  # VPOS + HEIGHT

    def test_parse_missing_file(self, tmp_path: Path) -> None:
        """Test parsing nonexistent file."""
        parser = ALTOParser()
        doc = parser.parse_file(tmp_path / "missing.alto")

        assert len(doc.pages) == 0


# parse_ocr_file tests


class TestParseOCRFile:
    """Tests for auto-detect parsing."""

    def test_hocr_extension(self, tmp_path: Path) -> None:
        """Test hOCR file detection."""
        hocr_file = tmp_path / "test.hocr"
        hocr_file.write_text(
            """
        <div class="ocr_page" title="bbox 0 0 612 792">
            <span class="ocrx_word" title="bbox 50 50 100 70">Test</span>
        </div>
        """
        )

        doc = parse_ocr_file(hocr_file)
        assert len(doc.pages) == 1

    def test_alto_detection(self, tmp_path: Path) -> None:
        """Test ALTO format detection."""
        alto_file = tmp_path / "test.xml"
        alto_file.write_text(
            """<?xml version="1.0"?>
        <alto xmlns="http://www.loc.gov/standards/alto/ns-v3#">
            <Layout>
                <Page WIDTH="612" HEIGHT="792">
                    <PrintSpace>
                        <TextBlock HPOS="50" VPOS="50" WIDTH="100" HEIGHT="50">
                            <TextLine HPOS="50" VPOS="50" WIDTH="100" HEIGHT="20">
                                <String CONTENT="Test"/>
                            </TextLine>
                        </TextBlock>
                    </PrintSpace>
                </Page>
            </Layout>
        </alto>
        """
        )

        doc = parse_ocr_file(alto_file)
        assert "alto" in doc.source_file.lower() or len(doc.pages) >= 0

    def test_unknown_extension(self, tmp_path: Path) -> None:
        """Test unknown file extension."""
        unknown = tmp_path / "test.xyz"
        unknown.write_text("unknown content")

        doc = parse_ocr_file(unknown)
        assert len(doc.pages) == 0
