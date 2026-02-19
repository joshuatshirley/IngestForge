"""Unit tests for BoundingBoxBridge."""


from ingestforge.ingest.ocr.spatial_parser import (
    BoundingBox,
    OCRElement,
    OCRPage,
    ElementType,
)
from ingestforge.ingest.ocr.bbox_bridge import (
    BoundingBoxBridge,
    ChunkBoundingBox,
    bbox_to_metadata,
    extract_bbox_from_elements,
)


class TestChunkBoundingBox:
    """Tests for ChunkBoundingBox dataclass."""

    def test_to_tuple(self) -> None:
        """Test converting to tuple."""
        bbox = ChunkBoundingBox(x1=10, y1=20, x2=100, y2=200, page_number=1)
        assert bbox.to_tuple() == (10, 20, 100, 200)

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        bbox = ChunkBoundingBox(x1=10, y1=20, x2=100, y2=200, page_number=3)
        result = bbox.to_dict()
        assert result["bbox_x1"] == 10
        assert result["bbox_y1"] == 20
        assert result["bbox_x2"] == 100
        assert result["bbox_y2"] == 200
        assert result["page_number"] == 3

    def test_from_ocr_bbox(self) -> None:
        """Test creating from OCR BoundingBox."""
        ocr_bbox = BoundingBox(x1=50, y1=60, x2=150, y2=160)
        chunk_bbox = ChunkBoundingBox.from_ocr_bbox(ocr_bbox, page_number=2)

        assert chunk_bbox.x1 == 50
        assert chunk_bbox.y1 == 60
        assert chunk_bbox.x2 == 150
        assert chunk_bbox.y2 == 160
        assert chunk_bbox.page_number == 2

    def test_width_and_height(self) -> None:
        """Test width and height properties."""
        bbox = ChunkBoundingBox(x1=10, y1=20, x2=110, y2=120, page_number=1)
        assert bbox.width == 100
        assert bbox.height == 100

    def test_center(self) -> None:
        """Test center point calculation."""
        bbox = ChunkBoundingBox(x1=0, y1=0, x2=100, y2=200, page_number=1)
        assert bbox.center == (50, 100)

    def test_contains_point(self) -> None:
        """Test point containment check."""
        bbox = ChunkBoundingBox(x1=10, y1=10, x2=100, y2=100, page_number=1)
        assert bbox.contains_point(50, 50) is True
        assert bbox.contains_point(10, 10) is True  # Edge
        assert bbox.contains_point(0, 0) is False
        assert bbox.contains_point(150, 150) is False


class TestBoundingBoxBridge:
    """Tests for BoundingBoxBridge class."""

    def test_combine_empty_elements(self) -> None:
        """Test combining empty element list."""
        bridge = BoundingBoxBridge()
        result = bridge.combine_element_boxes([])
        assert result is None

    def test_combine_elements_without_bbox(self) -> None:
        """Test combining elements that lack bounding boxes."""
        bridge = BoundingBoxBridge()
        elements = [
            OCRElement(element_type=ElementType.WORD, bbox=None, text="test"),
        ]
        result = bridge.combine_element_boxes(elements)
        assert result is None

    def test_combine_single_element(self) -> None:
        """Test combining a single element."""
        bridge = BoundingBoxBridge()
        elements = [
            OCRElement(
                element_type=ElementType.WORD,
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
                text="test",
            ),
        ]
        result = bridge.combine_element_boxes(elements, page_number=3)

        assert result is not None
        assert result.x1 == 10
        assert result.y1 == 20
        assert result.x2 == 100
        assert result.y2 == 50
        assert result.page_number == 3

    def test_combine_multiple_elements(self) -> None:
        """Test combining multiple elements into encompassing box."""
        bridge = BoundingBoxBridge()
        elements = [
            OCRElement(
                element_type=ElementType.WORD,
                bbox=BoundingBox(x1=10, y1=20, x2=50, y2=40),
                text="first",
            ),
            OCRElement(
                element_type=ElementType.WORD,
                bbox=BoundingBox(x1=60, y1=30, x2=100, y2=60),
                text="second",
            ),
        ]
        result = bridge.combine_element_boxes(elements, page_number=1)

        assert result is not None
        assert result.x1 == 10  # Minimum x1
        assert result.y1 == 20  # Minimum y1
        assert result.x2 == 100  # Maximum x2
        assert result.y2 == 60  # Maximum y2

    def test_get_page_bounding_boxes(self) -> None:
        """Test extracting bounding boxes from a page."""
        bridge = BoundingBoxBridge()
        page = OCRPage(
            page_number=2,
            width=800,
            height=600,
            elements=[
                OCRElement(
                    element_type=ElementType.BLOCK,
                    bbox=BoundingBox(x1=10, y1=10, x2=100, y2=50),
                    text="Block 1",
                ),
                OCRElement(
                    element_type=ElementType.BLOCK,
                    bbox=BoundingBox(x1=10, y1=60, x2=100, y2=100),
                    text="Block 2",
                ),
            ],
        )

        boxes = bridge.get_page_bounding_boxes(page)

        assert len(boxes) == 2
        assert boxes[0].page_number == 2
        assert boxes[1].page_number == 2

    def test_update_chunk_metadata_with_bbox(self) -> None:
        """Test updating metadata with bounding box."""
        bridge = BoundingBoxBridge()
        metadata = {"source_file": "test.pdf"}
        bbox = ChunkBoundingBox(x1=10, y1=20, x2=100, y2=200, page_number=5)

        result = bridge.update_chunk_metadata(metadata, bbox)

        assert result["source_file"] == "test.pdf"
        assert result["bbox_x1"] == 10
        assert result["bbox_y1"] == 20
        assert result["bbox_x2"] == 100
        assert result["bbox_y2"] == 200
        assert result["page_number"] == 5

    def test_update_chunk_metadata_without_bbox(self) -> None:
        """Test updating metadata with None bbox."""
        bridge = BoundingBoxBridge()
        metadata = {"source_file": "test.pdf"}

        result = bridge.update_chunk_metadata(metadata, None)

        assert result == metadata
        assert "bbox_x1" not in result


class TestConvenienceFunctions:
    """Tests for standalone convenience functions."""

    def test_extract_bbox_from_elements(self) -> None:
        """Test extracting bbox tuple from elements."""
        elements = [
            OCRElement(
                element_type=ElementType.WORD,
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
                text="test",
            ),
        ]
        result = extract_bbox_from_elements(elements, page_number=1)
        assert result == (10, 20, 100, 50)

    def test_extract_bbox_empty_elements(self) -> None:
        """Test extracting bbox from empty list."""
        result = extract_bbox_from_elements([])
        assert result is None

    def test_bbox_to_metadata(self) -> None:
        """Test converting bbox tuple to metadata dict."""
        result = bbox_to_metadata((10, 20, 100, 200), page_number=3)
        assert result["bbox_x1"] == 10
        assert result["bbox_y1"] == 20
        assert result["bbox_x2"] == 100
        assert result["bbox_y2"] == 200
        assert result["page_number"] == 3

    def test_bbox_to_metadata_none(self) -> None:
        """Test converting None bbox."""
        result = bbox_to_metadata(None)
        assert result == {}
