"""Tests for batch aggregator module (OCR-003.2).

Tests the image batch aggregator:
- ImageBatch validation
- AggregatedDocument properties
- BatchAggregator processing
- Batch splitting for large collections
- Error handling and recovery
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock

from ingestforge.ingest.ocr.batch_aggregator import (
    BatchAggregator,
    ImageBatch,
    AggregatedDocument,
    DEFAULT_MAX_BATCH_SIZE,
    SUPPORTED_EXTENSIONS,
)
from ingestforge.ingest.ocr_processor import OCRResult


class TestSupportedExtensions:
    """Test supported image extensions."""

    def test_common_formats_supported(self) -> None:
        """Common image formats should be supported."""
        assert ".png" in SUPPORTED_EXTENSIONS
        assert ".jpg" in SUPPORTED_EXTENSIONS
        assert ".jpeg" in SUPPORTED_EXTENSIONS
        assert ".tiff" in SUPPORTED_EXTENSIONS
        assert ".bmp" in SUPPORTED_EXTENSIONS
        assert ".webp" in SUPPORTED_EXTENSIONS

    def test_default_batch_size(self) -> None:
        """Default batch size should be reasonable."""
        assert DEFAULT_MAX_BATCH_SIZE == 50


class TestImageBatch:
    """Test ImageBatch dataclass."""

    def test_empty_batch(self) -> None:
        """Empty batch should be valid."""
        batch = ImageBatch()
        assert len(batch) == 0
        assert batch.is_valid()

    def test_batch_with_paths(self) -> None:
        """Batch should accept Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "test1.png"
            img2 = Path(tmpdir) / "test2.png"
            img1.touch()
            img2.touch()

            batch = ImageBatch(images=[img1, img2])

            assert len(batch) == 2
            assert batch.is_valid()

    def test_batch_with_strings(self) -> None:
        """Batch should convert string paths to Path objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "test1.png"
            img1.touch()

            # Pass as string
            batch = ImageBatch(images=[str(img1)])

            assert isinstance(batch.images[0], Path)

    def test_batch_sorting(self) -> None:
        """Batch should sort images by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_c = Path(tmpdir) / "c.png"
            img_a = Path(tmpdir) / "a.png"
            img_b = Path(tmpdir) / "b.png"

            for p in [img_c, img_a, img_b]:
                p.touch()

            batch = ImageBatch(images=[img_c, img_a, img_b], sort_images=True)

            assert batch.images[0].name == "a.png"
            assert batch.images[1].name == "b.png"
            assert batch.images[2].name == "c.png"

    def test_batch_no_sorting(self) -> None:
        """Batch should preserve order when sort_images=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_c = Path(tmpdir) / "c.png"
            img_a = Path(tmpdir) / "a.png"

            for p in [img_c, img_a]:
                p.touch()

            batch = ImageBatch(images=[img_c, img_a], sort_images=False)

            assert batch.images[0].name == "c.png"
            assert batch.images[1].name == "a.png"

    def test_batch_title(self) -> None:
        """Batch should accept optional title."""
        batch = ImageBatch(title="My Document")
        assert batch.title == "My Document"

    def test_validate_missing_files(self) -> None:
        """Validation should catch missing files."""
        batch = ImageBatch(images=[Path("/nonexistent/file.png")])

        errors = batch.validate()

        assert len(errors) == 1
        assert "not found" in errors[0].lower()
        assert not batch.is_valid()

    def test_validate_unsupported_format(self) -> None:
        """Validation should catch unsupported formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_file = Path(tmpdir) / "test.xyz"
            bad_file.touch()

            batch = ImageBatch(images=[bad_file])
            errors = batch.validate()

            assert len(errors) == 1
            assert "unsupported" in errors[0].lower()

    def test_validate_directory(self) -> None:
        """Validation should reject directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = ImageBatch(images=[Path(tmpdir)])
            errors = batch.validate()

            assert len(errors) == 1
            assert "not a file" in errors[0].lower()


class TestAggregatedDocument:
    """Test AggregatedDocument dataclass."""

    def test_basic_properties(self) -> None:
        """AggregatedDocument should expose basic properties."""
        ocr_result = OCRResult(
            text="Page 1\n\nPage 2",
            pages=["Page 1", "Page 2"],
            confidence=0.85,
            is_scanned=True,
            page_count=2,
        )

        doc = AggregatedDocument(
            ocr_result=ocr_result,
            source_images=[Path("img1.png"), Path("img2.png")],
            title="Test Doc",
        )

        assert doc.page_count == 2
        assert doc.title == "Test Doc"

    def test_success_count_no_failures(self) -> None:
        """Success count should equal total when no failures."""
        ocr_result = OCRResult(
            text="Test",
            pages=["Test"],
            confidence=0.9,
            is_scanned=True,
            page_count=1,
        )

        doc = AggregatedDocument(
            ocr_result=ocr_result,
            source_images=[Path("img1.png"), Path("img2.png")],
        )

        assert doc.success_count == 2
        assert not doc.has_failures

    def test_success_count_with_failures(self) -> None:
        """Success count should exclude failures."""
        ocr_result = OCRResult(
            text="Test",
            pages=["Test"],
            confidence=0.9,
            is_scanned=True,
            page_count=1,
        )

        doc = AggregatedDocument(
            ocr_result=ocr_result,
            source_images=[Path("img1.png"), Path("img2.png"), Path("img3.png")],
            failed_images=[(Path("img3.png"), "Error")],
        )

        assert doc.success_count == 2
        assert doc.has_failures

    def test_page_mapping(self) -> None:
        """Should retrieve source image for page."""
        ocr_result = OCRResult(
            text="Page 1\n\nPage 2",
            pages=["Page 1", "Page 2"],
            confidence=0.85,
            is_scanned=True,
            page_count=2,
        )

        doc = AggregatedDocument(
            ocr_result=ocr_result,
            source_images=[Path("img1.png"), Path("img2.png")],
            page_mapping={0: Path("img1.png"), 1: Path("img2.png")},
        )

        assert doc.get_source_image(0) == Path("img1.png")
        assert doc.get_source_image(1) == Path("img2.png")
        assert doc.get_source_image(99) is None


class TestBatchAggregatorInit:
    """Test BatchAggregator initialization."""

    def test_default_init(self) -> None:
        """Should initialize with defaults."""
        aggregator = BatchAggregator()

        assert aggregator.max_batch_size == DEFAULT_MAX_BATCH_SIZE
        assert aggregator.continue_on_error is True

    def test_custom_batch_size(self) -> None:
        """Should accept custom batch size."""
        aggregator = BatchAggregator(max_batch_size=25)

        assert aggregator.max_batch_size == 25

    def test_invalid_batch_size_zero(self) -> None:
        """Should reject zero batch size."""
        with pytest.raises(ValueError, match="positive"):
            BatchAggregator(max_batch_size=0)

    def test_invalid_batch_size_negative(self) -> None:
        """Should reject negative batch size."""
        with pytest.raises(ValueError, match="positive"):
            BatchAggregator(max_batch_size=-1)

    def test_invalid_batch_size_too_large(self) -> None:
        """Should reject batch size exceeding Rule #2 limit."""
        with pytest.raises(ValueError, match="exceed 100"):
            BatchAggregator(max_batch_size=101)

    def test_batch_size_at_limit(self) -> None:
        """Should accept batch size at limit."""
        aggregator = BatchAggregator(max_batch_size=100)
        assert aggregator.max_batch_size == 100

    def test_lazy_ocr_processor(self) -> None:
        """OCR processor should be lazy-initialized."""
        aggregator = BatchAggregator()

        # Not initialized yet
        assert aggregator._ocr_processor is None

        # Property access initializes it
        processor = aggregator.ocr_processor
        assert processor is not None


class TestBatchAggregatorValidation:
    """Test BatchAggregator validation."""

    def test_batch_exceeds_max_size(self) -> None:
        """Should reject batch exceeding max size."""
        aggregator = BatchAggregator(max_batch_size=2)

        batch = ImageBatch(
            images=[
                Path("img1.png"),
                Path("img2.png"),
                Path("img3.png"),
            ]
        )

        with pytest.raises(ValueError, match="exceeds maximum"):
            aggregator.aggregate_batch(batch)

    def test_batch_at_max_size(self) -> None:
        """Should accept batch at max size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [Path(tmpdir) / f"img{i}.png" for i in range(2)]
            for img in images:
                img.touch()

            # Create mock OCR processor
            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.return_value = OCRResult(
                text="Test",
                pages=["Test"],
                confidence=0.9,
                is_scanned=True,
                page_count=1,
            )

            aggregator = BatchAggregator(
                ocr_processor=mock_processor,
                max_batch_size=2,
            )

            batch = ImageBatch(images=images)
            doc = aggregator.aggregate_batch(batch)

            assert doc.page_count == 2


class TestBatchAggregatorProcessing:
    """Test BatchAggregator processing."""

    def test_aggregate_single_image(self) -> None:
        """Should process single image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "test.png"
            img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.return_value = OCRResult(
                text="Hello World",
                pages=["Hello World"],
                confidence=0.95,
                is_scanned=True,
                page_count=1,
            )

            aggregator = BatchAggregator(ocr_processor=mock_processor)
            doc = aggregator.aggregate([img], title="Test Doc")

            assert doc.page_count == 1
            assert doc.ocr_result.text == "Hello World"
            assert doc.ocr_result.confidence == 0.95
            assert doc.title == "Test Doc"

    def test_aggregate_multiple_images(self) -> None:
        """Should combine multiple images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [Path(tmpdir) / f"img{i}.png" for i in range(3)]
            for img in images:
                img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.side_effect = [
                OCRResult(
                    text=f"Page {i+1}",
                    pages=[f"Page {i+1}"],
                    confidence=0.8 + i * 0.05,
                    is_scanned=True,
                    page_count=1,
                )
                for i in range(3)
            ]

            aggregator = BatchAggregator(ocr_processor=mock_processor)
            doc = aggregator.aggregate(images)

            assert doc.page_count == 3
            assert "Page 1" in doc.ocr_result.text
            assert "Page 2" in doc.ocr_result.text
            assert "Page 3" in doc.ocr_result.text

    def test_page_mapping_tracked(self) -> None:
        """Should track page to source image mapping."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "a.png"
            img2 = Path(tmpdir) / "b.png"
            img1.touch()
            img2.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.side_effect = [
                OCRResult(
                    text="Page 1",
                    pages=["Page 1"],
                    confidence=0.9,
                    is_scanned=True,
                    page_count=1,
                ),
                OCRResult(
                    text="Page 2",
                    pages=["Page 2"],
                    confidence=0.85,
                    is_scanned=True,
                    page_count=1,
                ),
            ]

            aggregator = BatchAggregator(ocr_processor=mock_processor)
            doc = aggregator.aggregate([img1, img2])

            # Should be sorted alphabetically
            assert doc.get_source_image(0) == img1
            assert doc.get_source_image(1) == img2

    def test_continue_on_error_true(self) -> None:
        """Should continue processing after error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "good.png"
            img2 = Path(tmpdir) / "bad.png"
            img3 = Path(tmpdir) / "good2.png"

            for img in [img1, img2, img3]:
                img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.side_effect = [
                OCRResult(
                    text="Page 1",
                    pages=["Page 1"],
                    confidence=0.9,
                    is_scanned=True,
                    page_count=1,
                ),
                RuntimeError("OCR failed"),
                OCRResult(
                    text="Page 3",
                    pages=["Page 3"],
                    confidence=0.85,
                    is_scanned=True,
                    page_count=1,
                ),
            ]

            aggregator = BatchAggregator(
                ocr_processor=mock_processor,
                continue_on_error=True,
            )
            doc = aggregator.aggregate([img1, img2, img3])

            assert doc.page_count == 2
            assert doc.has_failures
            assert len(doc.failed_images) == 1

    def test_continue_on_error_false(self) -> None:
        """Should stop on error when continue_on_error=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img1 = Path(tmpdir) / "good.png"
            img2 = Path(tmpdir) / "bad.png"

            for img in [img1, img2]:
                img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.side_effect = [
                OCRResult(
                    text="Page 1",
                    pages=["Page 1"],
                    confidence=0.9,
                    is_scanned=True,
                    page_count=1,
                ),
                RuntimeError("OCR failed"),
            ]

            aggregator = BatchAggregator(
                ocr_processor=mock_processor,
                continue_on_error=False,
            )

            with pytest.raises(RuntimeError, match="OCR failed"):
                aggregator.aggregate([img1, img2])

    def test_tesseract_unavailable(self) -> None:
        """Should raise when Tesseract not available."""
        mock_processor = Mock()
        mock_processor.tesseract_available = False

        aggregator = BatchAggregator(ocr_processor=mock_processor)
        batch = ImageBatch(images=[Path("test.png")])

        with pytest.raises(RuntimeError, match="Tesseract"):
            aggregator.aggregate_batch(batch)


class TestBatchSplitting:
    """Test batch splitting for large collections."""

    def test_split_into_batches_single(self) -> None:
        """Should create single batch for small collection."""
        aggregator = BatchAggregator(max_batch_size=10)
        images = [Path(f"img{i}.png") for i in range(5)]

        batches = aggregator.split_into_batches(images)

        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_split_into_batches_multiple(self) -> None:
        """Should split large collection into batches."""
        aggregator = BatchAggregator(max_batch_size=3)
        images = [Path(f"img{i}.png") for i in range(10)]

        batches = aggregator.split_into_batches(images)

        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_split_exact_multiple(self) -> None:
        """Should handle exact multiple of batch size."""
        aggregator = BatchAggregator(max_batch_size=3)
        images = [Path(f"img{i}.png") for i in range(9)]

        batches = aggregator.split_into_batches(images)

        assert len(batches) == 3
        assert all(len(b) == 3 for b in batches)

    def test_split_empty(self) -> None:
        """Should handle empty collection."""
        aggregator = BatchAggregator()
        batches = aggregator.split_into_batches([])

        assert len(batches) == 0

    def test_aggregate_all_single_batch(self) -> None:
        """Should process single batch without splitting."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [Path(tmpdir) / f"img{i}.png" for i in range(3)]
            for img in images:
                img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.return_value = OCRResult(
                text="Test",
                pages=["Test"],
                confidence=0.9,
                is_scanned=True,
                page_count=1,
            )

            aggregator = BatchAggregator(
                ocr_processor=mock_processor,
                max_batch_size=10,
            )
            doc = aggregator.aggregate_all(images, title="Full Doc")

            assert doc.page_count == 3
            assert doc.title == "Full Doc"

    def test_aggregate_all_multiple_batches(self) -> None:
        """Should merge results from multiple batches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [Path(tmpdir) / f"img{i}.png" for i in range(5)]
            for img in images:
                img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.side_effect = [
                OCRResult(
                    text=f"Page {i+1}",
                    pages=[f"Page {i+1}"],
                    confidence=0.85,
                    is_scanned=True,
                    page_count=1,
                )
                for i in range(5)
            ]

            aggregator = BatchAggregator(
                ocr_processor=mock_processor,
                max_batch_size=2,
            )
            doc = aggregator.aggregate_all(images, title="Multi-Batch")

            assert doc.page_count == 5
            assert doc.title == "Multi-Batch"
            assert len(doc.source_images) == 5


class TestAverageConfidence:
    """Test confidence calculation."""

    def test_single_image_confidence(self) -> None:
        """Single image should use its confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "test.png"
            img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.return_value = OCRResult(
                text="Test",
                pages=["Test"],
                confidence=0.92,
                is_scanned=True,
                page_count=1,
            )

            aggregator = BatchAggregator(ocr_processor=mock_processor)
            doc = aggregator.aggregate([img])

            assert doc.ocr_result.confidence == 0.92

    def test_average_confidence_calculation(self) -> None:
        """Multiple images should average confidence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            images = [Path(tmpdir) / f"img{i}.png" for i in range(3)]
            for img in images:
                img.touch()

            mock_processor = Mock()
            mock_processor.tesseract_available = True
            mock_processor.process_image.side_effect = [
                OCRResult(
                    text="P1",
                    pages=["P1"],
                    confidence=0.80,
                    is_scanned=True,
                    page_count=1,
                ),
                OCRResult(
                    text="P2",
                    pages=["P2"],
                    confidence=0.90,
                    is_scanned=True,
                    page_count=1,
                ),
                OCRResult(
                    text="P3",
                    pages=["P3"],
                    confidence=1.00,
                    is_scanned=True,
                    page_count=1,
                ),
            ]

            aggregator = BatchAggregator(ocr_processor=mock_processor)
            doc = aggregator.aggregate(images)

            # (0.80 + 0.90 + 1.00) / 3 = 0.9
            assert doc.ocr_result.confidence == 0.9

    def test_empty_batch_zero_confidence(self) -> None:
        """Empty batch should have zero confidence."""
        mock_processor = Mock()
        mock_processor.tesseract_available = True

        aggregator = BatchAggregator(ocr_processor=mock_processor)
        batch = ImageBatch()
        doc = aggregator.aggregate_batch(batch)

        assert doc.ocr_result.confidence == 0.0
        assert doc.page_count == 0
