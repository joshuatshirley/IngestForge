"""Image batch aggregator for multi-scan documents (OCR-003.2).

This module provides functionality to merge multiple scanned images
into a single cohesive document with combined OCR results.

NASA JPL Rule #2 Compliance: Fixed upper bound on batch size prevents
unbounded memory consumption during batch processing.

Usage:
    from ingestforge.ingest.ocr.batch_aggregator import BatchAggregator

    aggregator = BatchAggregator()
    result = aggregator.aggregate(image_paths)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from ingestforge.core.logging import get_logger
from ingestforge.ingest.ocr_processor import OCRProcessor, OCRResult

logger = get_logger(__name__)
DEFAULT_MAX_BATCH_SIZE: int = 50

# Supported image extensions
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".png",
        ".jpg",
        ".jpeg",
        ".tiff",
        ".tif",
        ".bmp",
        ".webp",
        ".gif",
    }
)


@dataclass
class ImageBatch:
    """A batch of images to be processed together.

    Attributes:
        images: List of image file paths
        title: Optional title for the aggregated document
        sort_images: Whether to sort images by name before processing
    """

    images: List[Path] = field(default_factory=list)
    title: Optional[str] = None
    sort_images: bool = True

    def __post_init__(self) -> None:
        """Validate and normalize image paths."""
        # Convert strings to Path objects
        self.images = [Path(p) if isinstance(p, str) else p for p in self.images]

        # Sort if requested
        if self.sort_images:
            self.images = sorted(self.images, key=lambda p: p.name.lower())

    def __len__(self) -> int:
        """Return number of images in batch."""
        return len(self.images)

    def validate(self) -> List[str]:
        """Validate all images in the batch.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        for img_path in self.images:
            error = self._validate_single_image(img_path)
            if error:
                errors.append(error)

        return errors

    def _validate_single_image(self, img_path: Path) -> Optional[str]:
        """Validate a single image file.

        Args:
            img_path: Image path to validate

        Returns:
            Error message or None if valid
        """
        if not img_path.exists():
            return f"File not found: {img_path}"
        if not img_path.is_file():
            return f"Not a file: {img_path}"
        if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return f"Unsupported format: {img_path.suffix} for {img_path.name}"
        return None

    def is_valid(self) -> bool:
        """Check if batch is valid."""
        return len(self.validate()) == 0


@dataclass
class AggregatedDocument:
    """Result of aggregating multiple images into a document.

    Attributes:
        ocr_result: Combined OCR result from all images
        source_images: List of source image paths
        page_mapping: Maps page index to source image path
        failed_images: Images that failed processing
        title: Document title
    """

    ocr_result: OCRResult
    source_images: List[Path]
    page_mapping: dict[int, Path] = field(default_factory=dict)
    failed_images: List[tuple[Path, str]] = field(default_factory=list)
    title: Optional[str] = None

    @property
    def page_count(self) -> int:
        """Total number of pages in document."""
        return self.ocr_result.page_count

    @property
    def success_count(self) -> int:
        """Number of successfully processed images."""
        return len(self.source_images) - len(self.failed_images)

    @property
    def has_failures(self) -> bool:
        """Whether any images failed processing."""
        return len(self.failed_images) > 0

    def get_source_image(self, page_index: int) -> Optional[Path]:
        """Get source image for a specific page."""
        return self.page_mapping.get(page_index)


class BatchAggregator:
    """Aggregates multiple scanned images into a single document.

    This class processes multiple image files, applies OCR to each,
    and combines the results into a unified document.

    NASA JPL Rule #2: Enforces fixed upper bound on batch size.

    Args:
        ocr_processor: OCR processor instance (created if not provided)
        max_batch_size: Maximum images per batch (default: 50)
        continue_on_error: Whether to continue if individual images fail

    Example:
        aggregator = BatchAggregator(max_batch_size=25)

        batch = ImageBatch(
            images=[Path("scan1.png"), Path("scan2.png")],
            title="My Document"
        )

        doc = aggregator.aggregate_batch(batch)
        print(f"Processed {doc.page_count} pages")
    """

    def __init__(
        self,
        ocr_processor: Optional[OCRProcessor] = None,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        continue_on_error: bool = True,
    ) -> None:
        """Initialize the batch aggregator."""
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if max_batch_size > 100:
            raise ValueError(
                "max_batch_size cannot exceed 100 (Rule #2: fixed upper bound)"
            )

        self._ocr_processor = ocr_processor
        self.max_batch_size = max_batch_size
        self.continue_on_error = continue_on_error

    @property
    def ocr_processor(self) -> OCRProcessor:
        """Lazy-initialize OCR processor."""
        if self._ocr_processor is None:
            self._ocr_processor = OCRProcessor()
        return self._ocr_processor

    def aggregate(
        self,
        image_paths: Sequence[Path],
        title: Optional[str] = None,
        sort_images: bool = True,
    ) -> AggregatedDocument:
        """Aggregate multiple images into a single document.

        Convenience method that creates a batch and processes it.

        Args:
            image_paths: Sequence of image file paths
            title: Optional document title
            sort_images: Whether to sort images by filename

        Returns:
            AggregatedDocument containing combined OCR results

        Raises:
            ValueError: If batch exceeds max_batch_size
        """
        batch = ImageBatch(
            images=list(image_paths),
            title=title,
            sort_images=sort_images,
        )
        return self.aggregate_batch(batch)

    def aggregate_batch(self, batch: ImageBatch) -> AggregatedDocument:
        """Aggregate an ImageBatch into a document.

        Args:
            batch: ImageBatch containing images to process

        Returns:
            AggregatedDocument with combined OCR results

        Raises:
            ValueError: If batch exceeds max_batch_size or validation fails
            RuntimeError: If OCR processor is unavailable
        """
        if len(batch) > self.max_batch_size:
            raise ValueError(
                f"Batch size {len(batch)} exceeds maximum of {self.max_batch_size}. "
                f"Split into smaller batches."
            )

        # Validate batch
        errors = batch.validate()
        if errors and not self.continue_on_error:
            raise ValueError(f"Batch validation failed: {'; '.join(errors)}")

        # Check OCR availability
        if not self.ocr_processor.tesseract_available:
            raise RuntimeError(
                "Tesseract OCR is not available. "
                "Install with: pip install pytesseract"
            )

        return self._process_batch(batch)

    def _process_batch(self, batch: ImageBatch) -> AggregatedDocument:
        """Process all images in a batch.

        Args:
            batch: Validated ImageBatch to process

        Returns:
            AggregatedDocument with results
        """
        all_pages: List[str] = []
        confidences: List[float] = []
        page_mapping: dict[int, Path] = {}
        failed_images: List[tuple[Path, str]] = []
        current_page = 0

        logger.info(f"Processing batch of {len(batch)} images")

        for idx, img_path in enumerate(batch.images):
            try:
                result = self._process_single_image(img_path, idx, len(batch))

                # Map pages to source image
                for page_idx in range(len(result.pages)):
                    page_mapping[current_page + page_idx] = img_path

                all_pages.extend(result.pages)
                confidences.append(result.confidence)
                current_page += len(result.pages)

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Failed to process {img_path.name}: {error_msg}")
                failed_images.append((img_path, error_msg))

                if not self.continue_on_error:
                    raise

        # Calculate average confidence
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # Build combined OCR result
        combined_result = OCRResult(
            text="\n\n".join(all_pages),
            pages=all_pages,
            confidence=avg_confidence,
            is_scanned=True,
            page_count=len(all_pages),
            scanned_page_count=len(all_pages),
        )

        logger.info(
            f"Batch complete: {len(all_pages)} pages, "
            f"{len(failed_images)} failures, "
            f"{avg_confidence:.0%} avg confidence"
        )

        return AggregatedDocument(
            ocr_result=combined_result,
            source_images=batch.images,
            page_mapping=page_mapping,
            failed_images=failed_images,
            title=batch.title,
        )

    def _process_single_image(
        self,
        img_path: Path,
        index: int,
        total: int,
    ) -> OCRResult:
        """Process a single image file.

        Args:
            img_path: Path to image file
            index: Current image index (0-based)
            total: Total images in batch

        Returns:
            OCRResult for the image
        """
        logger.debug(f"Processing image {index + 1}/{total}: {img_path.name}")

        result = self.ocr_processor.process_image(img_path)

        logger.debug(
            f"Image {index + 1}/{total} complete: "
            f"{len(result.text)} chars, {result.confidence:.0%} confidence"
        )

        return result

    def split_into_batches(
        self,
        image_paths: Sequence[Path],
    ) -> List[ImageBatch]:
        """Split a large set of images into compliant batches.

        Use this when you have more images than max_batch_size.

        Args:
            image_paths: All image paths to split

        Returns:
            List of ImageBatch objects, each within size limit
        """
        batches: List[ImageBatch] = []
        paths_list = list(image_paths)
        num_batches = (len(paths_list) + self.max_batch_size - 1) // self.max_batch_size

        for i in range(num_batches):
            start = i * self.max_batch_size
            end = min(start + self.max_batch_size, len(paths_list))

            batch = ImageBatch(
                images=paths_list[start:end],
                title=f"Batch {i + 1}/{num_batches}" if num_batches > 1 else None,
            )
            batches.append(batch)

        return batches

    def aggregate_all(
        self,
        image_paths: Sequence[Path],
        title: Optional[str] = None,
    ) -> AggregatedDocument:
        """Aggregate any number of images, splitting into batches as needed.

        This method handles large collections by processing in batches
        and merging results.

        Args:
            image_paths: All image paths to process
            title: Document title

        Returns:
            Single AggregatedDocument combining all results
        """
        batches = self.split_into_batches(image_paths)

        if len(batches) == 1:
            # Simple case: single batch
            batches[0].title = title
            return self.aggregate_batch(batches[0])

        # Multiple batches: process and merge
        logger.info(f"Processing {len(image_paths)} images in {len(batches)} batches")

        all_pages: List[str] = []
        all_confidences: List[float] = []
        all_page_mapping: dict[int, Path] = {}
        all_failures: List[tuple[Path, str]] = []
        all_sources: List[Path] = []
        current_page = 0

        for batch_idx, batch in enumerate(batches):
            logger.info(f"Processing batch {batch_idx + 1}/{len(batches)}")

            doc = self.aggregate_batch(batch)

            # Merge results
            all_pages.extend(doc.ocr_result.pages)
            all_confidences.append(doc.ocr_result.confidence)
            all_sources.extend(batch.images)
            all_failures.extend(doc.failed_images)

            # Update page mapping with offset
            for page_idx, source in doc.page_mapping.items():
                all_page_mapping[current_page + page_idx] = source

            current_page += doc.page_count

        # Build merged result
        avg_confidence = (
            sum(all_confidences) / len(all_confidences) if all_confidences else 0.0
        )

        combined_result = OCRResult(
            text="\n\n".join(all_pages),
            pages=all_pages,
            confidence=avg_confidence,
            is_scanned=True,
            page_count=len(all_pages),
            scanned_page_count=len(all_pages),
        )

        return AggregatedDocument(
            ocr_result=combined_result,
            source_images=all_sources,
            page_mapping=all_page_mapping,
            failed_images=all_failures,
            title=title,
        )
