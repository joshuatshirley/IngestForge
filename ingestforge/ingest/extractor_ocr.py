"""
Slide Image OCR and Processing Classes.

Handles OCR extraction from slide images and image processing utilities.

This module is part of the slide_image_extractor refactoring (Sprint 3, Rule #4)
to reduce slide_image_extractor.py from 1,434 lines to <400 lines.
"""

import base64
import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

from ingestforge.core.logging import get_logger

# Import dataclasses and classes from main module
from ingestforge.ingest.slide_image_extractor import (
    ExtractedImage,
    ImageOCRResult,
    SlideImageExtractor,
    SlideImageExtractionResult,
)

logger = get_logger(__name__)


class SlideImageOCR:
    """OCR processing for slide images."""

    def __init__(self, ocr_engine: Any = None, language: str = "eng") -> None:
        """
        Initialize with an OCR engine.

        Args:
            ocr_engine: OCR engine instance (from OCR-001)
            language: Default OCR language
        """
        self.ocr_engine = ocr_engine
        self.language = language
        self._temp_files: list[str] = []

    def process_images(
        self,
        images: list[ExtractedImage],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[ImageOCRResult]:
        """
        Perform OCR on extracted images.

        Args:
            images: List of ExtractedImage objects
            progress_callback: Optional callback(current, total)

        Returns:
            List of ImageOCRResult objects
        """
        results = []
        total = len(images)

        for i, image in enumerate(images):
            if progress_callback:
                progress_callback(i + 1, total)

            result = self._process_single_image(image)
            results.append(result)

        # Cleanup temp files
        self._cleanup_temp_files()

        return results

    def _build_error_ocr_result(
        self, image: ExtractedImage, elapsed_ms: int, error: str
    ) -> ImageOCRResult:
        """
        Build ImageOCRResult for error cases.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            image: Source image
            elapsed_ms: Processing time
            error: Error message

        Returns:
            ImageOCRResult with error
        """
        return ImageOCRResult(
            image_id=image.image_id,
            slide_number=image.slide_number,
            text="",
            confidence=0.0,
            language=self.language,
            word_count=0,
            processing_time_ms=elapsed_ms,
            error=error,
        )

    def _get_image_data(self, image: ExtractedImage) -> bytes:
        """
        Get image data from either data or base64.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            image: Source image

        Returns:
            Image data bytes
        """
        if image.data is not None:
            return image.data
        if image.data_base64 is None:
            return b""
        return base64.b64decode(image.data_base64)

    def _save_image_to_temp(
        self, image_data: bytes, image_format: Optional[str]
    ) -> str:
        """
        Save image data to temporary file.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            image_data: Image bytes
            image_format: Image format (for file extension)

        Returns:
            Path to temporary file
        """
        import tempfile

        suffix = f".{image_format}" if image_format else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_data)
            temp_path = f.name
            self._temp_files.append(temp_path)
        return temp_path

    def _convert_ocr_result(
        self, ocr_result: Any, image: ExtractedImage, elapsed_ms: int
    ) -> ImageOCRResult:
        """
        Convert OCR engine result to ImageOCRResult.

        Rule #1: Early return for dict vs object
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            ocr_result: Result from OCR engine (dict or object)
            image: Source image
            elapsed_ms: Processing time

        Returns:
            ImageOCRResult
        """
        if isinstance(ocr_result, dict):
            return ImageOCRResult(
                image_id=image.image_id,
                slide_number=image.slide_number,
                text=ocr_result.get("text", ""),
                confidence=ocr_result.get("confidence", 0.0),
                language=ocr_result.get("language", self.language),
                word_count=ocr_result.get("word_count", 0),
                processing_time_ms=elapsed_ms,
                error=ocr_result.get("error"),
            )

        # Handle object result
        return ImageOCRResult(
            image_id=image.image_id,
            slide_number=image.slide_number,
            text=getattr(ocr_result, "text", ""),
            confidence=getattr(ocr_result, "confidence", 0.0),
            language=getattr(ocr_result, "language", self.language),
            word_count=getattr(ocr_result, "word_count", 0),
            processing_time_ms=elapsed_ms,
            error=getattr(ocr_result, "error", None),
        )

    def _process_single_image(self, image: ExtractedImage) -> ImageOCRResult:
        """
        Process a single image with OCR.

        Rule #1: Early returns for validation
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            image: Extracted image to process

        Returns:
            ImageOCRResult with OCR results or error
        """
        import time

        start_time = time.time()
        if self.ocr_engine is None:
            return self._build_error_ocr_result(image, 0, "OCR engine not configured")

        if image.data is None and image.data_base64 is None:
            return self._build_error_ocr_result(image, 0, "No image data available")

        try:
            # Process using helpers
            image_data = self._get_image_data(image)
            temp_path = self._save_image_to_temp(image_data, image.format)
            ocr_result = self.ocr_engine.extract_text(temp_path, language=self.language)
            elapsed_ms = int((time.time() - start_time) * 1000)

            return self._convert_ocr_result(ocr_result, image, elapsed_ms)

        except Exception as e:
            elapsed_ms = int((time.time() - start_time) * 1000)
            return self._build_error_ocr_result(image, elapsed_ms, str(e))

    def _cleanup_temp_files(self) -> None:
        """
        Clean up temporary files.

        Rule #5: Log cleanup failures
        """
        import os

        for temp_file in self._temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.debug(f"Failed to delete temporary file {temp_file}: {e}")
        self._temp_files = []


class PresentationImageProcessor:
    """High-level processor for extracting and OCRing presentation images."""

    def __init__(self, ocr_engine: Any = None, language: str = "eng") -> None:
        """
        Initialize the processor.

        Args:
            ocr_engine: OCR engine instance
            language: Default OCR language
        """
        self.extractor = SlideImageExtractor(include_data=True, include_base64=True)
        self.ocr = SlideImageOCR(ocr_engine, language)

    def _extract_images_with_progress(
        self,
        path: Path,
        slide_numbers: Optional[list[int]],
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> list[ExtractedImage]:
        """
        Extract images from PPTX with progress callback.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            path: Path to PPTX file
            slide_numbers: Optional specific slides
            progress_callback: Progress callback

        Returns:
            List of extracted images
        """
        if progress_callback:
            progress_callback(0, 1, "extracting")

        return self.extractor.extract_from_pptx(path, slide_numbers)

    def _perform_ocr_on_images(
        self,
        images: list[ExtractedImage],
        skip_ocr: bool,
        progress_callback: Optional[Callable[[int, int, str], None]],
    ) -> list[ImageOCRResult]:
        """
        Perform OCR on extracted images.

        Rule #1: Early return for skip/empty
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            images: List of extracted images
            skip_ocr: Whether to skip OCR
            progress_callback: Progress callback

        Returns:
            List of OCR results
        """
        if skip_ocr or not images:
            return []

        def ocr_progress(current: Any, total: Any) -> None:
            if progress_callback:
                progress_callback(current, total, "ocr")

        return self.ocr.process_images(images, ocr_progress)

    def _calculate_extraction_statistics(
        self, ocr_results: list[ImageOCRResult]
    ) -> tuple[int, int]:
        """
        Calculate statistics from OCR results.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            ocr_results: List of OCR results

        Returns:
            Tuple of (images_with_text, total_words)
        """
        images_with_text = sum(1 for r in ocr_results if r.is_successful)
        total_words = sum(r.word_count for r in ocr_results)
        return (images_with_text, total_words)

    def _combine_extraction_text(self, ocr_results: list[ImageOCRResult]) -> str:
        """
        Combine all extracted text from OCR results.

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            ocr_results: List of OCR results

        Returns:
            Combined text string
        """
        combined_parts = []
        for result in ocr_results:
            if result.text:
                combined_parts.append(
                    f"[Slide {result.slide_number} - {result.image_id}]\n{result.text}"
                )
        return "\n\n".join(combined_parts)

    def _build_error_result(
        self, path: Path, start_time: float, error: str
    ) -> SlideImageExtractionResult:
        """
        Build error result for exceptions.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            path: File path
            start_time: Processing start time
            error: Error message

        Returns:
            SlideImageExtractionResult with error
        """
        return SlideImageExtractionResult(
            source_file=str(path),
            total_images=0,
            images_processed=0,
            images_with_text=0,
            total_words_extracted=0,
            images=[],
            ocr_results=[],
            combined_text="",
            processing_time_ms=int((time.time() - start_time) * 1000),
            error=error,
        )

    def process_pptx(
        self,
        file_path: Union[str, Path],
        slide_numbers: Optional[list[int]] = None,
        skip_ocr: bool = False,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SlideImageExtractionResult:
        """
        Extract images from PPTX and perform OCR.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: Path to PPTX file
            slide_numbers: Optional specific slides to process
            skip_ocr: If True, skip OCR processing
            progress_callback: Optional callback(current, total, stage)

        Returns:
            SlideImageExtractionResult
        """
        start_time = time.time()
        path = Path(file_path)

        try:
            # Extract and process using helpers
            images = self._extract_images_with_progress(
                path, slide_numbers, progress_callback
            )
            ocr_results = self._perform_ocr_on_images(
                images, skip_ocr, progress_callback
            )
            images_with_text, total_words = self._calculate_extraction_statistics(
                ocr_results
            )
            combined_text = self._combine_extraction_text(ocr_results)

            elapsed_ms = int((time.time() - start_time) * 1000)

            return SlideImageExtractionResult(
                source_file=str(path),
                total_images=len(images),
                images_processed=len(ocr_results),
                images_with_text=images_with_text,
                total_words_extracted=total_words,
                images=[img.to_dict() for img in images],
                ocr_results=[r.to_dict() for r in ocr_results],
                combined_text=combined_text,
                processing_time_ms=elapsed_ms,
            )

        except FileNotFoundError as e:
            return self._build_error_result(path, start_time, str(e))
        except Exception as e:
            return self._build_error_result(path, start_time, str(e))

    def _extract_google_slides_images(
        self,
        presentation_data: dict[str, Any],
        fetch_image_func: Optional[Callable[[str], bytes]],
        skip_ocr: bool,
    ) -> list[ExtractedImage]:
        """
        Extract images from Google Slides presentation.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            presentation_data: Google Slides API response
            fetch_image_func: Function to fetch image data
            skip_ocr: Whether to skip OCR

        Returns:
            List of extracted images
        """
        extractor = SlideImageExtractor(
            include_data=not skip_ocr,
            include_base64=True,
        )
        return extractor.extract_from_google_slides(
            presentation_data,
            fetch_image_func,
        )

    def _perform_ocr_on_google_slides(
        self,
        images: list[ExtractedImage],
        skip_ocr: bool,
        progress_callback: Optional[Callable[[int, int], None]],
    ) -> list[ImageOCRResult]:
        """
        Perform OCR on Google Slides images.

        Rule #1: Early return for skip/empty
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            images: List of extracted images
            skip_ocr: Whether to skip OCR
            progress_callback: Progress callback

        Returns:
            List of OCR results
        """
        if skip_ocr or not images:
            return []

        return self.ocr.process_images(images, progress_callback)

    def _combine_google_slides_text(self, ocr_results: list[ImageOCRResult]) -> str:
        """
        Combine text from Google Slides OCR results.

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            ocr_results: List of OCR results

        Returns:
            Combined text string
        """
        combined_parts = []
        for result in ocr_results:
            if result.text:
                combined_parts.append(f"[Slide {result.slide_number}]\n{result.text}")
        return "\n\n".join(combined_parts)

    def _build_google_slides_error_result(
        self, start_time: float, error: str
    ) -> SlideImageExtractionResult:
        """
        Build error result for Google Slides.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            start_time: Processing start time
            error: Error message

        Returns:
            SlideImageExtractionResult with error
        """
        return SlideImageExtractionResult(
            source_file="Google Slides",
            total_images=0,
            images_processed=0,
            images_with_text=0,
            total_words_extracted=0,
            images=[],
            ocr_results=[],
            combined_text="",
            processing_time_ms=int((time.time() - start_time) * 1000),
            error=error,
        )

    def process_google_slides(
        self,
        presentation_data: dict[str, Any],
        fetch_image_func: Optional[Callable[[str], bytes]] = None,
        skip_ocr: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> SlideImageExtractionResult:
        """
        Extract images from Google Slides and perform OCR.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            presentation_data: Google Slides API response
            fetch_image_func: Function to fetch image data from URL
            skip_ocr: If True, skip OCR processing
            progress_callback: Optional callback

        Returns:
            SlideImageExtractionResult
        """
        start_time = time.time()

        try:
            # Adjust skip_ocr if no fetch function available
            if not skip_ocr and fetch_image_func is None:
                skip_ocr = True

            # Extract and process using helpers
            images = self._extract_google_slides_images(
                presentation_data, fetch_image_func, skip_ocr
            )
            ocr_results = self._perform_ocr_on_google_slides(
                images, skip_ocr, progress_callback
            )
            images_with_text, total_words = self._calculate_extraction_statistics(
                ocr_results
            )
            combined_text = self._combine_google_slides_text(ocr_results)

            elapsed_ms = int((time.time() - start_time) * 1000)
            title = presentation_data.get("title", "Google Slides")

            return SlideImageExtractionResult(
                source_file=title,
                total_images=len(images),
                images_processed=len(ocr_results),
                images_with_text=images_with_text,
                total_words_extracted=total_words,
                images=[img.to_dict() for img in images],
                ocr_results=[r.to_dict() for r in ocr_results],
                combined_text=combined_text,
                processing_time_ms=elapsed_ms,
            )

        except Exception as e:
            return self._build_google_slides_error_result(start_time, str(e))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Extracting images from: {file_path}")

        try:
            images = extract_pptx_images(file_path)
            print(f"\nFound {len(images)} images:")

            for img in images:
                print(f"  - {img['name']} (slide {img['slide_number']})")
                if img.get("alt_text"):
                    print(f"    Alt text: {img['alt_text']}")
                print(f"    Size: {img.get('width', '?')}x{img.get('height', '?')}")
                print(f"    Format: {img['format']}")

        except Exception as e:
            print(f"Error: {e}")
    else:
        print("Usage: python slide_image_extractor.py <file.pptx>")
