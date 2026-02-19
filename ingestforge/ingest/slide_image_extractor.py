#!/usr/bin/env python3
"""Slide image extraction and OCR for SplitAnalyze.

Extracts images embedded in slide presentations (PPTX, Google Slides)
and performs OCR to extract text content from those images.

Dependencies:
- SLIDES-001: PPTXParser for slide structure
- OCR-001: OCREngine for text extraction
"""

import zipfile
import base64
from dataclasses import dataclass, asdict
from typing import Any, Optional
from pathlib import Path

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# Sprint 3 (Rule #4): Import mixin modules to reduce file size
from ingestforge.ingest.extractor_pptx import PPTXExtractionMixin
from ingestforge.ingest.extractor_google import GoogleSlidesExtractionMixin

NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


@dataclass
class ExtractedImage:
    """An image extracted from a slide presentation."""

    image_id: str
    slide_number: int
    name: str
    format: str  # png, jpeg, gif, etc.
    width: Optional[int] = None
    height: Optional[int] = None
    data: Optional[bytes] = None
    data_base64: Optional[str] = None
    alt_text: str = ""
    title: str = ""
    path_in_archive: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (excludes raw bytes)."""
        return {
            "image_id": self.image_id,
            "slide_number": self.slide_number,
            "name": self.name,
            "format": self.format,
            "width": self.width,
            "height": self.height,
            "data_base64": self.data_base64,
            "alt_text": self.alt_text,
            "title": self.title,
            "path_in_archive": self.path_in_archive,
            "size_bytes": len(self.data) if self.data else 0,
        }


@dataclass
class ImageOCRResult:
    """OCR result for an extracted image."""

    image_id: str
    slide_number: int
    text: str
    confidence: float
    language: str
    word_count: int
    processing_time_ms: int
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.error is None and len(self.text) > 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SlideImageExtractionResult:
    """Result of extracting and OCRing images from a presentation."""

    source_file: str
    total_images: int
    images_processed: int
    images_with_text: int
    total_words_extracted: int
    images: list[dict[str, Any]]  # List of ExtractedImage dicts
    ocr_results: list[dict[str, Any]]  # List of ImageOCRResult dicts
    combined_text: str
    processing_time_ms: int
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "total_images": self.total_images,
            "images_processed": self.images_processed,
            "images_with_text": self.images_with_text,
            "total_words_extracted": self.total_words_extracted,
            "images": self.images,
            "ocr_results": self.ocr_results,
            "combined_text": self.combined_text,
            "processing_time_ms": self.processing_time_ms,
            "error": self.error,
        }


class SlideImageExtractor(
    PPTXExtractionMixin,
    GoogleSlidesExtractionMixin,
):
    """
    Extracts images from slide presentations.

    Sprint 3 (Rule #4): Inherits from mixin classes to reduce file size from 1,434 → <400 lines
    """

    """Extracts images from slide presentations."""

    SUPPORTED_IMAGE_FORMATS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".emf",
        ".wmf",
    }

    def __init__(self, include_data: bool = True, include_base64: bool = False) -> None:
        """
        Initialize the image extractor.

        Args:
            include_data: Include raw bytes in results (for OCR processing)
            include_base64: Include base64-encoded data in results (for serialization)
        """
        self.include_data = include_data
        self.include_base64 = include_base64

    def _filter_slide_numbers(
        self, slide_nums: list[int], slide_numbers: Optional[list[int]]
    ) -> Optional[list[int]]:
        """
        Filter slide numbers by requested slides.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            slide_nums: Slide numbers where image appears
            slide_numbers: Requested slide numbers (None = all)

        Returns:
            Filtered slide numbers, or None if image should be skipped
        """
        if slide_numbers is None:
            return slide_nums

        # Filter to requested slides
        filtered = [s for s in slide_nums if s in slide_numbers]
        if not filtered:
            return None

        return filtered

    def _read_image_data(
        self, pptx: zipfile.ZipFile, media_path: str
    ) -> tuple[Optional[bytes], Optional[str]]:
        """
        Read image data and optionally encode as base64.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pptx: PPTX ZIP file
            media_path: Path to image in archive

        Returns:
            Tuple of (data, data_base64)
        """
        data = None
        data_base64 = None
        if not (self.include_data or self.include_base64):
            return (data, data_base64)

        # Read data
        data = pptx.read(media_path)

        # Encode as base64 if requested
        if self.include_base64:
            data_base64 = base64.b64encode(data).decode("utf-8")

        # Clear data if not needed
        if not self.include_data:
            data = None

        return (data, data_base64)

    def _create_extracted_image(
        self,
        media_path: str,
        slide_num: int,
        metadata: dict[str, Any],
        data: Optional[bytes],
        data_base64: Optional[str],
    ) -> ExtractedImage:
        """
        Create ExtractedImage object from components.

        Rule #1: Simple construction
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            media_path: Path in archive
            slide_num: Slide number
            metadata: Image metadata
            data: Raw image bytes
            data_base64: Base64-encoded data

        Returns:
            ExtractedImage object
        """
        name = Path(media_path).name
        ext = Path(media_path).suffix.lower()

        return ExtractedImage(
            image_id=f"img_{Path(media_path).stem}",
            slide_number=slide_num,
            name=name,
            format=ext.lstrip("."),
            width=metadata.get("width"),
            height=metadata.get("height"),
            data=data,
            data_base64=data_base64,
            alt_text=metadata.get("alt_text", ""),
            title=metadata.get("title", ""),
            path_in_archive=media_path,
        )

    def _process_media_file(
        self,
        pptx: zipfile.ZipFile,
        media_path: str,
        slide_image_map: dict[str, list[int]],
        slide_numbers: Optional[list[int]],
    ) -> Optional[ExtractedImage]:
        """
        Process single media file from PPTX.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            pptx: PPTX ZIP file
            media_path: Path to media file
            slide_image_map: Map of media paths to slide numbers
            slide_numbers: Requested slide numbers

        Returns:
            ExtractedImage or None if skipped
        """
        ext = Path(media_path).suffix.lower()
        if ext not in self.SUPPORTED_IMAGE_FORMATS:
            return None

        # Find which slides use this image
        slide_nums = slide_image_map.get(media_path, [])

        # Filter by requested slides
        filtered_nums = self._filter_slide_numbers(slide_nums, slide_numbers)
        if filtered_nums is None:
            return None

        # Use first slide or 0 if not associated
        slide_num = filtered_nums[0] if filtered_nums else 0

        # Get image metadata
        metadata = self._get_image_metadata(pptx, media_path, filtered_nums)

        # Read image data
        data, data_base64 = self._read_image_data(pptx, media_path)

        # Create image object
        return self._create_extracted_image(
            media_path, slide_num, metadata, data, data_base64
        )

    # Sprint 3 (Rule #4): Methods moved to mixin modules
    # PPTX methods → extractor_pptx.py
    # Google Slides methods → extractor_google.py
    # OCR classes → extractor_ocr.py
