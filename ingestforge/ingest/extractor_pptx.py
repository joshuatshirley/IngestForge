"""
PPTX Image Extraction Mixin.

Handles extraction of images from PowerPoint (.pptx) files.

This module is part of the slide_image_extractor refactoring (Sprint 3, Rule #4)
to reduce slide_image_extractor.py from 1,434 lines to <400 lines.
"""

import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Union
from xml.etree import ElementTree as ET

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# XML namespaces for PPTX parsing
NAMESPACES = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


class PPTXExtractionMixin:
    """
    Mixin providing PPTX image extraction methods.

    Rule #4: Extracted from slide_image_extractor.py to reduce file size
    """

    def extract_from_pptx(
        self,
        file_path: Union[str, Path],
        slide_numbers: Optional[list[int]] = None,
    ) -> list[Any]:
        """
        Extract images from a PPTX file.

        Rule #1: Reduced nesting via helper extraction
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: Path to PPTX file
            slide_numbers: Optional list of specific slide numbers to extract from

        Returns:
            List of ExtractedImage objects
        """
        path = Path(file_path)
        if not path.exists():
            # SEC-002: Sanitize path disclosure
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError("File not found: [REDACTED]")

        if path.suffix.lower() != ".pptx":
            raise ValueError("Not a PPTX file")

        try:
            with zipfile.ZipFile(path, "r") as pptx:
                return self._extract_images_from_pptx_zip(pptx, slide_numbers)

        except zipfile.BadZipFile:
            raise ValueError("Invalid PPTX file (corrupted or not a valid ZIP)")

    def _extract_images_from_pptx_zip(
        self, pptx: zipfile.ZipFile, slide_numbers: Optional[list[int]]
    ) -> list[Any]:
        """Extract images from opened PPTX ZIP file.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        # Build relationship maps for each slide
        slide_image_map = self._build_slide_image_map(pptx)

        # Extract images from media folder
        media_files = [f for f in pptx.namelist() if f.startswith("ppt/media/")]

        images = []
        for media_path in media_files:
            image = self._process_media_file(
                pptx, media_path, slide_image_map, slide_numbers
            )
            if image:
                images.append(image)

        return images

    def _normalize_image_path(self, target: str) -> str:
        """
        Normalize image path from relationship target.

        Rule #1: Simple conditional logic
        Rule #4: Function <60 lines

        Args:
            target: Relationship target path

        Returns:
            Normalized path relative to ppt/
        """
        # Remove leading ../ if present
        if target.startswith("../"):
            # Relative to slide file: ../media/image1.png
            return "ppt/" + target[3:]

        # Absolute path from ppt/: media/image1.png
        if not target.startswith("ppt/"):
            return "ppt/" + target

        return target

    def _record_media_slide(
        self,
        media_to_slides: Dict[str, list[int]],
        normalized_path: str,
        slide_num: int,
    ) -> None:
        """
        Record which slide contains which media file.

        Rule #4: Extracted to reduce nesting
        Rule #9: Full type hints
        """
        if normalized_path not in media_to_slides:
            media_to_slides[normalized_path] = []
        media_to_slides[normalized_path].append(slide_num)

    def _process_slide_relationships(
        self,
        pptx: zipfile.ZipFile,
        slide_rels_path: str,
        slide_num: int,
        media_to_slides: Dict[str, list[int]],
    ) -> None:
        """
        Process relationships for a single slide.

        Rule #1: Reduced nesting with early returns
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pptx: Open PPTX archive
            slide_rels_path: Path to slide relationships XML
            slide_num: Slide number (1-indexed)
            media_to_slides: Dictionary mapping media paths to slide numbers
        """
        if slide_rels_path not in pptx.namelist():
            return

        try:
            rels_xml = pptx.read(slide_rels_path)
            rels_tree = ET.fromstring(rels_xml)

            for rel in rels_tree.findall(
                ".//{http://schemas.openxmlformats.org/package/2006/relationships}Relationship"
            ):
                rel_type = rel.get("Type", "")
                if "image" not in rel_type.lower():
                    continue

                target = rel.get("Target", "")
                if not target:
                    continue

                normalized_path = self._normalize_image_path(target)
                self._record_media_slide(media_to_slides, normalized_path, slide_num)

        except ET.ParseError as e:
            logger.warning(f"Failed to parse relationships for slide {slide_num}: {e}")

    def _build_slide_image_map(self, pptx: zipfile.ZipFile) -> dict[str, Any]:
        """
        Build mapping of media files to slide numbers.

        Rule #1: Simple loop structure
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pptx: Open PPTX archive

        Returns:
            Dict mapping media paths to list of slide numbers
        """
        media_to_slides: dict[str, list[int]] = {}

        # Find all slide relationship files
        slide_rels = [
            f
            for f in pptx.namelist()
            if f.startswith("ppt/slides/_rels/slide") and f.endswith(".xml.rels")
        ]

        for slide_rels_path in slide_rels:
            # Extract slide number from path like: ppt/slides/_rels/slide3.xml.rels
            try:
                slide_num = int(slide_rels_path.split("slide")[1].split(".")[0])
            except (IndexError, ValueError):
                continue

            self._process_slide_relationships(
                pptx, slide_rels_path, slide_num, media_to_slides
            )

        return media_to_slides

    def _convert_emu_to_pixels(self, emu_value: Optional[str]) -> Optional[int]:
        """
        Convert EMU (English Metric Units) to pixels.

        Rule #1: Early return for None
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            emu_value: EMU value as string

        Returns:
            Pixel value or None if conversion fails
        """
        if not emu_value:
            return None

        try:
            emu = int(emu_value)
            # 914400 EMUs = 1 inch, assume 96 DPI
            return int(emu / 914400 * 96)
        except ValueError:
            return None

    def _extract_pic_dimensions(self, pic: Any, metadata: Dict[str, Any]) -> None:
        """
        Extract picture dimensions from XML element.

        Rule #1: Reduced nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pic: Picture XML element
            metadata: Metadata dictionary to update
        """
        # Try to get dimensions from extent element
        extent = pic.find(".//a:ext", NAMESPACES)
        if extent is not None:
            cx = extent.get("cx")
            cy = extent.get("cy")

            if cx:
                metadata["width"] = self._convert_emu_to_pixels(cx)
            if cy:
                metadata["height"] = self._convert_emu_to_pixels(cy)

    def _extract_pic_metadata(self, pic: Any, metadata: Dict[str, Any]) -> None:
        """
        Extract picture metadata (alt text, title) from XML element.

        Rule #1: Reduced nesting with early checks
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pic: Picture XML element
            metadata: Metadata dictionary to update
        """
        # Extract alt text
        cnvpr = pic.find(".//p:cNvPr", NAMESPACES)
        if cnvpr is not None:
            metadata["alt_text"] = cnvpr.get("descr", "")
            metadata["title"] = cnvpr.get("name", "")

    def _get_image_metadata(
        self, pptx: zipfile.ZipFile, slide_nums: list[int], media_path: str
    ) -> Dict[str, Any]:
        """
        Get metadata for an image by examining slide XML.

        Rule #1: Reduced nesting with early returns
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            pptx: Open PPTX archive
            slide_nums: List of slide numbers containing this image
            media_path: Path to media file

        Returns:
            Metadata dictionary with width, height, alt_text, title
        """
        metadata: Dict[str, Any] = {
            "width": None,
            "height": None,
            "alt_text": "",
            "title": "",
        }
        if not slide_nums:
            return metadata

        # Try to get metadata from first slide containing the image
        slide_num = slide_nums[0]
        slide_path = f"ppt/slides/slide{slide_num}.xml"
        if slide_path not in pptx.namelist():
            return metadata

        try:
            slide_xml = pptx.read(slide_path)
            slide_tree = ET.fromstring(slide_xml)

            # Find all picture elements
            for pic in slide_tree.findall(".//p:pic", NAMESPACES):
                self._extract_pic_dimensions(pic, metadata)
                self._extract_pic_metadata(pic, metadata)
                if metadata["width"] or metadata["alt_text"]:
                    break

        except ET.ParseError as e:
            logger.debug(f"Failed to parse slide {slide_num} XML: {e}")

        return metadata
