"""
Google Slides Image Extraction Mixin.

Handles extraction of images from Google Slides presentations via API.

This module is part of the slide_image_extractor refactoring (Sprint 3, Rule #4)
to reduce slide_image_extractor.py from 1,434 lines to <400 lines.
"""

import base64
from typing import Any, Callable, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class GoogleSlidesExtractionMixin:
    """
    Mixin providing Google Slides image extraction methods.

    Rule #4: Extracted from slide_image_extractor.py to reduce file size
    """

    def _detect_image_format(self, url: str) -> str:
        """
        Detect image format from URL.

        Rule #1: Simple loop with early break
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            url: Image URL

        Returns:
            Image format (png, jpg, jpeg, gif)
        """
        if not url:
            return "png"

        # Check for format in URL
        url_lower = url.lower()
        for fmt in ["png", "jpg", "jpeg", "gif"]:
            if f".{fmt}" in url_lower:
                return fmt

        return "png"

    def _fetch_google_slides_image_data(
        self, fetch_image_func: Optional[Callable], url: str
    ) -> tuple:
        """
        Fetch image data from Google Slides URL.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            fetch_image_func: Function to fetch image data
            url: Image URL

        Returns:
            Tuple of (data, data_base64)
        """
        data = None
        data_base64 = None
        if not (fetch_image_func and url):
            return (data, data_base64)

        try:
            data = fetch_image_func(url)

            # Encode as base64 if requested
            if data and self.include_base64:
                data_base64 = base64.b64encode(data).decode("utf-8")

            # Clear data if not needed
            if not self.include_data:
                data = None

        except Exception as e:
            logger.debug(f"Failed to fetch image from URL {url}: {e}")

        return (data, data_base64)

    def _create_google_slides_image(
        self,
        elem: dict,
        slide_num: int,
        url: str,
        img_format: str,
        data: Optional[bytes],
        data_base64: Optional[str],
        images_count: int,
    ):
        """
        Create ExtractedImage from Google Slides element.

        Rule #1: Simple construction
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            elem: Page element dictionary
            slide_num: Slide number
            url: Image URL
            img_format: Image format
            data: Raw image bytes
            data_base64: Base64-encoded data
            images_count: Current image count for ID generation

        Returns:
            ExtractedImage object
        """
        from ingestforge.ingest.slide_image_extractor import ExtractedImage

        image_props = elem.get("image", {})
        size = elem.get("size", {})
        width = size.get("width", {}).get("magnitude")
        height = size.get("height", {}).get("magnitude")
        obj_id = elem.get("objectId", f"slide{slide_num}_img{images_count}")

        return ExtractedImage(
            image_id=obj_id,
            slide_number=slide_num,
            name=f"slide_{slide_num}_image",
            format=img_format,
            width=int(width) if width else None,
            height=int(height) if height else None,
            data=data,
            data_base64=data_base64,
            alt_text=image_props.get("title", ""),
            title=image_props.get("description", ""),
            path_in_archive=url,
        )

    def _process_google_slides_element(
        self,
        elem: dict,
        slide_num: int,
        fetch_image_func: Optional[Callable],
        images_count: int,
    ):
        """
        Process single page element from Google Slides.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            elem: Page element dictionary
            slide_num: Slide number
            fetch_image_func: Function to fetch image data
            images_count: Current image count

        Returns:
            ExtractedImage or None if not an image
        """
        if "image" not in elem:
            return None

        image_props = elem.get("image", {})
        content_url = image_props.get("contentUrl", "")
        source_url = image_props.get("sourceUrl", "")
        url = content_url or source_url

        # Detect image format
        img_format = self._detect_image_format(url)

        # Fetch image data
        data, data_base64 = self._fetch_google_slides_image_data(fetch_image_func, url)

        # Create image object
        return self._create_google_slides_image(
            elem, slide_num, url, img_format, data, data_base64, images_count
        )

    def extract_from_google_slides(
        self,
        presentation_data: dict,
        fetch_image_func: Optional[Callable] = None,
    ) -> list[Any]:
        """
        Extract images from Google Slides presentation data.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            presentation_data: Parsed Google Slides API response
            fetch_image_func: Optional function to fetch image data from URL

        Returns:
            List of ExtractedImage objects
        """
        images: list[ExtractedImage] = []
        slides = presentation_data.get("slides", [])

        for i, slide in enumerate(slides):
            slide_num = i + 1
            page_elements = slide.get("pageElements", [])

            for elem in page_elements:
                image = self._process_google_slides_element(
                    elem, slide_num, fetch_image_func, len(images)
                )
                if image is None:
                    continue

                images.append(image)

        return images
