"""Image Cropping Utilities for VLM Escalation.

Provides functions to crop regions from images for VLM processing."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from ingestforge.core.logging import get_logger
from ingestforge.ingest.ocr.spatial_parser import BoundingBox

logger = get_logger(__name__)
MAX_CROP_DIMENSION = 4096
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB


def crop_image_region(
    image_path: Path,
    bbox: BoundingBox,
    padding: int = 20,
) -> Optional[bytes]:
    """Crop a region from an image file.

    Args:
        image_path: Path to source image
        bbox: Bounding box to crop
        padding: Padding to add around bbox (pixels)

    Returns:
        Cropped image as PNG bytes, or None on error
    """
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return None

    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL/Pillow not available for image cropping")
        return None

    try:
        # Load image
        with Image.open(image_path) as img:
            return crop_pil_image(img, bbox, padding)
    except Exception as e:
        logger.error(f"Failed to crop image: {e}")
        return None


def crop_pil_image(
    image,
    bbox: BoundingBox,
    padding: int = 20,
) -> Optional[bytes]:
    """Crop a region from a PIL Image.

    Args:
        image: PIL Image object
        bbox: Bounding box to crop
        padding: Padding to add around bbox (pixels)

    Returns:
        Cropped image as PNG bytes, or None on error
    """
    if image is None:
        logger.error("No image provided")
        return None

    # Calculate crop box with padding
    img_width, img_height = image.size

    x1 = max(0, bbox.x1 - padding)
    y1 = max(0, bbox.y1 - padding)
    x2 = min(img_width, bbox.x2 + padding)
    y2 = min(img_height, bbox.y2 + padding)
    crop_width = x2 - x1
    crop_height = y2 - y1

    if crop_width > MAX_CROP_DIMENSION or crop_height > MAX_CROP_DIMENSION:
        logger.warning(f"Crop too large: {crop_width}x{crop_height}, clamping")
        if crop_width > MAX_CROP_DIMENSION:
            x2 = x1 + MAX_CROP_DIMENSION
        if crop_height > MAX_CROP_DIMENSION:
            y2 = y1 + MAX_CROP_DIMENSION

    # Validate crop box
    if x1 >= x2 or y1 >= y2:
        logger.error(f"Invalid crop box: ({x1}, {y1}, {x2}, {y2})")
        return None

    try:
        # Crop image
        cropped = image.crop((x1, y1, x2, y2))

        # Convert to PNG bytes
        output = io.BytesIO()
        cropped.save(output, format="PNG")
        data = output.getvalue()
        if len(data) > MAX_IMAGE_SIZE_BYTES:
            logger.error(f"Cropped image too large: {len(data)} bytes")
            return None

        return data

    except Exception as e:
        logger.error(f"Failed to crop image: {e}")
        return None


def load_and_crop_regions(
    image_path: Path,
    regions: list[BoundingBox],
    padding: int = 20,
) -> list[Optional[bytes]]:
    """Crop multiple regions from an image.

    Args:
        image_path: Path to source image
        regions: List of bounding boxes to crop
        padding: Padding around each region

    Returns:
        List of cropped images as bytes (None for failed crops)
    """
    if not image_path.exists():
        logger.error(f"Image file not found: {image_path}")
        return [None] * len(regions)

    try:
        from PIL import Image
    except ImportError:
        logger.error("PIL/Pillow not available")
        return [None] * len(regions)

    try:
        # Load image once
        with Image.open(image_path) as img:
            results = []
            for bbox in regions:
                cropped = crop_pil_image(img, bbox, padding)
                results.append(cropped)
            return results
    except Exception as e:
        logger.error(f"Failed to load image: {e}")
        return [None] * len(regions)
