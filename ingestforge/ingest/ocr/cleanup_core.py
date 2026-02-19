"""Image deskew and binarization for OCR preprocessing (OCR-002.1).

Provides automatic image quality improvement for scanned documents:
- Deskew (auto-rotation correction)
- Binarization (noise removal and contrast enhancement)
- Border removal

NASA JPL Commandments compliance:
- Rule #1: Simple control flow, no deep nesting
- Rule #2: Fixed upper bounds on image dimensions
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.ingest.ocr.cleanup_core import (
        ImagePreprocessor,
        preprocess_image,
    )

    preprocessor = ImagePreprocessor()
    cleaned = preprocessor.process(image_path)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_IMAGE_DIMENSION = 10000  # Max pixels in any dimension
MIN_IMAGE_DIMENSION = 50  # Min pixels
MAX_ROTATION_DEGREES = 45  # Max auto-rotation
DEFAULT_THRESHOLD = 128  # Binarization threshold


class BinarizationMethod(Enum):
    """Methods for image binarization."""

    OTSU = "otsu"  # Otsu's automatic threshold
    ADAPTIVE = "adaptive"  # Adaptive local threshold
    FIXED = "fixed"  # Fixed threshold value
    SAUVOLA = "sauvola"  # Sauvola's local method


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing.

    Attributes:
        enable_deskew: Enable automatic rotation correction
        enable_binarize: Enable noise removal/contrast
        enable_denoise: Enable noise reduction
        enable_border_removal: Remove dark borders
        binarization_method: Method for binarization
        threshold: Fixed threshold value (if using FIXED method)
        adaptive_block_size: Block size for adaptive methods
        denoise_strength: Strength of denoising (0-1)
        target_dpi: Target DPI for rescaling (0 = no rescale)
    """

    enable_deskew: bool = True
    enable_binarize: bool = True
    enable_denoise: bool = True
    enable_border_removal: bool = True
    binarization_method: BinarizationMethod = BinarizationMethod.OTSU
    threshold: int = DEFAULT_THRESHOLD
    adaptive_block_size: int = 11
    denoise_strength: float = 0.5
    target_dpi: int = 0


@dataclass
class PreprocessingResult:
    """Result of image preprocessing.

    Attributes:
        success: Whether preprocessing succeeded
        original_path: Path to original image
        processed_image: Processed image array (or None)
        rotation_angle: Applied rotation in degrees
        was_binarized: Whether binarization was applied
        was_denoised: Whether denoising was applied
        original_size: Original image dimensions
        processed_size: Processed image dimensions
        error: Error message if failed
    """

    success: bool
    original_path: Path
    processed_image: Optional["numpy.ndarray"] = None
    rotation_angle: float = 0.0
    was_binarized: bool = False
    was_denoised: bool = False
    original_size: Tuple[int, int] = (0, 0)
    processed_size: Tuple[int, int] = (0, 0)
    error: str = ""


class ImagePreprocessor:
    """Preprocesses images for improved OCR quality.

    Applies deskew, binarization, and denoising to improve
    OCR accuracy on scanned documents.

    Args:
        config: Preprocessing configuration
    """

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        """Initialize the image preprocessor."""
        self.config = config or PreprocessingConfig()
        self._cv2: Optional[object] = None
        self._numpy: Optional[object] = None

    @property
    def cv2(self):
        """Lazy-load OpenCV."""
        if self._cv2 is not None:
            return self._cv2

        try:
            import cv2

            self._cv2 = cv2
            return cv2
        except ImportError:
            logger.warning(
                "OpenCV not installed. Install with: pip install opencv-python"
            )
            return None

    @property
    def numpy(self):
        """Lazy-load numpy."""
        if self._numpy is not None:
            return self._numpy

        try:
            import numpy as np

            self._numpy = np
            return np
        except ImportError:
            logger.warning("NumPy not installed.")
            return None

    @property
    def is_available(self) -> bool:
        """Check if preprocessing is available."""
        return self.cv2 is not None and self.numpy is not None

    def process(
        self,
        image_path: Union[str, Path],
    ) -> PreprocessingResult:
        """Process a single image.

        Args:
            image_path: Path to image file

        Returns:
            PreprocessingResult with processed image
        """
        path = Path(image_path)
        if not path.exists():
            return PreprocessingResult(
                success=False,
                original_path=path,
                error=f"File not found: {path}",
            )

        if not self.is_available:
            return PreprocessingResult(
                success=False,
                original_path=path,
                error="OpenCV not available",
            )

        # Load image
        image = self.cv2.imread(str(path), self.cv2.IMREAD_GRAYSCALE)
        if image is None:
            return PreprocessingResult(
                success=False,
                original_path=path,
                error=f"Failed to load image: {path}",
            )

        original_size = (image.shape[1], image.shape[0])
        if not self._validate_dimensions(image):
            return PreprocessingResult(
                success=False,
                original_path=path,
                original_size=original_size,
                error="Image dimensions out of bounds",
            )

        # Apply preprocessing steps
        rotation_angle = 0.0
        was_binarized = False
        was_denoised = False

        # Deskew
        if self.config.enable_deskew:
            image, rotation_angle = self._deskew(image)

        # Binarize
        if self.config.enable_binarize:
            image = self._binarize(image)
            was_binarized = True

        # Denoise
        if self.config.enable_denoise:
            image = self._denoise(image)
            was_denoised = True

        # Border removal
        if self.config.enable_border_removal:
            image = self._remove_borders(image)

        processed_size = (image.shape[1], image.shape[0])

        return PreprocessingResult(
            success=True,
            original_path=path,
            processed_image=image,
            rotation_angle=rotation_angle,
            was_binarized=was_binarized,
            was_denoised=was_denoised,
            original_size=original_size,
            processed_size=processed_size,
        )

    def _validate_dimensions(self, image) -> bool:
        """Validate image dimensions."""
        height, width = image.shape[:2]

        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False

        return True

    def _deskew(self, image) -> Tuple[object, float]:
        """Detect and correct image skew.

        Args:
            image: Input grayscale image

        Returns:
            Tuple of (corrected image, rotation angle)
        """
        np = self.numpy
        cv2 = self.cv2

        # Find edges using Canny
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None or len(lines) == 0:
            return image, 0.0

        # Calculate average angle
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

            # Only consider near-horizontal or near-vertical lines
            if abs(angle) < MAX_ROTATION_DEGREES:
                angles.append(angle)

        if not angles:
            return image, 0.0

        # Use median angle for robustness
        median_angle = float(np.median(angles))
        if abs(median_angle) > MAX_ROTATION_DEGREES:
            return image, 0.0

        # Small angles only - skip negligible rotations
        if abs(median_angle) < 0.5:
            return image, 0.0

        # Rotate image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)

        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated, median_angle

    def _binarize(self, image) -> object:
        """Apply binarization to image.

        Args:
            image: Input grayscale image

        Returns:
            Binarized image
        """
        cv2 = self.cv2
        method = self.config.binarization_method

        if method == BinarizationMethod.OTSU:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary

        if method == BinarizationMethod.ADAPTIVE:
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.config.adaptive_block_size,
                2,
            )

        if method == BinarizationMethod.FIXED:
            _, binary = cv2.threshold(
                image, self.config.threshold, 255, cv2.THRESH_BINARY
            )
            return binary

        # Default to Otsu
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def _denoise(self, image) -> object:
        """Apply denoising to image.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        cv2 = self.cv2

        # Calculate filter strength based on config
        h = int(10 * self.config.denoise_strength)
        h = max(1, min(h, 15))  # Clamp to valid range

        return cv2.fastNlMeansDenoising(image, None, h, 7, 21)

    def _remove_borders(self, image) -> object:
        """Remove dark borders from image.

        Args:
            image: Input image

        Returns:
            Image with borders removed
        """
        np = self.numpy
        cv2 = self.cv2

        # Find contours
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        # Find largest contour (assumed to be content area)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # Add small padding
        padding = 5
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)

        return image[y : y + h, x : x + w]

    def save_result(
        self,
        result: PreprocessingResult,
        output_path: Union[str, Path],
    ) -> bool:
        """Save preprocessed image to file.

        Args:
            result: Preprocessing result
            output_path: Path to save to

        Returns:
            True if successful
        """
        if not result.success or result.processed_image is None:
            return False

        try:
            self.cv2.imwrite(str(output_path), result.processed_image)
            return True
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            return False


def preprocess_image(
    image_path: Union[str, Path],
    config: Optional[PreprocessingConfig] = None,
) -> PreprocessingResult:
    """Convenience function to preprocess a single image.

    Args:
        image_path: Path to image
        config: Optional configuration

    Returns:
        PreprocessingResult
    """
    preprocessor = ImagePreprocessor(config)
    return preprocessor.process(image_path)


def preprocess_batch(
    image_paths: List[Union[str, Path]],
    config: Optional[PreprocessingConfig] = None,
) -> List[PreprocessingResult]:
    """Preprocess multiple images.

    Args:
        image_paths: List of image paths
        config: Optional configuration

    Returns:
        List of PreprocessingResult
    """
    preprocessor = ImagePreprocessor(config)
    return [preprocessor.process(path) for path in image_paths]
