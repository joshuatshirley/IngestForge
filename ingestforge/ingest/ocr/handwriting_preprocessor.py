"""Handwriting Pre-processor for OCR enhancement (P3-SOURCE-006.2).

Detects handwritten regions in scanned images and applies preprocessing
filters to improve OCR accuracy. Routes low-confidence handwritten text
to VLM escalator for enhanced extraction.
Usage:
    from ingestforge.ingest.ocr.handwriting_preprocessor import (
        HandwritingPreprocessor,
        preprocess_handwriting,
    )

    preprocessor = HandwritingPreprocessor()
    result = preprocessor.process(image_path)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union, TYPE_CHECKING

from ingestforge.core.logging import get_logger
from ingestforge.ingest.ocr.spatial_parser import BoundingBox, OCRElement, ElementType

if TYPE_CHECKING:
    import numpy

logger = get_logger(__name__)
MAX_IMAGE_DIMENSION = 10000
MIN_IMAGE_DIMENSION = 50
MAX_REGIONS_PER_IMAGE = 100
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
VLM_ESCALATION_THRESHOLD = 0.5
MORPHOLOGY_KERNEL_SIZE = 3
ADAPTIVE_BLOCK_SIZE = 11
ADAPTIVE_C_VALUE = 2


class HandwritingMethod(str, Enum):
    """Methods for handwriting detection and preprocessing."""

    CONTOUR = "contour"  # Contour-based region detection
    GRADIENT = "gradient"  # Gradient variance detection
    TEXTURE = "texture"  # Texture analysis


class ThresholdMethod(str, Enum):
    """Methods for image thresholding."""

    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"


@dataclass
class HandwritingRegion:
    """A detected handwritten region in an image."""

    bbox: BoundingBox
    confidence: float = 0.0
    is_handwritten: bool = False
    preprocessed_data: Optional[bytes] = None
    extracted_text: str = ""
    needs_vlm_escalation: bool = False

    @property
    def requires_escalation(self) -> bool:
        """Check if region needs VLM escalation."""
        return self.needs_vlm_escalation or self.confidence < VLM_ESCALATION_THRESHOLD


@dataclass
class PreprocessorConfig:
    """Configuration for handwriting preprocessor.

    Attributes:
        enable_denoising: Apply noise reduction
        enable_binarization: Apply thresholding
        enable_contrast_enhancement: Apply CLAHE
        enable_deskewing: Correct document skew
        enable_morphology: Apply morphological operations
        threshold_method: Method for binarization
        detection_method: Method for handwriting detection
        vlm_threshold: Confidence below which to escalate to VLM
        denoise_strength: Denoising filter strength (1-15)
    """

    enable_denoising: bool = True
    enable_binarization: bool = True
    enable_contrast_enhancement: bool = True
    enable_deskewing: bool = True
    enable_morphology: bool = True
    threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE_GAUSSIAN
    detection_method: HandwritingMethod = HandwritingMethod.CONTOUR
    vlm_threshold: float = VLM_ESCALATION_THRESHOLD
    denoise_strength: int = 10


@dataclass
class PreprocessorResult:
    """Result of handwriting preprocessing.

    Attributes:
        success: Whether preprocessing succeeded
        original_path: Path to original image
        regions: Detected handwriting regions
        processed_image: Full processed image array
        escalation_candidates: Regions needing VLM escalation
        error: Error message if failed
    """

    success: bool
    original_path: Path
    regions: List[HandwritingRegion] = field(default_factory=list)
    processed_image: Optional["numpy.ndarray"] = None
    escalation_candidates: List[HandwritingRegion] = field(default_factory=list)
    error: str = ""

    @property
    def has_escalation_candidates(self) -> bool:
        """Check if any regions need VLM escalation."""
        return len(self.escalation_candidates) > 0

    @property
    def region_count(self) -> int:
        """Get count of detected regions."""
        return len(self.regions)


class HandwritingPreprocessor:
    """Preprocesses images to detect and enhance handwritten regions.

    Applies a pipeline of image processing steps to improve OCR
    accuracy on handwritten text. Routes low-confidence regions
    to VLM for enhanced extraction.

    Args:
        config: Preprocessing configuration
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None) -> None:
        """Initialize the handwriting preprocessor."""
        self.config = config or PreprocessorConfig()
        self._cv2 = None
        self._numpy = None

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
        """Check if preprocessing dependencies are available."""
        return self.cv2 is not None and self.numpy is not None

    def process(self, image_path: Union[str, Path]) -> PreprocessorResult:
        """Process an image for handwriting detection and enhancement.

        Args:
            image_path: Path to image file

        Returns:
            PreprocessorResult with detected regions and processed image
        """
        path = Path(image_path)
        if not path.exists():
            return PreprocessorResult(
                success=False,
                original_path=path,
                error=f"File not found: {path}",
            )

        if not self.is_available:
            return PreprocessorResult(
                success=False,
                original_path=path,
                error="OpenCV not available",
            )

        # Load image
        image = self.cv2.imread(str(path))
        if image is None:
            return PreprocessorResult(
                success=False,
                original_path=path,
                error=f"Failed to load image: {path}",
            )

        # Validate dimensions
        if not self._validate_dimensions(image):
            return PreprocessorResult(
                success=False,
                original_path=path,
                error="Image dimensions out of bounds",
            )

        return self._process_image(path, image)

    def process_array(
        self,
        image: "numpy.ndarray",
        source_path: Optional[Path] = None,
    ) -> PreprocessorResult:
        """Process a numpy array image.

        Args:
            image: Image as numpy array
            source_path: Optional source path for reference

        Returns:
            PreprocessorResult
        """
        path = source_path or Path("memory_image")

        if not self.is_available:
            return PreprocessorResult(
                success=False,
                original_path=path,
                error="OpenCV not available",
            )

        if image is None:
            return PreprocessorResult(
                success=False,
                original_path=path,
                error="No image provided",
            )

        if not self._validate_dimensions(image):
            return PreprocessorResult(
                success=False,
                original_path=path,
                error="Image dimensions out of bounds",
            )

        return self._process_image(path, image)

    def _process_image(
        self,
        path: Path,
        image: "numpy.ndarray",
    ) -> PreprocessorResult:
        """Process image through preprocessing pipeline.

        Args:
            path: Image path for reference
            image: Image array

        Returns:
            PreprocessorResult
        """
        # Convert to grayscale
        gray = self._to_grayscale(image)

        # Apply preprocessing pipeline
        processed = self._apply_pipeline(gray)

        # Detect handwriting regions
        regions = self._detect_handwriting_regions(processed, gray)

        # Identify escalation candidates
        escalation = [r for r in regions if r.requires_escalation]

        return PreprocessorResult(
            success=True,
            original_path=path,
            regions=regions,
            processed_image=processed,
            escalation_candidates=escalation,
        )

    def _validate_dimensions(self, image: "numpy.ndarray") -> bool:
        """Validate image dimensions are within bounds."""
        height, width = image.shape[:2]

        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False

        return True

    def _to_grayscale(self, image: "numpy.ndarray") -> "numpy.ndarray":
        """Convert image to grayscale.

        Args:
            image: Input image (BGR or grayscale)

        Returns:
            Grayscale image
        """
        cv2 = self.cv2

        if len(image.shape) == 2:
            return image

        if len(image.shape) == 3 and image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def _apply_pipeline(self, gray: "numpy.ndarray") -> "numpy.ndarray":
        """Apply preprocessing pipeline to grayscale image.

        Args:
            gray: Grayscale image

        Returns:
            Processed image
        """
        processed = gray.copy()

        # Contrast enhancement (CLAHE)
        if self.config.enable_contrast_enhancement:
            processed = self._enhance_contrast(processed)

        # Denoising
        if self.config.enable_denoising:
            processed = self._denoise(processed)

        # Binarization
        if self.config.enable_binarization:
            processed = self._binarize(processed)

        # Morphological operations
        if self.config.enable_morphology:
            processed = self._apply_morphology(processed)

        # Deskewing
        if self.config.enable_deskewing:
            processed = self._deskew(processed)

        return processed

    def _enhance_contrast(self, image: "numpy.ndarray") -> "numpy.ndarray":
        """Apply CLAHE contrast enhancement.

        Args:
            image: Grayscale image

        Returns:
            Contrast-enhanced image
        """
        cv2 = self.cv2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    def _denoise(self, image: "numpy.ndarray") -> "numpy.ndarray":
        """Apply noise reduction.

        Args:
            image: Input image

        Returns:
            Denoised image
        """
        cv2 = self.cv2

        strength = max(1, min(self.config.denoise_strength, 15))
        return cv2.fastNlMeansDenoising(image, None, strength, 7, 21)

    def _binarize(self, image: "numpy.ndarray") -> "numpy.ndarray":
        """Apply adaptive thresholding.

        Args:
            image: Grayscale image

        Returns:
            Binary image
        """
        cv2 = self.cv2
        method = self.config.threshold_method

        if method == ThresholdMethod.OTSU:
            _, binary = cv2.threshold(
                image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            return binary

        if method == ThresholdMethod.ADAPTIVE_MEAN:
            return cv2.adaptiveThreshold(
                image,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                ADAPTIVE_BLOCK_SIZE,
                ADAPTIVE_C_VALUE,
            )

        # Default: adaptive gaussian
        return cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C_VALUE,
        )

    def _apply_morphology(self, image: "numpy.ndarray") -> "numpy.ndarray":
        """Apply morphological operations for noise cleanup.

        Args:
            image: Binary image

        Returns:
            Cleaned image
        """
        cv2 = self.cv2
        np = self.numpy

        kernel = np.ones((MORPHOLOGY_KERNEL_SIZE, MORPHOLOGY_KERNEL_SIZE), np.uint8)

        # Opening: removes small noise
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

        # Closing: fills small gaps in text
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)

        return closed

    def _deskew(self, image: "numpy.ndarray") -> "numpy.ndarray":
        """Detect and correct image skew.

        Args:
            image: Input image

        Returns:
            Deskewed image
        """
        cv2 = self.cv2
        np = self.numpy

        # Find non-zero points
        coords = np.column_stack(np.where(image > 0))
        if len(coords) < 10:
            return image

        # Compute minimum area rectangle
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]

        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        elif angle > 45:
            angle = angle - 90

        # Skip small angles
        if abs(angle) < 0.5:
            return image

        # Clamp rotation
        angle = max(-15, min(15, angle))

        # Rotate image
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        return rotated

    def _detect_handwriting_regions(
        self,
        processed: "numpy.ndarray",
        original: "numpy.ndarray",
    ) -> List[HandwritingRegion]:
        """Detect handwritten regions in the image.

        Args:
            processed: Preprocessed binary image
            original: Original grayscale image

        Returns:
            List of detected regions
        """
        method = self.config.detection_method

        if method == HandwritingMethod.CONTOUR:
            return self._detect_by_contour(processed, original)

        if method == HandwritingMethod.GRADIENT:
            return self._detect_by_gradient(processed, original)

        return self._detect_by_contour(processed, original)

    def _detect_by_contour(
        self,
        binary: "numpy.ndarray",
        gray: "numpy.ndarray",
    ) -> List[HandwritingRegion]:
        """Detect regions using contour analysis.

        Args:
            binary: Binary image
            gray: Original grayscale for confidence estimation

        Returns:
            List of handwriting regions
        """
        cv2 = self.cv2
        np = self.numpy

        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        regions: List[HandwritingRegion] = []
        height, width = binary.shape[:2]
        min_area = (width * height) * 0.001  # Min 0.1% of image

        for contour in contours[:MAX_REGIONS_PER_IMAGE]:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            bbox = BoundingBox(x1=x, y1=y, x2=x + w, y2=y + h)

            # Estimate confidence based on region characteristics
            confidence = self._estimate_region_confidence(binary, gray, bbox)
            is_handwritten = self._classify_as_handwriting(binary, bbox)

            region = HandwritingRegion(
                bbox=bbox,
                confidence=confidence,
                is_handwritten=is_handwritten,
                needs_vlm_escalation=confidence < self.config.vlm_threshold,
            )
            regions.append(region)

        return regions

    def _detect_by_gradient(
        self,
        binary: "numpy.ndarray",
        gray: "numpy.ndarray",
    ) -> List[HandwritingRegion]:
        """Detect regions using gradient variance.

        Args:
            binary: Binary image
            gray: Original grayscale

        Returns:
            List of handwriting regions
        """
        cv2 = self.cv2
        np = self.numpy

        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold gradient magnitude
        thresh = np.percentile(magnitude, 90)
        high_grad = (magnitude > thresh).astype(np.uint8) * 255

        # Find regions
        return self._detect_by_contour(high_grad, gray)

    def _estimate_region_confidence(
        self,
        binary: "numpy.ndarray",
        gray: "numpy.ndarray",
        bbox: BoundingBox,
    ) -> float:
        """Estimate OCR confidence for a region.

        Args:
            binary: Binary image
            gray: Grayscale image
            bbox: Region bounding box

        Returns:
            Confidence score (0.0 - 1.0)
        """
        np = self.numpy

        # Extract region
        region = gray[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        if region.size == 0:
            return 0.0

        # Factors affecting confidence:
        # 1. Contrast (std dev of intensity)
        contrast = np.std(region) / 127.5  # Normalize to 0-1

        # 2. Stroke consistency (uniformity of binary region)
        binary_region = binary[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        ink_ratio = np.sum(binary_region == 0) / binary_region.size

        # 3. Edge clarity
        edge_score = min(1.0, contrast * 1.5)

        # Combined confidence
        confidence = (0.4 * contrast) + (0.3 * ink_ratio) + (0.3 * edge_score)

        return min(1.0, max(0.0, confidence))

    def _classify_as_handwriting(
        self,
        binary: "numpy.ndarray",
        bbox: BoundingBox,
    ) -> bool:
        """Classify region as likely handwriting.

        Args:
            binary: Binary image
            bbox: Region bounding box

        Returns:
            True if region appears to be handwriting
        """
        cv2 = self.cv2
        np = self.numpy

        # Extract region
        region = binary[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2]
        if region.size == 0:
            return False

        # Handwriting characteristics:
        # 1. Stroke width variance (handwriting has more variation)
        skeleton = cv2.ximgproc.thinning(region) if hasattr(cv2, "ximgproc") else region
        stroke_variance = np.var(skeleton) if skeleton.size > 0 else 0

        # 2. Aspect ratio (handwriting often horizontal)
        aspect = bbox.width / max(1, bbox.height)
        is_horizontal = 0.3 < aspect < 10.0

        # 3. Density pattern (handwriting less uniform than print)
        density = np.sum(region == 0) / region.size
        has_typical_density = 0.05 < density < 0.7

        # Simple heuristic
        return is_horizontal and has_typical_density

    def extract_region_for_vlm(
        self,
        image_path: Path,
        region: HandwritingRegion,
        padding: int = 20,
    ) -> Optional[bytes]:
        """Extract region image data for VLM escalation.

        Args:
            image_path: Source image path
            region: Handwriting region to extract
            padding: Padding around region

        Returns:
            PNG bytes of extracted region, or None
        """
        from ingestforge.ingest.ocr.image_cropper import crop_image_region

        return crop_image_region(image_path, region.bbox, padding)

    def to_ocr_elements(
        self,
        result: PreprocessorResult,
    ) -> List[OCRElement]:
        """Convert preprocessing result to OCRElements.

        Args:
            result: Preprocessing result

        Returns:
            List of OCRElement objects
        """
        elements: List[OCRElement] = []

        for region in result.regions:
            element = OCRElement(
                element_type=ElementType.BLOCK,
                bbox=region.bbox,
                text=region.extracted_text,
                confidence=region.confidence,
                attributes={
                    "is_handwritten": region.is_handwritten,
                    "needs_escalation": region.needs_vlm_escalation,
                },
            )
            elements.append(element)

        return elements


def preprocess_handwriting(
    image_path: Union[str, Path],
    config: Optional[PreprocessorConfig] = None,
) -> PreprocessorResult:
    """Convenience function to preprocess image for handwriting.

    Args:
        image_path: Path to image
        config: Optional configuration

    Returns:
        PreprocessorResult
    """
    preprocessor = HandwritingPreprocessor(config)
    return preprocessor.process(image_path)


def detect_handwriting_regions(
    image_path: Union[str, Path],
    vlm_threshold: float = VLM_ESCALATION_THRESHOLD,
) -> List[HandwritingRegion]:
    """Detect handwriting regions in an image.

    Args:
        image_path: Path to image
        vlm_threshold: Confidence below which to mark for VLM

    Returns:
        List of detected handwriting regions
    """
    config = PreprocessorConfig(vlm_threshold=vlm_threshold)
    result = preprocess_handwriting(image_path, config)

    if not result.success:
        logger.warning(f"Handwriting detection failed: {result.error}")
        return []

    return result.regions


def get_vlm_escalation_candidates(
    image_path: Union[str, Path],
    threshold: float = VLM_ESCALATION_THRESHOLD,
) -> List[HandwritingRegion]:
    """Get handwriting regions that need VLM escalation.

    Args:
        image_path: Path to image
        threshold: Confidence threshold for escalation

    Returns:
        List of regions needing VLM escalation
    """
    config = PreprocessorConfig(vlm_threshold=threshold)
    result = preprocess_handwriting(image_path, config)

    if not result.success:
        logger.warning(f"Handwriting detection failed: {result.error}")
        return []

    return result.escalation_candidates
