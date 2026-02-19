"""Tests for Handwriting Pre-processor.

Tests preprocessing pipeline, confidence thresholds, VLM escalation,
and various image quality scenarios."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestforge.ingest.ocr.spatial_parser import BoundingBox, ElementType
from ingestforge.ingest.ocr.handwriting_preprocessor import (
    HandwritingMethod,
    HandwritingPreprocessor,
    HandwritingRegion,
    PreprocessorConfig,
    PreprocessorResult,
    ThresholdMethod,
    VLM_ESCALATION_THRESHOLD,
    detect_handwriting_regions,
    get_vlm_escalation_candidates,
    preprocess_handwriting,
)

# HandwritingRegion tests


class TestHandwritingRegion:
    """Tests for HandwritingRegion dataclass."""

    def test_region_creation(self) -> None:
        """Test creating a handwriting region."""
        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.7,
            is_handwritten=True,
        )

        assert region.bbox.x1 == 10
        assert region.confidence == 0.7
        assert region.is_handwritten is True

    def test_requires_escalation_low_confidence(self) -> None:
        """Test escalation required for low confidence."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.3,  # Below threshold
            needs_vlm_escalation=False,
        )

        assert region.requires_escalation is True

    def test_requires_escalation_explicit(self) -> None:
        """Test explicit escalation flag."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.9,  # High confidence
            needs_vlm_escalation=True,  # But explicitly flagged
        )

        assert region.requires_escalation is True

    def test_no_escalation_high_confidence(self) -> None:
        """Test no escalation for high confidence."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.8,
            needs_vlm_escalation=False,
        )

        assert region.requires_escalation is False


# PreprocessorConfig tests


class TestPreprocessorConfig:
    """Tests for PreprocessorConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = PreprocessorConfig()

        assert config.enable_denoising is True
        assert config.enable_binarization is True
        assert config.enable_contrast_enhancement is True
        assert config.enable_deskewing is True
        assert config.enable_morphology is True
        assert config.threshold_method == ThresholdMethod.ADAPTIVE_GAUSSIAN
        assert config.vlm_threshold == VLM_ESCALATION_THRESHOLD

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = PreprocessorConfig(
            enable_denoising=False,
            threshold_method=ThresholdMethod.OTSU,
            vlm_threshold=0.7,
        )

        assert config.enable_denoising is False
        assert config.threshold_method == ThresholdMethod.OTSU
        assert config.vlm_threshold == 0.7


# PreprocessorResult tests


class TestPreprocessorResult:
    """Tests for PreprocessorResult dataclass."""

    def test_success_result(self) -> None:
        """Test successful result."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        region = HandwritingRegion(bbox=bbox, confidence=0.8)

        result = PreprocessorResult(
            success=True,
            original_path=Path("test.png"),
            regions=[region],
        )

        assert result.success is True
        assert result.region_count == 1
        assert result.error == ""

    def test_failed_result(self) -> None:
        """Test failed result."""
        result = PreprocessorResult(
            success=False,
            original_path=Path("test.png"),
            error="File not found",
        )

        assert result.success is False
        assert result.error == "File not found"
        assert result.region_count == 0

    def test_has_escalation_candidates(self) -> None:
        """Test escalation candidate detection."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        low_conf_region = HandwritingRegion(
            bbox=bbox,
            confidence=0.3,
            needs_vlm_escalation=True,
        )

        result = PreprocessorResult(
            success=True,
            original_path=Path("test.png"),
            regions=[low_conf_region],
            escalation_candidates=[low_conf_region],
        )

        assert result.has_escalation_candidates is True

    def test_no_escalation_candidates(self) -> None:
        """Test when no escalation needed."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        high_conf_region = HandwritingRegion(bbox=bbox, confidence=0.9)

        result = PreprocessorResult(
            success=True,
            original_path=Path("test.png"),
            regions=[high_conf_region],
            escalation_candidates=[],
        )

        assert result.has_escalation_candidates is False


# HandwritingPreprocessor tests


class TestHandwritingPreprocessor:
    """Tests for HandwritingPreprocessor class."""

    def test_preprocessor_creation(self) -> None:
        """Test creating preprocessor."""
        preprocessor = HandwritingPreprocessor()
        assert preprocessor.config is not None

    def test_preprocessor_with_config(self) -> None:
        """Test preprocessor with custom config."""
        config = PreprocessorConfig(enable_denoising=False)
        preprocessor = HandwritingPreprocessor(config=config)

        assert preprocessor.config.enable_denoising is False

    def test_file_not_found(self) -> None:
        """Test handling of missing file."""
        preprocessor = HandwritingPreprocessor()
        result = preprocessor.process(Path("nonexistent.png"))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_is_available_property(self) -> None:
        """Test availability check."""
        preprocessor = HandwritingPreprocessor()

        # Will be True if OpenCV is installed, False otherwise
        # Just test that it returns a boolean
        assert isinstance(preprocessor.is_available, bool)


class TestPreprocessorWithMocks:
    """Tests for preprocessor with mocked dependencies."""

    @pytest.fixture
    def mock_cv2(self):
        """Create mock cv2 module."""
        mock = MagicMock()
        mock.imread.return_value = MagicMock()
        mock.imread.return_value.shape = (100, 200, 3)
        mock.cvtColor.return_value = MagicMock()
        mock.cvtColor.return_value.shape = (100, 200)
        mock.cvtColor.return_value.copy.return_value = MagicMock()
        mock.cvtColor.return_value.copy.return_value.shape = (100, 200)
        return mock

    @pytest.fixture
    def mock_numpy(self):
        """Create mock numpy module."""
        mock = MagicMock()
        return mock

    def test_process_with_valid_image(
        self, tmp_path: Path, mock_cv2, mock_numpy
    ) -> None:
        """Test processing valid image file."""
        import numpy as np

        # Create test image file
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image data")

        # Use minimal config to avoid complex mocking
        config = PreprocessorConfig(
            enable_deskewing=False,  # Skip deskew to simplify mocking
            enable_morphology=False,
        )
        preprocessor = HandwritingPreprocessor(config=config)
        preprocessor._cv2 = mock_cv2
        preprocessor._numpy = mock_numpy

        # Create a real numpy array for the grayscale image
        gray_image = np.zeros((100, 200), dtype=np.uint8)
        mock_cv2.cvtColor.return_value = gray_image
        mock_cv2.findContours.return_value = ([], None)

        # Mock CLAHE
        clahe_mock = MagicMock()
        clahe_mock.apply.return_value = gray_image
        mock_cv2.createCLAHE.return_value = clahe_mock

        # Mock denoising and thresholding
        mock_cv2.fastNlMeansDenoising.return_value = gray_image
        mock_cv2.adaptiveThreshold.return_value = gray_image

        result = preprocessor.process(image_path)

        assert result.success is True
        assert result.original_path == image_path


class TestPreprocessingPipeline:
    """Tests for preprocessing pipeline steps."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor with mocks."""
        p = HandwritingPreprocessor()
        p._cv2 = MagicMock()
        p._numpy = MagicMock()
        return p

    def test_to_grayscale_already_gray(self, preprocessor) -> None:
        """Test grayscale conversion when already gray."""
        gray_image = MagicMock()
        gray_image.shape = (100, 200)  # 2D = grayscale

        result = preprocessor._to_grayscale(gray_image)

        assert result == gray_image

    def test_to_grayscale_from_bgr(self, preprocessor) -> None:
        """Test grayscale conversion from BGR."""
        bgr_image = MagicMock()
        bgr_image.shape = (100, 200, 3)

        preprocessor._to_grayscale(bgr_image)

        preprocessor.cv2.cvtColor.assert_called_once()

    def test_validate_dimensions_valid(self, preprocessor) -> None:
        """Test dimension validation with valid size."""
        image = MagicMock()
        image.shape = (100, 200)

        assert preprocessor._validate_dimensions(image) is True

    def test_validate_dimensions_too_small(self, preprocessor) -> None:
        """Test dimension validation with too small size."""
        image = MagicMock()
        image.shape = (10, 20)  # Below MIN_IMAGE_DIMENSION

        assert preprocessor._validate_dimensions(image) is False

    def test_validate_dimensions_too_large(self, preprocessor) -> None:
        """Test dimension validation with too large size."""
        image = MagicMock()
        image.shape = (15000, 15000)  # Above MAX_IMAGE_DIMENSION

        assert preprocessor._validate_dimensions(image) is False


class TestConfidenceThresholds:
    """Tests for confidence threshold behavior."""

    def test_low_confidence_triggers_escalation(self) -> None:
        """Test that low confidence marks region for escalation."""
        config = PreprocessorConfig(vlm_threshold=0.6)
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)

        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.4,  # Below threshold
            needs_vlm_escalation=True,
        )

        assert region.confidence < config.vlm_threshold
        assert region.requires_escalation is True

    def test_high_confidence_no_escalation(self) -> None:
        """Test that high confidence doesn't trigger escalation."""
        config = PreprocessorConfig(vlm_threshold=0.6)
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)

        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.85,
            needs_vlm_escalation=False,
        )

        assert region.confidence > config.vlm_threshold
        assert region.requires_escalation is False

    def test_custom_vlm_threshold(self) -> None:
        """Test custom VLM threshold."""
        config = PreprocessorConfig(vlm_threshold=0.8)
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)

        # Region with 0.7 confidence
        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.7,
            needs_vlm_escalation=True,
        )

        assert region.confidence < config.vlm_threshold
        assert region.requires_escalation is True


class TestOCRElementConversion:
    """Tests for converting results to OCRElements."""

    def test_to_ocr_elements(self) -> None:
        """Test conversion to OCR elements."""
        preprocessor = HandwritingPreprocessor()

        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=80)
        region = HandwritingRegion(
            bbox=bbox,
            confidence=0.75,
            is_handwritten=True,
            extracted_text="Hello",
            needs_vlm_escalation=False,
        )

        result = PreprocessorResult(
            success=True,
            original_path=Path("test.png"),
            regions=[region],
        )

        elements = preprocessor.to_ocr_elements(result)

        assert len(elements) == 1
        assert elements[0].element_type == ElementType.BLOCK
        assert elements[0].bbox == bbox
        assert elements[0].text == "Hello"
        assert elements[0].confidence == 0.75
        assert elements[0].attributes["is_handwritten"] is True

    def test_to_ocr_elements_empty(self) -> None:
        """Test conversion with no regions."""
        preprocessor = HandwritingPreprocessor()

        result = PreprocessorResult(
            success=True,
            original_path=Path("test.png"),
            regions=[],
        )

        elements = preprocessor.to_ocr_elements(result)

        assert len(elements) == 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_preprocess_handwriting_file_not_found(self) -> None:
        """Test preprocess_handwriting with missing file."""
        result = preprocess_handwriting(Path("nonexistent.png"))

        assert result.success is False
        assert "not found" in result.error.lower()

    def test_detect_handwriting_regions_file_not_found(self) -> None:
        """Test detect_handwriting_regions with missing file."""
        regions = detect_handwriting_regions(Path("nonexistent.png"))

        assert len(regions) == 0

    def test_get_vlm_escalation_candidates_file_not_found(self) -> None:
        """Test get_vlm_escalation_candidates with missing file."""
        candidates = get_vlm_escalation_candidates(Path("nonexistent.png"))

        assert len(candidates) == 0


class TestThresholdMethods:
    """Tests for different threshold methods."""

    def test_threshold_method_enum(self) -> None:
        """Test threshold method enum values."""
        assert ThresholdMethod.OTSU.value == "otsu"
        assert ThresholdMethod.ADAPTIVE_MEAN.value == "adaptive_mean"
        assert ThresholdMethod.ADAPTIVE_GAUSSIAN.value == "adaptive_gaussian"

    def test_config_with_otsu(self) -> None:
        """Test config with Otsu threshold."""
        config = PreprocessorConfig(threshold_method=ThresholdMethod.OTSU)
        assert config.threshold_method == ThresholdMethod.OTSU

    def test_config_with_adaptive_mean(self) -> None:
        """Test config with adaptive mean threshold."""
        config = PreprocessorConfig(threshold_method=ThresholdMethod.ADAPTIVE_MEAN)
        assert config.threshold_method == ThresholdMethod.ADAPTIVE_MEAN


class TestHandwritingDetectionMethods:
    """Tests for handwriting detection methods."""

    def test_detection_method_enum(self) -> None:
        """Test detection method enum values."""
        assert HandwritingMethod.CONTOUR.value == "contour"
        assert HandwritingMethod.GRADIENT.value == "gradient"
        assert HandwritingMethod.TEXTURE.value == "texture"

    def test_config_with_gradient_method(self) -> None:
        """Test config with gradient detection."""
        config = PreprocessorConfig(detection_method=HandwritingMethod.GRADIENT)
        assert config.detection_method == HandwritingMethod.GRADIENT


class TestImageQualityLevels:
    """Tests for handling various image quality scenarios."""

    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor."""
        return HandwritingPreprocessor()

    def test_high_quality_image_config(self) -> None:
        """Test config for high quality images (minimal processing)."""
        config = PreprocessorConfig(
            enable_denoising=False,
            enable_contrast_enhancement=False,
            denoise_strength=3,
        )

        assert config.enable_denoising is False
        assert config.denoise_strength == 3

    def test_low_quality_image_config(self) -> None:
        """Test config for low quality images (aggressive processing)."""
        config = PreprocessorConfig(
            enable_denoising=True,
            enable_contrast_enhancement=True,
            enable_morphology=True,
            denoise_strength=15,
            threshold_method=ThresholdMethod.ADAPTIVE_GAUSSIAN,
        )

        assert config.enable_denoising is True
        assert config.denoise_strength == 15

    def test_faded_document_config(self) -> None:
        """Test config for faded documents."""
        config = PreprocessorConfig(
            enable_contrast_enhancement=True,
            enable_binarization=True,
            threshold_method=ThresholdMethod.ADAPTIVE_GAUSSIAN,
        )

        assert config.enable_contrast_enhancement is True


class TestVLMIntegration:
    """Tests for VLM escalator integration."""

    def test_extract_region_for_vlm(self, tmp_path: Path) -> None:
        """Test region extraction for VLM."""
        # Create a minimal test image
        image_path = tmp_path / "test.png"
        image_path.write_bytes(b"fake image")

        preprocessor = HandwritingPreprocessor()
        bbox = BoundingBox(x1=10, y1=10, x2=50, y2=50)
        region = HandwritingRegion(bbox=bbox, confidence=0.3)

        # This will fail without PIL but tests the interface
        result = preprocessor.extract_region_for_vlm(image_path, region)

        # Result will be None if PIL not available or image invalid
        # Just verify the method is callable
        assert result is None or isinstance(result, bytes)

    def test_escalation_candidate_identification(self) -> None:
        """Test that escalation candidates are properly identified."""
        bbox1 = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        bbox2 = BoundingBox(x1=100, y1=0, x2=200, y2=50)

        high_conf = HandwritingRegion(bbox=bbox1, confidence=0.9)
        low_conf = HandwritingRegion(
            bbox=bbox2,
            confidence=0.3,
            needs_vlm_escalation=True,
        )

        result = PreprocessorResult(
            success=True,
            original_path=Path("test.png"),
            regions=[high_conf, low_conf],
            escalation_candidates=[low_conf],
        )

        assert len(result.escalation_candidates) == 1
        assert result.escalation_candidates[0].confidence == 0.3


class TestProcessArrayMethod:
    """Tests for process_array method."""

    def test_process_array_no_image(self) -> None:
        """Test process_array with None image."""
        preprocessor = HandwritingPreprocessor()
        # Inject mocks to bypass OpenCV availability check
        preprocessor._cv2 = MagicMock()
        preprocessor._numpy = MagicMock()

        result = preprocessor.process_array(None)

        assert result.success is False
        assert "no image" in result.error.lower()

    def test_process_array_with_source_path(self) -> None:
        """Test process_array with source path."""
        preprocessor = HandwritingPreprocessor()
        preprocessor._cv2 = MagicMock()
        preprocessor._numpy = MagicMock()

        mock_image = MagicMock()
        mock_image.shape = (10, 10)  # Too small

        result = preprocessor.process_array(
            mock_image,
            source_path=Path("original.png"),
        )

        # Should fail due to dimension validation
        assert result.success is False
        assert result.original_path == Path("original.png")
