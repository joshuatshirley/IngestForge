"""Tests for cleanup_core module (OCR-002.1).

Tests the image preprocessing pipeline:
- Configuration validation
- Preprocessing result structure
- Binarization methods
- Dimension validation
"""

from pathlib import Path
from unittest.mock import Mock

from ingestforge.ingest.ocr.cleanup_core import (
    BinarizationMethod,
    PreprocessingConfig,
    PreprocessingResult,
    ImagePreprocessor,
    preprocess_image,
    preprocess_batch,
    MAX_IMAGE_DIMENSION,
    MIN_IMAGE_DIMENSION,
    MAX_ROTATION_DEGREES,
)


class TestBinarizationMethod:
    """Test BinarizationMethod enum."""

    def test_all_methods_defined(self) -> None:
        """All methods should be defined."""
        assert BinarizationMethod.OTSU
        assert BinarizationMethod.ADAPTIVE
        assert BinarizationMethod.FIXED
        assert BinarizationMethod.SAUVOLA


class TestPreprocessingConfig:
    """Test PreprocessingConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config should have sensible values."""
        config = PreprocessingConfig()

        assert config.enable_deskew is True
        assert config.enable_binarize is True
        assert config.enable_denoise is True
        assert config.enable_border_removal is True
        assert config.binarization_method == BinarizationMethod.OTSU

    def test_custom_values(self) -> None:
        """Should accept custom values."""
        config = PreprocessingConfig(
            enable_deskew=False,
            binarization_method=BinarizationMethod.ADAPTIVE,
            threshold=150,
        )

        assert config.enable_deskew is False
        assert config.binarization_method == BinarizationMethod.ADAPTIVE
        assert config.threshold == 150


class TestPreprocessingResult:
    """Test PreprocessingResult dataclass."""

    def test_success_result(self) -> None:
        """Success result should have all fields."""
        result = PreprocessingResult(
            success=True,
            original_path=Path("test.png"),
            rotation_angle=2.5,
            was_binarized=True,
            was_denoised=True,
            original_size=(100, 200),
            processed_size=(100, 200),
        )

        assert result.success is True
        assert result.rotation_angle == 2.5
        assert result.was_binarized is True

    def test_failure_result(self) -> None:
        """Failure result should have error message."""
        result = PreprocessingResult(
            success=False,
            original_path=Path("test.png"),
            error="File not found",
        )

        assert result.success is False
        assert "not found" in result.error


class TestImagePreprocessorInit:
    """Test ImagePreprocessor initialization."""

    def test_default_init(self) -> None:
        """Should initialize with default config."""
        preprocessor = ImagePreprocessor()

        assert preprocessor.config is not None
        assert preprocessor.config.enable_deskew is True

    def test_custom_config(self) -> None:
        """Should accept custom config."""
        config = PreprocessingConfig(enable_deskew=False)
        preprocessor = ImagePreprocessor(config)

        assert preprocessor.config.enable_deskew is False


class TestImagePreprocessorAvailability:
    """Test availability checking."""

    def test_is_available_property(self) -> None:
        """Should report availability correctly."""
        preprocessor = ImagePreprocessor()

        # Result depends on OpenCV installation
        available = preprocessor.is_available
        assert isinstance(available, bool)


class TestImagePreprocessorValidation:
    """Test input validation."""

    def test_file_not_found(self) -> None:
        """Should handle missing file."""
        preprocessor = ImagePreprocessor()

        result = preprocessor.process(Path("/nonexistent/file.png"))

        assert result.success is False
        assert "not found" in result.error.lower()


class TestDimensionValidation:
    """Test dimension bounds (Rule #2)."""

    def test_max_dimension_constant(self) -> None:
        """Should have max dimension constant."""
        assert MAX_IMAGE_DIMENSION > 0
        assert MAX_IMAGE_DIMENSION <= 50000

    def test_min_dimension_constant(self) -> None:
        """Should have min dimension constant."""
        assert MIN_IMAGE_DIMENSION > 0
        assert MIN_IMAGE_DIMENSION < 100

    def test_max_rotation_constant(self) -> None:
        """Should have max rotation constant."""
        assert MAX_ROTATION_DEGREES > 0
        assert MAX_ROTATION_DEGREES <= 45


class TestPreprocessorWithMock:
    """Test preprocessor with mocked OpenCV."""

    def test_process_with_mock_cv2(self) -> None:
        """Should process with mocked OpenCV."""
        preprocessor = ImagePreprocessor()

        # Mock OpenCV
        mock_cv2 = Mock()
        mock_np = Mock()

        # Mock image loading
        mock_image = Mock()
        mock_image.shape = (100, 200)  # height, width
        mock_cv2.imread.return_value = mock_image
        mock_cv2.IMREAD_GRAYSCALE = 0

        # Mock processing functions
        mock_cv2.Canny.return_value = mock_image
        mock_cv2.HoughLinesP.return_value = None  # No lines found
        mock_cv2.threshold.return_value = (0, mock_image)
        mock_cv2.fastNlMeansDenoising.return_value = mock_image
        mock_cv2.findContours.return_value = ([], None)
        mock_cv2.THRESH_BINARY = 0
        mock_cv2.THRESH_OTSU = 8

        preprocessor._cv2 = mock_cv2
        preprocessor._numpy = mock_np

        # Mock numpy
        mock_np.pi = 3.14159

        # Create temp file for testing
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = preprocessor.process(temp_path)
            # Should attempt to load the image
            mock_cv2.imread.assert_called_once()
        finally:
            temp_path.unlink(missing_ok=True)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_preprocess_image_function(self) -> None:
        """Should work as standalone function."""
        result = preprocess_image(Path("/nonexistent.png"))

        # Should fail gracefully for missing file
        assert result.success is False

    def test_preprocess_batch_function(self) -> None:
        """Should process multiple images."""
        paths = [
            Path("/nonexistent1.png"),
            Path("/nonexistent2.png"),
        ]

        results = preprocess_batch(paths)

        assert len(results) == 2
        assert all(not r.success for r in results)


class TestConfigEnums:
    """Test configuration enum values."""

    def test_binarization_values(self) -> None:
        """Binarization methods should have string values."""
        assert BinarizationMethod.OTSU.value == "otsu"
        assert BinarizationMethod.ADAPTIVE.value == "adaptive"
        assert BinarizationMethod.FIXED.value == "fixed"
