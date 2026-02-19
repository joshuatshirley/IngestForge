"""Tests for Whisper backend integration.

Tests audio transcription functionality."""

from __future__ import annotations

from pathlib import Path


from ingestforge.ingest.audio.whisper_backend import (
    WhisperBackend,
    WhisperModel,
    ComputeType,
    TranscriptionConfig,
    TranscriptionSegment,
    TranscriptionResult,
    WordInfo,
    create_whisper_backend,
    is_whisper_available,
    SUPPORTED_FORMATS,
    DEFAULT_MODEL,
)

# WhisperModel tests


class TestWhisperModel:
    """Tests for WhisperModel enum."""

    def test_models_defined(self) -> None:
        """Test all model sizes are defined."""
        models = [m.value for m in WhisperModel]

        assert "tiny" in models
        assert "base" in models
        assert "large" in models

    def test_model_count(self) -> None:
        """Test correct number of models."""
        assert len(WhisperModel) == 5


# ComputeType tests


class TestComputeType:
    """Tests for ComputeType enum."""

    def test_types_defined(self) -> None:
        """Test all compute types are defined."""
        types = [t.value for t in ComputeType]

        assert "int8" in types
        assert "float16" in types
        assert "auto" in types


# TranscriptionConfig tests


class TestTranscriptionConfig:
    """Tests for TranscriptionConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = TranscriptionConfig()

        assert config.model == DEFAULT_MODEL
        assert config.device == "cpu"
        assert config.word_timestamps is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = TranscriptionConfig(
            model="small",
            language="es",
            device="cuda",
        )

        assert config.model == "small"
        assert config.language == "es"


# TranscriptionSegment tests


class TestTranscriptionSegment:
    """Tests for TranscriptionSegment dataclass."""

    def test_segment_creation(self) -> None:
        """Test creating a segment."""
        segment = TranscriptionSegment(
            id=0,
            start=0.0,
            end=5.5,
            text="Hello world",
        )

        assert segment.id == 0
        assert segment.text == "Hello world"

    def test_segment_duration(self) -> None:
        """Test segment duration property."""
        segment = TranscriptionSegment(
            id=1,
            start=10.0,
            end=25.0,
            text="Test",
        )

        assert segment.duration == 15.0

    def test_segment_with_words(self) -> None:
        """Test segment with word info."""
        words = [
            WordInfo(word="Hello", start=0.0, end=0.5, probability=0.95),
            WordInfo(word="world", start=0.6, end=1.0, probability=0.90),
        ]
        segment = TranscriptionSegment(
            id=0,
            start=0.0,
            end=1.0,
            text="Hello world",
            words=words,
        )

        assert len(segment.words) == 2


# WordInfo tests


class TestWordInfo:
    """Tests for WordInfo dataclass."""

    def test_word_info_creation(self) -> None:
        """Test creating word info."""
        word = WordInfo(
            word="test",
            start=1.0,
            end=1.5,
            probability=0.9,
        )

        assert word.word == "test"
        assert word.probability == 0.9


# TranscriptionResult tests


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_default_result(self) -> None:
        """Test default result values."""
        result = TranscriptionResult(text="")

        assert result.text == ""
        assert result.segments == []

    def test_result_with_data(self) -> None:
        """Test result with transcription data."""
        result = TranscriptionResult(
            text="Hello world test",
            language="en",
            duration=10.5,
            model_used="base",
        )

        assert result.text == "Hello world test"
        assert result.language == "en"

    def test_word_count(self) -> None:
        """Test word count property."""
        result = TranscriptionResult(
            text="Hello world this is a test",
        )

        assert result.word_count == 6


# WhisperBackend tests


class TestWhisperBackend:
    """Tests for WhisperBackend class."""

    def test_backend_creation(self) -> None:
        """Test creating backend."""
        backend = WhisperBackend()

        assert backend.config is not None

    def test_backend_with_config(self) -> None:
        """Test backend with custom config."""
        config = TranscriptionConfig(model="small")
        backend = WhisperBackend(config=config)

        assert backend.config.model == "small"

    def test_is_available_check(self) -> None:
        """Test availability check."""
        backend = WhisperBackend()

        # Returns bool regardless of installation
        result = backend.is_available
        assert isinstance(result, bool)


class TestSupportedFormats:
    """Tests for supported format checking."""

    def test_supported_formats_exist(self) -> None:
        """Test that supported formats are defined."""
        assert len(SUPPORTED_FORMATS) > 0
        assert ".mp3" in SUPPORTED_FORMATS
        assert ".wav" in SUPPORTED_FORMATS

    def test_is_supported_format(self) -> None:
        """Test format checking."""
        backend = WhisperBackend()

        assert backend._is_supported_format(Path("test.mp3")) is True
        assert backend._is_supported_format(Path("test.wav")) is True
        assert backend._is_supported_format(Path("test.txt")) is False


class TestTranscribeValidation:
    """Tests for transcription input validation."""

    def test_transcribe_nonexistent_file(self) -> None:
        """Test transcribing non-existent file."""
        backend = WhisperBackend()

        result = backend.transcribe(Path("/nonexistent/audio.mp3"))

        assert result.text == ""

    def test_transcribe_unsupported_format(self, tmp_path: Path) -> None:
        """Test transcribing unsupported format."""
        backend = WhisperBackend()

        # Create dummy text file
        test_file = tmp_path / "test.txt"
        test_file.write_text("not audio")

        result = backend.transcribe(test_file)

        assert result.text == ""


class TestModelLoading:
    """Tests for model loading."""

    def test_model_not_loaded_initially(self) -> None:
        """Test that model is not loaded initially."""
        backend = WhisperBackend()

        assert backend._model_loaded is False
        assert backend._model is None

    def test_unload_model(self) -> None:
        """Test unloading model."""
        backend = WhisperBackend()
        backend._model_loaded = True
        backend._model = "dummy"

        backend.unload_model()

        assert backend._model_loaded is False
        assert backend._model is None


# Factory function tests


class TestCreateWhisperBackend:
    """Tests for create_whisper_backend factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        backend = create_whisper_backend()

        assert backend.config.model == DEFAULT_MODEL
        assert backend.config.device == "cpu"

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        backend = create_whisper_backend(
            model="small",
            device="cuda",
            language="es",
        )

        assert backend.config.model == "small"
        assert backend.config.device == "cuda"
        assert backend.config.language == "es"


class TestIsWhisperAvailable:
    """Tests for is_whisper_available function."""

    def test_returns_bool(self) -> None:
        """Test that function returns bool."""
        result = is_whisper_available()

        assert isinstance(result, bool)
