"""Tests for Audio Processor.

Tests high-level audio processing with Whisper transcription and chunking."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


from ingestforge.ingest.audio.processor import (
    AudioProcessor,
    AudioChunk,
    AudioMetadata,
    AudioProcessingResult,
    DEFAULT_CHUNK_DURATION_SECONDS,
    MIN_CHUNK_DURATION_SECONDS,
    MAX_CHUNK_DURATION_SECONDS,
)

# AudioChunk tests


class TestAudioChunk:
    """Tests for AudioChunk dataclass."""

    def test_chunk_creation(self) -> None:
        """Test creating an audio chunk."""
        chunk = AudioChunk(
            content="This is test content.",
            timestamp_start=0.0,
            timestamp_end=10.0,
        )

        assert chunk.content == "This is test content."
        assert chunk.timestamp_start == 0.0
        assert chunk.timestamp_end == 10.0
        assert chunk.word_count == 4

    def test_duration_property(self) -> None:
        """Test duration calculation."""
        chunk = AudioChunk(
            content="Test",
            timestamp_start=5.0,
            timestamp_end=15.5,
        )

        assert chunk.duration == 10.5

    def test_format_timestamp_minutes(self) -> None:
        """Test timestamp formatting for minutes."""
        chunk = AudioChunk(
            content="Test",
            timestamp_start=125.0,
            timestamp_end=130.0,
        )

        assert chunk.start_formatted == "2:05"

    def test_format_timestamp_hours(self) -> None:
        """Test timestamp formatting for hours."""
        chunk = AudioChunk(
            content="Test",
            timestamp_start=3665.0,  # 1:01:05
            timestamp_end=3670.0,
        )

        assert chunk.start_formatted == "1:01:05"


# AudioMetadata tests


class TestAudioMetadata:
    """Tests for AudioMetadata dataclass."""

    def test_metadata_creation(self) -> None:
        """Test creating audio metadata."""
        metadata = AudioMetadata(
            file_path="/path/to/audio.mp3",
            file_name="audio.mp3",
            duration=120.5,
            language="en",
        )

        assert metadata.file_name == "audio.mp3"
        assert metadata.duration == 120.5
        assert metadata.language == "en"


# AudioProcessingResult tests


class TestAudioProcessingResult:
    """Tests for AudioProcessingResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating processing result."""
        metadata = AudioMetadata(
            file_path="/path/to/test.mp3",
            file_name="test.mp3",
        )

        result = AudioProcessingResult(
            metadata=metadata,
            text="Test transcription text.",
            success=True,
        )

        assert result.success is True
        assert result.text == "Test transcription text."
        assert result.word_count == 3

    def test_chunk_count_property(self) -> None:
        """Test chunk count property."""
        metadata = AudioMetadata(
            file_path="/path/to/test.mp3",
            file_name="test.mp3",
        )

        chunks = [
            AudioChunk("Test 1", 0.0, 10.0),
            AudioChunk("Test 2", 10.0, 20.0),
        ]

        result = AudioProcessingResult(
            metadata=metadata,
            chunks=chunks,
        )

        assert result.chunk_count == 2


# AudioProcessor tests


class TestAudioProcessor:
    """Tests for AudioProcessor class."""

    def test_init_default_params(self) -> None:
        """Test initialization with default parameters."""
        processor = AudioProcessor()

        assert processor.chunk_duration == DEFAULT_CHUNK_DURATION_SECONDS
        assert processor.whisper_model == "base"
        assert processor.language == "en"

    def test_init_custom_params(self) -> None:
        """Test initialization with custom parameters."""
        processor = AudioProcessor(
            chunk_duration=120.0,
            whisper_model="small",
            language="es",
        )

        assert processor.chunk_duration == 120.0
        assert processor.whisper_model == "small"
        assert processor.language == "es"

    def test_chunk_duration_bounds(self) -> None:
        """Test chunk duration respects bounds."""
        # Too small
        processor = AudioProcessor(chunk_duration=5.0)
        assert processor.chunk_duration == MIN_CHUNK_DURATION_SECONDS

        # Too large
        processor = AudioProcessor(chunk_duration=500.0)
        assert processor.chunk_duration == MAX_CHUNK_DURATION_SECONDS

    def test_process_file_not_found(self) -> None:
        """Test processing non-existent file."""
        processor = AudioProcessor()
        result = processor.process(Path("/nonexistent/file.mp3"))

        assert result.success is False
        assert "not found" in result.error.lower()

    @patch("ingestforge.ingest.audio.is_whisper_available")
    def test_process_whisper_not_available(self, mock_available: MagicMock) -> None:
        """Test processing when Whisper is not available."""
        mock_available.return_value = False
        processor = AudioProcessor()

        # Create a temp file
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = Path(f.name)

        try:
            result = processor.process(temp_path)

            assert result.success is False
            assert "faster-whisper" in result.error
        finally:
            temp_path.unlink()

    def test_finalize_chunk(self) -> None:
        """Test chunk finalization."""
        processor = AudioProcessor()

        texts = ["Hello world.", "This is a test."]
        chunk = processor._finalize_chunk(texts, 0.0, 10.0, 2)

        assert chunk.content == "Hello world. This is a test."
        assert chunk.timestamp_start == 0.0
        assert chunk.timestamp_end == 10.0
        assert chunk.segment_count == 2

    def test_to_chunk_records(self) -> None:
        """Test converting to ChunkRecord objects."""
        processor = AudioProcessor()

        metadata = AudioMetadata(
            file_path="/path/to/audio.mp3",
            file_name="audio.mp3",
            duration=120.0,
            language="en",
            model_used="base",
        )

        chunks = [
            AudioChunk("First chunk content.", 0.0, 60.0),
            AudioChunk("Second chunk content.", 60.0, 120.0),
        ]

        result = AudioProcessingResult(
            metadata=metadata,
            chunks=chunks,
            text="First chunk content. Second chunk content.",
            success=True,
        )

        records = processor.to_chunk_records(result, "doc_001")

        assert len(records) == 2
        assert records[0].chunk_id == "doc_001_chunk_0000"
        assert records[1].chunk_id == "doc_001_chunk_0001"
        assert records[0].content == "First chunk content."
        assert records[0].chunk_type == "transcript"
        assert records[0].metadata["audio_file"] == "audio.mp3"
        assert records[0].metadata["timestamp_start"] == 0.0
        assert records[0].source_location.source_type.value == "audio"
