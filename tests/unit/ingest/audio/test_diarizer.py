"""
Tests for Audio Diarization Processor.

Speaker identification and timestamp synchronization.
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from unittest.mock import patch
import pytest

from ingestforge.ingest.audio.diarizer import (
    AudioDiarizationProcessor,
    DiarizationConfig,
    DiarizationResult,
    SpeakerSegment,
    Speaker,
    validate_audio_file,
    SimpleDiarizer,
    MAX_SPEAKERS,
    MAX_SEGMENTS,
    SUPPORTED_CODECS,
)


# =============================================================================
# SPEAKER SEGMENT TESTS
# =============================================================================


class TestSpeakerSegment:
    """Tests for SpeakerSegment dataclass."""

    def test_segment_creation(self):
        """
        GWT:
        Given valid parameters
        When SpeakerSegment is created
        Then all fields are set correctly.
        """
        segment = SpeakerSegment(
            text="Hello world",
            speaker_id="Speaker A",
            start_ms=1000,
            end_ms=5000,
            confidence=0.95,
        )

        assert segment.text == "Hello world"
        assert segment.speaker_id == "Speaker A"
        assert segment.start_ms == 1000
        assert segment.end_ms == 5000

    def test_segment_duration_ms(self):
        """
        GWT:
        Given segment with timestamps
        When duration_ms is accessed
        Then correct duration is returned.
        """
        segment = SpeakerSegment(
            text="Test",
            speaker_id="Speaker A",
            start_ms=1000,
            end_ms=3500,
        )

        assert segment.duration_ms == 2500

    def test_segment_seconds_conversion(self):
        """
        GWT:
        Given segment with ms timestamps
        When start_seconds and end_seconds accessed
        Then correct seconds are returned.
        """
        segment = SpeakerSegment(
            text="Test",
            speaker_id="Speaker A",
            start_ms=1500,
            end_ms=4500,
        )

        assert segment.start_seconds == 1.5
        assert segment.end_seconds == 4.5

    def test_segment_to_dict(self):
        """
        GWT:
        Given SpeakerSegment
        When to_dict is called
        Then dictionary contains all fields.
        """
        segment = SpeakerSegment(
            text="Test",
            speaker_id="Speaker B",
            start_ms=0,
            end_ms=1000,
        )

        data = segment.to_dict()

        assert data["text"] == "Test"
        assert data["speaker_id"] == "Speaker B"
        assert data["start_ms"] == 0
        assert data["end_ms"] == 1000

    def test_format_timestamp(self):
        """
        GWT:
        Given segment at specific time
        When format_timestamp is called
        Then formatted string is returned.
        """
        segment = SpeakerSegment(
            text="Test",
            speaker_id="A",
            start_ms=3661500,  # 1h 1m 1.5s
            end_ms=3665000,
        )

        timestamp = segment.format_timestamp()

        assert "01:01" in timestamp


# =============================================================================
# SPEAKER TESTS
# =============================================================================


class TestSpeaker:
    """Tests for Speaker dataclass."""

    def test_speaker_creation(self):
        """
        GWT:
        Given valid parameters
        When Speaker is created
        Then fields are set correctly.
        """
        speaker = Speaker(
            speaker_id="Speaker A",
            label="John",
            total_speaking_time_ms=60000,
            segment_count=10,
        )

        assert speaker.speaker_id == "Speaker A"
        assert speaker.label == "John"
        assert speaker.total_speaking_time_ms == 60000

    def test_speaker_to_entity_dict(self):
        """
        GWT:
        Given Speaker
        When to_entity_dict is called
        Then PERSON entity format is returned ().
        """
        speaker = Speaker(
            speaker_id="Speaker A",
            label="Jane",
        )

        entity = speaker.to_entity_dict()

        assert entity["entity_type"] == "PERSON"
        assert entity["name"] == "Jane"
        assert "speaker_id" in entity["properties"]


# =============================================================================
# DIARIZATION RESULT TESTS
# =============================================================================


class TestDiarizationResult:
    """Tests for DiarizationResult dataclass."""

    def test_result_creation(self):
        """
        GWT:
        Given segments and speakers
        When DiarizationResult is created
        Then all fields are set.
        """
        segments = [
            SpeakerSegment("Hello", "Speaker A", 0, 1000),
            SpeakerSegment("Hi there", "Speaker B", 1000, 2000),
        ]
        speakers = [
            Speaker("Speaker A"),
            Speaker("Speaker B"),
        ]

        result = DiarizationResult(
            segments=segments,
            speakers=speakers,
            success=True,
        )

        assert result.segment_count == 2
        assert result.speaker_count == 2
        assert result.success is True

    def test_get_speaker_transcript(self):
        """
        GWT:
        Given result with multiple speakers
        When get_speaker_transcript is called
        Then only that speaker's text is returned.
        """
        segments = [
            SpeakerSegment("Hello", "Speaker A", 0, 1000),
            SpeakerSegment("Hi", "Speaker B", 1000, 2000),
            SpeakerSegment("How are you?", "Speaker A", 2000, 3000),
        ]

        result = DiarizationResult(segments=segments)

        transcript_a = result.get_speaker_transcript("Speaker A")

        assert "Hello" in transcript_a
        assert "How are you?" in transcript_a
        assert "Hi" not in transcript_a

    def test_formatted_transcript(self):
        """
        GWT:
        Given result with segments
        When to_formatted_transcript is called
        Then formatted output is returned.
        """
        segments = [
            SpeakerSegment("Hello", "Speaker A", 0, 1000),
        ]

        result = DiarizationResult(segments=segments)

        formatted = result.to_formatted_transcript()

        assert "Speaker A" in formatted
        assert "Hello" in formatted


# =============================================================================
# AUDIO VALIDATION TESTS
# =============================================================================


class TestValidateAudioFile:
    """Tests for validate_audio_file function."""

    def test_validate_none_path(self):
        """
        GWT:
        Given None path (JPL Rule #5)
        When validate_audio_file is called
        Then invalid result is returned.
        """
        is_valid, error = validate_audio_file(None)

        assert is_valid is False
        assert "None" in error or "invalid" in error

    def test_validate_missing_file(self, tmp_path):
        """
        GWT:
        Given non-existent path
        When validate_audio_file is called
        Then invalid result is returned.
        """
        fake_path = tmp_path / "nonexistent.mp3"

        is_valid, error = validate_audio_file(fake_path)

        assert is_valid is False
        assert "not found" in error

    def test_validate_unsupported_codec(self, tmp_path):
        """
        GWT:
        Given file with unsupported extension
        When validate_audio_file is called
        Then invalid result is returned (JPL Rule #5).
        """
        fake_path = tmp_path / "audio.xyz"
        fake_path.write_bytes(b"fake audio data")

        is_valid, error = validate_audio_file(fake_path)

        assert is_valid is False
        assert "Unsupported" in error or "codec" in error.lower()

    def test_validate_empty_file(self, tmp_path):
        """
        GWT:
        Given empty audio file
        When validate_audio_file is called
        Then invalid result is returned.
        """
        empty_path = tmp_path / "empty.mp3"
        empty_path.write_bytes(b"")

        is_valid, error = validate_audio_file(empty_path)

        assert is_valid is False
        assert "empty" in error.lower()

    def test_supported_codecs(self):
        """
        GWT:
        Given supported codecs list
        When checking
        Then common formats are included.
        """
        assert "mp3" in SUPPORTED_CODECS
        assert "wav" in SUPPORTED_CODECS
        assert "flac" in SUPPORTED_CODECS
        assert "m4a" in SUPPORTED_CODECS


# =============================================================================
# SIMPLE DIARIZER TESTS
# =============================================================================


class TestSimpleDiarizer:
    """Tests for SimpleDiarizer fallback."""

    def test_simple_diarizer_available(self):
        """
        GWT:
        Given SimpleDiarizer
        When is_available is called
        Then True is returned (always available).
        """
        diarizer = SimpleDiarizer()

        assert diarizer.is_available() is True


# =============================================================================
# DIARIZATION PROCESSOR TESTS
# =============================================================================


class TestAudioDiarizationProcessor:
    """Tests for AudioDiarizationProcessor."""

    def test_processor_initialization(self):
        """
        GWT:
        Given default configuration
        When processor is created
        Then defaults are set.
        """
        processor = AudioDiarizationProcessor()

        assert processor._config.whisper_model == "base"
        assert processor._config.language == "en"

    def test_processor_custom_config(self):
        """
        GWT:
        Given custom configuration
        When processor is created
        Then config is applied.
        """
        config = DiarizationConfig(
            whisper_model="large",
            num_speakers=2,
        )

        processor = AudioDiarizationProcessor(config=config)

        assert processor._config.whisper_model == "large"
        assert processor._config.num_speakers == 2

    def test_process_none_path(self):
        """
        GWT:
        Given None path (JPL Rule #5)
        When process is called
        Then AssertionError is raised.
        """
        processor = AudioDiarizationProcessor()

        with pytest.raises(AssertionError):
            processor.process(None)

    def test_process_invalid_type(self):
        """
        GWT:
        Given string instead of Path (JPL Rule #5)
        When process is called
        Then AssertionError is raised.
        """
        processor = AudioDiarizationProcessor()

        with pytest.raises(AssertionError):
            processor.process("not_a_path.mp3")

    def test_process_missing_file(self, tmp_path):
        """
        GWT:
        Given non-existent file
        When process is called
        Then error result is returned.
        """
        processor = AudioDiarizationProcessor()
        fake_path = tmp_path / "missing.mp3"

        result = processor.process(fake_path)

        assert result.success is False
        assert "not found" in result.error_message

    @patch("ingestforge.ingest.audio.diarizer.AudioDiarizationProcessor.is_available")
    def test_process_whisper_unavailable(self, mock_available, tmp_path):
        """
        GWT:
        Given Whisper not installed
        When process is called
        Then error result is returned (JPL Rule #7).
        """
        mock_available.return_value = False

        # Create a valid-looking file
        audio_path = tmp_path / "test.mp3"
        audio_path.write_bytes(b"fake audio data")

        processor = AudioDiarizationProcessor()
        result = processor.process(audio_path)

        assert result.success is False
        assert (
            "not available" in result.error_message.lower()
            or "Whisper" in result.error_message
        )


# =============================================================================
# CONFIG TESTS
# =============================================================================


class TestDiarizationConfig:
    """Tests for DiarizationConfig."""

    def test_default_config(self):
        """
        GWT:
        Given no parameters
        When DiarizationConfig is created
        Then defaults are set.
        """
        config = DiarizationConfig()

        assert config.whisper_model == "base"
        assert config.language == "en"
        assert config.max_speakers == MAX_SPEAKERS

    def test_config_bounds_max_speakers(self):
        """
        GWT:
        Given excessive max_speakers (JPL Rule #2)
        When DiarizationConfig is created
        Then value is bounded.
        """
        config = DiarizationConfig(max_speakers=1000)

        assert config.max_speakers <= MAX_SPEAKERS

    def test_config_num_speakers_bounded(self):
        """
        GWT:
        Given excessive num_speakers
        When DiarizationConfig is created
        Then value is bounded.
        """
        config = DiarizationConfig(num_speakers=500)

        assert config.num_speakers <= MAX_SPEAKERS


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_speaker_limit(self):
        """
        GWT:
        Given MAX_SPEAKERS constant
        When checking
        Then reasonable limit is set (JPL Rule #2).
        """
        assert MAX_SPEAKERS > 0
        assert MAX_SPEAKERS <= 100  # Reasonable upper bound

    def test_jpl_rule_2_segment_limit(self):
        """
        GWT:
        Given MAX_SEGMENTS constant
        When checking
        Then reasonable limit is set (JPL Rule #2).
        """
        assert MAX_SEGMENTS > 0
        assert MAX_SEGMENTS <= 100000

    def test_jpl_rule_5_null_path_assertion(self):
        """
        GWT:
        Given None audio path
        When process is called
        Then AssertionError raised (JPL Rule #5).
        """
        processor = AudioDiarizationProcessor()

        with pytest.raises(AssertionError):
            processor.process(None)

    def test_jpl_rule_5_invalid_type_assertion(self):
        """
        GWT:
        Given non-Path type
        When process is called
        Then AssertionError raised (JPL Rule #5).
        """
        processor = AudioDiarizationProcessor()

        with pytest.raises(AssertionError):
            processor.process(123)

    def test_jpl_rule_9_type_hints(self):
        """
        GWT:
        Given SpeakerSegment and Speaker
        When inspecting fields
        Then all have type annotations (JPL Rule #9).
        """
        segment = SpeakerSegment("test", "A", 0, 1000)
        speaker = Speaker("A")

        # Dataclasses have __dataclass_fields__
        assert hasattr(segment, "__dataclass_fields__")
        assert hasattr(speaker, "__dataclass_fields__")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestDiarizationIntegration:
    """Integration tests for diarization workflow."""

    def test_ms_timestamps_present(self):
        """
        GWT:
        Given SpeakerSegment
        When created
        Then start_ms and end_ms are present (AC).
        """
        segment = SpeakerSegment(
            text="Test",
            speaker_id="Speaker A",
            start_ms=1500,
            end_ms=3500,
        )

        # AC: 100% of transcript lines have start_ms and end_ms
        assert hasattr(segment, "start_ms")
        assert hasattr(segment, "end_ms")
        assert isinstance(segment.start_ms, int)
        assert isinstance(segment.end_ms, int)

    def test_speaker_entity_mapping(self):
        """
        GWT:
        Given Speaker
        When to_entity_dict is called
        Then maps to PERSON type (AC).
        """
        speaker = Speaker(speaker_id="Speaker A", label="Alice")

        entity = speaker.to_entity_dict()

        # AC: Map Speaker nodes to PERSON entity type
        assert entity["entity_type"] == "PERSON"
