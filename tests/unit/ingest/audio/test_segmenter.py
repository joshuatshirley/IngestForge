"""Tests for audio segmenter.

Tests audio segmentation functionality."""

from __future__ import annotations

from pathlib import Path


from ingestforge.ingest.audio.segmenter import (
    AudioSegmenter,
    AudioSegment,
    SegmentConfig,
    SegmentationResult,
    create_segmenter,
    segment_audio,
    is_segmenter_available,
    MAX_SEGMENT_DURATION_SECONDS,
    MIN_SEGMENT_DURATION_SECONDS,
    MAX_SEGMENTS,
    DEFAULT_SEGMENT_DURATION,
)

# AudioSegment tests


class TestAudioSegment:
    """Tests for AudioSegment dataclass."""

    def test_segment_creation(self) -> None:
        """Test creating a segment."""
        segment = AudioSegment(
            index=0,
            start_time=0.0,
            end_time=60.0,
        )

        assert segment.index == 0
        assert segment.start_time == 0.0

    def test_segment_duration(self) -> None:
        """Test segment duration calculation."""
        segment = AudioSegment(
            index=1,
            start_time=10.0,
            end_time=70.0,
        )

        assert segment.duration == 60.0

    def test_segment_with_path(self) -> None:
        """Test segment with file path."""
        segment = AudioSegment(
            index=0,
            start_time=0.0,
            end_time=30.0,
            path=Path("/tmp/segment_000.wav"),
        )

        assert segment.path is not None


# SegmentConfig tests


class TestSegmentConfig:
    """Tests for SegmentConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default configuration."""
        config = SegmentConfig()

        assert config.max_segment_duration == DEFAULT_SEGMENT_DURATION
        assert config.output_format == "wav"

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = SegmentConfig(
            max_segment_duration=120.0,
            silence_threshold_db=-50,
        )

        assert config.max_segment_duration == 120.0
        assert config.silence_threshold_db == -50

    def test_bounds_enforced(self) -> None:
        """Test that bounds are enforced."""
        config = SegmentConfig(
            max_segment_duration=MAX_SEGMENT_DURATION_SECONDS + 100,
        )

        assert config.max_segment_duration == MAX_SEGMENT_DURATION_SECONDS

    def test_minimum_enforced(self) -> None:
        """Test minimum duration is enforced."""
        config = SegmentConfig(max_segment_duration=1)

        assert config.max_segment_duration >= MIN_SEGMENT_DURATION_SECONDS


# SegmentationResult tests


class TestSegmentationResult:
    """Tests for SegmentationResult dataclass."""

    def test_default_result(self) -> None:
        """Test default result values."""
        result = SegmentationResult()

        assert result.segments == []
        assert result.total_duration == 0.0

    def test_segment_count(self) -> None:
        """Test segment count property."""
        segments = [
            AudioSegment(index=0, start_time=0.0, end_time=60.0),
            AudioSegment(index=1, start_time=60.0, end_time=120.0),
        ]
        result = SegmentationResult(segments=segments)

        assert result.segment_count == 2

    def test_result_with_data(self) -> None:
        """Test result with full data."""
        result = SegmentationResult(
            segments=[],
            total_duration=300.0,
            source_path=Path("/audio.mp3"),
        )

        assert result.total_duration == 300.0


# AudioSegmenter tests


class TestAudioSegmenter:
    """Tests for AudioSegmenter class."""

    def test_segmenter_creation(self) -> None:
        """Test creating segmenter."""
        segmenter = AudioSegmenter()

        assert segmenter.config is not None

    def test_segmenter_with_config(self) -> None:
        """Test segmenter with custom config."""
        config = SegmentConfig(max_segment_duration=120)
        segmenter = AudioSegmenter(config=config)

        assert segmenter.config.max_segment_duration == 120

    def test_is_available_check(self) -> None:
        """Test availability check."""
        segmenter = AudioSegmenter()

        # Returns bool regardless of installation
        result = segmenter.is_available
        assert isinstance(result, bool)


class TestSegmentValidation:
    """Tests for segmentation input validation."""

    def test_segment_nonexistent_file(self) -> None:
        """Test segmenting non-existent file."""
        segmenter = AudioSegmenter()

        result = segmenter.segment(Path("/nonexistent/audio.mp3"))

        assert result.segments == []

    def test_segment_returns_result(self) -> None:
        """Test that segment returns SegmentationResult."""
        segmenter = AudioSegmenter()

        result = segmenter.segment(Path("/nonexistent/file.mp3"))

        assert isinstance(result, SegmentationResult)


class TestSplitPointFinding:
    """Tests for split point finding."""

    def test_find_best_silence_no_silences(self) -> None:
        """Test finding silence with no silences."""
        segmenter = AudioSegmenter()

        result = segmenter._find_best_silence([], 0, 60, 120)

        assert result is None

    def test_find_best_silence_with_silences(self) -> None:
        """Test finding best silence in list."""
        segmenter = AudioSegmenter()
        silences = [
            (55.0, 56.0),  # Near target of 60
            (100.0, 101.0),  # Far from target
        ]

        result = segmenter._find_best_silence(silences, 0, 60, 120)

        # Should pick silence near target
        assert result is not None
        assert 55.0 <= result <= 56.0


class TestSegmentCreation:
    """Tests for segment creation."""

    def test_create_segments_single(self) -> None:
        """Test creating single segment."""
        segmenter = AudioSegmenter()
        split_points = [0.0]

        segments = segmenter._create_segments(split_points, 60.0)

        assert len(segments) == 1
        assert segments[0].end_time == 60.0

    def test_create_segments_multiple(self) -> None:
        """Test creating multiple segments."""
        segmenter = AudioSegmenter()
        split_points = [0.0, 60.0, 120.0]

        segments = segmenter._create_segments(split_points, 180.0)

        assert len(segments) == 3


class TestOverlap:
    """Tests for segment overlap."""

    def test_overlap_applied(self) -> None:
        """Test that overlap is applied to segments."""
        config = SegmentConfig(overlap_seconds=1.0)
        segmenter = AudioSegmenter(config=config)
        split_points = [0.0, 60.0]

        segments = segmenter._create_segments(split_points, 120.0)

        # Second segment should start earlier due to overlap
        assert segments[1].start_time < 60.0


# Factory function tests


class TestCreateSegmenter:
    """Tests for create_segmenter factory."""

    def test_create_default(self) -> None:
        """Test creating with defaults."""
        segmenter = create_segmenter()

        assert segmenter.config.max_segment_duration == DEFAULT_SEGMENT_DURATION

    def test_create_custom(self) -> None:
        """Test creating with custom options."""
        segmenter = create_segmenter(
            max_duration=120.0,
            silence_threshold=-50.0,
        )

        assert segmenter.config.max_segment_duration == 120.0
        assert segmenter.config.silence_threshold_db == -50.0


class TestSegmentAudio:
    """Tests for segment_audio function."""

    def test_segment_nonexistent(self) -> None:
        """Test segmenting non-existent file."""
        result = segment_audio(Path("/nonexistent.mp3"))

        assert isinstance(result, SegmentationResult)
        assert result.segments == []


class TestIsSegmenterAvailable:
    """Tests for is_segmenter_available function."""

    def test_returns_bool(self) -> None:
        """Test that function returns bool."""
        result = is_segmenter_available()

        assert isinstance(result, bool)


class TestMaxSegmentsLimit:
    """Tests for max segments limit."""

    def test_max_segments_constant(self) -> None:
        """Test MAX_SEGMENTS is defined."""
        assert MAX_SEGMENTS > 0
        assert MAX_SEGMENTS == 100
