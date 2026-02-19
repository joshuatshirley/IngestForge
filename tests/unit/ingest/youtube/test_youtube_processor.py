"""Unit tests for YouTube transcript processor.

Tests URL parsing, transcript processing, and chunk creation.
"""

import pytest
from unittest.mock import patch

from ingestforge.ingest.youtube import (
    YouTubeProcessor,
    TranscriptSegment,
    YouTubeMetadata,
    YouTubeProcessingResult,
    is_youtube_available,
    extract_video_id,
)


class TestExtractVideoId:
    """Tests for extract_video_id function."""

    def test_standard_watch_url(self) -> None:
        """Test standard YouTube watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_without_www(self) -> None:
        """Test YouTube URL without www."""
        url = "https://youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_watch_url_http(self) -> None:
        """Test HTTP URL."""
        url = "http://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_short_url(self) -> None:
        """Test youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_embed_url(self) -> None:
        """Test embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_shorts_url(self) -> None:
        """Test YouTube Shorts URL."""
        url = "https://www.youtube.com/shorts/dQw4w9WgXcQ"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_video_id_only(self) -> None:
        """Test raw video ID."""
        video_id = "dQw4w9WgXcQ"
        assert extract_video_id(video_id) == "dQw4w9WgXcQ"

    def test_url_with_extra_params(self) -> None:
        """Test URL with additional parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx0sYbCqOb8TBPRdmBHs&index=5"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_url_with_timestamp(self) -> None:
        """Test URL with timestamp parameter."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s"
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_invalid_url(self) -> None:
        """Test invalid URL returns None."""
        url = "https://example.com/video"
        assert extract_video_id(url) is None

    def test_empty_string(self) -> None:
        """Test empty string returns None."""
        assert extract_video_id("") is None

    def test_none_handling(self) -> None:
        """Test None-like values."""
        assert extract_video_id("   ") is None


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_end_property(self) -> None:
        """Test end time calculation."""
        segment = TranscriptSegment(text="Hello world", start=10.0, duration=5.0)
        assert segment.end == 15.0

    def test_format_timestamp_minutes(self) -> None:
        """Test timestamp formatting for minutes."""
        segment = TranscriptSegment(text="Test", start=65.0, duration=5.0)
        assert segment.start_formatted == "1:05"

    def test_format_timestamp_hours(self) -> None:
        """Test timestamp formatting for hours."""
        segment = TranscriptSegment(text="Test", start=3665.0, duration=5.0)
        assert segment.start_formatted == "1:01:05"

    def test_format_timestamp_seconds_only(self) -> None:
        """Test timestamp formatting for seconds only."""
        segment = TranscriptSegment(text="Test", start=45.0, duration=5.0)
        assert segment.start_formatted == "0:45"


class TestYouTubeMetadata:
    """Tests for YouTubeMetadata dataclass."""

    def test_watch_url(self) -> None:
        """Test watch_url property."""
        metadata = YouTubeMetadata(
            video_id="dQw4w9WgXcQ",
            video_url="https://youtu.be/dQw4w9WgXcQ",
        )
        assert metadata.watch_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_timestamp_url(self) -> None:
        """Test get_timestamp_url method."""
        metadata = YouTubeMetadata(
            video_id="dQw4w9WgXcQ",
            video_url="https://youtu.be/dQw4w9WgXcQ",
        )
        assert metadata.get_timestamp_url(120) == (
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s"
        )

    def test_thumbnail_url(self) -> None:
        """Test thumbnail_url generation."""
        metadata = YouTubeMetadata(
            video_id="dQw4w9WgXcQ",
            video_url="https://youtu.be/dQw4w9WgXcQ",
            thumbnail_url="https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
        )
        assert metadata.thumbnail_url == (
            "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg"
        )


class TestYouTubeProcessor:
    """Tests for YouTubeProcessor class."""

    def test_init_defaults(self) -> None:
        """Test default initialization."""
        processor = YouTubeProcessor()
        assert processor.chunk_duration == 60.0
        assert processor.language == "en"
        assert processor.include_auto_generated is True

    def test_init_custom_values(self) -> None:
        """Test custom initialization values."""
        processor = YouTubeProcessor(
            chunk_duration=120.0,
            language="es",
            include_auto_generated=False,
        )
        assert processor.chunk_duration == 120.0
        assert processor.language == "es"
        assert processor.include_auto_generated is False

    def test_chunk_duration_min_bound(self) -> None:
        """Test minimum chunk duration enforcement."""
        processor = YouTubeProcessor(chunk_duration=1.0)
        assert processor.chunk_duration == 10.0  # MIN_CHUNK_DURATION_SECONDS

    def test_chunk_duration_max_bound(self) -> None:
        """Test maximum chunk duration enforcement."""
        processor = YouTubeProcessor(chunk_duration=1000.0)
        assert processor.chunk_duration == 300.0  # MAX_CHUNK_DURATION_SECONDS

    def test_process_invalid_url(self) -> None:
        """Test processing invalid URL returns error."""
        processor = YouTubeProcessor()
        # Use a clearly invalid URL that won't match any pattern
        result = processor.process("https://example.com/not-youtube")

        assert result.success is False
        assert "Invalid YouTube URL" in result.error

    def test_process_without_library(self) -> None:
        """Test processing when youtube-transcript-api is not installed."""
        processor = YouTubeProcessor()

        # Mock is_youtube_available to return False
        with patch(
            "ingestforge.ingest.youtube.processor.is_youtube_available",
            return_value=False,
        ):
            result = processor.process("dQw4w9WgXcQ")

        assert result.success is False
        assert "youtube-transcript-api not installed" in result.error

    @pytest.mark.skipif(
        not is_youtube_available(), reason="youtube-transcript-api not installed"
    )
    def test_process_real_video(self) -> None:
        """Test processing a real YouTube video (integration test)."""
        processor = YouTubeProcessor()
        # Use a well-known video that should always have transcripts
        result = processor.process("dQw4w9WgXcQ")

        if result.success:
            assert len(result.segments) > 0
            assert len(result.chunks) > 0
            assert result.text != ""
            assert result.metadata.video_id == "dQw4w9WgXcQ"


class TestYouTubeProcessorChunking:
    """Tests for chunk creation logic."""

    def test_convert_to_segments(self) -> None:
        """Test converting raw transcript data to segments."""
        processor = YouTubeProcessor()
        raw_data = [
            {"text": "Hello world", "start": 0.0, "duration": 2.0},
            {"text": "How are you", "start": 2.0, "duration": 1.5},
            {"text": "", "start": 3.5, "duration": 0.5},  # Empty, should be skipped
            {"text": "Goodbye", "start": 4.0, "duration": 2.0},
        ]

        segments = processor._convert_to_segments(raw_data)

        assert len(segments) == 3  # Empty segment skipped
        assert segments[0].text == "Hello world"
        assert segments[0].start == 0.0
        assert segments[1].text == "How are you"
        assert segments[2].text == "Goodbye"

    def test_create_chunks(self) -> None:
        """Test chunk creation from segments."""
        # Use chunk_duration >= MIN_CHUNK_DURATION_SECONDS (10s)
        processor = YouTubeProcessor(chunk_duration=15.0)
        segments = [
            TranscriptSegment(text="Hello", start=0.0, duration=5.0),
            TranscriptSegment(text="world", start=5.0, duration=5.0),
            TranscriptSegment(text="how", start=10.0, duration=5.0),  # Ends at 15.0
            TranscriptSegment(text="are", start=15.0, duration=5.0),
            TranscriptSegment(text="you", start=20.0, duration=5.0),
        ]
        metadata = YouTubeMetadata(
            video_id="test123",
            video_url="https://youtube.com/watch?v=test123",
        )

        chunks = processor._create_chunks(segments, metadata)

        # First chunk includes "Hello", "world", "how" (0-15s, duration >= 15s)
        # Second chunk includes "are", "you" (15-25s)
        assert len(chunks) == 2
        assert "Hello" in chunks[0].content
        assert "world" in chunks[0].content
        assert "how" in chunks[0].content
        assert chunks[0].timestamp_start == 0.0
        assert chunks[0].timestamp_end == 15.0  # "how" ends at 15.0

    def test_create_chunks_empty_segments(self) -> None:
        """Test chunk creation with empty segments list."""
        processor = YouTubeProcessor()
        metadata = YouTubeMetadata(
            video_id="test123",
            video_url="https://youtube.com/watch?v=test123",
        )

        chunks = processor._create_chunks([], metadata)

        assert chunks == []


class TestYouTubeProcessorToChunkRecords:
    """Tests for to_chunk_records method."""

    def test_to_chunk_records(self) -> None:
        """Test conversion to ChunkRecord objects."""
        processor = YouTubeProcessor()
        metadata = YouTubeMetadata(
            video_id="test123",
            video_url="https://youtube.com/watch?v=test123",
            title="Test Video",
            language="en",
            is_generated=False,
        )

        from ingestforge.ingest.youtube.processor import YouTubeChunk

        chunks = [
            YouTubeChunk(
                content="Hello world",
                timestamp_start=0.0,
                timestamp_end=30.0,
                segment_count=5,
                metadata=metadata,
            ),
            YouTubeChunk(
                content="How are you",
                timestamp_start=30.0,
                timestamp_end=60.0,
                segment_count=3,
                metadata=metadata,
            ),
        ]

        result = YouTubeProcessingResult(
            metadata=metadata,
            chunks=chunks,
            text="Hello world How are you",
            success=True,
        )

        records = processor.to_chunk_records(result, "doc_001")

        assert len(records) == 2
        assert records[0].chunk_id == "doc_001_chunk_0000"
        assert records[0].document_id == "doc_001"
        assert records[0].content == "Hello world"
        assert records[0].chunk_type == "transcript"
        assert records[0].metadata["video_id"] == "test123"
        assert records[0].metadata["timestamp_start"] == 0.0
        assert records[0].chunk_index == 0
        assert records[0].total_chunks == 2

        assert records[1].chunk_id == "doc_001_chunk_0001"
        assert records[1].chunk_index == 1

    def test_to_chunk_records_includes_thumbnail(self) -> None:
        """Test that thumbnail_url is included in metadata."""
        processor = YouTubeProcessor()
        metadata = YouTubeMetadata(
            video_id="test123",
            video_url="https://youtube.com/watch?v=test123",
            title="Test Video",
            thumbnail_url="https://img.youtube.com/vi/test123/maxresdefault.jpg",
        )

        from ingestforge.ingest.youtube.processor import YouTubeChunk

        chunks = [
            YouTubeChunk(
                content="Test content",
                timestamp_start=0.0,
                timestamp_end=30.0,
                segment_count=1,
                metadata=metadata,
            ),
        ]

        result = YouTubeProcessingResult(
            metadata=metadata,
            chunks=chunks,
            text="Test content",
            success=True,
        )

        records = processor.to_chunk_records(result, "doc_001")

        assert len(records) == 1
        assert "thumbnail_url" in records[0].metadata
        assert records[0].metadata["thumbnail_url"] == (
            "https://img.youtube.com/vi/test123/maxresdefault.jpg"
        )
