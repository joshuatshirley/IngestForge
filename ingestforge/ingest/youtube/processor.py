"""YouTube transcript processor for video ingestion.

Extracts transcripts from YouTube videos using youtube-transcript-api
and creates timestamped chunks suitable for RAG retrieval."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_VIDEO_DURATION_SECONDS = 14400  # 4 hours
MAX_TRANSCRIPT_SEGMENTS = 10000
MIN_CHUNK_DURATION_SECONDS = 10.0
DEFAULT_CHUNK_DURATION_SECONDS = 60.0
MAX_CHUNK_DURATION_SECONDS = 300.0

# Supported URL patterns for YouTube
YOUTUBE_URL_PATTERNS = [
    r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/embed/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/v/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?youtu\.be/([a-zA-Z0-9_-]{11})",
    r"(?:https?://)?(?:www\.)?youtube\.com/shorts/([a-zA-Z0-9_-]{11})",
]


@dataclass
class TranscriptSegment:
    """A segment of transcript with timestamp metadata.

    Attributes:
        text: The transcript text for this segment.
        start: Start time in seconds.
        duration: Duration of segment in seconds.
    """

    text: str
    start: float
    duration: float

    @property
    def end(self) -> float:
        """Get end time in seconds."""
        return self.start + self.duration

    def format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS.

        Args:
            seconds: Time in seconds.

        Returns:
            Formatted timestamp string.
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"

    @property
    def start_formatted(self) -> str:
        """Get formatted start timestamp."""
        return self.format_timestamp(self.start)

    @property
    def end_formatted(self) -> str:
        """Get formatted end timestamp."""
        return self.format_timestamp(self.end)


@dataclass
class YouTubeMetadata:
    """Metadata about the YouTube video.

    Attributes:
        video_id: YouTube video ID (11 characters).
        video_url: Full YouTube URL.
        title: Video title (if available).
        language: Transcript language code.
        is_generated: Whether transcript is auto-generated.
        thumbnail_url: URL to video thumbnail image.
    """

    video_id: str
    video_url: str
    title: str = ""
    language: str = "en"
    is_generated: bool = False
    duration_seconds: float = 0.0
    thumbnail_url: Optional[str] = None

    @property
    def watch_url(self) -> str:
        """Get standard watch URL."""
        return f"https://www.youtube.com/watch?v={self.video_id}"

    def get_timestamp_url(self, seconds: float) -> str:
        """Get URL with timestamp.

        Args:
            seconds: Time in seconds.

        Returns:
            YouTube URL with timestamp parameter.
        """
        return f"{self.watch_url}&t={int(seconds)}s"


@dataclass
class YouTubeChunk:
    """A chunk of transcript content with timestamp provenance.

    Attributes:
        content: The combined transcript text.
        timestamp_start: Start time in seconds.
        timestamp_end: End time in seconds.
        segment_count: Number of original segments in this chunk.
        metadata: Reference to video metadata.
    """

    content: str
    timestamp_start: float
    timestamp_end: float
    segment_count: int
    metadata: YouTubeMetadata

    @property
    def duration(self) -> float:
        """Get chunk duration in seconds."""
        return self.timestamp_end - self.timestamp_start

    @property
    def timestamp_url(self) -> str:
        """Get URL to video at this timestamp."""
        return self.metadata.get_timestamp_url(self.timestamp_start)


@dataclass
class YouTubeProcessingResult:
    """Result of YouTube video processing.

    Attributes:
        metadata: Video metadata.
        segments: Raw transcript segments.
        chunks: Combined chunks with timestamps.
        text: Full transcript text.
        success: Whether processing succeeded.
        error: Error message if failed.
    """

    metadata: YouTubeMetadata
    segments: List[TranscriptSegment] = field(default_factory=list)
    chunks: List[YouTubeChunk] = field(default_factory=list)
    text: str = ""
    success: bool = True
    error: str = ""

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return len(self.text.split())

    @property
    def duration_seconds(self) -> float:
        """Get total transcript duration."""
        if not self.segments:
            return 0.0
        last = self.segments[-1]
        return last.start + last.duration


def is_youtube_available() -> bool:
    """Check if youtube-transcript-api is available.

    Returns:
        True if the library is installed.
    """
    try:
        from youtube_transcript_api import YouTubeTranscriptApi  # noqa: F401

        return True
    except ImportError:
        return False


def extract_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL.

    Supports various YouTube URL formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID

    Args:
        url: YouTube URL or video ID.

    Returns:
        Video ID string or None if not found.

    Rule #7: Parameter validation
    Rule #1: Early return
    """
    if not url:
        return None

    url = url.strip()

    # Check if already a video ID (11 alphanumeric chars)
    if re.match(r"^[a-zA-Z0-9_-]{11}$", url):
        return url

    # Try each URL pattern
    for pattern in YOUTUBE_URL_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    return None


class YouTubeProcessor:
    """Process YouTube videos by extracting transcripts.

    Creates timestamped chunks suitable for RAG retrieval with
    video timestamp provenance for citations.

    Example:
        processor = YouTubeProcessor()
        result = processor.process("https://youtube.com/watch?v=abc123")

        for chunk in result.chunks:
            print(f"[{chunk.timestamp_start}s] {chunk.content[:50]}...")
    """

    def __init__(
        self,
        chunk_duration: float = DEFAULT_CHUNK_DURATION_SECONDS,
        language: str = "en",
        include_auto_generated: bool = True,
    ) -> None:
        """Initialize the YouTube processor.

        Args:
            chunk_duration: Target duration for each chunk in seconds.
            language: Preferred transcript language code.
            include_auto_generated: Whether to use auto-generated transcripts.

        Rule #7: Parameter validation
        """
        self.chunk_duration = max(
            MIN_CHUNK_DURATION_SECONDS, min(chunk_duration, MAX_CHUNK_DURATION_SECONDS)
        )
        self.language = language
        self.include_auto_generated = include_auto_generated

    @property
    def is_available(self) -> bool:
        """Check if youtube-transcript-api is installed."""
        return is_youtube_available()

    def process(self, url_or_id: str) -> YouTubeProcessingResult:
        """Process a YouTube video and extract transcript.

        Args:
            url_or_id: YouTube URL or video ID.

        Returns:
            YouTubeProcessingResult with transcript data.

        Rule #1: Early returns for validation
        Rule #7: Parameter validation
        """
        video_id = extract_video_id(url_or_id)
        if not video_id:
            return self._create_error_result(
                url_or_id, "Invalid YouTube URL or video ID"
            )

        # Check library availability
        if not self.is_available:
            return self._create_error_result(
                video_id,
                "youtube-transcript-api not installed. "
                "Install with: pip install youtube-transcript-api",
            )

        # Fetch and process transcript
        return self._fetch_transcript(video_id)

    def _create_error_result(
        self, video_id: str, error: str
    ) -> YouTubeProcessingResult:
        """Create an error result.

        Args:
            video_id: Video ID or URL that failed.
            error: Error message.

        Returns:
            YouTubeProcessingResult with error status.
        """
        metadata = YouTubeMetadata(
            video_id=video_id,
            video_url=f"https://www.youtube.com/watch?v={video_id}",
        )
        return YouTubeProcessingResult(
            metadata=metadata,
            success=False,
            error=error,
        )

    def _fetch_transcript(self, video_id: str) -> YouTubeProcessingResult:
        """Fetch transcript from YouTube.

        Args:
            video_id: YouTube video ID.

        Returns:
            YouTubeProcessingResult with transcript data.

        Rule #4: Function <60 lines
        Rule #5: Log errors instead of silent failure
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            # Fetch available transcripts (v1.2+ uses instance methods)
            api = YouTubeTranscriptApi()
            transcript_list = api.list(video_id)

            # Try to get transcript in preferred language
            transcript_data, is_generated = self._select_transcript(transcript_list)

            # Convert to segments
            segments = self._convert_to_segments(transcript_data)

            # Validate segment count
            if len(segments) > MAX_TRANSCRIPT_SEGMENTS:
                logger.warning(
                    f"Transcript has {len(segments)} segments, "
                    f"truncating to {MAX_TRANSCRIPT_SEGMENTS}"
                )
                segments = segments[:MAX_TRANSCRIPT_SEGMENTS]

            # Build metadata
            metadata = self._build_metadata(video_id, segments, is_generated)

            # Create chunks from segments
            chunks = self._create_chunks(segments, metadata)

            # Build full text
            full_text = " ".join(seg.text for seg in segments)

            return YouTubeProcessingResult(
                metadata=metadata,
                segments=segments,
                chunks=chunks,
                text=full_text,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to fetch transcript for {video_id}: {e}")
            return self._create_error_result(video_id, str(e))

    def _select_transcript(self, transcript_list) -> Tuple[List[dict], bool]:
        """Select best available transcript.

        Args:
            transcript_list: YouTubeTranscriptApi transcript list.

        Returns:
            Tuple of (transcript data, is_auto_generated).

        Rule #1: Early returns
        Rule #4: Function <60 lines
        """
        # Try manual transcript in preferred language
        try:
            transcript = transcript_list.find_transcript([self.language])
            if not transcript.is_generated:
                return transcript.fetch(), False
        except Exception:
            pass

        # Try auto-generated if allowed
        if self.include_auto_generated:
            try:
                transcript = transcript_list.find_generated_transcript([self.language])
                return transcript.fetch(), True
            except Exception:
                pass

        # Fall back to any available transcript
        for transcript in transcript_list:
            try:
                return transcript.fetch(), transcript.is_generated
            except Exception:
                continue

        # No transcript found
        raise ValueError(f"No transcript available in language '{self.language}'")

    def _convert_to_segments(self, transcript_data: List) -> List[TranscriptSegment]:
        """Convert raw transcript data to TranscriptSegment objects.

        Args:
            transcript_data: Raw transcript data from API (dict or dataclass).

        Returns:
            List of TranscriptSegment objects.

        Rule #4: Function <60 lines
        """
        segments = []
        for item in transcript_data:
            # Support both dict format (old API) and dataclass (v1.2+)
            if hasattr(item, "text"):
                text = item.text
                start = item.start
                duration = item.duration
            else:
                text = item.get("text", "")
                start = item.get("start", 0)
                duration = item.get("duration", 0)

            text = text.strip() if text else ""
            if not text:
                continue

            segment = TranscriptSegment(
                text=text,
                start=float(start),
                duration=float(duration),
            )
            segments.append(segment)

        return segments

    def _build_metadata(
        self,
        video_id: str,
        segments: List[TranscriptSegment],
        is_generated: bool,
    ) -> YouTubeMetadata:
        """Build metadata from video and segments.

        Args:
            video_id: YouTube video ID.
            segments: Transcript segments.
            is_generated: Whether transcript is auto-generated.

        Returns:
            YouTubeMetadata object.

        Rule #4: Function <60 lines
        """
        duration = 0.0
        if segments:
            last = segments[-1]
            duration = last.start + last.duration

        # Generate thumbnail URL
        thumbnail_url = f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"

        return YouTubeMetadata(
            video_id=video_id,
            video_url=f"https://www.youtube.com/watch?v={video_id}",
            language=self.language,
            is_generated=is_generated,
            duration_seconds=duration,
            thumbnail_url=thumbnail_url,
        )

    def _create_chunks(
        self,
        segments: List[TranscriptSegment],
        metadata: YouTubeMetadata,
    ) -> List[YouTubeChunk]:
        """Create chunks from segments based on duration.

        Groups segments into chunks of approximately chunk_duration seconds.

        Args:
            segments: Transcript segments to combine.
            metadata: Video metadata.

        Returns:
            List of YouTubeChunk objects.

        Rule #1: Max 3 nesting levels
        Rule #4: Function <60 lines
        """
        if not segments:
            return []

        chunks = []
        current_texts: List[str] = []
        current_start = segments[0].start
        current_end = segments[0].end
        segment_count = 0

        for segment in segments:
            current_texts.append(segment.text)
            current_end = segment.end
            segment_count += 1

            # Check if chunk is long enough
            if (current_end - current_start) >= self.chunk_duration:
                chunk = self._finalize_chunk(
                    current_texts, current_start, current_end, segment_count, metadata
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_texts = []
                current_start = current_end
                segment_count = 0

        # Handle remaining segments
        if current_texts:
            chunk = self._finalize_chunk(
                current_texts, current_start, current_end, segment_count, metadata
            )
            chunks.append(chunk)

        return chunks

    def _finalize_chunk(
        self,
        texts: List[str],
        start: float,
        end: float,
        segment_count: int,
        metadata: YouTubeMetadata,
    ) -> YouTubeChunk:
        """Finalize a chunk from accumulated texts.

        Args:
            texts: List of segment texts.
            start: Chunk start time.
            end: Chunk end time.
            segment_count: Number of segments.
            metadata: Video metadata.

        Returns:
            YouTubeChunk object.

        Rule #4: Function <60 lines
        """
        content = " ".join(texts)
        return YouTubeChunk(
            content=content,
            timestamp_start=start,
            timestamp_end=end,
            segment_count=segment_count,
            metadata=metadata,
        )

    def to_chunk_records(
        self,
        result: YouTubeProcessingResult,
        document_id: str,
    ) -> List:
        """Convert processing result to ChunkRecord objects.

        Creates ChunkRecords with proper source location metadata
        for timestamp-based citations.

        Args:
            result: YouTube processing result.
            document_id: Document ID for the chunks.

        Returns:
            List of ChunkRecord objects.

        Rule #4: Function <60 lines
        """
        from ingestforge.chunking import ChunkRecord
        from ingestforge.core.provenance import SourceLocation, SourceType

        records = []
        total_chunks = len(result.chunks)

        for idx, chunk in enumerate(result.chunks):
            # Build source location with timestamp
            source_location = SourceLocation(
                source_type=SourceType.VIDEO,
                title=result.metadata.title or result.metadata.video_id,
                url=chunk.timestamp_url,
                timestamp_start=chunk.metadata.get_timestamp_url(
                    chunk.timestamp_start
                ).split("t=")[1],
                timestamp_end=chunk.metadata.get_timestamp_url(
                    chunk.timestamp_end
                ).split("t=")[1],
            )

            record = ChunkRecord(
                chunk_id=f"{document_id}_chunk_{idx:04d}",
                document_id=document_id,
                content=chunk.content,
                section_title=f"Timestamp {chunk.timestamp_start:.0f}s",
                chunk_type="transcript",
                source_file=result.metadata.video_url,
                word_count=len(chunk.content.split()),
                char_count=len(chunk.content),
                chunk_index=idx,
                total_chunks=total_chunks,
                source_location=source_location,
                metadata={
                    "video_id": result.metadata.video_id,
                    "video_url": result.metadata.video_url,
                    "thumbnail_url": result.metadata.thumbnail_url,
                    "timestamp_start": chunk.timestamp_start,
                    "timestamp_end": chunk.timestamp_end,
                    "timestamp_url": chunk.timestamp_url,
                    "is_auto_generated": result.metadata.is_generated,
                    "language": result.metadata.language,
                },
            )
            records.append(record)

        return records
