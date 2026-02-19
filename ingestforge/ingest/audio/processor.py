"""Audio processor for lecture/podcast ingestion.

Orchestrates Whisper transcription and chunking for audio files.
Creates timestamped chunks suitable for RAG retrieval with audio citations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_AUDIO_DURATION_SECONDS = 7200  # 2 hours
MIN_CHUNK_DURATION_SECONDS = 10.0
DEFAULT_CHUNK_DURATION_SECONDS = 60.0
MAX_CHUNK_DURATION_SECONDS = 300.0


@dataclass
class AudioChunk:
    """A chunk of transcribed audio with timestamp provenance.

    Attributes:
        content: The transcribed text content.
        timestamp_start: Start time in seconds.
        timestamp_end: End time in seconds.
        word_count: Number of words in content.
        segment_count: Number of original segments in this chunk.
    """

    content: str
    timestamp_start: float
    timestamp_end: float
    word_count: int = 0
    segment_count: int = 0

    def __post_init__(self) -> None:
        """Calculate word count if not provided."""
        if self.word_count == 0:
            self.word_count = len(self.content.split())

    @property
    def duration(self) -> float:
        """Get chunk duration in seconds."""
        return self.timestamp_end - self.timestamp_start

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
        return self.format_timestamp(self.timestamp_start)

    @property
    def end_formatted(self) -> str:
        """Get formatted end timestamp."""
        return self.format_timestamp(self.timestamp_end)


@dataclass
class AudioMetadata:
    """Metadata about the audio file.

    Attributes:
        file_path: Path to audio file.
        file_name: Name of audio file.
        duration: Total duration in seconds.
        language: Detected or specified language.
        model_used: Whisper model used for transcription.
        is_segmented: Whether file was segmented before processing.
    """

    file_path: str
    file_name: str
    duration: float = 0.0
    language: str = "en"
    model_used: str = "base"
    is_segmented: bool = False


@dataclass
class AudioProcessingResult:
    """Result of audio file processing.

    Attributes:
        metadata: Audio file metadata.
        chunks: Transcribed chunks with timestamps.
        text: Full transcribed text.
        success: Whether processing succeeded.
        error: Error message if failed.
    """

    metadata: AudioMetadata
    chunks: List[AudioChunk] = field(default_factory=list)
    text: str = ""
    success: bool = True
    error: str = ""

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return len(self.text.split())

    @property
    def chunk_count(self) -> int:
        """Get number of chunks."""
        return len(self.chunks)


class AudioProcessor:
    """Process audio files for lecture/podcast ingestion.

    Combines Whisper transcription with intelligent chunking to create
    timestamped chunks suitable for RAG retrieval.

    Example:
        processor = AudioProcessor()
        result = processor.process(Path("lecture.mp3"))

        for chunk in result.chunks:
            print(f"[{chunk.start_formatted}] {chunk.content[:50]}...")
    """

    def __init__(
        self,
        chunk_duration: float = DEFAULT_CHUNK_DURATION_SECONDS,
        whisper_model: str = "base",
        language: Optional[str] = "en",
    ) -> None:
        """Initialize the audio processor.

        Args:
            chunk_duration: Target duration for each chunk in seconds.
            whisper_model: Whisper model to use (tiny, base, small, medium, large).
            language: Language code or None for auto-detect.

        Rule #7: Parameter validation
        """
        self.chunk_duration = max(
            MIN_CHUNK_DURATION_SECONDS, min(chunk_duration, MAX_CHUNK_DURATION_SECONDS)
        )
        self.whisper_model = whisper_model
        self.language = language

    @property
    def is_available(self) -> bool:
        """Check if dependencies are installed."""
        from ingestforge.ingest.audio import is_whisper_available

        return is_whisper_available()

    def process(self, audio_path: Path) -> AudioProcessingResult:
        """Process an audio file and extract transcript.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioProcessingResult with transcript data.

        Rule #1: Early returns for validation
        Rule #7: Parameter validation
        """
        if not audio_path.exists():
            return self._create_error_result(
                audio_path, f"Audio file not found: {audio_path}"
            )

        # Check availability
        if not self.is_available:
            return self._create_error_result(
                audio_path,
                "faster-whisper not installed. Install with: pip install faster-whisper",
            )

        # Process the audio file
        return self._process_audio(audio_path)

    def _create_error_result(
        self, audio_path: Path, error: str
    ) -> AudioProcessingResult:
        """Create an error result.

        Args:
            audio_path: Path that failed.
            error: Error message.

        Returns:
            AudioProcessingResult with error status.
        """
        metadata = AudioMetadata(
            file_path=str(audio_path),
            file_name=audio_path.name,
        )
        return AudioProcessingResult(
            metadata=metadata,
            success=False,
            error=error,
        )

    def _process_audio(self, audio_path: Path) -> AudioProcessingResult:
        """Process audio file with Whisper.

        Args:
            audio_path: Path to audio file.

        Returns:
            AudioProcessingResult with transcription.

        Rule #4: Function <60 lines
        Rule #5: Log errors instead of silent failure
        """
        try:
            from ingestforge.ingest.audio import create_whisper_backend

            # Create Whisper backend
            backend = create_whisper_backend(
                model=self.whisper_model,
                language=self.language,
            )

            # Transcribe audio
            logger.info(f"Transcribing audio: {audio_path.name}")
            transcription_result = backend.transcribe(audio_path)

            # Check for transcription failure
            if not transcription_result.text:
                return self._create_error_result(
                    audio_path, "Transcription produced no text"
                )

            # Build metadata
            metadata = AudioMetadata(
                file_path=str(audio_path),
                file_name=audio_path.name,
                duration=transcription_result.duration,
                language=transcription_result.language,
                model_used=transcription_result.model_used,
            )

            # Create chunks from segments
            chunks = self._create_chunks_from_segments(transcription_result.segments)

            logger.info(
                f"Created {len(chunks)} chunks from {len(transcription_result.segments)} segments"
            )

            return AudioProcessingResult(
                metadata=metadata,
                chunks=chunks,
                text=transcription_result.text,
                success=True,
            )

        except Exception as e:
            logger.error(f"Failed to process audio {audio_path}: {e}")
            return self._create_error_result(audio_path, str(e))

    def _create_chunks_from_segments(self, segments: List) -> List[AudioChunk]:
        """Create chunks from transcription segments.

        Groups segments into chunks of approximately chunk_duration seconds.

        Args:
            segments: List of TranscriptionSegment objects.

        Returns:
            List of AudioChunk objects.

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
            current_texts.append(segment.text.strip())
            current_end = segment.end
            segment_count += 1

            # Check if chunk is long enough
            if (current_end - current_start) >= self.chunk_duration:
                chunk = self._finalize_chunk(
                    current_texts, current_start, current_end, segment_count
                )
                chunks.append(chunk)

                # Reset for next chunk
                current_texts = []
                current_start = current_end
                segment_count = 0

        # Handle remaining segments
        if current_texts:
            chunk = self._finalize_chunk(
                current_texts, current_start, current_end, segment_count
            )
            chunks.append(chunk)

        return chunks

    def _finalize_chunk(
        self,
        texts: List[str],
        start: float,
        end: float,
        segment_count: int,
    ) -> AudioChunk:
        """Finalize a chunk from accumulated texts.

        Args:
            texts: List of segment texts.
            start: Chunk start time.
            end: Chunk end time.
            segment_count: Number of segments.

        Returns:
            AudioChunk object.

        Rule #4: Function <60 lines
        """
        content = " ".join(texts)
        return AudioChunk(
            content=content,
            timestamp_start=start,
            timestamp_end=end,
            segment_count=segment_count,
        )

    def to_chunk_records(
        self,
        result: AudioProcessingResult,
        document_id: str,
    ) -> List:
        """Convert processing result to ChunkRecord objects.

        Creates ChunkRecords with proper source location metadata
        for timestamp-based citations.

        Args:
            result: Audio processing result.
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
                source_type=SourceType.AUDIO,
                title=result.metadata.file_name,
                file_path=result.metadata.file_path,
                timestamp_start=chunk.start_formatted,
                timestamp_end=chunk.end_formatted,
            )

            record = ChunkRecord(
                chunk_id=f"{document_id}_chunk_{idx:04d}",
                document_id=document_id,
                content=chunk.content,
                section_title=f"Timestamp {chunk.start_formatted}",
                chunk_type="transcript",
                source_file=result.metadata.file_path,
                word_count=chunk.word_count,
                char_count=len(chunk.content),
                chunk_index=idx,
                total_chunks=total_chunks,
                source_location=source_location,
                metadata={
                    "audio_file": result.metadata.file_name,
                    "duration": result.metadata.duration,
                    "language": result.metadata.language,
                    "model_used": result.metadata.model_used,
                    "timestamp_start": chunk.timestamp_start,
                    "timestamp_end": chunk.timestamp_end,
                    "timestamp_start_formatted": chunk.start_formatted,
                    "timestamp_end_formatted": chunk.end_formatted,
                },
            )
            records.append(record)

        return records
