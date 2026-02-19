"""Whisper Backend Integration for audio transcription.

Provides local audio-to-text processing using faster-whisper
for CPU-efficient transcription."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Iterator, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_AUDIO_DURATION_SECONDS = 7200  # 2 hours
MAX_FILE_SIZE_MB = 500
SUPPORTED_FORMATS = frozenset([".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"])
DEFAULT_MODEL = "base"
DEFAULT_LANGUAGE = "en"


class WhisperModel(str, Enum):
    """Available Whisper model sizes."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ComputeType(str, Enum):
    """Compute precision types."""

    INT8 = "int8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    AUTO = "auto"


@dataclass
class TranscriptionConfig:
    """Configuration for transcription."""

    model: str = DEFAULT_MODEL
    language: Optional[str] = DEFAULT_LANGUAGE
    compute_type: ComputeType = ComputeType.AUTO
    device: str = "cpu"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    word_timestamps: bool = True
    vad_filter: bool = True


@dataclass
class TranscriptionSegment:
    """A segment of transcribed text."""

    id: int
    start: float
    end: float
    text: str
    words: List["WordInfo"] = field(default_factory=list)
    confidence: float = 0.0

    @property
    def duration(self) -> float:
        """Get segment duration in seconds."""
        return self.end - self.start


@dataclass
class WordInfo:
    """Word-level transcription information."""

    word: str
    start: float
    end: float
    probability: float = 0.0


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    segments: List[TranscriptionSegment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    model_used: str = ""

    @property
    def word_count(self) -> int:
        """Get total word count."""
        return len(self.text.split())


# Type alias for progress callback
ProgressCallback = Callable[[float, str], None]


class WhisperBackend:
    """Whisper-based audio transcription backend.

    Uses faster-whisper for CPU-efficient transcription
    with optional GPU acceleration.
    """

    def __init__(
        self,
        config: Optional[TranscriptionConfig] = None,
    ) -> None:
        """Initialize the Whisper backend.

        Args:
            config: Transcription configuration
        """
        self.config = config or TranscriptionConfig()
        self._model = None
        self._model_loaded = False

    @property
    def is_available(self) -> bool:
        """Check if faster-whisper is available."""
        try:
            import faster_whisper  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self) -> bool:
        """Load the Whisper model.

        Returns:
            True if model loaded successfully
        """
        if self._model_loaded:
            return True

        if not self.is_available:
            logger.error("faster-whisper not installed")
            return False

        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self.config.model,
                device=self.config.device,
                compute_type=self.config.compute_type.value,
            )
            self._model_loaded = True
            logger.info(f"Loaded Whisper model: {self.config.model}")
            return True

        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            return False

    def transcribe(
        self,
        audio_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional callback for progress updates

        Returns:
            TranscriptionResult with transcribed text
        """
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return TranscriptionResult(text="", model_used=self.config.model)

        if not self._is_supported_format(audio_path):
            logger.error(f"Unsupported format: {audio_path.suffix}")
            return TranscriptionResult(text="", model_used=self.config.model)

        # Check file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            logger.error(f"File too large: {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB")
            return TranscriptionResult(text="", model_used=self.config.model)

        # Load model if needed
        if not self._model_loaded:
            if not self.load_model():
                return TranscriptionResult(text="", model_used=self.config.model)

        # Run transcription
        return self._run_transcription(audio_path, progress_callback)

    def _run_transcription(
        self,
        audio_path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> TranscriptionResult:
        """Run the actual transcription.

        Args:
            audio_path: Path to audio file
            progress_callback: Optional progress callback

        Returns:
            TranscriptionResult
        """
        try:
            if progress_callback:
                progress_callback(0.1, "Starting transcription...")

            # Run faster-whisper transcription
            segments_iter, info = self._model.transcribe(
                str(audio_path),
                language=self.config.language,
                beam_size=self.config.beam_size,
                best_of=self.config.best_of,
                temperature=self.config.temperature,
                word_timestamps=self.config.word_timestamps,
                vad_filter=self.config.vad_filter,
            )

            # Collect segments
            segments = self._collect_segments(
                segments_iter, info.duration, progress_callback
            )

            # Build full text
            full_text = " ".join(seg.text.strip() for seg in segments)

            if progress_callback:
                progress_callback(1.0, "Transcription complete")

            return TranscriptionResult(
                text=full_text,
                segments=segments,
                language=info.language,
                language_probability=info.language_probability,
                duration=info.duration,
                model_used=self.config.model,
            )

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(text="", model_used=self.config.model)

    def _collect_segments(
        self,
        segments_iter: Iterator,
        total_duration: float,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> List[TranscriptionSegment]:
        """Collect segments from iterator.

        Args:
            segments_iter: Segment iterator from faster-whisper
            total_duration: Total audio duration
            progress_callback: Optional progress callback

        Returns:
            List of TranscriptionSegment
        """
        segments: List[TranscriptionSegment] = []

        for segment in segments_iter:
            # Build word info list
            words = []
            if hasattr(segment, "words") and segment.words:
                for word in segment.words:
                    words.append(
                        WordInfo(
                            word=word.word,
                            start=word.start,
                            end=word.end,
                            probability=word.probability,
                        )
                    )

            # Create segment
            trans_seg = TranscriptionSegment(
                id=segment.id,
                start=segment.start,
                end=segment.end,
                text=segment.text,
                words=words,
                confidence=segment.avg_logprob
                if hasattr(segment, "avg_logprob")
                else 0.0,
            )
            segments.append(trans_seg)

            # Update progress
            if progress_callback and total_duration > 0:
                progress = min(0.9, 0.1 + (segment.end / total_duration) * 0.8)
                progress_callback(progress, f"Transcribed {segment.end:.1f}s...")

        return segments

    def _is_supported_format(self, path: Path) -> bool:
        """Check if file format is supported.

        Args:
            path: File path

        Returns:
            True if supported
        """
        return path.suffix.lower() in SUPPORTED_FORMATS

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._model = None
        self._model_loaded = False
        logger.info("Whisper model unloaded")


def create_whisper_backend(
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
    language: Optional[str] = DEFAULT_LANGUAGE,
) -> WhisperBackend:
    """Factory function to create Whisper backend.

    Args:
        model: Model size (tiny, base, small, medium, large)
        device: Device to use (cpu, cuda)
        language: Language code or None for auto-detect

    Returns:
        Configured WhisperBackend
    """
    config = TranscriptionConfig(
        model=model,
        device=device,
        language=language,
    )
    return WhisperBackend(config=config)


def transcribe_audio(
    audio_path: Path,
    model: str = DEFAULT_MODEL,
) -> TranscriptionResult:
    """Convenience function to transcribe audio.

    Args:
        audio_path: Path to audio file
        model: Model size to use

    Returns:
        TranscriptionResult
    """
    backend = create_whisper_backend(model=model)
    return backend.transcribe(audio_path)


def is_whisper_available() -> bool:
    """Check if Whisper is available.

    Returns:
        True if faster-whisper is installed
    """
    try:
        import faster_whisper  # noqa: F401

        return True
    except ImportError:
        return False
