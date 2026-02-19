"""
Audio Diarization Processor.

Speaker identification and timestamp synchronization.
Enhances Whisper transcription with speaker diarization.

Features:
- Diarization support (Speaker A, Speaker B, etc.)
- Millisecond-precision timestamps (start_ms, end_ms)
- Integration with pyannote.audio for speaker identification
- Fallback to simple speaker detection when unavailable
- JPL Rule #5: Assertions for audio file validity

NASA JPL Power of Ten Rules:
- Rule #2: Bounded speaker counts and segment sizes
- Rule #4: Functions <60 lines
- Rule #5: Assert preconditions (audio file not null, valid codec)
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_SPEAKERS = 20  # Maximum distinct speakers
MAX_SEGMENTS = 10000  # Maximum transcript segments
MAX_AUDIO_DURATION_MS = 7200000  # 2 hours in milliseconds
MIN_SEGMENT_DURATION_MS = 100  # Minimum 100ms segment

# Supported audio formats
SUPPORTED_CODECS = frozenset(["mp3", "wav", "flac", "m4a", "ogg", "webm", "wma", "aac"])


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SpeakerSegment:
    """
    A transcript segment with speaker identification.

    AC: 100% of transcript lines have start_ms and end_ms.
    Rule #9: Complete type hints.
    """

    text: str
    speaker_id: str  # "Speaker A", "Speaker B", etc.
    start_ms: int  # Start time in milliseconds
    end_ms: int  # End time in milliseconds
    confidence: float = 0.0  # Speaker identification confidence
    words: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        """Get segment duration in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def start_seconds(self) -> float:
        """Get start time in seconds."""
        return self.start_ms / 1000.0

    @property
    def end_seconds(self) -> float:
        """Get end time in seconds."""
        return self.end_ms / 1000.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for metadata storage."""
        return {
            "text": self.text,
            "speaker_id": self.speaker_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "confidence": self.confidence,
            "words": self.words,
        }

    def format_timestamp(self) -> str:
        """Format as HH:MM:SS.mmm."""
        total_seconds = self.start_ms / 1000.0
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


@dataclass
class Speaker:
    """
    Identified speaker in audio.

    Map Speaker nodes to PERSON entity type.
    Rule #9: Complete type hints.
    """

    speaker_id: str  # "Speaker A", "Speaker B", etc.
    label: Optional[str] = None  # User-provided name if known
    total_speaking_time_ms: int = 0
    segment_count: int = 0
    embedding: Optional[List[float]] = None  # Speaker embedding if available

    def to_entity_dict(self) -> Dict[str, Any]:
        """Convert to entity dict for Knowledge Graph."""
        return {
            "entity_type": "PERSON",
            "name": self.label or self.speaker_id,
            "properties": {
                "speaker_id": self.speaker_id,
                "total_speaking_time_ms": self.total_speaking_time_ms,
                "segment_count": self.segment_count,
            },
        }


@dataclass
class DiarizationResult:
    """
    Result of audio diarization.

    Rule #9: Complete type hints.
    """

    segments: List[SpeakerSegment] = field(default_factory=list)
    speakers: List[Speaker] = field(default_factory=list)
    full_text: str = ""
    duration_ms: int = 0
    language: str = "en"
    success: bool = False
    error_message: str = ""

    @property
    def speaker_count(self) -> int:
        """Get number of identified speakers."""
        return len(self.speakers)

    @property
    def segment_count(self) -> int:
        """Get number of segments."""
        return len(self.segments)

    def get_speaker_transcript(self, speaker_id: str) -> str:
        """Get transcript for a specific speaker."""
        texts = [seg.text for seg in self.segments if seg.speaker_id == speaker_id]
        return " ".join(texts)

    def to_formatted_transcript(self) -> str:
        """Format transcript with speaker labels."""
        lines = []
        for seg in self.segments:
            timestamp = seg.format_timestamp()
            lines.append(f"[{timestamp}] {seg.speaker_id}: {seg.text}")
        return "\n".join(lines)


@dataclass
class DiarizationConfig:
    """
    Configuration for diarization processor.

    Rule #2: Bounded parameters.
    Rule #9: Complete type hints.
    """

    whisper_model: str = "base"
    language: Optional[str] = "en"
    num_speakers: Optional[int] = None  # Auto-detect if None
    min_speakers: int = 1
    max_speakers: int = MAX_SPEAKERS
    huggingface_token: Optional[str] = None  # For pyannote.audio
    device: str = "cpu"
    batch_size: int = 8

    def __post_init__(self) -> None:
        """Validate and bound configuration."""
        self.max_speakers = min(self.max_speakers, MAX_SPEAKERS)
        if self.num_speakers:
            self.num_speakers = min(self.num_speakers, MAX_SPEAKERS)


# =============================================================================
# AUDIO VALIDATION
# =============================================================================


def validate_audio_file(audio_path: Path) -> Tuple[bool, str]:
    """
    Validate audio file exists and has valid codec.

    AC / JPL Rule #5: Assert audio file not null and valid codec.
    Rule #4: <60 lines.
    Rule #7: Return validation result.

    Args:
        audio_path: Path to audio file.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Check file exists
    if not audio_path or not isinstance(audio_path, Path):
        return False, "Audio path is None or invalid type"

    if not audio_path.exists():
        return False, f"Audio file not found: {audio_path}"

    # Check file size
    file_size = audio_path.stat().st_size
    if file_size == 0:
        return False, "Audio file is empty"

    # Check codec by extension
    extension = audio_path.suffix.lower().lstrip(".")
    if extension not in SUPPORTED_CODECS:
        return (
            False,
            f"Unsupported audio codec: {extension}. Supported: {', '.join(SUPPORTED_CODECS)}",
        )

    # Try to probe audio with pydub if available
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(audio_path))
        duration_ms = len(audio)

        if duration_ms < MIN_SEGMENT_DURATION_MS:
            return False, f"Audio too short: {duration_ms}ms"

        if duration_ms > MAX_AUDIO_DURATION_MS:
            return (
                False,
                f"Audio too long: {duration_ms}ms (max {MAX_AUDIO_DURATION_MS}ms)",
            )

    except ImportError:
        # pydub not available, skip detailed validation
        logger.debug("pydub not available for audio validation")
    except Exception as e:
        return False, f"Cannot read audio file: {e}"

    return True, ""


def get_audio_duration_ms(audio_path: Path) -> int:
    """
    Get audio duration in milliseconds.

    Rule #4: <60 lines.
    Rule #7: Return 0 on error.
    """
    try:
        from pydub import AudioSegment

        audio = AudioSegment.from_file(str(audio_path))
        return len(audio)
    except Exception:
        # Fallback: estimate from file size (rough approximation)
        return 0


# =============================================================================
# DIARIZATION BACKENDS
# =============================================================================


class DiarizationBackend:
    """Base class for diarization backends."""

    def diarize(
        self,
        audio_path: Path,
        config: DiarizationConfig,
    ) -> List[Tuple[float, float, str]]:
        """
        Perform speaker diarization.

        Args:
            audio_path: Path to audio file.
            config: Diarization configuration.

        Returns:
            List of (start_seconds, end_seconds, speaker_id) tuples.
        """
        raise NotImplementedError

    def is_available(self) -> bool:
        """Check if backend is available."""
        return False


class PyAnnoteDiarizer(DiarizationBackend):
    """
    Pyannote.audio based speaker diarization.

    Requires HuggingFace token for model access.
    """

    def __init__(self, hf_token: Optional[str] = None):
        """Initialize with HuggingFace token."""
        self._hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._pipeline = None

    def is_available(self) -> bool:
        """Check if pyannote.audio is available."""
        if not self._hf_token:
            return False
        try:
            from pyannote.audio import Pipeline  # noqa: F401

            return True
        except ImportError:
            return False

    def diarize(
        self,
        audio_path: Path,
        config: DiarizationConfig,
    ) -> List[Tuple[float, float, str]]:
        """
        Perform diarization using pyannote.audio.

        Rule #4: <60 lines.
        Rule #7: Check pipeline result.
        """
        from pyannote.audio import Pipeline

        if self._pipeline is None:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self._hf_token,
            )

        # Run diarization
        diarization = self._pipeline(str(audio_path))

        # Extract segments
        segments: List[Tuple[float, float, str]] = []
        speaker_map: Dict[str, str] = {}
        speaker_counter = ord("A")

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Map speaker labels to "Speaker A", "Speaker B", etc.
            if speaker not in speaker_map:
                if len(speaker_map) < MAX_SPEAKERS:
                    speaker_map[speaker] = f"Speaker {chr(speaker_counter)}"
                    speaker_counter += 1
                else:
                    speaker_map[speaker] = "Speaker Other"

            segments.append(
                (
                    turn.start,
                    turn.end,
                    speaker_map[speaker],
                )
            )

        return segments


class SimpleDiarizer(DiarizationBackend):
    """
    Simple fallback diarizer using silence detection.

    When pyannote is not available, uses pause detection
    to create speaker turns (alternating speakers).
    """

    def is_available(self) -> bool:
        """Always available as fallback."""
        return True

    def diarize(
        self,
        audio_path: Path,
        config: DiarizationConfig,
    ) -> List[Tuple[float, float, str]]:
        """
        Simple diarization using silence detection.

        Rule #4: <60 lines.

        Note: This is a basic fallback. Results may not reflect
        actual speaker changes.
        """
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_silence

            audio = AudioSegment.from_file(str(audio_path))
            duration_ms = len(audio)

            # Detect silences
            silences = detect_silence(audio, min_silence_len=1000, silence_thresh=-40)

            segments: List[Tuple[float, float, str]] = []
            current_speaker = "Speaker A"
            last_end = 0.0

            for silence_start, silence_end in silences:
                if silence_start > last_end:
                    # Add segment before silence
                    segments.append(
                        (
                            last_end / 1000.0,
                            silence_start / 1000.0,
                            current_speaker,
                        )
                    )

                    # Alternate speaker after long silence
                    if (silence_end - silence_start) > 2000:  # 2+ second pause
                        current_speaker = (
                            "Speaker B"
                            if current_speaker == "Speaker A"
                            else "Speaker A"
                        )

                last_end = silence_end

            # Add final segment
            if last_end < duration_ms:
                segments.append(
                    (
                        last_end / 1000.0,
                        duration_ms / 1000.0,
                        current_speaker,
                    )
                )

            return segments if segments else [(0.0, duration_ms / 1000.0, "Speaker A")]

        except Exception as e:
            logger.warning(f"Simple diarization failed: {e}")
            # Return single speaker for entire audio
            duration_ms = get_audio_duration_ms(audio_path) or 60000
            return [(0.0, duration_ms / 1000.0, "Speaker A")]


# =============================================================================
# AUDIO DIARIZATION PROCESSOR
# =============================================================================


class AudioDiarizationProcessor:
    """
    Audio processor with speaker diarization support.

    Refactor meeting transcription to include speaker
    identification and timestamp sync.

    Features:
    - Integration with Whisper for high-fidelity transcription
    - Diarization support (Speaker A, Speaker B)
    - 100% of transcript lines have start_ms and end_ms metadata
    - JPL Rule #5: Asserts audio file validity

    Example:
        processor = AudioDiarizationProcessor()
        result = processor.process(Path("meeting.mp3"))

        for segment in result.segments:
            print(f"[{segment.speaker_id}] {segment.start_ms}ms: {segment.text}")
    """

    def __init__(
        self,
        config: Optional[DiarizationConfig] = None,
    ) -> None:
        """
        Initialize processor.

        Args:
            config: Diarization configuration.
        """
        self._config = config or DiarizationConfig()
        self._diarizer: Optional[DiarizationBackend] = None

    def _get_diarizer(self) -> DiarizationBackend:
        """Get or create diarization backend."""
        if self._diarizer is not None:
            return self._diarizer

        # Try pyannote first
        pyannote = PyAnnoteDiarizer(self._config.huggingface_token)
        if pyannote.is_available():
            self._diarizer = pyannote
            logger.info("Using pyannote.audio for speaker diarization")
        else:
            self._diarizer = SimpleDiarizer()
            logger.info("Using simple silence-based speaker detection")

        return self._diarizer

    def is_available(self) -> bool:
        """Check if processor dependencies are available."""
        try:
            from ingestforge.ingest.audio import is_whisper_available

            return is_whisper_available()
        except ImportError:
            return False

    def process(self, audio_path: Path) -> DiarizationResult:
        """
        Process audio file with diarization.

        AC: All criteria.
        Rule #4: <60 lines (delegates to helper methods).
        Rule #5: Assert audio file validity.
        Rule #7: Check all return values.

        Args:
            audio_path: Path to audio file.

        Returns:
            DiarizationResult with speaker-labeled segments.
        """
        # JPL Rule #5: Assert preconditions
        assert audio_path is not None, "Audio path cannot be None"
        assert isinstance(audio_path, Path), "Audio path must be Path object"

        # Validate audio file
        is_valid, error = validate_audio_file(audio_path)
        if not is_valid:
            return DiarizationResult(
                success=False,
                error_message=error,
            )

        # Check availability
        if not self.is_available():
            return DiarizationResult(
                success=False,
                error_message="Whisper not available. Install with: pip install faster-whisper",
            )

        # Process audio
        return self._process_audio(audio_path)

    def _process_audio(self, audio_path: Path) -> DiarizationResult:
        """
        Internal audio processing.

        Rule #4: <60 lines.
        Rule #7: Handle errors gracefully.
        """
        try:
            # Step 1: Transcribe with Whisper
            transcription = self._transcribe(audio_path)
            if not transcription:
                return DiarizationResult(
                    success=False,
                    error_message="Transcription failed",
                )

            # Step 2: Perform diarization
            diarizer = self._get_diarizer()
            speaker_turns = diarizer.diarize(audio_path, self._config)

            # Step 3: Align transcription with speaker turns
            segments = self._align_with_speakers(transcription, speaker_turns)

            # Step 4: Build speaker list
            speakers = self._build_speaker_list(segments)

            # Step 5: Build full text
            full_text = " ".join(seg.text for seg in segments)

            duration_ms = get_audio_duration_ms(audio_path)

            return DiarizationResult(
                segments=segments,
                speakers=speakers,
                full_text=full_text,
                duration_ms=duration_ms,
                language=transcription.get("language", "en"),
                success=True,
            )

        except Exception as e:
            logger.error(f"Diarization failed for {audio_path}: {e}")
            return DiarizationResult(
                success=False,
                error_message=str(e),
            )

    def _transcribe(self, audio_path: Path) -> Optional[Dict[str, Any]]:
        """
        Transcribe audio with Whisper.

        Rule #4: <60 lines.
        Rule #7: Return None on error.
        """
        try:
            from ingestforge.ingest.audio import create_whisper_backend

            backend = create_whisper_backend(
                model=self._config.whisper_model,
                language=self._config.language,
            )

            result = backend.transcribe(audio_path)

            return {
                "text": result.text,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                        "words": [
                            {"word": w.word, "start": w.start, "end": w.end}
                            for w in seg.words
                        ]
                        if seg.words
                        else [],
                    }
                    for seg in result.segments
                ],
                "language": result.language,
            }

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    def _align_with_speakers(
        self,
        transcription: Dict[str, Any],
        speaker_turns: List[Tuple[float, float, str]],
    ) -> List[SpeakerSegment]:
        """
        Align transcription segments with speaker turns.

        AC: 100% of transcript lines have start_ms and end_ms.
        Rule #4: <60 lines.
        """
        segments: List[SpeakerSegment] = []

        for trans_seg in transcription.get("segments", []):
            seg_start = trans_seg["start"]
            seg_end = trans_seg["end"]
            seg_text = trans_seg["text"].strip()

            if not seg_text:
                continue

            # Find matching speaker turn
            speaker_id = self._find_speaker_for_segment(
                seg_start, seg_end, speaker_turns
            )

            # Create segment with ms timestamps (AC)
            segment = SpeakerSegment(
                text=seg_text,
                speaker_id=speaker_id,
                start_ms=int(seg_start * 1000),
                end_ms=int(seg_end * 1000),
                words=trans_seg.get("words", []),
            )
            segments.append(segment)

            # Bound segments (Rule #2)
            if len(segments) >= MAX_SEGMENTS:
                logger.warning(f"Segment limit reached ({MAX_SEGMENTS})")
                break

        return segments

    def _find_speaker_for_segment(
        self,
        start: float,
        end: float,
        speaker_turns: List[Tuple[float, float, str]],
    ) -> str:
        """
        Find the speaker for a transcript segment.

        Rule #4: <60 lines.
        """
        seg_mid = (start + end) / 2

        for turn_start, turn_end, speaker_id in speaker_turns:
            if turn_start <= seg_mid <= turn_end:
                return speaker_id

        # Fallback: find closest turn
        if speaker_turns:
            closest = min(speaker_turns, key=lambda t: abs((t[0] + t[1]) / 2 - seg_mid))
            return closest[2]

        return "Speaker A"

    def _build_speaker_list(
        self,
        segments: List[SpeakerSegment],
    ) -> List[Speaker]:
        """
        Build speaker statistics from segments.

        Map Speaker nodes to PERSON entity type.
        Rule #4: <60 lines.
        """
        speaker_stats: Dict[str, Speaker] = {}

        for seg in segments:
            if seg.speaker_id not in speaker_stats:
                speaker_stats[seg.speaker_id] = Speaker(speaker_id=seg.speaker_id)

            speaker = speaker_stats[seg.speaker_id]
            speaker.total_speaking_time_ms += seg.duration_ms
            speaker.segment_count += 1

        return list(speaker_stats.values())

    def to_artifact(
        self,
        result: DiarizationResult,
        audio_path: Path,
    ) -> "IFAudioArtifact":
        """
        Convert DiarizationResult to IFAudioArtifact.

        Integration with IFAudioArtifact schema.
        Rule #7: Explicit return type.

        Args:
            result: Diarization result to convert.
            audio_path: Path to source audio file.

        Returns:
            IFAudioArtifact with diarized segments.
        """
        from ingestforge.core.pipeline.artifacts import (
            IFAudioArtifact,
            AudioSegment as IFAudioSegment,
        )

        # Convert SpeakerSegment to AudioSegment
        segments = [
            IFAudioSegment(
                speaker_id=seg.speaker_id,
                start_ms=seg.start_ms,
                end_ms=seg.end_ms,
                text=seg.text,
                confidence=seg.confidence,
            )
            for seg in result.segments
        ]

        return IFAudioArtifact(
            audio_path=str(audio_path),
            duration_ms=result.duration_ms,
            full_transcript=result.full_text,
            segments=segments,
            language=result.language,
            word_count=len(result.full_text.split()),
            transcription_model=self._config.whisper_model,
            metadata={
                "speaker_count": result.speaker_count,
                "segment_count": result.segment_count,
                "speakers": [s.speaker_id for s in result.speakers],
            },
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def diarize_audio(
    audio_path: Path,
    whisper_model: str = "base",
    language: Optional[str] = "en",
) -> DiarizationResult:
    """
    Convenience function to diarize audio.

    Simple API for audio diarization.
    Rule #4: <60 lines.

    Args:
        audio_path: Path to audio file.
        whisper_model: Whisper model size.
        language: Language code or None for auto-detect.

    Returns:
        DiarizationResult with speaker-labeled segments.
    """
    config = DiarizationConfig(
        whisper_model=whisper_model,
        language=language,
    )
    processor = AudioDiarizationProcessor(config=config)
    return processor.process(audio_path)


def diarize_to_artifact(
    audio_path: Path,
    whisper_model: str = "base",
    language: Optional[str] = "en",
) -> "IFAudioArtifact":
    """
    Diarize audio and return IFAudioArtifact.

    Direct artifact creation.
    Rule #7: Returns failure artifact on error.

    Args:
        audio_path: Path to audio file.
        whisper_model: Whisper model size.
        language: Language code.

    Returns:
        IFAudioArtifact with diarized segments.
    """
    from ingestforge.core.pipeline.artifacts import (
        IFFailureArtifact,
    )

    config = DiarizationConfig(
        whisper_model=whisper_model,
        language=language,
    )
    processor = AudioDiarizationProcessor(config=config)
    result = processor.process(audio_path)

    if not result.success:
        return IFFailureArtifact(
            error_message=result.error_message,
            metadata={"audio_path": str(audio_path)},
        )

    return processor.to_artifact(result, audio_path)
