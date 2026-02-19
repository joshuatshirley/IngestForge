"""Audio ingestion utilities for transcription and processing.

This package provides audio processing functionality:
- whisper_backend: Faster-whisper based transcription (P3-SOURCE-002.1)
- segmenter: Pause-based audio segmentation (P3-SOURCE-002.2)
- processor: High-level audio processor (TICKET-201)
- diarizer: Speaker diarization processor ()
"""

from ingestforge.ingest.audio.whisper_backend import (
    WhisperBackend,
    WhisperModel,
    ComputeType,
    TranscriptionConfig,
    TranscriptionSegment,
    TranscriptionResult,
    WordInfo,
    create_whisper_backend,
    transcribe_audio,
    is_whisper_available,
)
from ingestforge.ingest.audio.segmenter import (
    AudioSegmenter,
    AudioSegment,
    SegmentConfig,
    SegmentationResult,
    create_segmenter,
    segment_audio,
    is_segmenter_available,
)
from ingestforge.ingest.audio.processor import (
    AudioProcessor,
    AudioChunk,
    AudioMetadata,
    AudioProcessingResult,
)
from ingestforge.ingest.audio.diarizer import (
    AudioDiarizationProcessor,
    DiarizationConfig,
    DiarizationResult,
    SpeakerSegment,
    Speaker,
    validate_audio_file,
)

__all__ = [
    # Whisper Backend (P3-SOURCE-002.1)
    "WhisperBackend",
    "WhisperModel",
    "ComputeType",
    "TranscriptionConfig",
    "TranscriptionSegment",
    "TranscriptionResult",
    "WordInfo",
    "create_whisper_backend",
    "transcribe_audio",
    "is_whisper_available",
    # Audio Segmenter (P3-SOURCE-002.2)
    "AudioSegmenter",
    "AudioSegment",
    "SegmentConfig",
    "SegmentationResult",
    "create_segmenter",
    "segment_audio",
    "is_segmenter_available",
    # Audio Processor (TICKET-201)
    "AudioProcessor",
    "AudioChunk",
    "AudioMetadata",
    "AudioProcessingResult",
    # Diarization Processor ()
    "AudioDiarizationProcessor",
    "DiarizationConfig",
    "DiarizationResult",
    "SpeakerSegment",
    "Speaker",
    "validate_audio_file",
]
