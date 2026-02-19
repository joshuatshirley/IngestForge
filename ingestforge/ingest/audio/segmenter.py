"""Audio Segmenter for pre-processing large audio files.

Provides pause-based segmentation to avoid memory floods
when processing long audio recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
MAX_SEGMENT_DURATION_SECONDS = 600  # 10 minutes max per segment
MIN_SEGMENT_DURATION_SECONDS = 5  # At least 5 seconds
MAX_SEGMENTS = 100
DEFAULT_SEGMENT_DURATION = 300  # 5 minutes default
MIN_SILENCE_DURATION = 0.3  # 300ms silence threshold
SILENCE_THRESHOLD_DB = -40


@dataclass
class AudioSegment:
    """A segment of audio."""

    index: int
    start_time: float
    end_time: float
    path: Optional[Path] = None
    duration: float = 0.0

    def __post_init__(self) -> None:
        """Calculate duration."""
        self.duration = self.end_time - self.start_time


@dataclass
class SegmentConfig:
    """Configuration for audio segmentation."""

    max_segment_duration: float = DEFAULT_SEGMENT_DURATION
    min_silence_duration: float = MIN_SILENCE_DURATION
    silence_threshold_db: float = SILENCE_THRESHOLD_DB
    overlap_seconds: float = 0.5
    output_format: str = "wav"

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        self.max_segment_duration = min(
            self.max_segment_duration, MAX_SEGMENT_DURATION_SECONDS
        )
        self.max_segment_duration = max(
            self.max_segment_duration, MIN_SEGMENT_DURATION_SECONDS
        )


@dataclass
class SegmentationResult:
    """Result of audio segmentation."""

    segments: List[AudioSegment] = field(default_factory=list)
    total_duration: float = 0.0
    source_path: Optional[Path] = None
    output_dir: Optional[Path] = None

    @property
    def segment_count(self) -> int:
        """Get number of segments."""
        return len(self.segments)


class AudioSegmenter:
    """Segments audio files for efficient processing.

    Uses pause detection to find natural break points
    and splits audio into manageable chunks.
    """

    def __init__(
        self,
        config: Optional[SegmentConfig] = None,
    ) -> None:
        """Initialize the segmenter.

        Args:
            config: Segmentation configuration
        """
        self.config = config or SegmentConfig()
        self._pydub_available = self._check_pydub()

    def _check_pydub(self) -> bool:
        """Check if pydub is available."""
        try:
            import pydub  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def is_available(self) -> bool:
        """Check if segmenter dependencies are available."""
        return self._pydub_available

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Duration in seconds, or 0 if failed
        """
        if not self._pydub_available:
            return 0.0

        try:
            from pydub import AudioSegment as PydubSegment

            audio = PydubSegment.from_file(str(audio_path))
            return len(audio) / 1000.0  # Convert ms to seconds

        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            return 0.0

    def detect_silences(self, audio_path: Path) -> List[Tuple[float, float]]:
        """Detect silence regions in audio.

        Args:
            audio_path: Path to audio file

        Returns:
            List of (start, end) tuples for silent regions
        """
        if not self._pydub_available:
            return []

        try:
            from pydub import AudioSegment as PydubSegment
            from pydub.silence import detect_silence

            audio = PydubSegment.from_file(str(audio_path))

            # Detect silences (returns milliseconds)
            silences_ms = detect_silence(
                audio,
                min_silence_len=int(self.config.min_silence_duration * 1000),
                silence_thresh=self.config.silence_threshold_db,
            )

            # Convert to seconds
            return [(start / 1000.0, end / 1000.0) for start, end in silences_ms]

        except Exception as e:
            logger.error(f"Silence detection failed: {e}")
            return []

    def segment(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
    ) -> SegmentationResult:
        """Segment audio file.

        Args:
            audio_path: Path to audio file
            output_dir: Optional directory for segment files

        Returns:
            SegmentationResult with segments
        """
        if not audio_path.exists():
            logger.error(f"Audio file not found: {audio_path}")
            return SegmentationResult(source_path=audio_path)

        if not self._pydub_available:
            logger.error("pydub not available for segmentation")
            return SegmentationResult(source_path=audio_path)

        # Get duration
        duration = self.get_audio_duration(audio_path)
        if duration <= 0:
            return SegmentationResult(source_path=audio_path)

        # If short enough, return single segment
        if duration <= self.config.max_segment_duration:
            segment = AudioSegment(
                index=0,
                start_time=0.0,
                end_time=duration,
                path=audio_path,
            )
            return SegmentationResult(
                segments=[segment],
                total_duration=duration,
                source_path=audio_path,
            )

        # Find split points using silence detection
        split_points = self._find_split_points(audio_path, duration)

        # Create segments
        segments = self._create_segments(split_points, duration)

        # Export segments if output_dir provided
        if output_dir:
            segments = self._export_segments(audio_path, segments, output_dir)

        return SegmentationResult(
            segments=segments,
            total_duration=duration,
            source_path=audio_path,
            output_dir=output_dir,
        )

    def _find_split_points(self, audio_path: Path, duration: float) -> List[float]:
        """Find optimal split points.

        Args:
            audio_path: Path to audio file
            duration: Total audio duration

        Returns:
            List of split times in seconds
        """
        split_points = [0.0]

        # Detect silences for natural break points
        silences = self.detect_silences(audio_path)

        # Group silences into potential split points
        target_duration = self.config.max_segment_duration
        current_pos = 0.0

        while current_pos < duration - MIN_SEGMENT_DURATION_SECONDS:
            next_target = current_pos + target_duration

            # Find best silence near target
            best_split = self._find_best_silence(
                silences, current_pos, next_target, duration
            )

            if best_split and best_split > current_pos + MIN_SEGMENT_DURATION_SECONDS:
                split_points.append(best_split)
                current_pos = best_split
            else:
                # No silence found, split at target
                split_point = min(next_target, duration)
                if split_point > current_pos + MIN_SEGMENT_DURATION_SECONDS:
                    split_points.append(split_point)
                current_pos = split_point
            if len(split_points) >= MAX_SEGMENTS:
                break

        return split_points

    def _find_best_silence(
        self,
        silences: List[Tuple[float, float]],
        start: float,
        target: float,
        max_time: float,
    ) -> Optional[float]:
        """Find best silence near target time.

        Args:
            silences: List of silence regions
            start: Start of search window
            target: Target split time
            max_time: Maximum allowed time

        Returns:
            Best split time or None
        """
        # Search window around target
        window_start = target - 30  # 30 seconds before
        window_end = min(target + 30, max_time)  # 30 seconds after

        best_silence = None
        best_distance = float("inf")

        for silence_start, silence_end in silences:
            # Check if silence is in window and after start
            if silence_start < start:
                continue
            if silence_start < window_start or silence_start > window_end:
                continue

            # Use middle of silence as split point
            silence_mid = (silence_start + silence_end) / 2
            distance = abs(silence_mid - target)

            if distance < best_distance:
                best_distance = distance
                best_silence = silence_mid

        return best_silence

    def _create_segments(
        self, split_points: List[float], duration: float
    ) -> List[AudioSegment]:
        """Create segment objects from split points.

        Args:
            split_points: List of split times
            duration: Total duration

        Returns:
            List of AudioSegment
        """
        segments: List[AudioSegment] = []

        for i, start in enumerate(split_points):
            # Determine end time
            if i + 1 < len(split_points):
                end = split_points[i + 1]
            else:
                end = duration

            # Add overlap for context
            if i > 0:
                start = max(0, start - self.config.overlap_seconds)

            segment = AudioSegment(
                index=i,
                start_time=start,
                end_time=end,
            )
            segments.append(segment)

        return segments

    def _export_segments(
        self,
        audio_path: Path,
        segments: List[AudioSegment],
        output_dir: Path,
    ) -> List[AudioSegment]:
        """Export segments to files.

        Args:
            audio_path: Source audio file
            segments: Segment definitions
            output_dir: Output directory

        Returns:
            Segments with updated paths
        """
        if not self._pydub_available:
            return segments

        try:
            from pydub import AudioSegment as PydubSegment

            audio = PydubSegment.from_file(str(audio_path))
            output_dir.mkdir(parents=True, exist_ok=True)

            for segment in segments:
                # Extract segment (pydub uses milliseconds)
                start_ms = int(segment.start_time * 1000)
                end_ms = int(segment.end_time * 1000)
                chunk = audio[start_ms:end_ms]

                # Export
                filename = f"segment_{segment.index:03d}.{self.config.output_format}"
                output_path = output_dir / filename
                chunk.export(str(output_path), format=self.config.output_format)

                segment.path = output_path
                logger.debug(f"Exported segment {segment.index} to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export segments: {e}")

        return segments


def create_segmenter(
    max_duration: float = DEFAULT_SEGMENT_DURATION,
    silence_threshold: float = SILENCE_THRESHOLD_DB,
) -> AudioSegmenter:
    """Factory function to create segmenter.

    Args:
        max_duration: Maximum segment duration in seconds
        silence_threshold: Silence detection threshold in dB

    Returns:
        Configured AudioSegmenter
    """
    config = SegmentConfig(
        max_segment_duration=max_duration,
        silence_threshold_db=silence_threshold,
    )
    return AudioSegmenter(config=config)


def segment_audio(
    audio_path: Path,
    output_dir: Optional[Path] = None,
    max_duration: float = DEFAULT_SEGMENT_DURATION,
) -> SegmentationResult:
    """Convenience function to segment audio.

    Args:
        audio_path: Path to audio file
        output_dir: Optional output directory
        max_duration: Maximum segment duration

    Returns:
        SegmentationResult
    """
    segmenter = create_segmenter(max_duration=max_duration)
    return segmenter.segment(audio_path, output_dir)


def is_segmenter_available() -> bool:
    """Check if segmenter is available.

    Returns:
        True if pydub is installed
    """
    try:
        import pydub  # noqa: F401

        return True
    except ImportError:
        return False
