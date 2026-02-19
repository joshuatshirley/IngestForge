"""YouTube transcript ingestion module.

Provides YouTube video transcript extraction using youtube-transcript-api.
Creates timestamped chunks suitable for citation and retrieval.

Architecture Position
---------------------
    CLI (outermost)
      -> Feature Modules (you are here)
            -> Shared (patterns, interfaces, utilities)
                  -> Core (innermost)

Key Components
--------------
**YouTubeProcessor**
    Main processor for extracting transcripts from YouTube videos.
    Handles URL parsing, transcript fetching, and chunk creation.

**TranscriptSegment**
    Represents a segment of transcript with timestamp metadata.

**YouTubeMetadata**
    Metadata about the YouTube video (id, url, title, etc.).

Usage Example
-------------
    from ingestforge.ingest.youtube import YouTubeProcessor

    processor = YouTubeProcessor()
    result = processor.process("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    for chunk in result.chunks:
        print(f"{chunk.timestamp_start}s: {chunk.content[:50]}...")
"""

from ingestforge.ingest.youtube.processor import (
    YouTubeProcessor,
    TranscriptSegment,
    YouTubeMetadata,
    YouTubeProcessingResult,
    is_youtube_available,
    extract_video_id,
)

__all__ = [
    "YouTubeProcessor",
    "TranscriptSegment",
    "YouTubeMetadata",
    "YouTubeProcessingResult",
    "is_youtube_available",
    "extract_video_id",
]
