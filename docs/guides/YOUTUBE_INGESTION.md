# YouTube Video Ingestion Guide

IngestForge now supports direct ingestion of YouTube video transcripts via the CLI.

## Quick Start

```bash
# Ingest a YouTube video
ingestforge ingest https://youtube.com/watch?v=dQw4w9WgXcQ

# Preview what would be processed (dry run)
ingestforge ingest https://youtube.com/watch?v=dQw4w9WgXcQ --dry-run

# Quiet mode for scripts/CI
ingestforge ingest https://youtube.com/watch?v=dQw4w9WgXcQ --quiet
```

## Supported URL Formats

All standard YouTube URL formats are supported:

- **Standard:** `https://www.youtube.com/watch?v=VIDEO_ID`
- **Short:** `https://youtu.be/VIDEO_ID`
- **Embed:** `https://www.youtube.com/embed/VIDEO_ID`
- **Shorts:** `https://www.youtube.com/shorts/VIDEO_ID`
- **With params:** `https://youtube.com/watch?v=VIDEO_ID&t=120s`

## How It Works

1. **URL Detection**: The CLI automatically detects YouTube URLs
2. **Transcript Extraction**: Uses youtube-transcript-api to fetch transcripts
3. **Chunking**: Creates semantic chunks with timestamp metadata
4. **Storage**: Adds chunks to your knowledge base

## Chunk Metadata

Each YouTube chunk includes rich metadata:

```python
{
    "video_id": "dQw4w9WgXcQ",
    "video_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "thumbnail_url": "https://img.youtube.com/vi/dQw4w9WgXcQ/maxresdefault.jpg",
    "timestamp_start": 0.0,
    "timestamp_end": 60.0,
    "timestamp_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=0s",
    "is_auto_generated": false,
    "language": "en"
}
```

## Citations

When querying content from YouTube videos, citations include:

- Video title and URL
- Timestamp link (click to jump to exact moment)
- Thumbnail image
- Auto-generated vs manual transcript indicator

## Requirements

Install the youtube-transcript-api package:

```bash
pip install youtube-transcript-api
```

## Examples

### Basic Ingestion
```bash
ingestforge ingest https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

### Check What Would Happen
```bash
ingestforge ingest https://youtu.be/dQw4w9WgXcQ --dry-run
```

### Use with Specific Project
```bash
ingestforge ingest https://youtube.com/watch?v=dQw4w9WgXcQ --project /path/to/project
```

### Quiet Mode for Scripts
```bash
ingestforge ingest https://youtube.com/watch?v=dQw4w9WgXcQ --quiet
```

## Advanced Usage

### Custom Chunk Duration

Modify the chunk duration by using the YouTube processor directly:

```python
from ingestforge.ingest.youtube import YouTubeProcessor

# Create processor with 2-minute chunks
processor = YouTubeProcessor(chunk_duration=120.0)
result = processor.process("https://youtube.com/watch?v=VIDEO_ID")

# Convert to ChunkRecords
records = processor.to_chunk_records(result, "my_doc_id")
```

### Language Selection

```python
# Prefer Spanish transcripts
processor = YouTubeProcessor(language="es")
```

### Disable Auto-Generated Transcripts

```python
# Only use manual transcripts
processor = YouTubeProcessor(include_auto_generated=False)
```

## Error Handling

Common errors and solutions:

### "youtube-transcript-api not installed"
**Solution:** `pip install youtube-transcript-api`

### "No transcript available"
**Solution:** Video may not have transcripts enabled. Try a different video.

### "Invalid YouTube URL"
**Solution:** Ensure URL is a valid YouTube video link.

## Integration with Other Commands

After ingesting a YouTube video, you can:

### Query the Content
```bash
ingestforge query "What was discussed about X?"
```

### Export to Markdown
```bash
ingestforge export markdown output.md
```

### Study the Content
```bash
ingestforge study quiz --topic "YouTube content"
```

## Performance Notes

- Transcript extraction is network-bound
- Embedding generation depends on available VRAM/RAM
- Average processing time: 5-30 seconds per video
- Chunk count depends on video length (default: 60 seconds per chunk)

## Troubleshooting

### Slow Processing
- Check network connection
- Verify VRAM/RAM availability
- Consider using `--quiet` to reduce console overhead

### Missing Transcripts
- Not all videos have transcripts
- Some videos only have auto-generated transcripts
- Regional restrictions may apply

### Character Encoding Issues
- Most transcripts use UTF-8
- Non-English transcripts are fully supported
- Emoji and special characters are preserved

## See Also

- [YouTube Processor API](../api/youtube.md)
- [Ingest Command Reference](../cli/ingest.md)
- [Citation System](../guides/citations.md)
