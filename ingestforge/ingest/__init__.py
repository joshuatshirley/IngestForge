"""
Document Ingestion Module.

This module handles the first two stages of the pipeline: Split and Extract.
It transforms raw documents (PDF, HTML, DOCX, etc.) into plain text with
metadata ready for chunking.

Architecture Position
---------------------
    CLI (outermost)
      └── **Feature Modules** (you are here)
            └── Shared (patterns, interfaces, utilities)
                  └── Core (innermost)

Pipeline Stage: 1-2 (Split → Extract)

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │   Raw Document  │────→│    Split        │────→│    Extract      │
    │   (PDF/HTML)    │     │  (chapters)     │     │  (plain text)   │
    └─────────────────┘     └─────────────────┘     └─────────────────┘

Key Components
--------------
**PDFSplitter**
    Splits large PDFs into chapter-level files using table of contents.
    Enables parallel processing and better citation granularity.

**TextExtractor**
    Extracts plain text from various formats (PDF, EPUB, DOCX, TXT, MD).
    Handles encoding issues and preserves structural markers.

**DocumentProcessor**
    High-level processor that routes files to appropriate handlers.
    Coordinates the full document-to-text transformation.

**DirectoryWatcher**
    Monitors the pending directory for new documents.
    Triggers automatic ingestion when files are dropped.

Supporting Components
---------------------
- HTMLProcessor: Web page extraction with metadata (trafilatura)
- OCRProcessor: Image-to-text using Tesseract OCR
- GoogleSlidesParser: Extract content from Google Slides exports
- CitationMetadataExtractor: Parse citation info from academic PDFs
- StructureExtractor: Extract document structure (chapters, sections)
- YouTubeProcessor: YouTube video transcript extraction with timestamps
- AudioProcessor: Audio transcription with Whisper (TICKET-201)

File Format Support
-------------------
| Format | Extension | Processor |
|--------|-----------|-----------|
| PDF | .pdf | PDFSplitter + TextExtractor |
| HTML | .html, .htm | HTMLProcessor |
| Word | .docx | TextExtractor (python-docx) |
| E-book | .epub | TextExtractor (ebooklib) |
| Markdown | .md | TextExtractor (direct read) |
| Plain text | .txt | TextExtractor (direct read) |
| Images | .png, .jpg | OCRProcessor (pytesseract) |
| YouTube | URL | YouTubeProcessor (youtube-transcript-api) |
| Audio | .mp3, .wav, .m4a, .flac | AudioProcessor (faster-whisper) |

Usage Example
-------------
    from ingestforge.ingest import PDFSplitter, TextExtractor

    # Split a large PDF
    splitter = PDFSplitter(config)
    chapters = splitter.split(Path("textbook.pdf"), "doc_123")

    # Extract text from each chapter
    extractor = TextExtractor(config)
    for chapter in chapters:
        text = extractor.extract(chapter)
        # text is now ready for chunking

    # Or use high-level processor
    processor = DocumentProcessor(config)
    result = processor.process(Path("document.pdf"))
    print(result.text)

YouTube Videos
--------------
    # Process YouTube video transcripts
    from ingestforge.ingest.youtube import YouTubeProcessor

    yt_processor = YouTubeProcessor()
    result = yt_processor.process("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # Get timestamped chunks
    for chunk in result.chunks:
        print(f"[{chunk.timestamp_start}s] {chunk.content[:50]}...")

    # Convert to ChunkRecords for storage
    records = yt_processor.to_chunk_records(result, "video_001")

Audio Files (TICKET-201)
------------------------
    # Process audio files with Whisper transcription
    from ingestforge.ingest.audio import AudioProcessor

    audio_processor = AudioProcessor(whisper_model="base", language="en")
    result = audio_processor.process(Path("lecture.mp3"))

    # Get timestamped chunks
    for chunk in result.chunks:
        print(f"[{chunk.start_formatted}] {chunk.content[:50]}...")

    # Convert to ChunkRecords for storage
    records = audio_processor.to_chunk_records(result, "audio_001")

Watch Mode
----------
    # Start watching for new documents
    watcher = DirectoryWatcher(config)
    watcher.start()  # Blocks, processes files as they appear
"""

from ingestforge.ingest.pdf_splitter import PDFSplitter
from ingestforge.ingest.text_extractor import TextExtractor
from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.ingest.watcher import DirectoryWatcher

__all__ = ["PDFSplitter", "TextExtractor", "DocumentProcessor", "DirectoryWatcher"]
