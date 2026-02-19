# Ingest Module

## Purpose

Document processing - extract text and metadata from various formats (PDF, HTML, EPUB, DOCX, images). Provides processors for each format, intelligent content extraction, OCR for scanned documents, and automatic directory watching.

## Architecture Context

The `ingest/` module is the entry point of the document processing pipeline. It converts raw documents into clean text that can be chunked and enriched.

```
┌─────────────────────────────────────────┐
│   User drops document in pending/       │
│            ↓                             │
│   ingest/ - Document Processing         │  ← You are here
│   (PDF, HTML, OCR, text extraction)     │
│            ↓                             │
│   chunking/ - Split into chunks         │
│            ↓                             │
│   enrichment/ - Add metadata            │
└─────────────────────────────────────────┘
```

## Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| `processor.py` | DocumentProcessor - route to format-specific processors | ✅ Complete |
| `text_extractor.py` | TextExtractor - extract text from PDF, EPUB, DOCX | ✅ Complete |
| `html_processor.py` | HTMLProcessor - smart HTML article extraction | ✅ Complete |
| `ocr_processor.py` | OCRProcessor - OCR for scanned PDFs/images | ✅ Complete |
| `pdf_splitter.py` | PDFSplitter - split PDFs by chapters/TOC | ✅ Complete |
| `watcher.py` | DirectoryWatcher - monitor pending/ for new files | ✅ Complete |
| `formats.py` | Format detection and validation | ✅ Complete |
| `structure_extractor.py` | Extract document structure (headings, sections) | ✅ Complete |

## Data Flow

```
Document File → DocumentProcessor
                      ↓
              ┌───────┴────────┐
              │  Route by type │
              └───────┬────────┘
       ┌──────┬──────┼──────┬──────┐
       ↓      ↓      ↓      ↓      ↓
     PDF   HTML   EPUB  DOCX  Image
       ↓      ↓      ↓      ↓      ↓
   Extract Extract Extract Extract OCR
       ↓      ↓      ↓      ↓      ↓
       └──────┴──────┴──────┴──────┘
                      ↓
            Clean Text + Metadata
```

## DocumentProcessor

**Purpose:** Main dispatcher that routes documents to appropriate processors based on file type.

**Usage:**

```python
from ingestforge.core import load_config
from ingestforge.ingest import DocumentProcessor
from pathlib import Path

config = load_config
processor = DocumentProcessor(config)

# Process any supported document
result = processor.process(
    file_path=Path("document.pdf"),
    document_id="doc-001"
)

print(f"Type: {result.file_type}")
print(f"Chapters: {len(result.chapters)}")
print(f"Text: {result.texts[0][:100]}...")
print(f"Metadata: {result.metadata}")

# Access citation info
if result.source_location:
    cite = result.source_location.to_short_cite
    print(f"Citation: {cite}")
```

**ProcessedDocument Structure:**

```python
@dataclass
class ProcessedDocument:
    document_id: str                        # Unique identifier
    source_file: str                        # Original file path
    file_type: str                          # pdf, html, epub, etc.
    chapters: List[Path]                    # Chapter files (for PDFs)
    texts: List[str]                        # Extracted text per chapter
    metadata: Dict[str, Any]                # Title, author, etc.
    source_location: Optional[SourceLocation]  # For citations
```

**Supported Formats:**

| Format | Extensions | Processor | Features |
|--------|-----------|-----------|----------|
| PDF | `.pdf` | PDFSplitter + TextExtractor | TOC-based splitting, metadata extraction |
| HTML | `.html`, `.htm`, `.mhtml` | HTMLProcessor | Article extraction, structure parsing |
| EPUB | `.epub` | TextExtractor | Chapter detection, metadata |
| DOCX | `.docx` | TextExtractor | Formatting preservation |
| Text | `.txt`, `.md` | TextExtractor | Direct read with encoding fallback |
| Images | `.png`, `.jpg`, `.tiff`, `.bmp` | OCRProcessor | Tesseract OCR |

## TextExtractor

**Purpose:** Extract clean text from various document formats.

**Features:**
- PDF text extraction with PyMuPDF (preserves structure)
- EPUB chapter-by-chapter extraction
- DOCX with formatting preservation
- Text file reading with encoding fallback
- Advanced text cleaning (ligatures, hyphenation, spacing)

**Usage:**

```python
from ingestforge.ingest import TextExtractor

extractor = TextExtractor(config)

# Extract from PDF
text = extractor.extract(Path("paper.pdf"))

# Extract from EPUB
text = extractor.extract(Path("book.epub"))

# Handles encoding automatically
text = extractor.extract(Path("legacy.txt"))  # Tries multiple encodings
```

**Text Cleaning:**

The extractor automatically:
- Fixes ligatures (ﬁ → fi, ﬂ → fl)
- Removes hyphenation at line breaks
- Normalizes whitespace
- Preserves paragraph structure
- Removes headers/footers (PDF)

## HTMLProcessor

**Purpose:** Extract article content from HTML files with intelligent content detection.

**Features:**
- Smart article extraction (filters ads, navigation, etc.)
- Metadata extraction (title, author, date)
- Structure preservation (headings, sections)
- Table of contents generation
- Citation-ready SourceLocation

**Usage:**

```python
from ingestforge.ingest.html_processor import HTMLProcessor

processor = HTMLProcessor(
    include_tables=True,
    include_links=True,
    favor_precision=True  # Fewer false positives
)

# Process HTML file
result = processor.process(Path("article.html"))

# Access content
print(result.title)
print(result.markdown)  # Markdown formatted

# Access metadata
print(f"Authors: {result.authors}")
print(f"Published: {result.publication_date}")
print(f"Site: {result.site_name}")

# Access structure
for section in result.sections:
    print(f"  {'  ' * (section.level - 1)}{section.title}")

# Get citation
cite = result.source_location.to_short_cite
print(f"Cite as: {cite}")
```

**Extracted Metadata:**

```python
@dataclass
class ExtractedHTML:
    # Content
    text: str                    # Clean text
    markdown: str                # Markdown formatted
    html_clean: str              # Cleaned HTML

    # Metadata
    title: str
    authors: List[str]
    publication_date: Optional[str]
    description: Optional[str]
    site_name: Optional[str]
    url: Optional[str]
    language: Optional[str]

    # Structure
    sections: List[HTMLSection]
    headings: List[Dict[str, Any]]

    # Provenance
    source_location: SourceLocation
```

**Smart Extraction:**

Uses trafilatura to:
- Identify main article content
- Remove boilerplate (ads, navigation, footers)
- Extract metadata from HTML meta tags
- Preserve article structure
- Handle various article layouts

## OCRProcessor

**Purpose:** Extract text from scanned PDFs and images using Tesseract OCR.

**Features:**
- Scanned PDF detection
- Multi-page OCR processing
- Confidence scoring
- Graceful fallback if Tesseract not installed
- Progress tracking

**Usage:**

```python
from ingestforge.ingest.ocr_processor import OCRProcessor

processor = OCRProcessor

# Check if OCR is available
if processor.tesseract_available:
    # Check if PDF needs OCR
    if processor.is_scanned_pdf(Path("scan.pdf")):
        # Perform OCR
        result = processor.process_pdf(Path("scan.pdf"))

        print(f"Extracted: {len(result.text)} chars")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Pages: {result.page_count}")

        # Access per-page text
        for i, page_text in enumerate(result.pages):
            print(f"Page {i+1}: {page_text[:100]}...")
else:
    print("Tesseract not available - OCR disabled")
```

**OCRResult Structure:**

```python
@dataclass
class OCRResult:
    text: str              # Full extracted text
    pages: List[str]       # Text per page
    confidence: float      # Average confidence 0-1
    is_scanned: bool       # Whether OCR was needed
    page_count: int
```

**Detection Logic:**

```python
# Determines if PDF is scanned
is_scanned = processor.is_scanned_pdf(Path("document.pdf"))

# Criteria: < 100 chars per page suggests scanned
# Checks first 3 pages for speed
```

## PDFSplitter

**Purpose:** Split PDFs into chapters based on table of contents.

**Features:**
- TOC-based chapter detection
- Metadata extraction (title, author, keywords)
- Fallback to single file if no TOC
- Smart chapter naming
- Configurable minimum chapter size

**Usage:**

```python
from ingestforge.ingest import PDFSplitter

splitter = PDFSplitter(config)

# Get metadata first
metadata = splitter.get_metadata(Path("book.pdf"))
print(f"Title: {metadata.get('title')}")
print(f"Author: {metadata.get('author')}")

# Split into chapters
chapters = splitter.split(
    file_path=Path("book.pdf"),
    document_id="book-001"
)

print(f"Created {len(chapters)} chapters:")
for chapter_path in chapters:
    print(f"  - {chapter_path.name}")
```

**Configuration:**

```yaml
# config.yaml
split:
  use_toc: true                # Use TOC for splitting
  deep_split: false            # Split subsections
  min_chapter_size_kb: 5       # Minimum chapter size
  fallback_single_file: true   # Fallback if no TOC
```

**Chapter Naming:**

```
book-001_chapter_00_introduction.pdf
book-001_chapter_01_background.pdf
book-001_chapter_02_methods.pdf
```

## DirectoryWatcher

**Purpose:** Monitor `.ingest/pending/` directory for new documents and trigger processing automatically.

**Features:**
- Cross-platform file watching (watchdog library)
- Polling fallback if watchdog unavailable
- Duplicate detection
- Format filtering
- Graceful shutdown

**Usage:**

```python
from ingestforge.ingest import DirectoryWatcher
from pathlib import Path

def process_new_file(file_path: Path):
    """Called when new file appears."""
    print(f"Processing {file_path}")
    # ... process document ...

# Create watcher
watcher = DirectoryWatcher(
    config=config,
    on_new_file=process_new_file
)

# Start watching (non-blocking)
watcher.start

# ... do other work ...

# Stop watching
watcher.stop
```

**Watched Directory:**

```
.ingest/
├── pending/          ← Watched directory
│   └── new_doc.pdf   ← Triggers callback
├── processing/       ← Move here during processing
└── completed/        ← Move here when done
```

**Workflow:**

1. User drops file in `pending/`
2. Watcher detects file
3. Calls `on_new_file(file_path)`
4. File is processed
5. File moved to `completed/` (if config.ingest.move_completed=true)

## Extension Points

### Adding a New Processor

Implement `IProcessor` interface from `shared/patterns`:

```python
from ingestforge.shared.patterns import IProcessor, ExtractedContent
from pathlib import Path

class EPUBProcessor(IProcessor):
    """Process EPUB files."""

    def can_process(self, file_path: Path) -> bool:
        return file_path.suffix.lower == ".epub"

    def process(self, file_path: Path) -> ExtractedContent:
        # Extract text from EPUB
        import ebooklib
        from ebooklib import epub

        book = epub.read_epub(str(file_path))

        # Extract text from all chapters
        chapters = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            chapters.append(item.get_content.decode('utf-8'))

        text = "\n\n".join(chapters)

        # Extract metadata
        metadata = {
            'title': book.get_metadata('DC', 'title'),
            'author': book.get_metadata('DC', 'creator'),
        }

        return ExtractedContent(
            text=text,
            metadata=metadata
        )

    def get_supported_extensions(self):
        return [".epub"]

# Register with DocumentProcessor
processor = DocumentProcessor(config)
processor.register_processor(EPUBProcessor)
```

### Custom Text Cleaning

Override `_clean_text_advanced` in TextExtractor:

```python
class CustomTextExtractor(TextExtractor):
    def _clean_text_advanced(self, text: str) -> str:
        # Call parent cleaning
        text = super._clean_text_advanced(text)

        # Add custom cleaning
        text = text.replace("®", "")  # Remove trademark symbols
        text = re.sub(r'\[citation needed\]', '', text)

        return text
```

### Custom Watcher Actions

```python
def custom_file_handler(file_path: Path):
    """Custom processing logic."""
    # Validate file
    if file_path.stat.st_size > 100_000_000:  # 100MB
        logger.warning(f"File too large: {file_path}")
        return

    # Process based on type
    if file_path.suffix == ".pdf":
        process_pdf(file_path)
    elif file_path.suffix == ".html":
        process_html(file_path)

watcher = DirectoryWatcher(config, on_new_file=custom_file_handler)
watcher.start
```

## Dependencies

### Required
- `pymupdf>=1.23.0` - PDF processing

### Optional
- `trafilatura>=1.6.0` - HTML article extraction (for `HTMLProcessor`)
- `pytesseract>=0.3.10` - OCR (for `OCRProcessor`)
- `tesseract` - Tesseract OCR engine (system dependency)
- `python-docx>=0.8.11` - DOCX processing
- `ebooklib>=0.18` - EPUB processing
- `watchdog>=3.0.0` - File system watching

### Installation

```bash
# Minimal (PDF only)
pip install pymupdf

# Full features
pip install pymupdf trafilatura pytesseract python-docx ebooklib watchdog

# System dependencies
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr

# macOS:
brew install tesseract

# Windows:
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

## Testing

### Running Tests

```bash
# Run all ingest tests
pytest tests/test_ingest_*.py -v

# Test specific processor
pytest tests/test_ingest_pdf.py -v
pytest tests/test_ingest_html.py -v
pytest tests/test_ingest_ocr.py -v

# Test with coverage
pytest tests/test_ingest_*.py --cov=ingestforge.ingest --cov-report=html
```

### Key Test Files

- `tests/test_ingest_processor.py` - DocumentProcessor routing
- `tests/test_ingest_text_extractor.py` - Text extraction
- `tests/test_ingest_html.py` - HTML processing
- `tests/test_ingest_ocr.py` - OCR processing
- `tests/test_ingest_watcher.py` - Directory watching

## Common Patterns

### Pattern 1: Process Document with Error Handling

```python
from ingestforge.ingest import DocumentProcessor
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)
processor = DocumentProcessor(config)

try:
    result = processor.process(file_path, document_id)
    logger.info(
        "Document processed",
        document_id=document_id,
        chapters=len(result.chapters),
        text_length=sum(len(t) for t in result.texts)
    )
except ValueError as e:
    logger.error(f"Unsupported format: {e}")
except Exception as e:
    logger.error(f"Processing failed: {e}")
```

### Pattern 2: OCR with Confidence Threshold

```python
from ingestforge.ingest.ocr_processor import OCRProcessor

processor = OCRProcessor

if processor.tesseract_available:
    result = processor.process_pdf(file_path)

    if result.confidence < 0.7:
        logger.warning(
            f"Low OCR confidence: {result.confidence:.1%}",
            document=file_path.name
        )
        # Maybe skip or flag for manual review
    else:
        # Proceed with text
        process_text(result.text)
```

### Pattern 3: HTML with Citation

```python
from ingestforge.ingest.html_processor import HTMLProcessor

processor = HTMLProcessor
result = processor.process(Path("article.html"))

# Store source location with chunks
for chunk in chunks:
    chunk.metadata["source"] = result.source_location.to_dict
    chunk.metadata["cite"] = result.source_location.to_short_cite

# Later, generate citations
for chunk in search_results:
    source = SourceLocation.from_dict(chunk.metadata["source"])
    print(f"Source: {source.to_citation(CitationStyle.APA)}")
```

### Pattern 4: Automatic Ingestion Pipeline

```python
from ingestforge.ingest import DirectoryWatcher, DocumentProcessor
from ingestforge.chunking import SemanticChunker
from ingestforge.storage import ChromaDBStorage

# Setup components
processor = DocumentProcessor(config)
chunker = SemanticChunker(config)
storage = ChromaDBStorage(config)

def ingest_pipeline(file_path: Path):
    """Full ingestion pipeline."""
    # 1. Process document
    doc = processor.process(file_path, str(file_path.stem))

    # 2. Chunk text
    all_chunks = []
    for text in doc.texts:
        chunks = chunker.chunk(text, doc.document_id)
        all_chunks.extend(chunks)

    # 3. Store chunks
    storage.save_chunks(all_chunks)

    logger.info(f"Ingested {len(all_chunks)} chunks from {file_path.name}")

# Start watching
watcher = DirectoryWatcher(config, on_new_file=ingest_pipeline)
watcher.start
```

## Troubleshooting

### Issue 1: PyMuPDF Import Error

**Symptom:** `ImportError: PyMuPDF is required for PDF processing`

**Cause:** pymupdf not installed

**Fix:**

```bash
pip install pymupdf
```

### Issue 2: Tesseract Not Found

**Symptom:** `tesseract_available` returns False

**Cause:** Tesseract OCR engine not installed on system

**Fix:**

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

### Issue 3: HTML Extraction Returns Empty

**Symptom:** `HTMLProcessor` returns empty text

**Cause:** Page structure not recognized, or trafilatura not installed

**Fix:**

```bash
# Install trafilatura
pip install trafilatura

# Try with different settings
processor = HTMLProcessor(favor_precision=False)  # More permissive
```

### Issue 4: Encoding Errors in Text Files

**Symptom:** `UnicodeDecodeError` when reading text files

**Cause:** File uses non-UTF-8 encoding

**Fix:**

```python
# TextExtractor already handles this with fallback
# But if using direct read:
from ingestforge.shared import read_text_with_fallback

text = read_text_with_fallback(Path("file.txt"))
# Tries: utf-8, utf-8-sig, latin-1, cp1252, then fallback
```

### Issue 5: Watcher Not Detecting Files

**Symptom:** DirectoryWatcher doesn't trigger on new files

**Cause:** Watchdog not installed, or polling too slow

**Fix:**

```bash
# Install watchdog for better performance
pip install watchdog

# Or adjust polling interval
# In config.yaml:
ingest:
  watch_interval_sec: 2  # Faster polling (default: 5)
```

## References

- [ARCHITECTURE.md](../../ARCHITECTURE.md) - System overview
- [ingestforge/core/README.md](../core/README.md) - Configuration and provenance
- [ingestforge/shared/README.md](../shared/README.md) - IProcessor interface
- [ingestforge/chunking/README.md](../chunking/README.md) - Next step: chunking
