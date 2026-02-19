# Audit Trail Design

> Every answer traceable to the exact source location

## Goal

From any piece of output (answer, quote, flashcard, glossary term), the user can:
1. See exactly which file it came from
2. Click a link to open that file
3. Jump directly to the exact location (page, paragraph, line)
4. Verify the extracted text matches the original

---

## Source Location Model (Enhanced)

```python
@dataclass
class SourceLocation:
    # === File Identity ===
    source_id: str                      # Unique hash of file content
    source_file: str                    # Original filename
    source_path: Path                   # Full local path at ingest time
    source_path_exists: bool            # Verified at query time

    # === Document Metadata ===
    title: Optional[str]
    authors: list[Author]
    publication_date: Optional[str]
    url: Optional[str]                  # If fetched from web

    # === Structural Location ===
    chapter: Optional[str]
    section: Optional[str]
    subsection: Optional[str]

    # === Precise Location ===
    page_start: Optional[int]           # PDF/DOCX page number
    page_end: Optional[int]
    paragraph_number: Optional[int]     # Within page/section
    line_start: Optional[int]           # Line number in source
    line_end: Optional[int]
    char_offset_start: Optional[int]    # Character offset from file start
    char_offset_end: Optional[int]

    # === Format-Specific ===
    pdf_page: Optional[int]             # 0-indexed PDF page
    docx_paragraph_id: Optional[str]    # DOCX paragraph XML ID
    html_anchor: Optional[str]          # HTML element ID or generated anchor
    epub_chapter_href: Optional[str]    # EPUB spine reference
    slide_number: Optional[int]         # PPTX slide index

    # === Extraction Metadata ===
    extracted_at: datetime
    extraction_method: str              # "native_text", "ocr", "html_parse"
    extraction_confidence: float        # 0.0-1.0
    original_text_hash: str             # Hash of extracted text for verification
```

---

## Deep Link Generation

### PDF Files

```python
class PDFLinker:
    def generate_link(self, location: SourceLocation) -> str:
        """Generate a link that opens PDF at specific page."""
        path = location.source_path
        page = location.pdf_page or location.page_start or 1

        # File URI with page parameter (works in most PDF readers)
        # file:///C:/docs/paper.pdf#page=47
        return f"file:///{path}#page={page}"

    def generate_link_with_highlight(self, location: SourceLocation, text: str) -> str:
        """Generate link with search text for highlighting."""
        # Some PDF readers support search parameter
        # file:///C:/docs/paper.pdf#page=47&search=inflation
        base = self.generate_link(location)
        search_term = text[:50].replace(" ", "%20")  # First 50 chars
        return f"{base}&search={search_term}"
```

### DOCX Files

```python
class DOCXLinker:
    def generate_link(self, location: SourceLocation) -> str:
        """Generate link to DOCX file."""
        # DOCX doesn't support deep linking natively
        # But we can provide paragraph number for manual navigation
        path = location.source_path
        return f"file:///{path}"

    def get_navigation_hint(self, location: SourceLocation) -> str:
        """Provide human-readable navigation instructions."""
        hints = []
        if location.page_start:
            hints.append(f"Page {location.page_start}")
        if location.section:
            hints.append(f"Section: {location.section}")
        if location.paragraph_number:
            hints.append(f"Paragraph {location.paragraph_number}")
        return " → ".join(hints) if hints else "See document"
```

### HTML Files

```python
class HTMLLinker:
    def generate_link(self, location: SourceLocation) -> str:
        """Generate link with anchor to specific section."""
        path = location.source_path

        if location.html_anchor:
            return f"file:///{path}#{location.html_anchor}"
        return f"file:///{path}"

    def inject_anchors(self, html_path: Path, chunks: list[Chunk]) -> None:
        """Add anchor IDs to HTML for deep linking (at ingest time)."""
        # Modify saved HTML to add id="chunk-{chunk_id}" to source elements
        pass
```

### Text/Markdown Files

```python
class TextLinker:
    def generate_link(self, location: SourceLocation) -> str:
        """Generate link to text file."""
        path = location.source_path

        # VS Code / editors support line numbers
        # file:///C:/docs/notes.md:47
        if location.line_start:
            return f"file:///{path}:{location.line_start}"
        return f"file:///{path}"

    def generate_vscode_link(self, location: SourceLocation) -> str:
        """Generate VS Code-specific link."""
        path = location.source_path
        line = location.line_start or 1
        col = 1
        # vscode://file/C:/docs/notes.md:47:1
        return f"vscode://file/{path}:{line}:{col}"
```

---

## Link Factory

```python
class SourceLinker:
    """Generate appropriate links based on file type."""

    linkers = {
        ".pdf": PDFLinker,
        ".docx": DOCXLinker,
        ".doc": DOCXLinker,
        ".html": HTMLLinker,
        ".htm": HTMLLinker,
        ".txt": TextLinker,
        ".md": TextLinker,
        ".epub": EPUBLinker,
        ".pptx": PPTXLinker,
    }

    def get_link(self, location: SourceLocation) -> SourceLink:
        """Get appropriate link for source type."""
        ext = Path(location.source_file).suffix.lower
        linker = self.linkers.get(ext, TextLinker)

        return SourceLink(
            file_uri=linker.generate_link(location),
            file_path=location.source_path,
            file_exists=location.source_path.exists,
            navigation_hint=linker.get_navigation_hint(location),
            line_number=location.line_start,
            page_number=location.page_start,
        )

@dataclass
class SourceLink:
    file_uri: str              # Clickable URI
    file_path: Path            # Raw path for display
    file_exists: bool          # Verified at render time
    navigation_hint: str       # Human-readable location
    line_number: Optional[int]
    page_number: Optional[int]

    def render_cli(self) -> str:
        """Render for CLI output."""
        status = "✓" if self.file_exists else "✗ (file moved)"
        return f"[{status}] {self.file_path}\n    → {self.navigation_hint}"

    def render_markdown(self) -> str:
        """Render for Markdown export."""
        if self.file_exists:
            return f"[{self.file_path.name}]({self.file_uri}) — {self.navigation_hint}"
        return f"{self.file_path.name} (file not found) — {self.navigation_hint}"
```

---

## CLI Output with Audit Links

### Query Results

```
Query: "What causes inflation?"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Result 1 [HIGH CONFIDENCE]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"Demand-pull inflation occurs when aggregate demand exceeds
aggregate supply, leading to upward pressure on prices."

┌─────────────────────────────────────────────────────────────────┐
│ SOURCE AUDIT TRAIL                                              │
├─────────────────────────────────────────────────────────────────┤
│ File:     Macroeconomics_Smith_2023.pdf                        │
│ Path:     C:\Research\Economics\Macroeconomics_Smith_2023.pdf  │
│ Status:   ✓ File exists                                        │
│ Location: Chapter 3 → Section 3.2 → Page 47, Paragraph 3       │
│ Link:     file:///C:/Research/Economics/Macro...pdf#page=47    │
│ Extracted: 2024-01-15 14:32:00 (OCR confidence: 98%)           │
└─────────────────────────────────────────────────────────────────┘

[o] Open file  [c] Copy citation  [v] Verify text  [n] Next result
```

### Flashcard Export

```csv
Term,Definition,Source,SourcePath,Page,VerificationLink
"Demand-pull inflation","When aggregate demand exceeds supply...","Smith (2023)","C:\Research\Macro.pdf","47","file:///C:/Research/Macro.pdf#page=47"
```

### Markdown Export

```markdown
## Key Finding: Causes of Inflation

> "Demand-pull inflation occurs when aggregate demand exceeds
> aggregate supply, leading to upward pressure on prices."

**Source**: Smith, J. (2023). *Macroeconomics*. Chapter 3, p. 47.

📎 **Verify**: [Open source file](file:///C:/Research/Economics/Macroeconomics_Smith_2023.pdf#page=47)
   - Location: Chapter 3 → Section 3.2 → Paragraph 3
   - Extracted: 2024-01-15 (confidence: 98%)
```

---

## Verification Workflow

### At Query Time

```python
class AuditTrailVerifier:
    def verify_source(self, location: SourceLocation) -> VerificationResult:
        """Verify source file still exists and text matches."""

        result = VerificationResult(location=location)

        # 1. Check file exists
        if not location.source_path.exists:
            result.file_status = "missing"
            result.warnings.append(f"Source file not found: {location.source_path}")
            return result

        result.file_status = "found"

        # 2. Check file hasn't changed (optional, expensive)
        current_hash = hash_file(location.source_path)
        if current_hash != location.source_id:
            result.file_status = "modified"
            result.warnings.append("Source file has been modified since ingestion")

        # 3. Verify extracted text still present (for text files)
        if self.can_verify_text(location):
            original_text = self.extract_text_at_location(location)
            if hash(original_text) != location.original_text_hash:
                result.text_status = "mismatch"
                result.warnings.append("Extracted text no longer matches source")
            else:
                result.text_status = "verified"

        return result

@dataclass
class VerificationResult:
    location: SourceLocation
    file_status: str = "unknown"    # found, missing, modified
    text_status: str = "unknown"    # verified, mismatch, unverifiable
    warnings: list[str] = field(default_factory=list)

    @property
    def is_trustworthy(self) -> bool:
        return self.file_status == "found" and self.text_status in ("verified", "unknown")
```

### Verification Command

```bash
ingestforge verify-sources

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SOURCE VERIFICATION REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Total sources: 47
✓ Verified:    42 (89%)
⚠ Modified:    3  (6%)
✗ Missing:     2  (4%)

MISSING FILES:
  ✗ C:\Research\OldPaper.pdf
    Last seen: 2024-01-10
    Chunks affected: 12
    Action: [Re-ingest] [Remove from corpus] [Ignore]

  ✗ C:\Downloads\Article.html
    Last seen: 2024-01-12
    Chunks affected: 3
    Action: [Re-ingest] [Remove from corpus] [Ignore]

MODIFIED FILES:
  ⚠ C:\Research\Draft.docx
    Modified: 2024-01-20 (after ingest on 2024-01-15)
    Chunks affected: 8
    Action: [Re-ingest] [Keep old version] [Ignore]
```

---

## Chunk-Level Audit Trail

Every chunk stores complete provenance:

```python
@dataclass
class Chunk:
    id: str
    content: str

    # === Audit Trail ===
    source_location: SourceLocation      # Full location info
    extraction_timestamp: datetime        # When extracted
    extraction_method: str                # How extracted
    extraction_confidence: float          # Quality score
    content_hash: str                     # For verification

    # === Processing History ===
    processing_steps: list[ProcessingStep]  # Full history

    def get_audit_trail(self) -> AuditTrail:
        """Get complete audit trail for this chunk."""
        return AuditTrail(
            chunk_id=self.id,
            source=self.source_location,
            extraction=ExtractionInfo(
                timestamp=self.extraction_timestamp,
                method=self.extraction_method,
                confidence=self.extraction_confidence,
            ),
            processing=self.processing_steps,
            verification=self.verify,
        )

@dataclass
class ProcessingStep:
    step_name: str          # "chunking", "embedding", "enrichment"
    timestamp: datetime
    input_hash: str         # Hash of input to this step
    output_hash: str        # Hash of output
    parameters: dict        # Config used

@dataclass
class AuditTrail:
    chunk_id: str
    source: SourceLocation
    extraction: ExtractionInfo
    processing: list[ProcessingStep]
    verification: VerificationResult

    def to_markdown(self) -> str:
        """Render full audit trail as markdown."""
        pass

    def to_json(self) -> dict:
        """Export audit trail as JSON."""
        pass
```

---

## Audit Trail in LLM Answers

```python
class GroundedAnswer:
    text: str
    claims: list[GroundedClaim]

@dataclass
class GroundedClaim:
    claim_text: str
    supporting_chunks: list[Chunk]
    confidence: float

    def get_audit_trails(self) -> list[AuditTrail]:
        """Get audit trail for each supporting chunk."""
        return [chunk.get_audit_trail for chunk in self.supporting_chunks]
```

**User Output**:

```
Answer: Inflation is primarily caused by demand exceeding supply [1].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CLAIM AUDIT: "Inflation is primarily caused by demand exceeding supply"
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Supporting Evidence [1]:
  Text:     "Demand-pull inflation occurs when aggregate demand..."
  Source:   Macroeconomics_Smith_2023.pdf
  Location: Chapter 3, Page 47, Paragraph 3
  Link:     file:///C:/Research/Macro.pdf#page=47

  Extraction:
    Method:     Native PDF text
    Confidence: 99%
    Timestamp:  2024-01-15 14:32:00

  Verification:
    File:       ✓ Exists at original path
    Content:    ✓ Matches extracted text

  [Open File] [Copy Quote] [View Full Context]
```

---

## Export with Audit Trail

### Full Audit Report

```bash
ingestforge export-audit --output audit_report.json
```

```json
{
  "generated_at": "2024-01-20T10:30:00",
  "corpus_stats": {
    "total_chunks": 1234,
    "total_sources": 47,
    "verified_sources": 42
  },
  "chunks": [
    {
      "id": "chunk_abc123",
      "content": "Demand-pull inflation occurs...",
      "audit_trail": {
        "source": {
          "file": "Macroeconomics_Smith_2023.pdf",
          "path": "C:\\Research\\Economics\\Macroeconomics_Smith_2023.pdf",
          "exists": true,
          "page": 47,
          "paragraph": 3,
          "link": "file:///C:/Research/Economics/Macro...pdf#page=47"
        },
        "extraction": {
          "method": "native_pdf",
          "confidence": 0.99,
          "timestamp": "2024-01-15T14:32:00"
        },
        "verification": {
          "file_status": "found",
          "text_status": "verified",
          "last_verified": "2024-01-20T10:30:00"
        }
      }
    }
  ]
}
```

---

## Implementation Priority

| Priority | Feature | Why |
|----------|---------|-----|
| 1 | **Source path storage** | Foundation - store full path at ingest |
| 2 | **Page/line tracking** | Know exact location in document |
| 3 | **File URI generation** | Clickable links to source |
| 4 | **CLI audit display** | Show source info in query results |
| 5 | **File existence check** | Warn if source moved/deleted |
| 6 | **Markdown export links** | Verifiable exports |
| 7 | **Content hash verification** | Detect source modifications |
| 8 | **Full audit command** | `verify-sources` command |
| 9 | **JSON audit export** | Machine-readable provenance |
| 10 | **Editor deep links** | VS Code line number links |

---

## File Movement Handling

When source files are moved:

```bash
ingestforge sources --check

2 sources have moved since ingestion:

  OldPath: C:\Downloads\Paper.pdf
  Status:  Not found

  Possible matches:
    1. C:\Research\Papers\Paper.pdf (same hash)
    2. C:\Backup\Paper.pdf (same hash)

  [1] Update path  [2] Update path  [s] Skip  [r] Remove from corpus

ingestforge sources --update-path "C:\Downloads\Paper.pdf" "C:\Research\Papers\Paper.pdf"
✓ Updated 15 chunks with new source path
```

---

*Every claim must be verifiable. Every quote must be traceable.*

*Last updated: 2026-02-07*
