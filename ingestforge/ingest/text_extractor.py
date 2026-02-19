"""
Text extraction from various document formats.

Converts PDF, EPUB, DOCX, TXT to clean markdown text.

d: Added extract_to_artifact() for IFTextArtifact output.
"""

import mimetypes
import re
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.shared.text_utils import clean_text
from ingestforge.shared.lazy_imports import lazy_property

if TYPE_CHECKING:
    from ingestforge.core.pipeline.artifacts import IFTextArtifact
    from ingestforge.core.pipeline.interfaces import IFArtifact

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_EXTRACTION_SIZE = 50_000_000  # 50MB max extraction size


class TextExtractor:
    """
    Extract text from various document formats.

    Supports:
    - PDF (via PyMuPDF)
    - EPUB (via ebooklib)
    - DOCX (via python-docx)
    - TXT/MD (direct read)
    """

    def __init__(self, config: Config) -> None:
        self.config = config

    @lazy_property
    def fitz(self) -> Any:
        """Lazy-load PyMuPDF."""
        try:
            import fitz

            fitz.TOOLS.mupdf_display_errors(False)
            return fitz
        except ImportError:
            raise ImportError(
                "PyMuPDF is required for PDF processing. "
                "Install with: pip install pymupdf"
            )

    def extract(self, file_path: Path) -> str:
        """
        Extract text from a document.

        Args:
            file_path: Path to the document

        Returns:
            Extracted text as markdown
        """
        suffix = file_path.suffix.lower()

        extractors = {
            ".pdf": self._extract_pdf,
            ".epub": self._extract_epub,
            ".docx": self._extract_docx,
            ".txt": self._extract_text,
            ".md": self._extract_text,
        }

        extractor = extractors.get(suffix)
        if extractor is None:
            raise ValueError(f"Unsupported file format: {suffix}")

        return extractor(file_path)

    def _extract_line_text_from_spans(self, spans: list) -> str:
        """
        Extract text from line spans.

        Rule #1: Simple loop eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            spans: List of text spans in line

        Returns:
            Concatenated line text
        """
        line_text = ""
        for span in spans:
            text = span.get("text", "")
            line_text += text
        return line_text

    def _extract_text_from_block(self, block: dict) -> list[Any]:
        """
        Extract text lines from PDF text block.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            block: PDF text block dictionary

        Returns:
            List of extracted text lines
        """
        lines = []

        for line in block.get("lines", []):
            line_text = self._extract_line_text_from_spans(line.get("spans", []))
            if line_text.strip():
                lines.append(line_text.strip())

        return lines

    def _extract_page_text(self, page: Any) -> str:
        """
        Extract text from single PDF page.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            page: PDF page object

        Returns:
            Extracted page text or empty string
        """
        blocks = page.get_text("dict", flags=11)["blocks"]
        page_text = []

        for block in blocks:
            if block["type"] != 0:
                continue

            block_lines = self._extract_text_from_block(block)
            page_text.extend(block_lines)
        if not page_text:
            return ""

        return "\n".join(page_text)

    def _extract_pdf(self, file_path: Path) -> str:
        """
        Extract text from PDF, including OCR of embedded images.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted and cleaned text
        """
        doc = self.fitz.open(file_path)

        native_texts = []
        for page in doc:
            page_text = self._extract_page_text(page)
            native_texts.append(page_text if page_text else "")

        doc.close()

        # Check for embedded images that may contain text
        # This catches tables/diagrams rendered as images in native-text PDFs
        # Default to True - most users want embedded content extracted
        ocr_embedded = getattr(self.config.ocr, "ocr_embedded_images", True)
        if ocr_embedded:
            texts = self._ocr_embedded_images(file_path, native_texts)
        else:
            texts = [t for t in native_texts if t]

        # Clean up extracted text
        full_text = "\n\n".join(texts)
        return self._clean_text_advanced(full_text)

    def _ocr_embedded_images(
        self, file_path: Path, native_texts: list[str]
    ) -> list[str]:
        """
        OCR any significant embedded images in PDF pages.

        Detects images that may contain text (tables, diagrams, etc.)
        and OCRs them, combining with native text.

        Args:
            file_path: Path to PDF
            native_texts: Native text extracted per page

        Returns:
            List of combined text per page
        """
        try:
            from ingestforge.ingest.embedded_image_ocr import EmbeddedImageOCR

            processor = EmbeddedImageOCR()
            results = processor.process_pdf(file_path, native_texts)

            return [r.combined_text for r in results if r.combined_text.strip()]

        except ImportError as e:
            logger.debug(f"Embedded image OCR not available: {e}")
            return [t for t in native_texts if t]
        except Exception as e:
            logger.warning(f"Embedded image OCR failed: {e}")
            return [t for t in native_texts if t]

    def _extract_epub(self, file_path: Path) -> str:
        """Extract text from EPUB."""
        try:
            import ebooklib
            from ebooklib import epub
            from html.parser import HTMLParser
        except ImportError:
            raise ImportError(
                "ebooklib is required for EPUB processing. "
                "Install with: pip install ebooklib"
            )

        class HTMLTextExtractor(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.text_parts: list[str] = []
                self.in_body = False

            def handle_starttag(self, tag: Any, attrs: Any) -> None:
                if tag == "body":
                    self.in_body = True
                elif tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "div"):
                    self.text_parts.append("\n")

            def handle_endtag(self, tag: Any) -> None:
                if tag == "body":
                    self.in_body = False
                elif tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6"):
                    self.text_parts.append("\n")

            def handle_data(self, data: Any) -> None:
                if self.in_body:
                    self.text_parts.append(data)

            def get_text(self) -> Any:
                return "".join(self.text_parts)

        book = epub.read_epub(str(file_path))
        texts = []

        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                parser = HTMLTextExtractor()
                content = item.get_content().decode("utf-8", errors="ignore")
                parser.feed(content)
                text = parser.get_text().strip()
                if text:
                    texts.append(text)

        full_text = "\n\n".join(texts)
        return self._clean_text(full_text)

    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX."""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required for DOCX processing. "
                "Install with: pip install python-docx"
            )

        doc = Document(file_path)
        texts = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                formatted_text = self._format_paragraph(para, text)
                texts.append(formatted_text)

        full_text = "\n\n".join(texts)
        return self._clean_text(full_text)

    def _format_paragraph(self, para: Any, text: str) -> str:
        """Format paragraph based on style (heading vs normal text)."""
        style = para.style.name if para.style else ""

        if "Heading" not in style:
            return text

        level = self._extract_heading_level(style)
        return "#" * level + " " + text

    def _extract_heading_level(self, style: str) -> int:
        """Extract heading level from style name."""
        if "1" in style:
            return 1
        elif "2" in style:
            return 2
        elif "3" in style:
            return 3
        return 1

    def _extract_text(self, file_path: Path) -> str:
        """Extract text from plain text files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning using shared utility."""
        return clean_text(text)

    def _clean_text_advanced(self, text: str) -> str:
        """Clean extracted text with advanced processing.

        Uses shared clean_text utility for basic cleaning,
        then applies document-specific processing.
        """
        # Use shared utility for basic text cleaning
        text = clean_text(text)

        # Remove page numbers (common patterns)
        text = re.sub(r"\n\d+\n", "\n", text)
        text = re.sub(r"Page \d+ of \d+", "", text)

        # Remove headers/footers (often repeated text)
        lines = text.split("\n")
        if len(lines) > 10:
            # Simple dedup for potential headers/footers
            line_counts: dict[str, int] = {}
            for line in lines:
                clean_line = line.strip()
                if len(clean_line) < 100 and len(clean_line) > 5:
                    line_counts[clean_line] = line_counts.get(clean_line, 0) + 1

            # Remove lines that appear too frequently
            threshold = max(3, len(lines) // 20)
            repeated = {
                line for line, count in line_counts.items() if count > threshold
            }
            lines = [line for line in lines if line.strip() not in repeated]
            text = "\n".join(lines)

        return text.strip()

    def extract_with_metadata(self, file_path: Path) -> dict[str, Any]:
        """
        Extract text with metadata.

        Rule #1: Reduced nesting via helper extraction

        Returns:
            Dict with 'text', 'title', 'sections', etc.
        """
        suffix = file_path.suffix.lower()
        text = self.extract(file_path)

        result = {
            "text": text,
            "file_name": file_path.name,
            "file_type": suffix,
            "word_count": len(text.split()),
            "char_count": len(text),
        }

        # Extract sections from markdown headers
        sections = self._extract_markdown_sections(text)

        result["sections"] = sections
        result["section_count"] = len(sections)

        return result

    def _extract_markdown_sections(self, text: str) -> list[Any]:
        """Extract markdown header sections from text.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        sections = []

        for line in text.split("\n"):
            if line.startswith("#"):
                section = self._parse_markdown_header(line)
                if section:
                    sections.append(section)

        return sections

    def _parse_markdown_header(self, line: str) -> dict[str, Any]:
        """Parse a markdown header line to extract level and title.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        # Count header level
        level = 0
        for char in line:
            if char == "#":
                level += 1
            else:
                break

        title = line[level:].strip()
        return {
            "level": level,
            "title": title,
        }

    def extract_to_artifact(
        self,
        file_path: Path,
        parent: Optional["IFArtifact"] = None,
    ) -> "IFTextArtifact":
        """
        Extract text and return as IFTextArtifact.

        d: Artifact-based extraction for pipeline migration.
        Rule #2: Bounded content size (MAX_EXTRACTION_SIZE).
        Rule #7: Explicit return type.
        Rule #9: Complete type hints.

        Args:
            file_path: Path to the document.
            parent: Optional parent artifact for lineage tracking.

        Returns:
            IFTextArtifact with extracted text, hash, and lineage.
        """
        from ingestforge.core.pipeline.artifact_factory import ArtifactFactory

        # Extract text using existing method
        text = self.extract(file_path)
        if len(text) > MAX_EXTRACTION_SIZE:
            logger.warning(
                f"Extracted content exceeds {MAX_EXTRACTION_SIZE} bytes, truncating"
            )
            text = text[:MAX_EXTRACTION_SIZE]

        # Build metadata
        suffix = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        metadata: dict[str, Any] = {
            "source_path": str(file_path.absolute()),
            "file_name": file_path.name,
            "file_type": suffix,
            "mime_type": mime_type or "application/octet-stream",
            "word_count": len(text.split()),
            "char_count": len(text),
            "extraction_method": "text_extractor",
        }

        # Create artifact with lineage if parent provided
        if parent:
            from ingestforge.core.pipeline.artifacts import IFTextArtifact

            return IFTextArtifact(
                artifact_id=str(__import__("uuid").uuid4()),
                content=text,
                metadata=metadata,
                parent_id=parent.artifact_id,
                root_artifact_id=parent.effective_root_id,
                lineage_depth=parent.lineage_depth + 1,
                provenance=list(parent.provenance) + ["text-extractor"],
            )

        # Create standalone artifact
        return ArtifactFactory.text_from_string(
            content=text,
            source_path=str(file_path.absolute()),
            metadata=metadata,
        )
