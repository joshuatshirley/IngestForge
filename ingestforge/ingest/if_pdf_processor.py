"""
IF-Protocol PDF Processor.

Migration - PDF Parity
Provides IF-Protocol-compliant PDF text extraction with parity
to the legacy TextExtractor implementation.

Follows NASA JPL Power of Ten rules.
"""

import re
import uuid
from pathlib import Path
from typing import Any, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_PAGES = 10000
MAX_TEXT_SIZE = 100_000_000  # 100MB text content


class IFPDFProcessor(IFProcessor):
    """
    IF-Protocol processor for PDF text extraction.

    Achieves functional parity with legacy TextExtractor._extract_pdf
    while providing proper artifact tracking and type safety.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        processor_id: Optional[str] = None,
        version: str = "1.0.0",
        ocr_embedded: bool = True,
    ):
        """
        Initialize the PDF processor.

        Args:
            processor_id: Optional custom ID. Defaults to 'if-pdf-extractor'.
            version: SemVer version string.
            ocr_embedded: Whether to OCR embedded images (default True).
        """
        self._processor_id = processor_id or "if-pdf-extractor"
        self._version = version
        self._ocr_embedded = ocr_embedded
        self._fitz: Any = None

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return self._processor_id

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Functional capabilities provided by this processor."""
        return ["ingest.pdf", "text-extraction", "ocr"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 256  # PDF processing can be memory-intensive

    def is_available(self) -> bool:
        """
        Check if PyMuPDF is available.

        Rule #7: Check return values.
        """
        try:
            import fitz  # noqa: F401

            return True
        except ImportError:
            return False

    def _load_fitz(self) -> Any:
        """
        Lazy-load PyMuPDF.

        Rule #4: Helper function < 60 lines.
        """
        if self._fitz is None:
            import fitz

            fitz.TOOLS.mupdf_display_errors(False)
            self._fitz = fitz
        return self._fitz

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Extract text from PDF file.

        Rule #4: Method should be < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact (must be IFFileArtifact with PDF).

        Returns:
            IFTextArtifact on success, IFFailureArtifact on error.
        """
        # Validate input type
        if not isinstance(artifact, IFFileArtifact):
            return self._create_failure(
                artifact,
                f"IFPDFProcessor requires IFFileArtifact, got {type(artifact).__name__}",
            )

        # Validate file is PDF
        if artifact.mime_type != "application/pdf":
            return self._create_failure(
                artifact, f"IFPDFProcessor requires PDF, got {artifact.mime_type}"
            )

        # Validate file exists
        if not artifact.file_path.exists():
            return self._create_failure(
                artifact, f"File not found: {artifact.file_path}"
            )

        try:
            text = self._extract_pdf_text(artifact.file_path)
            return self._create_text_artifact(artifact, text)
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return self._create_failure(artifact, str(e))

    def _extract_pdf_text(self, file_path: Path) -> str:
        """
        Extract text from PDF file.

        Rule #4: Function < 60 lines.

        Args:
            file_path: Path to PDF file.

        Returns:
            Extracted text content.
        """
        fitz = self._load_fitz()
        doc = fitz.open(file_path)

        # JPL Rule #2: Enforce page limit
        page_count = min(len(doc), MAX_PAGES)
        native_texts: List[str] = []

        for i in range(page_count):
            page_text = self._extract_page_text(doc[i])
            native_texts.append(page_text)

        doc.close()

        # Handle OCR of embedded images
        if self._ocr_embedded:
            texts = self._ocr_embedded_images(file_path, native_texts)
        else:
            texts = [t for t in native_texts if t]

        full_text = "\n\n".join(texts)
        return self._clean_text(full_text)

    def _extract_page_text(self, page: Any) -> str:
        """
        Extract text from a single PDF page.

        Rule #4: Function < 60 lines.
        """
        blocks = page.get_text("dict", flags=11)["blocks"]
        page_lines: List[str] = []

        for block in blocks:
            if block["type"] != 0:
                continue
            block_lines = self._extract_block_lines(block)
            page_lines.extend(block_lines)

        return "\n".join(page_lines) if page_lines else ""

    def _extract_block_lines(self, block: dict) -> List[str]:
        """
        Extract text lines from a PDF block.

        Rule #4: Function < 60 lines.
        """
        lines: List[str] = []
        for line in block.get("lines", []):
            line_text = self._extract_line_spans(line.get("spans", []))
            if line_text.strip():
                lines.append(line_text.strip())
        return lines

    def _extract_line_spans(self, spans: list) -> str:
        """
        Concatenate text from line spans.

        Rule #4: Function < 60 lines.
        """
        return "".join(span.get("text", "") for span in spans)

    def _ocr_embedded_images(
        self, file_path: Path, native_texts: List[str]
    ) -> List[str]:
        """
        OCR embedded images in PDF pages.

        Rule #4: Function < 60 lines.
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

    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.

        Rule #4: Function < 60 lines.
        """
        # Normalize whitespace
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove page numbers
        text = re.sub(r"\n\d+\n", "\n", text)
        text = re.sub(r"Page \d+ of \d+", "", text)

        # Remove repeated headers/footers
        lines = text.split("\n")
        if len(lines) > 10:
            text = self._remove_repeated_lines(lines)

        return text.strip()

    def _remove_repeated_lines(self, lines: List[str]) -> str:
        """
        Remove lines that appear too frequently (headers/footers).

        Rule #4: Function < 60 lines.
        """
        line_counts: dict[str, int] = {}
        for line in lines:
            clean = line.strip()
            if 5 < len(clean) < 100:
                line_counts[clean] = line_counts.get(clean, 0) + 1

        threshold = max(3, len(lines) // 20)
        repeated = {ln for ln, cnt in line_counts.items() if cnt > threshold}
        filtered = [ln for ln in lines if ln.strip() not in repeated]
        return "\n".join(filtered)

    def _create_text_artifact(
        self, source: IFFileArtifact, content: str
    ) -> IFTextArtifact:
        """
        Create IFTextArtifact from extracted content.

        Rule #4: Function < 60 lines.
        """
        # JPL Rule #2: Enforce text size limit
        if len(content) > MAX_TEXT_SIZE:
            content = content[:MAX_TEXT_SIZE]
            logger.warning(f"Truncated text to {MAX_TEXT_SIZE} bytes")

        return IFTextArtifact(
            artifact_id=f"{source.artifact_id}-text-{uuid.uuid4().hex[:8]}",
            content=content,
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id,
            lineage_depth=source.lineage_depth + 1,
            provenance=source.provenance + [self._processor_id],
            metadata={
                "source_file": str(source.file_path),
                "source_mime": source.mime_type,
                "extractor": self._processor_id,
                "word_count": len(content.split()),
                "char_count": len(content),
            },
        )

    def _create_failure(
        self, artifact: IFArtifact, error_message: str
    ) -> IFFailureArtifact:
        """
        Create IFFailureArtifact for error cases.

        Rule #4: Function < 60 lines.
        Rule #7: Always return explicit result.
        """
        return IFFailureArtifact(
            artifact_id=f"{artifact.artifact_id}-pdf-failed",
            error_message=error_message,
            failed_processor_id=self._processor_id,
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.effective_root_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self._processor_id],
        )

    def teardown(self) -> bool:
        """
        Perform resource cleanup.

        Rule #7: Check return values.
        """
        self._fitz = None
        return True
