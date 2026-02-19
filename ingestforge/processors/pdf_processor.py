"""
Standard PDF Processor.

IFProcessor implementation for PDF document splitting.
NASA JPL Power of Ten compliant.
"""

import logging
from pathlib import Path
from typing import Any, List, Optional

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.core.pipeline.registry import register_processor

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_PDF_PAGES = 10000
MAX_CHAPTERS = 1000


@register_processor(
    processor_id="standard-pdf-processor",
    capabilities=["pdf-splitting", "ocr-detection"],
    mime_types=["application/pdf"],
)
class StandardPDFProcessor(IFProcessor):
    """
    IFProcessor for PDF document splitting and OCR detection.

    Replaces PipelineSplittersMixin._split_pdf_document().
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(self, config: Optional[Any] = None) -> None:
        """
        Initialize PDF processor.

        Args:
            config: Pipeline configuration for splitter and OCR settings.
        """
        self._config = config
        self._splitter = None
        self._current_pdf_structure: Optional[Any] = None
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "standard-pdf-processor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return self._version

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return ["pdf-splitting", "ocr-detection"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement."""
        return 500  # PDF processing can be memory-intensive

    def is_available(self) -> bool:
        """Check if PDF processing is available."""
        try:
            from ingestforge.ingest.pdf_splitter import PDFSplitter

            return True
        except ImportError:
            return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process a PDF file artifact.

        Rule #7: Check return values.

        Args:
            artifact: Input IFFileArtifact pointing to PDF file.

        Returns:
            IFTextArtifact with extracted content or IFFailureArtifact on error.
        """
        if not isinstance(artifact, IFFileArtifact):
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-pdf-error",
                error_message=f"Expected IFFileArtifact, got {type(artifact).__name__}",
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

        try:
            file_path = artifact.file_path
            return self._process_pdf(file_path, artifact)
        except Exception as e:
            logger.exception(f"PDF processing failed: {e}")
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-pdf-error",
                error_message=str(e),
                parent_id=artifact.artifact_id,
                provenance=artifact.provenance + [self.processor_id],
            )

    def _process_pdf(self, file_path: Path, artifact: IFFileArtifact) -> IFArtifact:
        """
        Internal PDF processing logic.

        Rule #4: Function < 60 lines.
        """
        from ingestforge.ingest.ocr_manager import get_best_available_engine

        # Check for scanned PDF
        ocr = get_best_available_engine(self._config) if self._config else None
        if ocr:
            ocr_result = ocr.process_pdf(file_path)
            if ocr_result.is_majority_scanned():
                return self._handle_scanned_pdf(file_path, ocr_result, artifact)

        # Standard PDF splitting
        chapters = self._get_splitter().split(file_path, artifact.artifact_id)

        # JPL Rule #2: Bound chapters
        if len(chapters) > MAX_CHAPTERS:
            chapters = chapters[:MAX_CHAPTERS]
            logger.warning(f"Truncated chapters to {MAX_CHAPTERS}")

        source_loc = self._extract_pdf_structure(file_path)

        # Build text content from chapters (simplified for processor)
        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-text",
            content=f"PDF with {len(chapters)} chapters",
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "source_type": "pdf",
                "chapter_count": len(chapters),
                "chapter_paths": [str(c) for c in chapters],
                "source_location": source_loc.title if source_loc else None,
            },
        )

    def _handle_scanned_pdf(
        self,
        file_path: Path,
        ocr_result: Any,
        artifact: IFFileArtifact,
    ) -> IFTextArtifact:
        """Handle scanned PDF with OCR."""
        logger.info(
            f"Scanned PDF: {file_path.name} "
            f"({ocr_result.scanned_page_count}/{ocr_result.page_count} pages)"
        )

        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-ocr",
            content=ocr_result.text,
            parent_id=artifact.artifact_id,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "source_type": "scanned_pdf",
                "ocr_engine": ocr_result.engine,
                "confidence": ocr_result.confidence,
                "page_count": ocr_result.page_count,
                "scanned_pages": ocr_result.scanned_page_count,
            },
        )

    def _get_splitter(self) -> Any:
        """Get or create PDF splitter."""
        if self._splitter is None:
            from ingestforge.ingest.pdf_splitter import PDFSplitter

            self._splitter = PDFSplitter(self._config)
        return self._splitter

    def _extract_pdf_structure(self, file_path: Path) -> Optional[SourceLocation]:
        """Extract PDF structure for citations."""
        try:
            from ingestforge.ingest.structure_extractor import StructureExtractor

            structure_extractor = StructureExtractor()
            pdf_structure = structure_extractor.extract_from_pdf(file_path)
            self._current_pdf_structure = pdf_structure

            return SourceLocation(
                source_type=SourceType.PDF,
                title=pdf_structure.title or file_path.stem,
                file_path=str(file_path),
            )
        except Exception as e:
            logger.warning(f"PDF structure extraction failed: {e}")
            self._current_pdf_structure = None
            return SourceLocation(
                source_type=SourceType.PDF,
                title=file_path.stem,
                file_path=str(file_path),
            )

    def teardown(self) -> bool:
        """Clean up resources."""
        self._splitter = None
        self._current_pdf_structure = None
        return True
