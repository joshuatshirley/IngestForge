"""
Document processor dispatcher.

Routes documents to appropriate processors based on file type.

Enhanced with SmartIngestRouter for content-based routing.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.ingest.content_hash_verifier import hash_content
from ingestforge.ingest.content_router import SmartIngestRouter, RoutingDecision
from ingestforge.ingest.pdf_splitter import PDFSplitter
from ingestforge.ingest.text_extractor import TextExtractor

logger = get_logger(__name__)


@dataclass
class ProcessedDocument:
    """
    Result of document processing.

    Added routing_decision for content-based routing provenance.
    """

    document_id: str
    source_file: str
    file_type: str
    chapters: List[Path]
    texts: List[str]
    metadata: Dict[str, Any]
    source_location: Optional[SourceLocation] = None  # For citations
    routing_decision: Optional[RoutingDecision] = None  # Routing provenance


# Handler registry: maps file suffixes to handler method names
# Using method names (strings) allows registration before instance creation
_HANDLER_REGISTRY: Dict[str, str] = {
    # PDF
    ".pdf": "_process_pdf",
    # EPUB
    ".epub": "_process_epub",
    # Plain text
    ".txt": "_process_text",
    # Markdown variants
    ".md": "_process_markdown",
    ".markdown": "_process_markdown",
    ".mdown": "_process_markdown",
    # Office documents
    ".docx": "_process_docx",
    ".pptx": "_process_pptx",
    # HTML variants
    ".html": "_process_html",
    ".htm": "_process_html",
    ".mhtml": "_process_html",
    ".xhtml": "_process_html",
    # Image formats
    ".png": "_process_image",
    ".jpg": "_process_image",
    ".jpeg": "_process_image",
    ".tiff": "_process_image",
    ".bmp": "_process_image",
    ".gif": "_process_image",
    # Audio formats (TICKET-201)
    ".mp3": "_process_audio",
    ".wav": "_process_audio",
    ".m4a": "_process_audio",
    ".flac": "_process_audio",
    ".ogg": "_process_audio",
    ".webm": "_process_audio",
    # Salesforce code
    ".cls": "_process_apex",
    ".trigger": "_process_apex",
    ".js": "_process_lwc",
    # LaTeX formats
    ".tex": "_process_latex",
    ".latex": "_process_latex",
    ".ltx": "_process_latex",
    # Jupyter notebooks
    ".ipynb": "_process_jupyter",
}


def register_handler(suffix: str, method_name: str) -> None:
    """
    Register a handler for a file suffix.

    Args:
        suffix: File extension including dot (e.g., ".pdf")
        method_name: Name of the handler method (e.g., "_process_pdf")
    """
    _HANDLER_REGISTRY[suffix.lower()] = method_name


def get_supported_suffixes() -> Set[str]:
    """Return all registered file suffixes."""
    return set(_HANDLER_REGISTRY.keys())


class DocumentProcessor:
    """
    Process documents through split and extraction stages.

    Enhanced with SmartIngestRouter for content-based routing.
    Dispatches to appropriate handlers based on file type.
    """

    def __init__(
        self,
        config: Config,
        enable_smart_routing: bool = True,
    ) -> None:
        """
        Initialize document processor.

        Args:
            config: Configuration object.
            enable_smart_routing: Use SmartIngestRouter for content-based routing (default: True).
                                  Set to False to use legacy extension-only routing.
        """
        self.config = config
        self.splitter = PDFSplitter(config)
        self.extractor = TextExtractor(config)

        # Initialize SmartIngestRouter
        self._enable_smart_routing = enable_smart_routing
        self._router: Optional[SmartIngestRouter] = None
        if enable_smart_routing:
            self._router = SmartIngestRouter(
                handler_registry=_HANDLER_REGISTRY,
                enable_caching=True,
            )

        # Validate handler registry on initialization
        self._validate_handlers()

    def _validate_handlers(self) -> None:
        """Validate that all registered handlers exist as methods."""
        for suffix, method_name in _HANDLER_REGISTRY.items():
            assert hasattr(
                self, method_name
            ), f"Handler method '{method_name}' not found for suffix '{suffix}'"
            handler = getattr(self, method_name)
            assert callable(
                handler
            ), f"Handler '{method_name}' for suffix '{suffix}' is not callable"

    def _get_handler_for_suffix(self, suffix: str) -> Optional[Callable]:
        """
        Get the appropriate handler function for a file type.

        Rule #1: Dictionary dispatch eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints
        """
        method_name = _HANDLER_REGISTRY.get(suffix)
        if method_name is None:
            return None
        return getattr(self, method_name)

    def process(
        self,
        file_path: Path,
        document_id: str,
        content_type: Optional[str] = None,
    ) -> ProcessedDocument:
        """
        Process a document file.

        Enhanced with SmartIngestRouter for content-based routing.
        Maintains backwards compatibility with extension-based routing.

        Rule #1: No nesting - pure dictionary dispatch
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: Path to the document
            document_id: Unique document identifier
            content_type: Optional MIME type for content-based routing ()

        Returns:
            ProcessedDocument with extracted content and routing metadata
        """
        routing_decision: Optional[RoutingDecision] = None

        # Use SmartIngestRouter if enabled
        if self._enable_smart_routing and self._router:
            try:
                routing_decision = self._router.route(file_path, content_type)
                handler = getattr(self, routing_decision.processor_id)
                logger.info(
                    f"Processing {file_path.name} via smart routing "
                    f"(method: {routing_decision.detection_method.value}, "
                    f"confidence: {routing_decision.confidence:.2f})"
                )
            except (ValueError, AttributeError) as e:
                # Fallback to extension-based routing on error
                logger.warning(
                    f"Smart routing failed for {file_path.name}, falling back to extension: {e}"
                )
                suffix = file_path.suffix.lower()
                handler = self._get_handler_for_suffix(suffix)
                if not handler:
                    raise ValueError(f"Unsupported file type: {suffix}")
        else:
            # Legacy extension-based routing (AC-F10: backwards compatibility)
            suffix = file_path.suffix.lower()
            logger.info(f"Processing {file_path.name} (type: {suffix})")
            handler = self._get_handler_for_suffix(suffix)
            if not handler:
                raise ValueError(f"Unsupported file type: {suffix}")

        # Compute content hash for integrity verification
        _content_hash = hash_content(file_path)

        # Process document and add routing metadata
        result = handler(file_path, document_id)
        result.routing_decision = routing_decision  # Add routing provenance
        return result

    def _process_pdf(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process PDF document with splitting.

        Uses a two-tier strategy:
        - Majority-scanned PDFs (>50% pages need OCR): use hybrid OCR path
        - Mixed/digital PDFs: use standard split+extract+enrich path
          (preserves chapter splitting, citation metadata, PDF structure)
        """
        # Try OCR path for scanned PDFs
        ocr_result = self._try_ocr_processing(file_path, document_id)
        if ocr_result:
            return ocr_result

        # Standard path for digital/mixed PDFs
        return self._process_digital_pdf(file_path, document_id)

    def _try_ocr_processing(
        self, file_path: Path, document_id: str
    ) -> Optional[ProcessedDocument]:
        """Attempt OCR processing for scanned PDFs."""
        from ingestforge.ingest.ocr_manager import get_best_available_engine

        ocr = get_best_available_engine(self.config)
        if not ocr:
            return None

        result = ocr.process_pdf(file_path)
        if not result.is_majority_scanned():
            return None

        logger.info(
            f"Scanned PDF detected: {file_path.name} "
            f"({result.scanned_page_count}/{result.page_count} pages), "
            f"OCR via {result.engine} ({result.confidence:.0%} confidence)"
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="pdf",
            chapters=[file_path],
            texts=[result.text],
            metadata={
                "is_scanned": True,
                "ocr_engine": result.engine,
                "ocr_confidence": result.confidence,
                "page_count": result.page_count,
                "scanned_page_count": result.scanned_page_count,
            },
        )

    def _process_digital_pdf(
        self, file_path: Path, document_id: str
    ) -> ProcessedDocument:
        """Process digital/mixed PDF through standard pipeline."""
        # Get metadata first
        metadata = self.splitter.get_metadata(file_path)

        # Enrich with citation identifiers
        self._enrich_pdf_metadata(metadata)

        # Split into chapters and extract text
        chapters = self.splitter.split(file_path, document_id)
        texts = [self.extractor.extract(ch) for ch in chapters]

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="pdf",
            chapters=chapters,
            texts=texts,
            metadata=metadata,
        )

    def _enrich_pdf_metadata(self, metadata: Dict[str, Any]) -> None:
        """Enrich metadata with citation identifiers from PDF metadata."""
        try:
            from ingestforge.ingest.citation_metadata_extractor import (
                CitationMetadataExtractor,
            )

            citation_extractor = CitationMetadataExtractor()
            citation_meta = citation_extractor.extract_from_pdf_metadata(metadata)

            if citation_meta.doi:
                metadata["doi"] = citation_meta.doi
            if citation_meta.isbn:
                metadata["isbn"] = citation_meta.isbn
            if citation_meta.arxiv_id:
                metadata["arxiv_id"] = citation_meta.arxiv_id
        except Exception as e:
            logger.debug(f"Citation metadata extraction skipped: {e}")

    def _process_epub(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process EPUB document."""
        # EPUB doesn't need splitting - extract directly
        result = self.extractor.extract_with_metadata(file_path)

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="epub",
            chapters=[file_path],
            texts=[result["text"]],
            metadata={
                "word_count": result["word_count"],
                "section_count": result["section_count"],
                "sections": result["sections"],
            },
        )

    def _process_text(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process plain text/markdown document.

        Detects ADO work item markdown exports and routes to ADO processor.
        """
        # Check if this is an ADO work item export
        if file_path.suffix.lower() == ".md":
            try:
                content = file_path.read_text(encoding="utf-8")
                # ADO exports have a specific format with | ID | Type | State | table
                if "| ID |" in content and (
                    "| Type |" in content or "| Work Item Type |" in content
                ):
                    return self._process_ado(file_path, document_id)
            except Exception as e:
                logger.debug(
                    f"Failed ADO format detection, falling through to normal text processing: {e}"
                )

        result = self.extractor.extract_with_metadata(file_path)

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type=file_path.suffix.lower()[1:],  # txt or md
            chapters=[file_path],
            texts=[result["text"]],
            metadata={
                "word_count": result["word_count"],
                "section_count": result["section_count"],
            },
        )

    def _process_ado(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process ADO work item markdown export."""
        from ingestforge.ingest.ado_processor import ADOProcessor
        from ingestforge.core.provenance import SourceLocation, SourceType

        processor = ADOProcessor()
        result = processor.process(file_path)

        # Build source location for ADO work item
        ado_id = result.metadata.get("ado_id", 0)
        title = result.metadata.get("title", file_path.stem)

        source_location = SourceLocation(
            source_type=SourceType.ADO_WORK_ITEM,
            title=f"#{ado_id}: {title}" if ado_id else title,
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="ado",
            chapters=[file_path],
            texts=[result.text],
            metadata=result.metadata,
            source_location=source_location,
        )

    def _process_docx(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process DOCX document."""
        result = self.extractor.extract_with_metadata(file_path)

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="docx",
            chapters=[file_path],
            texts=[result["text"]],
            metadata={
                "word_count": result["word_count"],
                "section_count": result["section_count"],
                "sections": result["sections"],
            },
        )

    def _process_pptx(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process PPTX presentation."""
        from ingestforge.ingest.pptx_processor import PptxProcessor

        processor = PptxProcessor()
        result = processor.process(file_path)

        # Attempt slide image OCR if an engine is available
        slide_image_text = ""
        from ingestforge.ingest.ocr_manager import get_best_available_engine

        ocr = get_best_available_engine(self.config)
        if ocr:
            try:
                from ingestforge.ingest.slide_image_extractor import (
                    PresentationImageProcessor,
                )

                img_processor = PresentationImageProcessor(
                    ocr_engine=ocr,
                    language=self.config.ocr.language,
                )
                img_result = img_processor.process_pptx(file_path)
                if img_result.combined_text.strip():
                    slide_image_text = (
                        f"\n\n[Slide Image Text]\n{img_result.combined_text}"
                    )
            except Exception as e:
                logger.debug(f"Slide image OCR skipped: {e}")

        full_text = result.full_text
        if slide_image_text:
            full_text += slide_image_text

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="pptx",
            chapters=[file_path],
            texts=[full_text],
            metadata={
                "title": result.title,
                "total_slides": result.total_slides,
                "word_count": result.word_count,
                "slides": [
                    {
                        "slide_number": s.slide_number,
                        "title": s.title,
                        "word_count": s.word_count,
                    }
                    for s in result.slides
                ],
            },
        )

    def _process_html(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process HTML document with full provenance tracking."""
        from ingestforge.ingest.html_processor import HTMLProcessor

        processor = HTMLProcessor()
        result = processor.process(file_path)

        # Build source location from extracted metadata
        source_location = result.source_location

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="html",
            chapters=[file_path],
            texts=[result.text],
            metadata={
                "title": result.title,
                "authors": result.authors,
                "publication_date": result.publication_date,
                "description": result.description,
                "site_name": result.site_name,
                "url": result.url,
                "headings": result.headings,
                "word_count": len(result.text.split()),
            },
            source_location=source_location,
        )

    def _process_image(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process image file with OCR."""
        from ingestforge.ingest.ocr_manager import get_ocr_engine

        engine = get_ocr_engine(self.config)

        if not engine.is_available:
            raise RuntimeError(
                "No OCR engine available for image processing. "
                "Install with: pip install 'ingestforge[ocr]'"
            )

        result = engine.process_image(file_path)

        if not result.text.strip():
            logger.warning(f"No text extracted from image: {file_path.name}")

        # Create source location
        source_location = SourceLocation(
            source_type=SourceType.IMAGE,
            title=file_path.stem,
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="image",
            chapters=[file_path],
            texts=[result.text],
            metadata={
                "title": file_path.stem,
                "ocr_confidence": result.confidence,
                "ocr_engine": result.engine,
                "word_count": len(result.text.split()),
            },
            source_location=source_location,
        )

    def _process_audio(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process audio file with Whisper transcription (TICKET-201).

        Args:
            file_path: Path to audio file.
            document_id: Document identifier.

        Returns:
            ProcessedDocument with transcription.
        """
        from ingestforge.ingest.audio import AudioProcessor

        processor = AudioProcessor(
            whisper_model=self.config.get("audio.whisper_model", "base"),
            language=self.config.get("audio.language", "en"),
        )

        if not processor.is_available:
            raise RuntimeError(
                "faster-whisper not available for audio processing. "
                "Install with: pip install faster-whisper"
            )

        result = processor.process(file_path)

        if not result.success:
            raise RuntimeError(f"Audio processing failed: {result.error}")

        if not result.text.strip():
            logger.warning(f"No text transcribed from audio: {file_path.name}")

        # Create source location
        source_location = SourceLocation(
            source_type=SourceType.AUDIO,
            title=file_path.stem,
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="audio",
            chapters=[file_path],
            texts=[result.text],
            metadata={
                "title": file_path.stem,
                "duration": result.metadata.duration,
                "language": result.metadata.language,
                "model_used": result.metadata.model_used,
                "word_count": result.word_count,
                "chunk_count": result.chunk_count,
            },
            source_location=source_location,
        )

    def _process_apex(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process Apex class or trigger file."""
        from ingestforge.ingest.apex_processor import ApexProcessor
        from ingestforge.core.provenance import SourceLocation, SourceType

        processor = ApexProcessor()
        result = processor.process(file_path)

        # Build source location
        source_location = SourceLocation(
            source_type=SourceType.CODE,
            title=result.metadata.get("name", file_path.stem),
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="apex",
            chapters=[file_path],
            texts=[result.text],
            metadata=result.metadata,
            source_location=source_location,
        )

    def _process_lwc(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process Lightning Web Component JavaScript file."""
        from ingestforge.ingest.lwc_processor import LWCProcessor
        from ingestforge.core.provenance import SourceLocation, SourceType

        processor = LWCProcessor()

        # Only process if it's actually an LWC file (in lwc folder)
        if "lwc" not in str(file_path).lower():
            # Fallback to text processing for non-LWC JS files
            result = self.extractor.extract_with_metadata(file_path)
            return ProcessedDocument(
                document_id=document_id,
                source_file=str(file_path),
                file_type="js",
                chapters=[file_path],
                texts=[result["text"]],
                metadata={"word_count": result["word_count"]},
            )

        result = processor.process(file_path)

        source_location = SourceLocation(
            source_type=SourceType.CODE,
            title=result.metadata.get("name", file_path.stem),
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="lwc",
            chapters=[file_path],
            texts=[result.text],
            metadata=result.metadata,
            source_location=source_location,
        )

    def _process_latex(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process LaTeX document file."""
        from ingestforge.ingest.latex_processor import LaTeXProcessor
        from ingestforge.core.provenance import SourceLocation, SourceType

        processor = LaTeXProcessor()
        result = processor.process(file_path)

        # Build source location
        title = result["metadata"].get("title", file_path.stem)
        source_location = SourceLocation(
            source_type=SourceType.TEXT,
            title=title,
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="latex",
            chapters=[file_path],
            texts=[result["text"]],
            metadata=result["metadata"],
            source_location=source_location,
        )

    def _process_jupyter(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process Jupyter notebook file."""
        from ingestforge.ingest.jupyter_processor import JupyterProcessor
        from ingestforge.core.provenance import SourceLocation, SourceType

        processor = JupyterProcessor()
        result = processor.process(file_path)

        # Build source location
        title = result["metadata"].get("title", file_path.stem)
        source_location = SourceLocation(
            source_type=SourceType.CODE,
            title=title,
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="jupyter",
            chapters=[file_path],
            texts=[result["text"]],
            metadata=result["metadata"],
            source_location=source_location,
        )

    def _process_markdown(self, file_path: Path, document_id: str) -> ProcessedDocument:
        """Process Markdown document file with enhanced extraction."""
        from ingestforge.ingest.markdown_processor import MarkdownProcessor
        from ingestforge.core.provenance import SourceLocation, SourceType

        # Check if this is an ADO work item export first
        try:
            content = file_path.read_text(encoding="utf-8")
            # ADO exports have a specific format with | ID | Type | State | table
            if "| ID |" in content and (
                "| Type |" in content or "| Work Item Type |" in content
            ):
                return self._process_ado(file_path, document_id)
        except Exception as e:
            logger.debug(f"Failed ADO format detection: {e}")

        processor = MarkdownProcessor()
        result = processor.process(file_path)

        # Build source location
        title = result["metadata"].get("title", file_path.stem)
        source_location = SourceLocation(
            source_type=SourceType.MARKDOWN,
            title=title,
            file_path=str(file_path),
        )

        return ProcessedDocument(
            document_id=document_id,
            source_file=str(file_path),
            file_type="markdown",
            chapters=[file_path],
            texts=[result["text"]],
            metadata=result["metadata"],
            source_location=source_location,
        )

    def is_supported(self, file_path: Path) -> bool:
        """Check if file type is supported.

        Uses the consolidated handler registry for consistency.
        """
        from ingestforge.ingest.html_processor import HTMLProcessor

        suffix = file_path.suffix.lower()

        # Check handler registry (single source of truth)
        if suffix in get_supported_suffixes():
            return True

        # Check config-based supported formats
        if suffix in self.config.ingest.supported_formats:
            return True

        # Use HTMLProcessor.can_process() for special HTML detection
        html_processor = HTMLProcessor()
        return html_processor.can_process(file_path)
