"""
Unified OCR Engine Manager.

Provides a single entry point for all OCR operations, wrapping
Tesseract and EasyOCR engines behind a common interface.

Follows the factory pattern used in llm/factory.py — callers use
get_ocr_engine() or get_best_available_engine() and get back an
OCREngine that delegates to whichever backend is configured/available.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.ingest.ocr_processor import OCRResult

logger = get_logger(__name__)


class OCREngine:
    """
    Unified OCR engine wrapping Tesseract and EasyOCR.

    Lazily resolves which backend to use based on config and availability.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self._engine_name: Optional[str] = None
        self._tesseract_processor = None  # cached OCRProcessor for Tesseract path

    @property
    def engine_name(self) -> str:
        """Resolved engine name: 'tesseract', 'easyocr', or 'none'."""
        if self._engine_name is None:
            self._engine_name = self._resolve_engine()
        return self._engine_name

    @property
    def is_available(self) -> bool:
        """Whether any OCR engine is available."""
        return self.engine_name != "none"

    def _resolve_engine(self) -> str:
        """Determine which engine to use based on config preference and availability."""
        preferred = self.config.ocr.preferred_engine.lower()

        if preferred == "tesseract":
            return "tesseract" if self._tesseract_available() else "none"
        elif preferred == "easyocr":
            return "easyocr" if self._easyocr_available() else "none"
        else:
            # "auto" — try Tesseract first (lighter weight), then EasyOCR
            if self._tesseract_available():
                return "tesseract"
            if self._easyocr_available():
                return "easyocr"
            return "none"

    def _tesseract_available(self) -> bool:
        """Check if Tesseract is usable without importing pytesseract."""
        import importlib.util
        import shutil

        # Lightweight check: is the Python package installable?
        if importlib.util.find_spec("pytesseract") is None:
            return False
        # Is the tesseract binary on PATH?
        if shutil.which("tesseract") is None:
            return False
        return True

    def _easyocr_available(self) -> bool:
        """Check if EasyOCR is installed without importing it."""
        import importlib.util

        return importlib.util.find_spec("easyocr") is not None

    # ── Public API ──────────────────────────────────────────────

    def process_image(self, file_path: Path) -> OCRResult:
        """
        OCR a single image file.

        Args:
            file_path: Path to image file (PNG, JPG, TIFF, etc.)

        Returns:
            OCRResult with extracted text

        Raises:
            RuntimeError: If no OCR engine is available
        """
        if not self.is_available:
            raise RuntimeError(
                "No OCR engine available. "
                "Install with: pip install 'ingestforge[ocr]'"
            )

        if self.engine_name == "easyocr":
            from ingestforge.ingest.easyocr_processor import process_image_easyocr

            result = process_image_easyocr(
                file_path,
                languages=self.config.ocr.languages,
                use_gpu=self.config.ocr.use_gpu,
            )
        else:
            if self._tesseract_processor is None:
                from ingestforge.ingest.ocr_processor import OCRProcessor

                self._tesseract_processor = OCRProcessor()
            result = self._tesseract_processor.process_image(file_path)

        # Apply confidence threshold filtering
        if result.confidence < self.config.ocr.confidence_threshold:
            logger.warning(
                f"OCR confidence {result.confidence:.0%} below threshold "
                f"{self.config.ocr.confidence_threshold:.0%} for {file_path.name}"
            )

        return result

    def process_pdf(self, file_path: Path) -> OCRResult:
        """
        Process a PDF with per-page hybrid analysis.

        Uses a two-tier strategy:
        1. Fast 3-page sample via is_scanned_pdf() — if the sample shows
           a digital PDF, return immediately without iterating every page.
        2. Full per-page hybrid analysis via _hybrid_pdf() for PDFs that
           appear to have scanned content.

        Args:
            file_path: Path to PDF file

        Returns:
            OCRResult with extracted text
        """
        if not self.is_available:
            logger.warning("No OCR engine available for PDF processing")
            return OCRResult(
                text="",
                pages=[],
                confidence=0.0,
                is_scanned=True,
                page_count=0,
                engine="none",
            )

        # Fast-path: 3-page sample detects clearly digital PDFs without
        # iterating every page (avoids full _hybrid_pdf overhead for the
        # common case of text-based documents).
        if not self.is_scanned_pdf(file_path):
            try:
                import fitz

                with fitz.open(str(file_path)) as doc:
                    page_count = len(doc)
            except Exception:
                page_count = 0
            return OCRResult(
                text="",
                pages=[],
                confidence=1.0,
                is_scanned=False,
                page_count=page_count,
                engine=self.engine_name,
                scanned_page_count=0,
            )

        return self._hybrid_pdf(file_path)

    def is_scanned_pdf(self, file_path: Path) -> bool:
        """
        Detect if a PDF is scanned (image-based) vs text-based.

        Checks the first few pages' text density against the configured
        scanned_threshold. Uses context manager for safe resource cleanup.
        """
        try:
            import fitz  # PyMuPDF

            with fitz.open(str(file_path)) as doc:
                pages_to_check = min(3, len(doc))
                text_chars = 0

                for i in range(pages_to_check):
                    page = doc[i]
                    text = page.get_text()
                    text_chars += len(text.strip())

            avg_chars_per_page = (
                text_chars / pages_to_check if pages_to_check > 0 else 0
            )
            return avg_chars_per_page < self.config.ocr.scanned_threshold

        except ImportError:
            logger.warning("PyMuPDF not available for PDF analysis")
            return False
        except Exception as e:
            logger.warning(f"Error checking if PDF is scanned: {e}")
            return False

    def extract_text(self, file_path: Any, language: str = None) -> dict[str, Any]:
        """
        Duck-type interface for SlideImageOCR compatibility.

        SlideImageOCR calls ocr_engine.extract_text(path, language=...) and
        expects a dict with 'text', 'confidence', 'language', 'word_count'.
        """
        path = Path(file_path)
        lang = language or self.config.ocr.language

        if not self.is_available:
            return {
                "text": "",
                "confidence": 0.0,
                "language": lang,
                "word_count": 0,
                "error": "No OCR engine available",
            }

        try:
            result = self.process_image(path)
            text = result.text
            return {
                "text": text,
                "confidence": result.confidence,
                "language": lang,
                "word_count": len(text.split()) if text else 0,
            }
        except Exception as e:
            return {
                "text": "",
                "confidence": 0.0,
                "language": lang,
                "word_count": 0,
                "error": str(e),
            }

    # ── Private helpers ─────────────────────────────────────────

    def _hybrid_pdf(self, file_path: Path) -> OCRResult:
        """
        Per-page hybrid PDF processing.

        Classifies each page by text density: pages above the scanned
        threshold use direct text extraction; pages below it are OCR'd.
        """
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) required for PDF processing")

        page_texts: List[Optional[str]] = []
        page_confidences: List[float] = []
        scanned_indices: List[int] = []

        with fitz.open(str(file_path)) as doc:
            # First pass: classify and extract digital text
            self._classify_pages(doc, page_texts, page_confidences, scanned_indices)

            # Second pass: OCR scanned pages
            if scanned_indices:
                self._ocr_scanned_pages(
                    doc, scanned_indices, page_texts, page_confidences
                )

        return self._build_ocr_result(page_texts, page_confidences, scanned_indices)

    def _classify_pages(
        self, doc: Any, page_texts: List, page_confidences: List, scanned_indices: List
    ):
        """Classify pages as digital or scanned based on text density."""
        threshold = self.config.ocr.scanned_threshold

        for page_num in range(len(doc)):
            page = doc[page_num]
            native_text = page.get_text().strip()

            if len(native_text) >= threshold:
                page_texts.append(native_text)
                page_confidences.append(1.0)
            else:
                page_texts.append(None)
                page_confidences.append(0.0)
                scanned_indices.append(page_num)

    def _ocr_scanned_pages(
        self,
        doc: Any,
        scanned_indices: List[int],
        page_texts: List,
        page_confidences: List,
    ):
        """OCR scanned pages in batches to limit memory."""
        max_workers = min(self.config.ocr.max_workers, len(scanned_indices))
        batch_size = max(max_workers, 1)
        total = len(doc)

        for batch_start in range(0, len(scanned_indices), batch_size):
            batch = scanned_indices[batch_start : batch_start + batch_size]
            rendered = self._render_pages_to_png(doc, batch)
            self._process_batch(
                rendered, batch, total, max_workers, page_texts, page_confidences
            )
            del rendered

    def _render_pages_to_png(self, doc: Any, batch: List[int]) -> dict[str, Any]:
        """Render PDF pages to PNG bytes for OCR."""
        try:
            import fitz
        except ImportError:
            raise RuntimeError("PyMuPDF required")

        rendered = {}
        for idx in batch:
            page = doc[idx]
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            rendered[idx] = pix.tobytes("png")
        return rendered

    def _process_batch(
        self,
        rendered: dict,
        batch: List[int],
        total: int,
        max_workers: int,
        page_texts: List,
        page_confidences: List,
    ):
        """Process batch of rendered pages with OCR."""
        if max_workers > 1:
            self._process_batch_parallel(
                rendered, batch, total, max_workers, page_texts, page_confidences
            )
        else:
            self._process_batch_sequential(
                rendered, batch, total, page_texts, page_confidences
            )

    def _process_batch_parallel(
        self,
        rendered: dict,
        batch: List[int],
        total: int,
        max_workers: int,
        page_texts: List,
        page_confidences: List,
    ):
        """Process batch in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor

        timeout = self.config.ocr.page_timeout or None

        def _ocr_rendered(page_num: int) -> Tuple[int, str, float]:
            text, conf = self._ocr_from_bytes(
                rendered[page_num], page_num, total, use_timeout=False
            )
            return page_num, text, conf

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_ocr_rendered, idx) for idx in batch]
            for future in futures:
                try:
                    pn, text, conf = future.result(timeout=timeout)
                    page_texts[pn] = text
                    page_confidences[pn] = conf
                except Exception as e:
                    logger.warning(f"OCR worker failed: {e}, skipping page")

    def _process_batch_sequential(
        self,
        rendered: dict,
        batch: List[int],
        total: int,
        page_texts: List,
        page_confidences: List,
    ):
        """Process batch sequentially."""
        for idx in batch:
            ocr_text, conf = self._ocr_from_bytes(rendered[idx], idx, total)
            page_texts[idx] = ocr_text
            page_confidences[idx] = conf

    def _build_ocr_result(
        self, page_texts: List, page_confidences: List, scanned_indices: List
    ) -> OCRResult:
        """Build final OCR result from processed pages."""
        pages: List[str] = [t or "" for t in page_texts]

        avg_confidence = (
            sum(page_confidences) / len(page_confidences) if page_confidences else 0.0
        )

        return OCRResult(
            text="\n\n".join(pages),
            pages=pages,
            confidence=avg_confidence,
            is_scanned=len(scanned_indices) > 0,
            page_count=len(pages),
            language=self.config.ocr.language,
            engine=self.engine_name,
            scanned_page_count=len(scanned_indices),
        )

    def _ocr_from_bytes(
        self,
        img_data: bytes,
        page_num: int,
        total: int,
        use_timeout: bool = True,
    ) -> tuple:
        """
        OCR from pre-rendered PNG bytes. Thread-safe (no fitz dependency).

        Rule #4: Reduced from 78 lines to <60 lines via helper extraction

        When use_timeout=True (default, single-page path), wraps the OCR
        call in a single-thread executor for timeout protection. When
        use_timeout=False (concurrent path), runs OCR directly since the
        outer executor already handles timeouts.

        Returns (text, confidence).
        """
        try:
            if use_timeout:
                text, conf = self._execute_ocr_with_timeout(img_data)
            else:
                text, conf = self._execute_ocr_direct(img_data)
        except Exception as e:
            # Covers FuturesTimeout, RuntimeError, etc.
            self._log_ocr_error(e, page_num, total)
            return "", 0.0

        logger.info(
            f"OCR page {page_num + 1}/{total}: "
            f"{len(text)} chars, {conf:.0%} confidence"
        )
        return text, conf

    def _execute_ocr_with_timeout(self, img_data: bytes) -> tuple:
        """
        Execute OCR with timeout protection.

        Rule #4: Extracted to reduce function size
        """
        from concurrent.futures import ThreadPoolExecutor

        timeout = self.config.ocr.page_timeout or None
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._execute_ocr_direct, img_data)
            return future.result(timeout=timeout)

    def _execute_ocr_direct(self, img_data: bytes) -> tuple:
        """
        Execute OCR directly (no timeout wrapper).

        Rule #4: Extracted to reduce function size
        """
        if self.engine_name == "easyocr":
            return self._ocr_easyocr_from_bytes(img_data)
        return self._ocr_tesseract_from_bytes(img_data)

    def _ocr_easyocr_from_bytes(self, img_data: bytes) -> tuple:
        """
        OCR using EasyOCR engine.

        Rule #4: Extracted to reduce function size
        """
        import tempfile
        import os
        from ingestforge.ingest.easyocr_processor import process_image_easyocr

        fd, tmp = tempfile.mkstemp(suffix=".png")
        try:
            os.write(fd, img_data)
            os.close(fd)
            result = process_image_easyocr(
                Path(tmp),
                languages=self.config.ocr.languages,
                use_gpu=self.config.ocr.use_gpu,
            )
            return result.text, result.confidence
        finally:
            try:
                os.unlink(tmp)
            except OSError as e:
                logger.debug(f"Failed to delete temporary OCR file {tmp}: {e}")

    def _ocr_tesseract_from_bytes(self, img_data: bytes) -> tuple:
        """
        OCR using Tesseract engine.

        Rule #4: Extracted to reduce function size
        """
        from PIL import Image
        import io

        img = Image.open(io.BytesIO(img_data))
        # Reuse cached processor to avoid re-creating per page
        if self._tesseract_processor is None:
            from ingestforge.ingest.ocr_processor import OCRProcessor

            self._tesseract_processor = OCRProcessor()
        return self._tesseract_processor._ocr_pil_image(img)

    def _log_ocr_error(self, error: Exception, page_num: int, total: int) -> None:
        """
        Log OCR error with appropriate message.

        Rule #4: Extracted to reduce function size
        """
        if "timeout" in type(error).__name__.lower() or "timeout" in str(error).lower():
            timeout_val = self.config.ocr.page_timeout
            logger.warning(
                f"OCR timeout on page {page_num + 1}/{total} "
                f"(limit: {timeout_val}s), skipping"
            )
        else:
            logger.warning(
                f"OCR error on page {page_num + 1}/{total}: {error}, skipping"
            )


# ── Factory functions ─────────────────────────────────────────────


def get_ocr_engine(config: Config) -> OCREngine:
    """
    Get an OCR engine based on configuration.

    Always returns an OCREngine instance. Check .is_available to see
    if any backend was found.
    """
    return OCREngine(config)


def get_best_available_engine(config: Config) -> Optional[OCREngine]:
    """
    Get an OCR engine only if one is available.

    Returns None if no OCR backend can be found, making it safe to use
    in optional-OCR code paths.
    """
    engine = OCREngine(config)
    return engine if engine.is_available else None
