"""OCR processing for scanned documents."""

from pathlib import Path
from typing import Any, List, Tuple
from dataclasses import dataclass

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OCRResult:
    """Result of OCR processing."""

    text: str
    pages: List[str]  # Text per page
    confidence: float  # Average confidence 0-1
    is_scanned: bool  # Whether document needed OCR
    page_count: int
    language: str = "eng"  # Language used for OCR
    engine: str = "tesseract"  # OCR engine used
    scanned_page_count: int = 0  # Number of pages that required OCR

    @property
    def scanned_ratio(self) -> float:
        """Fraction of pages that required OCR (0.0 to 1.0)."""
        if self.page_count <= 0:
            return 0.0
        return self.scanned_page_count / self.page_count

    def is_majority_scanned(self, threshold: float = 0.5) -> bool:
        """Whether the document is majority-scanned above the given threshold."""
        return self.is_scanned and self.scanned_ratio > threshold


class OCRProcessor:
    """
    OCR processor for scanned documents.

    Uses Tesseract for OCR. Falls back gracefully if not installed.
    """

    def __init__(self) -> None:
        self._tesseract_available = None
        self._pytesseract = None

    @property
    def tesseract_available(self) -> bool:
        """Check if Tesseract is available."""
        if self._tesseract_available is None:
            try:
                import pytesseract

                # Try to get version to verify it works
                pytesseract.get_tesseract_version()
                self._pytesseract = pytesseract
                self._tesseract_available = True
            except Exception:
                self._tesseract_available = False
        return self._tesseract_available

    def is_scanned_pdf(self, file_path: Path) -> bool:
        """
        Detect if a PDF is scanned (image-based) vs text-based.

        Args:
            file_path: Path to PDF file

        Returns:
            True if PDF appears to be scanned/image-based
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(file_path))

            # Check first few pages
            pages_to_check = min(3, len(doc))
            text_chars = 0

            for i in range(pages_to_check):
                page = doc[i]
                text = page.get_text()
                text_chars += len(text.strip())

            doc.close()

            # If very little text extracted, likely scanned
            avg_chars_per_page = (
                text_chars / pages_to_check if pages_to_check > 0 else 0
            )

            # Threshold: less than 100 chars per page suggests scanned
            return avg_chars_per_page < 100

        except ImportError:
            logger.warning("PyMuPDF not available for PDF analysis")
            return False
        except Exception as e:
            logger.warning(f"Error checking if PDF is scanned: {e}")
            return False

    def process_pdf(self, file_path: Path) -> OCRResult:
        """
        Process a PDF with OCR if needed.

        Args:
            file_path: Path to PDF file

        Returns:
            OCRResult with extracted text
        """
        is_scanned = self.is_scanned_pdf(file_path)

        if not is_scanned:
            # Not scanned, extract text normally
            return self._extract_text_pdf(file_path)

        if not self.tesseract_available:
            logger.warning(
                "PDF appears to be scanned but Tesseract is not available. "
                "Install with: pip install pytesseract"
            )
            # Return empty result
            return OCRResult(
                text="",
                pages=[],
                confidence=0.0,
                is_scanned=True,
                page_count=0,
            )

        return self._ocr_pdf(file_path)

    def process_image(self, file_path: Path) -> OCRResult:
        """
        Process an image file with OCR.

        Args:
            file_path: Path to image file

        Returns:
            OCRResult with extracted text
        """
        if not self.tesseract_available:
            raise RuntimeError(
                "Tesseract is not available. " "Install with: pip install pytesseract"
            )

        return self._ocr_image(file_path)

    def _extract_text_pdf(self, file_path: Path) -> OCRResult:
        """Extract text from a text-based PDF."""
        try:
            import fitz

            doc = fitz.open(str(file_path))
            pages = []

            for page in doc:
                text = page.get_text()
                pages.append(text)

            doc.close()

            return OCRResult(
                text="\n\n".join(pages),
                pages=pages,
                confidence=1.0,  # Native text extraction
                is_scanned=False,
                page_count=len(pages),
            )

        except ImportError:
            raise RuntimeError("PyMuPDF required for PDF processing")

    def _ocr_pdf(self, file_path: Path) -> OCRResult:
        """OCR a scanned PDF."""
        try:
            import fitz
            from PIL import Image
            import io

            doc = fitz.open(str(file_path))
            pages = []
            confidences = []

            for page_num, page in enumerate(doc):
                # Render page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better OCR
                pix = page.get_pixmap(matrix=mat)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # OCR the image
                text, conf = self._ocr_pil_image(img)
                pages.append(text)
                confidences.append(conf)

                logger.info(
                    f"OCR page {page_num + 1}/{len(doc)}: {len(text)} chars, {conf:.0%} confidence"
                )

            doc.close()

            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            return OCRResult(
                text="\n\n".join(pages),
                pages=pages,
                confidence=avg_confidence,
                is_scanned=True,
                page_count=len(pages),
            )

        except ImportError as e:
            raise RuntimeError(f"Required library not available: {e}")

    def _ocr_image(self, file_path: Path) -> OCRResult:
        """OCR a single image file."""
        from PIL import Image

        img = Image.open(file_path)
        text, confidence = self._ocr_pil_image(img)

        return OCRResult(
            text=text,
            pages=[text],
            confidence=confidence,
            is_scanned=True,
            page_count=1,
        )

    def _ocr_pil_image(self, img: Any) -> Tuple[str, float]:
        """OCR a PIL Image and return text and confidence."""
        import pytesseract

        # Get detailed data including confidence
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Extract text
        text = pytesseract.image_to_string(img)

        # Calculate average confidence (filter out -1 which means no text)
        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0

        return text.strip(), avg_conf
