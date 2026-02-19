"""
Embedded image OCR detector and processor.

Detects images embedded in otherwise text-rich PDF pages and OCRs them.
This handles the common case where tables, diagrams, or formatted content
is rendered as an image within a native-text PDF.

Example: A PDF page with 1500 chars of text AND a 400x300 table image.
Standard extraction gets the text but misses the table content entirely.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple
import io

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


# Minimum image dimensions to consider for OCR (pixels)
MIN_IMAGE_WIDTH = 100
MIN_IMAGE_HEIGHT = 50

# Minimum image area as fraction of page area to consider (0.05 = 5%)
MIN_IMAGE_AREA_RATIO = 0.03

# Maximum aspect ratio (width/height or height/width) - filters out thin lines
MAX_ASPECT_RATIO = 20.0


@dataclass
class EmbeddedImage:
    """An image embedded in a PDF page."""

    page_num: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    width: float
    height: float
    area: float
    image_data: Optional[bytes] = None
    ocr_text: str = ""
    ocr_confidence: float = 0.0

    @property
    def aspect_ratio(self) -> float:
        """Width/height ratio."""
        if self.height == 0:
            return float("inf")
        return self.width / self.height


@dataclass
class PageOCRResult:
    """OCR results for a single page."""

    page_num: int
    native_text: str
    embedded_images: List[EmbeddedImage] = field(default_factory=list)
    combined_text: str = ""
    had_embedded_images: bool = False
    ocr_performed: bool = False

    @property
    def image_text(self) -> str:
        """Combined text from all OCR'd images."""
        return "\n\n".join(
            img.ocr_text for img in self.embedded_images if img.ocr_text.strip()
        )


class EmbeddedImageOCR:
    """
    Detect and OCR images embedded in PDF pages.

    This processor identifies pages that have significant image content
    alongside native text, extracts those images, and OCRs them to
    capture content that would otherwise be missed.
    """

    def __init__(
        self,
        min_image_width: int = MIN_IMAGE_WIDTH,
        min_image_height: int = MIN_IMAGE_HEIGHT,
        min_area_ratio: float = MIN_IMAGE_AREA_RATIO,
        max_aspect_ratio: float = MAX_ASPECT_RATIO,
    ) -> None:
        self.min_image_width = min_image_width
        self.min_image_height = min_image_height
        self.min_area_ratio = min_area_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self._fitz: Any = None
        self._ocr_engine: Optional[str] = None

    @property
    def fitz(self) -> Any:
        """Lazy-load PyMuPDF."""
        if self._fitz is None:
            import fitz

            fitz.TOOLS.mupdf_display_errors(False)
            self._fitz = fitz
        return self._fitz

    def _get_ocr_engine(self) -> Optional[str]:
        """Detect available OCR engine."""
        if self._ocr_engine is not None:
            return self._ocr_engine

        # Try EasyOCR first (pure Python, no system deps)
        try:
            from ingestforge.ingest.easyocr_processor import is_easyocr_available

            if is_easyocr_available():
                self._ocr_engine = "easyocr"
                return self._ocr_engine
        except ImportError:
            pass

        # Fall back to Tesseract
        try:
            from ingestforge.ingest.easyocr_processor import is_tesseract_available

            if is_tesseract_available():
                self._ocr_engine = "tesseract"
                return self._ocr_engine
        except ImportError:
            pass

        self._ocr_engine = ""  # Empty string = none available
        return None

    def detect_embedded_images(
        self,
        file_path: Path,
        page_nums: Optional[List[int]] = None,
    ) -> List[EmbeddedImage]:
        """
        Detect significant embedded images in a PDF.

        Args:
            file_path: Path to PDF file
            page_nums: Specific pages to check (1-indexed), or None for all

        Returns:
            List of EmbeddedImage objects for images worth OCRing
        """
        doc = self.fitz.open(file_path)
        images = []

        pages_to_check = (
            range(len(doc)) if page_nums is None else [p - 1 for p in page_nums]
        )

        for page_idx in pages_to_check:
            if page_idx < 0 or page_idx >= len(doc):
                continue

            page = doc[page_idx]
            page_area = page.rect.width * page.rect.height

            # Get all images on the page
            image_list = page.get_images(full=True)

            for img_info in image_list:
                xref = img_info[0]

                # Get image bbox on page
                img_rects = page.get_image_rects(xref)
                if not img_rects:
                    continue

                for rect in img_rects:
                    width = rect.width
                    height = rect.height
                    area = width * height

                    # Filter: too small
                    if width < self.min_image_width or height < self.min_image_height:
                        continue

                    # Filter: too small relative to page
                    if area / page_area < self.min_area_ratio:
                        continue

                    # Filter: extreme aspect ratio (likely decorative)
                    aspect = width / height if height > 0 else float("inf")
                    if (
                        aspect > self.max_aspect_ratio
                        or aspect < 1 / self.max_aspect_ratio
                    ):
                        continue

                    images.append(
                        EmbeddedImage(
                            page_num=page_idx + 1,
                            bbox=(rect.x0, rect.y0, rect.x1, rect.y1),
                            width=width,
                            height=height,
                            area=area,
                        )
                    )

        doc.close()
        logger.info(
            f"Found {len(images)} significant embedded images in {file_path.name}"
        )
        return images

    def extract_image_region(
        self,
        file_path: Path,
        page_num: int,
        bbox: Tuple[float, float, float, float],
        zoom: float = 2.0,
    ) -> bytes:
        """
        Extract an image region from a PDF page as PNG bytes.

        Args:
            file_path: Path to PDF
            page_num: Page number (1-indexed)
            bbox: Bounding box (x0, y0, x1, y1)
            zoom: Zoom factor for better OCR quality

        Returns:
            PNG image data as bytes
        """
        doc = self.fitz.open(file_path)
        page = doc[page_num - 1]

        # Create clip rectangle
        clip = self.fitz.Rect(bbox)

        # Render at higher resolution for OCR
        mat = self.fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)

        img_data = pix.tobytes("png")
        doc.close()

        return img_data

    def ocr_image_bytes(self, image_data: bytes) -> Tuple[str, float]:
        """
        OCR image data and return text with confidence.

        Args:
            image_data: PNG image bytes

        Returns:
            Tuple of (extracted_text, confidence)
        """
        engine = self._get_ocr_engine()

        if engine == "easyocr":
            return self._ocr_with_easyocr(image_data)
        elif engine == "tesseract":
            return self._ocr_with_tesseract(image_data)
        else:
            logger.warning("No OCR engine available (install easyocr or pytesseract)")
            return "", 0.0

    def _ocr_with_easyocr(self, image_data: bytes) -> Tuple[str, float]:
        """OCR using EasyOCR."""
        from PIL import Image
        import easyocr

        # Load image
        img = Image.open(io.BytesIO(image_data))

        # Get or create reader (cached)
        from ingestforge.ingest.easyocr_processor import _reader_cache, _reader_lock

        cache_key = (("en",), False)  # Default: English, no GPU for quick processing
        with _reader_lock:
            if cache_key not in _reader_cache:
                _reader_cache[cache_key] = easyocr.Reader(["en"], gpu=False)
            reader = _reader_cache[cache_key]

        # Convert PIL to numpy for EasyOCR
        import numpy as np

        img_array = np.array(img)

        results = reader.readtext(img_array)

        if not results:
            return "", 0.0

        # Combine results
        texts = []
        total_conf = 0.0
        for bbox, text, conf in results:
            texts.append(text)
            total_conf += conf

        avg_conf = total_conf / len(results)
        return " ".join(texts), avg_conf

    def _ocr_with_tesseract(self, image_data: bytes) -> Tuple[str, float]:
        """OCR using Tesseract."""
        from PIL import Image
        import pytesseract

        img = Image.open(io.BytesIO(image_data))

        # Get text
        text = pytesseract.image_to_string(img)

        # Get confidence
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        avg_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0

        return text.strip(), avg_conf

    def process_page(
        self,
        file_path: Path,
        page_num: int,
        native_text: str,
    ) -> PageOCRResult:
        """
        Process a single page, OCRing any embedded images.

        Args:
            file_path: Path to PDF
            page_num: Page number (1-indexed)
            native_text: Already-extracted native text

        Returns:
            PageOCRResult with combined text
        """
        result = PageOCRResult(
            page_num=page_num,
            native_text=native_text,
        )

        # Detect images on this page
        images = self.detect_embedded_images(file_path, page_nums=[page_num])

        if not images:
            result.combined_text = native_text
            return result

        result.had_embedded_images = True

        # OCR each image
        for img in images:
            logger.info(
                f"Page {page_num}: OCRing embedded image "
                f"({img.width:.0f}x{img.height:.0f} at {img.bbox[:2]})"
            )

            # Extract image region
            img.image_data = self.extract_image_region(file_path, page_num, img.bbox)

            # OCR it
            img.ocr_text, img.ocr_confidence = self.ocr_image_bytes(img.image_data)

            if img.ocr_text.strip():
                logger.info(
                    f"  -> Extracted {len(img.ocr_text)} chars "
                    f"({img.ocr_confidence:.0%} confidence)"
                )
                result.ocr_performed = True

        result.embedded_images = images

        # Combine native text with OCR'd image text
        # Strategy: Append image text after native text, with clear OCR markers
        # Format is LLM-friendly and clearly indicates reduced certainty
        parts = [native_text.strip()] if native_text.strip() else []

        for img in images:
            if img.ocr_text.strip():
                confidence_pct = int(img.ocr_confidence * 100)
                ocr_block = (
                    f"\n[OCR_START confidence={confidence_pct}%]\n"
                    f"{img.ocr_text.strip()}\n"
                    f"[OCR_END]"
                )
                parts.append(ocr_block)

        result.combined_text = "\n\n".join(parts)
        return result

    def process_pdf(
        self,
        file_path: Path,
        native_texts: Optional[List[str]] = None,
    ) -> List[PageOCRResult]:
        """
        Process entire PDF, detecting and OCRing embedded images.

        Args:
            file_path: Path to PDF
            native_texts: Pre-extracted native text per page, or None to extract

        Returns:
            List of PageOCRResult, one per page
        """
        doc = self.fitz.open(file_path)
        page_count = len(doc)

        # Get native text if not provided
        if native_texts is None:
            native_texts = []
            for page in doc:
                native_texts.append(page.get_text())

        doc.close()

        # Process each page
        results = []
        for i, native_text in enumerate(native_texts):
            page_num = i + 1
            result = self.process_page(file_path, page_num, native_text)
            results.append(result)

        # Summary
        pages_with_images = sum(1 for r in results if r.had_embedded_images)
        pages_ocrd = sum(1 for r in results if r.ocr_performed)

        if pages_with_images > 0:
            logger.info(
                f"Processed {file_path.name}: "
                f"{pages_with_images} pages had embedded images, "
                f"{pages_ocrd} required OCR"
            )

        return results


def detect_and_ocr_embedded_images(
    file_path: Path,
    page_nums: Optional[List[int]] = None,
) -> List[PageOCRResult]:
    """
    Convenience function to detect and OCR embedded images in a PDF.

    Args:
        file_path: Path to PDF file
        page_nums: Specific pages to process (1-indexed), or None for all

    Returns:
        List of PageOCRResult objects
    """
    processor = EmbeddedImageOCR()

    if page_nums:
        # Process specific pages
        import fitz

        doc = fitz.open(file_path)
        results = []

        for page_num in page_nums:
            if 1 <= page_num <= len(doc):
                native_text = doc[page_num - 1].get_text()
                result = processor.process_page(file_path, page_num, native_text)
                results.append(result)

        doc.close()
        return results
    else:
        return processor.process_pdf(file_path)


# =============================================================================
# OCR Block Parsing Utilities
# =============================================================================

# Regex pattern for OCR blocks
import re

OCR_BLOCK_PATTERN = re.compile(
    r"\[OCR_START confidence=(\d+)%\]\n(.*?)\n\[OCR_END\]", re.DOTALL
)


@dataclass
class OCRBlock:
    """A parsed OCR block from extracted text."""

    text: str
    confidence: int  # 0-100
    start_pos: int
    end_pos: int

    @property
    def is_high_confidence(self) -> bool:
        """Whether OCR confidence is >= 80%."""
        return self.confidence >= 80

    @property
    def is_low_confidence(self) -> bool:
        """Whether OCR confidence is < 50%."""
        return self.confidence < 50


def contains_ocr_content(text: str) -> bool:
    """
    Check if text contains any OCR-extracted content.

    Args:
        text: Text to check

    Returns:
        True if text contains [OCR_START] blocks
    """
    return "[OCR_START" in text


def extract_ocr_blocks(text: str) -> List[OCRBlock]:
    """
    Extract all OCR blocks from text.

    Args:
        text: Text containing OCR blocks

    Returns:
        List of OCRBlock objects with text, confidence, and positions
    """
    blocks = []
    for match in OCR_BLOCK_PATTERN.finditer(text):
        blocks.append(
            OCRBlock(
                text=match.group(2).strip(),
                confidence=int(match.group(1)),
                start_pos=match.start(),
                end_pos=match.end(),
            )
        )
    return blocks


def get_native_text_only(text: str) -> str:
    """
    Remove all OCR blocks from text, returning only native content.

    Args:
        text: Text potentially containing OCR blocks

    Returns:
        Text with OCR blocks removed
    """
    return OCR_BLOCK_PATTERN.sub("", text).strip()


def get_ocr_text_only(text: str) -> str:
    """
    Extract only OCR'd content from text.

    Args:
        text: Text containing OCR blocks

    Returns:
        Combined OCR text (without markers)
    """
    blocks = extract_ocr_blocks(text)
    return "\n\n".join(b.text for b in blocks)


def get_average_ocr_confidence(text: str) -> Optional[float]:
    """
    Get average OCR confidence across all blocks.

    Args:
        text: Text containing OCR blocks

    Returns:
        Average confidence (0-100) or None if no OCR blocks
    """
    blocks = extract_ocr_blocks(text)
    if not blocks:
        return None
    return sum(b.confidence for b in blocks) / len(blocks)
