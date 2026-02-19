"""EasyOCR processor - Alternative OCR engine to Tesseract."""

import threading
from pathlib import Path
from typing import Dict, Optional, List, Tuple

from ingestforge.core.logging import get_logger
from ingestforge.ingest.ocr_processor import OCRResult

logger = get_logger(__name__)

# Module-level cache: keyed by (languages, gpu) to reuse Reader.
# EasyOCR Reader loads ~100-500MB of model weights, so re-creating it per
# call is extremely wasteful. Protected by _reader_lock for thread-safety
# when called from concurrent OCR in _hybrid_pdf.
_reader_cache: Dict[Tuple, object] = {}
_reader_lock = threading.Lock()


def is_easyocr_available() -> bool:
    """Check if EasyOCR is installed without importing it."""
    import importlib.util

    return importlib.util.find_spec("easyocr") is not None


def is_tesseract_available() -> bool:
    """Check if Tesseract OCR is available without importing pytesseract."""
    import importlib.util
    import shutil

    if importlib.util.find_spec("pytesseract") is None:
        return False
    return shutil.which("tesseract") is not None


def process_image_easyocr(
    image_path: Path,
    languages: Optional[List[str]] = None,
    use_gpu: Optional[bool] = None,
) -> OCRResult:
    """
    Process an image using EasyOCR.

    Args:
        image_path: Path to the image file
        languages: List of language codes (default: ["en"])
        use_gpu: Force GPU on/off; None uses auto-detection

    Returns:
        OCRResult with extracted text
    """
    try:
        import easyocr
    except ImportError:
        raise ImportError("EasyOCR is not installed. Install with: pip install easyocr")

    if languages is None:
        languages = ["en"]

    gpu = use_gpu if use_gpu is not None else _has_gpu()

    # Cache key: (sorted languages, gpu flag) — avoids reloading ~100-500MB model
    cache_key = (tuple(sorted(languages)), gpu)
    with _reader_lock:
        if cache_key not in _reader_cache:
            logger.info(f"Creating EasyOCR Reader (languages={languages}, gpu={gpu})")
            _reader_cache[cache_key] = easyocr.Reader(languages, gpu=gpu)
        reader = _reader_cache[cache_key]

    results = reader.readtext(str(image_path))

    # Combine all text blocks
    text_blocks = []
    total_confidence = 0.0

    for bbox, text, confidence in results:
        text_blocks.append(text)
        total_confidence += confidence

    full_text = " ".join(text_blocks)
    avg_confidence = total_confidence / len(results) if results else 0.0

    return OCRResult(
        text=full_text,
        pages=[full_text],
        confidence=avg_confidence,
        is_scanned=True,
        page_count=1,
        language=languages[0],
        engine="easyocr",
    )


def _has_gpu() -> bool:
    """Check if GPU is available for EasyOCR.

    Uses find_spec first to avoid eagerly importing torch (~2-5s overhead)
    when it's not installed.
    """
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
