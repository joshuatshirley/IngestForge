"""
Document type detection for files and URLs.

Provides multiple detection strategies:
1. Extension-based detection
2. Magic byte/signature detection
3. URL-based detection with MIME type inference
4. Content-Type header parsing for URLs
"""

import mimetypes
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse, unquote

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class DocumentType(Enum):
    """Supported document types."""

    PDF = "pdf"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    XLS = "xls"
    PPTX = "pptx"
    PPT = "ppt"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    EPUB = "epub"
    RTF = "rtf"
    ODT = "odt"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    # Code and notebook formats
    LATEX = "tex"
    JUPYTER = "ipynb"
    # Images
    IMAGE_PNG = "png"
    IMAGE_JPG = "jpg"
    IMAGE_GIF = "gif"
    IMAGE_TIFF = "tiff"
    IMAGE_BMP = "bmp"
    IMAGE_WEBP = "webp"
    UNKNOWN = "unknown"


@dataclass
class DetectionResult:
    """Result of document type detection."""

    document_type: DocumentType
    extension: str
    mime_type: str
    confidence: float  # 0.0 to 1.0
    detection_method: str  # 'extension', 'magic', 'url', 'content_type'
    original_source: str


# Magic byte signatures for common document formats
MAGIC_SIGNATURES: Dict[bytes, Tuple[DocumentType, str]] = {
    # PDF
    b"%PDF": (DocumentType.PDF, "application/pdf"),
    # ZIP-based formats (need further inspection)
    b"PK\x03\x04": (DocumentType.UNKNOWN, "application/zip"),  # ZIP archive
    # MS Office legacy formats
    b"\xd0\xcf\x11\xe0": (DocumentType.DOC, "application/msword"),  # OLE compound
    # Images
    b"\x89PNG\r\n\x1a\n": (DocumentType.IMAGE_PNG, "image/png"),
    b"\xff\xd8\xff": (DocumentType.IMAGE_JPG, "image/jpeg"),
    b"GIF87a": (DocumentType.IMAGE_GIF, "image/gif"),
    b"GIF89a": (DocumentType.IMAGE_GIF, "image/gif"),
    b"II*\x00": (DocumentType.IMAGE_TIFF, "image/tiff"),  # TIFF little-endian
    b"MM\x00*": (DocumentType.IMAGE_TIFF, "image/tiff"),  # TIFF big-endian
    b"BM": (DocumentType.IMAGE_BMP, "image/bmp"),
    b"RIFF": (DocumentType.IMAGE_WEBP, "image/webp"),  # WEBP (needs WEBP check)
    # RTF
    b"{\\rtf": (DocumentType.RTF, "application/rtf"),
    # XML
    b"<?xml": (DocumentType.XML, "application/xml"),
    # HTML
    b"<!DOCTYPE html": (DocumentType.HTML, "text/html"),
    b"<!doctype html": (DocumentType.HTML, "text/html"),
    b"<html": (DocumentType.HTML, "text/html"),
    b"<HTML": (DocumentType.HTML, "text/html"),
}

# ZIP-based format detection (checks internal files)
ZIP_INTERNAL_MARKERS: Dict[str, Tuple[DocumentType, str]] = {
    "word/document.xml": (
        DocumentType.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    "xl/workbook.xml": (
        DocumentType.XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    "ppt/presentation.xml": (
        DocumentType.PPTX,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    "META-INF/container.xml": (DocumentType.EPUB, "application/epub+zip"),
    "content.xml": (DocumentType.ODT, "application/vnd.oasis.opendocument.text"),  # ODF
}

# Extension to type mapping
EXTENSION_MAP: Dict[str, Tuple[DocumentType, str]] = {
    ".pdf": (DocumentType.PDF, "application/pdf"),
    ".docx": (
        DocumentType.DOCX,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
    ".doc": (DocumentType.DOC, "application/msword"),
    ".xlsx": (
        DocumentType.XLSX,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ),
    ".xls": (DocumentType.XLS, "application/vnd.ms-excel"),
    ".pptx": (
        DocumentType.PPTX,
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ),
    ".ppt": (DocumentType.PPT, "application/vnd.ms-powerpoint"),
    ".txt": (DocumentType.TXT, "text/plain"),
    ".md": (DocumentType.MD, "text/markdown"),
    ".markdown": (DocumentType.MD, "text/markdown"),
    ".mdown": (DocumentType.MD, "text/markdown"),
    ".html": (DocumentType.HTML, "text/html"),
    ".htm": (DocumentType.HTML, "text/html"),
    ".mhtml": (DocumentType.HTML, "message/rfc822"),
    ".epub": (DocumentType.EPUB, "application/epub+zip"),
    ".rtf": (DocumentType.RTF, "application/rtf"),
    ".odt": (DocumentType.ODT, "application/vnd.oasis.opendocument.text"),
    ".csv": (DocumentType.CSV, "text/csv"),
    ".json": (DocumentType.JSON, "application/json"),
    ".xml": (DocumentType.XML, "application/xml"),
    # LaTeX formats
    ".tex": (DocumentType.LATEX, "application/x-latex"),
    ".latex": (DocumentType.LATEX, "application/x-latex"),
    ".ltx": (DocumentType.LATEX, "application/x-latex"),
    # Jupyter notebooks
    ".ipynb": (DocumentType.JUPYTER, "application/x-ipynb+json"),
    # Images
    ".png": (DocumentType.IMAGE_PNG, "image/png"),
    ".jpg": (DocumentType.IMAGE_JPG, "image/jpeg"),
    ".jpeg": (DocumentType.IMAGE_JPG, "image/jpeg"),
    ".gif": (DocumentType.IMAGE_GIF, "image/gif"),
    ".tiff": (DocumentType.IMAGE_TIFF, "image/tiff"),
    ".tif": (DocumentType.IMAGE_TIFF, "image/tiff"),
    ".bmp": (DocumentType.IMAGE_BMP, "image/bmp"),
    ".webp": (DocumentType.IMAGE_WEBP, "image/webp"),
}

# MIME type to document type mapping
MIME_TYPE_MAP: Dict[str, DocumentType] = {
    "application/pdf": DocumentType.PDF,
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": DocumentType.DOCX,
    "application/msword": DocumentType.DOC,
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": DocumentType.XLSX,
    "application/vnd.ms-excel": DocumentType.XLS,
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": DocumentType.PPTX,
    "application/vnd.ms-powerpoint": DocumentType.PPT,
    "text/plain": DocumentType.TXT,
    "text/markdown": DocumentType.MD,
    "text/x-markdown": DocumentType.MD,
    "text/html": DocumentType.HTML,
    "application/xhtml+xml": DocumentType.HTML,
    "application/epub+zip": DocumentType.EPUB,
    "application/rtf": DocumentType.RTF,
    "application/vnd.oasis.opendocument.text": DocumentType.ODT,
    "text/csv": DocumentType.CSV,
    "application/json": DocumentType.JSON,
    "application/xml": DocumentType.XML,
    "text/xml": DocumentType.XML,
    # LaTeX
    "application/x-latex": DocumentType.LATEX,
    "text/x-latex": DocumentType.LATEX,
    "application/x-tex": DocumentType.LATEX,
    # Jupyter
    "application/x-ipynb+json": DocumentType.JUPYTER,
    # Images
    "image/png": DocumentType.IMAGE_PNG,
    "image/jpeg": DocumentType.IMAGE_JPG,
    "image/gif": DocumentType.IMAGE_GIF,
    "image/tiff": DocumentType.IMAGE_TIFF,
    "image/bmp": DocumentType.IMAGE_BMP,
    "image/webp": DocumentType.IMAGE_WEBP,
}


class DocumentTypeDetector:
    """
    Detect document types from files and URLs.

    Uses multiple strategies to maximize detection accuracy:
    - Extension-based (fast, works for most cases)
    - Magic byte detection (reliable for file content)
    - URL parsing (for web resources)
    - MIME type from HTTP headers (when available)
    """

    def __init__(self) -> None:
        # Initialize mimetypes with common types
        mimetypes.init()
        for ext, (doc_type, mime) in EXTENSION_MAP.items():
            if not mimetypes.guess_type(f"file{ext}")[0]:
                mimetypes.add_type(mime, ext)

    def detect_from_path(self, file_path: Path) -> DetectionResult:
        """
        Detect document type from a local file path.

        Tries magic byte detection first, falls back to extension.

        Args:
            file_path: Path to the file

        Returns:
            DetectionResult with detected type and confidence
        """
        source = str(file_path)

        # Try magic byte detection first if file exists
        if file_path.exists():
            magic_result = self._detect_from_magic(file_path)
            if magic_result and magic_result.document_type != DocumentType.UNKNOWN:
                return magic_result

        # Fall back to extension-based detection
        return self._detect_from_extension(file_path.suffix.lower(), source)

    def detect_from_url(
        self,
        url: str,
        content_type: Optional[str] = None,
    ) -> DetectionResult:
        """
        Detect document type from URL.

        Rule #4: Reduced from 66 → 40 lines via helper extraction
        """
        parsed = urlparse(url)
        path = unquote(parsed.path)
        if content_type:
            ct_result = self._detect_from_content_type(content_type, url)
            if ct_result.document_type != DocumentType.UNKNOWN:
                return ct_result
        pattern_result = self._detect_from_url_patterns(url)
        if pattern_result:
            return pattern_result
        if path:
            service_result = self._detect_from_url_services(parsed, path, url)
            if service_result:
                return service_result

        # Unknown type
        return DetectionResult(
            document_type=DocumentType.UNKNOWN,
            extension="",
            mime_type="application/octet-stream",
            confidence=0.0,
            detection_method="url",
            original_source=url,
        )

    def _detect_from_url_services(
        self, parsed: Any, path: str, url: str
    ) -> Optional[DetectionResult]:
        """
        Detect from URL service (Google Docs, Dropbox, etc).

        Rule #4: Extracted to reduce detect_from_url() size
        """
        clean_path = path.split("?")[0]

        # Handle Google Docs/Drive URLs
        if "docs.google.com" in parsed.netloc:
            return self._detect_google_docs_type(url)

        # Handle Dropbox URLs
        if "dropbox.com" in parsed.netloc:
            return self._detect_dropbox_type(url)

        # Get extension from path
        path_obj = Path(clean_path)
        if path_obj.suffix:
            ext_result = self._detect_from_extension(path_obj.suffix.lower(), url)
            # URL extension is less reliable
            ext_result.confidence = 0.7
            ext_result.detection_method = "url"
            return ext_result

        return None

    def detect_from_bytes(
        self,
        data: bytes,
        source: str = "<bytes>",
    ) -> DetectionResult:
        """
        Detect document type from raw bytes.

        Rule #4: Reduced from 66 → 32 lines via helper extraction
        """
        magic_result = self._check_magic_signatures(data, source)
        if magic_result:
            return magic_result
        text_result = self._detect_text_based_format(data, source)
        if text_result:
            return text_result

        return DetectionResult(
            document_type=DocumentType.UNKNOWN,
            extension="",
            mime_type="application/octet-stream",
            confidence=0.0,
            detection_method="magic",
            original_source=source,
        )

    def _check_magic_signatures(
        self, data: bytes, source: str
    ) -> Optional[DetectionResult]:
        """
        Check binary magic signatures.

        Rule #1: Reduced nesting via helper extraction
        Rule #4: Extracted to reduce detect_from_bytes() size
        """
        for signature, (doc_type, mime_type) in MAGIC_SIGNATURES.items():
            if data.startswith(signature):
                return self._create_magic_result(
                    data, source, signature, doc_type, mime_type
                )
        return None

    def _create_magic_result(
        self,
        data: bytes,
        source: str,
        signature: bytes,
        doc_type: DocumentType,
        mime_type: str,
    ) -> Optional[DetectionResult]:
        """Create detection result for magic signature match.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        # Special handling for ZIP-based formats
        if doc_type == DocumentType.UNKNOWN and signature == b"PK\x03\x04":
            zip_result = self._detect_zip_based_format(data, source)
            if zip_result:
                return zip_result

        return DetectionResult(
            document_type=doc_type,
            extension=f".{doc_type.value}" if doc_type != DocumentType.UNKNOWN else "",
            mime_type=mime_type,
            confidence=0.95,
            detection_method="magic",
            original_source=source,
        )

    def _detect_text_based_format(
        self, data: bytes, source: str
    ) -> Optional[DetectionResult]:
        """
        Detect HTML/XML from text content.

        Rule #4: Extracted to reduce detect_from_bytes() size
        """
        try:
            text_start = data[:1000].decode("utf-8", errors="ignore").strip().lower()
            if text_start.startswith("<!doctype html") or text_start.startswith(
                "<html"
            ):
                return DetectionResult(
                    document_type=DocumentType.HTML,
                    extension=".html",
                    mime_type="text/html",
                    confidence=0.9,
                    detection_method="magic",
                    original_source=source,
                )
            if text_start.startswith("<?xml"):
                return DetectionResult(
                    document_type=DocumentType.XML,
                    extension=".xml",
                    mime_type="application/xml",
                    confidence=0.9,
                    detection_method="magic",
                    original_source=source,
                )
        except Exception as e:
            logger.debug(f"Failed to detect file type using magic bytes: {e}")
        return None

    def _detect_from_magic(self, file_path: Path) -> Optional[DetectionResult]:
        """Detect from file magic bytes."""
        try:
            with open(file_path, "rb") as f:
                header = f.read(8192)  # Read first 8KB
            return self.detect_from_bytes(header, str(file_path))
        except (IOError, OSError):
            return None

    def _detect_from_extension(
        self,
        extension: str,
        source: str,
    ) -> DetectionResult:
        """Detect from file extension."""
        if not extension.startswith("."):
            extension = f".{extension}"

        if extension in EXTENSION_MAP:
            doc_type, mime_type = EXTENSION_MAP[extension]
            return DetectionResult(
                document_type=doc_type,
                extension=extension,
                mime_type=mime_type,
                confidence=0.8,
                detection_method="extension",
                original_source=source,
            )

        # Try mimetypes module
        guessed_mime, _ = mimetypes.guess_type(f"file{extension}")
        if guessed_mime and guessed_mime in MIME_TYPE_MAP:
            return DetectionResult(
                document_type=MIME_TYPE_MAP[guessed_mime],
                extension=extension,
                mime_type=guessed_mime,
                confidence=0.6,
                detection_method="extension",
                original_source=source,
            )

        return DetectionResult(
            document_type=DocumentType.UNKNOWN,
            extension=extension,
            mime_type="application/octet-stream",
            confidence=0.0,
            detection_method="extension",
            original_source=source,
        )

    def _detect_from_content_type(
        self,
        content_type: str,
        source: str,
    ) -> DetectionResult:
        """Detect from HTTP Content-Type header."""
        # Parse content type (remove charset and other params)
        mime_type = content_type.split(";")[0].strip().lower()

        if mime_type in MIME_TYPE_MAP:
            doc_type = MIME_TYPE_MAP[mime_type]
            # Get standard extension for this type
            ext = mimetypes.guess_extension(mime_type) or f".{doc_type.value}"

            return DetectionResult(
                document_type=doc_type,
                extension=ext,
                mime_type=mime_type,
                confidence=0.9,
                detection_method="content_type",
                original_source=source,
            )

        return DetectionResult(
            document_type=DocumentType.UNKNOWN,
            extension="",
            mime_type=mime_type,
            confidence=0.0,
            detection_method="content_type",
            original_source=source,
        )

    def _detect_zip_based_format(
        self,
        data: bytes,
        source: str,
    ) -> Optional[DetectionResult]:
        """Detect ZIP-based formats by inspecting archive contents.

        Rule #1: Reduced nesting via helper extraction
        """
        try:
            import zipfile
            import io

            with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
                names = zf.namelist()
                return self._check_zip_markers(names, source)

        except Exception as e:
            logger.debug(f"Failed to detect file type from magic library: {e}")

        return None

    def _check_zip_markers(
        self, filenames: list[str], source: str
    ) -> Optional[DetectionResult]:
        """Check ZIP filenames for internal format markers.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        for marker, (doc_type, mime_type) in ZIP_INTERNAL_MARKERS.items():
            if marker in filenames:
                ext = f".{doc_type.value}"
                return DetectionResult(
                    document_type=doc_type,
                    extension=ext,
                    mime_type=mime_type,
                    confidence=0.95,
                    detection_method="magic",
                    original_source=source,
                )
        return None

    def _detect_google_docs_type(self, url: str) -> DetectionResult:
        """
        Detect Google Docs document type from URL.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        """
        GOOGLE_DOCS_PATTERNS = {
            "/document/": (
                DocumentType.DOCX,
                ".docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            "/spreadsheets/": (
                DocumentType.XLSX,
                ".xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
            "/presentation/": (
                DocumentType.PPTX,
                ".pptx",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ),
        }
        for pattern, (doc_type, ext, mime) in GOOGLE_DOCS_PATTERNS.items():
            if pattern in url:
                return DetectionResult(
                    document_type=doc_type,
                    extension=ext,
                    mime_type=mime,
                    confidence=0.8,
                    detection_method="url",
                    original_source=url,
                )

        # Default: HTML
        return DetectionResult(
            document_type=DocumentType.HTML,
            extension=".html",
            mime_type="text/html",
            confidence=0.5,
            detection_method="url",
            original_source=url,
        )

    def _detect_dropbox_type(self, url: str) -> DetectionResult:
        """Detect Dropbox document type from URL."""
        # Dropbox URLs often have the extension in the path
        parsed = urlparse(url)
        path = unquote(parsed.path)

        for ext, (doc_type, mime_type) in EXTENSION_MAP.items():
            if path.lower().endswith(ext):
                return DetectionResult(
                    document_type=doc_type,
                    extension=ext,
                    mime_type=mime_type,
                    confidence=0.75,
                    detection_method="url",
                    original_source=url,
                )

        return DetectionResult(
            document_type=DocumentType.UNKNOWN,
            extension="",
            mime_type="application/octet-stream",
            confidence=0.0,
            detection_method="url",
            original_source=url,
        )

    def _detect_from_url_patterns(self, url: str) -> Optional[DetectionResult]:
        """
        Detect from common URL patterns.

        Rule #1: Dictionary dispatch eliminates if/elif chain
        """
        url_lower = url.lower()
        # Check ArXiv first (has multiple subtypes)
        if "arxiv.org" in url_lower:
            return self._detect_arxiv_type(url_lower, url)
        DOMAIN_PATTERNS = {
            "wikipedia.org": (DocumentType.HTML, ".html", "text/html", 0.9),
        }

        for domain, (doc_type, ext, mime, conf) in DOMAIN_PATTERNS.items():
            if domain in url_lower:
                return DetectionResult(
                    document_type=doc_type,
                    extension=ext,
                    mime_type=mime,
                    confidence=conf,
                    detection_method="url",
                    original_source=url,
                )

        # GitHub raw files - special handling
        if "raw.githubusercontent.com" in url_lower:
            path = urlparse(url).path
            path_obj = Path(path)
            if path_obj.suffix:
                return self._detect_from_extension(path_obj.suffix, url)

        return None

    def _detect_arxiv_type(self, url_lower: str, url: str) -> DetectionResult:
        """
        Detect ArXiv document type.

        Rule #1: Dictionary dispatch for ArXiv patterns
        Rule #4: Extracted to reduce _detect_from_url_patterns() size
        """
        ARXIV_PATTERNS = {
            "/pdf/": (DocumentType.PDF, ".pdf", "application/pdf", 0.85),
            ".pdf": (DocumentType.PDF, ".pdf", "application/pdf", 0.85),
            "/abs/": (DocumentType.HTML, ".html", "text/html", 0.8),
        }

        for pattern, (doc_type, ext, mime, conf) in ARXIV_PATTERNS.items():
            if pattern in url_lower:
                return DetectionResult(
                    document_type=doc_type,
                    extension=ext,
                    mime_type=mime,
                    confidence=conf,
                    detection_method="url",
                    original_source=url,
                )

        # Default to HTML for unknown ArXiv pages
        return DetectionResult(
            document_type=DocumentType.HTML,
            extension=".html",
            mime_type="text/html",
            confidence=0.6,
            detection_method="url",
            original_source=url,
        )


# Convenience functions
_detector = DocumentTypeDetector()


def detect_type(source: str, content_type: Optional[str] = None) -> DetectionResult:
    """
    Auto-detect document type from file path or URL.

    Args:
        source: File path or URL string
        content_type: Optional Content-Type header for URLs

    Returns:
        DetectionResult with detected type
    """
    # Check if it's a URL
    if source.startswith(("http://", "https://", "ftp://")):
        return _detector.detect_from_url(source, content_type)

    # Treat as file path
    return _detector.detect_from_path(Path(source))


# Aliases for backwards compatibility
ContentType = DocumentType
detect_content_type = detect_type
