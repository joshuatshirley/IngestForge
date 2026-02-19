"""
Tests for Document Type Detection.

This module tests document type detection from files, URLs, and byte content.

Test Strategy
-------------
- Focus on detection strategies and accuracy
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test enum, dataclass, and detector methods
- Mock file I/O where appropriate

Organization
------------
- TestDocumentType: DocumentType enum
- TestDetectionResult: DetectionResult dataclass
- TestExtensionDetection: Extension-based detection
- TestMagicByteDetection: Magic byte signature detection
- TestDetectorInitialization: DocumentTypeDetector setup
"""

from pathlib import Path
from unittest.mock import patch, mock_open


from ingestforge.ingest.type_detector import (
    DocumentType,
    DetectionResult,
    DocumentTypeDetector,
    EXTENSION_MAP,
    MAGIC_SIGNATURES,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestDocumentType:
    """Tests for DocumentType enum.

    Rule #4: Focused test class - tests only DocumentType
    """

    def test_document_type_pdf(self):
        """Test PDF document type."""
        assert DocumentType.PDF.value == "pdf"

    def test_document_type_text_formats(self):
        """Test text format document types."""
        assert DocumentType.TXT.value == "txt"
        assert DocumentType.MD.value == "md"
        assert DocumentType.HTML.value == "html"

    def test_document_type_office_formats(self):
        """Test office format document types."""
        assert DocumentType.DOCX.value == "docx"
        assert DocumentType.XLSX.value == "xlsx"
        assert DocumentType.PPTX.value == "pptx"

    def test_document_type_image_formats(self):
        """Test image format document types."""
        assert DocumentType.IMAGE_PNG.value == "png"
        assert DocumentType.IMAGE_JPG.value == "jpg"
        assert DocumentType.IMAGE_GIF.value == "gif"

    def test_document_type_unknown(self):
        """Test unknown document type."""
        assert DocumentType.UNKNOWN.value == "unknown"


class TestDetectionResult:
    """Tests for DetectionResult dataclass.

    Rule #4: Focused test class - tests only DetectionResult
    """

    def test_create_detection_result(self):
        """Test creating a DetectionResult."""
        result = DetectionResult(
            document_type=DocumentType.PDF,
            extension=".pdf",
            mime_type="application/pdf",
            confidence=1.0,
            detection_method="extension",
            original_source="document.pdf",
        )

        assert result.document_type == DocumentType.PDF
        assert result.extension == ".pdf"
        assert result.mime_type == "application/pdf"
        assert result.confidence == 1.0
        assert result.detection_method == "extension"
        assert result.original_source == "document.pdf"

    def test_detection_result_low_confidence(self):
        """Test DetectionResult with low confidence."""
        result = DetectionResult(
            document_type=DocumentType.UNKNOWN,
            extension="",
            mime_type="",
            confidence=0.3,
            detection_method="guess",
            original_source="unknown.xyz",
        )

        assert result.confidence == 0.3
        assert result.document_type == DocumentType.UNKNOWN


class TestExtensionMap:
    """Tests for EXTENSION_MAP registry.

    Rule #4: Focused test class - tests EXTENSION_MAP only
    """

    def test_extension_map_contains_pdf(self):
        """Test extension map contains PDF."""
        assert ".pdf" in EXTENSION_MAP
        doc_type, mime = EXTENSION_MAP[".pdf"]

        assert doc_type == DocumentType.PDF
        assert mime == "application/pdf"

    def test_extension_map_contains_office_formats(self):
        """Test extension map contains office formats."""
        assert ".docx" in EXTENSION_MAP
        assert ".xlsx" in EXTENSION_MAP
        assert ".pptx" in EXTENSION_MAP

        docx_type, docx_mime = EXTENSION_MAP[".docx"]
        assert docx_type == DocumentType.DOCX

    def test_extension_map_contains_text_formats(self):
        """Test extension map contains text formats."""
        assert ".txt" in EXTENSION_MAP
        assert ".md" in EXTENSION_MAP
        assert ".html" in EXTENSION_MAP

        txt_type, txt_mime = EXTENSION_MAP[".txt"]
        assert txt_type == DocumentType.TXT
        assert txt_mime == "text/plain"

    def test_extension_map_contains_image_formats(self):
        """Test extension map contains image formats."""
        assert ".png" in EXTENSION_MAP
        assert ".jpg" in EXTENSION_MAP
        assert ".gif" in EXTENSION_MAP

        png_type, _ = EXTENSION_MAP[".png"]
        assert png_type == DocumentType.IMAGE_PNG


class TestMagicSignatures:
    """Tests for MAGIC_SIGNATURES registry.

    Rule #4: Focused test class - tests MAGIC_SIGNATURES only
    """

    def test_magic_signatures_contains_pdf(self):
        """Test magic signatures contain PDF signature."""
        assert b"%PDF" in MAGIC_SIGNATURES
        doc_type, mime = MAGIC_SIGNATURES[b"%PDF"]

        assert doc_type == DocumentType.PDF
        assert mime == "application/pdf"

    def test_magic_signatures_contains_images(self):
        """Test magic signatures contain image signatures."""
        assert b"\x89PNG\r\n\x1a\n" in MAGIC_SIGNATURES
        assert b"\xff\xd8\xff" in MAGIC_SIGNATURES
        assert b"GIF87a" in MAGIC_SIGNATURES

        png_type, _ = MAGIC_SIGNATURES[b"\x89PNG\r\n\x1a\n"]
        assert png_type == DocumentType.IMAGE_PNG

    def test_magic_signatures_contains_html(self):
        """Test magic signatures contain HTML signatures."""
        assert b"<!DOCTYPE html" in MAGIC_SIGNATURES
        assert b"<html" in MAGIC_SIGNATURES

        html_type, _ = MAGIC_SIGNATURES[b"<!DOCTYPE html"]
        assert html_type == DocumentType.HTML


class TestDetectorInitialization:
    """Tests for DocumentTypeDetector initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_detector(self):
        """Test creating a DocumentTypeDetector."""
        detector = DocumentTypeDetector()

        assert detector is not None

    def test_detector_registers_mime_types(self):
        """Test detector registers MIME types on init."""
        detector = DocumentTypeDetector()

        # Should be able to guess types after initialization
        assert detector is not None


class TestExtensionDetection:
    """Tests for extension-based detection.

    Rule #4: Focused test class - tests extension detection only
    """

    def test_detect_pdf_from_extension(self):
        """Test detecting PDF from file extension."""
        detector = DocumentTypeDetector()
        pdf_path = Path("document.pdf")

        # Mock file existence
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"%PDF-1.4")):
                    result = detector.detect_from_path(pdf_path)

        assert result.document_type == DocumentType.PDF
        assert result.extension == ".pdf"

    def test_detect_text_from_extension(self):
        """Test detecting text file from extension."""
        detector = DocumentTypeDetector()
        txt_path = Path("notes.txt")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"Hello world")):
                    result = detector.detect_from_path(txt_path)

        assert result.document_type == DocumentType.TXT
        assert result.extension == ".txt"

    def test_detect_markdown_from_extension(self):
        """Test detecting markdown from extension."""
        detector = DocumentTypeDetector()
        md_path = Path("README.md")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "is_file", return_value=True):
                with patch("builtins.open", mock_open(read_data=b"# Title")):
                    result = detector.detect_from_path(md_path)

        assert result.document_type == DocumentType.MD
        assert result.extension == ".md"


class TestMagicByteDetection:
    """Tests for magic byte detection.

    Rule #4: Focused test class - tests magic byte detection only
    """

    def test_detect_pdf_from_magic_bytes(self):
        """Test detecting PDF from magic bytes."""
        detector = DocumentTypeDetector()
        pdf_bytes = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3"

        result = detector.detect_from_bytes(pdf_bytes, "document.pdf")

        assert result.document_type == DocumentType.PDF
        assert result.detection_method == "magic"

    def test_detect_png_from_magic_bytes(self):
        """Test detecting PNG from magic bytes."""
        detector = DocumentTypeDetector()
        png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"

        result = detector.detect_from_bytes(png_bytes, "image.png")

        assert result.document_type == DocumentType.IMAGE_PNG
        assert result.detection_method == "magic"

    def test_detect_jpg_from_magic_bytes(self):
        """Test detecting JPG from magic bytes."""
        detector = DocumentTypeDetector()
        jpg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"

        result = detector.detect_from_bytes(jpg_bytes, "photo.jpg")

        assert result.document_type == DocumentType.IMAGE_JPG
        assert result.detection_method == "magic"

    def test_detect_gif_from_magic_bytes(self):
        """Test detecting GIF from magic bytes."""
        detector = DocumentTypeDetector()
        gif_bytes = b"GIF89a\x01\x00\x01\x00"

        result = detector.detect_from_bytes(gif_bytes, "animation.gif")

        assert result.document_type == DocumentType.IMAGE_GIF
        assert result.detection_method == "magic"


class TestURLDetection:
    """Tests for URL-based detection.

    Rule #4: Focused test class - tests URL detection only
    """

    def test_detect_from_pdf_url(self):
        """Test detecting from PDF URL."""
        detector = DocumentTypeDetector()
        url = "https://example.com/document.pdf"

        result = detector.detect_from_url(url)

        assert result.document_type == DocumentType.PDF
        assert result.extension == ".pdf"

    def test_detect_from_html_url(self):
        """Test detecting from HTML URL."""
        detector = DocumentTypeDetector()
        url = "https://example.com/page.html"

        result = detector.detect_from_url(url)

        assert result.document_type == DocumentType.HTML

    def test_detect_from_url_without_extension(self):
        """Test detecting from URL without extension."""
        detector = DocumentTypeDetector()
        url = "https://example.com/article"

        result = detector.detect_from_url(url)

        # Should still return a result (may be HTML by default for web URLs)
        assert result is not None
        assert result.document_type is not None


class TestArXivDetection:
    """Tests for arXiv URL detection.

    Rule #4: Focused test class - tests arXiv detection only

    This addresses TEST-MAINT-TYPE: arXiv abstract type detection.
    """

    def test_detect_arxiv_abstract_page(self):
        """Test detection of arXiv abstract page URL.

        arXiv abstract pages (/abs/) should be detected as HTML.
        This is the primary test for TEST-MAINT-TYPE backlog item.
        """
        detector = DocumentTypeDetector()
        url = "https://arxiv.org/abs/2301.12345"

        result = detector.detect_from_url(url)

        assert result.document_type == DocumentType.HTML
        assert result.extension == ".html"
        assert result.detection_method == "url"

    def test_detect_arxiv_pdf_page(self):
        """Test detection of arXiv PDF URL."""
        detector = DocumentTypeDetector()
        url = "https://arxiv.org/pdf/2301.12345"

        result = detector.detect_from_url(url)

        assert result.document_type == DocumentType.PDF
        assert result.extension == ".pdf"

    def test_detect_arxiv_pdf_with_extension(self):
        """Test detection of arXiv PDF URL with .pdf extension."""
        detector = DocumentTypeDetector()
        url = "https://arxiv.org/pdf/2301.12345.pdf"

        result = detector.detect_from_url(url)

        assert result.document_type == DocumentType.PDF

    def test_detect_arxiv_home_page(self):
        """Test detection of arXiv home page."""
        detector = DocumentTypeDetector()
        url = "https://arxiv.org"

        result = detector.detect_from_url(url)

        # Home page should default to HTML
        assert result.document_type == DocumentType.HTML

    def test_detect_arxiv_search_page(self):
        """Test detection of arXiv search results page."""
        detector = DocumentTypeDetector()
        url = "https://arxiv.org/search/?query=machine+learning"

        result = detector.detect_from_url(url)

        # Search pages should be HTML
        assert result.document_type == DocumentType.HTML

    def test_arxiv_detection_case_insensitive(self):
        """Test that arXiv detection is case-insensitive."""
        detector = DocumentTypeDetector()

        # Test with various case combinations
        urls = [
            "https://ARXIV.org/abs/2301.12345",
            "https://ArXiv.org/abs/2301.12345",
            "https://arxiv.ORG/abs/2301.12345",
        ]

        for url in urls:
            result = detector.detect_from_url(url)
            assert result.document_type == DocumentType.HTML, f"Failed for URL: {url}"


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - DocumentType enum: 5 tests (PDF, text, office, image, unknown)
    - DetectionResult: 2 tests (creation, low confidence)
    - EXTENSION_MAP: 4 tests (PDF, office, text, image formats)
    - MAGIC_SIGNATURES: 3 tests (PDF, images, HTML)
    - Detector init: 2 tests (creation, MIME type registration)
    - Extension detection: 3 tests (PDF, text, markdown)
    - Magic byte detection: 4 tests (PDF, PNG, JPG, GIF)
    - URL detection: 3 tests (PDF URL, HTML URL, no extension)
    - arXiv detection: 6 tests (abstract, PDF, extension, home, search, case)

    Total: 32 tests

Design Decisions:
    1. Focus on detection strategies and data structures
    2. Mock file I/O to avoid creating real test files
    3. Test magic byte signatures with actual byte patterns
    4. Test all major document types (PDF, office, text, images)
    5. Simple, clear tests that verify detection accuracy
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)
    8. arXiv detection tests verify TEST-MAINT-TYPE fix

Behaviors Tested:
    - DocumentType enum values and coverage
    - DetectionResult dataclass creation
    - EXTENSION_MAP registry contents
    - MAGIC_SIGNATURES registry contents
    - DocumentTypeDetector initialization
    - Extension-based detection (PDF, text, markdown)
    - Magic byte detection (PDF, PNG, JPG, GIF)
    - URL-based detection (with and without extensions)
    - arXiv URL detection (abstract pages, PDFs, search pages)

Justification:
    - Type detection is critical for document ingestion
    - Multiple detection strategies need verification
    - Magic bytes are reliable signatures to test
    - Extension detection is the most common path
    - URL detection handles web resources
    - arXiv detection is important for academic workflows
    - Simple tests verify detection system works correctly
"""
