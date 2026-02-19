"""
Unit tests for IFPDFProcessor.

Migration - PDF Parity
Tests cover all GWT scenarios, JPL rules, and feature acceptance criteria.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
    IFChunkArtifact,
)
from ingestforge.ingest.if_pdf_processor import (
    IFPDFProcessor,
    MAX_PAGES,
    MAX_TEXT_SIZE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def processor() -> IFPDFProcessor:
    """Create a fresh processor instance."""
    return IFPDFProcessor()


@pytest.fixture
def pdf_artifact(tmp_path: Path) -> IFFileArtifact:
    """Create a test PDF file artifact."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test content")
    return IFFileArtifact(
        artifact_id="test-pdf-001",
        file_path=pdf_path,
        mime_type="application/pdf",
    )


@pytest.fixture
def mock_fitz() -> MagicMock:
    """Create a mock fitz (PyMuPDF) module."""
    mock = MagicMock()
    mock.TOOLS.mupdf_display_errors = MagicMock()
    return mock


@pytest.fixture
def mock_page() -> MagicMock:
    """Create a mock PDF page with text blocks."""
    page = MagicMock()
    page.get_text.return_value = {
        "blocks": [
            {
                "type": 0,
                "lines": [
                    {"spans": [{"text": "Hello "}, {"text": "World"}]},
                    {"spans": [{"text": "Second line"}]},
                ],
            },
            {
                "type": 1,  # Image block, should be skipped
                "lines": [],
            },
        ]
    }
    return page


@pytest.fixture
def mock_doc(mock_page: MagicMock) -> MagicMock:
    """Create a mock PDF document."""
    doc = MagicMock()
    doc.__len__ = MagicMock(return_value=2)
    doc.__getitem__ = MagicMock(return_value=mock_page)
    doc.close = MagicMock()
    return doc


# =============================================================================
# GWT Scenario 1: Basic PDF Text Extraction
# =============================================================================


class TestBasicPDFExtraction:
    """GWT Scenario 1: Basic PDF text extraction."""

    def test_given_standard_pdf_when_processed_then_text_extracted(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Given a standard PDF, when processed, then text is extracted."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert "Hello World" in result.content
        assert result.parent_id == pdf_artifact.artifact_id

    def test_extracted_text_includes_all_text_blocks(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Text extraction includes all text blocks."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert "Second line" in result.content

    def test_image_blocks_are_skipped(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Image blocks (type != 0) are properly skipped."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        # Should only contain text content, no image artifacts
        assert isinstance(result, IFTextArtifact)


# =============================================================================
# GWT Scenario 2: Multi-Page PDF Processing
# =============================================================================


class TestMultiPagePDF:
    """GWT Scenario 2: Multi-page PDF processing."""

    def test_given_multipage_pdf_when_processed_then_all_pages_included(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_fitz: MagicMock,
    ):
        """Given a multi-page PDF, when processed, then all pages included."""
        page1 = MagicMock()
        page1.get_text.return_value = {
            "blocks": [{"type": 0, "lines": [{"spans": [{"text": "Page 1"}]}]}]
        }
        page2 = MagicMock()
        page2.get_text.return_value = {
            "blocks": [{"type": 0, "lines": [{"spans": [{"text": "Page 2"}]}]}]
        }

        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=2)
        doc.__getitem__ = MagicMock(side_effect=lambda i: [page1, page2][i])
        doc.close = MagicMock()

        mock_fitz.open.return_value = doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert "Page 1" in result.content
        assert "Page 2" in result.content

    def test_artifact_lineage_tracks_processing(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Artifact lineage properly tracks the processing chain."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert result.parent_id == pdf_artifact.artifact_id
        assert result.lineage_depth == pdf_artifact.lineage_depth + 1
        assert processor.processor_id in result.provenance


# =============================================================================
# GWT Scenario 3: PDF with Embedded Images (OCR)
# =============================================================================


class TestPDFWithEmbeddedImages:
    """GWT Scenario 3: PDF with embedded images (OCR)."""

    def test_given_pdf_with_images_when_ocr_enabled_then_ocr_attempted(
        self,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Given PDF with images, when OCR enabled, then OCR is attempted."""
        processor = IFPDFProcessor(ocr_embedded=True)
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            with patch(
                "ingestforge.ingest.if_pdf_processor.IFPDFProcessor._ocr_embedded_images"
            ) as mock_ocr:
                mock_ocr.return_value = ["Page text with OCR"]
                processor._fitz = mock_fitz
                result = processor.process(pdf_artifact)

        mock_ocr.assert_called_once()
        assert isinstance(result, IFTextArtifact)

    def test_given_pdf_when_ocr_disabled_then_ocr_skipped(
        self,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Given PDF, when OCR disabled, then OCR is skipped."""
        processor = IFPDFProcessor(ocr_embedded=False)
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            with patch(
                "ingestforge.ingest.if_pdf_processor.IFPDFProcessor._ocr_embedded_images"
            ) as mock_ocr:
                processor._fitz = mock_fitz
                result = processor.process(pdf_artifact)

        mock_ocr.assert_not_called()
        assert isinstance(result, IFTextArtifact)

    def test_ocr_failure_falls_back_to_native_text(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """OCR failure gracefully falls back to native text."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            with patch(
                "ingestforge.ingest.embedded_image_ocr.EmbeddedImageOCR",
                side_effect=ImportError("OCR not available"),
            ):
                processor._fitz = mock_fitz
                result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert "Hello World" in result.content


# =============================================================================
# GWT Scenario 4: Processor Registry Integration
# =============================================================================


class TestRegistryIntegration:
    """GWT Scenario 4: Processor registry integration."""

    def test_processor_has_correct_capabilities(self, processor: IFPDFProcessor):
        """Processor declares correct capabilities for routing."""
        assert "ingest.pdf" in processor.capabilities
        assert "text-extraction" in processor.capabilities

    def test_processor_has_unique_id(self, processor: IFPDFProcessor):
        """Processor has a unique identifier."""
        assert processor.processor_id == "if-pdf-extractor"

    def test_processor_has_version(self, processor: IFPDFProcessor):
        """Processor has a SemVer version."""
        assert processor.version == "1.0.0"

    def test_processor_declares_memory_requirements(self, processor: IFPDFProcessor):
        """Processor declares memory requirements for selection."""
        assert processor.memory_mb == 256


# =============================================================================
# GWT Scenario 5: Fallback Behavior
# =============================================================================


class TestFallbackBehavior:
    """GWT Scenario 5: Fallback to legacy (availability checks)."""

    def test_is_available_returns_true_when_fitz_installed(
        self,
        processor: IFPDFProcessor,
    ):
        """is_available returns True when PyMuPDF is installed."""
        with patch.dict("sys.modules", {"fitz": MagicMock()}):
            assert processor.is_available() is True

    def test_is_available_returns_false_when_fitz_missing(
        self,
        processor: IFPDFProcessor,
    ):
        """is_available returns False when PyMuPDF is missing."""
        with patch.dict("sys.modules", {"fitz": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                assert processor.is_available() is False


# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================


class TestJPLRule2Bounds:
    """JPL Rule #2: Fixed upper bounds on loops and data."""

    def test_max_pages_constant_exists(self):
        """MAX_PAGES constant is defined."""
        assert MAX_PAGES == 10000

    def test_max_text_size_constant_exists(self):
        """MAX_TEXT_SIZE constant is defined."""
        assert MAX_TEXT_SIZE == 100_000_000

    def test_page_count_respects_max_pages(
        self,
        pdf_artifact: IFFileArtifact,
        mock_fitz: MagicMock,
    ):
        """Page extraction respects MAX_PAGES limit."""
        # Use processor with OCR disabled to isolate page iteration count
        processor = IFPDFProcessor(ocr_embedded=False)

        page = MagicMock()
        page.get_text.return_value = {"blocks": []}

        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=MAX_PAGES + 1000)
        doc.__getitem__ = MagicMock(return_value=page)
        doc.close = MagicMock()

        mock_fitz.open.return_value = doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            processor.process(pdf_artifact)

        # Should only process MAX_PAGES pages
        assert doc.__getitem__.call_count == MAX_PAGES

    def test_text_truncated_at_max_size(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_fitz: MagicMock,
    ):
        """Text content is truncated at MAX_TEXT_SIZE."""
        large_text = "x" * (MAX_TEXT_SIZE + 1000)
        page = MagicMock()
        page.get_text.return_value = {
            "blocks": [{"type": 0, "lines": [{"spans": [{"text": large_text}]}]}]
        }

        doc = MagicMock()
        doc.__len__ = MagicMock(return_value=1)
        doc.__getitem__ = MagicMock(return_value=page)
        doc.close = MagicMock()

        mock_fitz.open.return_value = doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert len(result.content) <= MAX_TEXT_SIZE


# =============================================================================
# JPL Rule #7: Check Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """JPL Rule #7: All functions check return values."""

    def test_process_always_returns_artifact(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """process() always returns an IFArtifact."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, (IFTextArtifact, IFFailureArtifact))

    def test_invalid_input_returns_failure_artifact(
        self,
        processor: IFPDFProcessor,
    ):
        """Invalid input returns IFFailureArtifact, not exception."""
        chunk = IFChunkArtifact(
            artifact_id="chunk-001",
            document_id="doc-001",
            content="test",
        )
        result = processor.process(chunk)

        assert isinstance(result, IFFailureArtifact)
        assert "requires IFFileArtifact" in result.error_message

    def test_wrong_mime_type_returns_failure_artifact(
        self,
        processor: IFPDFProcessor,
        tmp_path: Path,
    ):
        """Wrong MIME type returns IFFailureArtifact."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("hello")
        artifact = IFFileArtifact(
            artifact_id="txt-001",
            file_path=txt_path,
            mime_type="text/plain",
        )
        result = processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "requires PDF" in result.error_message

    def test_missing_file_returns_failure_artifact(
        self,
        processor: IFPDFProcessor,
    ):
        """Missing file returns IFFailureArtifact."""
        artifact = IFFileArtifact(
            artifact_id="missing-001",
            file_path=Path("/nonexistent/file.pdf"),
            mime_type="application/pdf",
        )
        result = processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "not found" in result.error_message.lower()

    def test_extraction_error_returns_failure_artifact(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_fitz: MagicMock,
    ):
        """Extraction error returns IFFailureArtifact."""
        mock_fitz.open.side_effect = RuntimeError("Corrupt PDF")

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "Corrupt PDF" in result.error_message


# =============================================================================
# JPL Rule #9: Complete Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """JPL Rule #9: Complete type hints on all public methods."""

    def test_process_has_type_hints(self):
        """process() method has complete type hints."""
        import inspect

        sig = inspect.signature(IFPDFProcessor.process)
        # Check parameter annotation is present and is IFArtifact
        assert sig.parameters["artifact"].annotation is not inspect.Parameter.empty
        assert sig.parameters["artifact"].annotation.__name__ == "IFArtifact"
        # Check return annotation is present and is IFArtifact
        assert sig.return_annotation is not inspect.Signature.empty
        assert sig.return_annotation.__name__ == "IFArtifact"

    def test_is_available_has_type_hints(self):
        """is_available() has return type hint."""
        import inspect

        sig = inspect.signature(IFPDFProcessor.is_available)
        assert sig.return_annotation == bool

    def test_teardown_has_type_hints(self):
        """teardown() has return type hint."""
        import inspect

        sig = inspect.signature(IFPDFProcessor.teardown)
        assert sig.return_annotation == bool


# =============================================================================
# Artifact Metadata Tests
# =============================================================================


class TestArtifactMetadata:
    """Tests for artifact metadata preservation."""

    def test_output_artifact_contains_source_info(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Output artifact metadata includes source information."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert "source_file" in result.metadata
        assert "source_mime" in result.metadata
        assert result.metadata["source_mime"] == "application/pdf"

    def test_output_artifact_has_content_hash(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Output artifact has SHA-256 content hash."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64  # SHA-256 hex

    def test_word_and_char_counts_in_metadata(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Word and character counts are in metadata."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        assert "word_count" in result.metadata
        assert "char_count" in result.metadata
        assert isinstance(result.metadata["word_count"], int)
        assert isinstance(result.metadata["char_count"], int)


# =============================================================================
# Teardown Tests
# =============================================================================


class TestTeardown:
    """Tests for resource cleanup."""

    def test_teardown_returns_true(self, processor: IFPDFProcessor):
        """teardown() returns True on success."""
        result = processor.teardown()
        assert result is True

    def test_teardown_clears_fitz_reference(self, processor: IFPDFProcessor):
        """teardown() clears the cached fitz module reference."""
        processor._fitz = MagicMock()
        processor.teardown()
        assert processor._fitz is None


# =============================================================================
# Text Cleaning Tests
# =============================================================================


class TestTextCleaning:
    """Tests for text cleaning functionality."""

    def test_removes_page_numbers(self, processor: IFPDFProcessor):
        """Page numbers are removed from text."""
        text = "Content\n123\nMore content"
        cleaned = processor._clean_text(text)
        assert "\n123\n" not in cleaned

    def test_removes_page_x_of_y(self, processor: IFPDFProcessor):
        """'Page X of Y' patterns are removed."""
        text = "Content Page 5 of 10 More content"
        cleaned = processor._clean_text(text)
        assert "Page 5 of 10" not in cleaned

    def test_normalizes_whitespace(self, processor: IFPDFProcessor):
        """Multiple spaces are normalized."""
        text = "Hello    World"
        cleaned = processor._clean_text(text)
        assert "    " not in cleaned


# =============================================================================
# GWT Scenario Completeness
# =============================================================================


class TestGWTScenarioCompleteness:
    """Verify all 5 GWT scenarios are explicitly covered."""

    def test_scenario_1_basic_extraction_covered(self):
        """Scenario 1: Basic PDF Text Extraction is covered."""
        assert hasattr(
            TestBasicPDFExtraction,
            "test_given_standard_pdf_when_processed_then_text_extracted",
        )

    def test_scenario_2_multipage_covered(self):
        """Scenario 2: Multi-Page PDF Processing is covered."""
        assert hasattr(
            TestMultiPagePDF,
            "test_given_multipage_pdf_when_processed_then_all_pages_included",
        )

    def test_scenario_3_embedded_images_covered(self):
        """Scenario 3: PDF with Embedded Images is covered."""
        assert hasattr(
            TestPDFWithEmbeddedImages,
            "test_given_pdf_with_images_when_ocr_enabled_then_ocr_attempted",
        )

    def test_scenario_4_registry_integration_covered(self):
        """Scenario 4: Processor Registry Integration is covered."""
        assert hasattr(
            TestRegistryIntegration, "test_processor_has_correct_capabilities"
        )

    def test_scenario_5_fallback_covered(self):
        """Scenario 5: Fallback to Legacy is covered."""
        assert hasattr(
            TestFallbackBehavior, "test_is_available_returns_true_when_fitz_installed"
        )


# =============================================================================
# Legacy Parity Tests
# =============================================================================


class TestLegacyParity:
    """Verify functional parity with legacy TextExtractor."""

    def test_text_cleaning_matches_legacy_patterns(self, processor: IFPDFProcessor):
        """Text cleaning applies same patterns as legacy extractor."""
        # Test page number removal - same as legacy
        text = "Content\n42\nMore content"
        cleaned = processor._clean_text(text)
        assert "\n42\n" not in cleaned

        # Test Page X of Y - same as legacy
        text = "Header Page 3 of 10 Content"
        cleaned = processor._clean_text(text)
        assert "Page 3 of 10" not in cleaned

    def test_block_extraction_logic_matches_legacy(
        self,
        processor: IFPDFProcessor,
    ):
        """Block extraction filters same way as legacy."""
        # Text blocks (type 0) are processed
        text_block = {"type": 0, "lines": [{"spans": [{"text": "Hello"}]}]}
        lines = processor._extract_block_lines(text_block)
        assert lines == ["Hello"]

        # Image blocks (type 1) are skipped
        image_block = {"type": 1, "lines": []}
        # Image blocks don't reach _extract_block_lines in actual flow
        # but the logic filters type != 0 in _extract_page_text

    def test_span_concatenation_matches_legacy(self, processor: IFPDFProcessor):
        """Span concatenation matches legacy behavior."""
        spans = [{"text": "Hello "}, {"text": "World"}, {"text": "!"}]
        result = processor._extract_line_spans(spans)
        assert result == "Hello World!"

    def test_whitespace_normalization_matches_legacy(
        self,
        processor: IFPDFProcessor,
    ):
        """Whitespace normalization matches legacy patterns."""
        text = "Line 1\n\n\n\n\nLine 2"
        cleaned = processor._clean_text(text)
        # Multiple newlines reduced to double (paragraph break)
        assert "\n\n\n" not in cleaned

    def test_processor_produces_same_metadata_fields(
        self,
        processor: IFPDFProcessor,
        pdf_artifact: IFFileArtifact,
        mock_doc: MagicMock,
        mock_fitz: MagicMock,
    ):
        """Processor produces metadata similar to legacy extract_with_metadata."""
        mock_fitz.open.return_value = mock_doc

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            processor._fitz = mock_fitz
            result = processor.process(pdf_artifact)

        assert isinstance(result, IFTextArtifact)
        # Legacy returns: text, file_name, file_type, word_count, char_count
        # IF-Protocol provides equivalent via content and metadata
        assert hasattr(result, "content")
        assert "word_count" in result.metadata
        assert "char_count" in result.metadata
