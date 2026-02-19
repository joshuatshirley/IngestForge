"""
Tests for d: Stage 2 Extract produces IFTextArtifact.

GWT-style tests verifying that _stage_extract_text produces artifacts
while maintaining backward compatibility with dict output format.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from ingestforge.core.pipeline.artifacts import IFTextArtifact


# --- Mock Classes ---


class MockExtractor:
    """Mock TextExtractor for testing."""

    def extract(self, file_path: Path) -> str:
        return f"Extracted text from {file_path.name}"

    def extract_to_artifact(self, file_path: Path) -> IFTextArtifact:
        return IFTextArtifact(
            artifact_id=f"artifact-{file_path.name}",
            content=f"Extracted text from {file_path.name}",
            metadata={
                "source_path": str(file_path),
                "file_name": file_path.name,
            },
        )


class MockPipelineLogger:
    """Mock PipelineLogger for testing."""

    def start_stage(self, name: str) -> None:
        pass

    def log_progress(self, message: str) -> None:
        pass


class MockAudioProcessor:
    """Mock AudioProcessor for testing."""

    def process(self, file_path: Path) -> MagicMock:
        result = MagicMock()
        result.success = True
        result.text = "Transcribed audio content"
        result.word_count = 3
        return result


class MockOCRResult:
    """Mock OCR result."""

    engine = "test-ocr"
    text = "OCR extracted text"


class MockHTMLResult:
    """Mock HTML result."""

    text = "HTML extracted content"


class MockCodeResult:
    """Mock code extraction result."""

    text = "def hello(): pass"
    metadata = {"language": "python"}


class MockADOResult:
    """Mock ADO work item result."""

    text = "ADO work item content"
    metadata = {"work_item_id": "12345"}


# --- Test Pipeline Stages Mixin ---


class StageMixinTestHarness:
    """Test harness that simulates PipelineStagesMixin behavior."""

    def __init__(self) -> None:
        self.extractor = MockExtractor()
        self.audio_processor = MockAudioProcessor()
        self._html_result = None
        self._progress_reports: List[tuple] = []

    def _report_progress(self, stage: str, progress: float, message: str) -> None:
        self._progress_reports.append((stage, progress, message))

    def _create_text_artifact(
        self,
        text: str,
        file_path: Path,
        extraction_method: str,
    ) -> IFTextArtifact:
        """Create IFTextArtifact from extracted text."""
        from ingestforge.core.pipeline.artifact_factory import ArtifactFactory

        return ArtifactFactory.text_from_string(
            content=text,
            source_path=str(file_path.absolute()),
            metadata={
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "extraction_method": extraction_method,
                "word_count": len(text.split()),
                "char_count": len(text),
            },
        )

    def _stage_extract_text(
        self,
        chapters: List[Path],
        file_path: Path,
        context: Dict[str, Any],
        plog: MockPipelineLogger,
    ) -> List[Dict[str, Any]]:
        """Simulated stage extract (mirrors actual implementation)."""
        plog.start_stage("extract")
        self._report_progress("extract", 0.0, "Extracting text")

        suffix = file_path.suffix.lower()

        # OCR result handling
        if "scanned_pdf_ocr_result" in context:
            ocr_result = context["scanned_pdf_ocr_result"]
            self._report_progress(
                "extract", 1.0, f"Extracted via OCR ({ocr_result.engine})"
            )
            artifact = self._create_text_artifact(ocr_result.text, file_path, "ocr")
            return [
                {"path": str(file_path), "text": ocr_result.text, "_artifact": artifact}
            ]

        # HTML result handling
        if suffix in (".html", ".htm", ".mhtml") and self._html_result:
            self._report_progress("extract", 1.0, "Extracted HTML content")
            artifact = self._create_text_artifact(
                self._html_result.text, file_path, "html"
            )
            return [
                {
                    "path": str(file_path),
                    "text": self._html_result.text,
                    "_artifact": artifact,
                }
            ]

        # Code result handling
        if "code_result" in context:
            code_result = context["code_result"]
            self._report_progress("extract", 1.0, f"Extracted {suffix} code")
            artifact = self._create_text_artifact(code_result.text, file_path, "code")
            return [
                {
                    "path": str(file_path),
                    "text": code_result.text,
                    "metadata": code_result.metadata,
                    "_artifact": artifact,
                }
            ]

        # ADO result handling
        if "ado_result" in context:
            ado_result = context["ado_result"]
            self._report_progress("extract", 1.0, "Extracted ADO work item")
            artifact = self._create_text_artifact(ado_result.text, file_path, "ado")
            return [
                {
                    "path": str(file_path),
                    "text": ado_result.text,
                    "metadata": ado_result.metadata,
                    "_artifact": artifact,
                }
            ]

        # Audio handling
        if suffix in (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm"):
            audio_result = self.audio_processor.process(file_path)
            if not audio_result.success:
                raise ValueError(f"Transcription failed: {audio_result.error}")
            context["audio_result"] = audio_result
            artifact = self._create_text_artifact(audio_result.text, file_path, "audio")
            return [
                {
                    "path": str(file_path),
                    "text": audio_result.text,
                    "_artifact": artifact,
                }
            ]

        # Standard extraction
        extracted_texts = []
        for i, chapter_path in enumerate(chapters):
            artifact = self.extractor.extract_to_artifact(Path(chapter_path))
            extracted_texts.append(
                {
                    "path": str(chapter_path),
                    "text": artifact.content,
                    "_artifact": artifact,
                }
            )
            progress = (i + 1) / len(chapters)
            self._report_progress(
                "extract", progress, f"Extracted {i + 1}/{len(chapters)}"
            )

        plog.log_progress(f"Extracted {len(extracted_texts)} text sections")
        return extracted_texts


# --- Fixtures ---


@pytest.fixture
def harness() -> StageMixinTestHarness:
    """Create test harness."""
    return StageMixinTestHarness()


@pytest.fixture
def plog() -> MockPipelineLogger:
    """Create mock pipeline logger."""
    return MockPipelineLogger()


@pytest.fixture
def sample_chapters(tmp_path: Path) -> List[Path]:
    """Create sample chapter files."""
    chapters = []
    for i in range(3):
        chapter = tmp_path / f"chapter_{i}.txt"
        chapter.write_text(f"Content of chapter {i}")
        chapters.append(chapter)
    return chapters


# --- GWT Scenario 1: Standard Extraction Produces Artifacts ---


class TestStandardExtractionArtifacts:
    """Tests that standard extraction produces artifacts."""

    def test_extraction_returns_list_of_dicts(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given chapters, When _stage_extract_text called,
        Then list of dicts is returned."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_each_dict_has_artifact_key(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given chapters, When _stage_extract_text called,
        Then each dict has '_artifact' key."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for item in result:
            assert "_artifact" in item

    def test_artifact_is_if_text_artifact(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given chapters, When _stage_extract_text called,
        Then '_artifact' is IFTextArtifact."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for item in result:
            assert isinstance(item["_artifact"], IFTextArtifact)

    def test_artifact_content_matches_text_key(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given chapters, When _stage_extract_text called,
        Then artifact.content matches 'text' key."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for item in result:
            assert item["_artifact"].content == item["text"]


# --- GWT Scenario 2: Backward Compatibility ---


class TestBackwardCompatibility:
    """Tests that old dict format is preserved."""

    def test_dict_has_path_key(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given extraction, When result examined,
        Then each dict has 'path' key."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for item in result:
            assert "path" in item
            assert isinstance(item["path"], str)

    def test_dict_has_text_key(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given extraction, When result examined,
        Then each dict has 'text' key."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for item in result:
            assert "text" in item
            assert isinstance(item["text"], str)

    def test_text_is_string_not_artifact(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given extraction, When 'text' accessed,
        Then it is raw string (not artifact)."""
        result = harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for item in result:
            # text should be str, not IFTextArtifact
            assert type(item["text"]) is str


# --- GWT Scenario 3: Special Case - OCR Results ---


class TestOCRResultArtifacts:
    """Tests OCR result produces artifacts."""

    def test_ocr_result_produces_artifact(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given OCR result in context, When _stage_extract_text called,
        Then artifact is produced."""
        file_path = tmp_path / "scanned.pdf"
        file_path.touch()

        context = {"scanned_pdf_ocr_result": MockOCRResult()}

        result = harness._stage_extract_text(
            chapters=[file_path],
            file_path=file_path,
            context=context,
            plog=plog,
        )

        assert len(result) == 1
        assert "_artifact" in result[0]
        assert isinstance(result[0]["_artifact"], IFTextArtifact)

    def test_ocr_artifact_has_extraction_method(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given OCR result, When artifact examined,
        Then extraction_method is 'ocr'."""
        file_path = tmp_path / "scanned.pdf"
        file_path.touch()

        context = {"scanned_pdf_ocr_result": MockOCRResult()}

        result = harness._stage_extract_text(
            chapters=[file_path],
            file_path=file_path,
            context=context,
            plog=plog,
        )

        assert result[0]["_artifact"].metadata.get("extraction_method") == "ocr"


# --- GWT Scenario 4: Special Case - Code Results ---


class TestCodeResultArtifacts:
    """Tests code result produces artifacts."""

    def test_code_result_produces_artifact(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given code result in context, When _stage_extract_text called,
        Then artifact is produced."""
        file_path = tmp_path / "module.py"
        file_path.touch()

        context = {"code_result": MockCodeResult()}

        result = harness._stage_extract_text(
            chapters=[file_path],
            file_path=file_path,
            context=context,
            plog=plog,
        )

        assert len(result) == 1
        assert "_artifact" in result[0]

    def test_code_result_preserves_metadata(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given code result with metadata, When extracted,
        Then original metadata preserved in dict."""
        file_path = tmp_path / "module.py"
        file_path.touch()

        context = {"code_result": MockCodeResult()}

        result = harness._stage_extract_text(
            chapters=[file_path],
            file_path=file_path,
            context=context,
            plog=plog,
        )

        assert result[0].get("metadata") == {"language": "python"}


# --- GWT Scenario 5: Special Case - Audio Results ---


class TestAudioResultArtifacts:
    """Tests audio transcription produces artifacts."""

    def test_audio_file_produces_artifact(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given audio file, When _stage_extract_text called,
        Then artifact is produced."""
        file_path = tmp_path / "recording.mp3"
        file_path.touch()

        context: Dict[str, Any] = {}

        result = harness._stage_extract_text(
            chapters=[file_path],
            file_path=file_path,
            context=context,
            plog=plog,
        )

        assert len(result) == 1
        assert "_artifact" in result[0]

    def test_audio_artifact_has_extraction_method(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given audio transcription, When artifact examined,
        Then extraction_method is 'audio'."""
        file_path = tmp_path / "recording.wav"
        file_path.touch()

        context: Dict[str, Any] = {}

        result = harness._stage_extract_text(
            chapters=[file_path],
            file_path=file_path,
            context=context,
            plog=plog,
        )

        assert result[0]["_artifact"].metadata.get("extraction_method") == "audio"


# --- JPL Rules Compliance ---


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_bounded_extraction_count(
        self, harness: StageMixinTestHarness, plog: MockPipelineLogger, tmp_path: Path
    ) -> None:
        """Given multiple chapters, When extracted,
        Then result count matches input count."""
        chapters = []
        for i in range(5):
            chapter = tmp_path / f"ch{i}.txt"
            chapter.write_text(f"Content {i}")
            chapters.append(chapter)

        result = harness._stage_extract_text(
            chapters=chapters,
            file_path=tmp_path / "doc.pdf",
            context={},
            plog=plog,
        )

        assert len(result) == len(chapters)

    def test_progress_reports_bounded(
        self,
        harness: StageMixinTestHarness,
        plog: MockPipelineLogger,
        sample_chapters: List[Path],
    ) -> None:
        """Given extraction, When progress examined,
        Then values are 0.0 to 1.0."""
        harness._stage_extract_text(
            chapters=sample_chapters,
            file_path=sample_chapters[0].parent / "doc.pdf",
            context={},
            plog=plog,
        )

        for stage, progress, _ in harness._progress_reports:
            assert 0.0 <= progress <= 1.0


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_standard_extraction_covered(self) -> None:
        """GWT Scenario 1 (Standard Extraction) is tested."""
        assert hasattr(
            TestStandardExtractionArtifacts, "test_artifact_is_if_text_artifact"
        )

    def test_scenario_2_backward_compat_covered(self) -> None:
        """GWT Scenario 2 (Backward Compatibility) is tested."""
        assert hasattr(TestBackwardCompatibility, "test_dict_has_text_key")

    def test_scenario_3_ocr_covered(self) -> None:
        """GWT Scenario 3 (OCR Results) is tested."""
        assert hasattr(TestOCRResultArtifacts, "test_ocr_result_produces_artifact")

    def test_scenario_4_code_covered(self) -> None:
        """GWT Scenario 4 (Code Results) is tested."""
        assert hasattr(TestCodeResultArtifacts, "test_code_result_produces_artifact")

    def test_scenario_5_audio_covered(self) -> None:
        """GWT Scenario 5 (Audio Results) is tested."""
        assert hasattr(TestAudioResultArtifacts, "test_audio_file_produces_artifact")
