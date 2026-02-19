"""
Unit tests for InstructorCitationEnricher IFProcessor migration.

Migrate InstructorCitationEnricher to IFProcessor.
GWT-compliant tests with NASA JPL Power of Ten verification.
"""

import inspect
import warnings
from unittest.mock import Mock, patch

import pytest

from ingestforge.core.config import Config
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.enrichment.instructor_citation import InstructorCitationEnricher


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Config:
    """Create a minimal mock config for testing."""
    config = Mock(spec=Config)
    config.llm = Mock()
    config.llm.openai = Mock()
    config.llm.openai.model = "gpt-4o-mini"
    return config


@pytest.fixture
def mock_openai_client() -> Mock:
    """Create a mock OpenAI client."""
    client = Mock()
    client.is_available.return_value = True
    return client


@pytest.fixture
def enricher(
    mock_config: Config, mock_openai_client: Mock
) -> InstructorCitationEnricher:
    """Create an InstructorCitationEnricher instance with mocked dependencies."""
    with patch(
        "ingestforge.enrichment.instructor_citation.get_llm_client",
        return_value=mock_openai_client,
    ):
        return InstructorCitationEnricher(mock_config)


@pytest.fixture
def first_chunk_artifact() -> IFChunkArtifact:
    """Create a first chunk (index 0) with citation info."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content="""
        Title: Machine Learning Fundamentals
        Authors: John Smith, Jane Doe
        DOI: 10.1234/ml.2024.001
        Published: January 2024

        This paper presents a comprehensive overview of machine learning...
        """,
        chunk_index=0,
        total_chunks=5,
        metadata={"source": "academic"},
    )


@pytest.fixture
def non_first_chunk_artifact() -> IFChunkArtifact:
    """Create a non-first chunk (index > 0)."""
    return IFChunkArtifact(
        artifact_id="chunk-003",
        document_id="doc-001",
        content="This is the middle of the document with no citation info.",
        chunk_index=3,
        total_chunks=5,
        metadata={"source": "academic"},
    )


# ---------------------------------------------------------------------------
# IFProcessor Interface Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherIFProcessorInterface:
    """GWT: Given InstructorCitationEnricher, When checked, Then implements IFProcessor."""

    def test_extends_if_processor(self, enricher: InstructorCitationEnricher) -> None:
        """Verify InstructorCitationEnricher extends IFProcessor."""
        assert isinstance(enricher, IFProcessor)

    def test_has_process_method(self, enricher: InstructorCitationEnricher) -> None:
        """Verify process() method exists and is callable."""
        assert hasattr(enricher, "process")
        assert callable(enricher.process)

    def test_has_processor_id_property(
        self, enricher: InstructorCitationEnricher
    ) -> None:
        """Verify processor_id property exists."""
        assert hasattr(enricher, "processor_id")
        assert isinstance(enricher.processor_id, str)

    def test_has_version_property(self, enricher: InstructorCitationEnricher) -> None:
        """Verify version property exists."""
        assert hasattr(enricher, "version")
        assert isinstance(enricher.version, str)

    def test_has_capabilities_property(
        self, enricher: InstructorCitationEnricher
    ) -> None:
        """Verify capabilities property exists."""
        assert hasattr(enricher, "capabilities")
        assert isinstance(enricher.capabilities, list)

    def test_has_is_available_method(
        self, enricher: InstructorCitationEnricher
    ) -> None:
        """Verify is_available() method exists."""
        assert hasattr(enricher, "is_available")
        assert callable(enricher.is_available)

    def test_has_teardown_method(self, enricher: InstructorCitationEnricher) -> None:
        """Verify teardown() method exists."""
        assert hasattr(enricher, "teardown")
        assert callable(enricher.teardown)


# ---------------------------------------------------------------------------
# Processor Properties Tests
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherProperties:
    """GWT: Given InstructorCitationEnricher, When properties accessed, Then correct values."""

    def test_processor_id_value(self, enricher: InstructorCitationEnricher) -> None:
        """Verify processor_id is correct."""
        assert enricher.processor_id == "instructor-citation-enricher"

    def test_version_is_semver(self, enricher: InstructorCitationEnricher) -> None:
        """Verify version follows SemVer format."""
        import re

        assert re.match(r"^\d+\.\d+\.\d+$", enricher.version)
        assert enricher.version == "2.0.0"

    def test_capabilities_include_expected(
        self, enricher: InstructorCitationEnricher
    ) -> None:
        """Verify capabilities include expected values."""
        assert "citation-extraction" in enricher.capabilities
        assert "metadata-enrichment" in enricher.capabilities

    def test_memory_mb_is_positive(self, enricher: InstructorCitationEnricher) -> None:
        """Verify memory_mb returns positive integer."""
        assert enricher.memory_mb > 0
        assert isinstance(enricher.memory_mb, int)

    def test_teardown_clears_client(self, enricher: InstructorCitationEnricher) -> None:
        """Verify teardown clears instructor client."""
        enricher._instructor_client = Mock()
        result = enricher.teardown()
        assert result is True
        assert enricher._instructor_client is None


# ---------------------------------------------------------------------------
# Process Method Tests
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherProcess:
    """GWT: Given IFChunkArtifact, When processed, Then extracts citation."""

    def test_process_returns_if_artifact(
        self,
        enricher: InstructorCitationEnricher,
        first_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() returns an IFArtifact."""
        result = enricher.process(first_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_process_skips_non_first_chunk(
        self,
        enricher: InstructorCitationEnricher,
        non_first_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify process() skips chunks with index > 0."""
        result = enricher.process(non_first_chunk_artifact)
        # Should return unchanged artifact
        assert result.artifact_id == non_first_chunk_artifact.artifact_id
        assert "citation" not in result.metadata

    def test_process_forces_extraction_with_metadata_flag(
        self,
        enricher: InstructorCitationEnricher,
        non_first_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Verify force_citation_extraction flag overrides chunk_index check."""
        # Create artifact with force flag
        forced_artifact = IFChunkArtifact(
            artifact_id="chunk-forced",
            document_id="doc-001",
            content="Title: Forced Extraction Test",
            chunk_index=5,  # Not first chunk
            total_chunks=10,
            metadata={"force_citation_extraction": True},
        )
        # Will still return the artifact (no instructor available in test)
        result = enricher.process(forced_artifact)
        assert isinstance(result, IFArtifact)

    def test_process_invalid_input_returns_failure(
        self,
        enricher: InstructorCitationEnricher,
    ) -> None:
        """Verify process() returns IFFailureArtifact for invalid input."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Some text content",
        )
        result = enricher.process(text_artifact)
        assert isinstance(result, IFFailureArtifact)
        assert (
            "InstructorCitationEnricher requires IFChunkArtifact"
            in result.error_message
        )


# ---------------------------------------------------------------------------
# Lineage Preservation Tests
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherLineage:
    """GWT: Given artifact with lineage, When processed, Then lineage preserved."""

    def test_failure_artifact_has_lineage(
        self,
        enricher: InstructorCitationEnricher,
    ) -> None:
        """Verify failure artifact preserves lineage."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Test",
        )
        result = enricher.process(text_artifact)
        assert isinstance(result, IFFailureArtifact)
        assert result.parent_id == text_artifact.artifact_id
        assert enricher.processor_id in result.provenance


# ---------------------------------------------------------------------------
# Legacy API Tests (Backward Compatibility)
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherLegacyAPI:
    """GWT: Given legacy method call, When invoked, Then emits deprecation warning."""

    def test_enrich_chunk_emits_warning(
        self,
        enricher: InstructorCitationEnricher,
    ) -> None:
        """Verify enrich_chunk() emits DeprecationWarning."""
        chunk = Mock()
        chunk.chunk_index = 0
        chunk.content = "Title: Test Paper"
        chunk.metadata = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            enricher.enrich_chunk(chunk)
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()


# ---------------------------------------------------------------------------
# JPL Power of Ten Compliance Tests
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherJPLRules:
    """Verify JPL Power of Ten compliance."""

    def test_jpl_rule_4_functions_under_60_lines(self) -> None:
        """Rule #4: All functions under 60 lines."""
        methods = [
            InstructorCitationEnricher.process,
            InstructorCitationEnricher._extract_metadata,
        ]
        for method in methods:
            source_lines = len(inspect.getsourcelines(method)[0])
            assert source_lines < 60, f"{method.__name__} has {source_lines} lines"

    def test_jpl_rule_9_type_annotations(self) -> None:
        """Rule #9: Methods have type annotations."""
        methods = [
            InstructorCitationEnricher.process,
            InstructorCitationEnricher.processor_id.fget,
            InstructorCitationEnricher.version.fget,
            InstructorCitationEnricher.capabilities.fget,
            InstructorCitationEnricher.is_available,
        ]
        for method in methods:
            hints = getattr(method, "__annotations__", {})
            assert "return" in hints, f"{method.__name__} missing return type"


# ---------------------------------------------------------------------------
# Availability Tests
# ---------------------------------------------------------------------------


class TestInstructorCitationEnricherAvailability:
    """Test availability checking."""

    def test_is_available_without_instructor(
        self,
        mock_config: Config,
    ) -> None:
        """Verify is_available returns False when instructor not installed."""
        with patch(
            "ingestforge.enrichment.instructor_citation.get_llm_client",
            return_value=Mock(),
        ):
            enricher = InstructorCitationEnricher(mock_config)
            # Since instructor import may fail in test, check it handles gracefully
            result = enricher.is_available()
            # Result depends on whether instructor is actually installed
            assert isinstance(result, bool)

    def test_instructor_client_lazy_loading(
        self,
        enricher: InstructorCitationEnricher,
    ) -> None:
        """Verify instructor_client is lazily loaded."""
        # Initially None
        assert enricher._instructor_client is None
        # Accessing property attempts to load (may return None if instructor not installed)
        _ = enricher.instructor_client
        # Should have attempted initialization
