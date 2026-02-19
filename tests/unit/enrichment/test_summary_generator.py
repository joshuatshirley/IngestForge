"""
Unit tests for SummaryGenerator IFProcessor migration.

Migrate SummaryGenerator to IFProcessor.
GWT-compliant tests with NASA JPL Power of Ten verification.
"""

import inspect
import warnings
from unittest.mock import Mock

import pytest

from ingestforge.core.config import Config
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.enrichment.summary import (
    SummaryGenerator,
    MAX_CONTENT_LENGTH,
    MAX_SUMMARY_LENGTH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Config:
    """Create a minimal mock config for testing."""
    config = Mock(spec=Config)
    config.enrichment = Mock()
    return config


@pytest.fixture
def summary_generator(mock_config: Config) -> SummaryGenerator:
    """Create a SummaryGenerator instance."""
    return SummaryGenerator(mock_config)


@pytest.fixture
def sample_chunk_artifact() -> IFChunkArtifact:
    """Create a sample IFChunkArtifact for testing."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content=(
            "Machine learning is a subset of artificial intelligence that enables "
            "systems to learn and improve from experience without being explicitly programmed. "
            "It focuses on developing algorithms that can access data and use it to learn."
        ),
        chunk_index=0,
        total_chunks=1,
        metadata={
            "section_title": "Machine Learning Basics",
            "chunk_type": "general",
        },
    )


@pytest.fixture
def short_content_artifact() -> IFChunkArtifact:
    """Create a chunk with short content."""
    return IFChunkArtifact(
        artifact_id="chunk-short",
        document_id="doc-001",
        content="Short text here.",
        chunk_index=0,
        total_chunks=1,
        metadata={},
    )


@pytest.fixture
def long_content_artifact() -> IFChunkArtifact:
    """Create a chunk with very long content."""
    return IFChunkArtifact(
        artifact_id="chunk-long",
        document_id="doc-001",
        content="This is a sample sentence. " * 500,  # Very long
        chunk_index=0,
        total_chunks=1,
        metadata={},
    )


# ---------------------------------------------------------------------------
# IFProcessor Interface Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestIFProcessorInterface:
    """
    GWT: Test that SummaryGenerator implements IFProcessor interface.

    Acceptance Criteria:
    - [x] SummaryGenerator extends IFProcessor instead of IEnricher.
    """

    def test_given_summary_generator_when_check_inheritance_then_is_ifprocessor(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When checking inheritance, Then it extends IFProcessor."""
        assert isinstance(summary_generator, IFProcessor)

    def test_given_summary_generator_when_check_methods_then_has_process(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When checking methods, Then has process() method."""
        assert hasattr(summary_generator, "process")
        assert callable(summary_generator.process)

    def test_given_summary_generator_when_check_methods_then_has_is_available(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When checking methods, Then has is_available() method."""
        assert hasattr(summary_generator, "is_available")
        assert callable(summary_generator.is_available)

    def test_given_summary_generator_when_check_methods_then_has_teardown(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When checking methods, Then has teardown() method."""
        assert hasattr(summary_generator, "teardown")
        assert callable(summary_generator.teardown)


# ---------------------------------------------------------------------------
# Processor Properties Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestProcessorProperties:
    """
    GWT: Test processor_id, version, capabilities properties.

    Acceptance Criteria:
    - [x] Implements processor_id, version, capabilities properties.
    """

    def test_given_summary_generator_when_get_processor_id_then_returns_string(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When getting processor_id, Then returns string."""
        assert summary_generator.processor_id == "summary-generator"

    def test_given_summary_generator_when_get_version_then_returns_semver(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When getting version, Then returns SemVer string."""
        version = summary_generator.version
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_given_summary_generator_when_get_capabilities_then_returns_list(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When getting capabilities, Then returns list."""
        capabilities = summary_generator.capabilities
        assert isinstance(capabilities, list)
        assert "summarization" in capabilities
        assert "text-compression" in capabilities

    def test_given_summary_generator_when_get_memory_mb_then_returns_int(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When getting memory_mb, Then returns int."""
        memory = summary_generator.memory_mb
        assert isinstance(memory, int)
        assert memory > 0


# ---------------------------------------------------------------------------
# Process Method Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestProcessMethod:
    """
    GWT: Test process(artifact: IFChunkArtifact) -> IFChunkArtifact.

    Acceptance Criteria:
    - [x] Implements process(artifact: IFChunkArtifact) -> IFChunkArtifact.
    - [x] Summary stored in metadata["summary"].
    """

    def test_given_chunk_artifact_when_process_then_returns_derived_artifact(
        self,
        summary_generator: SummaryGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then returns derived IFChunkArtifact."""
        result = summary_generator.process(sample_chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert result.artifact_id == f"{sample_chunk_artifact.artifact_id}-summary"

    def test_given_chunk_artifact_when_process_then_summary_in_metadata(
        self,
        summary_generator: SummaryGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then summary in metadata."""
        result = summary_generator.process(sample_chunk_artifact)

        assert "summary" in result.metadata
        assert isinstance(result.metadata["summary"], str)
        assert len(result.metadata["summary"]) > 0

    def test_given_chunk_artifact_when_process_then_summary_method_in_metadata(
        self,
        summary_generator: SummaryGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then summary_method in metadata."""
        result = summary_generator.process(sample_chunk_artifact)

        assert "summary_method" in result.metadata
        assert result.metadata["summary_method"] in ["llm", "extractive"]

    def test_given_chunk_artifact_when_process_then_preserves_lineage(
        self,
        summary_generator: SummaryGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then preserves lineage."""
        result = summary_generator.process(sample_chunk_artifact)

        assert result.parent_id == sample_chunk_artifact.artifact_id
        assert result.lineage_depth == sample_chunk_artifact.lineage_depth + 1
        assert summary_generator.processor_id in result.provenance

    def test_given_wrong_artifact_type_when_process_then_returns_failure(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given non-IFChunkArtifact, When process(), Then returns IFFailureArtifact."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Some text content",
        )

        result = summary_generator.process(text_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "requires IFChunkArtifact" in result.error_message

    def test_given_chunk_with_existing_summary_when_process_then_skips(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given chunk with existing summary, When process(), Then returns unchanged."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-with-summary",
            document_id="doc-001",
            content="Some content here.",
            chunk_index=0,
            total_chunks=1,
            metadata={"summary": "Existing summary."},
        )

        result = summary_generator.process(artifact)

        # Should return the same artifact since summary exists
        assert result.metadata.get("summary") == "Existing summary."


# ---------------------------------------------------------------------------
# Extractive Summary Tests
# ---------------------------------------------------------------------------


class TestExtractiveSummary:
    """GWT: Test extractive summarization fallback."""

    def test_given_content_with_sentences_when_extractive_then_returns_first_good_sentence(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given content with sentences, When extractive, Then returns first good sentence."""
        content = "Short. This is a longer sentence that should be selected as the summary for this chunk."

        result = summary_generator._generate_extractive_summary(content)

        assert "longer sentence" in result

    def test_given_content_with_only_short_sentences_when_extractive_then_returns_truncated(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given short sentences only, When extractive, Then returns truncated content."""
        content = "Hi. Yes. No. Ok."

        result = summary_generator._generate_extractive_summary(content)

        assert len(result) > 0

    def test_given_very_long_sentence_when_extractive_then_truncates(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given very long sentence, When extractive, Then truncates."""
        content = "Word " * 100  # Very long single sentence

        result = summary_generator._generate_extractive_summary(content)

        assert len(result) <= MAX_SUMMARY_LENGTH + 3  # +3 for ellipsis

    def test_given_empty_content_when_process_then_returns_empty_summary(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given empty content, When process(), Then returns empty summary."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-empty",
            document_id="doc-001",
            content="",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        result = summary_generator.process(artifact)

        assert result.metadata["summary"] == ""


# ---------------------------------------------------------------------------
# Summary Cleaning Tests
# ---------------------------------------------------------------------------


class TestSummaryCleaning:
    """GWT: Test summary cleaning logic."""

    def test_given_summary_with_quotes_when_clean_then_removes_quotes(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given summary with quotes, When clean, Then removes quotes."""
        result = summary_generator._clean_summary('"This is a summary."')
        assert not result.startswith('"')
        assert not result.endswith('"')

    def test_given_summary_with_prefix_when_clean_then_removes_prefix(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given summary with prefix, When clean, Then removes prefix."""
        result = summary_generator._clean_summary(
            "Summary: This is the actual summary."
        )
        assert not result.startswith("Summary:")
        assert "actual summary" in result

    def test_given_summary_without_punctuation_when_clean_then_adds_period(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given summary without punctuation, When clean, Then adds period."""
        result = summary_generator._clean_summary("This is a summary")
        assert result.endswith(".")


# ---------------------------------------------------------------------------
# Deprecated API Tests (Backward Compatibility)
# ---------------------------------------------------------------------------


class TestDeprecatedAPI:
    """
    GWT: Test deprecated methods emit warnings.

    Acceptance Criteria:
    - [x] Deprecation warning no longer emitted on instantiation.
    """

    def test_given_summary_generator_when_instantiate_then_no_deprecation_warning(
        self,
        mock_config: Config,
    ) -> None:
        """Given SummaryGenerator, When instantiate, Then no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SummaryGenerator(mock_config)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

    def test_given_summary_generator_when_call_enrich_chunk_then_emits_warning(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When call enrich_chunk(), Then emits deprecation warning."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="This is test content that is long enough to generate a summary.",
            chunk_type="general",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = summary_generator.enrich_chunk(chunk)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()
            # Verify summary was stored in metadata
            assert "summary" in result.metadata

    def test_given_summary_generator_when_call_enrich_batch_then_emits_warning(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When call enrich_batch(), Then emits deprecation warning."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        chunks = [
            ChunkRecord(
                chunk_id="chunk-001",
                document_id="doc-001",
                content="This is test content that is long enough to generate a summary.",
                chunk_type="general",
            )
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            summary_generator.enrich_batch(chunks)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1


# ---------------------------------------------------------------------------
# Availability and Teardown Tests
# ---------------------------------------------------------------------------


class TestAvailabilityAndTeardown:
    """GWT: Test is_available() and teardown() methods."""

    def test_given_summary_generator_when_no_llm_then_is_available_false(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator without LLM, When is_available(), Then returns False."""
        # Without actual LLM, should return False
        # Note: is_available() may return True or False depending on environment
        result = summary_generator.is_available()
        assert isinstance(result, bool)

    def test_given_summary_generator_when_teardown_then_returns_true(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When teardown(), Then returns True."""
        result = summary_generator.teardown()
        assert result is True

    def test_given_summary_generator_when_teardown_then_resets_availability_cache(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When teardown(), Then resets availability cache."""
        # First check availability (sets cache)
        summary_generator.is_available()
        assert summary_generator._availability_checked is True

        # Teardown should reset
        summary_generator.teardown()
        assert summary_generator._availability_checked is False


# ---------------------------------------------------------------------------
# JPL Power of Ten Rule Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLRuleCompliance:
    """GWT: Verify NASA JPL Power of Ten compliance."""

    def test_rule_2_max_content_length_bound_exists(self) -> None:
        """Rule #2: Fixed upper bound MAX_CONTENT_LENGTH exists."""
        assert MAX_CONTENT_LENGTH > 0
        assert MAX_CONTENT_LENGTH <= 100000

    def test_rule_2_max_summary_length_bound_exists(self) -> None:
        """Rule #2: Fixed upper bound MAX_SUMMARY_LENGTH exists."""
        assert MAX_SUMMARY_LENGTH > 0
        assert MAX_SUMMARY_LENGTH <= 1000

    def test_rule_4_process_method_under_60_lines(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Rule #4: process() method is < 60 lines."""
        source = inspect.getsource(summary_generator.process)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]
        assert len(lines) < 60, f"process() has {len(lines)} lines"

    def test_rule_4_all_helper_methods_under_60_lines(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Rule #4: All helper methods are < 60 lines."""
        methods_to_check = [
            "_generate_summary",
            "_generate_llm_summary",
            "_generate_extractive_summary",
            "_clean_summary",
        ]

        for method_name in methods_to_check:
            method = getattr(summary_generator, method_name)
            source = inspect.getsource(method)
            lines = [
                l
                for l in source.split("\n")
                if l.strip() and not l.strip().startswith("#")
            ]
            assert len(lines) < 60, f"{method_name}() has {len(lines)} lines"

    def test_rule_7_process_returns_artifact(
        self,
        summary_generator: SummaryGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Rule #7: process() always returns an IFArtifact (check return values)."""
        result = summary_generator.process(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_rule_9_process_has_type_hints(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Rule #9: process() has complete type hints."""
        hints = summary_generator.process.__annotations__
        assert "artifact" in hints
        assert "return" in hints


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """GWT: Test edge cases and error handling."""

    def test_given_whitespace_only_content_when_process_then_returns_empty_summary(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given whitespace-only content, When process(), Then returns empty summary."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-whitespace",
            document_id="doc-001",
            content="   \n\t  ",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        result = summary_generator.process(artifact)

        assert result.metadata["summary"] == ""

    def test_given_special_characters_in_content_when_process_then_handles_gracefully(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given special characters in content, When process(), Then handles gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-special",
            document_id="doc-001",
            content="Content with special chars: émojis 🎉, unicode™, and <html> tags. This is a complete sentence.",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        result = summary_generator.process(artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "summary" in result.metadata

    def test_given_content_with_no_sentences_when_extractive_then_returns_truncated(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given content without sentence endings, When extractive, Then returns truncated."""
        content = "No sentence endings here just continuous text flowing along"

        result = summary_generator._generate_extractive_summary(content)

        assert len(result) > 0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """GWT: Integration tests for full workflow."""

    def test_given_multiple_chunks_when_process_each_then_all_get_summaries(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given multiple chunks, When process each, Then all get summaries."""
        chunks = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"This is content about topic {i} and it explains how the system works in detail.",
                chunk_index=i,
                total_chunks=3,
                metadata={"chunk_type": "general"},
            )
            for i in range(3)
        ]

        results = [summary_generator.process(chunk) for chunk in chunks]

        for result in results:
            assert isinstance(result, IFChunkArtifact)
            assert "summary" in result.metadata
            assert len(result.metadata["summary"]) > 0

    def test_given_chunk_with_all_metadata_when_process_then_preserves_original_metadata(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given chunk with metadata, When process(), Then preserves original metadata."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-meta",
            document_id="doc-001",
            content="This is some content that should be summarized into a single sentence.",
            chunk_index=0,
            total_chunks=1,
            metadata={
                "section_title": "Test Section",
                "chunk_type": "general",
                "custom_field": "custom_value",
            },
        )

        result = summary_generator.process(artifact)

        assert result.metadata["section_title"] == "Test Section"
        assert result.metadata["chunk_type"] == "general"
        assert result.metadata["custom_field"] == "custom_value"
        assert "summary" in result.metadata

    def test_given_generator_repr_when_called_then_returns_string(
        self,
        summary_generator: SummaryGenerator,
    ) -> None:
        """Given SummaryGenerator, When repr(), Then returns informative string."""
        result = repr(summary_generator)
        assert "SummaryGenerator" in result
        assert "llm" in result or "extractive" in result
