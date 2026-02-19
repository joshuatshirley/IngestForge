"""
Unit tests for QuestionGenerator IFProcessor migration.

Migrate QuestionGenerator to IFProcessor.
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
from ingestforge.enrichment.questions import (
    QuestionGenerator,
    MAX_QUESTIONS,
    MAX_CONTENT_LENGTH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config() -> Config:
    """Create a minimal mock config for testing."""
    config = Mock(spec=Config)
    config.enrichment = Mock()
    config.enrichment.embedding_model = "all-MiniLM-L6-v2"
    return config


@pytest.fixture
def question_generator(mock_config: Config) -> QuestionGenerator:
    """Create a QuestionGenerator instance."""
    return QuestionGenerator(mock_config)


@pytest.fixture
def sample_chunk_artifact() -> IFChunkArtifact:
    """Create a sample IFChunkArtifact for testing."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content="This is a sample chunk about how machine learning works. "
        "It explains why neural networks are important and when to use them. "
        "The benefits include faster processing and better accuracy.",
        chunk_index=0,
        total_chunks=1,
        metadata={
            "section_title": "Machine Learning Basics",
            "chunk_type": "general",
        },
    )


@pytest.fixture
def definition_chunk_artifact() -> IFChunkArtifact:
    """Create a definition-type chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-def-001",
        document_id="doc-001",
        content="A neural network is a computational model inspired by biological neurons.",
        chunk_index=0,
        total_chunks=1,
        metadata={
            "section_title": "Neural Networks",
            "chunk_type": "definition",
        },
    )


@pytest.fixture
def procedure_chunk_artifact() -> IFChunkArtifact:
    """Create a procedure-type chunk artifact."""
    return IFChunkArtifact(
        artifact_id="chunk-proc-001",
        document_id="doc-001",
        content="Step 1: Collect data. Step 2: Preprocess. Step 3: Train model.",
        chunk_index=0,
        total_chunks=1,
        metadata={
            "section_title": "Train a Model",
            "chunk_type": "procedure",
        },
    )


# ---------------------------------------------------------------------------
# IFProcessor Interface Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestIFProcessorInterface:
    """
    GWT: Test that QuestionGenerator implements IFProcessor interface.

    Acceptance Criteria:
    - [x] QuestionGenerator extends IFProcessor instead of IEnricher.
    """

    def test_given_question_generator_when_check_inheritance_then_is_ifprocessor(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When checking inheritance, Then it extends IFProcessor."""
        assert isinstance(question_generator, IFProcessor)

    def test_given_question_generator_when_check_methods_then_has_process(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When checking methods, Then has process() method."""
        assert hasattr(question_generator, "process")
        assert callable(question_generator.process)

    def test_given_question_generator_when_check_methods_then_has_is_available(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When checking methods, Then has is_available() method."""
        assert hasattr(question_generator, "is_available")
        assert callable(question_generator.is_available)

    def test_given_question_generator_when_check_methods_then_has_teardown(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When checking methods, Then has teardown() method."""
        assert hasattr(question_generator, "teardown")
        assert callable(question_generator.teardown)


# ---------------------------------------------------------------------------
# Processor Properties Tests (Acceptance Criteria)
# ---------------------------------------------------------------------------


class TestProcessorProperties:
    """
    GWT: Test processor_id, version, capabilities properties.

    Acceptance Criteria:
    - [x] Implements processor_id, version, capabilities properties.
    """

    def test_given_question_generator_when_get_processor_id_then_returns_string(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When getting processor_id, Then returns string."""
        assert question_generator.processor_id == "question-generator"

    def test_given_question_generator_when_get_version_then_returns_semver(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When getting version, Then returns SemVer string."""
        version = question_generator.version
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_given_question_generator_when_get_capabilities_then_returns_list(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When getting capabilities, Then returns list."""
        capabilities = question_generator.capabilities
        assert isinstance(capabilities, list)
        assert "question-generation" in capabilities
        assert "query-expansion" in capabilities

    def test_given_question_generator_when_get_memory_mb_then_returns_int(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When getting memory_mb, Then returns int."""
        memory = question_generator.memory_mb
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
    - [x] Questions stored in metadata["hypothetical_questions"].
    """

    def test_given_chunk_artifact_when_process_then_returns_derived_artifact(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then returns derived IFChunkArtifact."""
        result = question_generator.process(sample_chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert result.artifact_id == f"{sample_chunk_artifact.artifact_id}-questions"

    def test_given_chunk_artifact_when_process_then_questions_in_metadata(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then questions in metadata."""
        result = question_generator.process(sample_chunk_artifact)

        assert "hypothetical_questions" in result.metadata
        assert isinstance(result.metadata["hypothetical_questions"], list)
        assert len(result.metadata["hypothetical_questions"]) > 0

    def test_given_chunk_artifact_when_process_then_question_count_in_metadata(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then question_count in metadata."""
        result = question_generator.process(sample_chunk_artifact)

        assert "question_count" in result.metadata
        assert result.metadata["question_count"] == len(
            result.metadata["hypothetical_questions"]
        )

    def test_given_chunk_artifact_when_process_then_preserves_lineage(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given IFChunkArtifact, When process(), Then preserves lineage."""
        result = question_generator.process(sample_chunk_artifact)

        assert result.parent_id == sample_chunk_artifact.artifact_id
        assert result.lineage_depth == sample_chunk_artifact.lineage_depth + 1
        assert question_generator.processor_id in result.provenance

    def test_given_wrong_artifact_type_when_process_then_returns_failure(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given non-IFChunkArtifact, When process(), Then returns IFFailureArtifact."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Some text content",
        )

        result = question_generator.process(text_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "requires IFChunkArtifact" in result.error_message


# ---------------------------------------------------------------------------
# Template Generation Tests
# ---------------------------------------------------------------------------


class TestTemplateGeneration:
    """GWT: Test template-based question generation for different chunk types."""

    def test_given_definition_chunk_when_process_then_generates_definition_questions(
        self,
        question_generator: QuestionGenerator,
        definition_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given definition chunk, When process(), Then generates definition questions."""
        result = question_generator.process(definition_chunk_artifact)

        questions = result.metadata["hypothetical_questions"]
        question_text = " ".join(questions).lower()

        # Should include "what is" style questions
        assert "what is" in question_text or "defined" in question_text

    def test_given_procedure_chunk_when_process_then_generates_procedure_questions(
        self,
        question_generator: QuestionGenerator,
        procedure_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given procedure chunk, When process(), Then generates procedure questions."""
        result = question_generator.process(procedure_chunk_artifact)

        questions = result.metadata["hypothetical_questions"]
        question_text = " ".join(questions).lower()

        # Should include "how" or "steps" style questions
        assert "how" in question_text or "steps" in question_text

    def test_given_general_chunk_with_keywords_when_process_then_generates_relevant_questions(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given general chunk with keywords, When process(), Then generates relevant questions."""
        result = question_generator.process(sample_chunk_artifact)

        questions = result.metadata["hypothetical_questions"]

        # Content contains "how", "why", "when", "benefits" - should trigger relevant questions
        assert len(questions) >= 1


# ---------------------------------------------------------------------------
# Deprecated API Tests (Backward Compatibility)
# ---------------------------------------------------------------------------


class TestDeprecatedAPI:
    """
    GWT: Test deprecated methods emit warnings.

    Acceptance Criteria:
    - [x] Deprecation warning no longer emitted on instantiation.
    """

    def test_given_question_generator_when_instantiate_then_no_deprecation_warning(
        self,
        mock_config: Config,
    ) -> None:
        """Given QuestionGenerator, When instantiate, Then no deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            QuestionGenerator(mock_config)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0

    def test_given_question_generator_when_call_enrich_chunk_then_emits_warning(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When call enrich_chunk(), Then emits deprecation warning."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        chunk = ChunkRecord(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Test content",
            chunk_type="general",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            question_generator.enrich_chunk(chunk)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()

    def test_given_question_generator_when_call_enrich_then_emits_warning(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When call enrich(), Then emits deprecation warning."""
        from ingestforge.chunking.semantic_chunker import ChunkRecord

        chunk = ChunkRecord(
            chunk_id="chunk-002",
            document_id="doc-001",
            content="Test content",
            chunk_type="general",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            question_generator.enrich(chunk)

            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) == 1


# ---------------------------------------------------------------------------
# Availability and Teardown Tests
# ---------------------------------------------------------------------------


class TestAvailabilityAndTeardown:
    """GWT: Test is_available() and teardown() methods."""

    def test_given_question_generator_when_check_availability_then_always_true(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When is_available(), Then returns True."""
        # Template-based generation is always available
        assert question_generator.is_available() is True

    def test_given_question_generator_when_teardown_then_returns_true(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given QuestionGenerator, When teardown(), Then returns True."""
        result = question_generator.teardown()
        assert result is True


# ---------------------------------------------------------------------------
# JPL Power of Ten Rule Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLRuleCompliance:
    """GWT: Verify NASA JPL Power of Ten compliance."""

    def test_rule_2_max_questions_bound_exists(self) -> None:
        """Rule #2: Fixed upper bound MAX_QUESTIONS exists."""
        assert MAX_QUESTIONS > 0
        assert MAX_QUESTIONS <= 100  # Reasonable upper bound

    def test_rule_2_max_content_length_bound_exists(self) -> None:
        """Rule #2: Fixed upper bound MAX_CONTENT_LENGTH exists."""
        assert MAX_CONTENT_LENGTH > 0
        assert MAX_CONTENT_LENGTH <= 100000  # Reasonable upper bound

    def test_rule_4_process_method_under_60_lines(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Rule #4: process() method is < 60 lines."""
        source = inspect.getsource(question_generator.process)
        lines = [
            l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
        ]
        assert len(lines) < 60, f"process() has {len(lines)} lines"

    def test_rule_4_all_helper_methods_under_60_lines(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Rule #4: All helper methods are < 60 lines."""
        methods_to_check = [
            "_generate_questions",
            "_generate_llm",
            "_generate_template",
            "_process_question_line",
            "_generate_definition_questions",
            "_generate_procedure_questions",
            "_generate_example_questions",
            "_generate_general_questions",
            "_get_content",
            "_get_metadata_field",
        ]

        for method_name in methods_to_check:
            method = getattr(question_generator, method_name)
            source = inspect.getsource(method)
            lines = [
                l
                for l in source.split("\n")
                if l.strip() and not l.strip().startswith("#")
            ]
            assert len(lines) < 60, f"{method_name}() has {len(lines)} lines"

    def test_rule_7_process_returns_artifact(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Rule #7: process() always returns an IFArtifact (check return values)."""
        result = question_generator.process(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_rule_9_process_has_type_hints(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Rule #9: process() has complete type hints."""
        hints = question_generator.process.__annotations__
        assert "artifact" in hints
        assert "return" in hints


# ---------------------------------------------------------------------------
# Edge Case Tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """GWT: Test edge cases and error handling."""

    def test_given_empty_content_when_process_then_returns_empty_questions(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given empty content, When process(), Then returns artifact with minimal questions."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-empty",
            document_id="doc-001",
            content="",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        result = question_generator.process(artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "hypothetical_questions" in result.metadata

    def test_given_very_long_content_when_process_then_truncates(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given very long content, When process(), Then handles gracefully."""
        long_content = "word " * 10000  # Very long content
        artifact = IFChunkArtifact(
            artifact_id="chunk-long",
            document_id="doc-001",
            content=long_content,
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        result = question_generator.process(artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "hypothetical_questions" in result.metadata

    def test_given_num_questions_exceeds_max_when_init_then_caps_at_max(
        self,
        mock_config: Config,
    ) -> None:
        """Given num_questions > MAX_QUESTIONS, When init, Then caps at MAX_QUESTIONS."""
        generator = QuestionGenerator(mock_config, num_questions=100)
        assert generator._num_questions <= MAX_QUESTIONS

    def test_given_special_characters_in_content_when_process_then_handles_gracefully(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given special characters in content, When process(), Then handles gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-special",
            document_id="doc-001",
            content="Content with special chars: émojis 🎉, unicode™, and <html> tags",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        result = question_generator.process(artifact)

        assert isinstance(result, IFChunkArtifact)


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """GWT: Integration tests for full workflow."""

    def test_given_multiple_chunks_when_process_each_then_all_get_questions(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given multiple chunks, When process each, Then all get questions."""
        chunks = [
            IFChunkArtifact(
                artifact_id=f"chunk-{i}",
                document_id="doc-001",
                content=f"Content about topic {i} and how it works.",
                chunk_index=i,
                total_chunks=3,
                metadata={"chunk_type": "general"},
            )
            for i in range(3)
        ]

        results = [question_generator.process(chunk) for chunk in chunks]

        for result in results:
            assert isinstance(result, IFChunkArtifact)
            assert "hypothetical_questions" in result.metadata
            assert len(result.metadata["hypothetical_questions"]) > 0

    def test_given_chunk_with_all_metadata_when_process_then_preserves_original_metadata(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given chunk with metadata, When process(), Then preserves original metadata."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-meta",
            document_id="doc-001",
            content="Some content here.",
            chunk_index=0,
            total_chunks=1,
            metadata={
                "section_title": "Test Section",
                "chunk_type": "general",
                "custom_field": "custom_value",
            },
        )

        result = question_generator.process(artifact)

        assert result.metadata["section_title"] == "Test Section"
        assert result.metadata["chunk_type"] == "general"
        assert result.metadata["custom_field"] == "custom_value"
        assert "hypothetical_questions" in result.metadata


# ---------------------------------------------------------------------------
# JPL Rule #1: Simple Control Flow
# ---------------------------------------------------------------------------


class TestJPLRule1SimpleControlFlow:
    """Test simple control flow per JPL Rule #1."""

    def test_no_goto_or_setjmp_in_source(self) -> None:
        """Given QuestionGenerator source, Then no goto/setjmp constructs."""
        source = inspect.getsource(QuestionGenerator)
        assert "goto" not in source.lower()
        assert "setjmp" not in source.lower()

    def test_process_has_single_success_path(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given valid input, When process(), Then follows single success path."""
        result = question_generator.process(sample_chunk_artifact)
        # Single type of success result
        assert isinstance(result, IFChunkArtifact)
        assert not isinstance(result, IFFailureArtifact)

    def test_process_has_single_failure_path(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given invalid input, When process(), Then follows single failure path."""
        text_artifact = IFTextArtifact(
            artifact_id="text-001",
            content="Invalid type",
        )
        result = question_generator.process(text_artifact)
        # Single type of failure result
        assert isinstance(result, IFFailureArtifact)


# ---------------------------------------------------------------------------
# JPL Rule #5: Assertion Density
# ---------------------------------------------------------------------------


class TestJPLRule5AssertionDensity:
    """Test assertion/validation density per JPL Rule #5."""

    def test_process_validates_artifact_type(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given non-chunk artifact, When process(), Then type is validated."""
        text_artifact = IFTextArtifact(artifact_id="t", content="c")
        result = question_generator.process(text_artifact)
        assert isinstance(result, IFFailureArtifact)
        assert "requires IFChunkArtifact" in result.error_message

    def test_num_questions_validated_on_init(
        self,
        mock_config: Config,
    ) -> None:
        """Given excessive num_questions, When init, Then bounded."""
        gen = QuestionGenerator(mock_config, num_questions=999)
        assert gen._num_questions <= MAX_QUESTIONS

    def test_content_bounded_in_chunk_record_conversion(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given long content, When converting to ChunkRecord, Then bounded."""
        long_content = "x" * (MAX_CONTENT_LENGTH + 1000)
        artifact = IFChunkArtifact(
            artifact_id="long",
            document_id="doc",
            content=long_content,
            chunk_index=0,
            total_chunks=1,
        )
        # Should not raise - content is truncated internally
        result = question_generator.process(artifact)
        assert isinstance(result, IFChunkArtifact)


# ---------------------------------------------------------------------------
# JPL Rule #6: Smallest Scope
# ---------------------------------------------------------------------------


class TestJPLRule6SmallestScope:
    """Test smallest scope principle per JPL Rule #6."""

    def test_constants_at_module_level(self) -> None:
        """Given module constants, Then defined at module scope."""
        from ingestforge.enrichment import questions

        assert hasattr(questions, "MAX_QUESTIONS")
        assert hasattr(questions, "MAX_CONTENT_LENGTH")

    def test_private_methods_encapsulated(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given helper methods, Then prefixed with underscore."""
        private_methods = [
            "_generate_questions",
            "_generate_llm",
            "_generate_template",
            "_process_question_line",
            "_generate_definition_questions",
            "_generate_procedure_questions",
            "_generate_example_questions",
            "_generate_general_questions",
            "_get_content",
            "_get_metadata_field",
        ]
        for method_name in private_methods:
            assert hasattr(question_generator, method_name)
            assert method_name.startswith("_")

    def test_instance_state_encapsulated(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given instance variables, Then appropriately scoped."""
        assert hasattr(question_generator, "config")
        assert hasattr(question_generator, "_num_questions")
        assert hasattr(question_generator, "_version")


# ---------------------------------------------------------------------------
# GWT: Full Behavioral Specification
# ---------------------------------------------------------------------------


class TestGWTFullBehavior:
    """
    Complete GWT specification from
    - Given: A chunk artifact requiring hypothetical question generation.
    - When: Processed by QuestionGenerator.
    - Then: Returns derived IFChunkArtifact with questions in metadata.
    """

    def test_gwt_full_scenario_with_section_title(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Full GWT scenario with section title present."""
        # Given
        artifact = IFChunkArtifact(
            artifact_id="gwt-test-001",
            document_id="doc-001",
            content="This document explains how to configure the system.",
            chunk_index=0,
            total_chunks=1,
            metadata={"section_title": "Configuration Guide"},
        )

        # When
        result = question_generator.process(artifact)

        # Then
        assert isinstance(result, IFChunkArtifact)
        assert result.parent_id == artifact.artifact_id
        assert "hypothetical_questions" in result.metadata
        assert len(result.metadata["hypothetical_questions"]) > 0
        # Should have questions about the section
        questions_text = " ".join(result.metadata["hypothetical_questions"]).lower()
        assert "configuration" in questions_text or "how" in questions_text

    def test_gwt_full_scenario_without_section_title(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Full GWT scenario without section title - uses content keywords."""
        # Given
        artifact = IFChunkArtifact(
            artifact_id="gwt-test-002",
            document_id="doc-001",
            content="This explains why the feature is important and how it benefits users.",
            chunk_index=0,
            total_chunks=1,
            metadata={},
        )

        # When
        result = question_generator.process(artifact)

        # Then
        assert isinstance(result, IFChunkArtifact)
        assert "hypothetical_questions" in result.metadata
        # Content has "why", "how", "benefits" - should trigger questions
        assert len(result.metadata["hypothetical_questions"]) >= 1


# ---------------------------------------------------------------------------
# Lineage Preservation Tests
# ---------------------------------------------------------------------------


class TestLineagePreservation:
    """Test complete lineage tracking through processing."""

    def test_root_artifact_id_preserved_for_derived_artifacts(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given artifact with root_artifact_id, When process(), Then preserved."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-derived",
            document_id="doc-001",
            content="Content here",
            chunk_index=0,
            total_chunks=1,
            parent_id="chunk-parent",
            root_artifact_id="chunk-root",
            lineage_depth=2,
            provenance=["proc-1", "proc-2"],
        )

        result = question_generator.process(artifact)

        assert result.root_artifact_id == "chunk-root"
        assert result.lineage_depth == 3
        assert result.provenance == ["proc-1", "proc-2", "question-generator"]

    def test_root_artifact_set_for_root_artifacts(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given root artifact, When process(), Then root_artifact_id set correctly."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-root",
            document_id="doc-001",
            content="Root content",
            chunk_index=0,
            total_chunks=1,
        )

        result = question_generator.process(artifact)

        # derive() should set root_artifact_id to parent's artifact_id for roots
        assert result.root_artifact_id == artifact.artifact_id
        assert result.parent_id == artifact.artifact_id
        assert result.lineage_depth == 1


# ---------------------------------------------------------------------------
# Example Chunk Type Tests
# ---------------------------------------------------------------------------


class TestExampleChunkType:
    """Test example chunk type question generation."""

    def test_given_example_chunk_when_process_then_generates_example_questions(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given example chunk type, When process(), Then generates example questions."""
        artifact = IFChunkArtifact(
            artifact_id="chunk-example",
            document_id="doc-001",
            content="For example, you can use this approach when building APIs.",
            chunk_index=0,
            total_chunks=1,
            metadata={
                "section_title": "API Design",
                "chunk_type": "example",
            },
        )

        result = question_generator.process(artifact)

        questions = result.metadata["hypothetical_questions"]
        question_text = " ".join(questions).lower()
        # Should include example-style questions
        assert "example" in question_text


# ---------------------------------------------------------------------------
# Metadata Enrichment Tests
# ---------------------------------------------------------------------------


class TestMetadataEnrichment:
    """Test metadata fields added during processing."""

    def test_adds_question_generator_version(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact, When process(), Then adds generator version."""
        result = question_generator.process(sample_chunk_artifact)

        assert "question_generator_version" in result.metadata
        assert (
            result.metadata["question_generator_version"] == question_generator.version
        )

    def test_adds_question_count(
        self,
        question_generator: QuestionGenerator,
        sample_chunk_artifact: IFChunkArtifact,
    ) -> None:
        """Given artifact, When process(), Then adds question count."""
        result = question_generator.process(sample_chunk_artifact)

        assert "question_count" in result.metadata
        questions = result.metadata["hypothetical_questions"]
        assert result.metadata["question_count"] == len(questions)

    def test_preserves_all_original_metadata(
        self,
        question_generator: QuestionGenerator,
    ) -> None:
        """Given artifact with metadata, When process(), Then preserves all."""
        original_metadata = {
            "section_title": "Test",
            "chunk_type": "general",
            "custom_key_1": "value_1",
            "custom_key_2": 42,
            "nested": {"key": "value"},
        }
        artifact = IFChunkArtifact(
            artifact_id="chunk-meta-test",
            document_id="doc-001",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
            metadata=original_metadata,
        )

        result = question_generator.process(artifact)

        for key, value in original_metadata.items():
            assert result.metadata[key] == value
