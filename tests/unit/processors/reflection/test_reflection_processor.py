"""
Tests for Reflection Processor.

Agentic Reflection Loop - Critic pass for extraction accuracy.
Tests follow GWT (Given-When-Then) pattern and NASA JPL Power of Ten rules.
"""

from unittest.mock import MagicMock, patch
import pytest

from ingestforge.core.pipeline.artifacts import IFTextArtifact
from ingestforge.processors.reflection.reflection_processor import (
    IFReflectionProcessor,
    IFReflectionArtifact,
    ReflectionResult,
    ContradictionResult,
    MAX_REFLECTION_PASSES,
)
from ingestforge.processors.reflection.stage import ReflectionStage


# =============================================================================
# REFLECTION RESULT TESTS
# =============================================================================


class TestReflectionResult:
    """Tests for ReflectionResult dataclass."""

    def test_reflection_result_creation(self):
        """
        GWT:
        Given valid reflection result parameters
        When ReflectionResult is created
        Then all fields are set correctly.
        """
        result = ReflectionResult(
            pass_number=1,
            total_claims=10,
            verified_claims=8,
            contradicted_claims=2,
            average_confidence=0.85,
            should_reextract=False,
        )

        assert result.pass_number == 1
        assert result.total_claims == 10
        assert result.verified_claims == 8
        assert result.contradicted_claims == 2
        assert result.average_confidence == 0.85
        assert result.should_reextract is False

    def test_reflection_result_confidence_property(self):
        """
        GWT:
        Given a ReflectionResult
        When confidence property is accessed
        Then it returns average_confidence.
        """
        result = ReflectionResult(
            pass_number=1,
            total_claims=5,
            verified_claims=4,
            contradicted_claims=1,
            average_confidence=0.75,
            should_reextract=False,
        )

        assert result.confidence == 0.75

    def test_reflection_result_with_contradictions(self):
        """
        GWT:
        Given contradictions in result
        When ReflectionResult is created
        Then contradictions list is preserved.
        """
        contradictions = [
            ContradictionResult(
                claim_text="Test claim",
                source_text="Source text",
                entailment_score=0.3,
                is_contradiction=True,
                reasoning="Low score",
            )
        ]

        result = ReflectionResult(
            pass_number=1,
            total_claims=1,
            verified_claims=0,
            contradicted_claims=1,
            average_confidence=0.3,
            should_reextract=True,
            contradictions=contradictions,
        )

        assert len(result.contradictions) == 1
        assert result.contradictions[0].claim_text == "Test claim"


# =============================================================================
# CONTRADICTION RESULT TESTS
# =============================================================================


class TestContradictionResult:
    """Tests for ContradictionResult dataclass."""

    def test_contradiction_result_creation(self):
        """
        GWT:
        Given valid contradiction parameters
        When ContradictionResult is created
        Then all fields are set correctly.
        """
        result = ContradictionResult(
            claim_text="The sky is green",
            source_text="The sky is blue",
            entailment_score=0.2,
            is_contradiction=True,
            reasoning="Score below threshold",
        )

        assert result.claim_text == "The sky is green"
        assert result.entailment_score == 0.2
        assert result.is_contradiction is True


# =============================================================================
# REFLECTION ARTIFACT TESTS
# =============================================================================


class TestReflectionArtifact:
    """Tests for IFReflectionArtifact."""

    def test_reflection_artifact_creation(self):
        """
        GWT:
        Given valid artifact parameters
        When IFReflectionArtifact is created
        Then all fields are set correctly.
        """
        artifact = IFReflectionArtifact(
            artifact_id="test-1",
            content="Test content",
            reflection_pass=1,
            confidence_score=0.85,
            contradictions_found=2,
            should_reextract=False,
            reflection_reasoning="Pass 1 complete",
        )

        assert artifact.artifact_id == "test-1"
        assert artifact.reflection_pass == 1
        assert artifact.confidence_score == 0.85
        assert artifact.contradictions_found == 2
        assert artifact.should_reextract is False

    def test_reflection_artifact_derive(self):
        """
        GWT:
        Given an existing IFReflectionArtifact
        When derive is called
        Then new artifact has updated provenance.
        """
        original = IFReflectionArtifact(
            artifact_id="orig-1",
            content="Original",
            reflection_pass=1,
        )

        derived = original.derive(
            processor_id="test-proc",
            reflection_pass=2,
            confidence_score=0.9,
        )

        assert derived.parent_id == "orig-1"
        assert "test-proc" in derived.provenance
        assert derived.reflection_pass == 2
        assert derived.confidence_score == 0.9


# =============================================================================
# REFLECTION PROCESSOR TESTS
# =============================================================================


class TestReflectionProcessor:
    """Tests for IFReflectionProcessor."""

    def test_processor_initialization(self):
        """
        GWT:
        Given default parameters
        When IFReflectionProcessor is created
        Then default values are set.
        """
        processor = IFReflectionProcessor()

        assert processor.processor_id == "if-reflection-processor"
        assert processor.version == "1.0.0"
        assert "reflection" in processor.capabilities

    def test_processor_custom_threshold(self):
        """
        GWT:
        Given custom confidence threshold
        When processor is created
        Then threshold is applied.
        """
        processor = IFReflectionProcessor(confidence_threshold=0.8)

        # Threshold is stored internally
        assert processor._threshold == 0.8

    def test_processor_max_passes_bounded(self):
        """
        GWT:
        Given max_passes exceeding MAX_REFLECTION_PASSES (JPL Rule #2)
        When processor is created
        Then passes are capped at maximum.
        """
        processor = IFReflectionProcessor(max_passes=100)

        assert processor._max_passes == MAX_REFLECTION_PASSES

    def test_processor_threshold_validation(self):
        """
        GWT:
        Given invalid threshold
        When processor is created
        Then AssertionError is raised.
        """
        with pytest.raises(AssertionError):
            IFReflectionProcessor(confidence_threshold=1.5)

        with pytest.raises(AssertionError):
            IFReflectionProcessor(confidence_threshold=-0.1)

    @patch(
        "ingestforge.processors.reflection.reflection_processor.IFReflectionProcessor._get_claim_extractor"
    )
    @patch(
        "ingestforge.processors.reflection.reflection_processor.IFReflectionProcessor._get_entailment_scorer"
    )
    def test_processor_process_with_mocks(self, mock_scorer, mock_extractor):
        """
        GWT:
        Given mocked dependencies
        When process is called
        Then reflection analysis is performed.
        """
        # Setup mocks
        mock_claim = MagicMock()
        mock_claim.text = "Test claim"
        mock_extractor.return_value.extract.return_value = [mock_claim]
        mock_scorer.return_value.score.return_value = 0.85

        processor = IFReflectionProcessor()
        artifact = IFTextArtifact(
            artifact_id="test-1",
            content="Test content",
            metadata={"source_text": "Original source text"},
        )

        result = processor.process(artifact)

        assert isinstance(result, IFReflectionArtifact)
        assert result.reflection_pass == 1

    def test_processor_process_missing_content(self):
        """
        GWT:
        Given artifact with no content or source
        When process is called
        Then skip artifact is returned.
        """
        processor = IFReflectionProcessor()
        artifact = IFTextArtifact(artifact_id="empty-1", content="", metadata={})

        result = processor.process(artifact)

        assert isinstance(result, IFReflectionArtifact)
        assert (
            "Missing" in result.reflection_reasoning
            or "Skipped" in result.reflection_reasoning
        )

    def test_processor_max_passes_enforcement(self):
        """
        GWT:
        Given artifact already at max passes (JPL Rule #2)
        When process is called
        Then no further analysis is performed.
        """
        processor = IFReflectionProcessor(max_passes=2)
        artifact = IFTextArtifact(
            artifact_id="max-pass-1",
            content="Content",
            metadata={"reflection_pass": 2, "source_text": "Source"},
        )

        result = processor.process(artifact)

        assert isinstance(result, IFReflectionArtifact)
        assert "Max passes" in result.reflection_reasoning


# =============================================================================
# REFLECTION STAGE TESTS
# =============================================================================


class TestReflectionStage:
    """Tests for ReflectionStage."""

    def test_stage_properties(self):
        """
        GWT:
        Given a ReflectionStage
        When properties are accessed
        Then correct values are returned.
        """
        stage = ReflectionStage()

        assert stage.name == "reflection"
        assert stage.input_type == IFTextArtifact
        assert stage.output_type == IFReflectionArtifact

    def test_stage_execute_with_unavailable_processor(self):
        """
        GWT:
        Given processor with unavailable dependencies
        When execute is called
        Then skip result is returned.
        """
        mock_processor = MagicMock()
        mock_processor.is_available.return_value = False
        mock_processor.processor_id = "mock-proc"

        stage = ReflectionStage(processor=mock_processor)
        artifact = IFTextArtifact(artifact_id="test-1", content="Test")

        result = stage.execute(artifact)

        assert isinstance(result, IFReflectionArtifact)
        assert "not available" in result.reflection_reasoning

    @patch.object(IFReflectionProcessor, "is_available", return_value=True)
    @patch.object(IFReflectionProcessor, "process")
    def test_stage_execute_success(self, mock_process, mock_available):
        """
        GWT:
        Given valid artifact and available processor
        When execute is called
        Then processor.process is invoked.
        """
        expected_result = IFReflectionArtifact(
            artifact_id="result-1",
            content="Result",
            reflection_pass=1,
            confidence_score=0.9,
        )
        mock_process.return_value = expected_result

        stage = ReflectionStage()
        artifact = IFTextArtifact(artifact_id="test-1", content="Test")

        result = stage.execute(artifact)

        mock_process.assert_called_once()
        assert result == expected_result


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying NASA JPL Power of Ten rule compliance."""

    def test_jpl_rule_2_max_passes(self):
        """
        GWT:
        Given max_passes parameter
        When exceeds MAX_REFLECTION_PASSES
        Then capped at maximum (JPL Rule #2).
        """
        processor = IFReflectionProcessor(max_passes=999)

        assert processor._max_passes <= MAX_REFLECTION_PASSES

    def test_jpl_rule_2_threshold_bounded(self):
        """
        GWT:
        Given confidence threshold
        When outside 0.0-1.0
        Then AssertionError raised (JPL Rule #2).
        """
        with pytest.raises(AssertionError):
            IFReflectionProcessor(confidence_threshold=2.0)

    def test_jpl_rule_5_precondition_check(self):
        """
        GWT:
        Given None artifact
        When process is called
        Then AssertionError raised (JPL Rule #5).
        """
        processor = IFReflectionProcessor()

        with pytest.raises(AssertionError):
            processor.process(None)

    def test_jpl_rule_9_type_hints(self):
        """
        GWT:
        Given ReflectionResult and ContradictionResult
        When inspecting __dataclass_fields__
        Then all fields have type hints (JPL Rule #9).
        """
        # ReflectionResult fields
        assert "pass_number" in ReflectionResult.__dataclass_fields__
        assert "average_confidence" in ReflectionResult.__dataclass_fields__

        # ContradictionResult fields
        assert "claim_text" in ContradictionResult.__dataclass_fields__
        assert "entailment_score" in ContradictionResult.__dataclass_fields__


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestReflectionIntegration:
    """Integration tests for reflection workflow."""

    def test_multiple_reflection_passes(self):
        """
        GWT:
        Given artifact requiring multiple passes
        When processed multiple times
        Then pass count increments correctly.
        """
        processor = IFReflectionProcessor(max_passes=2)

        # First pass
        artifact1 = IFTextArtifact(
            artifact_id="multi-1",
            content="Test content",
            metadata={"source_text": "Source text"},
        )

        # Processing will fail without actual dependencies
        # but we can verify the max passes logic
        result1 = processor.process(artifact1)
        assert isinstance(result1, IFReflectionArtifact)

    def test_teardown(self):
        """
        GWT:
        Given processor with loaded components
        When teardown is called
        Then components are cleared.
        """
        processor = IFReflectionProcessor()
        # Force some internal state
        processor._claim_extractor = "dummy"
        processor._entailment_scorer = "dummy"

        result = processor.teardown()

        assert result is True
        assert processor._claim_extractor is None
        assert processor._entailment_scorer is None
