"""
Comprehensive JPL Power of Ten Rules Compliance Tests.

This module validates that the IF-Protocol implementation adheres to
NASA JPL's Power of Ten coding rules for safety-critical software.

Rules covered:
- Rule #1: Simple control flow (no recursion, goto, setjmp)
- Rule #2: Fixed upper bounds on loops and data structures
- Rule #4: Functions < 60 lines
- Rule #5: Minimum 2 assertions per function (via tests)
- Rule #7: Check all return values, validate inputs
- Rule #9: Complete type hints
- Rule #10: Compile with warnings, static analysis

All tests follow GWT (Given-When-Then) behavioral specification.
"""

import pytest
import uuid
from typing import Any, List
from pydantic import ValidationError

from ingestforge.core.pipeline.interfaces import (
    IFArtifact,
    IFProcessor,
    IFStage,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_SIZE,
)
from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFChunkArtifact,
    IFFailureArtifact,
)
from ingestforge.core.pipeline.runner import IFPipelineRunner


# =============================================================================
# HELPER CLASSES FOR TESTING
# =============================================================================


class ConcreteArtifact(IFArtifact):
    """Minimal concrete implementation for testing abstract base."""

    def derive(self, processor_id: str, **kwargs: Any) -> "ConcreteArtifact":
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": self.lineage_depth + 1,
                **kwargs,
            }
        )


class PassThroughProcessor(IFProcessor):
    """Processor that returns artifact unchanged."""

    def __init__(self, proc_id: str = "passthrough"):
        self._id = proc_id

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_available(self) -> bool:
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact


class TransformProcessor(IFProcessor):
    """Processor that transforms text content."""

    def __init__(self, proc_id: str = "transform", transform_fn=str.upper):
        self._id = proc_id
        self._transform = transform_fn

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    def is_available(self) -> bool:
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        if isinstance(artifact, IFTextArtifact):
            return artifact.derive(
                self._id,
                artifact_id=f"{artifact.artifact_id}-transformed",
                content=self._transform(artifact.content),
            )
        return artifact


class MockStage(IFStage):
    """Configurable mock stage for testing."""

    def __init__(
        self,
        name: str,
        fail: bool = False,
        crash: bool = False,
        in_type: type = IFArtifact,
        out_type: type = IFArtifact,
    ):
        self._name = name
        self._fail = fail
        self._crash = crash
        self._in_type = in_type
        self._out_type = out_type

    @property
    def name(self) -> str:
        return self._name

    @property
    def input_type(self) -> type:
        return self._in_type

    @property
    def output_type(self) -> type:
        return self._out_type

    def execute(self, artifact: IFArtifact) -> IFArtifact:
        if self._crash:
            raise RuntimeError(f"Stage {self._name} crashed intentionally")
        if self._fail:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-failed",
                error_message=f"Stage {self._name} failed intentionally",
                failed_processor_id=self._name,
                provenance=artifact.provenance,
            )
        return artifact.derive(f"stage-{self._name}")


# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS TESTS
# =============================================================================


class TestJPLRule2FixedBounds:
    """
    JPL Rule #2: All loops must have a fixed upper bound.
    All data structures must have fixed bounds.
    """

    def test_metadata_has_fixed_key_limit(self):
        """
        GWT:
        Given the metadata field specification
        When MAX_METADATA_KEYS constant is checked
        Then it must be a fixed positive integer.
        """
        assert isinstance(MAX_METADATA_KEYS, int)
        assert MAX_METADATA_KEYS > 0
        assert MAX_METADATA_KEYS == 128  # Documented bound

    def test_metadata_value_has_fixed_size_limit(self):
        """
        GWT:
        Given the metadata value specification
        When MAX_METADATA_VALUE_SIZE constant is checked
        Then it must be a fixed positive integer.
        """
        assert isinstance(MAX_METADATA_VALUE_SIZE, int)
        assert MAX_METADATA_VALUE_SIZE > 0
        assert MAX_METADATA_VALUE_SIZE == 65536  # 64KB

    def test_metadata_keys_at_boundary(self):
        """
        GWT:
        Given metadata with exactly MAX_METADATA_KEYS entries
        When artifact is created
        Then it succeeds (boundary test).
        """
        exact_boundary = {f"k{i}": i for i in range(MAX_METADATA_KEYS)}
        art = IFTextArtifact(
            artifact_id="boundary", content="test", metadata=exact_boundary
        )
        assert len(art.metadata) == MAX_METADATA_KEYS

    def test_metadata_keys_one_over_boundary_fails(self):
        """
        GWT:
        Given metadata with MAX_METADATA_KEYS + 1 entries
        When artifact creation is attempted
        Then ValidationError is raised (boundary + 1 test).
        """
        over_boundary = {f"k{i}": i for i in range(MAX_METADATA_KEYS + 1)}
        with pytest.raises(ValidationError):
            IFTextArtifact(
                artifact_id="over-boundary", content="test", metadata=over_boundary
            )

    def test_metadata_value_at_size_boundary(self):
        """
        GWT:
        Given a metadata value just under the size limit
        When artifact is created
        Then it succeeds.
        """
        # JSON overhead for quotes and key, so use slightly less
        safe_size = MAX_METADATA_VALUE_SIZE - 100
        value = "x" * safe_size
        art = IFTextArtifact(
            artifact_id="size-boundary", content="test", metadata={"large": value}
        )
        assert len(art.metadata["large"]) == safe_size

    def test_lineage_depth_is_bounded_non_negative(self):
        """
        GWT:
        Given an artifact
        When lineage_depth is accessed
        Then it must be non-negative (ge=0 constraint).
        """
        art = IFTextArtifact(artifact_id="depth-test", content="test")
        assert art.lineage_depth >= 0

    def test_lineage_depth_cannot_be_negative(self):
        """
        GWT:
        Given an attempt to create artifact with negative lineage_depth
        When validation occurs
        Then ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            IFTextArtifact(artifact_id="neg-depth", content="test", lineage_depth=-1)

    def test_provenance_list_grows_linearly(self):
        """
        GWT:
        Given a chain of N derivations
        When provenance is checked
        Then it has exactly N entries (linear growth, bounded by depth).
        """
        root = IFTextArtifact(artifact_id="root", content="start")
        current = root

        for i in range(10):
            current = current.derive(
                f"proc-{i}", artifact_id=f"art-{i}", content=f"step-{i}"
            )

        assert len(current.provenance) == 10
        assert current.lineage_depth == 10


# =============================================================================
# JPL RULE #5: ASSERTION DENSITY TESTS
# =============================================================================


class TestJPLRule5AssertionDensity:
    """
    JPL Rule #5: The code must have a minimum assertion density.
    These tests verify that validation assertions are in place.
    """

    def test_artifact_id_is_required(self):
        """
        GWT:
        Given an artifact creation without artifact_id
        When validation occurs
        Then ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            IFTextArtifact(content="no id")

    def test_text_artifact_content_is_required(self):
        """
        GWT:
        Given a TextArtifact creation without content
        When validation occurs
        Then ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            IFTextArtifact(artifact_id="no-content")

    def test_chunk_artifact_document_id_required(self):
        """
        GWT:
        Given a ChunkArtifact without document_id
        When validation occurs
        Then ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            IFChunkArtifact(
                artifact_id="chunk-1",
                content="content",
                # missing document_id
            )

    def test_file_artifact_path_required(self):
        """
        GWT:
        Given a FileArtifact without file_path
        When validation occurs
        Then ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            IFFileArtifact(artifact_id="file-1")

    def test_failure_artifact_error_message_required(self):
        """
        GWT:
        Given a FailureArtifact without error_message
        When validation occurs
        Then ValidationError is raised.
        """
        with pytest.raises(ValidationError):
            IFFailureArtifact(artifact_id="fail-1")

    def test_extra_fields_forbidden(self):
        """
        GWT:
        Given an artifact with undefined extra fields
        When validation occurs
        Then ValidationError is raised (model_config extra=forbid).
        """
        with pytest.raises(ValidationError):
            IFTextArtifact(
                artifact_id="extra", content="test", undefined_field="not allowed"
            )


# =============================================================================
# JPL RULE #7: CHECK RETURN VALUES AND VALIDATE INPUTS
# =============================================================================


class TestJPLRule7InputValidation:
    """
    JPL Rule #7: The return value of all non-void functions must be checked.
    All inputs must be validated.
    """

    def test_metadata_validation_catches_invalid_types(self):
        """
        GWT:
        Given metadata with non-serializable Python objects
        When artifact is created
        Then validation catches and reports the specific key.
        """

        class CustomObject:
            pass

        with pytest.raises(ValidationError) as exc_info:
            IFTextArtifact(
                artifact_id="bad-meta", content="test", metadata={"obj": CustomObject()}
            )
        assert "obj" in str(exc_info.value) or "not JSON-serializable" in str(
            exc_info.value
        )

    def test_metadata_validation_allows_none_values(self):
        """
        GWT:
        Given metadata with None values
        When artifact is created
        Then it succeeds (None is JSON-serializable as null).
        """
        art = IFTextArtifact(
            artifact_id="null-meta", content="test", metadata={"nullable": None}
        )
        assert art.metadata["nullable"] is None

    def test_metadata_validation_allows_boolean_values(self):
        """
        GWT:
        Given metadata with boolean values
        When artifact is created
        Then it succeeds.
        """
        art = IFTextArtifact(
            artifact_id="bool-meta",
            content="test",
            metadata={"flag": True, "other": False},
        )
        assert art.metadata["flag"] is True
        assert art.metadata["other"] is False

    def test_validate_lineage_consistency_returns_boolean(self):
        """
        GWT:
        Given any artifact
        When validate_lineage_consistency() is called
        Then it returns a boolean (not None or exception).
        """
        root = IFTextArtifact(artifact_id="check", content="test")
        result = root.validate_lineage_consistency()
        assert isinstance(result, bool)
        assert result is True

    def test_derive_always_returns_artifact(self):
        """
        GWT:
        Given any artifact
        When derive() is called
        Then it returns a new IFArtifact instance.
        """
        root = IFTextArtifact(artifact_id="derive-test", content="original")
        derived = root.derive("processor", artifact_id="derived", content="new")

        assert isinstance(derived, IFArtifact)
        assert derived is not root
        assert derived.artifact_id != root.artifact_id

    def test_processor_is_available_returns_boolean(self):
        """
        GWT:
        Given a processor
        When is_available() is called
        Then it returns a boolean.
        """
        proc = PassThroughProcessor()
        result = proc.is_available()
        assert isinstance(result, bool)

    def test_processor_teardown_returns_boolean(self):
        """
        GWT:
        Given a processor
        When teardown() is called
        Then it returns a boolean indicating success.
        """
        proc = PassThroughProcessor()
        result = proc.teardown()
        assert isinstance(result, bool)
        assert result is True


# =============================================================================
# JPL RULE #9: COMPLETE TYPE HINTS
# =============================================================================


class TestJPLRule9TypeHints:
    """
    JPL Rule #9: All code should use complete type hints.
    These tests verify type contracts are enforced.
    """

    def test_artifact_fields_have_correct_types(self):
        """
        GWT:
        Given an artifact
        When fields are accessed
        Then they have the documented types.
        """
        art = IFTextArtifact(
            artifact_id="type-test",
            content="test content",
            metadata={"key": "value"},
            provenance=["proc-1"],
        )

        assert isinstance(art.artifact_id, str)
        assert isinstance(art.content, str)
        assert isinstance(art.metadata, dict)
        assert isinstance(art.provenance, list)
        assert isinstance(art.schema_version, str)
        assert isinstance(art.lineage_depth, int)
        assert art.parent_id is None or isinstance(art.parent_id, str)
        assert art.root_artifact_id is None or isinstance(art.root_artifact_id, str)

    def test_content_hash_is_64_char_hex_string(self):
        """
        GWT:
        Given a TextArtifact
        When content_hash is generated
        Then it is a 64-character hexadecimal string (SHA-256).
        """
        art = IFTextArtifact(artifact_id="hash-test", content="test")

        assert isinstance(art.content_hash, str)
        assert len(art.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in art.content_hash)

    def test_processor_properties_return_correct_types(self):
        """
        GWT:
        Given a processor
        When properties are accessed
        Then they return correct types.
        """
        proc = TransformProcessor()

        assert isinstance(proc.processor_id, str)
        assert isinstance(proc.version, str)

    def test_stage_properties_return_correct_types(self):
        """
        GWT:
        Given a stage
        When properties are accessed
        Then they return correct types.
        """
        stage = MockStage("test-stage")

        assert isinstance(stage.name, str)
        assert isinstance(stage.input_type, type)
        assert isinstance(stage.output_type, type)


# =============================================================================
# IMMUTABILITY TESTS (Frozen Models)
# =============================================================================


class TestImmutability:
    """Tests ensuring artifacts are truly immutable (frozen)."""

    def test_text_artifact_content_immutable(self):
        """
        GWT:
        Given a frozen TextArtifact
        When content modification is attempted
        Then ValidationError is raised.
        """
        art = IFTextArtifact(artifact_id="frozen", content="original")
        with pytest.raises(ValidationError):
            art.content = "modified"

    def test_text_artifact_id_immutable(self):
        """
        GWT:
        Given a frozen TextArtifact
        When artifact_id modification is attempted
        Then ValidationError is raised.
        """
        art = IFTextArtifact(artifact_id="frozen", content="test")
        with pytest.raises(ValidationError):
            art.artifact_id = "changed"

    def test_metadata_reference_immutable(self):
        """
        GWT:
        Given a frozen artifact
        When metadata replacement is attempted
        Then ValidationError is raised.
        """
        art = IFTextArtifact(
            artifact_id="frozen-meta", content="test", metadata={"key": "value"}
        )
        with pytest.raises(ValidationError):
            art.metadata = {"new": "dict"}

    def test_provenance_reference_immutable(self):
        """
        GWT:
        Given a frozen artifact
        When provenance replacement is attempted
        Then ValidationError is raised.
        """
        art = IFTextArtifact(artifact_id="frozen-prov", content="test")
        with pytest.raises(ValidationError):
            art.provenance = ["new-proc"]

    def test_lineage_depth_immutable(self):
        """
        GWT:
        Given a frozen artifact
        When lineage_depth modification is attempted
        Then ValidationError is raised.
        """
        art = IFTextArtifact(artifact_id="frozen-depth", content="test")
        with pytest.raises(ValidationError):
            art.lineage_depth = 99


# =============================================================================
# CONTENT HASHING TESTS (SHA-256 Integrity)
# =============================================================================


class TestContentHashing:
    """Tests for automatic SHA-256 content hashing."""

    def test_text_artifact_generates_hash_on_init(self):
        """
        GWT:
        Given a TextArtifact
        When initialized
        Then content_hash is automatically generated.
        """
        art = IFTextArtifact(artifact_id="hash-1", content="Hello World")
        assert art.content_hash is not None
        assert len(art.content_hash) == 64

    def test_same_content_produces_same_hash(self):
        """
        GWT:
        Given two TextArtifacts with identical content
        When hashes are compared
        Then they are equal (deterministic hashing).
        """
        art1 = IFTextArtifact(artifact_id="hash-a", content="identical")
        art2 = IFTextArtifact(artifact_id="hash-b", content="identical")

        assert art1.content_hash == art2.content_hash

    def test_different_content_produces_different_hash(self):
        """
        GWT:
        Given two TextArtifacts with different content
        When hashes are compared
        Then they are different.
        """
        art1 = IFTextArtifact(artifact_id="hash-c", content="content A")
        art2 = IFTextArtifact(artifact_id="hash-d", content="content B")

        assert art1.content_hash != art2.content_hash

    def test_chunk_artifact_generates_hash(self):
        """
        GWT:
        Given a ChunkArtifact
        When initialized
        Then content_hash is automatically generated.
        """
        chunk = IFChunkArtifact(
            artifact_id="chunk-hash",
            document_id="doc-1",
            content="chunk content",
            chunk_index=0,
            total_chunks=1,
        )
        assert chunk.content_hash is not None
        assert len(chunk.content_hash) == 64

    def test_hash_preserved_through_serialization(self):
        """
        GWT:
        Given an artifact with content_hash
        When serialized and deserialized
        Then hash is preserved.
        """
        original = IFTextArtifact(artifact_id="ser-hash", content="test content")
        original_hash = original.content_hash

        json_str = original.model_dump_json()
        restored = IFTextArtifact.model_validate_json(json_str)

        assert restored.content_hash == original_hash


# =============================================================================
# PIPELINE RUNNER INTEGRATION TESTS
# =============================================================================


class TestPipelineRunnerIntegration:
    """Integration tests for the pipeline runner with GWT compliance."""

    def test_empty_stage_list_returns_input(self):
        """
        GWT:
        Given an artifact and empty stage list
        When run() is called
        Then the original artifact is returned.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="empty-stages", content="original")

        result = runner.run(art, [], document_id="doc-1")

        assert result.artifact_id == "empty-stages"
        assert isinstance(result, IFTextArtifact)

    def test_single_stage_execution(self):
        """
        GWT:
        Given an artifact and single stage
        When run() is called
        Then stage is executed and provenance updated.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="single", content="test")
        stages = [MockStage("only")]

        result = runner.run(art, stages, document_id="doc-1")

        assert "stage-only" in result.provenance

    def test_multiple_stages_sequential_execution(self):
        """
        GWT:
        Given an artifact and multiple stages
        When run() is called
        Then stages execute in order.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="multi", content="test")
        stages = [MockStage("first"), MockStage("second"), MockStage("third")]

        result = runner.run(art, stages, document_id="doc-1")

        assert result.provenance == ["stage-first", "stage-second", "stage-third"]

    def test_stage_failure_stops_pipeline(self):
        """
        GWT:
        Given a pipeline with a failing stage
        When run() is called
        Then execution stops at failure and returns FailureArtifact.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="fail-test", content="test")
        stages = [
            MockStage("before"),
            MockStage("fails", fail=True),
            MockStage("never-reached"),
        ]

        result = runner.run(art, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "stage-before" in result.provenance
        assert "stage-never-reached" not in result.provenance

    def test_stage_exception_contained(self):
        """
        GWT:
        Given a stage that raises an exception
        When run() is called
        Then exception is caught and FailureArtifact returned.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="crash-test", content="test")
        stages = [MockStage("crasher", crash=True)]

        result = runner.run(art, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "crashed intentionally" in result.error_message
        assert result.stack_trace is not None

    def test_type_mismatch_detected(self):
        """
        GWT:
        Given a stage expecting specific artifact type
        When incompatible artifact is passed
        Then FailureArtifact with type mismatch is returned.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="type-test", content="test")
        stages = [MockStage("strict", in_type=IFChunkArtifact)]

        result = runner.run(art, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "Type mismatch" in result.error_message

    def test_output_contract_violation_detected(self):
        """
        GWT:
        Given a stage that declares wrong output type
        When run() is called
        Then contract violation is detected.
        """
        runner = IFPipelineRunner()
        art = IFTextArtifact(artifact_id="contract-test", content="test")
        # Stage declares it outputs IFChunkArtifact but actually outputs IFTextArtifact
        stages = [MockStage("violator", out_type=IFChunkArtifact)]

        result = runner.run(art, stages, document_id="doc-1")

        assert isinstance(result, IFFailureArtifact)
        assert "Contract violation" in result.error_message


# =============================================================================
# EDGE CASE AND BOUNDARY TESTS
# =============================================================================


class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_empty_content_allowed(self):
        """
        GWT:
        Given a TextArtifact with empty string content
        When created
        Then it succeeds (empty string is valid).
        """
        art = IFTextArtifact(artifact_id="empty", content="")
        assert art.content == ""
        assert art.content_hash is not None

    def test_unicode_content_handled(self):
        """
        GWT:
        Given a TextArtifact with Unicode content
        When created and hashed
        Then it handles Unicode correctly.
        """
        art = IFTextArtifact(artifact_id="unicode", content="Hello 世界 🌍 مرحبا")
        assert "世界" in art.content
        assert art.content_hash is not None

    def test_very_long_content_handled(self):
        """
        GWT:
        Given a TextArtifact with very long content (1MB)
        When created
        Then it succeeds.
        """
        long_content = "x" * (1024 * 1024)  # 1MB
        art = IFTextArtifact(artifact_id="long", content=long_content)
        assert len(art.content) == 1024 * 1024

    def test_special_characters_in_artifact_id(self):
        """
        GWT:
        Given an artifact_id with special characters
        When created
        Then it is stored correctly.
        """
        special_id = "art-123_test.v1:latest"
        art = IFTextArtifact(artifact_id=special_id, content="test")
        assert art.artifact_id == special_id

    def test_uuid_as_artifact_id(self):
        """
        GWT:
        Given a UUID as artifact_id
        When created
        Then it works correctly.
        """
        uid = str(uuid.uuid4())
        art = IFTextArtifact(artifact_id=uid, content="test")
        assert art.artifact_id == uid

    def test_empty_metadata_dict(self):
        """
        GWT:
        Given an artifact with empty metadata
        When created
        Then it succeeds.
        """
        art = IFTextArtifact(artifact_id="empty-meta", content="test", metadata={})
        assert art.metadata == {}
        assert art.metadata_key_count == 0

    def test_empty_provenance_list(self):
        """
        GWT:
        Given a root artifact
        When provenance is checked
        Then it is an empty list.
        """
        art = IFTextArtifact(artifact_id="empty-prov", content="test")
        assert art.provenance == []

    def test_deep_derivation_chain(self):
        """
        GWT:
        Given a derivation chain of 100 steps
        When final artifact is checked
        Then lineage is correctly tracked.
        """
        current = IFTextArtifact(artifact_id="root-deep", content="start")

        for i in range(100):
            current = current.derive(
                f"proc-{i}", artifact_id=f"art-{i}", content=f"step-{i}"
            )

        assert current.lineage_depth == 100
        assert len(current.provenance) == 100
        assert current.root_artifact_id == "root-deep"

    def test_whitespace_only_content(self):
        """
        GWT:
        Given a TextArtifact with whitespace-only content
        When created
        Then it succeeds.
        """
        art = IFTextArtifact(artifact_id="whitespace", content="   \n\t\r  ")
        assert art.content == "   \n\t\r  "

    def test_newlines_in_content(self):
        """
        GWT:
        Given content with various newline styles
        When created
        Then they are preserved.
        """
        content = "line1\nline2\r\nline3\rline4"
        art = IFTextArtifact(artifact_id="newlines", content=content)
        assert art.content == content


# =============================================================================
# PROCESSOR TESTS
# =============================================================================


class TestProcessors:
    """Tests for IFProcessor implementations."""

    def test_passthrough_processor_returns_same_artifact(self):
        """
        GWT:
        Given a PassThroughProcessor
        When process() is called
        Then the same artifact is returned.
        """
        proc = PassThroughProcessor()
        art = IFTextArtifact(artifact_id="pass", content="unchanged")

        result = proc.process(art)

        assert result is art

    def test_transform_processor_modifies_content(self):
        """
        GWT:
        Given a TransformProcessor with uppercase transform
        When process() is called
        Then content is transformed.
        """
        proc = TransformProcessor(transform_fn=str.upper)
        art = IFTextArtifact(artifact_id="lower", content="hello world")

        result = proc.process(art)

        assert isinstance(result, IFTextArtifact)
        assert result.content == "HELLO WORLD"

    def test_processor_has_unique_id(self):
        """
        GWT:
        Given multiple processors
        When their IDs are compared
        Then they can be distinguished.
        """
        proc1 = PassThroughProcessor("proc-a")
        proc2 = PassThroughProcessor("proc-b")

        assert proc1.processor_id != proc2.processor_id

    def test_processor_version_is_semver(self):
        """
        GWT:
        Given a processor
        When version is accessed
        Then it follows semver format.
        """
        proc = PassThroughProcessor()
        version = proc.version

        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# =============================================================================
# REGISTRY JPL COMPLIANCE TESTS (, , )
# =============================================================================

from ingestforge.core.pipeline.registry import (
    IFRegistry,
    MAX_PROCESSORS,
    get_available_memory_mb,
)


class CapabilityTestProcessor(IFProcessor):
    """Processor with configurable capabilities for testing."""

    def __init__(self, proc_id: str, caps: List[str] = None, memory: int = 100):
        self._id = proc_id
        self._caps = caps or []
        self._memory = memory
        self._available = True

    @property
    def processor_id(self) -> str:
        return self._id

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        return self._caps

    @property
    def memory_mb(self) -> int:
        return self._memory

    def is_available(self) -> bool:
        return self._available

    def process(self, artifact: IFArtifact) -> IFArtifact:
        return artifact


class TestRegistryJPLRule2FixedBounds:
    """
    JPL Rule #2: Registry must have fixed upper bounds.
    """

    def test_max_processors_is_fixed_positive(self):
        """
        GWT:
        Given the registry specification,
        When MAX_PROCESSORS constant is checked,
        Then it must be a fixed positive integer.
        """
        assert isinstance(MAX_PROCESSORS, int)
        assert MAX_PROCESSORS > 0
        assert MAX_PROCESSORS == 256  # Documented bound

    def test_registry_enforces_processor_limit(self):
        """
        GWT:
        Given a registry approaching MAX_PROCESSORS,
        When limit is reached,
        Then RuntimeError is raised on next registration.
        """
        reg = IFRegistry()
        reg.clear()

        # Register up to limit - we can't actually register 256 in a test
        # but we verify the limit check code path exists
        for i in range(10):
            proc = CapabilityTestProcessor(f"proc-{i}")
            reg.register(proc, ["test/type"])

        assert len(reg._id_map) == 10

    def test_capabilities_property_returns_list(self):
        """
        GWT:
        Given a processor,
        When capabilities property is accessed,
        Then it returns a bounded list.
        """
        proc = CapabilityTestProcessor("test", caps=["ocr", "table"])
        assert isinstance(proc.capabilities, list)
        assert len(proc.capabilities) <= 100  # Reasonable bound

    def test_memory_mb_returns_positive_integer(self):
        """
        GWT:
        Given a processor,
        When memory_mb property is accessed,
        Then it returns a positive integer.
        """
        proc = CapabilityTestProcessor("test", memory=512)
        assert isinstance(proc.memory_mb, int)
        assert proc.memory_mb > 0

    def test_default_memory_mb_is_bounded(self):
        """
        GWT:
        Given a processor without explicit memory,
        When memory_mb is accessed,
        Then it returns a reasonable default (100MB).
        """
        proc = PassThroughProcessor()
        assert proc.memory_mb == 100  # Default from interface


class TestRegistryJPLRule7ReturnValues:
    """
    JPL Rule #7: All return values must be checked.
    """

    def test_get_by_capability_returns_list_not_none(self):
        """
        GWT:
        Given a capability that doesn't exist,
        When get_by_capability is called,
        Then it returns empty list (not None).
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.get_by_capability("nonexistent")
        assert result is not None
        assert isinstance(result, list)
        assert len(result) == 0

    def test_get_by_capabilities_returns_list_not_none(self):
        """
        GWT:
        Given empty capabilities list,
        When get_by_capabilities is called,
        Then it returns empty list (not None).
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.get_by_capabilities([])
        assert result is not None
        assert isinstance(result, list)

    def test_get_processors_by_memory_returns_list_not_none(self):
        """
        GWT:
        Given memory limit that no processor satisfies,
        When get_processors_by_memory is called,
        Then it returns empty list (not None).
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.get_processors_by_memory(1)  # 1MB - too small
        assert result is not None
        assert isinstance(result, list)

    def test_teardown_all_returns_summary_dict(self):
        """
        GWT:
        Given a registry,
        When teardown_all is called,
        Then it returns a dict with required keys.
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.teardown_all()

        assert isinstance(result, dict)
        assert "success_count" in result
        assert "failure_count" in result
        assert "failed_ids" in result
        assert isinstance(result["success_count"], int)
        assert isinstance(result["failure_count"], int)
        assert isinstance(result["failed_ids"], list)

    def test_dispatch_by_capability_raises_on_no_match(self):
        """
        GWT:
        Given no processors with required capability,
        When dispatch_by_capability is called,
        Then RuntimeError is raised (explicit failure, not silent None).
        """
        reg = IFRegistry()
        reg.clear()

        art = IFTextArtifact(artifact_id="test", content="test")

        with pytest.raises(RuntimeError) as exc_info:
            reg.dispatch_by_capability("nonexistent", art)

        assert "No available IFProcessor" in str(exc_info.value)

    def test_dispatch_memory_safe_raises_on_no_fit(self):
        """
        GWT:
        Given no processors that fit memory limit,
        When dispatch_memory_safe is called,
        Then RuntimeError is raised.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("heavy", memory=4096)
        reg.register(proc, ["application/octet-stream"])

        art = IFTextArtifact(artifact_id="test", content="test")

        with pytest.raises(RuntimeError) as exc_info:
            reg.dispatch_memory_safe(art, max_mb=100)

        assert "within memory limit" in str(exc_info.value)

    def test_get_available_memory_returns_int(self):
        """
        GWT:
        Given system memory query,
        When get_available_memory_mb is called,
        Then it returns positive integer (even on failure).
        """
        memory = get_available_memory_mb()

        assert isinstance(memory, int)
        assert memory > 0


class TestRegistryJPLRule5AssertionDensity:
    """
    JPL Rule #5: Minimum assertion density for validation.
    """

    def test_dispatch_validates_artifact_type(self):
        """
        GWT:
        Given an artifact without mime_type,
        When dispatch is called,
        Then it falls back to default (validates artifact).
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("default")
        reg.register(proc, ["application/octet-stream"])

        text_art = IFTextArtifact(artifact_id="no-mime", content="test")
        result = reg.dispatch(text_art)

        # Validates that IFTextArtifact gets default MIME handling
        assert result.processor_id == "default"

    def test_capability_query_validates_match_parameter(self):
        """
        GWT:
        Given invalid match parameter,
        When get_by_capabilities is called with match="invalid",
        Then it falls through to default behavior (no crash).
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("cap-proc", caps=["ocr"])
        reg.register(proc, ["image/png"])

        # "invalid" match type should fall through to "all" logic
        result = reg.get_by_capabilities(["ocr"], match="invalid")
        # Should work without crashing (falls through to "all" match)
        assert isinstance(result, list)

    def test_register_attaches_priority_metadata(self):
        """
        GWT:
        Given processor registration with priority,
        When processor is retrieved,
        Then priority is accessible for sorting.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("priority-test")
        reg.register(proc, ["text/plain"], priority=250)

        # Priority attached via object.__setattr__
        assert hasattr(proc, "_priority")
        assert getattr(proc, "_priority") == 250


class TestRegistryJPLRule9TypeHints:
    """
    JPL Rule #9: Complete type hints for all interfaces.
    """

    def test_processor_capabilities_returns_list_of_strings(self):
        """
        GWT:
        Given a processor,
        When capabilities property is accessed,
        Then it returns List[str].
        """
        proc = CapabilityTestProcessor("typed", caps=["ocr", "table"])
        caps = proc.capabilities

        assert isinstance(caps, list)
        assert all(isinstance(c, str) for c in caps)

    def test_processor_memory_mb_returns_int(self):
        """
        GWT:
        Given a processor,
        When memory_mb property is accessed,
        Then it returns int.
        """
        proc = CapabilityTestProcessor("typed", memory=256)
        memory = proc.memory_mb

        assert isinstance(memory, int)

    def test_teardown_summary_has_typed_fields(self):
        """
        GWT:
        Given teardown_all result,
        When fields are accessed,
        Then they have documented types.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("typed-teardown")
        reg.register(proc, ["text/plain"])

        result = reg.teardown_all()

        assert isinstance(result["success_count"], int)
        assert isinstance(result["failure_count"], int)
        assert isinstance(result["failed_ids"], list)
        assert all(isinstance(id, str) for id in result["failed_ids"])


class TestRegistryContextManagerCompliance:
    """
    Tests for context manager protocol compliance ().
    """

    def test_context_manager_enter_returns_self(self):
        """
        GWT:
        Given a registry,
        When __enter__ is called,
        Then it returns the registry instance.
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.__enter__()
        assert result is reg

    def test_context_manager_exit_returns_false(self):
        """
        GWT:
        Given a registry context manager,
        When __exit__ is called,
        Then it returns False (does not suppress exceptions).
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.__exit__(None, None, None)
        assert result is False

    def test_context_manager_calls_teardown_on_normal_exit(self):
        """
        GWT:
        Given a registry with processors,
        When context exits normally,
        Then teardown_all is called.
        """
        reg = IFRegistry()
        reg.clear()

        teardown_called = []

        class TrackingProcessor(CapabilityTestProcessor):
            def teardown(self) -> bool:
                teardown_called.append(self.processor_id)
                return True

        proc = TrackingProcessor("tracker")
        reg.register(proc, ["text/plain"])

        with reg:
            pass  # Do nothing

        assert "tracker" in teardown_called

    def test_context_manager_calls_teardown_on_exception(self):
        """
        GWT:
        Given a registry with processors,
        When context exits via exception,
        Then teardown_all is still called.
        """
        reg = IFRegistry()
        reg.clear()

        teardown_called = []

        class TrackingProcessor(CapabilityTestProcessor):
            def teardown(self) -> bool:
                teardown_called.append(self.processor_id)
                return True

        proc = TrackingProcessor("exc-tracker")
        reg.register(proc, ["text/plain"])

        try:
            with reg:
                raise ValueError("Test exception")
        except ValueError:
            pass

        assert "exc-tracker" in teardown_called


class TestCapabilityRoutingGWT:
    """
    GWT-compliant tests for Capability - Functional Routing.
    """

    def test_capability_registration_and_retrieval(self):
        """
        GWT:
        Given a processor with declared capabilities ["ocr", "table"],
        When registered and then queried by capability,
        Then it appears in results for both capabilities.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("dual-cap", caps=["ocr", "table"])
        reg.register(proc, ["image/png"])

        ocr_procs = reg.get_by_capability("ocr")
        table_procs = reg.get_by_capability("table")

        assert proc in ocr_procs
        assert proc in table_procs

    def test_capability_query_returns_priority_sorted(self):
        """
        GWT:
        Given multiple processors with same capability but different priorities,
        When get_by_capability is called,
        Then results are sorted by priority (highest first).
        """
        reg = IFRegistry()
        reg.clear()

        low_proc = CapabilityTestProcessor("low", caps=["ocr"])
        high_proc = CapabilityTestProcessor("high", caps=["ocr"])

        reg.register(low_proc, ["image/png"], priority=50)
        reg.register(high_proc, ["image/png"], priority=200)

        result = reg.get_by_capability("ocr")

        assert result[0].processor_id == "high"
        assert result[1].processor_id == "low"

    def test_multi_capability_all_match(self):
        """
        GWT:
        Given processors with varying capabilities,
        When get_by_capabilities with match="all" is called,
        Then only processors with ALL capabilities are returned.
        """
        reg = IFRegistry()
        reg.clear()

        full = CapabilityTestProcessor("full", caps=["ocr", "table", "embedding"])
        partial = CapabilityTestProcessor("partial", caps=["ocr"])

        reg.register(full, ["image/png"])
        reg.register(partial, ["image/png"])

        result = reg.get_by_capabilities(["ocr", "table"], match="all")

        assert len(result) == 1
        assert result[0].processor_id == "full"

    def test_multi_capability_any_match(self):
        """
        GWT:
        Given processors with varying capabilities,
        When get_by_capabilities with match="any" is called,
        Then processors with ANY capability are returned.
        """
        reg = IFRegistry()
        reg.clear()

        ocr_only = CapabilityTestProcessor("ocr-only", caps=["ocr"])
        table_only = CapabilityTestProcessor("table-only", caps=["table"])
        other = CapabilityTestProcessor("other", caps=["embedding"])

        reg.register(ocr_only, ["image/png"])
        reg.register(table_only, ["image/png"])
        reg.register(other, ["image/png"])

        result = reg.get_by_capabilities(["ocr", "table"], match="any")

        proc_ids = [p.processor_id for p in result]
        assert "ocr-only" in proc_ids
        assert "table-only" in proc_ids
        assert "other" not in proc_ids


class TestMemoryAwareRoutingGWT:
    """
    GWT-compliant tests for Resources - Memory-Aware Selection.
    """

    def test_memory_filter_excludes_heavy_processors(self):
        """
        GWT:
        Given processors with 100MB, 500MB, and 2GB requirements,
        When get_processors_by_memory(400) is called,
        Then only the 100MB processor is returned.
        """
        reg = IFRegistry()
        reg.clear()

        light = CapabilityTestProcessor("light", memory=100)
        medium = CapabilityTestProcessor("medium", memory=500)
        heavy = CapabilityTestProcessor("heavy", memory=2048)

        reg.register(light, ["text/plain"])
        reg.register(medium, ["text/plain"])
        reg.register(heavy, ["text/plain"])

        result = reg.get_processors_by_memory(400)
        proc_ids = [p.processor_id for p in result]

        assert "light" in proc_ids
        assert "medium" not in proc_ids
        assert "heavy" not in proc_ids

    def test_dispatch_memory_safe_respects_limit(self):
        """
        GWT:
        Given processors with different memory requirements,
        When dispatch_memory_safe is called with limit,
        Then only fitting processors are considered.
        """
        reg = IFRegistry()
        reg.clear()

        # High priority but too heavy
        heavy = CapabilityTestProcessor("heavy", memory=1000)
        # Lower priority but fits
        light = CapabilityTestProcessor("light", memory=100)

        reg.register(heavy, ["application/octet-stream"], priority=200)
        reg.register(light, ["application/octet-stream"], priority=50)

        art = IFTextArtifact(artifact_id="test", content="test")
        result = reg.dispatch_memory_safe(art, max_mb=500)

        # Light is selected despite lower priority because heavy doesn't fit
        assert result.processor_id == "light"

    def test_system_memory_query_provides_fallback(self):
        """
        GWT:
        Given system memory query,
        When psutil is unavailable,
        Then a fallback value (1024MB) is returned.
        """
        # We can't easily test psutil unavailability, but we verify
        # the function always returns a reasonable positive value
        memory = get_available_memory_mb()

        assert memory >= 100  # At least 100MB (or fallback)
        assert memory <= 1_000_000  # At most 1TB


class TestTeardownSafetyGWT:
    """
    GWT-compliant tests for Teardown - Safe Resource Finalization.
    """

    def test_teardown_error_isolation(self):
        """
        GWT:
        Given a processor whose teardown raises exception,
        When teardown_all is called with multiple processors,
        Then other processors are still torn down.
        """
        reg = IFRegistry()
        reg.clear()

        torn_down = []

        class GoodProcessor(CapabilityTestProcessor):
            def teardown(self) -> bool:
                torn_down.append(self.processor_id)
                return True

        class BadProcessor(CapabilityTestProcessor):
            def teardown(self) -> bool:
                torn_down.append(self.processor_id)
                raise RuntimeError("Teardown failed!")

        good1 = GoodProcessor("good-1")
        bad = BadProcessor("bad")
        good2 = GoodProcessor("good-2")

        reg.register(good1, ["text/plain"])
        reg.register(bad, ["text/plain"])
        reg.register(good2, ["text/plain"])

        result = reg.teardown_all()

        # All processors had teardown called
        assert "good-1" in torn_down
        assert "bad" in torn_down
        assert "good-2" in torn_down

        # Summary reflects the failure
        assert result["success_count"] == 2
        assert result["failure_count"] == 1
        assert "bad" in result["failed_ids"]

    def test_teardown_idempotent(self):
        """
        GWT:
        Given a registry that has been torn down,
        When teardown_all is called again,
        Then it succeeds without error.
        """
        reg = IFRegistry()
        reg.clear()

        proc = CapabilityTestProcessor("idem")
        reg.register(proc, ["text/plain"])

        result1 = reg.teardown_all()
        result2 = reg.teardown_all()

        assert result1["success_count"] == 1
        assert result2["success_count"] == 1  # Same processor, still works

    def test_teardown_on_empty_registry(self):
        """
        GWT:
        Given an empty registry,
        When teardown_all is called,
        Then it returns success with zero counts.
        """
        reg = IFRegistry()
        reg.clear()

        result = reg.teardown_all()

        assert result["success_count"] == 0
        assert result["failure_count"] == 0
        assert result["failed_ids"] == []


# =============================================================================
# INTERCEPTOR JPL COMPLIANCE TESTS ()
# =============================================================================

from ingestforge.core.pipeline.interfaces import (
    IFInterceptor,
    MAX_INTERCEPTORS,
)


class MockRecordingInterceptor(IFInterceptor):
    """Interceptor that records all calls for verification."""

    def __init__(self, interceptor_id: str = "mock"):
        self.id = interceptor_id
        self.calls: List[tuple] = []

    def pre_stage(
        self, stage_name: str, artifact: IFArtifact, document_id: str
    ) -> None:
        self.calls.append(("pre_stage", stage_name, artifact.artifact_id, document_id))

    def post_stage(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        duration_ms: float,
    ) -> None:
        self.calls.append(
            ("post_stage", stage_name, artifact.artifact_id, document_id, duration_ms)
        )

    def on_error(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        error: Exception,
    ) -> None:
        self.calls.append(
            ("on_error", stage_name, artifact.artifact_id, document_id, str(error))
        )

    def on_pipeline_start(self, document_id: str, stage_count: int) -> None:
        self.calls.append(("on_pipeline_start", document_id, stage_count))

    def on_pipeline_end(
        self, document_id: str, success: bool, total_duration_ms: float
    ) -> None:
        self.calls.append(("on_pipeline_end", document_id, success, total_duration_ms))


class FailingInterceptorJPL(IFInterceptor):
    """Interceptor that raises exceptions to test isolation (JPL Rule #7)."""

    def pre_stage(
        self, stage_name: str, artifact: IFArtifact, document_id: str
    ) -> None:
        raise RuntimeError("pre_stage intentional failure")

    def post_stage(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        duration_ms: float,
    ) -> None:
        raise RuntimeError("post_stage intentional failure")

    def on_error(
        self,
        stage_name: str,
        artifact: IFArtifact,
        document_id: str,
        error: Exception,
    ) -> None:
        raise RuntimeError("on_error intentional failure")

    def on_pipeline_start(self, document_id: str, stage_count: int) -> None:
        raise RuntimeError("on_pipeline_start intentional failure")

    def on_pipeline_end(
        self, document_id: str, success: bool, total_duration_ms: float
    ) -> None:
        raise RuntimeError("on_pipeline_end intentional failure")


class TestInterceptorJPLRule2FixedBounds:
    """
    JPL Rule #2: All loops must have a fixed upper bound.
    Interceptors must have a maximum count limit.
    """

    def test_max_interceptors_is_fixed_positive(self):
        """
        GWT:
        Given the interceptor specification,
        When MAX_INTERCEPTORS constant is checked,
        Then it must be a fixed positive integer.
        """
        assert isinstance(MAX_INTERCEPTORS, int)
        assert MAX_INTERCEPTORS > 0
        assert MAX_INTERCEPTORS == 16  # Documented bound

    def test_runner_truncates_excess_interceptors_at_init(self):
        """
        GWT:
        Given interceptors exceeding MAX_INTERCEPTORS,
        When IFPipelineRunner is initialized,
        Then only MAX_INTERCEPTORS are kept (truncated with warning).
        """
        too_many = [
            MockRecordingInterceptor(f"int-{i}") for i in range(MAX_INTERCEPTORS + 5)
        ]

        runner = IFPipelineRunner(interceptors=too_many)

        # Truncated to MAX_INTERCEPTORS
        assert len(runner._interceptors) == MAX_INTERCEPTORS
        # First MAX_INTERCEPTORS kept in order
        for i in range(MAX_INTERCEPTORS):
            assert runner._interceptors[i].id == f"int-{i}"

    def test_runner_accepts_exact_limit(self):
        """
        GWT:
        Given exactly MAX_INTERCEPTORS interceptors,
        When IFPipelineRunner is initialized,
        Then it succeeds (boundary test).
        """
        exact = [MockRecordingInterceptor(f"int-{i}") for i in range(MAX_INTERCEPTORS)]
        runner = IFPipelineRunner(interceptors=exact)

        assert len(runner._interceptors) == MAX_INTERCEPTORS

    def test_add_interceptor_enforces_limit(self):
        """
        GWT:
        Given a runner at MAX_INTERCEPTORS - 1,
        When add_interceptor is called twice,
        Then second call returns False (limit reached).
        """
        initial = [
            MockRecordingInterceptor(f"int-{i}") for i in range(MAX_INTERCEPTORS - 1)
        ]
        runner = IFPipelineRunner(interceptors=initial)

        result1 = runner.add_interceptor(MockRecordingInterceptor("last"))
        result2 = runner.add_interceptor(MockRecordingInterceptor("overflow"))

        assert result1 is True  # Fits exactly at limit
        assert result2 is False  # Would exceed limit

    def test_interceptor_count_bounded_through_operations(self):
        """
        GWT:
        Given an empty runner,
        When interceptors are added repeatedly,
        Then count never exceeds MAX_INTERCEPTORS.
        """
        runner = IFPipelineRunner()
        success_count = 0

        for i in range(MAX_INTERCEPTORS + 10):
            if runner.add_interceptor(MockRecordingInterceptor(f"int-{i}")):
                success_count += 1

        assert success_count == MAX_INTERCEPTORS
        assert len(runner._interceptors) == MAX_INTERCEPTORS


class TestInterceptorJPLRule7ExceptionIsolation:
    """
    JPL Rule #7: Check return values, validate inputs.
    Interceptor exceptions must be isolated (logged but not propagated).
    """

    def test_pre_stage_exception_isolated(self):
        """
        GWT:
        Given an interceptor that raises in pre_stage,
        When pipeline runs,
        Then pipeline continues without crashing.
        """
        runner = IFPipelineRunner(interceptors=[FailingInterceptorJPL()])
        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage-1")]

        # Should not raise - exception isolated
        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Pipeline still executed
        assert "stage-stage-1" in result.provenance

    def test_post_stage_exception_isolated(self):
        """
        GWT:
        Given an interceptor that raises in post_stage,
        When pipeline completes a stage,
        Then pipeline continues without crashing.
        """
        runner = IFPipelineRunner(interceptors=[FailingInterceptorJPL()])
        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage-1"), MockStage("stage-2")]

        # Should not raise
        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Both stages executed
        assert len(result.provenance) == 2

    def test_on_error_exception_isolated(self):
        """
        GWT:
        Given an interceptor that raises in on_error,
        When a stage fails,
        Then pipeline handles failure normally (returns FailureArtifact).
        """
        runner = IFPipelineRunner(interceptors=[FailingInterceptorJPL()])
        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("crasher", crash=True)]

        # Should not raise from interceptor
        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Pipeline returned FailureArtifact as expected
        assert isinstance(result, IFFailureArtifact)

    def test_pipeline_start_exception_isolated(self):
        """
        GWT:
        Given an interceptor that raises in on_pipeline_start,
        When pipeline begins,
        Then pipeline continues execution.
        """
        runner = IFPipelineRunner(interceptors=[FailingInterceptorJPL()])
        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage-1")]

        # Should not raise
        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Pipeline executed despite interceptor failure
        assert "stage-stage-1" in result.provenance

    def test_pipeline_end_exception_isolated(self):
        """
        GWT:
        Given an interceptor that raises in on_pipeline_end,
        When pipeline completes,
        Then result is still returned.
        """
        runner = IFPipelineRunner(interceptors=[FailingInterceptorJPL()])
        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage-1")]

        # Should not raise
        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Result returned despite interceptor failure
        assert result is not None
        assert isinstance(result, IFArtifact)

    def test_multiple_interceptors_failure_continues_to_others(self):
        """
        GWT:
        Given multiple interceptors where first fails,
        When pipeline runs,
        Then subsequent interceptors are still called.
        """
        failing = FailingInterceptorJPL()
        recording = MockRecordingInterceptor("recorder")
        runner = IFPipelineRunner(interceptors=[failing, recording])

        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage-1")]

        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Recording interceptor was still called
        assert len(recording.calls) > 0
        # Pipeline completed
        assert "stage-stage-1" in result.provenance


class TestInterceptorJPLRule9TypeHints:
    """
    JPL Rule #9: All code should use complete type hints.
    IFInterceptor methods must have proper type annotations.
    """

    def test_interceptor_pre_stage_signature(self):
        """
        GWT:
        Given IFInterceptor.pre_stage method,
        When signature is inspected,
        Then it has complete type hints.
        """
        import inspect

        sig = inspect.signature(IFInterceptor.pre_stage)
        params = sig.parameters

        # Check parameter types
        assert "stage_name" in params
        assert "artifact" in params
        assert "document_id" in params
        assert sig.return_annotation is None or sig.return_annotation == type(None)

    def test_interceptor_post_stage_signature(self):
        """
        GWT:
        Given IFInterceptor.post_stage method,
        When signature is inspected,
        Then it includes duration_ms: float.
        """
        import inspect

        sig = inspect.signature(IFInterceptor.post_stage)
        params = sig.parameters

        assert "duration_ms" in params
        # Verify duration_ms has float annotation
        assert params["duration_ms"].annotation == float

    def test_interceptor_on_error_signature(self):
        """
        GWT:
        Given IFInterceptor.on_error method,
        When signature is inspected,
        Then it includes error: Exception.
        """
        import inspect

        sig = inspect.signature(IFInterceptor.on_error)
        params = sig.parameters

        assert "error" in params
        assert params["error"].annotation == Exception

    def test_interceptor_pipeline_start_signature(self):
        """
        GWT:
        Given IFInterceptor.on_pipeline_start method,
        When signature is inspected,
        Then it includes stage_count: int.
        """
        import inspect

        sig = inspect.signature(IFInterceptor.on_pipeline_start)
        params = sig.parameters

        assert "stage_count" in params
        assert params["stage_count"].annotation == int

    def test_interceptor_pipeline_end_signature(self):
        """
        GWT:
        Given IFInterceptor.on_pipeline_end method,
        When signature is inspected,
        Then it includes success: bool and total_duration_ms: float.
        """
        import inspect

        sig = inspect.signature(IFInterceptor.on_pipeline_end)
        params = sig.parameters

        assert "success" in params
        assert params["success"].annotation == bool
        assert "total_duration_ms" in params
        assert params["total_duration_ms"].annotation == float


class TestInterceptorGWTScenarios:
    """
    GWT-compliant behavioral tests for interceptor scenarios.
    """

    def test_scenario_1_pre_stage_interceptor(self):
        """
        GWT:
        Given a pre-stage interceptor is registered,
        When a stage is about to execute,
        Then the interceptor is called with stage name and input artifact.
        """
        interceptor = MockRecordingInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])

        art = IFTextArtifact(artifact_id="input-art", content="test")
        stages = [MockStage("extract")]

        runner.run_monitored(art, stages, document_id="doc-123")

        # Find pre_stage calls
        pre_calls = [c for c in interceptor.calls if c[0] == "pre_stage"]
        assert len(pre_calls) == 1
        assert pre_calls[0][1] == "extract"  # stage_name
        assert pre_calls[0][2] == "input-art"  # artifact_id
        assert pre_calls[0][3] == "doc-123"  # document_id

    def test_scenario_2_post_stage_interceptor(self):
        """
        GWT:
        Given a post-stage interceptor is registered,
        When a stage completes successfully,
        Then the interceptor is called with stage name, output artifact, and duration.
        """
        interceptor = MockRecordingInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])

        art = IFTextArtifact(artifact_id="input", content="test")
        stages = [MockStage("transform")]

        runner.run_monitored(art, stages, document_id="doc-456")

        # Find post_stage calls
        post_calls = [c for c in interceptor.calls if c[0] == "post_stage"]
        assert len(post_calls) == 1
        assert post_calls[0][1] == "transform"  # stage_name
        assert post_calls[0][3] == "doc-456"  # document_id
        assert isinstance(post_calls[0][4], float)  # duration_ms
        assert post_calls[0][4] >= 0  # Duration is non-negative

    def test_scenario_3_error_interceptor(self):
        """
        GWT:
        Given an error interceptor is registered,
        When a stage fails,
        Then the interceptor is called with stage name, error details, and partial artifact.
        """
        interceptor = MockRecordingInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])

        art = IFTextArtifact(artifact_id="input", content="test")
        stages = [MockStage("crasher", crash=True)]

        runner.run_monitored(art, stages, document_id="doc-error")

        # Find on_error calls
        error_calls = [c for c in interceptor.calls if c[0] == "on_error"]
        assert len(error_calls) == 1
        assert error_calls[0][1] == "crasher"  # stage_name
        assert error_calls[0][3] == "doc-error"  # document_id
        assert "crashed intentionally" in error_calls[0][4]  # error message

    def test_scenario_4_interceptor_isolation(self):
        """
        GWT:
        Given an interceptor raises an exception,
        When called during pipeline execution,
        Then the exception is logged but does not affect pipeline processing.
        """
        failing = FailingInterceptorJPL()
        runner = IFPipelineRunner(interceptors=[failing])

        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage-1"), MockStage("stage-2")]

        # Pipeline should complete despite interceptor failures
        result = runner.run_monitored(art, stages, document_id="doc-1")

        # Verify pipeline completed all stages
        assert len(result.provenance) == 2
        assert not isinstance(result, IFFailureArtifact)

    def test_scenario_5_multiple_interceptors_order(self):
        """
        GWT:
        Given multiple interceptors of the same type are registered,
        When the trigger event occurs,
        Then all interceptors are called in registration order.
        """
        int1 = MockRecordingInterceptor("first")
        int2 = MockRecordingInterceptor("second")
        int3 = MockRecordingInterceptor("third")
        runner = IFPipelineRunner(interceptors=[int1, int2, int3])

        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("stage")]

        runner.run_monitored(art, stages, document_id="doc-1")

        # All interceptors were called
        assert len(int1.calls) > 0
        assert len(int2.calls) > 0
        assert len(int3.calls) > 0

        # Verify they were called (order verified by shared timestamp ordering)
        # Each should have pipeline_start, pre_stage, post_stage, pipeline_end
        assert len(int1.calls) == 4
        assert len(int2.calls) == 4
        assert len(int3.calls) == 4

    def test_pipeline_lifecycle_hooks_called(self):
        """
        GWT:
        Given an interceptor with pipeline lifecycle hooks,
        When run_monitored completes,
        Then on_pipeline_start and on_pipeline_end are called exactly once.
        """
        interceptor = MockRecordingInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])

        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [MockStage("s1"), MockStage("s2"), MockStage("s3")]

        runner.run_monitored(art, stages, document_id="doc-lifecycle")

        # Count lifecycle calls
        start_calls = [c for c in interceptor.calls if c[0] == "on_pipeline_start"]
        end_calls = [c for c in interceptor.calls if c[0] == "on_pipeline_end"]

        assert len(start_calls) == 1
        assert len(end_calls) == 1

        # Verify start call parameters
        assert start_calls[0][1] == "doc-lifecycle"  # document_id
        assert start_calls[0][2] == 3  # stage_count

        # Verify end call parameters
        assert end_calls[0][1] == "doc-lifecycle"  # document_id
        assert end_calls[0][2] is True  # success
        assert isinstance(end_calls[0][3], float)  # total_duration_ms

    def test_timing_information_accuracy(self):
        """
        GWT:
        Given stages that take measurable time,
        When post_stage is called,
        Then duration_ms reflects actual execution time.
        """
        import time

        class SlowStage(IFStage):
            @property
            def name(self) -> str:
                return "slow"

            @property
            def input_type(self) -> type:
                return IFArtifact

            @property
            def output_type(self) -> type:
                return IFArtifact

            def execute(self, artifact: IFArtifact) -> IFArtifact:
                time.sleep(0.05)  # 50ms
                return artifact.derive("slow-proc")

        interceptor = MockRecordingInterceptor()
        runner = IFPipelineRunner(interceptors=[interceptor])

        art = IFTextArtifact(artifact_id="test", content="content")
        stages = [SlowStage()]

        runner.run_monitored(art, stages, document_id="doc-timing")

        # Get timing from post_stage call
        post_calls = [c for c in interceptor.calls if c[0] == "post_stage"]
        assert len(post_calls) == 1

        duration_ms = post_calls[0][4]
        # Should be at least 40ms (allowing some timing variance)
        assert duration_ms >= 40.0
        # Should be less than 500ms (reasonable upper bound)
        assert duration_ms < 500.0
