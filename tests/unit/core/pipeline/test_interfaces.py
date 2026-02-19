import pytest
from typing import Any
from pydantic import ValidationError
from ingestforge.core.pipeline.interfaces import (
    IFArtifact,
    IFProcessor,
    MAX_METADATA_KEYS,
    MAX_METADATA_VALUE_SIZE,
)


def test_if_artifact_immutability():
    """
    GWT:
    Given a concrete implementation of IFArtifact
    When an attribute is modified after initialization
    Then a ValidationError (FrozenInstanceError) must be raised.
    """

    class ConcreteArtifact(IFArtifact):
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

    art = ConcreteArtifact(artifact_id="test-123")

    with pytest.raises(ValidationError):
        # Pydantic v2 raises ValidationError for frozen instances
        art.artifact_id = "changed"


def test_if_processor_interface():
    """
    GWT:
    Given a class that inherits from IFProcessor
    When it doesn't implement abstract methods
    Then it cannot be instantiated.
    """

    class IncompleteProcessor(IFProcessor):
        pass

    with pytest.raises(TypeError):
        IncompleteProcessor()


def test_if_processor_implementation():
    """
    GWT:
    Given a correctly implemented IFProcessor
    When process() is called
    Then it returns an IFArtifact.
    """

    class MockArtifact(IFArtifact):
        def derive(self, processor_id: str, **kwargs: Any) -> "MockArtifact":
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

    class MockProcessor(IFProcessor):
        @property
        def processor_id(self) -> str:
            return "mock-proc"

        @property
        def version(self) -> str:
            return "1.0.0"

        def is_available(self) -> bool:
            return True

        def process(self, artifact: IFArtifact) -> IFArtifact:
            return artifact

    proc = MockProcessor()
    art = MockArtifact(artifact_id="1")
    result = proc.process(art)
    assert result.artifact_id == "1"
    assert isinstance(result, IFArtifact)


# Helper for metadata tests
def _create_test_artifact(**kwargs: Any) -> IFArtifact:
    """Create a concrete artifact for testing."""

    class TestArtifact(IFArtifact):
        def derive(self, processor_id: str, **kw: Any) -> "TestArtifact":
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
                    **kw,
                }
            )

    return TestArtifact(**kwargs)


# Serialization - Validated Metadata Tests


def test_metadata_accepts_valid_dict():
    """
    GWT:
    Given valid metadata with JSON-serializable values
    When artifact is created
    Then metadata is stored successfully.
    """
    art = _create_test_artifact(
        artifact_id="test-1",
        metadata={"key1": "value1", "key2": 123, "nested": {"a": [1, 2, 3]}},
    )
    assert art.metadata["key1"] == "value1"
    assert art.metadata["key2"] == 123
    assert art.metadata["nested"]["a"] == [1, 2, 3]


def test_metadata_rejects_exceeding_max_keys():
    """
    GWT:
    Given metadata with more than MAX_METADATA_KEYS entries
    When artifact creation is attempted
    Then ValidationError is raised.
    """
    too_many_keys = {f"key_{i}": i for i in range(MAX_METADATA_KEYS + 1)}

    with pytest.raises(ValidationError) as exc_info:
        _create_test_artifact(artifact_id="test-1", metadata=too_many_keys)

    assert "exceeds maximum" in str(exc_info.value).lower()


def test_metadata_accepts_exactly_max_keys():
    """
    GWT:
    Given metadata with exactly MAX_METADATA_KEYS entries
    When artifact is created
    Then it succeeds.
    """
    exact_keys = {f"key_{i}": i for i in range(MAX_METADATA_KEYS)}
    art = _create_test_artifact(artifact_id="test-1", metadata=exact_keys)
    assert len(art.metadata) == MAX_METADATA_KEYS


def test_metadata_rejects_non_serializable_values():
    """
    GWT:
    Given metadata with non-JSON-serializable values
    When artifact creation is attempted
    Then ValidationError is raised.
    """

    class CustomObject:
        pass

    with pytest.raises(ValidationError) as exc_info:
        _create_test_artifact(
            artifact_id="test-1", metadata={"bad_value": CustomObject()}
        )

    assert "not json-serializable" in str(exc_info.value).lower()


def test_metadata_rejects_oversized_values():
    """
    GWT:
    Given metadata with a value exceeding MAX_METADATA_VALUE_SIZE
    When artifact creation is attempted
    Then ValidationError is raised.
    """
    # Create a string that exceeds the limit when JSON-serialized
    oversized_value = "x" * (MAX_METADATA_VALUE_SIZE + 100)

    with pytest.raises(ValidationError) as exc_info:
        _create_test_artifact(
            artifact_id="test-1", metadata={"oversized": oversized_value}
        )

    assert "exceeds" in str(exc_info.value).lower()


def test_metadata_to_json_serialization():
    """
    GWT:
    Given an artifact with metadata
    When metadata_to_json() is called
    Then a valid JSON string is returned with sorted keys.
    """
    art = _create_test_artifact(
        artifact_id="test-1", metadata={"zebra": 1, "alpha": 2, "beta": 3}
    )
    json_str = art.metadata_to_json()

    # Keys should be sorted
    assert json_str == '{"alpha": 2, "beta": 3, "zebra": 1}'


def test_metadata_key_count_property():
    """
    GWT:
    Given an artifact with metadata
    When metadata_key_count is accessed
    Then correct count is returned.
    """
    art = _create_test_artifact(artifact_id="test-1", metadata={"a": 1, "b": 2, "c": 3})
    assert art.metadata_key_count == 3


def test_can_add_metadata_keys_property():
    """
    GWT:
    Given an artifact with some metadata
    When can_add_metadata_keys is accessed
    Then remaining capacity is returned.
    """
    art = _create_test_artifact(artifact_id="test-1", metadata={"a": 1, "b": 2})
    assert art.can_add_metadata_keys == MAX_METADATA_KEYS - 2


def test_empty_metadata_is_valid():
    """
    GWT:
    Given an artifact with empty metadata
    When created
    Then it succeeds with empty dict.
    """
    art = _create_test_artifact(artifact_id="test-1")
    assert art.metadata == {}
    assert art.metadata_key_count == 0
    assert art.can_add_metadata_keys == MAX_METADATA_KEYS
