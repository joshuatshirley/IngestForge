"""
Tests for Multi-Format Serializers.

Multi-Format Serializers.
Verifies JSON, XML, and CSV serialization with validation.

Test Categories:
- GWT (Given-When-Then) behavioral specifications
- JPL Power of Ten rule compliance verification
- Validation status verification
"""

import csv
import inspect
import io
import json
import xml.etree.ElementTree as ET

import pytest

from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact
from ingestforge.core.pipeline.serializers import (
    IFJSONSerializer,
    IFXMLSerializer,
    IFCSVSerializer,
    SerializationResult,
    serialize_artifacts,
    MAX_ITEMS,
    MAX_FIELD_LENGTH,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def text_artifact():
    """Create a sample IFTextArtifact for testing."""
    return IFTextArtifact(
        artifact_id="test-text-001",
        content="This is sample text content for testing serialization.",
        metadata={"source": "test", "author": "pytest"},
    )


@pytest.fixture
def chunk_artifact():
    """Create a sample IFChunkArtifact for testing."""
    return IFChunkArtifact(
        artifact_id="test-chunk-001",
        document_id="doc-001",
        content="This is a chunk of content from a larger document.",
        chunk_index=0,
        total_chunks=5,
        metadata={"section": "introduction", "page": 1},
    )


@pytest.fixture
def json_serializer():
    """Create a JSON serializer instance."""
    return IFJSONSerializer()


@pytest.fixture
def xml_serializer():
    """Create an XML serializer instance."""
    return IFXMLSerializer()


@pytest.fixture
def csv_serializer():
    """Create a CSV serializer instance."""
    return IFCSVSerializer()


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLRule2FixedUpperBounds:
    """
    JPL Rule #2: All loops must have a fixed upper-bound.

    Tests verify that constants are defined and enforced.
    """

    def test_max_items_constant_defined(self):
        """
        GWT:
        Given the serializers module.
        When MAX_ITEMS is accessed.
        Then it is a positive integer with reasonable bound.
        """
        assert isinstance(MAX_ITEMS, int)
        assert MAX_ITEMS > 0
        assert MAX_ITEMS <= 100000

    def test_max_field_length_constant_defined(self):
        """
        GWT:
        Given the serializers module.
        When MAX_FIELD_LENGTH is accessed.
        Then it is a positive integer with reasonable bound.
        """
        assert isinstance(MAX_FIELD_LENGTH, int)
        assert MAX_FIELD_LENGTH > 0
        assert MAX_FIELD_LENGTH <= 10000000


class TestJPLRule4FunctionLength:
    """
    JPL Rule #4: No function should be longer than 60 lines.
    """

    def test_json_process_method_length(self):
        """
        GWT:
        Given the IFJSONSerializer.process() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(IFJSONSerializer.process)[0]
        assert len(source_lines) < 60, f"process() has {len(source_lines)} lines"

    def test_xml_process_method_length(self):
        """
        GWT:
        Given the IFXMLSerializer.process() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(IFXMLSerializer.process)[0]
        assert len(source_lines) < 60, f"process() has {len(source_lines)} lines"

    def test_csv_process_method_length(self):
        """
        GWT:
        Given the IFCSVSerializer.process() method.
        When its source code is inspected.
        Then it has fewer than 60 lines.
        """
        source_lines = inspect.getsourcelines(IFCSVSerializer.process)[0]
        assert len(source_lines) < 60, f"process() has {len(source_lines)} lines"


class TestJPLRule9TypeSafety:
    """
    JPL Rule #9: Data integrity and type safety.
    """

    def test_serializers_have_processor_id(self):
        """
        GWT:
        Given each serializer class.
        When processor_id is accessed.
        Then a non-empty string is returned.
        """
        assert IFJSONSerializer().processor_id == "json-serializer"
        assert IFXMLSerializer().processor_id == "xml-serializer"
        assert IFCSVSerializer().processor_id == "csv-serializer"

    def test_serializers_have_version(self):
        """
        GWT:
        Given each serializer class.
        When version is accessed.
        Then a valid semver string is returned.
        """
        assert IFJSONSerializer().version == "1.0.0"
        assert IFXMLSerializer().version == "1.0.0"
        assert IFCSVSerializer().version == "1.0.0"

    def test_serializers_have_capabilities(self):
        """
        GWT:
        Given each serializer class.
        When capabilities is accessed.
        Then a list containing 'serialize' is returned.
        """
        assert "serialize" in IFJSONSerializer().capabilities
        assert "serialize" in IFXMLSerializer().capabilities
        assert "serialize" in IFCSVSerializer().capabilities


# =============================================================================
# JSON SERIALIZER TESTS
# =============================================================================


class TestIFJSONSerializer:
    """Test JSON serializer functionality."""

    def test_process_text_artifact(self, json_serializer, text_artifact):
        """
        GWT:
        Given a text artifact.
        When process() is called.
        Then a valid JSON IFTextArtifact is returned.
        """
        result = json_serializer.process(text_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == "test-text-001-json"
        assert result.metadata["serialization_format"] == "json"
        assert result.metadata["validation_status"] is True

    def test_process_chunk_artifact(self, json_serializer, chunk_artifact):
        """
        GWT:
        Given a chunk artifact.
        When process() is called.
        Then a valid JSON IFTextArtifact is returned.
        """
        result = json_serializer.process(chunk_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == "test-chunk-001-json"
        assert result.metadata["validation_status"] is True

    def test_json_output_is_valid(self, json_serializer, text_artifact):
        """
        GWT:
        Given a serialized artifact.
        When the content is parsed as JSON.
        Then no parsing errors occur.
        """
        result = json_serializer.process(text_artifact)
        parsed = json.loads(result.content)

        assert isinstance(parsed, dict)
        assert "artifact_id" in parsed
        assert parsed["artifact_id"] == "test-text-001"

    def test_json_pretty_printing(self, text_artifact):
        """
        GWT:
        Given a JSON serializer with indent=4.
        When process() is called.
        Then output is indented with 4 spaces.
        """
        serializer = IFJSONSerializer(indent=4)
        result = serializer.process(text_artifact)

        assert "    " in result.content  # 4-space indentation

    def test_json_sort_keys(self, text_artifact):
        """
        GWT:
        Given a JSON serializer with sort_keys=True.
        When process() is called.
        Then keys are alphabetically sorted.
        """
        serializer = IFJSONSerializer(sort_keys=True)
        result = serializer.process(text_artifact)
        parsed = json.loads(result.content)

        keys = list(parsed.keys())
        assert keys == sorted(keys)

    def test_json_is_available(self, json_serializer):
        """
        GWT:
        Given a JSON serializer.
        When is_available() is called.
        Then True is returned (json is always available).
        """
        assert json_serializer.is_available() is True

    def test_json_memory_mb(self, json_serializer):
        """
        GWT:
        Given a JSON serializer.
        When memory_mb is accessed.
        Then a reasonable value is returned.
        """
        assert json_serializer.memory_mb == 50


# =============================================================================
# XML SERIALIZER TESTS
# =============================================================================


class TestIFXMLSerializer:
    """Test XML serializer functionality."""

    def test_process_text_artifact(self, xml_serializer, text_artifact):
        """
        GWT:
        Given a text artifact.
        When process() is called.
        Then a valid XML IFTextArtifact is returned.
        """
        result = xml_serializer.process(text_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == "test-text-001-xml"
        assert result.metadata["serialization_format"] == "xml"
        assert result.metadata["validation_status"] is True

    def test_process_chunk_artifact(self, xml_serializer, chunk_artifact):
        """
        GWT:
        Given a chunk artifact.
        When process() is called.
        Then a valid XML IFTextArtifact is returned.
        """
        result = xml_serializer.process(chunk_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == "test-chunk-001-xml"
        assert result.metadata["validation_status"] is True

    def test_xml_output_is_valid(self, xml_serializer, text_artifact):
        """
        GWT:
        Given a serialized artifact.
        When the content is parsed as XML.
        Then no parsing errors occur.
        """
        result = xml_serializer.process(text_artifact)
        root = ET.fromstring(result.content)

        assert root.tag == "artifact"

    def test_xml_custom_root_tag(self, text_artifact):
        """
        GWT:
        Given an XML serializer with custom root_tag.
        When process() is called.
        Then the output uses the custom root tag.
        """
        serializer = IFXMLSerializer(root_tag="document")
        result = serializer.process(text_artifact)
        root = ET.fromstring(result.content)

        assert root.tag == "document"

    def test_xml_tag_sanitization(self):
        """
        GWT:
        Given a field name with invalid XML characters.
        When _sanitize_tag() is called.
        Then the tag is sanitized to valid XML.
        """
        serializer = IFXMLSerializer()

        assert serializer._sanitize_tag("123field") == "_123field"
        assert serializer._sanitize_tag("field-name") == "field-name"
        assert serializer._sanitize_tag("field name") == "field_name"
        assert serializer._sanitize_tag("") == "item"

    def test_xml_is_available(self, xml_serializer):
        """
        GWT:
        Given an XML serializer.
        When is_available() is called.
        Then True is returned.
        """
        assert xml_serializer.is_available() is True


# =============================================================================
# CSV SERIALIZER TESTS
# =============================================================================


class TestIFCSVSerializer:
    """Test CSV serializer functionality."""

    def test_process_text_artifact(self, csv_serializer, text_artifact):
        """
        GWT:
        Given a text artifact.
        When process() is called.
        Then a valid CSV IFTextArtifact is returned.
        """
        result = csv_serializer.process(text_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == "test-text-001-csv"
        assert result.metadata["serialization_format"] == "csv"
        assert result.metadata["validation_status"] is True

    def test_process_chunk_artifact(self, csv_serializer, chunk_artifact):
        """
        GWT:
        Given a chunk artifact.
        When process() is called.
        Then a valid CSV IFTextArtifact is returned.
        """
        result = csv_serializer.process(chunk_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == "test-chunk-001-csv"
        assert result.metadata["validation_status"] is True

    def test_csv_output_is_valid(self, csv_serializer, text_artifact):
        """
        GWT:
        Given a serialized artifact.
        When the content is parsed as CSV.
        Then no parsing errors occur.
        """
        result = csv_serializer.process(text_artifact)
        reader = csv.DictReader(io.StringIO(result.content))
        rows = list(reader)

        assert len(rows) == 1
        assert "artifact_id" in rows[0]

    def test_csv_header_included(self, text_artifact):
        """
        GWT:
        Given a CSV serializer with include_header=True.
        When process() is called.
        Then output includes header row.
        """
        serializer = IFCSVSerializer(include_header=True)
        result = serializer.process(text_artifact)

        lines = result.content.strip().split("\n")
        assert len(lines) == 2  # header + data row

    def test_csv_no_header(self, text_artifact):
        """
        GWT:
        Given a CSV serializer with include_header=False.
        When process() is called.
        Then output has no header row.
        """
        serializer = IFCSVSerializer(include_header=False)
        result = serializer.process(text_artifact)

        lines = result.content.strip().split("\n")
        assert len(lines) == 1  # data row only

    def test_csv_custom_delimiter(self, text_artifact):
        """
        GWT:
        Given a CSV serializer with custom delimiter.
        When process() is called.
        Then output uses the custom delimiter.
        """
        serializer = IFCSVSerializer(delimiter=";")
        result = serializer.process(text_artifact)

        assert ";" in result.content

    def test_csv_is_available(self, csv_serializer):
        """
        GWT:
        Given a CSV serializer.
        When is_available() is called.
        Then True is returned.
        """
        assert csv_serializer.is_available() is True


# =============================================================================
# VALIDATION STATUS TESTS
# =============================================================================


class TestValidationStatus:
    """Test validation_status flag in serializer output."""

    def test_json_validation_status_valid(self, json_serializer, text_artifact):
        """
        GWT:
        Given a valid artifact.
        When JSON serialization succeeds.
        Then validation_status is True.
        """
        result = json_serializer.process(text_artifact)
        assert result.metadata["validation_status"] is True

    def test_xml_validation_status_valid(self, xml_serializer, text_artifact):
        """
        GWT:
        Given a valid artifact.
        When XML serialization succeeds.
        Then validation_status is True.
        """
        result = xml_serializer.process(text_artifact)
        assert result.metadata["validation_status"] is True

    def test_csv_validation_status_valid(self, csv_serializer, text_artifact):
        """
        GWT:
        Given a valid artifact.
        When CSV serialization succeeds.
        Then validation_status is True.
        """
        result = csv_serializer.process(text_artifact)
        assert result.metadata["validation_status"] is True


# =============================================================================
# SERIALIZATION RESULT TESTS
# =============================================================================


class TestSerializationResult:
    """Test SerializationResult dataclass."""

    def test_serialization_result_creation(self):
        """
        GWT:
        Given valid parameters.
        When SerializationResult is created.
        Then all fields are set correctly.
        """
        result = SerializationResult(
            content='{"test": true}',
            format_type="json",
            item_count=1,
            validation_status=True,
            validation_errors=[],
        )

        assert result.content == '{"test": true}'
        assert result.format_type == "json"
        assert result.item_count == 1
        assert result.validation_status is True
        assert result.validation_errors == []

    def test_serialization_result_with_errors(self):
        """
        GWT:
        Given validation errors.
        When SerializationResult is created.
        Then errors are captured.
        """
        result = SerializationResult(
            content="",
            format_type="json",
            item_count=0,
            validation_status=False,
            validation_errors=["Parse error", "Invalid data"],
        )

        assert result.validation_status is False
        assert len(result.validation_errors) == 2


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestSerializeArtifacts:
    """Test serialize_artifacts convenience function."""

    def test_serialize_single_artifact_json(self, text_artifact):
        """
        GWT:
        Given a single artifact and format_type="json".
        When serialize_artifacts() is called.
        Then a JSON IFTextArtifact is returned.
        """
        result = serialize_artifacts([text_artifact], format_type="json")

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["serialization_format"] == "json"

    def test_serialize_single_artifact_xml(self, text_artifact):
        """
        GWT:
        Given a single artifact and format_type="xml".
        When serialize_artifacts() is called.
        Then an XML IFTextArtifact is returned.
        """
        result = serialize_artifacts([text_artifact], format_type="xml")

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["serialization_format"] == "xml"

    def test_serialize_single_artifact_csv(self, text_artifact):
        """
        GWT:
        Given a single artifact and format_type="csv".
        When serialize_artifacts() is called.
        Then a CSV IFTextArtifact is returned.
        """
        result = serialize_artifacts([text_artifact], format_type="csv")

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["serialization_format"] == "csv"

    def test_serialize_empty_list_raises(self):
        """
        GWT:
        Given an empty artifact list.
        When serialize_artifacts() is called.
        Then ValueError is raised.
        """
        with pytest.raises(ValueError, match="No artifacts"):
            serialize_artifacts([])

    def test_serialize_unsupported_format_raises(self, text_artifact):
        """
        GWT:
        Given an unsupported format_type.
        When serialize_artifacts() is called.
        Then ValueError is raised.
        """
        with pytest.raises(ValueError, match="Unsupported format"):
            serialize_artifacts([text_artifact], format_type="yaml")


# =============================================================================
# LINEAGE AND PROVENANCE TESTS
# =============================================================================


class TestLineageTracking:
    """Test that serializers properly track lineage."""

    def test_json_updates_provenance(self, json_serializer, text_artifact):
        """
        GWT:
        Given an artifact with empty provenance.
        When JSON serialization occurs.
        Then the serializer ID is added to provenance.
        """
        result = json_serializer.process(text_artifact)

        assert "json-serializer" in result.provenance

    def test_json_increments_lineage_depth(self, json_serializer, text_artifact):
        """
        GWT:
        Given an artifact with lineage_depth=0.
        When JSON serialization occurs.
        Then lineage_depth is incremented to 1.
        """
        result = json_serializer.process(text_artifact)

        assert result.lineage_depth == text_artifact.lineage_depth + 1

    def test_json_sets_parent_id(self, json_serializer, text_artifact):
        """
        GWT:
        Given an artifact.
        When JSON serialization occurs.
        Then parent_id is set to original artifact ID.
        """
        result = json_serializer.process(text_artifact)

        assert result.parent_id == text_artifact.artifact_id
