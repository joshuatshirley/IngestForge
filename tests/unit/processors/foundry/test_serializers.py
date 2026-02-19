"""
Tests for Multi-Format Serializers.

Verifies JSON, CSV, and XML serialization of IFEntityArtifacts.

Test Categories:
- GWT (Given-When-Then) behavioral specifications
- JPL Power of Ten rule compliance
- Error handling and edge cases
"""

import csv
import inspect
import io
import json
import xml.etree.ElementTree as ET
from typing import get_type_hints

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFEntityArtifact,
    IFTextArtifact,
    IFFailureArtifact,
    IFChunkArtifact,
    EntityNode,
    RelationshipEdge,
    SourceProvenance,
)
from ingestforge.processors.foundry.serializers import (
    IFJSONSerializer,
    IFJSONLDSerializer,
    IFCSVSerializer,
    IFXMLSerializer,
    _escape_xml_text,
    _sanitize_xml_tag,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_provenance():
    """Create sample SourceProvenance for testing."""
    return SourceProvenance(
        source_artifact_id="source-artifact-001",
        char_offset_start=0,
        char_offset_end=100,
        confidence=0.95,
        extraction_method="test",
    )


@pytest.fixture
def sample_nodes(sample_provenance):
    """Create sample EntityNode list for testing."""
    return [
        EntityNode(
            entity_id="e1",
            entity_type="PERSON",
            name="Alice Smith",
            aliases=["A. Smith", "Alice"],
            confidence=0.95,
            source_provenance=sample_provenance,
            properties={"role": "engineer"},
        ),
        EntityNode(
            entity_id="e2",
            entity_type="ORG",
            name="Acme Corp",
            aliases=["Acme"],
            confidence=0.88,
            source_provenance=sample_provenance,
            properties={"industry": "tech"},
        ),
        EntityNode(
            entity_id="e3",
            entity_type="GPE",
            name="New York",
            aliases=[],
            confidence=0.99,
            source_provenance=sample_provenance,
            properties={},
        ),
    ]


@pytest.fixture
def sample_edges():
    """Create sample RelationshipEdge list for testing."""
    return [
        RelationshipEdge(
            source_entity_id="e1",
            target_entity_id="e2",
            predicate="WORKS_FOR",
            confidence=0.85,
        ),
        RelationshipEdge(
            source_entity_id="e2",
            target_entity_id="e3",
            predicate="LOCATED_IN",
            confidence=0.92,
        ),
    ]


@pytest.fixture
def entity_artifact(sample_nodes, sample_edges):
    """Create a sample IFEntityArtifact for testing."""
    return IFEntityArtifact(
        artifact_id="test-entity-123",
        nodes=sample_nodes,
        edges=sample_edges,
        extraction_model="test_model",
        source_document_id="doc-456",
    )


@pytest.fixture
def empty_entity_artifact():
    """Create an empty IFEntityArtifact for testing."""
    return IFEntityArtifact(
        artifact_id="empty-entity-789",
        nodes=[],
        edges=[],
        extraction_model="test_model",
    )


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestHelperFunctions:
    """Test helper functions for XML processing."""

    def test_escape_xml_text_basic(self):
        """Test XML escaping for special characters."""
        assert _escape_xml_text("hello") == "hello"
        assert _escape_xml_text("a & b") == "a &amp; b"
        assert _escape_xml_text("<tag>") == "&lt;tag&gt;"
        assert _escape_xml_text('"quoted"') == "&quot;quoted&quot;"

    def test_sanitize_xml_tag_valid(self):
        """Test XML tag sanitization for valid names."""
        assert _sanitize_xml_tag("entity") == "entity"
        assert _sanitize_xml_tag("my_tag") == "my_tag"
        assert _sanitize_xml_tag("tag-name") == "tag-name"

    def test_sanitize_xml_tag_invalid_chars(self):
        """Test XML tag sanitization removes invalid characters."""
        assert _sanitize_xml_tag("my tag") == "my_tag"
        assert _sanitize_xml_tag("my@tag") == "my_tag"
        assert _sanitize_xml_tag("123tag") == "_123tag"


# =============================================================================
# JSON SERIALIZER TESTS
# =============================================================================


class TestIFJSONSerializer:
    """Test IFJSONSerializer."""

    def test_process_basic(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with nodes and edges.
        When IFJSONSerializer.process() is called.
        Then it returns an IFTextArtifact with valid JSON.
        """
        serializer = IFJSONSerializer()
        result = serializer.process(entity_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["format"] == "json"

        # Verify JSON is valid and parseable
        data = json.loads(result.content)
        assert data["node_count"] == 3
        assert data["edge_count"] == 2
        assert len(data["nodes"]) == 3
        assert len(data["edges"]) == 2

    def test_process_empty_artifact(self, empty_entity_artifact):
        """
        GWT:
        Given an empty IFEntityArtifact.
        When IFJSONSerializer.process() is called.
        Then it returns valid JSON with empty arrays.
        """
        serializer = IFJSONSerializer()
        result = serializer.process(empty_entity_artifact)

        assert isinstance(result, IFTextArtifact)
        data = json.loads(result.content)
        assert data["nodes"] == []
        assert data["edges"] == []

    def test_process_wrong_artifact_type(self):
        """
        GWT:
        Given a non-IFEntityArtifact (e.g., IFChunkArtifact).
        When IFJSONSerializer.process() is called.
        Then it returns an IFFailureArtifact.
        """
        chunk = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="test content",
        )
        serializer = IFJSONSerializer()
        result = serializer.process(chunk)

        assert isinstance(result, IFFailureArtifact)
        assert "Expected IFEntityArtifact" in result.error_message

    def test_process_preserves_lineage(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact.
        When IFJSONSerializer.process() is called.
        Then the result preserves parent_id and lineage.
        """
        serializer = IFJSONSerializer()
        result = serializer.process(entity_artifact)

        assert result.parent_id == entity_artifact.artifact_id
        assert serializer.processor_id in result.provenance

    def test_is_available(self):
        """JSON serialization should always be available."""
        serializer = IFJSONSerializer()
        assert serializer.is_available() is True

    def test_teardown(self):
        """Teardown should succeed."""
        serializer = IFJSONSerializer()
        assert serializer.teardown() is True

    def test_indent_option(self, entity_artifact):
        """Test custom indentation."""
        serializer = IFJSONSerializer(indent=4)
        result = serializer.process(entity_artifact)

        # With indent=4, should have deeper indentation
        assert "    " in result.content

    def test_method_length_under_60_lines(self):
        """JPL Rule #4: process() method under 60 lines."""
        source = inspect.getsourcelines(IFJSONSerializer.process)[0]
        assert len(source) < 60, f"process() has {len(source)} lines"


# =============================================================================
# JSON-LD SERIALIZER TESTS (AC#2: Schema.org Compliance)
# =============================================================================


class TestIFJSONLDSerializer:
    """Test IFJSONLDSerializer for Schema.org compliant JSON-LD output."""

    def test_process_basic(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with nodes and edges.
        When IFJSONLDSerializer.process() is called.
        Then it returns an IFTextArtifact with valid JSON-LD.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["format"] == "jsonld"

        # Verify JSON-LD is valid and parseable
        data = json.loads(result.content)
        assert "@context" in data
        assert "@type" in data
        assert "itemListElement" in data
        assert len(data["itemListElement"]) == 3

    def test_process_schema_org_context(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact.
        When IFJSONLDSerializer.process() is called.
        Then the @context includes Schema.org vocabulary.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        data = json.loads(result.content)
        context = data["@context"]

        assert "@vocab" in context
        assert "schema.org" in context["@vocab"]
        assert "if" in context  # Custom namespace for IngestForge

    def test_process_entity_type_mapping(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with PERSON and ORG types.
        When IFJSONLDSerializer.process() is called.
        Then types are mapped to Schema.org types.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        data = json.loads(result.content)
        items = data["itemListElement"]

        # Find PERSON entity
        person_item = next(i for i in items if i["name"] == "Alice Smith")
        assert person_item["@type"] == "Person"

        # Find ORG entity
        org_item = next(i for i in items if i["name"] == "Acme Corp")
        assert org_item["@type"] == "Organization"

    def test_process_entity_id_as_urn(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact.
        When IFJSONLDSerializer.process() is called.
        Then each entity has a URN-formatted @id.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        data = json.loads(result.content)
        items = data["itemListElement"]

        for item in items:
            assert "@id" in item
            assert item["@id"].startswith("urn:ingestforge:entity:")

    def test_process_aliases_as_alternate_name(self, entity_artifact):
        """
        GWT:
        Given an entity with aliases.
        When IFJSONLDSerializer.process() is called.
        Then aliases are mapped to alternateName (Schema.org property).
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        data = json.loads(result.content)
        items = data["itemListElement"]

        # Find entity with aliases
        person_item = next(i for i in items if i["name"] == "Alice Smith")
        assert "alternateName" in person_item
        assert "A. Smith" in person_item["alternateName"]
        assert "Alice" in person_item["alternateName"]

    def test_process_relationships(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with edges.
        When IFJSONLDSerializer.process() is called.
        Then relationships are included in the output.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        data = json.loads(result.content)
        assert "if:relationships" in data
        relationships = data["if:relationships"]

        assert len(relationships) == 2
        first_rel = relationships[0]
        assert "if:source" in first_rel
        assert "if:target" in first_rel
        assert "if:predicate" in first_rel

    def test_process_empty_artifact(self, empty_entity_artifact):
        """
        GWT:
        Given an empty IFEntityArtifact.
        When IFJSONLDSerializer.process() is called.
        Then it returns valid JSON-LD with empty itemListElement.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(empty_entity_artifact)

        assert isinstance(result, IFTextArtifact)
        data = json.loads(result.content)
        assert data["itemListElement"] == []
        assert data["numberOfItems"] == 0

    def test_process_wrong_artifact_type(self):
        """
        GWT:
        Given a non-IFEntityArtifact (e.g., IFChunkArtifact).
        When IFJSONLDSerializer.process() is called.
        Then it returns an IFFailureArtifact.
        """
        chunk = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="test content",
        )
        serializer = IFJSONLDSerializer()
        result = serializer.process(chunk)

        assert isinstance(result, IFFailureArtifact)
        assert "Expected IFEntityArtifact" in result.error_message

    def test_process_preserves_lineage(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact.
        When IFJSONLDSerializer.process() is called.
        Then the result preserves parent_id and lineage.
        """
        serializer = IFJSONLDSerializer()
        result = serializer.process(entity_artifact)

        assert result.parent_id == entity_artifact.artifact_id
        assert serializer.processor_id in result.provenance

    def test_is_available(self):
        """JSON-LD serialization should always be available."""
        serializer = IFJSONLDSerializer()
        assert serializer.is_available() is True

    def test_teardown(self):
        """Teardown should succeed."""
        serializer = IFJSONLDSerializer()
        assert serializer.teardown() is True

    def test_custom_context_url(self, entity_artifact):
        """Test custom context URL."""
        serializer = IFJSONLDSerializer(context_url="https://example.org/context")
        result = serializer.process(entity_artifact)

        data = json.loads(result.content)
        assert "example.org" in data["@context"]["@vocab"]

    def test_indent_option(self, entity_artifact):
        """Test custom indentation."""
        serializer = IFJSONLDSerializer(indent=4)
        result = serializer.process(entity_artifact)

        # With indent=4, should have deeper indentation
        assert "    " in result.content

    def test_unmapped_type_defaults_to_thing(self, sample_provenance):
        """
        GWT:
        Given an entity with unmapped type (e.g., GPE).
        When IFJSONLDSerializer.process() is called.
        Then the type defaults to Schema.org "Thing".
        """
        node = EntityNode(
            entity_id="g1",
            entity_type="GPE",  # Not in mapping
            name="Unknown Place",
            confidence=0.9,
            source_provenance=sample_provenance,
        )
        artifact = IFEntityArtifact(
            artifact_id="unmapped-test",
            nodes=[node],
            edges=[],
            extraction_model="test",
        )

        serializer = IFJSONLDSerializer()
        result = serializer.process(artifact)

        data = json.loads(result.content)
        assert data["itemListElement"][0]["@type"] == "Thing"

    def test_method_length_under_60_lines(self):
        """JPL Rule #4: process() method under 60 lines."""
        source = inspect.getsourcelines(IFJSONLDSerializer.process)[0]
        assert len(source) < 60, f"process() has {len(source)} lines"

    def test_rule5_assertion_null_artifact(self):
        """JPL Rule #5: Assert artifact is not None."""
        serializer = IFJSONLDSerializer()
        with pytest.raises(AssertionError):
            serializer.process(None)


# =============================================================================
# CSV SERIALIZER TESTS
# =============================================================================


class TestIFCSVSerializer:
    """Test IFCSVSerializer."""

    def test_process_basic(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with nodes.
        When IFCSVSerializer.process() is called.
        Then it returns an IFTextArtifact with valid Excel-compatible CSV.
        """
        serializer = IFCSVSerializer()
        result = serializer.process(entity_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["format"] == "csv"

        # Parse CSV and verify
        reader = csv.reader(io.StringIO(result.content))
        rows = list(reader)

        # Header + 3 data rows
        assert len(rows) == 4
        assert rows[0] == ["entity_id", "entity_type", "name", "aliases", "confidence"]
        assert rows[1][0] == "e1"
        assert rows[1][2] == "Alice Smith"

    def test_process_with_aliases(self, entity_artifact):
        """
        GWT:
        Given an entity with multiple aliases.
        When IFCSVSerializer.process() is called.
        Then aliases are pipe-delimited.
        """
        serializer = IFCSVSerializer()
        result = serializer.process(entity_artifact)

        reader = csv.reader(io.StringIO(result.content))
        rows = list(reader)

        # First data row should have pipe-delimited aliases
        assert "A. Smith|Alice" in rows[1][3]

    def test_process_no_header(self, entity_artifact):
        """
        GWT:
        Given include_header=False.
        When IFCSVSerializer.process() is called.
        Then CSV has no header row.
        """
        serializer = IFCSVSerializer(include_header=False)
        result = serializer.process(entity_artifact)

        reader = csv.reader(io.StringIO(result.content))
        rows = list(reader)

        # Should have only data rows
        assert len(rows) == 3
        assert rows[0][0] == "e1"

    def test_process_empty_artifact(self, empty_entity_artifact):
        """
        GWT:
        Given an empty IFEntityArtifact.
        When IFCSVSerializer.process() is called.
        Then it returns CSV with only header.
        """
        serializer = IFCSVSerializer()
        result = serializer.process(empty_entity_artifact)

        reader = csv.reader(io.StringIO(result.content))
        rows = list(reader)

        assert len(rows) == 1  # Header only

    def test_process_wrong_artifact_type(self):
        """
        GWT:
        Given a non-IFEntityArtifact.
        When IFCSVSerializer.process() is called.
        Then it returns an IFFailureArtifact.
        """
        chunk = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="test content",
        )
        serializer = IFCSVSerializer()
        result = serializer.process(chunk)

        assert isinstance(result, IFFailureArtifact)

    def test_custom_delimiter(self, entity_artifact):
        """Test custom delimiter (tab)."""
        serializer = IFCSVSerializer(delimiter="\t")
        result = serializer.process(entity_artifact)

        reader = csv.reader(io.StringIO(result.content), delimiter="\t")
        rows = list(reader)

        assert len(rows) == 4

    def test_is_available(self):
        """CSV serialization should always be available."""
        serializer = IFCSVSerializer()
        assert serializer.is_available() is True

    def test_method_length_under_60_lines(self):
        """JPL Rule #4: process() method under 60 lines."""
        source = inspect.getsourcelines(IFCSVSerializer.process)[0]
        assert len(source) < 60, f"process() has {len(source)} lines"


# =============================================================================
# XML SERIALIZER TESTS
# =============================================================================


class TestIFXMLSerializer:
    """Test IFXMLSerializer."""

    def test_process_basic(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with nodes and edges.
        When IFXMLSerializer.process() is called.
        Then it returns an IFTextArtifact with valid XML.
        """
        serializer = IFXMLSerializer()
        result = serializer.process(entity_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["format"] == "xml"

        # Verify XML is valid and parseable
        root = ET.fromstring(
            result.content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        )
        assert root.tag == "entities"
        assert root.get("node_count") == "3"
        assert root.get("edge_count") == "2"

    def test_process_nodes_structure(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact.
        When IFXMLSerializer.process() is called.
        Then nodes are properly structured in XML.
        """
        serializer = IFXMLSerializer()
        result = serializer.process(entity_artifact)

        root = ET.fromstring(
            result.content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        )
        nodes_elem = root.find("nodes")

        assert nodes_elem is not None
        entities = nodes_elem.findall("entity")
        assert len(entities) == 3

        # Check first entity
        first = entities[0]
        assert first.get("id") == "e1"
        assert first.get("type") == "PERSON"
        assert first.find("name").text == "Alice Smith"

    def test_process_edges_structure(self, entity_artifact):
        """
        GWT:
        Given an IFEntityArtifact with edges.
        When IFXMLSerializer.process() is called.
        Then relationships are properly structured.
        """
        serializer = IFXMLSerializer()
        result = serializer.process(entity_artifact)

        root = ET.fromstring(
            result.content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        )
        rels_elem = root.find("relationships")

        assert rels_elem is not None
        rels = rels_elem.findall("relationship")
        assert len(rels) == 2

        first_rel = rels[0]
        assert first_rel.get("source") == "e1"
        assert first_rel.get("target") == "e2"
        assert first_rel.get("predicate") == "WORKS_FOR"

    def test_process_empty_artifact(self, empty_entity_artifact):
        """
        GWT:
        Given an empty IFEntityArtifact.
        When IFXMLSerializer.process() is called.
        Then it returns valid XML with empty nodes/relationships.
        """
        serializer = IFXMLSerializer()
        result = serializer.process(empty_entity_artifact)

        assert isinstance(result, IFTextArtifact)
        root = ET.fromstring(
            result.content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        )
        assert len(root.find("nodes").findall("entity")) == 0

    def test_process_wrong_artifact_type(self):
        """
        GWT:
        Given a non-IFEntityArtifact.
        When IFXMLSerializer.process() is called.
        Then it returns an IFFailureArtifact.
        """
        chunk = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="test content",
        )
        serializer = IFXMLSerializer()
        result = serializer.process(chunk)

        assert isinstance(result, IFFailureArtifact)

    def test_custom_root_tag(self, entity_artifact):
        """Test custom root tag name."""
        serializer = IFXMLSerializer(root_tag="knowledge_graph")
        result = serializer.process(entity_artifact)

        root = ET.fromstring(
            result.content.replace('<?xml version="1.0" encoding="UTF-8"?>\n', "")
        )
        assert root.tag == "knowledge_graph"

    def test_xml_declaration(self, entity_artifact):
        """Test XML declaration is present."""
        serializer = IFXMLSerializer()
        result = serializer.process(entity_artifact)

        assert result.content.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_is_available(self):
        """XML serialization should always be available."""
        serializer = IFXMLSerializer()
        assert serializer.is_available() is True

    def test_method_length_under_60_lines(self):
        """JPL Rule #4: process() method under 60 lines."""
        source = inspect.getsourcelines(IFXMLSerializer.process)[0]
        assert len(source) < 60, f"process() has {len(source)} lines"


# =============================================================================
# JPL RULE COMPLIANCE TESTS
# =============================================================================


class TestJPLRuleCompliance:
    """Test JPL Power of Ten rule compliance across all serializers."""

    def test_rule5_assertions_in_process(self, entity_artifact):
        """
        JPL Rule #5: Assertions should be used liberally.
        All process methods should assert artifact is not None.
        """
        # This is implicitly tested by the implementation
        # If assertions fail, tests would error
        for SerializerClass in [
            IFJSONSerializer,
            IFJSONLDSerializer,
            IFCSVSerializer,
            IFXMLSerializer,
        ]:
            serializer = SerializerClass()
            with pytest.raises(AssertionError):
                serializer.process(None)

    def test_rule7_return_values_checked(self, entity_artifact):
        """
        JPL Rule #7: Check return values of all formatting operations.
        Serializers should return valid artifacts or failures.
        """
        for SerializerClass in [
            IFJSONSerializer,
            IFJSONLDSerializer,
            IFCSVSerializer,
            IFXMLSerializer,
        ]:
            serializer = SerializerClass()
            result = serializer.process(entity_artifact)

            # Result must be an artifact type
            assert isinstance(result, (IFTextArtifact, IFFailureArtifact))

    def test_rule9_type_hints(self):
        """
        JPL Rule #9: Complete type hints.
        All public methods should have type hints.
        """
        for SerializerClass in [
            IFJSONSerializer,
            IFJSONLDSerializer,
            IFCSVSerializer,
            IFXMLSerializer,
        ]:
            hints = get_type_hints(SerializerClass.process)
            assert "return" in hints
            assert "artifact" in hints


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for serializers."""

    def test_utf8_encoding(self, sample_edges, sample_provenance):
        """
        GWT:
        Given entities with Unicode characters.
        When serializers process them.
        Then output is valid UTF-8.
        """
        unicode_node = EntityNode(
            entity_id="u1",
            entity_type="PERSON",
            name="日本語テスト",
            aliases=["Ñoño", "Müller"],
            confidence=0.9,
            source_provenance=sample_provenance,
        )
        artifact = IFEntityArtifact(
            artifact_id="unicode-test",
            nodes=[unicode_node],
            edges=[],
            extraction_model="test",
        )

        for SerializerClass in [
            IFJSONSerializer,
            IFJSONLDSerializer,
            IFCSVSerializer,
            IFXMLSerializer,
        ]:
            serializer = SerializerClass()
            result = serializer.process(artifact)

            assert isinstance(result, IFTextArtifact)
            # Should be able to encode to UTF-8
            encoded = result.content.encode("utf-8")
            assert len(encoded) > 0

    def test_special_characters_csv(self, sample_provenance):
        """
        GWT:
        Given entities with CSV-problematic characters.
        When IFCSVSerializer processes them.
        Then output is properly escaped.
        """
        problem_node = EntityNode(
            entity_id="p1",
            entity_type="TEXT",
            name='Text with "quotes" and, commas',
            aliases=["Line\nbreak"],
            confidence=0.9,
            source_provenance=sample_provenance,
        )
        artifact = IFEntityArtifact(
            artifact_id="csv-escape-test",
            nodes=[problem_node],
            edges=[],
            extraction_model="test",
        )

        serializer = IFCSVSerializer()
        result = serializer.process(artifact)

        # Should parse without error
        reader = csv.reader(io.StringIO(result.content))
        rows = list(reader)
        assert len(rows) == 2  # Header + data

    def test_special_characters_xml(self, sample_provenance):
        """
        GWT:
        Given entities with XML-problematic characters.
        When IFXMLSerializer processes them.
        Then output is properly escaped.
        """
        problem_node = EntityNode(
            entity_id="x1",
            entity_type="TEXT",
            name="Text with <brackets> & ampersand",
            aliases=[],
            confidence=0.9,
            source_provenance=sample_provenance,
        )
        artifact = IFEntityArtifact(
            artifact_id="xml-escape-test",
            nodes=[problem_node],
            edges=[],
            extraction_model="test",
        )

        serializer = IFXMLSerializer()
        result = serializer.process(artifact)

        # Should parse without error
        xml_content = result.content.replace(
            '<?xml version="1.0" encoding="UTF-8"?>\n', ""
        )
        root = ET.fromstring(xml_content)
        assert root is not None
