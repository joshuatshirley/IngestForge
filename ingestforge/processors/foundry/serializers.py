"""
Multi-Format Serializers for IngestForge Foundry.

Transforms IFEntityArtifacts into standard string formats (JSON, CSV, XML).
Follows NASA JPL Power of Ten rules.
"""

import csv
import io
import json
import logging
import re
import uuid
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from ingestforge.core.pipeline.artifacts import (
    IFEntityArtifact,
    IFTextArtifact,
    IFFailureArtifact,
    EntityNode,
    RelationshipEdge,
)
from ingestforge.core.pipeline.interfaces import IFArtifact, IFProcessor

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_NODES_FOR_EXPORT = 10000
MAX_EDGES_FOR_EXPORT = 50000
MAX_OUTPUT_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB


def _create_failure(
    processor_id: str, parent_id: str, message: str
) -> IFFailureArtifact:
    """Create a failure artifact. Rule #1: Extracted helper."""
    return IFFailureArtifact(
        artifact_id=str(uuid.uuid4()),
        error_message=message,
        failed_processor_id=processor_id,
        parent_id=parent_id,
    )


def _create_text_result(
    content: str, artifact: IFEntityArtifact, processor_id: str, fmt: str
) -> IFTextArtifact:
    """Create successful text artifact result. Rule #1: Extracted helper."""
    return IFTextArtifact(
        artifact_id=str(uuid.uuid4()),
        content=content,
        parent_id=artifact.artifact_id,
        provenance=artifact.provenance + [processor_id],
        root_artifact_id=artifact.effective_root_id,
        lineage_depth=artifact.lineage_depth + 1,
        metadata={"format": fmt, "source_nodes": artifact.node_count},
    )


def _escape_xml_text(text: str) -> str:
    """
    Escape special XML characters in text content.

    Rule #4: Function < 60 lines.
    Rule #7: Check return value is valid.
    """
    assert text is not None, "text cannot be None"
    # Replace XML special characters
    result = text.replace("&", "&amp;")
    result = result.replace("<", "&lt;")
    result = result.replace(">", "&gt;")
    result = result.replace('"', "&quot;")
    result = result.replace("'", "&apos;")
    return result


def _sanitize_xml_tag(name: str) -> str:
    """
    Sanitize a string to be a valid XML tag name.

    Rule #4: Function < 60 lines.
    Rule #7: Check return value is valid.
    """
    assert name, "name cannot be empty"
    # Replace invalid characters with underscore
    sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "_", name)
    # Ensure starts with letter or underscore
    if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
        sanitized = "_" + sanitized
    return sanitized or "_entity"


class IFJSONSerializer(IFProcessor):
    """
    Serializer that converts IFEntityArtifact to pretty-printed JSON.

    Multi-Format Serializers.
    Rule #4: Methods < 60 lines.
    Rule #7: Check return values of all formatting operations.
    Rule #9: Complete type hints.
    """

    def __init__(self, indent: int = 2, ensure_ascii: bool = False) -> None:
        """
        Initialize the JSON serializer.

        Args:
            indent: Number of spaces for indentation (default: 2).
            ensure_ascii: If True, escape non-ASCII characters.
        """
        assert indent >= 0, "indent must be non-negative"
        self._indent = indent
        self._ensure_ascii = ensure_ascii
        self._processor_id = f"IFJSONSerializer_{uuid.uuid4().hex[:8]}"
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Return the unique processor ID."""
        return self._processor_id

    @property
    def version(self) -> str:
        """Return the processor version."""
        return self._version

    def _serialize_node(self, node: EntityNode) -> Dict[str, Any]:
        """Serialize a single EntityNode to dict. Rule #4: < 60 lines."""
        return {
            "entity_id": node.entity_id,
            "entity_type": node.entity_type,
            "name": node.name,
            "aliases": list(node.aliases) if node.aliases else [],
            "confidence": node.confidence,
            "properties": dict(node.properties) if node.properties else {},
        }

    def _serialize_edge(self, edge: RelationshipEdge) -> Dict[str, Any]:
        """Serialize a single RelationshipEdge to dict. Rule #4: < 60 lines."""
        return {
            "source_entity_id": edge.source_entity_id,
            "target_entity_id": edge.target_entity_id,
            "predicate": edge.predicate,
            "confidence": edge.confidence,
            "properties": dict(edge.properties) if edge.properties else {},
        }

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Convert IFEntityArtifact to pretty-printed JSON text artifact.

        Rule #4: Function < 60 lines. Rule #5: Assert preconditions.
        """
        assert artifact is not None, "artifact cannot be None"

        if not isinstance(artifact, IFEntityArtifact):
            msg = f"Expected IFEntityArtifact, got {type(artifact).__name__}"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        if len(artifact.nodes) > MAX_NODES_FOR_EXPORT:
            msg = (
                f"Too many nodes ({len(artifact.nodes)}), max is {MAX_NODES_FOR_EXPORT}"
            )
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        output = {
            "source_artifact_id": artifact.artifact_id,
            "extraction_model": artifact.extraction_model,
            "node_count": artifact.node_count,
            "edge_count": artifact.edge_count,
            "nodes": [self._serialize_node(n) for n in artifact.nodes],
            "edges": [self._serialize_edge(e) for e in artifact.edges],
        }

        try:
            json_str = json.dumps(
                output, indent=self._indent, ensure_ascii=self._ensure_ascii
            )
        except (TypeError, ValueError) as e:
            return _create_failure(
                self._processor_id,
                artifact.artifact_id,
                f"JSON serialization failed: {e}",
            )

        if len(json_str.encode("utf-8")) > MAX_OUTPUT_SIZE_BYTES:
            msg = f"Output exceeds max size of {MAX_OUTPUT_SIZE_BYTES} bytes"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        logger.info(f"Serialized {artifact.node_count} nodes to JSON")
        return _create_text_result(json_str, artifact, self._processor_id, "json")

    def is_available(self) -> bool:
        """JSON serialization is always available."""
        return True

    def teardown(self) -> bool:
        """No resources to clean up."""
        return True


class IFCSVSerializer(IFProcessor):
    """
    Serializer that converts IFEntityArtifact nodes to properly escaped CSV.

    Multi-Format Serializers.
    Rule #4: Methods < 60 lines.
    Rule #7: Check return values of all formatting operations.
    """

    def __init__(self, delimiter: str = ",", include_header: bool = True) -> None:
        """
        Initialize the CSV serializer.

        Args:
            delimiter: CSV field delimiter (default: comma).
            include_header: Whether to include header row.
        """
        assert delimiter, "delimiter cannot be empty"
        self._delimiter = delimiter
        self._include_header = include_header
        self._processor_id = f"IFCSVSerializer_{uuid.uuid4().hex[:8]}"
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Return the unique processor ID."""
        return self._processor_id

    @property
    def version(self) -> str:
        """Return the processor version."""
        return self._version

    def _get_csv_headers(self) -> List[str]:
        """Return standard CSV headers for entity nodes."""
        return [
            "entity_id",
            "entity_type",
            "name",
            "aliases",
            "confidence",
        ]

    def _node_to_row(self, node: EntityNode) -> List[str]:
        """Convert EntityNode to CSV row values. Rule #4: < 60 lines."""
        aliases_str = "|".join(node.aliases) if node.aliases else ""
        return [
            node.entity_id,
            node.entity_type,
            node.name,
            aliases_str,
            f"{node.confidence:.4f}",
        ]

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Convert IFEntityArtifact nodes to Excel-compatible CSV.

        Rule #4: Function < 60 lines. Rule #5: Assert preconditions.
        """
        assert artifact is not None, "artifact cannot be None"

        if not isinstance(artifact, IFEntityArtifact):
            msg = f"Expected IFEntityArtifact, got {type(artifact).__name__}"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        if len(artifact.nodes) > MAX_NODES_FOR_EXPORT:
            msg = (
                f"Too many nodes ({len(artifact.nodes)}), max is {MAX_NODES_FOR_EXPORT}"
            )
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        try:
            output = io.StringIO()
            writer = csv.writer(
                output, delimiter=self._delimiter, quoting=csv.QUOTE_ALL
            )
            if self._include_header:
                writer.writerow(self._get_csv_headers())
            for node in artifact.nodes:
                writer.writerow(self._node_to_row(node))
            csv_str = output.getvalue()
        except Exception as e:
            return _create_failure(
                self._processor_id,
                artifact.artifact_id,
                f"CSV serialization failed: {e}",
            )

        if len(csv_str.encode("utf-8")) > MAX_OUTPUT_SIZE_BYTES:
            msg = f"Output exceeds max size of {MAX_OUTPUT_SIZE_BYTES} bytes"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        logger.info(f"Serialized {artifact.node_count} nodes to CSV")
        return _create_text_result(csv_str, artifact, self._processor_id, "csv")

    def is_available(self) -> bool:
        """CSV serialization is always available."""
        return True

    def teardown(self) -> bool:
        """No resources to clean up."""
        return True


class IFJSONLDSerializer(IFProcessor):
    """
    Serializer that converts IFEntityArtifact to Schema.org compliant JSON-LD.

    Multi-Format Serializers - Schema.org Compliance.
    Rule #4: Methods < 60 lines.
    Rule #7: Check return values of all formatting operations.
    Rule #9: Complete type hints.
    """

    # Schema.org type mappings for common entity types
    SCHEMA_ORG_TYPE_MAP: Dict[str, str] = {
        "PERSON": "Person",
        "ORG": "Organization",
        "ORGANIZATION": "Organization",
        "LOC": "Place",
        "LOCATION": "Place",
        "DATE": "Date",
        "EVENT": "Event",
        "PRODUCT": "Product",
        "WORK": "CreativeWork",
        "MONEY": "MonetaryAmount",
        "URL": "URL",
        "EMAIL": "ContactPoint",
    }

    def __init__(
        self, context_url: str = "https://schema.org", indent: int = 2
    ) -> None:
        """
        Initialize the JSON-LD serializer.

        Args:
            context_url: URL for the JSON-LD @context (default: Schema.org).
            indent: Number of spaces for indentation (default: 2).
        """
        assert context_url, "context_url cannot be empty"
        assert indent >= 0, "indent must be non-negative"
        self._context_url = context_url
        self._indent = indent
        self._processor_id = f"IFJSONLDSerializer_{uuid.uuid4().hex[:8]}"
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Return the unique processor ID."""
        return self._processor_id

    @property
    def version(self) -> str:
        """Return the processor version."""
        return self._version

    def _map_entity_type(self, entity_type: str) -> str:
        """Map internal entity type to Schema.org type. Rule #4: < 60 lines."""
        normalized = entity_type.upper()
        return self.SCHEMA_ORG_TYPE_MAP.get(normalized, "Thing")

    def _node_to_jsonld(self, node: EntityNode) -> Dict[str, Any]:
        """
        Convert EntityNode to JSON-LD object with Schema.org types.

        Rule #4: Function < 60 lines.
        Rule #7: Check return value is valid dict.
        """
        schema_type = self._map_entity_type(node.entity_type)

        jsonld_obj: Dict[str, Any] = {
            "@type": schema_type,
            "@id": f"urn:ingestforge:entity:{node.entity_id}",
            "name": node.name,
            "identifier": node.entity_id,
        }

        # Add aliases as alternateName (Schema.org property)
        if node.aliases:
            jsonld_obj["alternateName"] = list(node.aliases)

        # Add confidence as custom property with namespace
        jsonld_obj["if:confidence"] = node.confidence

        # Map properties to Schema.org where possible
        if node.properties:
            for key, value in node.properties.items():
                # Prefix non-standard properties with custom namespace
                prop_key = key if key in ("name", "url", "email") else f"if:{key}"
                jsonld_obj[prop_key] = value

        return jsonld_obj

    def _edge_to_jsonld(self, edge: RelationshipEdge) -> Dict[str, Any]:
        """
        Convert RelationshipEdge to JSON-LD relationship object.

        Rule #4: Function < 60 lines.
        """
        return {
            "@type": "if:Relationship",
            "if:source": {"@id": f"urn:ingestforge:entity:{edge.source_entity_id}"},
            "if:target": {"@id": f"urn:ingestforge:entity:{edge.target_entity_id}"},
            "if:predicate": edge.predicate,
            "if:confidence": edge.confidence,
        }

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Convert IFEntityArtifact to Schema.org compliant JSON-LD.

        Rule #4: Function < 60 lines. Rule #5: Assert preconditions.
        Rule #7: Check return values of serialization calls.
        """
        assert artifact is not None, "artifact cannot be None"

        if not isinstance(artifact, IFEntityArtifact):
            msg = f"Expected IFEntityArtifact, got {type(artifact).__name__}"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        if len(artifact.nodes) > MAX_NODES_FOR_EXPORT:
            msg = (
                f"Too many nodes ({len(artifact.nodes)}), max is {MAX_NODES_FOR_EXPORT}"
            )
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        # Build JSON-LD document with Schema.org context
        jsonld_doc: Dict[str, Any] = {
            "@context": {
                "@vocab": self._context_url + "/",
                "if": "urn:ingestforge:schema:",
            },
            "@type": "ItemList",
            "@id": f"urn:ingestforge:extraction:{artifact.artifact_id}",
            "numberOfItems": artifact.node_count,
            "if:edgeCount": artifact.edge_count,
            "if:extractionModel": artifact.extraction_model,
            "itemListElement": [self._node_to_jsonld(n) for n in artifact.nodes],
        }

        # Add relationships as separate graph
        if artifact.edges:
            jsonld_doc["if:relationships"] = [
                self._edge_to_jsonld(e) for e in artifact.edges
            ]

        try:
            jsonld_str = json.dumps(jsonld_doc, indent=self._indent, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            return _create_failure(
                self._processor_id,
                artifact.artifact_id,
                f"JSON-LD serialization failed: {e}",
            )

        if len(jsonld_str.encode("utf-8")) > MAX_OUTPUT_SIZE_BYTES:
            msg = f"Output exceeds max size of {MAX_OUTPUT_SIZE_BYTES} bytes"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        logger.info(f"Serialized {artifact.node_count} nodes to JSON-LD (Schema.org)")
        return _create_text_result(jsonld_str, artifact, self._processor_id, "jsonld")

    def is_available(self) -> bool:
        """JSON-LD serialization is always available (uses json module)."""
        return True

    def teardown(self) -> bool:
        """No resources to clean up."""
        return True


class IFXMLSerializer(IFProcessor):
    """
    Serializer that converts IFEntityArtifact to tag-safe XML.

    Multi-Format Serializers.
    Rule #4: Methods < 60 lines.
    Rule #7: Check return values of all formatting operations.
    """

    def __init__(self, root_tag: str = "entities", indent: bool = True) -> None:
        """
        Initialize the XML serializer.

        Args:
            root_tag: Name of the root XML element.
            indent: Whether to pretty-print the XML output.
        """
        assert root_tag, "root_tag cannot be empty"
        self._root_tag = _sanitize_xml_tag(root_tag)
        self._indent = indent
        self._processor_id = f"IFXMLSerializer_{uuid.uuid4().hex[:8]}"
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        """Return the unique processor ID."""
        return self._processor_id

    @property
    def version(self) -> str:
        """Return the processor version."""
        return self._version

    def _node_to_element(self, node: EntityNode) -> ET.Element:
        """Convert EntityNode to XML Element. Rule #4: < 60 lines."""
        elem = ET.Element("entity")
        elem.set("id", node.entity_id)
        elem.set("type", node.entity_type)

        name_elem = ET.SubElement(elem, "name")
        name_elem.text = node.name

        if node.aliases:
            aliases_elem = ET.SubElement(elem, "aliases")
            for alias in node.aliases:
                alias_elem = ET.SubElement(aliases_elem, "alias")
                alias_elem.text = alias

        conf_elem = ET.SubElement(elem, "confidence")
        conf_elem.text = f"{node.confidence:.4f}"

        return elem

    def _edge_to_element(self, edge: RelationshipEdge) -> ET.Element:
        """Convert RelationshipEdge to XML Element. Rule #4: < 60 lines."""
        elem = ET.Element("relationship")
        elem.set("source", edge.source_entity_id)
        elem.set("target", edge.target_entity_id)
        elem.set("predicate", edge.predicate)
        elem.set("confidence", f"{edge.confidence:.4f}")
        return elem

    def _indent_xml(self, elem: ET.Element, level: int = 0) -> None:
        """
        Add indentation to XML element tree using iterative approach.

        Rule #1: No recursion - uses explicit stack.
        Rule #2: Bounded by MAX_NODES_FOR_EXPORT.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.
        """
        assert elem is not None, "elem cannot be None"
        assert level >= 0, "level must be non-negative"

        # JPL Rule #1: Use stack instead of recursion
        # Stack entries: (element, level, phase) where phase=0 means pre-visit, phase=1 means post-visit
        stack: list = [(elem, level, 0)]
        iterations = 0

        while stack:
            iterations += 1
            assert (
                iterations <= MAX_NODES_FOR_EXPORT * 2
            ), "Indentation iteration limit exceeded"

            current, lvl, phase = stack.pop()
            indent_str = "\n" + "  " * lvl

            if phase == 0:  # Pre-visit: set text and push children
                if len(current):
                    if not current.text or not current.text.strip():
                        current.text = indent_str + "  "
                    if not current.tail or not current.tail.strip():
                        current.tail = indent_str
                    # Push post-visit for this element, then children in reverse
                    stack.append((current, lvl, 1))
                    for child in reversed(list(current)):
                        stack.append((child, lvl + 1, 0))
                else:
                    if lvl and (not current.tail or not current.tail.strip()):
                        current.tail = indent_str
            else:  # Post-visit: fix last child's tail
                children = list(current)
                if children:
                    last_child = children[-1]
                    if not last_child.tail or not last_child.tail.strip():
                        last_child.tail = indent_str

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Convert IFEntityArtifact to tag-safe XML.

        Rule #4: Function < 60 lines. Rule #5: Assert preconditions.
        """
        assert artifact is not None, "artifact cannot be None"

        if not isinstance(artifact, IFEntityArtifact):
            msg = f"Expected IFEntityArtifact, got {type(artifact).__name__}"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        if len(artifact.nodes) > MAX_NODES_FOR_EXPORT:
            msg = (
                f"Too many nodes ({len(artifact.nodes)}), max is {MAX_NODES_FOR_EXPORT}"
            )
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        try:
            root = ET.Element(self._root_tag)
            root.set("source_artifact", artifact.artifact_id)
            root.set("node_count", str(artifact.node_count))
            root.set("edge_count", str(artifact.edge_count))
            nodes_elem = ET.SubElement(root, "nodes")
            for node in artifact.nodes:
                nodes_elem.append(self._node_to_element(node))
            edges_elem = ET.SubElement(root, "relationships")
            for edge in artifact.edges:
                edges_elem.append(self._edge_to_element(edge))
            if self._indent:
                self._indent_xml(root)
            xml_str = '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(
                root, encoding="unicode"
            )
        except Exception as e:
            return _create_failure(
                self._processor_id,
                artifact.artifact_id,
                f"XML serialization failed: {e}",
            )

        if len(xml_str.encode("utf-8")) > MAX_OUTPUT_SIZE_BYTES:
            msg = f"Output exceeds max size of {MAX_OUTPUT_SIZE_BYTES} bytes"
            return _create_failure(self._processor_id, artifact.artifact_id, msg)

        logger.info(f"Serialized {artifact.node_count} nodes to XML")
        return _create_text_result(xml_str, artifact, self._processor_id, "xml")

    def is_available(self) -> bool:
        """XML serialization is always available."""
        return True

    def teardown(self) -> bool:
        """No resources to clean up."""
        return True
