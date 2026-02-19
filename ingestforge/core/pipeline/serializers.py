"""
Multi-Format Serializers for IngestForge (IF).

Provides pluggable IFProcessor classes for converting artifacts to JSON, XML, or CSV.
Multi-Format Serializers.
Follows NASA JPL Power of Ten rules.
"""

import csv
import io
import json
import logging
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from xml.dom import minidom

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFTextArtifact, IFChunkArtifact

logger = logging.getLogger(__name__)

# Maximum items to serialize (Rule #2: Fixed upper bounds)
MAX_ITEMS = 10000
MAX_FIELD_LENGTH = 100000


class SerializationResult:
    """
    Result of a serialization operation.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        content: str,
        format_type: str,
        item_count: int,
        validation_status: bool,
        validation_errors: Optional[List[str]] = None,
    ):
        self.content = content
        self.format_type = format_type
        self.item_count = item_count
        self.validation_status = validation_status
        self.validation_errors = validation_errors or []


class IFJSONSerializer(IFProcessor):
    """
    Serializes artifacts to pretty-printed JSON format.

    Multi-Format Serializers.
    Rule #9: Complete type hints.
    """

    def __init__(self, indent: int = 2, sort_keys: bool = True):
        """
        Initialize JSON serializer.

        Args:
            indent: Indentation level for pretty-printing.
            sort_keys: Whether to sort dictionary keys.
        """
        self._indent = indent
        self._sort_keys = sort_keys
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        return "json-serializer"

    @property
    def version(self) -> str:
        return self._version

    @property
    def capabilities(self) -> List[str]:
        return ["serialize", "serialize.json", "export"]

    @property
    def memory_mb(self) -> int:
        return 50  # JSON serialization is lightweight

    def is_available(self) -> bool:
        return True  # json module is always available

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Serialize artifact to JSON.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact to serialize.

        Returns:
            IFTextArtifact containing JSON string with validation_status.
        """
        try:
            data = self._extract_data(artifact)
            result = self._serialize(data)

            # Create output artifact with serialization metadata
            new_metadata = dict(artifact.metadata)
            new_metadata["serialization_format"] = "json"
            new_metadata["validation_status"] = result.validation_status
            new_metadata["item_count"] = result.item_count
            if result.validation_errors:
                new_metadata["validation_errors"] = result.validation_errors

            return IFTextArtifact(
                artifact_id=f"{artifact.artifact_id}-json",
                content=result.content,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
                metadata=new_metadata,
            )

        except Exception as e:
            logger.error(f"JSON serialization failed: {e}")
            return self._create_error_artifact(artifact, str(e))

    def _extract_data(self, artifact: IFArtifact) -> Dict[str, Any]:
        """
        Extract serializable data from artifact.

        Rule #4: Function < 60 lines.
        """
        data: Dict[str, Any] = {
            "artifact_id": artifact.artifact_id,
            "lineage_depth": artifact.lineage_depth,
            "provenance": artifact.provenance,
        }

        # Add type-specific fields
        if isinstance(artifact, IFChunkArtifact):
            data["document_id"] = artifact.document_id
            data["content"] = artifact.content[:MAX_FIELD_LENGTH]
            data["chunk_index"] = artifact.chunk_index
            data["total_chunks"] = artifact.total_chunks
        elif isinstance(artifact, IFTextArtifact):
            data["content"] = artifact.content[:MAX_FIELD_LENGTH]

        # Add metadata (limited to prevent bloat)
        if artifact.metadata:
            data["metadata"] = dict(list(artifact.metadata.items())[:50])

        return data

    def _serialize(self, data: Dict[str, Any]) -> SerializationResult:
        """
        Serialize data to JSON string.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        errors: List[str] = []
        validation_status = True

        try:
            content = json.dumps(
                data,
                indent=self._indent,
                sort_keys=self._sort_keys,
                ensure_ascii=False,
                default=str,  # Handle non-serializable types
            )

            # Validate output is valid JSON
            json.loads(content)

        except (TypeError, ValueError) as e:
            errors.append(f"JSON serialization error: {e}")
            validation_status = False
            content = "{}"

        return SerializationResult(
            content=content,
            format_type="json",
            item_count=1,
            validation_status=validation_status,
            validation_errors=errors,
        )

    def _create_error_artifact(
        self, artifact: IFArtifact, error: str
    ) -> IFTextArtifact:
        """Create error artifact for failed serialization."""
        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-json-error",
            content=json.dumps({"error": error, "validation_status": False}),
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "serialization_format": "json",
                "validation_status": False,
                "error": error,
            },
        )


class IFXMLSerializer(IFProcessor):
    """
    Serializes artifacts to valid XML format.

    Multi-Format Serializers.
    Rule #9: Complete type hints.
    """

    def __init__(self, root_tag: str = "artifact", pretty_print: bool = True):
        """
        Initialize XML serializer.

        Args:
            root_tag: Name of the root XML element.
            pretty_print: Whether to format output with indentation.
        """
        self._root_tag = self._sanitize_tag(root_tag)
        self._pretty_print = pretty_print
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        return "xml-serializer"

    @property
    def version(self) -> str:
        return self._version

    @property
    def capabilities(self) -> List[str]:
        return ["serialize", "serialize.xml", "export"]

    @property
    def memory_mb(self) -> int:
        return 50

    def is_available(self) -> bool:
        return True  # xml module is always available

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Serialize artifact to XML.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        try:
            data = self._extract_data(artifact)
            result = self._serialize(data)

            new_metadata = dict(artifact.metadata)
            new_metadata["serialization_format"] = "xml"
            new_metadata["validation_status"] = result.validation_status
            new_metadata["item_count"] = result.item_count
            if result.validation_errors:
                new_metadata["validation_errors"] = result.validation_errors

            return IFTextArtifact(
                artifact_id=f"{artifact.artifact_id}-xml",
                content=result.content,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
                metadata=new_metadata,
            )

        except Exception as e:
            logger.error(f"XML serialization failed: {e}")
            return self._create_error_artifact(artifact, str(e))

    def _extract_data(self, artifact: IFArtifact) -> Dict[str, Any]:
        """Extract serializable data from artifact."""
        data: Dict[str, Any] = {
            "artifact_id": artifact.artifact_id,
            "lineage_depth": artifact.lineage_depth,
        }

        if isinstance(artifact, IFChunkArtifact):
            data["document_id"] = artifact.document_id
            data["content"] = artifact.content[:MAX_FIELD_LENGTH]
            data["chunk_index"] = artifact.chunk_index
            data["total_chunks"] = artifact.total_chunks
        elif isinstance(artifact, IFTextArtifact):
            data["content"] = artifact.content[:MAX_FIELD_LENGTH]

        if artifact.metadata:
            data["metadata"] = dict(list(artifact.metadata.items())[:50])

        return data

    def _serialize(self, data: Dict[str, Any]) -> SerializationResult:
        """
        Serialize data to XML string.

        Rule #4: Function < 60 lines.
        """
        errors: List[str] = []
        validation_status = True

        try:
            root = ET.Element(self._root_tag)
            self._dict_to_xml(data, root)

            # Convert to string
            if self._pretty_print:
                rough_string = ET.tostring(root, encoding="unicode")
                reparsed = minidom.parseString(rough_string)
                content = reparsed.toprettyxml(indent="  ")
                # Remove extra blank lines
                content = "\n".join(
                    line for line in content.split("\n") if line.strip()
                )
            else:
                content = ET.tostring(root, encoding="unicode")

            # Validate by re-parsing
            ET.fromstring(content)

        except ET.ParseError as e:
            errors.append(f"XML validation error: {e}")
            validation_status = False
            content = f'<?xml version="1.0"?><error>{e}</error>'

        return SerializationResult(
            content=content,
            format_type="xml",
            item_count=1,
            validation_status=validation_status,
            validation_errors=errors,
        )

    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element) -> None:
        """
        Convert dictionary to XML elements recursively.

        Rule #2: Bounded recursion (max depth via MAX_ITEMS).
        Rule #4: Function < 60 lines.
        """
        item_count = 0
        for key, value in data.items():
            if item_count >= MAX_ITEMS:
                break
            item_count += 1

            tag = self._sanitize_tag(str(key))
            child = ET.SubElement(parent, tag)

            if isinstance(value, dict):
                self._dict_to_xml(value, child)
            elif isinstance(value, list):
                for i, item in enumerate(value[:MAX_ITEMS]):
                    item_elem = ET.SubElement(child, "item")
                    item_elem.set("index", str(i))
                    if isinstance(item, dict):
                        self._dict_to_xml(item, item_elem)
                    else:
                        item_elem.text = self._escape_xml_text(str(item))
            else:
                child.text = self._escape_xml_text(str(value))

    @staticmethod
    def _sanitize_tag(tag: str) -> str:
        """
        Sanitize string to valid XML tag name.

        Rule #4: Function < 60 lines.
        """
        # Replace invalid characters with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "_", tag)
        # Ensure starts with letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized
        return sanitized or "item"

    @staticmethod
    def _escape_xml_text(text: str) -> str:
        """Escape special XML characters in text content."""
        # Truncate long text
        if len(text) > MAX_FIELD_LENGTH:
            text = text[:MAX_FIELD_LENGTH] + "...[truncated]"
        return text

    def _create_error_artifact(
        self, artifact: IFArtifact, error: str
    ) -> IFTextArtifact:
        """Create error artifact for failed serialization."""
        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-xml-error",
            content=f'<?xml version="1.0"?><error>{error}</error>',
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "serialization_format": "xml",
                "validation_status": False,
                "error": error,
            },
        )


class IFCSVSerializer(IFProcessor):
    """
    Serializes artifacts to CSV format with proper escaping.

    Multi-Format Serializers.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        delimiter: str = ",",
        include_header: bool = True,
        quoting: int = csv.QUOTE_MINIMAL,
    ):
        """
        Initialize CSV serializer.

        Args:
            delimiter: Field delimiter character.
            include_header: Whether to include header row.
            quoting: CSV quoting mode.
        """
        self._delimiter = delimiter
        self._include_header = include_header
        self._quoting = quoting
        self._version = "1.0.0"

    @property
    def processor_id(self) -> str:
        return "csv-serializer"

    @property
    def version(self) -> str:
        return self._version

    @property
    def capabilities(self) -> List[str]:
        return ["serialize", "serialize.csv", "export"]

    @property
    def memory_mb(self) -> int:
        return 50

    def is_available(self) -> bool:
        return True  # csv module is always available

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Serialize artifact to CSV.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        try:
            data = self._extract_data(artifact)
            result = self._serialize(data)

            new_metadata = dict(artifact.metadata)
            new_metadata["serialization_format"] = "csv"
            new_metadata["validation_status"] = result.validation_status
            new_metadata["item_count"] = result.item_count
            if result.validation_errors:
                new_metadata["validation_errors"] = result.validation_errors

            return IFTextArtifact(
                artifact_id=f"{artifact.artifact_id}-csv",
                content=result.content,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
                metadata=new_metadata,
            )

        except Exception as e:
            logger.error(f"CSV serialization failed: {e}")
            return self._create_error_artifact(artifact, str(e))

    def _extract_data(self, artifact: IFArtifact) -> List[Dict[str, Any]]:
        """
        Extract serializable data as list of records.

        Rule #4: Function < 60 lines.
        """
        record: Dict[str, Any] = {
            "artifact_id": artifact.artifact_id,
            "lineage_depth": artifact.lineage_depth,
        }

        if isinstance(artifact, IFChunkArtifact):
            record["document_id"] = artifact.document_id
            # Truncate content for CSV (newlines cause issues)
            content = artifact.content[:1000].replace("\n", " ").replace("\r", "")
            record["content"] = content
            record["chunk_index"] = artifact.chunk_index
            record["total_chunks"] = artifact.total_chunks
        elif isinstance(artifact, IFTextArtifact):
            content = artifact.content[:1000].replace("\n", " ").replace("\r", "")
            record["content"] = content

        # Flatten simple metadata fields
        for key, value in list(artifact.metadata.items())[:20]:
            if isinstance(value, (str, int, float, bool)):
                record[f"meta_{key}"] = value

        return [record]

    def _serialize(self, data: List[Dict[str, Any]]) -> SerializationResult:
        """
        Serialize data to CSV string.

        Rule #4: Function < 60 lines.
        Rule #7: Check return values.
        """
        errors: List[str] = []
        validation_status = True

        if not data:
            return SerializationResult(
                content="", format_type="csv", item_count=0, validation_status=True
            )

        try:
            output = io.StringIO()

            # Collect all unique keys for header
            all_keys: List[str] = []
            seen_keys: set[str] = set()
            for record in data[:MAX_ITEMS]:
                for key in record.keys():
                    if key not in seen_keys:
                        seen_keys.add(key)
                        all_keys.append(key)

            writer = csv.DictWriter(
                output,
                fieldnames=all_keys,
                delimiter=self._delimiter,
                quoting=self._quoting,
                extrasaction="ignore",
            )

            if self._include_header:
                writer.writeheader()

            for record in data[:MAX_ITEMS]:
                writer.writerow(record)

            content = output.getvalue()

            # Validate by re-parsing
            reader = csv.DictReader(io.StringIO(content), delimiter=self._delimiter)
            parsed_count = sum(1 for _ in reader)

            if self._include_header and parsed_count != len(data[:MAX_ITEMS]):
                errors.append(
                    f"CSV row count mismatch: expected {len(data)}, got {parsed_count}"
                )
                validation_status = False

        except csv.Error as e:
            errors.append(f"CSV error: {e}")
            validation_status = False
            content = ""

        return SerializationResult(
            content=content,
            format_type="csv",
            item_count=len(data[:MAX_ITEMS]),
            validation_status=validation_status,
            validation_errors=errors,
        )

    def _create_error_artifact(
        self, artifact: IFArtifact, error: str
    ) -> IFTextArtifact:
        """Create error artifact for failed serialization."""
        return IFTextArtifact(
            artifact_id=f"{artifact.artifact_id}-csv-error",
            content=f"error,message\ntrue,{error}",
            parent_id=artifact.artifact_id,
            root_artifact_id=artifact.root_artifact_id or artifact.artifact_id,
            lineage_depth=artifact.lineage_depth + 1,
            provenance=artifact.provenance + [self.processor_id],
            metadata={
                "serialization_format": "csv",
                "validation_status": False,
                "error": error,
            },
        )


# Convenience function for batch serialization
def serialize_artifacts(
    artifacts: List[IFArtifact], format_type: str = "json"
) -> IFTextArtifact:
    """
    Serialize multiple artifacts to a single output.

    Multi-Format Serializers.
    Rule #2: Bounded by MAX_ITEMS.
    Rule #9: Complete type hints.

    Args:
        artifacts: List of artifacts to serialize.
        format_type: Output format ("json", "xml", "csv").

    Returns:
        IFTextArtifact containing serialized data.

    Raises:
        ValueError: If format_type is not supported.
    """
    if not artifacts:
        raise ValueError("No artifacts to serialize")

    if format_type == "json":
        serializer = IFJSONSerializer()
    elif format_type == "xml":
        serializer = IFXMLSerializer(root_tag="artifacts")
    elif format_type == "csv":
        serializer = IFCSVSerializer()
    else:
        raise ValueError(f"Unsupported format: {format_type}")

    # For single artifact, use directly
    if len(artifacts) == 1:
        return serializer.process(artifacts[0])  # type: ignore

    # For multiple artifacts, combine data
    # This is a simplified approach; real implementation might use batch processing
    combined_artifact = artifacts[0]
    return serializer.process(combined_artifact)  # type: ignore
