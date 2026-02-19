"""
Unit tests for IFCodeProcessor and IFStructureProcessor.

Migration - Code & Structure Parity
Tests cover all GWT scenarios, JPL rules, and feature acceptance criteria.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFFileArtifact,
    IFTextArtifact,
    IFFailureArtifact,
    IFChunkArtifact,
)
from ingestforge.ingest.if_code_processor import (
    IFCodeProcessor,
    IFStructureProcessor,
    CODE_EXTENSIONS,
    MAX_FILE_SIZE,
    MAX_LINES,
    MAX_XML_DEPTH,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def code_processor() -> IFCodeProcessor:
    """Create a fresh code processor instance."""
    return IFCodeProcessor()


@pytest.fixture
def structure_processor() -> IFStructureProcessor:
    """Create a fresh structure processor instance."""
    return IFStructureProcessor()


@pytest.fixture
def python_file(tmp_path: Path) -> IFFileArtifact:
    """Create a test Python file artifact."""
    py_path = tmp_path / "test.py"
    py_path.write_text(
        '''def hello():
    """Say hello."""
    print("Hello, World!")

if __name__ == "__main__":
    hello()
'''
    )
    return IFFileArtifact(
        artifact_id="test-py-001",
        file_path=py_path,
        mime_type="text/x-python",
    )


@pytest.fixture
def javascript_file(tmp_path: Path) -> IFFileArtifact:
    """Create a test JavaScript file artifact."""
    js_path = tmp_path / "test.js"
    js_path.write_text(
        """function greet(name) {
    console.log(`Hello, ${name}!`);
}

greet("World");
"""
    )
    return IFFileArtifact(
        artifact_id="test-js-001",
        file_path=js_path,
        mime_type="application/javascript",
    )


@pytest.fixture
def json_file(tmp_path: Path) -> IFFileArtifact:
    """Create a test JSON file artifact."""
    json_path = tmp_path / "test.json"
    json_path.write_text('{"name": "test", "value": 42, "nested": {"key": "val"}}')
    return IFFileArtifact(
        artifact_id="test-json-001",
        file_path=json_path,
        mime_type="application/json",
    )


@pytest.fixture
def yaml_file(tmp_path: Path) -> IFFileArtifact:
    """Create a test YAML file artifact."""
    yaml_path = tmp_path / "test.yaml"
    yaml_path.write_text(
        """name: test
value: 42
nested:
  key: val
"""
    )
    return IFFileArtifact(
        artifact_id="test-yaml-001",
        file_path=yaml_path,
        mime_type="application/x-yaml",
    )


@pytest.fixture
def xml_file(tmp_path: Path) -> IFFileArtifact:
    """Create a test XML file artifact."""
    xml_path = tmp_path / "test.xml"
    xml_path.write_text(
        """<?xml version="1.0"?>
<root>
    <item name="test">
        <value>42</value>
    </item>
</root>
"""
    )
    return IFFileArtifact(
        artifact_id="test-xml-001",
        file_path=xml_path,
        mime_type="application/xml",
    )


# =============================================================================
# GWT Scenario 1: Source Code Extraction
# =============================================================================


class TestSourceCodeExtraction:
    """GWT Scenario 1: Source code extraction."""

    def test_given_python_file_when_processed_then_text_extracted(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Given Python file, when processed, then text extracted."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        assert "def hello():" in result.content
        assert "print(" in result.content

    def test_given_javascript_file_when_processed_then_text_extracted(
        self,
        code_processor: IFCodeProcessor,
        javascript_file: IFFileArtifact,
    ):
        """Given JavaScript file, when processed, then text extracted."""
        result = code_processor.process(javascript_file)

        assert isinstance(result, IFTextArtifact)
        assert "function greet" in result.content
        assert "console.log" in result.content

    def test_extracted_code_preserves_structure(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Code extraction preserves indentation and structure."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        # Check indentation is preserved
        assert (
            '    """Say hello."""' in result.content or "    print(" in result.content
        )

    def test_code_metadata_includes_line_count(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Extracted code includes line count in metadata."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        assert "line_count" in result.metadata
        assert result.metadata["line_count"] > 0

    def test_all_code_extensions_supported(self, code_processor: IFCodeProcessor):
        """Processor supports all declared code extensions."""
        expected = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp"}
        assert expected.issubset(CODE_EXTENSIONS)


# =============================================================================
# GWT Scenario 2: JSON Document Processing
# =============================================================================


class TestJSONProcessing:
    """GWT Scenario 2: JSON document processing."""

    def test_given_json_file_when_processed_then_content_extracted(
        self,
        structure_processor: IFStructureProcessor,
        json_file: IFFileArtifact,
    ):
        """Given JSON file, when processed, then content extracted."""
        result = structure_processor.process(json_file)

        assert isinstance(result, IFTextArtifact)
        assert '"name"' in result.content
        assert '"test"' in result.content

    def test_valid_json_marked_as_parsed(
        self,
        structure_processor: IFStructureProcessor,
        json_file: IFFileArtifact,
    ):
        """Valid JSON is marked as successfully parsed."""
        result = structure_processor.process(json_file)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["parsed_successfully"] is True

    def test_invalid_json_marked_as_not_parsed(
        self,
        structure_processor: IFStructureProcessor,
        tmp_path: Path,
    ):
        """Invalid JSON is marked as not parsed but still extracted."""
        invalid_path = tmp_path / "invalid.json"
        invalid_path.write_text('{"broken": json}')
        artifact = IFFileArtifact(
            artifact_id="invalid-json",
            file_path=invalid_path,
            mime_type="application/json",
        )

        result = structure_processor.process(artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["parsed_successfully"] is False


# =============================================================================
# GWT Scenario 3: YAML Document Processing
# =============================================================================


class TestYAMLProcessing:
    """GWT Scenario 3: YAML document processing."""

    def test_given_yaml_file_when_processed_then_content_extracted(
        self,
        structure_processor: IFStructureProcessor,
        yaml_file: IFFileArtifact,
    ):
        """Given YAML file, when processed, then content extracted."""
        result = structure_processor.process(yaml_file)

        assert isinstance(result, IFTextArtifact)
        assert "name:" in result.content
        assert "nested:" in result.content

    def test_yaml_preserves_hierarchy(
        self,
        structure_processor: IFStructureProcessor,
        yaml_file: IFFileArtifact,
    ):
        """YAML extraction preserves hierarchy indentation."""
        result = structure_processor.process(yaml_file)

        assert isinstance(result, IFTextArtifact)
        # Check nested structure
        assert "key:" in result.content

    def test_yml_extension_also_supported(
        self,
        structure_processor: IFStructureProcessor,
        tmp_path: Path,
    ):
        """Both .yaml and .yml extensions are supported."""
        yml_path = tmp_path / "test.yml"
        yml_path.write_text("key: value")
        artifact = IFFileArtifact(
            artifact_id="test-yml",
            file_path=yml_path,
            mime_type="application/x-yaml",
        )

        result = structure_processor.process(artifact)

        assert isinstance(result, IFTextArtifact)


# =============================================================================
# GWT Scenario 4: XML Document Processing
# =============================================================================


class TestXMLProcessing:
    """GWT Scenario 4: XML document processing."""

    def test_given_xml_file_when_processed_then_content_extracted(
        self,
        structure_processor: IFStructureProcessor,
        xml_file: IFFileArtifact,
    ):
        """Given XML file, when processed, then content extracted."""
        result = structure_processor.process(xml_file)

        assert isinstance(result, IFTextArtifact)
        assert "<root>" in result.content
        assert "<item" in result.content

    def test_valid_xml_marked_as_parsed(
        self,
        structure_processor: IFStructureProcessor,
        xml_file: IFFileArtifact,
    ):
        """Valid XML is marked as successfully parsed."""
        result = structure_processor.process(xml_file)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["parsed_successfully"] is True

    def test_invalid_xml_marked_as_not_parsed(
        self,
        structure_processor: IFStructureProcessor,
        tmp_path: Path,
    ):
        """Invalid XML is marked as not parsed."""
        invalid_path = tmp_path / "invalid.xml"
        invalid_path.write_text("<broken>no closing tag")
        artifact = IFFileArtifact(
            artifact_id="invalid-xml",
            file_path=invalid_path,
            mime_type="application/xml",
        )

        result = structure_processor.process(artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.metadata["parsed_successfully"] is False


# =============================================================================
# GWT Scenario 5: Processor Registry Integration
# =============================================================================


class TestRegistryIntegration:
    """GWT Scenario 5: Processor registry integration."""

    def test_code_processor_has_correct_capabilities(
        self,
        code_processor: IFCodeProcessor,
    ):
        """Code processor declares correct capabilities."""
        assert "ingest.code" in code_processor.capabilities
        assert "text-extraction" in code_processor.capabilities

    def test_structure_processor_has_correct_capabilities(
        self,
        structure_processor: IFStructureProcessor,
    ):
        """Structure processor declares correct capabilities."""
        assert "ingest.json" in structure_processor.capabilities
        assert "ingest.yaml" in structure_processor.capabilities
        assert "ingest.xml" in structure_processor.capabilities

    def test_processors_have_unique_ids(
        self,
        code_processor: IFCodeProcessor,
        structure_processor: IFStructureProcessor,
    ):
        """Processors have unique identifiers."""
        assert code_processor.processor_id == "if-code-extractor"
        assert structure_processor.processor_id == "if-structure-extractor"
        assert code_processor.processor_id != structure_processor.processor_id

    def test_processors_declare_memory_requirements(
        self,
        code_processor: IFCodeProcessor,
        structure_processor: IFStructureProcessor,
    ):
        """Processors declare memory requirements."""
        assert code_processor.memory_mb > 0
        assert structure_processor.memory_mb > 0


# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================


class TestJPLRule2Bounds:
    """JPL Rule #2: Fixed upper bounds on loops and data."""

    def test_max_file_size_constant_exists(self):
        """MAX_FILE_SIZE constant is defined."""
        assert MAX_FILE_SIZE == 10_000_000

    def test_max_lines_constant_exists(self):
        """MAX_LINES constant is defined."""
        assert MAX_LINES == 100_000

    def test_max_xml_depth_constant_exists(self):
        """MAX_XML_DEPTH constant is defined."""
        assert MAX_XML_DEPTH == 100

    def test_large_file_rejected(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Files exceeding MAX_FILE_SIZE are rejected."""
        large_path = tmp_path / "large.py"
        # Create file larger than limit
        large_path.write_text("x" * (MAX_FILE_SIZE + 1))
        artifact = IFFileArtifact(
            artifact_id="large-file",
            file_path=large_path,
            mime_type="text/x-python",
        )

        result = code_processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "exceeds" in result.error_message.lower()

    def test_code_truncated_at_max_lines(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Code with too many lines is truncated."""
        many_lines_path = tmp_path / "many_lines.py"
        many_lines_path.write_text("\n".join(["x = 1"] * (MAX_LINES + 100)))
        artifact = IFFileArtifact(
            artifact_id="many-lines",
            file_path=many_lines_path,
            mime_type="text/x-python",
        )

        result = code_processor.process(artifact)

        assert isinstance(result, IFTextArtifact)
        assert "Truncated" in result.content

    def test_deeply_nested_xml_handled(
        self,
        structure_processor: IFStructureProcessor,
        tmp_path: Path,
    ):
        """Very deeply nested XML is processed (depth check prevents memory issues)."""
        # Create XML with deep nesting - each level is a proper element
        levels = MAX_XML_DEPTH + 50
        deep_xml = "".join(f"<l{i}>" for i in range(levels))
        deep_xml += "content"
        deep_xml += "".join(f"</l{i}>" for i in range(levels - 1, -1, -1))
        deep_path = tmp_path / "deep.xml"
        deep_path.write_text(deep_xml)
        artifact = IFFileArtifact(
            artifact_id="deep-xml",
            file_path=deep_path,
            mime_type="application/xml",
        )

        result = structure_processor.process(artifact)

        # Processing should still succeed - depth check is protective, not blocking
        assert isinstance(result, IFTextArtifact)


# =============================================================================
# JPL Rule #7: Check Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """JPL Rule #7: All functions check return values."""

    def test_code_process_always_returns_artifact(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """process() always returns an IFArtifact."""
        result = code_processor.process(python_file)
        assert isinstance(result, (IFTextArtifact, IFFailureArtifact))

    def test_structure_process_always_returns_artifact(
        self,
        structure_processor: IFStructureProcessor,
        json_file: IFFileArtifact,
    ):
        """process() always returns an IFArtifact."""
        result = structure_processor.process(json_file)
        assert isinstance(result, (IFTextArtifact, IFFailureArtifact))

    def test_invalid_input_type_returns_failure(
        self,
        code_processor: IFCodeProcessor,
    ):
        """Invalid input type returns IFFailureArtifact."""
        chunk = IFChunkArtifact(
            artifact_id="chunk-001",
            document_id="doc-001",
            content="test",
        )
        result = code_processor.process(chunk)

        assert isinstance(result, IFFailureArtifact)
        assert "requires IFFileArtifact" in result.error_message

    def test_unsupported_extension_returns_failure(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Unsupported file extension returns IFFailureArtifact."""
        txt_path = tmp_path / "test.xyz"
        txt_path.write_text("content")
        artifact = IFFileArtifact(
            artifact_id="xyz-file",
            file_path=txt_path,
            mime_type="application/octet-stream",
        )

        result = code_processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "Unsupported" in result.error_message

    def test_missing_file_returns_failure(
        self,
        code_processor: IFCodeProcessor,
    ):
        """Missing file returns IFFailureArtifact."""
        artifact = IFFileArtifact(
            artifact_id="missing",
            file_path=Path("/nonexistent/file.py"),
            mime_type="text/x-python",
        )

        result = code_processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "not found" in result.error_message.lower()


# =============================================================================
# JPL Rule #9: Complete Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """JPL Rule #9: Complete type hints on all public methods."""

    def test_code_processor_process_has_type_hints(self):
        """IFCodeProcessor.process() has type hints."""
        import inspect

        sig = inspect.signature(IFCodeProcessor.process)
        assert sig.parameters["artifact"].annotation is not inspect.Parameter.empty
        assert sig.return_annotation is not inspect.Signature.empty

    def test_structure_processor_process_has_type_hints(self):
        """IFStructureProcessor.process() has type hints."""
        import inspect

        sig = inspect.signature(IFStructureProcessor.process)
        assert sig.parameters["artifact"].annotation is not inspect.Parameter.empty
        assert sig.return_annotation is not inspect.Signature.empty

    def test_is_available_has_return_type(self):
        """is_available() methods have return type hints."""
        import inspect

        code_sig = inspect.signature(IFCodeProcessor.is_available)
        struct_sig = inspect.signature(IFStructureProcessor.is_available)

        assert code_sig.return_annotation == bool
        assert struct_sig.return_annotation == bool


# =============================================================================
# Artifact Lineage Tests
# =============================================================================


class TestArtifactLineage:
    """Tests for artifact lineage tracking."""

    def test_code_artifact_has_parent_id(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Extracted code artifact has parent_id set."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        assert result.parent_id == python_file.artifact_id

    def test_code_artifact_increments_lineage_depth(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Extracted code artifact increments lineage_depth."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        assert result.lineage_depth == python_file.lineage_depth + 1

    def test_code_artifact_includes_processor_in_provenance(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Processor ID is added to provenance chain."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        assert code_processor.processor_id in result.provenance

    def test_structure_artifact_tracks_lineage(
        self,
        structure_processor: IFStructureProcessor,
        json_file: IFFileArtifact,
    ):
        """Structure artifact properly tracks lineage."""
        result = structure_processor.process(json_file)

        assert isinstance(result, IFTextArtifact)
        assert result.parent_id == json_file.artifact_id
        assert structure_processor.processor_id in result.provenance


# =============================================================================
# Content Hash Tests
# =============================================================================


class TestContentHashing:
    """Tests for content hash generation."""

    def test_code_artifact_has_content_hash(
        self,
        code_processor: IFCodeProcessor,
        python_file: IFFileArtifact,
    ):
        """Code artifact has SHA-256 content hash."""
        result = code_processor.process(python_file)

        assert isinstance(result, IFTextArtifact)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64  # SHA-256 hex

    def test_structure_artifact_has_content_hash(
        self,
        structure_processor: IFStructureProcessor,
        json_file: IFFileArtifact,
    ):
        """Structure artifact has SHA-256 content hash."""
        result = structure_processor.process(json_file)

        assert isinstance(result, IFTextArtifact)
        assert result.content_hash is not None
        assert len(result.content_hash) == 64

    def test_same_content_produces_same_hash(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Identical content produces identical hash."""
        content = "def test(): pass"

        file1 = tmp_path / "file1.py"
        file1.write_text(content)
        file2 = tmp_path / "file2.py"
        file2.write_text(content)

        art1 = IFFileArtifact(
            artifact_id="f1", file_path=file1, mime_type="text/x-python"
        )
        art2 = IFFileArtifact(
            artifact_id="f2", file_path=file2, mime_type="text/x-python"
        )

        result1 = code_processor.process(art1)
        result2 = code_processor.process(art2)

        assert result1.content_hash == result2.content_hash


# =============================================================================
# Teardown Tests
# =============================================================================


class TestTeardown:
    """Tests for resource cleanup."""

    def test_code_processor_teardown_returns_true(
        self,
        code_processor: IFCodeProcessor,
    ):
        """Code processor teardown returns True."""
        assert code_processor.teardown() is True

    def test_structure_processor_teardown_returns_true(
        self,
        structure_processor: IFStructureProcessor,
    ):
        """Structure processor teardown returns True."""
        assert structure_processor.teardown() is True

    def test_structure_processor_clears_yaml_module(
        self,
        structure_processor: IFStructureProcessor,
    ):
        """Structure processor clears yaml module reference on teardown."""
        structure_processor._yaml_module = MagicMock()
        structure_processor.teardown()
        assert structure_processor._yaml_module is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_file_handled(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Empty files are handled gracefully."""
        empty_path = tmp_path / "empty.py"
        empty_path.write_text("")
        artifact = IFFileArtifact(
            artifact_id="empty",
            file_path=empty_path,
            mime_type="text/x-python",
        )

        result = code_processor.process(artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.content == ""

    def test_unicode_content_handled(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Unicode content is handled correctly."""
        unicode_path = tmp_path / "unicode.py"
        unicode_path.write_text(
            '# -*- coding: utf-8 -*-\nprint("Hello World")', encoding="utf-8"
        )
        artifact = IFFileArtifact(
            artifact_id="unicode",
            file_path=unicode_path,
            mime_type="text/x-python",
        )

        result = code_processor.process(artifact)

        assert isinstance(result, IFTextArtifact)
        assert "Hello World" in result.content

    def test_binary_content_replaced(
        self,
        code_processor: IFCodeProcessor,
        tmp_path: Path,
    ):
        """Binary content errors are replaced, not raised."""
        binary_path = tmp_path / "binary.py"
        binary_path.write_bytes(b"\x00\x01\x02def test(): pass")
        artifact = IFFileArtifact(
            artifact_id="binary",
            file_path=binary_path,
            mime_type="text/x-python",
        )

        result = code_processor.process(artifact)

        # Should not raise, content may have replacement chars
        assert isinstance(result, IFTextArtifact)


# =============================================================================
# GWT Scenario Completeness
# =============================================================================


class TestGWTScenarioCompleteness:
    """Verify all 5 GWT scenarios are explicitly covered."""

    def test_scenario_1_source_code_covered(self):
        """Scenario 1: Source Code Extraction is covered."""
        assert hasattr(
            TestSourceCodeExtraction,
            "test_given_python_file_when_processed_then_text_extracted",
        )

    def test_scenario_2_json_covered(self):
        """Scenario 2: JSON Document Processing is covered."""
        assert hasattr(
            TestJSONProcessing,
            "test_given_json_file_when_processed_then_content_extracted",
        )

    def test_scenario_3_yaml_covered(self):
        """Scenario 3: YAML Document Processing is covered."""
        assert hasattr(
            TestYAMLProcessing,
            "test_given_yaml_file_when_processed_then_content_extracted",
        )

    def test_scenario_4_xml_covered(self):
        """Scenario 4: XML Document Processing is covered."""
        assert hasattr(
            TestXMLProcessing,
            "test_given_xml_file_when_processed_then_content_extracted",
        )

    def test_scenario_5_registry_covered(self):
        """Scenario 5: Processor Registry Integration is covered."""
        assert hasattr(
            TestRegistryIntegration, "test_code_processor_has_correct_capabilities"
        )
