"""
Tests for JSON Export Command.

This module tests exporting knowledge base to JSON format.

Test Strategy
-------------
- Focus on JSON serialization and conversion logic
- Test metadata serialization with complex types
- Test pretty vs compact formatting
- Keep tests simple and readable (NASA JPL Rule #1)

Organization
------------
- TestJSONExportInit: Initialization
- TestValidation: Parameter validation
- TestPrimitiveCheck: Primitive type checking
- TestValueSerialization: Value conversion
- TestMetadataSerialization: Metadata conversion
- TestChunkConversion: Chunk to JSON conversion
- TestJSONGeneration: Full JSON structure generation
"""

from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime


from ingestforge.cli.export.json import JSONExportCommand


# ============================================================================
# Test Helpers
# ============================================================================


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    content: str = "Test content",
    source_file: str = "test.txt",
):
    """Create a mock chunk object."""
    chunk = Mock()
    chunk.chunk_id = chunk_id
    chunk.content = content
    chunk.source_file = source_file
    return chunk


# ============================================================================
# Test Classes
# ============================================================================


class TestJSONExportInit:
    """Tests for JSONExportCommand initialization.

    Rule #4: Focused test class - tests initialization only
    """

    def test_create_json_export_command(self):
        """Test creating JSONExportCommand instance."""
        cmd = JSONExportCommand()

        assert cmd is not None

    def test_inherits_from_export_command(self):
        """Test JSONExportCommand inherits from ExportCommand."""
        from ingestforge.cli.export.base import ExportCommand

        cmd = JSONExportCommand()

        assert isinstance(cmd, ExportCommand)


class TestValidation:
    """Tests for parameter validation.

    Rule #4: Focused test class - tests validation logic
    """

    def test_validate_limit_positive(self):
        """Test limit must be positive."""
        cmd = JSONExportCommand()
        output = Path("output.json")

        # Execute returns 1 (error) for invalid limit
        result = cmd.execute(output, limit=0)
        assert result == 1

        result = cmd.execute(output, limit=-1)
        assert result == 1

    @patch.object(JSONExportCommand, "validate_output_path")
    @patch.object(JSONExportCommand, "initialize_context")
    @patch.object(JSONExportCommand, "search_filtered_chunks")
    @patch.object(JSONExportCommand, "_save_json_file")
    def test_validate_limit_accepts_valid(
        self, mock_save, mock_search, mock_init, mock_validate
    ):
        """Test validation accepts valid limit values."""
        cmd = JSONExportCommand()
        output = Path("output.json")

        # Mock return values
        mock_init.return_value = {"storage": Mock(), "config": Mock()}
        mock_search.return_value = [make_mock_chunk()]

        # Should not raise for valid values
        result = cmd.execute(output, limit=1)
        assert result == 0

        result = cmd.execute(output, limit=100)
        assert result == 0


class TestPrimitiveCheck:
    """Tests for primitive type checking.

    Rule #4: Focused test class - tests _serialize_primitive()
    """

    def test_primitive_types(self):
        """Test recognizing primitive types."""
        cmd = JSONExportCommand()

        # Primitives
        assert cmd._serialize_primitive("string") is True
        assert cmd._serialize_primitive(42) is True
        assert cmd._serialize_primitive(3.14) is True
        assert cmd._serialize_primitive(True) is True
        assert cmd._serialize_primitive(None) is True

    def test_non_primitive_types(self):
        """Test recognizing non-primitive types."""
        cmd = JSONExportCommand()

        # Non-primitives
        assert cmd._serialize_primitive([1, 2, 3]) is False
        assert cmd._serialize_primitive({"key": "value"}) is False
        assert cmd._serialize_primitive(Path("/test")) is False
        assert cmd._serialize_primitive(datetime.now()) is False


class TestValueSerialization:
    """Tests for value serialization.

    Rule #4: Focused test class - tests _serialize_value()
    """

    def test_serialize_primitives(self):
        """Test serializing primitive values."""
        cmd = JSONExportCommand()

        assert cmd._serialize_value("test") == "test"
        assert cmd._serialize_value(42) == 42
        assert cmd._serialize_value(3.14) == 3.14
        assert cmd._serialize_value(True) is True
        assert cmd._serialize_value(None) is None

    def test_serialize_list(self):
        """Test serializing list to list."""
        cmd = JSONExportCommand()

        result = cmd._serialize_value([1, 2, 3])

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_serialize_tuple(self):
        """Test serializing tuple to list."""
        cmd = JSONExportCommand()

        result = cmd._serialize_value((1, 2, 3))

        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_serialize_complex_type(self):
        """Test serializing complex type to string."""
        cmd = JSONExportCommand()

        path = Path("/test/path")
        result = cmd._serialize_value(path)

        # Should convert to string
        assert isinstance(result, str)
        assert "test" in result.lower()


class TestMetadataSerialization:
    """Tests for metadata serialization.

    Rule #4: Focused test class - tests _serialize_metadata()
    """

    def test_serialize_simple_metadata(self):
        """Test serializing simple metadata."""
        cmd = JSONExportCommand()
        metadata = {
            "source": "test.txt",
            "page": 1,
            "valid": True,
        }

        result = cmd._serialize_metadata(metadata)

        assert result["source"] == "test.txt"
        assert result["page"] == 1
        assert result["valid"] is True

    def test_serialize_nested_metadata(self):
        """Test serializing nested metadata."""
        cmd = JSONExportCommand()
        metadata = {
            "source": "test.txt",
            "details": {
                "chapter": "Introduction",
                "section": 1,
            },
        }

        result = cmd._serialize_metadata(metadata)

        assert result["source"] == "test.txt"
        assert result["details"]["chapter"] == "Introduction"
        assert result["details"]["section"] == 1

    def test_serialize_complex_types(self):
        """Test serializing complex types in metadata."""
        cmd = JSONExportCommand()
        metadata = {
            "path": Path("/test/file.txt"),
            "timestamp": datetime(2024, 1, 1, 12, 0),
        }

        result = cmd._serialize_metadata(metadata)

        # Complex types should be converted to strings
        assert isinstance(result["path"], str)
        assert isinstance(result["timestamp"], str)


class TestChunkConversion:
    """Tests for chunk conversion.

    Rule #4: Focused test class - tests _convert_chunks()
    """

    @patch.object(JSONExportCommand, "extract_chunk_text")
    @patch.object(JSONExportCommand, "extract_chunk_metadata")
    def test_convert_single_chunk(self, mock_metadata, mock_text):
        """Test converting a single chunk."""
        cmd = JSONExportCommand()
        chunks = [make_mock_chunk()]
        mock_text.return_value = "Test content"
        mock_metadata.return_value = {"source": "test.txt"}

        result = cmd._convert_chunks(chunks)

        assert len(result) == 1
        assert result[0]["id"] == 0
        assert result[0]["text"] == "Test content"
        assert result[0]["metadata"]["source"] == "test.txt"

    @patch.object(JSONExportCommand, "extract_chunk_text")
    @patch.object(JSONExportCommand, "extract_chunk_metadata")
    def test_convert_multiple_chunks(self, mock_metadata, mock_text):
        """Test converting multiple chunks."""
        cmd = JSONExportCommand()
        chunks = [
            make_mock_chunk("chunk_1", "Content 1"),
            make_mock_chunk("chunk_2", "Content 2"),
        ]
        mock_text.side_effect = ["Content 1", "Content 2"]
        mock_metadata.return_value = {}

        result = cmd._convert_chunks(chunks)

        assert len(result) == 2
        assert result[0]["id"] == 0
        assert result[1]["id"] == 1


class TestJSONGeneration:
    """Tests for full JSON structure generation.

    Rule #4: Focused test class - tests _convert_to_json()
    """

    @patch.object(JSONExportCommand, "_convert_chunks")
    def test_generate_json_structure(self, mock_convert):
        """Test generating full JSON structure."""
        cmd = JSONExportCommand()
        chunks = [make_mock_chunk()]
        mock_convert.return_value = [{"id": 0, "text": "Content"}]

        result = cmd._convert_to_json(chunks, None)

        assert "metadata" in result
        assert "chunks" in result
        assert result["metadata"]["total_chunks"] == 1
        assert result["metadata"]["filter_query"] is None
        assert "generated" in result["metadata"]

    @patch.object(JSONExportCommand, "_convert_chunks")
    def test_generate_json_with_query(self, mock_convert):
        """Test generating JSON with query filter."""
        cmd = JSONExportCommand()
        chunks = [make_mock_chunk()]
        mock_convert.return_value = [{"id": 0, "text": "Content"}]

        result = cmd._convert_to_json(chunks, "Python")

        assert result["metadata"]["filter_query"] == "Python"
        assert result["metadata"]["total_chunks"] == 1


class TestNoChunks:
    """Tests for handling no chunks found.

    Rule #4: Focused test class - tests _handle_no_chunks()
    """

    def test_handle_no_chunks_with_query(self):
        """Test handling no chunks with query filter."""
        cmd = JSONExportCommand()

        # Should not raise
        cmd._handle_no_chunks("Python")

    def test_handle_no_chunks_without_query(self):
        """Test handling no chunks without query."""
        cmd = JSONExportCommand()

        # Should not raise
        cmd._handle_no_chunks(None)


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - JSON export init: 2 tests (creation, inheritance)
    - Validation: 2 tests (limit validation)
    - Primitive check: 2 tests (primitives, non-primitives)
    - Value serialization: 4 tests (primitives, list, tuple, complex types)
    - Metadata serialization: 3 tests (simple, nested, complex types)
    - Chunk conversion: 2 tests (single, multiple chunks)
    - JSON generation: 2 tests (basic, with query)
    - No chunks: 2 tests (with/without query)

    Total: 19 tests

Design Decisions:
    1. Focus on JSON serialization logic
    2. Test metadata conversion with complex types
    3. Test primitive type detection
    4. Test nested metadata handling
    5. Don't test file I/O (tested in base ExportCommand)
    6. Simple tests that verify JSON export works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - JSONExportCommand initialization
    - Inheritance from ExportCommand base class
    - Limit parameter validation (must be positive)
    - Primitive type recognition (str, int, float, bool, None)
    - Non-primitive type recognition (list, dict, Path, datetime)
    - Value serialization (primitives, lists, tuples, complex types)
    - Metadata serialization (simple, nested, complex types)
    - Chunk conversion to JSON structure
    - Full JSON generation with metadata
    - Query filter inclusion in metadata
    - No chunks handling (with/without query)

Justification:
    - JSON export is critical for programmatic access
    - Serialization logic needs verification
    - Complex type handling ensures data integrity
    - Metadata conversion enables full data export
    - Simple tests verify export system works correctly
"""
