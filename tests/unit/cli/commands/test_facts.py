"""Tests for facts command.

Copy-Paste Ready CLI Interfaces
Tests for GWT-3 (Fact Sheet Generation) and GWT-4 (Provenance Embedding).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ingestforge.cli.commands.facts import (
    MAX_FACTS,
    MAX_SOURCES,
    ExtractedFact,
    FactsCommand,
    FactSheet,
)


# =============================================================================
# Test ExtractedFact Dataclass
# =============================================================================


class TestExtractedFact:
    """Tests for ExtractedFact dataclass."""

    def test_create_basic_fact(self) -> None:
        """Test creating basic fact."""
        fact = ExtractedFact(field="title", value="Test Title")
        assert fact.field == "title"
        assert fact.value == "Test Title"
        assert fact.source == "Unknown"
        assert fact.confidence == 1.0

    def test_create_full_fact(self) -> None:
        """Test creating fact with all fields."""
        fact = ExtractedFact(
            field="author",
            value="John Doe",
            source="paper.pdf",
            confidence=0.95,
        )
        assert fact.source == "paper.pdf"
        assert fact.confidence == 0.95

    def test_none_field_raises(self) -> None:
        """Test that None field raises assertion."""
        with pytest.raises(AssertionError, match="cannot be None"):
            ExtractedFact(field=None, value="test")  # type: ignore

    def test_long_field_raises(self) -> None:
        """Test that long field name raises assertion."""
        with pytest.raises(AssertionError, match="field name too long"):
            ExtractedFact(field="x" * 100, value="test")

    def test_invalid_confidence_raises(self) -> None:
        """Test that confidence outside 0-1 raises assertion."""
        with pytest.raises(AssertionError, match="confidence must be 0-1"):
            ExtractedFact(field="test", value="val", confidence=1.5)


# =============================================================================
# Test FactSheet Dataclass
# =============================================================================


class TestFactSheet:
    """Tests for FactSheet dataclass."""

    def test_create_empty_sheet(self) -> None:
        """Test creating empty fact sheet."""
        sheet = FactSheet(schema_name="test")
        assert sheet.schema_name == "test"
        assert sheet.facts == []
        assert sheet.source_count == 0
        assert sheet.validation_passed is True

    def test_create_with_facts(self) -> None:
        """Test creating sheet with facts."""
        facts = [
            ExtractedFact(field="title", value="Test"),
            ExtractedFact(field="author", value="John"),
        ]
        sheet = FactSheet(
            schema_name="research",
            facts=facts,
            source_count=2,
        )
        assert len(sheet.facts) == 2

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        facts = [ExtractedFact(field="title", value="Test", source="doc.pdf")]
        sheet = FactSheet(schema_name="test", facts=facts, source_count=1)

        d = sheet.to_dict()

        assert d["schema"] == "test"
        assert d["source_count"] == 1
        assert d["validation_passed"] is True
        assert len(d["facts"]) == 1
        assert d["facts"][0]["field"] == "title"

    def test_to_markdown(self) -> None:
        """Test converting to markdown."""
        facts = [
            ExtractedFact(field="title", value="Test", source="doc.pdf"),
            ExtractedFact(field="year", value=2024, source="doc.pdf"),
        ]
        sheet = FactSheet(schema_name="research", facts=facts, source_count=1)

        md = sheet.to_markdown()

        assert "# Fact Sheet: research" in md
        assert "**Sources:** 1" in md
        assert "**Validation:** PASSED" in md
        assert "| title |" in md
        assert "| year |" in md

    def test_to_markdown_truncates_long_values(self) -> None:
        """Test markdown truncates long values."""
        facts = [ExtractedFact(field="long", value="x" * 100, source="doc.pdf")]
        sheet = FactSheet(schema_name="test", facts=facts)

        md = sheet.to_markdown()

        # Value should be truncated with "..."
        assert "..." in md

    def test_to_csv(self) -> None:
        """Test converting to CSV."""
        facts = [
            ExtractedFact(field="title", value="Test", source="doc.pdf"),
            ExtractedFact(field="desc", value='Has "quotes"', source="doc.pdf"),
        ]
        sheet = FactSheet(schema_name="test", facts=facts)

        csv = sheet.to_csv()

        assert "field,value,source,confidence" in csv
        assert '"title"' in csv
        # Escaped quotes
        assert '""quotes""' in csv

    def test_to_dict_bounded_by_max_facts(self) -> None:
        """Test to_dict respects MAX_FACTS."""
        facts = [ExtractedFact(field=f"f{i}", value=i) for i in range(MAX_FACTS + 10)]
        sheet = FactSheet(schema_name="test", facts=facts)

        d = sheet.to_dict()

        assert len(d["facts"]) <= MAX_FACTS


# =============================================================================
# Test FactsCommand - Execute
# =============================================================================


class TestFactsCommandExecute:
    """Tests for execute method."""

    def test_execute_file_not_found(self) -> None:
        """Test execute with non-existent file."""
        cmd = FactsCommand()
        exit_code = cmd.execute(source=Path("/nonexistent/file.pdf"))
        assert exit_code == 1

    def test_execute_file_success(self) -> None:
        """Test execute with existing file."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = Path(f.name)

        try:
            with patch.object(cmd, "_extract_from_file") as mock_extract:
                mock_extract.return_value = [ExtractedFact(field="test", value="value")]
                exit_code = cmd.execute(source=temp_path)
                assert exit_code == 0
        finally:
            temp_path.unlink()

    def test_execute_directory_success(self) -> None:
        """Test execute with directory."""
        cmd = FactsCommand()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "test.txt").write_text("content")

            with patch.object(cmd, "_extract_from_file") as mock_extract:
                mock_extract.return_value = [ExtractedFact(field="test", value="value")]
                exit_code = cmd.execute(source=tmp_path)
                assert exit_code == 0

    def test_execute_with_output_file(self) -> None:
        """Test execute saves to output file."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            source_path = Path(f.name)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.md"

            try:
                with patch.object(cmd, "_extract_from_file") as mock_extract:
                    mock_extract.return_value = [
                        ExtractedFact(field="test", value="value")
                    ]
                    exit_code = cmd.execute(source=source_path, output=output_path)

                    assert exit_code == 0
                    assert output_path.exists()
            finally:
                source_path.unlink()

    def test_execute_with_clipboard(self) -> None:
        """Test execute copies to clipboard."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            source_path = Path(f.name)

        try:
            with (
                patch.object(cmd, "_extract_from_file") as mock_extract,
                patch(
                    "ingestforge.cli.commands.facts.is_clipboard_available",
                    return_value=True,
                ),
                patch("ingestforge.cli.commands.facts.copy_to_clipboard") as mock_copy,
            ):
                mock_extract.return_value = [ExtractedFact(field="test", value="value")]
                from ingestforge.core.clipboard import (
                    ClipboardBackend,
                    ClipboardResult,
                )

                mock_copy.return_value = ClipboardResult(
                    success=True, backend=ClipboardBackend.PYPERCLIP, chars_copied=100
                )

                exit_code = cmd.execute(source=source_path, clip=True)

                assert exit_code == 0
                mock_copy.assert_called_once()
        finally:
            source_path.unlink()

    def test_execute_json_format(self) -> None:
        """Test execute with JSON format."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            source_path = Path(f.name)

        try:
            with patch.object(cmd, "_extract_from_file") as mock_extract:
                mock_extract.return_value = [ExtractedFact(field="test", value="value")]
                exit_code = cmd.execute(source=source_path, format_type="json")
                assert exit_code == 0
        finally:
            source_path.unlink()


# =============================================================================
# Test FactsCommand - Extraction
# =============================================================================


class TestFactsCommandExtraction:
    """Tests for fact extraction."""

    def test_extract_facts_single_file(self) -> None:
        """Test extracting facts from single file."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            source_path = Path(f.name)

        try:
            with patch.object(cmd, "_extract_from_file") as mock_extract:
                mock_extract.return_value = [ExtractedFact(field="title", value="Test")]

                sheet = cmd._extract_facts(source_path, "test", None)

                assert sheet.source_count == 1
                assert len(sheet.facts) == 1
        finally:
            source_path.unlink()

    def test_extract_facts_directory(self) -> None:
        """Test extracting facts from directory."""
        cmd = FactsCommand()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            (tmp_path / "a.txt").write_text("content a")
            (tmp_path / "b.txt").write_text("content b")

            with patch.object(cmd, "_extract_from_file") as mock_extract:
                mock_extract.return_value = [ExtractedFact(field="test", value="val")]

                sheet = cmd._extract_facts(tmp_path, "test", None)

                assert sheet.source_count == 2

    def test_extract_facts_bounded_sources(self) -> None:
        """Test extraction bounded by MAX_SOURCES."""
        cmd = FactsCommand()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create more files than MAX_SOURCES
            for i in range(MAX_SOURCES + 10):
                (tmp_path / f"file{i}.txt").write_text(f"content {i}")

            with patch.object(cmd, "_extract_from_file") as mock_extract:
                mock_extract.return_value = [ExtractedFact(field="test", value="val")]

                sheet = cmd._extract_facts(tmp_path, "test", None)

                assert sheet.source_count <= MAX_SOURCES

    def test_extract_facts_bounded_facts(self) -> None:
        """Test extraction bounded by MAX_FACTS."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            source_path = Path(f.name)

        try:
            with patch.object(cmd, "_extract_from_file") as mock_extract:
                # Return more facts than MAX_FACTS
                mock_extract.return_value = [
                    ExtractedFact(field=f"f{i}", value=i) for i in range(MAX_FACTS + 10)
                ]

                sheet = cmd._extract_facts(source_path, "test", None)

                assert len(sheet.facts) <= MAX_FACTS
        finally:
            source_path.unlink()


# =============================================================================
# Test FactsCommand - File Support
# =============================================================================


class TestFactsCommandFileSupport:
    """Tests for file type support."""

    def test_supported_file_types(self) -> None:
        """Test supported file type detection."""
        cmd = FactsCommand()

        assert cmd._is_supported_file(Path("test.pdf")) is True
        assert cmd._is_supported_file(Path("test.txt")) is True
        assert cmd._is_supported_file(Path("test.md")) is True
        assert cmd._is_supported_file(Path("test.json")) is True
        assert cmd._is_supported_file(Path("test.csv")) is True
        assert cmd._is_supported_file(Path("test.html")) is True
        assert cmd._is_supported_file(Path("test.xml")) is True

    def test_unsupported_file_types(self) -> None:
        """Test unsupported file type detection."""
        cmd = FactsCommand()

        assert cmd._is_supported_file(Path("test.exe")) is False
        assert cmd._is_supported_file(Path("test.bin")) is False
        assert cmd._is_supported_file(Path("test.zip")) is False


# =============================================================================
# Test FactsCommand - Formatting
# =============================================================================


class TestFactsCommandFormatting:
    """Tests for output formatting."""

    def test_format_json(self) -> None:
        """Test JSON formatting."""
        cmd = FactsCommand()
        sheet = FactSheet(
            schema_name="test",
            facts=[ExtractedFact(field="title", value="Test")],
        )

        output = cmd._format_output(sheet, "json")

        # Should be valid JSON
        parsed = json.loads(output)
        assert parsed["schema"] == "test"

    def test_format_csv(self) -> None:
        """Test CSV formatting."""
        cmd = FactsCommand()
        sheet = FactSheet(
            schema_name="test",
            facts=[ExtractedFact(field="title", value="Test")],
        )

        output = cmd._format_output(sheet, "csv")

        assert "field,value,source,confidence" in output

    def test_format_markdown_default(self) -> None:
        """Test markdown is default format."""
        cmd = FactsCommand()
        sheet = FactSheet(
            schema_name="test",
            facts=[ExtractedFact(field="title", value="Test")],
        )

        output = cmd._format_output(sheet, "markdown")

        assert "# Fact Sheet" in output


# =============================================================================
# Test JPL Compliance
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_constants_defined(self) -> None:
        """Test all required constants are defined."""
        assert MAX_FACTS == 100
        assert MAX_SOURCES == 50

    def test_fact_validates_on_creation(self) -> None:
        """Test ExtractedFact validates in __post_init__."""
        with pytest.raises(AssertionError):
            ExtractedFact(field="test", value="val", confidence=2.0)

    def test_none_source_raises(self) -> None:
        """Test None source raises assertion in execute."""
        cmd = FactsCommand()
        with pytest.raises(AssertionError, match="cannot be None"):
            cmd.execute(source=None)  # type: ignore

    def test_schema_default_value(self) -> None:
        """Test schema has sensible default."""
        cmd = FactsCommand()

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"content")
            source_path = Path(f.name)

        try:
            with patch.object(cmd, "_extract_from_file", return_value=[]):
                sheet = cmd._extract_facts(source_path, None, None)
                assert sheet.schema_name == "default"
        finally:
            source_path.unlink()
