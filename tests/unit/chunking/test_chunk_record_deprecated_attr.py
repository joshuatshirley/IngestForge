"""
Unit tests for i: ChunkRecord Deprecation Enhancements.

Tests verify:
1. __deprecated__ attribute is True on ChunkRecord
2. __deprecated__ attribute is True on SemanticChunk
3. ChunkRecordCompat alias works correctly
4. Deprecation warning message content
5. Backward compatibility maintained

GWT Format: Given-When-Then
NASA JPL Power of Ten: Rule #7 (explicit assertions), Rule #9 (type hints)
"""

import warnings
from typing import Any, Dict


from ingestforge.chunking import ChunkRecord, ChunkRecordCompat
from ingestforge.chunking.semantic_chunker import SemanticChunk


class TestChunkRecordDeprecatedAttribute:
    """GWT: Given ChunkRecord, When checking __deprecated__, Then it should be True."""

    def test_chunk_record_has_deprecated_attribute(self) -> None:
        """Given ChunkRecord class, When accessing __deprecated__, Then returns True."""
        # Given: ChunkRecord class
        # When: Accessing __deprecated__ class attribute
        # Then: It should be True
        assert hasattr(ChunkRecord, "__deprecated__")
        assert ChunkRecord.__deprecated__ is True

    def test_chunk_record_deprecated_is_class_var(self) -> None:
        """Given ChunkRecord, When checking __deprecated__ type, Then it's ClassVar."""
        # Given: ChunkRecord class
        # When: Accessing __deprecated__
        # Then: It should be accessible at class level (not instance level)
        assert ChunkRecord.__deprecated__ is True
        # Instance should also have access
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            record = ChunkRecord(
                chunk_id="test", document_id="doc", content="test content"
            )
        assert record.__deprecated__ is True


class TestSemanticChunkDeprecatedAttribute:
    """GWT: Given SemanticChunk, When checking __deprecated__, Then it should be True."""

    def test_semantic_chunk_has_deprecated_attribute(self) -> None:
        """Given SemanticChunk class, When accessing __deprecated__, Then returns True."""
        assert hasattr(SemanticChunk, "__deprecated__")
        assert SemanticChunk.__deprecated__ is True

    def test_semantic_chunk_deprecated_is_class_var(self) -> None:
        """Given SemanticChunk, When checking __deprecated__ type, Then it's ClassVar."""
        assert SemanticChunk.__deprecated__ is True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            chunk = SemanticChunk(
                text="test",
                start_index=0,
                end_index=4,
                coherence_score=0.9,
                metadata={},
            )
        assert chunk.__deprecated__ is True


class TestChunkRecordCompatAlias:
    """GWT: Given ChunkRecordCompat, When used, Then it works like ChunkRecord."""

    def test_compat_alias_is_same_class(self) -> None:
        """Given ChunkRecordCompat, When compared to ChunkRecord, Then they're identical."""
        assert ChunkRecordCompat is ChunkRecord

    def test_compat_alias_creates_chunk_record(self) -> None:
        """Given ChunkRecordCompat, When instantiated, Then creates ChunkRecord."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            record = ChunkRecordCompat(
                chunk_id="compat-test",
                document_id="doc-compat",
                content="test content via compat alias",
            )
        assert isinstance(record, ChunkRecord)
        assert record.chunk_id == "compat-test"

    def test_compat_alias_has_deprecated_attribute(self) -> None:
        """Given ChunkRecordCompat, When checking __deprecated__, Then returns True."""
        assert ChunkRecordCompat.__deprecated__ is True


class TestDeprecationWarningContent:
    """GWT: Given ChunkRecord creation, When warning emitted, Then message is helpful."""

    def test_warning_mentions_ifchunkartifact(self) -> None:
        """Given ChunkRecord instantiation, When warning emitted, Then mentions IFChunkArtifact."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(chunk_id="test", document_id="doc", content="test")
            assert len(w) == 1
            assert "IFChunkArtifact" in str(w[0].message)

    def test_warning_mentions_from_chunk_record(self) -> None:
        """Given ChunkRecord instantiation, When warning emitted, Then mentions conversion method."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(chunk_id="test", document_id="doc", content="test")
            assert "from_chunk_record" in str(w[0].message)

    def test_warning_mentions_artifact_factory(self) -> None:
        """Given ChunkRecord instantiation, When warning emitted, Then mentions ArtifactFactory."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(chunk_id="test", document_id="doc", content="test")
            assert "ArtifactFactory" in str(w[0].message)

    def test_warning_is_deprecation_warning_type(self) -> None:
        """Given ChunkRecord instantiation, When warning emitted, Then is DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(chunk_id="test", document_id="doc", content="test")
            assert issubclass(w[0].category, DeprecationWarning)


class TestBackwardCompatibility:
    """GWT: Given existing code using ChunkRecord, When executed, Then still works."""

    def test_chunk_record_fields_still_work(self) -> None:
        """Given ChunkRecord with all fields, When accessed, Then all values correct."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            record = ChunkRecord(
                chunk_id="bc-test",
                document_id="doc-bc",
                content="backward compatible content",
                section_title="Test Section",
                chunk_type="content",
                word_count=3,
                library="test-lib",
                tags=["tag1", "tag2"],
                metadata={"key": "value"},
            )
        assert record.chunk_id == "bc-test"
        assert record.content == "backward compatible content"
        assert record.section_title == "Test Section"
        assert record.tags == ["tag1", "tag2"]
        assert record.metadata["key"] == "value"

    def test_to_dict_still_works(self) -> None:
        """Given ChunkRecord, When calling to_dict(), Then returns valid dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            record = ChunkRecord(
                chunk_id="dict-test", document_id="doc", content="content"
            )
            data = record.to_dict()
        assert isinstance(data, dict)
        assert data["chunk_id"] == "dict-test"
        assert data["content"] == "content"

    def test_from_dict_still_works(self) -> None:
        """Given valid dict, When calling from_dict(), Then creates ChunkRecord."""
        data: Dict[str, Any] = {
            "chunk_id": "from-dict-test",
            "document_id": "doc",
            "content": "from dict content",
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            record = ChunkRecord.from_dict(data)
        assert record.chunk_id == "from-dict-test"
        assert record.content == "from dict content"


class TestProgrammaticDeprecationDetection:
    """GWT: Given code checking for deprecated classes, When using __deprecated__, Then detects correctly."""

    def test_detect_deprecated_class_programmatically(self) -> None:
        """Given class with __deprecated__, When checking attribute, Then can detect."""
        deprecated_classes = []
        for cls in [ChunkRecord, SemanticChunk]:
            if getattr(cls, "__deprecated__", False):
                deprecated_classes.append(cls.__name__)

        assert "ChunkRecord" in deprecated_classes
        assert "SemanticChunk" in deprecated_classes

    def test_non_deprecated_class_lacks_attribute(self) -> None:
        """Given regular class, When checking __deprecated__, Then not present or False."""
        # Regular dataclass should not have __deprecated__
        from dataclasses import dataclass

        @dataclass
        class NotDeprecated:
            value: str

        assert not getattr(NotDeprecated, "__deprecated__", False)


class TestDocstringContent:
    """GWT: Given ChunkRecord docstring, When read, Then contains deprecation notice."""

    def test_docstring_mentions_deprecated(self) -> None:
        """Given ChunkRecord docstring, When examined, Then contains DEPRECATED."""
        assert "DEPRECATED" in ChunkRecord.__doc__

    def test_docstring_mentions_ifchunkartifact(self) -> None:
        """Given ChunkRecord docstring, When examined, Then mentions replacement."""
        assert "IFChunkArtifact" in ChunkRecord.__doc__

    def test_docstring_has_migration_guide(self) -> None:
        """Given ChunkRecord docstring, When examined, Then has migration examples."""
        assert "from_chunk_record" in ChunkRecord.__doc__
        assert "Migration Guide" in ChunkRecord.__doc__

    def test_semantic_chunk_docstring_deprecated(self) -> None:
        """Given SemanticChunk docstring, When examined, Then contains DEPRECATED."""
        assert "DEPRECATED" in SemanticChunk.__doc__
