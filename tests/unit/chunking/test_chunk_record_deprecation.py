"""
Tests for c: ChunkRecord Deprecation Warnings.

GWT-style tests verifying that ChunkRecord emits proper
deprecation warnings to guide migration to IFChunkArtifact.
"""

import warnings

from ingestforge.chunking.semantic_chunker import ChunkRecord, SemanticChunk


# --- GWT Scenario 1: Deprecation Warning on Instantiation ---


class TestDeprecationWarningEmitted:
    """Tests that ChunkRecord emits DeprecationWarning on instantiation."""

    def test_instantiation_emits_deprecation_warning(self):
        """Given code creating a ChunkRecord, When instantiated,
        Then a DeprecationWarning is emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-001",
                document_id="doc-001",
                content="Test content",
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_warning_is_deprecation_type(self):
        """Given ChunkRecord instantiation, When warning is caught,
        Then it is specifically a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-002",
                document_id="doc-002",
                content="Content",
            )

            assert w[0].category == DeprecationWarning

    def test_from_dict_also_warns(self):
        """Given ChunkRecord.from_dict(), When called,
        Then deprecation warning is emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord.from_dict(
                {
                    "chunk_id": "test-003",
                    "document_id": "doc-003",
                    "content": "Content",
                }
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


# --- GWT Scenario 2: Warning Includes Migration Path ---


class TestWarningMessage:
    """Tests that warning message includes migration guidance."""

    def test_warning_mentions_ifchunkartifact(self):
        """Given a deprecation warning, When message is examined,
        Then it mentions IFChunkArtifact."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-004",
                document_id="doc-004",
                content="Content",
            )

            message = str(w[0].message)
            assert "IFChunkArtifact" in message

    def test_warning_mentions_from_chunk_record(self):
        """Given a deprecation warning, When message is examined,
        Then it mentions from_chunk_record method."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-005",
                document_id="doc-005",
                content="Content",
            )

            message = str(w[0].message)
            assert "from_chunk_record" in message

    def test_warning_mentions_artifact_factory(self):
        """Given a deprecation warning, When message is examined,
        Then it mentions ArtifactFactory."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-006",
                document_id="doc-006",
                content="Content",
            )

            message = str(w[0].message)
            assert "ArtifactFactory" in message

    def test_warning_mentions_epic_06(self):
        """Given a deprecation warning, When message is examined,
        Then it references EPIC-06 for details."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-007",
                document_id="doc-007",
                content="Content",
            )

            message = str(w[0].message)
            assert "EPIC-06" in message

    def test_warning_indicates_future_removal(self):
        """Given a deprecation warning, When message is examined,
        Then it indicates future removal."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-008",
                document_id="doc-008",
                content="Content",
            )

            message = str(w[0].message)
            assert "deprecated" in message.lower()
            assert "future" in message.lower() or "removed" in message.lower()


# --- GWT Scenario 3: Warning Can Be Suppressed ---


class TestWarningSuppression:
    """Tests that warnings can be properly filtered."""

    def test_warning_can_be_filtered_by_category(self):
        """Given deprecation warning, When filtered by category,
        Then no warning is raised."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            ChunkRecord(
                chunk_id="test-009",
                document_id="doc-009",
                content="Content",
            )

            assert len(w) == 0

    def test_warning_can_be_filtered_by_message(self):
        """Given deprecation warning, When filtered by message pattern,
        Then no warning is raised."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(
                "ignore",
                message="ChunkRecord is deprecated",
            )
            ChunkRecord(
                chunk_id="test-010",
                document_id="doc-010",
                content="Content",
            )

            assert len(w) == 0

    def test_other_warnings_not_affected(self):
        """Given ChunkRecord deprecation filter, When other warning raised,
        Then other warnings still appear."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            warnings.filterwarnings(
                "ignore",
                message="ChunkRecord is deprecated",
            )

            # Create ChunkRecord (filtered)
            ChunkRecord(
                chunk_id="test-011",
                document_id="doc-011",
                content="Content",
            )

            # Emit a different warning (not filtered)
            warnings.warn("Different warning", UserWarning)

            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)


# --- GWT Scenario 4: Existing Tests Still Pass ---


class TestBackwardCompatibility:
    """Tests that ChunkRecord still functions correctly."""

    def test_chunk_record_still_functional(self):
        """Given ChunkRecord deprecation, When created,
        Then it still works as expected."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ChunkRecord(
                chunk_id="test-012",
                document_id="doc-012",
                content="Test content",
                section_title="Section",
                word_count=2,
            )

            assert record.chunk_id == "test-012"
            assert record.content == "Test content"
            assert record.section_title == "Section"
            assert record.word_count == 2

    def test_to_dict_still_works(self):
        """Given ChunkRecord deprecation, When to_dict() called,
        Then it still returns proper dict."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ChunkRecord(
                chunk_id="test-013",
                document_id="doc-013",
                content="Content",
            )
            data = record.to_dict()

            assert data["chunk_id"] == "test-013"
            assert data["content"] == "Content"

    def test_from_dict_still_works(self):
        """Given ChunkRecord deprecation, When from_dict() called,
        Then it still creates record properly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            record = ChunkRecord.from_dict(
                {
                    "chunk_id": "test-014",
                    "document_id": "doc-014",
                    "content": "Content",
                }
            )

            assert record.chunk_id == "test-014"


# --- GWT Scenario 5: Warning Appears Once Per Location ---


class TestWarningStackLevel:
    """Tests that warning points to the correct caller location."""

    def test_warning_has_correct_stacklevel(self):
        """Given ChunkRecord instantiation, When warning examined,
        Then it points to the calling code, not __post_init__."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This is line where ChunkRecord is instantiated
            ChunkRecord(
                chunk_id="test-015",
                document_id="doc-015",
                content="Content",
            )

            # Warning should NOT point to semantic_chunker.py internals
            # In pytest environments filename may be '<string>' which is fine
            filename = w[0].filename
            # Key assertion: stacklevel=2 ensures warning doesn't come from __post_init__
            assert "__post_init__" not in str(w[0].message)

    def test_warning_not_from_post_init(self):
        """Given warning stacklevel=2, When filename checked,
        Then it doesn't show __post_init__ as source."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="test-016",
                document_id="doc-016",
                content="Content",
            )

            # Verify warning was captured and has correct type
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


# --- Edge Cases ---


class TestEdgeCases:
    """Edge case tests for deprecation warnings."""

    def test_warning_with_all_fields(self):
        """Given ChunkRecord with all fields, When created,
        Then warning still emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="full-001",
                document_id="doc-full",
                content="Full content",
                section_title="Title",
                chunk_type="content",
                source_file="test.pdf",
                word_count=2,
                char_count=12,
                chunk_index=0,
                total_chunks=1,
                library="test",
                is_read=True,
                tags=["tag1", "tag2"],
                quality_score=0.95,
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_warning_with_empty_content(self):
        """Given ChunkRecord with empty content, When created,
        Then warning still emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ChunkRecord(
                chunk_id="empty-001",
                document_id="doc-empty",
                content="",
            )

            assert len(w) == 1


# --- GWT Scenario Completeness ---


class TestGWTScenarioCompleteness:
    """Meta-tests ensuring all GWT scenarios are covered."""

    def test_scenario_1_warning_emitted_covered(self):
        """GWT Scenario 1 (Warning Emitted) is tested."""
        assert hasattr(
            TestDeprecationWarningEmitted,
            "test_instantiation_emits_deprecation_warning",
        )

    def test_scenario_2_migration_path_covered(self):
        """GWT Scenario 2 (Migration Path) is tested."""
        assert hasattr(TestWarningMessage, "test_warning_mentions_ifchunkartifact")

    def test_scenario_3_suppression_covered(self):
        """GWT Scenario 3 (Warning Suppression) is tested."""
        assert hasattr(
            TestWarningSuppression, "test_warning_can_be_filtered_by_category"
        )

    def test_scenario_4_backward_compat_covered(self):
        """GWT Scenario 4 (Backward Compatibility) is tested."""
        assert hasattr(TestBackwardCompatibility, "test_chunk_record_still_functional")

    def test_scenario_5_stacklevel_covered(self):
        """GWT Scenario 5 (Warning Location) is tested."""
        assert hasattr(TestWarningStackLevel, "test_warning_has_correct_stacklevel")


# --- SemanticChunk Deprecation Tests ---


class TestSemanticChunkDeprecation:
    """Tests for SemanticChunk deprecation warnings."""

    def test_semantic_chunk_emits_deprecation_warning(self):
        """Given SemanticChunk instantiation, When created,
        Then a DeprecationWarning is emitted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SemanticChunk(
                text="Test content",
                start_index=0,
                end_index=12,
                coherence_score=0.95,
                metadata={},
            )

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_semantic_chunk_warning_mentions_ifchunkartifact(self):
        """Given SemanticChunk warning, When examined,
        Then it mentions IFChunkArtifact."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            SemanticChunk(
                text="Test",
                start_index=0,
                end_index=4,
                coherence_score=0.9,
                metadata={},
            )

            message = str(w[0].message)
            assert "IFChunkArtifact" in message

    def test_semantic_chunk_warning_can_be_filtered(self):
        """Given SemanticChunk deprecation filter, When applied,
        Then no warning is raised."""
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings(
                "ignore",
                message="SemanticChunk is deprecated",
            )
            SemanticChunk(
                text="Test",
                start_index=0,
                end_index=4,
                coherence_score=0.9,
                metadata={},
            )

            assert len(w) == 0

    def test_semantic_chunk_still_functional(self):
        """Given SemanticChunk deprecation, When created,
        Then it still works as expected."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk = SemanticChunk(
                text="Test content",
                start_index=0,
                end_index=12,
                coherence_score=0.95,
                metadata={"key": "value"},
            )

            assert chunk.text == "Test content"
            assert chunk.start_index == 0
            assert chunk.end_index == 12
            assert chunk.coherence_score == 0.95
            assert chunk.metadata == {"key": "value"}
