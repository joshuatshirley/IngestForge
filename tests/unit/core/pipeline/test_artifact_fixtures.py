"""
Unit tests for j: Artifact Test Fixtures.

Tests verify:
1. make_chunk_artifact fixture creates valid artifacts
2. sample_chunk_artifact fixture provides single artifact
3. sample_artifacts fixture provides batch
4. make_text_artifact fixture creates text artifacts
5. artifact_with_lineage fixture has complete lineage
6. Fixtures follow NASA JPL Power of Ten rules

GWT Format: Given-When-Then
"""


from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFTextArtifact


class TestMakeChunkArtifactFixture:
    """GWT: Given make_chunk_artifact fixture, When used, Then creates valid artifacts."""

    def test_creates_valid_artifact(self, make_chunk_artifact) -> None:
        """Given make_chunk_artifact, When called, Then returns IFChunkArtifact."""
        artifact = make_chunk_artifact()
        assert isinstance(artifact, IFChunkArtifact)

    def test_auto_generates_artifact_id(self, make_chunk_artifact) -> None:
        """Given no artifact_id, When created, Then auto-generates ID."""
        artifact = make_chunk_artifact()
        assert artifact.artifact_id.startswith("test-chunk-")
        assert len(artifact.artifact_id) > 10

    def test_uses_provided_artifact_id(self, make_chunk_artifact) -> None:
        """Given artifact_id, When created, Then uses provided ID."""
        artifact = make_chunk_artifact(artifact_id="my-custom-id")
        assert artifact.artifact_id == "my-custom-id"

    def test_auto_generates_content(self, make_chunk_artifact) -> None:
        """Given no content, When created, Then auto-generates content."""
        artifact = make_chunk_artifact()
        assert "Test content" in artifact.content
        assert len(artifact.content) > 20

    def test_uses_provided_content(self, make_chunk_artifact) -> None:
        """Given content, When created, Then uses provided content."""
        artifact = make_chunk_artifact(content="My custom content")
        assert artifact.content == "My custom content"

    def test_default_metadata_populated(self, make_chunk_artifact) -> None:
        """Given default creation, When checking metadata, Then has standard fields."""
        artifact = make_chunk_artifact()
        assert "section_title" in artifact.metadata
        assert "chunk_type" in artifact.metadata
        assert "source_file" in artifact.metadata
        assert "word_count" in artifact.metadata

    def test_custom_metadata_merged(self, make_chunk_artifact) -> None:
        """Given custom metadata, When created, Then merged with defaults."""
        artifact = make_chunk_artifact(metadata={"custom_key": "custom_value"})
        assert artifact.metadata["custom_key"] == "custom_value"
        assert "section_title" in artifact.metadata  # Default still present

    def test_supports_lineage_parameters(self, make_chunk_artifact) -> None:
        """Given lineage params, When created, Then sets lineage correctly."""
        artifact = make_chunk_artifact(
            parent_id="parent-123",
            root_artifact_id="root-456",
            lineage_depth=3,
            provenance=["processor_a", "processor_b"],
        )
        assert artifact.parent_id == "parent-123"
        assert artifact.root_artifact_id == "root-456"
        assert artifact.lineage_depth == 3
        assert artifact.provenance == ["processor_a", "processor_b"]


class TestSampleChunkArtifactFixture:
    """GWT: Given sample_chunk_artifact fixture, When used, Then provides valid artifact."""

    def test_is_ifchunkartifact(self, sample_chunk_artifact) -> None:
        """Given sample_chunk_artifact, When checking type, Then is IFChunkArtifact."""
        assert isinstance(sample_chunk_artifact, IFChunkArtifact)

    def test_has_content(self, sample_chunk_artifact) -> None:
        """Given sample_chunk_artifact, When checking content, Then has content."""
        assert len(sample_chunk_artifact.content) > 0
        assert "sample" in sample_chunk_artifact.content.lower()

    def test_has_artifact_id(self, sample_chunk_artifact) -> None:
        """Given sample_chunk_artifact, When checking ID, Then has sample ID."""
        assert sample_chunk_artifact.artifact_id == "sample-artifact"


class TestSampleArtifactsFixture:
    """GWT: Given sample_artifacts fixture, When used, Then provides list of artifacts."""

    def test_returns_list(self, sample_artifacts) -> None:
        """Given sample_artifacts, When checking type, Then is list."""
        assert isinstance(sample_artifacts, list)

    def test_contains_five_artifacts(self, sample_artifacts) -> None:
        """Given sample_artifacts, When checking length, Then has 5 items."""
        assert len(sample_artifacts) == 5

    def test_all_are_ifchunkartifact(self, sample_artifacts) -> None:
        """Given sample_artifacts, When checking items, Then all are IFChunkArtifact."""
        for artifact in sample_artifacts:
            assert isinstance(artifact, IFChunkArtifact)

    def test_unique_artifact_ids(self, sample_artifacts) -> None:
        """Given sample_artifacts, When checking IDs, Then all unique."""
        ids = [a.artifact_id for a in sample_artifacts]
        assert len(ids) == len(set(ids))

    def test_varied_content(self, sample_artifacts) -> None:
        """Given sample_artifacts, When checking content, Then varied topics."""
        contents = [a.content for a in sample_artifacts]
        assert "Python" in contents[0]
        assert "Machine learning" in contents[1]
        assert "Natural language" in contents[2]


class TestMakeTextArtifactFixture:
    """GWT: Given make_text_artifact fixture, When used, Then creates text artifacts."""

    def test_creates_valid_artifact(self, make_text_artifact) -> None:
        """Given make_text_artifact, When called, Then returns IFTextArtifact."""
        artifact = make_text_artifact()
        assert isinstance(artifact, IFTextArtifact)

    def test_auto_generates_id(self, make_text_artifact) -> None:
        """Given no artifact_id, When created, Then auto-generates ID."""
        artifact = make_text_artifact()
        assert artifact.artifact_id.startswith("test-text-")

    def test_uses_provided_content(self, make_text_artifact) -> None:
        """Given content, When created, Then uses provided content."""
        artifact = make_text_artifact(content="Custom text content")
        assert artifact.content == "Custom text content"

    def test_supports_parent_id(self, make_text_artifact) -> None:
        """Given parent_id, When created, Then sets parent_id."""
        artifact = make_text_artifact(parent_id="file-artifact-123")
        assert artifact.parent_id == "file-artifact-123"


class TestSampleTextArtifactFixture:
    """GWT: Given sample_text_artifact fixture, When used, Then provides text artifact."""

    def test_is_iftextartifact(self, sample_text_artifact) -> None:
        """Given sample_text_artifact, When checking type, Then is IFTextArtifact."""
        assert isinstance(sample_text_artifact, IFTextArtifact)

    def test_has_content(self, sample_text_artifact) -> None:
        """Given sample_text_artifact, When checking content, Then has content."""
        assert len(sample_text_artifact.content) > 0


class TestArtifactWithLineageFixture:
    """GWT: Given artifact_with_lineage fixture, When used, Then has complete lineage."""

    def test_has_parent_id(self, artifact_with_lineage) -> None:
        """Given artifact_with_lineage, When checking parent_id, Then is set."""
        assert artifact_with_lineage.parent_id == "parent-text-artifact"

    def test_has_root_artifact_id(self, artifact_with_lineage) -> None:
        """Given artifact_with_lineage, When checking root_artifact_id, Then is set."""
        assert artifact_with_lineage.root_artifact_id == "root-file-artifact"

    def test_has_lineage_depth(self, artifact_with_lineage) -> None:
        """Given artifact_with_lineage, When checking lineage_depth, Then is 2."""
        assert artifact_with_lineage.lineage_depth == 2

    def test_has_provenance(self, artifact_with_lineage) -> None:
        """Given artifact_with_lineage, When checking provenance, Then has 3 processors."""
        assert len(artifact_with_lineage.provenance) == 3
        assert "semantic_chunker" in artifact_with_lineage.provenance


class TestFixturesJPLCompliance:
    """GWT: Given artifact fixtures, When reviewing code, Then follows JPL rules."""

    def test_rule_7_explicit_types(self, make_chunk_artifact) -> None:
        """Rule #7: Factory returns IFChunkArtifact, not Any."""
        artifact = make_chunk_artifact()
        # Type is explicit, not Any
        assert type(artifact).__name__ == "IFChunkArtifact"

    def test_rule_9_type_hints_in_metadata(self, make_chunk_artifact) -> None:
        """Rule #9: Metadata values are typed (str, int, not Any)."""
        artifact = make_chunk_artifact()
        # word_count should be int, not arbitrary
        assert isinstance(artifact.metadata["word_count"], int)

    def test_rule_2_bounded_content(self, sample_artifacts) -> None:
        """Rule #2: Fixed upper bound - exactly 5 artifacts."""
        assert len(sample_artifacts) == 5  # Known fixed bound


class TestArtifactFixtureInteroperability:
    """GWT: Given artifact fixtures, When used together, Then interoperable."""

    def test_artifact_to_chunk_record_conversion(self, sample_chunk_artifact) -> None:
        """Given artifact, When converting to ChunkRecord, Then succeeds."""
        record = sample_chunk_artifact.to_chunk_record()
        assert record.chunk_id == sample_chunk_artifact.artifact_id
        assert record.content == sample_chunk_artifact.content

    def test_artifact_derive_preserves_lineage(self, sample_chunk_artifact) -> None:
        """Given artifact, When deriving, Then lineage preserved."""
        derived = sample_chunk_artifact.derive("test_processor")
        assert derived.parent_id == sample_chunk_artifact.artifact_id
        assert "test_processor" in derived.provenance

    def test_batch_artifacts_can_be_stored(
        self, sample_artifacts, mock_storage
    ) -> None:
        """Given artifacts, When storing via mock, Then no errors."""
        # Convert to ChunkRecords for storage (current pattern)
        records = [a.to_chunk_record() for a in sample_artifacts]
        mock_storage.add_chunks(records)
        mock_storage.add_chunks.assert_called_once()
