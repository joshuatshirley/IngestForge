import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.obsidian import ObsidianMetadataRefiner


class TestObsidianMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return ObsidianMetadataRefiner()

    def test_enrich_tags_and_aliases(self, refiner):
        metadata = {
            "frontmatter": {"tags": ["pkm", "research"], "aliases": "Vault, My Brain"}
        }
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            content="Content with tags",
            metadata=metadata,
        )

        enriched = refiner.enrich(chunk)

        assert "pkm" in enriched.concepts
        assert "research" in enriched.concepts
        assert "Vault" in enriched.metadata["aliases"]
        assert "My Brain" in enriched.metadata["aliases"]

    def test_string_tags(self, refiner):
        metadata = {"tags": "pkm, research #brain"}
        chunk = ChunkRecord(
            chunk_id="c2", document_id="d2", content="Content", metadata=metadata
        )

        enriched = refiner.enrich(chunk)
        assert "pkm" in enriched.concepts
        assert "research" in enriched.concepts
        assert "brain" in enriched.concepts
