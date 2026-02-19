import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.hr import HRMetadataRefiner


class TestHRMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return HRMetadataRefiner()

    def test_skill_normalization(self, refiner):
        metadata = {"resume_skills": ["JS", "React.js", "Python"]}
        chunk = ChunkRecord(
            chunk_id="c1", document_id="d1", content="Text", metadata=metadata
        )

        enriched = refiner.enrich(chunk)

        normalized = enriched.metadata["resume_skills_normalized"]
        assert "Javascript" in normalized
        assert "React" in normalized
        assert "Python" in normalized
        assert "Javascript" in enriched.concepts
