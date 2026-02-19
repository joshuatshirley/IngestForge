import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.political import PoliticalMetadataRefiner


class TestPoliticalMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return PoliticalMetadataRefiner()

    def test_enrich_voting_record(self, refiner):
        text = """
        Senator: Jane Doe.
        Vote: Yea.
        Action: Voted in favor of SB-101.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["political_candidate"] == "Jane Doe"
        assert enriched.metadata["political_vote"] == "Yea"

    def test_enrich_contribution(self, refiner):
        text = """
        Donor: Tech PAC.
        Contribution: $25,000.50.
        Entity: Silicon Valley Associates.
        """
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert "Tech PAC" in enriched.metadata["political_donors"]
        assert "Silicon Valley Associates" in enriched.metadata["political_donors"]
        assert enriched.metadata["political_contribution"] == 25000.50
