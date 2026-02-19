import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.grant import GrantMetadataRefiner


class TestGrantMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return GrantMetadataRefiner()

    def test_enrich_grant_details(self, refiner):
        text = """
        Funding Opportunity Number: HHS-2024-ABC-123
        Award Ceiling: $500,000
        Deadline: September 30, 2024
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["grant_id"] == "HHS-2024-ABC-123"
        assert enriched.metadata["grant_amount"] == 500000.0
        assert "September 30, 2024" in enriched.metadata["grant_deadline"]

    def test_amount_with_cents(self, refiner):
        text = "Maximum award is $1,250,000.50 for this project."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["grant_amount"] == 1250000.50
