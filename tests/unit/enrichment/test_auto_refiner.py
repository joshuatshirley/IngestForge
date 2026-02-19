import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.auto import AutoMetadataRefiner


class TestAutoMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return AutoMetadataRefiner()

    def test_enrich_restoration_log(self, refiner):
        text = """
        Vehicle VIN: 1G1YY26U455100001
        Status: Original engine, but transmission was Replaced.
        Part Ref: 12345-ABC-001 installed.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["auto_vin"] == "1G1YY26U455100001"
        assert "12345-ABC-001" in enriched.metadata["auto_part_numbers"]
        # It should pick up 'Original' as first status match or whatever logic
        assert "auto_restoration_status" in enriched.metadata

    def test_shorthand_parts(self, refiner):
        text = "P/N: 99999Z. Ref: 88888Y."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert "99999Z" in enriched.metadata["auto_part_numbers"]
        assert "88888Y" in enriched.metadata["auto_part_numbers"]
