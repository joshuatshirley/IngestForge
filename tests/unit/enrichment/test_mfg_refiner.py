import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.mfg import MfgMetadataRefiner


class TestMfgMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return MfgMetadataRefiner()

    def test_enrich_manual_snippet(self, refiner):
        text = """
        Equipment: Hydraulic Pump. 
        Part Number: HP-998822. 
        Maintenance Cycle: every 500 hours.
        Error Code: ERR-404.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["mfg_part_number"] == "HP-998822"
        assert enriched.metadata["mfg_maintenance_cycle"] == "every 500 hours"
        assert "ERR-404" in enriched.metadata["mfg_error_codes"]

    def test_p_n_shorthand(self, refiner):
        text = "Order replacement for P/N 12345ABC."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["mfg_part_number"] == "12345ABC"
