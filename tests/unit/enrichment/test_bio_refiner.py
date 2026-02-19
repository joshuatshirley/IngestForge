import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.bio import BioMetadataRefiner


class TestBioMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return BioMetadataRefiner()

    def test_enrich_lab_notebook(self, refiner):
        text = """
        Experiment ID: EXP-2026-02
        Protocol: PCR Amplification
        Reagents: Added H2O and NaCl to the solution.
        Target: C6H12O6 detection.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["bio_exp_id"] == "EXP-2026-02"
        assert enriched.metadata["bio_protocol"] == "PCR Amplification"
        assert "H2O" in enriched.metadata["bio_formulas"]
        assert "NaCl" in enriched.metadata["bio_formulas"]
        assert "C6H12O6" in enriched.metadata["bio_formulas"]

    def test_alternate_labels(self, refiner):
        text = "Lab Ref: BIO-99. SOP: Standard Clean."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["bio_exp_id"] == "BIO-99"
        assert enriched.metadata["bio_protocol"] == "Standard Clean"
