import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.urban import UrbanMetadataRefiner


class TestUrbanMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return UrbanMetadataRefiner()

    def test_enrich_zoning_ordinance(self, refiner):
        text = """
        The subject property is located in the R-1 district.
        It has a permitted FAR of 1.5.
        Target density is 12 units/acre.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["urban_zoning_code"] == "R-1"
        assert enriched.metadata["urban_far_ratio"] == 1.5
        assert enriched.metadata["urban_density_target"] == "12 units/acre"

    def test_alternate_codes(self, refiner):
        text = "Zoning District: C2. Floor Area Ratio: 4.0. Target: High Density."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["urban_zoning_code"] == "C2"
        assert enriched.metadata["urban_far_ratio"] == 4.0
        assert enriched.metadata["urban_density_target"] == "High Density"
