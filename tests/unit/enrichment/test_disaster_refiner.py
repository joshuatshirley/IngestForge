import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.disaster import DisasterMetadataRefiner


class TestDisasterMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return DisasterMetadataRefiner()

    def test_enrich_sitrep(self, refiner):
        text = """
        SITREP: Wildfire in sector 7.
        Incident Type: Brush Fire.
        Coordinates: 34.0522, -118.2437.
        Urgency: Critical.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["disaster_incident_type"] == "Brush fire"
        assert enriched.metadata["disaster_coordinates"] == "34.0522, -118.2437"
        assert enriched.metadata["disaster_urgency"] == "Critical"

    def test_negative_coordinates(self, refiner):
        text = "Search rescue at -33.8688, 151.2093 needed immediately."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["disaster_coordinates"] == "-33.8688, 151.2093"
