import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.museum import MuseumMetadataRefiner


class TestMuseumMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return MuseumMetadataRefiner()

    def test_enrich_catalog_entry(self, refiner):
        text = """
        Accession Number: 2024.15.1
        Artist: Vincent van Gogh
        Medium: Oil on canvas
        Era: Post-Impressionism
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["museum_accession_id"] == "2024.15.1"
        assert enriched.metadata["museum_artist"] == "Vincent van Gogh"
        assert enriched.metadata["museum_medium"] == "Oil on canvas"
        assert enriched.metadata["museum_era"] == "Post-impressionism"

    def test_alternate_labels(self, refiner):
        text = "ID: 1999.5.2. Maker: Leonardo da Vinci. Material: Marble."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["museum_accession_id"] == "1999.5.2"
        assert enriched.metadata["museum_artist"] == "Leonardo da Vinci"
        assert enriched.metadata["museum_medium"] == "Marble"
