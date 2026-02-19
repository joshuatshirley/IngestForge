import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.spiritual import SpiritualMetadataRefiner


class TestSpiritualMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return SpiritualMetadataRefiner()

    def test_enrich_bible_citation(self, refiner):
        text = "In the beginning was the Word (John 1:1). This speaks of Grace."
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["spiritual_book"] == "John"
        assert enriched.metadata["spiritual_citation"] == "John 1:1"
        assert "Grace" in enriched.metadata["spiritual_themes"]

    def test_enrich_surah_citation(self, refiner):
        text = "Recite in the name of your Lord (Surah 96:1). Divine wisdom is key."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["spiritual_book"] == "Surah"
        assert enriched.metadata["spiritual_citation"] == "Surah 96:1"
        assert "Wisdom" in enriched.metadata["spiritual_themes"]

    def test_numbered_book(self, refiner):
        text = "Faith without works is dead (2 James 2:26)."
        chunk = ChunkRecord(chunk_id="c3", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["spiritual_book"] == "2 James"
        assert enriched.metadata["spiritual_citation"] == "2 James 2:26"
