import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.rpg import RPGMetadataRefiner


class TestRPGMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return RPGMetadataRefiner()

    def test_enrich_monster_stats(self, refiner):
        text = """
Goblin
Small humanoid (goblinoid), neutral evil
Armor Class: 15 (leather armor, shield)
Hit Points: 7 (2d6)
Speed: 30 ft.
Challenge Rating: 1/4
"""
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["rpg_stats"]["hp"] == 7
        assert enriched.metadata["rpg_stats"]["ac"] == 15
        assert enriched.metadata["rpg_stats"]["cr"] == "1/4"

    def test_detect_location(self, refiner):
        text = "The Rusty Tankard is a Village Tavern located in the center of town."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["rpg_type"] == "Location"
