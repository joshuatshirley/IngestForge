import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.gaming import GamingMetadataRefiner


class TestGamingMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return GamingMetadataRefiner()

    def test_enrich_patch_notes(self, refiner):
        text = """
        Patch v1.4.2 Notes:
        Champion: Jinx
        Base Health: 580 -> 610
        Attack Damage: +5%
        Armor: 28 -> 32
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["gaming_patch_version"] == "1.4.2"
        assert "Jinx" in enriched.metadata["gaming_characters"]
        # Stat changes check
        stats = enriched.metadata["gaming_stat_changes"]
        assert any("Base Health: 580 -> 610" in s for s in stats)
        assert any("Attack Damage: +5%" in s for s in stats)
        assert any("Armor: 28 -> 32" in s for s in stats)

    def test_alternate_labels(self, refiner):
        text = "Version: 2.0a. Hero: Master Chief. Movement Speed: -10."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["gaming_patch_version"] == "2.0a"
        assert "Master Chief" in enriched.metadata["gaming_characters"]
        assert any(
            "Movement Speed: -10" in s for s in enriched.metadata["gaming_stat_changes"]
        )
