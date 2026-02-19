import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.wellness import WellnessMetadataRefiner


class TestWellnessMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return WellnessMetadataRefiner()

    def test_enrich_meal_plan(self, refiner):
        text = """
        Breakfast: Oatmeal with berries.
        Total Calories: 350 kcal.
        Macros: Protein: 12g, Fat: 5.5g, Carbs: 45g.
        Allergens: May contain traces of Nuts.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["wellness_calories"] == 350.0
        assert enriched.metadata["wellness_macros"]["Protein"] == 12.0
        assert enriched.metadata["wellness_macros"]["Fat"] == 5.5
        assert enriched.metadata["wellness_macros"]["Carbs"] == 45.0
        assert "Nuts" in enriched.metadata["wellness_allergens"]

    def test_shorthand_macros(self, refiner):
        text = "Snack: 150 calories. 10g protein, 2g fat."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["wellness_calories"] == 150.0
        assert enriched.metadata["wellness_macros"]["Protein"] == 10.0
