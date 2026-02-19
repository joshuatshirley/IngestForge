import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.ai_safety import AISafetyMetadataRefiner


class TestAISafetyMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return AISafetyMetadataRefiner()

    def test_enrich_model_card(self, refiner):
        text = """
        Model: GPT-4o
        Scale: 1.8 Trillion parameters.
        Performance: TruthfulQA score of 85.5% and MMLU: 88.2.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["ai_model_name"] == "GPT-4o"
        assert enriched.metadata["ai_param_count"] == "1.8 TRILLION"
        assert enriched.metadata["ai_safety_benchmarks"]["TruthfulQA"] == 85.5
        assert enriched.metadata["ai_safety_benchmarks"]["MMLU"] == 88.2

    def test_shorthand_params(self, refiner):
        text = "LLM: Llama-3. Scale: 70B params."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["ai_model_name"] == "Llama-3"
        assert enriched.metadata["ai_param_count"] == "70B"
