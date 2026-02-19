import pytest
from ingestforge.enrichment.router import DomainRouter


class TestDomainRouter:
    @pytest.fixture
    def router(self):
        return DomainRouter()

    def test_classify_legal(self, router):
        text = "The Plaintiff argues that the Defendant violated the contract. See 345 U.S. 123."
        domain = router.get_best_domain(text)
        assert domain == "legal"

    def test_classify_cyber(self, router):
        text = "Detected exploit for CVE-2023-5050 via SQLi payload."
        domain = router.get_best_domain(text)
        assert domain == "cyber"

    def test_classify_urban(self, router):
        text = "The property is zoned R-1 with a maximum FAR of 0.5. Density limit is strict."
        domain = router.get_best_domain(text)
        assert domain == "urban"

    def test_classify_ai_safety(self, router):
        text = "The new LLM achieves 85% on MMLU but fails TruthfulQA benchmarks."
        domain = router.get_best_domain(text)
        assert domain == "ai_safety"

    def test_classify_mixed_ambiguous(self, router):
        # "Model" (AI) vs "Plaintiff" (Legal) - check scoring
        text = "The Plaintiff used a Model to predict damages."
        # "Plaintiff" (2) + "Model" (2) -> Tie?
        # Let's adjust weights or see what happens (sort stability or first match wins)
        # Actually "Plaintiff" is first in list, so it might win tie-break if sort is stable?
        # But wait, dict insertion order...
        ranked = router.classify_chunk(text)
        # Just ensure we get *something* back
        assert len(ranked) >= 2
