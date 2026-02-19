import pytest
from ingestforge.query.domain_classifier import QueryDomainClassifier


class TestQueryDomainClassifier:
    @pytest.fixture
    def classifier(self):
        return QueryDomainClassifier()

    def test_classify_legal_query(self, classifier):
        query = "What did the court say in 345 U.S. 123 about the Plaintiff?"
        domains = classifier.classify_query(query)
        assert "legal" in domains

        strategy = classifier.get_query_strategy(query)
        assert strategy.name in ["legal", "merged"]
        assert strategy.bm25_modifier > 1.0

    def test_classify_cyber_query(self, classifier):
        query = "Show me all exploits for CVE-2024-5050."
        domains = classifier.classify_query(query)
        assert "cyber" in domains

        strategy = classifier.get_query_strategy(query)
        assert strategy.boost_fields["cve_id"] > 1.0

    def test_classify_urban_query(self, classifier):
        query = "Zoning rules for R-1 districts and FAR limits."
        domains = classifier.classify_query(query)
        assert "urban" in domains

    def test_classify_ai_safety_query(self, classifier):
        query = "How does GPT-4o perform on the MMLU benchmark?"
        domains = classifier.classify_query(query)
        assert "ai_safety" in domains

    def test_classify_wellness_query(self, classifier):
        query = "What is the protein content and calories in a banana?"
        domains = classifier.classify_query(query)
        assert "wellness" in domains
