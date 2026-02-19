import pytest
from ingestforge.query.classifier import DomainClassifier, QueryClassifier
from ingestforge.query.routing import get_merged_strategy


class TestDomainRouting:
    @pytest.fixture
    def domain_classifier(self):
        return DomainClassifier()

    def test_classify_legal_query(self, domain_classifier):
        query = "What is the jurisdiction for docket 20-1234?"
        domains = domain_classifier.classify(query)
        assert "legal" in domains

    def test_classify_tech_query(self, domain_classifier):
        query = "How to implement tree-sitter imports in python?"
        domains = domain_classifier.classify(query)
        assert "tech" in domains

    def test_classify_cyber_query(self, domain_classifier):
        query = "Check for CVE-2024-12345 in cloudtrail logs"
        domains = domain_classifier.classify(query)
        assert "cyber" in domains

    def test_merged_strategy(self):
        # Ambiguous query matching both tech and cyber
        domains = ["tech", "cyber"]
        strategy = get_merged_strategy(domains)

        # mods should be averaged: (1.1 + 1.3) / 2 = 1.2
        assert strategy.bm25_modifier == pytest.approx(1.2)
        # required metadata should be union
        assert "imports" in strategy.required_metadata
        assert "cve_id" in strategy.required_metadata

    def test_query_classifier_full(self):
        qc = QueryClassifier()
        query = "Who is the judge in this legal case?"
        classification = qc.classify_full(query)

        assert classification.intent == "factual"
        assert "legal" in classification.domains
