import pytest
from unittest.mock import MagicMock
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.core.config import Config


class TestQueryRoutingRetrieval:
    @pytest.fixture
    def mock_storage(self):
        return MagicMock()

    @pytest.fixture
    def config(self):
        c = Config()
        c.retrieval.hybrid.bm25_weight = 0.5
        c.retrieval.hybrid.semantic_weight = 0.5
        return c

    def test_retrieval_weight_adjustment(self, config, mock_storage):
        """Test that weights shift based on query domain."""
        retriever = HybridRetriever(config, mock_storage)

        # 1. Neutral Query (Default weights 0.5/0.5)
        with retriever._temporary_weights(query_intent=None, domain_strategy=None):
            assert retriever.bm25_weight == 0.5

        # 2. Cyber Query (bm25_modifier=1.3)
        from ingestforge.query.domain_classifier import QueryDomainClassifier

        classifier = QueryDomainClassifier()
        strategy = classifier.get_query_strategy("Show me CVE-2024-1234")

        assert strategy.name == "cyber"

        with retriever._temporary_weights(query_intent=None, domain_strategy=strategy):
            # Cyber bm25_mod=1.3, sem_mod=1.0
            # New weights: 0.5*1.3 = 0.65, 0.5*1.0 = 0.5
            # Total: 1.15. Normalized: 0.65/1.15 = 0.565, 0.5/1.15 = 0.435
            assert retriever.bm25_weight > 0.55
            assert retriever.semantic_weight < 0.45

    def test_search_integration_flow(self, config, mock_storage):
        """Verify the search method calls classification."""
        retriever = HybridRetriever(config, mock_storage)
        retriever._execute_searches = MagicMock(return_value=([], []))

        # Should not raise
        retriever.search("Zoning rules for R-1", use_rerank=False)

        # Verify it handled classification (indirectly by checking logger or mocking classifier)
        # For now, just ensuring it runs through the new logic
        assert retriever._execute_searches.called

    def test_metadata_field_boosting(self, config, mock_storage):
        """Verify that specific metadata fields increase result ranking."""
        from ingestforge.retrieval.rescorer import MetadataRescorer
        from ingestforge.query.routing import DomainStrategy
        from ingestforge.storage.base import SearchResult

        rescorer = MetadataRescorer()
        strategy = DomainStrategy(name="auto", boost_fields={"auto_part_number": 3.0})

        # Result A: Good semantic match, no metadata
        res_a = SearchResult(
            chunk_id="a",
            content="Engine parts",
            score=0.8,
            document_id="doc1",
            section_title="Intro",
            chunk_type="text",
            source_file="file1.txt",
            word_count=10,
        )

        # Result B: Lower semantic match, but has matching Part Number
        res_b = SearchResult(
            chunk_id="b",
            content="Details for HP-99",
            score=0.5,
            document_id="doc2",
            section_title="Details",
            chunk_type="text",
            source_file="file2.txt",
            word_count=10,
            metadata={"auto_part_number": "HP-99"},
        )

        results = [res_a, res_b]
        query = "Find HP-99 details"

        rescored = rescorer.rescore(results, query, strategy)

        # B should now be #1 because of 3.0x boost (0.5 * 3 = 1.5 > 0.8)
        assert rescored[0].chunk_id == "b"
        assert rescored[0].score == 1.5
