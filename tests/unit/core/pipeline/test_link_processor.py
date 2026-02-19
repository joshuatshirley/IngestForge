"""Tests for IFLinkProcessor.

Real-Time Semantic Linkage
Epic: EP-12 (Knowledge Graph Foundry)
Feature: FE-12-02 (Real-time Semantic Linkage)

GWT-based tests for link processor pipeline integration.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from ingestforge.core.pipeline.artifacts import IFTextArtifact
from ingestforge.core.pipeline.link_processor import (
    MAX_CANDIDATES_TO_CHECK,
    MAX_ENTITIES_TO_LINK,
    IFLinkProcessor,
    LinkProcessorConfig,
    LinkProcessorResult,
    create_link_processor,
)
from ingestforge.enrichment.semantic_linker import (
    LinkType,
    SemanticLink,
    SemanticLinker,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_artifact(
    artifact_id: str,
    metadata: Dict[str, Any] | None = None,
) -> IFTextArtifact:
    """Create test artifact using IFTextArtifact."""
    return IFTextArtifact(
        artifact_id=artifact_id,
        content=f"Test content for {artifact_id}",
        metadata=metadata or {},
    )


def make_link(
    source: str,
    target: str,
    link_type: LinkType = LinkType.SHARED_ENTITY,
    confidence: float = 0.9,
) -> SemanticLink:
    """Create test semantic link."""
    return SemanticLink(
        source_artifact_id=source,
        target_artifact_id=target,
        link_type=link_type,
        confidence=confidence,
    )


# =============================================================================
# TestLinkProcessorConfig
# =============================================================================


class TestLinkProcessorConfig:
    """Tests for LinkProcessorConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = LinkProcessorConfig()

        assert config.enable_entity_linking is True
        assert config.enable_similarity_linking is True
        assert config.enable_citation_linking is True
        assert config.enable_temporal_linking is True
        assert config.similarity_threshold == 0.85
        assert config.entity_confidence == 0.7

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = LinkProcessorConfig(
            enable_entity_linking=False,
            enable_similarity_linking=True,
            similarity_threshold=0.9,
            entity_confidence=0.8,
        )

        assert config.enable_entity_linking is False
        assert config.similarity_threshold == 0.9
        assert config.entity_confidence == 0.8

    def test_invalid_similarity_threshold_low(self) -> None:
        """Test assertion for invalid similarity threshold (too low)."""
        with pytest.raises(AssertionError):
            LinkProcessorConfig(similarity_threshold=-0.1)

    def test_invalid_similarity_threshold_high(self) -> None:
        """Test assertion for invalid similarity threshold (too high)."""
        with pytest.raises(AssertionError):
            LinkProcessorConfig(similarity_threshold=1.5)

    def test_invalid_entity_confidence_low(self) -> None:
        """Test assertion for invalid entity confidence (too low)."""
        with pytest.raises(AssertionError):
            LinkProcessorConfig(entity_confidence=-0.1)

    def test_invalid_entity_confidence_high(self) -> None:
        """Test assertion for invalid entity confidence (too high)."""
        with pytest.raises(AssertionError):
            LinkProcessorConfig(entity_confidence=1.5)


# =============================================================================
# TestLinkProcessorResult
# =============================================================================


class TestLinkProcessorResult:
    """Tests for LinkProcessorResult dataclass."""

    def test_default_values(self) -> None:
        """Test default result values."""
        result = LinkProcessorResult(artifact_id="art-001")

        assert result.artifact_id == "art-001"
        assert result.links_created == 0
        assert result.links_by_type == {}
        assert result.errors == []

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        result = LinkProcessorResult(
            artifact_id="art-001",
            links_created=5,
            links_by_type={"shared_entity": 3, "citation": 2},
            errors=["warning1"],
        )

        d = result.to_dict()

        assert d["artifact_id"] == "art-001"
        assert d["links_created"] == 5
        assert d["links_by_type"] == {"shared_entity": 3, "citation": 2}
        assert d["errors"] == ["warning1"]

    def test_accumulate_link_types(self) -> None:
        """Test accumulating link types."""
        result = LinkProcessorResult(artifact_id="art-001")
        result.links_by_type["shared_entity"] = 2
        result.links_by_type["citation"] = 1
        result.links_created = 3

        assert result.links_created == 3
        assert result.links_by_type["shared_entity"] == 2


# =============================================================================
# TestIFLinkProcessorInit
# =============================================================================


class TestIFLinkProcessorInit:
    """Tests for IFLinkProcessor initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        processor = IFLinkProcessor()

        assert processor.config is not None
        assert processor.linker is not None
        assert processor._embedding_cache == {}
        assert processor._artifact_index == {}
        assert processor._temporal_index == {}

    def test_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = LinkProcessorConfig(similarity_threshold=0.9)
        processor = IFLinkProcessor(config=config)

        assert processor.config.similarity_threshold == 0.9

    def test_custom_linker(self) -> None:
        """Test initialization with custom linker."""
        mock_linker = MagicMock(spec=SemanticLinker)
        processor = IFLinkProcessor(linker=mock_linker)

        assert processor.linker is mock_linker


# =============================================================================
# TestIFLinkProcessorProcess - GWT-1: Entity Linking
# =============================================================================


class TestGWT1EntityLinking:
    """GWT-1: Entity-based linking in processor."""

    def test_given_artifact_with_entities_when_processed_then_links_created(
        self,
    ) -> None:
        """Test entity-based linking creates links."""
        # Given
        artifact = make_artifact(
            "art-001",
            metadata={
                "entities": [
                    {"text": "Einstein", "type": "PERSON", "confidence": 0.9},
                    {"text": "MIT", "type": "ORG", "confidence": 0.85},
                ]
            },
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = [
            make_link("art-001", "art-002", LinkType.SHARED_ENTITY)
        ]
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)

        # When
        result = processor.process(artifact)

        # Then
        assert "semantic_links" in result.metadata
        assert len(result.metadata["semantic_links"]) == 1
        mock_linker.link_by_entity.assert_called_once()

    def test_entity_linking_disabled(self) -> None:
        """Test entity linking can be disabled."""
        config = LinkProcessorConfig(enable_entity_linking=False)
        artifact = make_artifact(
            "art-001",
            metadata={"entities": [{"text": "Einstein", "type": "PERSON"}]},
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(config=config, linker=mock_linker)
        processor.process(artifact)

        mock_linker.link_by_entity.assert_not_called()

    def test_max_entities_bounded(self) -> None:
        """Test entity count is bounded by MAX_ENTITIES_TO_LINK."""
        # Create more entities than the max
        entities = [
            {"text": f"Entity{i}", "type": "PERSON"}
            for i in range(MAX_ENTITIES_TO_LINK + 10)
        ]
        artifact = make_artifact("art-001", metadata={"entities": entities})

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        processor.process(artifact)

        # Check entities were bounded
        call_args = mock_linker.link_by_entity.call_args
        passed_entities = call_args[0][1]
        assert len(passed_entities) <= MAX_ENTITIES_TO_LINK


# =============================================================================
# TestIFLinkProcessorProcess - GWT-2: Similarity Linking
# =============================================================================


class TestGWT2SimilarityLinking:
    """GWT-2: Similarity-based linking in processor."""

    def test_given_artifact_with_embedding_when_processed_then_similarity_links(
        self,
    ) -> None:
        """Test similarity-based linking with embeddings."""
        # Given
        embedding = [0.1, 0.2, 0.3, 0.4]
        artifact = make_artifact("art-001", metadata={"embedding": embedding})

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = [
            make_link("art-001", "art-002", LinkType.SEMANTIC_SIMILARITY, 0.92)
        ]
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)

        # When
        result = processor.process(artifact)

        # Then
        assert len(result.metadata["semantic_links"]) == 1
        mock_linker.link_by_similarity.assert_called_once()

    def test_embedding_cached_for_future_comparisons(self) -> None:
        """Test embeddings are cached for cross-artifact comparison."""
        embedding = [0.1, 0.2, 0.3]
        artifact = make_artifact("art-001", metadata={"embedding": embedding})

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        processor.process(artifact)

        assert "art-001" in processor._embedding_cache
        assert processor._embedding_cache["art-001"] == embedding

    def test_similarity_linking_disabled(self) -> None:
        """Test similarity linking can be disabled."""
        config = LinkProcessorConfig(enable_similarity_linking=False)
        artifact = make_artifact("art-001", metadata={"embedding": [0.1, 0.2, 0.3]})

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(config=config, linker=mock_linker)
        processor.process(artifact)

        mock_linker.link_by_similarity.assert_not_called()

    def test_invalid_embedding_ignored(self) -> None:
        """Test non-numeric embeddings are ignored."""
        artifact = make_artifact("art-001", metadata={"embedding": ["not", "numeric"]})

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        processor.process(artifact)

        # Should not call similarity linking with invalid embedding
        mock_linker.link_by_similarity.assert_not_called()


# =============================================================================
# TestIFLinkProcessorProcess - GWT-3: Citation Linking
# =============================================================================


class TestGWT3CitationLinking:
    """GWT-3: Citation-based linking in processor."""

    def test_given_artifact_with_citations_when_processed_then_citation_links(
        self,
    ) -> None:
        """Test citation-based linking."""
        artifact = make_artifact(
            "art-001",
            metadata={
                "citations": [
                    {"title": "Referenced Paper", "doi": "10.1234/test"},
                ]
            },
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = [
            make_link("art-001", "art-002", LinkType.CITATION)
        ]
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        result = processor.process(artifact)

        assert len(result.metadata["semantic_links"]) == 1
        mock_linker.link_by_citation.assert_called_once()

    def test_artifact_title_indexed(self) -> None:
        """Test artifact titles are indexed for citation matching."""
        artifact = make_artifact(
            "art-001",
            metadata={
                "title": "My Research Paper",
                "citations": [],
            },
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        processor.process(artifact)

        assert "my research paper" in processor._artifact_index
        assert processor._artifact_index["my research paper"] == "art-001"

    def test_citation_linking_disabled(self) -> None:
        """Test citation linking can be disabled."""
        config = LinkProcessorConfig(enable_citation_linking=False)
        artifact = make_artifact(
            "art-001",
            metadata={"citations": [{"title": "Test"}]},
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(config=config, linker=mock_linker)
        processor.process(artifact)

        mock_linker.link_by_citation.assert_not_called()


# =============================================================================
# TestIFLinkProcessorProcess - GWT-4: Temporal Linking
# =============================================================================


class TestGWT4TemporalLinking:
    """GWT-4: Temporal-based linking in processor."""

    def test_given_artifact_with_dates_when_processed_then_temporal_links(
        self,
    ) -> None:
        """Test temporal linking with date entities."""
        artifact = make_artifact(
            "art-001",
            metadata={
                "entities": [{"date": "2024-01-15", "type": "DATE"}],
            },
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = [
            make_link("art-001", "art-002", LinkType.TEMPORAL)
        ]

        processor = IFLinkProcessor(linker=mock_linker)
        result = processor.process(artifact)

        assert len(result.metadata["semantic_links"]) == 1
        mock_linker.link_by_temporal.assert_called_once()

    def test_document_date_extracted(self) -> None:
        """Test document-level date is extracted for temporal linking."""
        artifact = make_artifact(
            "art-001",
            metadata={"date": "2024-06-15"},
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        processor.process(artifact)

        # Check temporal linking was called with date info
        call_args = mock_linker.link_by_temporal.call_args
        dates = call_args[0][1]
        assert len(dates) > 0
        assert any(d.get("date") == "2024-06-15" for d in dates)

    def test_temporal_linking_disabled(self) -> None:
        """Test temporal linking can be disabled."""
        config = LinkProcessorConfig(enable_temporal_linking=False)
        artifact = make_artifact(
            "art-001",
            metadata={"date": "2024-01-01"},
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = []
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []

        processor = IFLinkProcessor(config=config, linker=mock_linker)
        processor.process(artifact)

        mock_linker.link_by_temporal.assert_not_called()


# =============================================================================
# TestIFLinkProcessorProcess - GWT-5: Link Metadata
# =============================================================================


class TestGWT5LinkMetadata:
    """GWT-5: Link metadata in processor results."""

    def test_link_stats_added_to_artifact(self) -> None:
        """Test link statistics are added to artifact metadata."""
        # Provide metadata so helper methods don't return early
        artifact = make_artifact(
            "art-001",
            metadata={
                "entities": [{"text": "Test", "type": "PERSON"}],
                "citations": [{"title": "Cited Paper"}],
            },
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.return_value = [
            make_link("art-001", "art-002", LinkType.SHARED_ENTITY),
            make_link("art-001", "art-003", LinkType.SHARED_ENTITY),
        ]
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = [
            make_link("art-001", "art-004", LinkType.CITATION),
        ]
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        result = processor.process(artifact)

        assert "link_stats" in result.metadata
        stats = result.metadata["link_stats"]
        assert stats["links_created"] == 3
        assert stats["links_by_type"]["shared_entity"] == 2
        assert stats["links_by_type"]["citation"] == 1

    def test_links_serialized_to_dict(self) -> None:
        """Test links are serialized as dictionaries."""
        # Provide entities so linker gets called
        artifact = make_artifact(
            "art-001",
            metadata={"entities": [{"text": "Test", "type": "PERSON"}]},
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        link = make_link("art-001", "art-002", LinkType.SHARED_ENTITY, 0.95)
        mock_linker.link_by_entity.return_value = [link]
        mock_linker.link_by_similarity.return_value = []
        mock_linker.link_by_citation.return_value = []
        mock_linker.link_by_temporal.return_value = []

        processor = IFLinkProcessor(linker=mock_linker)
        result = processor.process(artifact)

        links = result.metadata["semantic_links"]
        assert len(links) == 1
        assert isinstance(links[0], dict)
        assert links[0]["source_artifact_id"] == "art-001"
        assert links[0]["target_artifact_id"] == "art-002"

    def test_errors_recorded_on_failure(self) -> None:
        """Test errors are recorded when processing fails."""
        # Provide entities so linker gets called and can raise error
        artifact = make_artifact(
            "art-001",
            metadata={"entities": [{"text": "Test", "type": "PERSON"}]},
        )

        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.link_by_entity.side_effect = ValueError("Test error")

        processor = IFLinkProcessor(linker=mock_linker)
        result = processor.process(artifact)

        assert "link_stats" in result.metadata
        stats = result.metadata["link_stats"]
        assert len(stats["errors"]) > 0
        assert "Test error" in stats["errors"][0]


# =============================================================================
# TestIFLinkProcessorHelpers
# =============================================================================


class TestIFLinkProcessorHelpers:
    """Tests for processor helper methods."""

    def test_extract_entities_returns_list(self) -> None:
        """Test entity extraction returns list."""
        artifact = make_artifact(
            "art-001",
            metadata={
                "entities": [
                    {"text": "Einstein", "type": "PERSON"},
                    {"text": "MIT", "type": "ORG"},
                ]
            },
        )

        processor = IFLinkProcessor()
        entities = processor._extract_entities(artifact)

        assert len(entities) == 2
        assert entities[0]["text"] == "Einstein"

    def test_extract_entities_handles_non_list(self) -> None:
        """Test entity extraction handles non-list metadata."""
        artifact = make_artifact(
            "art-001",
            metadata={"entities": "not a list"},
        )

        processor = IFLinkProcessor()
        entities = processor._extract_entities(artifact)

        assert entities == []

    def test_get_embedding_returns_vector(self) -> None:
        """Test embedding extraction returns vector."""
        embedding = [0.1, 0.2, 0.3, 0.4]
        artifact = make_artifact("art-001", metadata={"embedding": embedding})

        processor = IFLinkProcessor()
        result = processor._get_embedding(artifact)

        assert result == embedding

    def test_get_embedding_handles_missing(self) -> None:
        """Test embedding extraction handles missing embedding."""
        artifact = make_artifact("art-001", metadata={})

        processor = IFLinkProcessor()
        result = processor._get_embedding(artifact)

        assert result is None

    def test_extract_citations_returns_list(self) -> None:
        """Test citation extraction returns list."""
        artifact = make_artifact(
            "art-001",
            metadata={"citations": [{"title": "Paper A"}, {"title": "Paper B"}]},
        )

        processor = IFLinkProcessor()
        citations = processor._extract_citations(artifact)

        assert len(citations) == 2

    def test_extract_dates_combines_sources(self) -> None:
        """Test date extraction combines entity and document dates."""
        artifact = make_artifact(
            "art-001",
            metadata={
                "entities": [{"date": "2024-01-15", "type": "DATE"}],
                "date": "2024-06-01",
            },
        )

        processor = IFLinkProcessor()
        dates = processor._extract_dates(artifact)

        assert len(dates) == 2


# =============================================================================
# TestIFLinkProcessorStats
# =============================================================================


class TestIFLinkProcessorStats:
    """Tests for processor statistics."""

    def test_get_stats_returns_combined_stats(self) -> None:
        """Test get_stats returns combined statistics."""
        mock_linker = MagicMock(spec=SemanticLinker)
        mock_linker.get_link_stats.return_value = {
            "total_links": 10,
            "links_by_type": {"shared_entity": 5, "citation": 5},
        }

        processor = IFLinkProcessor(linker=mock_linker)
        processor._embedding_cache["art-001"] = [0.1, 0.2]
        processor._artifact_index["paper a"] = "art-001"

        stats = processor.get_stats()

        assert stats["total_links"] == 10
        assert stats["cached_embeddings"] == 1
        assert stats["indexed_artifacts"] == 1

    def test_clear_cache_empties_all_caches(self) -> None:
        """Test clear_cache empties all internal caches."""
        processor = IFLinkProcessor()
        processor._embedding_cache["art-001"] = [0.1, 0.2]
        processor._artifact_index["paper a"] = "art-001"
        processor._temporal_index["2024-01-01"] = ["art-001"]

        processor.clear_cache()

        assert processor._embedding_cache == {}
        assert processor._artifact_index == {}
        assert processor._temporal_index == {}


# =============================================================================
# TestCreateLinkProcessor
# =============================================================================


class TestCreateLinkProcessor:
    """Tests for create_link_processor convenience function."""

    def test_creates_default_processor(self) -> None:
        """Test creating processor with defaults."""
        processor = create_link_processor()

        assert processor.config.enable_entity_linking is True
        assert processor.config.enable_similarity_linking is True
        assert processor.config.similarity_threshold == 0.85

    def test_creates_custom_processor(self) -> None:
        """Test creating processor with custom options."""
        processor = create_link_processor(
            enable_entity=False,
            enable_similarity=True,
            similarity_threshold=0.9,
        )

        assert processor.config.enable_entity_linking is False
        assert processor.config.similarity_threshold == 0.9

    def test_selective_link_types(self) -> None:
        """Test creating processor with selective link types."""
        processor = create_link_processor(
            enable_entity=True,
            enable_similarity=False,
            enable_citation=True,
            enable_temporal=False,
        )

        assert processor.config.enable_entity_linking is True
        assert processor.config.enable_similarity_linking is False
        assert processor.config.enable_citation_linking is True
        assert processor.config.enable_temporal_linking is False


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_max_entities_bound(self) -> None:
        """JPL Rule #2: Verify MAX_ENTITIES_TO_LINK bound."""
        assert MAX_ENTITIES_TO_LINK > 0
        assert MAX_ENTITIES_TO_LINK <= 100  # Reasonable upper bound

    def test_rule_2_max_candidates_bound(self) -> None:
        """JPL Rule #2: Verify MAX_CANDIDATES_TO_CHECK bound."""
        assert MAX_CANDIDATES_TO_CHECK > 0
        assert MAX_CANDIDATES_TO_CHECK <= 200  # Reasonable upper bound

    def test_rule_5_precondition_null_artifact(self) -> None:
        """JPL Rule #5: Assert preconditions for null artifact."""
        processor = IFLinkProcessor()

        with pytest.raises(AssertionError):
            processor.process(None)

    def test_rule_5_precondition_missing_id(self) -> None:
        """JPL Rule #5: Assert preconditions for missing artifact ID."""
        processor = IFLinkProcessor()
        artifact = make_artifact("")  # Empty ID

        with pytest.raises(AssertionError):
            processor.process(artifact)

    def test_rule_9_type_hints_present(self) -> None:
        """JPL Rule #9: Verify type hints on key methods."""
        import inspect

        processor = IFLinkProcessor()

        # Check process method
        sig = inspect.signature(processor.process)
        assert sig.return_annotation != inspect.Parameter.empty

        # Check get_stats method
        sig = inspect.signature(processor.get_stats)
        assert sig.return_annotation != inspect.Parameter.empty


# =============================================================================
# TestProcessorIntegration
# =============================================================================


class TestProcessorIntegration:
    """Integration tests for processor with real linker."""

    def test_full_processing_flow(self) -> None:
        """Test complete processing flow with multiple link types."""
        artifact = make_artifact(
            "art-001",
            metadata={
                "title": "Research Paper A",
                "entities": [{"text": "Einstein", "type": "PERSON", "confidence": 0.9}],
                "embedding": [0.1, 0.2, 0.3],
                "citations": [{"title": "Paper B"}],
                "date": "2024-01-15",
            },
        )

        # Use real linker (no links will be found without pre-indexed data)
        processor = create_link_processor()
        result = processor.process(artifact)

        # Should have metadata fields even with no links
        assert "semantic_links" in result.metadata
        assert "link_stats" in result.metadata

        # Caches should be populated
        assert "art-001" in processor._embedding_cache
        assert "research paper a" in processor._artifact_index

    def test_multi_artifact_processing(self) -> None:
        """Test processing multiple artifacts builds indexes."""
        processor = create_link_processor()

        # Process first artifact
        art1 = make_artifact(
            "art-001",
            metadata={
                "title": "Paper A",
                "embedding": [0.1, 0.2, 0.3],
            },
        )
        processor.process(art1)

        # Process second artifact
        art2 = make_artifact(
            "art-002",
            metadata={
                "title": "Paper B",
                "embedding": [0.15, 0.25, 0.35],
            },
        )
        processor.process(art2)

        # Both should be indexed
        assert len(processor._embedding_cache) == 2
        assert len(processor._artifact_index) == 2
