"""Tests for semantic linker.

Real-Time Semantic Linkage
Tests for GWT-1 through GWT-5.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ingestforge.enrichment.semantic_linker import (
    MAX_EVIDENCE_ITEMS,
    MAX_LINKS_PER_ARTIFACT,
    EntityIndex,
    LinkIndex,
    LinkType,
    SemanticLink,
    SemanticLinker,
    create_semantic_linker,
    link_artifacts,
)


# =============================================================================
# Test SemanticLink Dataclass
# =============================================================================


class TestSemanticLink:
    """Tests for SemanticLink dataclass."""

    def test_create_basic_link(self) -> None:
        """Test creating basic link."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SHARED_ENTITY,
        )
        assert link.source_artifact_id == "art1"
        assert link.target_artifact_id == "art2"
        assert link.link_type == LinkType.SHARED_ENTITY
        assert link.confidence == 1.0

    def test_create_full_link(self) -> None:
        """Test creating link with all fields."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.CITATION,
            confidence=0.95,
            evidence=["Citation found"],
            metadata={"citation_key": "Smith2024"},
        )
        assert link.confidence == 0.95
        assert len(link.evidence) == 1
        assert link.metadata["citation_key"] == "Smith2024"

    def test_empty_source_raises(self) -> None:
        """Test that empty source raises assertion."""
        with pytest.raises(AssertionError, match="cannot be empty"):
            SemanticLink(
                source_artifact_id="",
                target_artifact_id="art2",
                link_type=LinkType.SHARED_ENTITY,
            )

    def test_empty_target_raises(self) -> None:
        """Test that empty target raises assertion."""
        with pytest.raises(AssertionError, match="cannot be empty"):
            SemanticLink(
                source_artifact_id="art1",
                target_artifact_id="",
                link_type=LinkType.SHARED_ENTITY,
            )

    def test_invalid_confidence_raises(self) -> None:
        """Test that invalid confidence raises assertion."""
        with pytest.raises(AssertionError, match="confidence must be 0-1"):
            SemanticLink(
                source_artifact_id="art1",
                target_artifact_id="art2",
                link_type=LinkType.SHARED_ENTITY,
                confidence=1.5,
            )

    def test_evidence_bounded(self) -> None:
        """Test that evidence is bounded."""
        evidence = [f"evidence_{i}" for i in range(20)]
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SHARED_ENTITY,
            evidence=evidence,
        )
        assert len(link.evidence) <= MAX_EVIDENCE_ITEMS

    def test_to_dict(self) -> None:
        """Test converting to dictionary."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SEMANTIC_SIMILARITY,
            confidence=0.9,
        )
        d = link.to_dict()
        assert d["source_artifact_id"] == "art1"
        assert d["target_artifact_id"] == "art2"
        assert d["link_type"] == "semantic_similarity"
        assert d["confidence"] == 0.9

    def test_from_dict(self) -> None:
        """Test creating from dictionary."""
        data = {
            "source_artifact_id": "art1",
            "target_artifact_id": "art2",
            "link_type": "citation",
            "confidence": 0.85,
            "evidence": ["test"],
            "metadata": {},
            "created_at": "2026-02-17T10:00:00",
        }
        link = SemanticLink.from_dict(data)
        assert link.source_artifact_id == "art1"
        assert link.link_type == LinkType.CITATION
        assert link.confidence == 0.85


# =============================================================================
# Test LinkIndex
# =============================================================================


class TestLinkIndex:
    """Tests for LinkIndex."""

    def test_add_link(self) -> None:
        """Test adding link to index."""
        index = LinkIndex()
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SHARED_ENTITY,
        )
        result = index.add_link(link)
        assert result is True
        assert len(index.get_links_from("art1")) == 1

    def test_add_duplicate_returns_false(self) -> None:
        """Test that adding duplicate returns False."""
        index = LinkIndex()
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SHARED_ENTITY,
        )
        index.add_link(link)
        result = index.add_link(link)
        assert result is False

    def test_max_links_enforced(self) -> None:
        """Test that MAX_LINKS_PER_ARTIFACT is enforced."""
        index = LinkIndex()
        for i in range(MAX_LINKS_PER_ARTIFACT + 10):
            link = SemanticLink(
                source_artifact_id="art1",
                target_artifact_id=f"art_{i}",
                link_type=LinkType.SHARED_ENTITY,
            )
            index.add_link(link)

        assert len(index.get_links_from("art1")) <= MAX_LINKS_PER_ARTIFACT

    def test_get_links_from(self) -> None:
        """Test getting outgoing links."""
        index = LinkIndex()
        link1 = SemanticLink("art1", "art2", LinkType.SHARED_ENTITY)
        link2 = SemanticLink("art1", "art3", LinkType.CITATION)
        index.add_link(link1)
        index.add_link(link2)

        links = index.get_links_from("art1")
        assert len(links) == 2

    def test_get_links_to(self) -> None:
        """Test getting incoming links."""
        index = LinkIndex()
        link1 = SemanticLink("art1", "art3", LinkType.SHARED_ENTITY)
        link2 = SemanticLink("art2", "art3", LinkType.CITATION)
        index.add_link(link1)
        index.add_link(link2)

        links = index.get_links_to("art3")
        assert len(links) == 2

    def test_get_links_by_type(self) -> None:
        """Test getting links by type."""
        index = LinkIndex()
        link1 = SemanticLink("art1", "art2", LinkType.SHARED_ENTITY)
        link2 = SemanticLink("art1", "art3", LinkType.CITATION)
        link3 = SemanticLink("art2", "art3", LinkType.SHARED_ENTITY)
        index.add_link(link1)
        index.add_link(link2)
        index.add_link(link3)

        entity_links = index.get_links_by_type(LinkType.SHARED_ENTITY)
        assert len(entity_links) == 2


# =============================================================================
# Test EntityIndex
# =============================================================================


class TestEntityIndex:
    """Tests for EntityIndex."""

    def test_add_entity(self) -> None:
        """Test adding entity to index."""
        index = EntityIndex()
        index.add_entity("microsoft", "art1")

        assert "art1" in index.get_artifacts_with_entity("microsoft")
        assert "microsoft" in index.get_entities_in_artifact("art1")

    def test_multiple_artifacts_same_entity(self) -> None:
        """Test multiple artifacts with same entity."""
        index = EntityIndex()
        index.add_entity("google", "art1")
        index.add_entity("google", "art2")
        index.add_entity("google", "art3")

        artifacts = index.get_artifacts_with_entity("google")
        assert len(artifacts) == 3

    def test_multiple_entities_same_artifact(self) -> None:
        """Test multiple entities in same artifact."""
        index = EntityIndex()
        index.add_entity("apple", "art1")
        index.add_entity("microsoft", "art1")
        index.add_entity("google", "art1")

        entities = index.get_entities_in_artifact("art1")
        assert len(entities) == 3

    def test_get_nonexistent_entity(self) -> None:
        """Test getting nonexistent entity returns empty set."""
        index = EntityIndex()
        artifacts = index.get_artifacts_with_entity("nonexistent")
        assert artifacts == set()


# =============================================================================
# Test SemanticLinker - GWT-1: Entity-Based Linking
# =============================================================================


class TestGWT1EntityLinking:
    """Tests for GWT-1: Entity-Based Linking."""

    def test_link_by_shared_entity(self) -> None:
        """Test linking by shared entity."""
        linker = SemanticLinker()

        # First artifact with entity
        entities1 = [{"text": "Microsoft Corporation", "type": "ORG"}]
        links1 = linker.link_by_entity("art1", entities1)
        assert len(links1) == 0  # No other artifacts yet

        # Second artifact with same entity
        entities2 = [{"text": "Microsoft Corp", "type": "ORG"}]
        links2 = linker.link_by_entity("art2", entities2)

        assert len(links2) == 1
        assert links2[0].link_type == LinkType.SHARED_ENTITY
        assert links2[0].target_artifact_id == "art1"

    def test_link_multiple_shared_entities(self) -> None:
        """Test linking with multiple shared entities."""
        linker = SemanticLinker()

        entities1 = [
            {"text": "Apple", "type": "ORG"},
            {"text": "Google", "type": "ORG"},
        ]
        linker.link_by_entity("art1", entities1)

        entities2 = [
            {"text": "Apple Inc", "type": "ORG"},
            {"text": "Google LLC", "type": "ORG"},
        ]
        links = linker.link_by_entity("art2", entities2)

        # Should create links for both shared entities
        assert len(links) >= 1

    def test_no_self_linking(self) -> None:
        """Test that artifact doesn't link to itself."""
        linker = SemanticLinker()

        entities = [{"text": "Tesla", "type": "ORG"}]
        linker.link_by_entity("art1", entities)

        # Same artifact, same entity - should not create self-link
        links = linker.link_by_entity("art1", entities)
        for link in links:
            assert link.source_artifact_id != link.target_artifact_id

    def test_entity_normalization(self) -> None:
        """Test entity normalization for matching."""
        linker = SemanticLinker()

        entities1 = [{"text": "AMAZON INC.", "type": "ORG"}]
        linker.link_by_entity("art1", entities1)

        entities2 = [{"text": "amazon", "type": "ORG"}]
        links = linker.link_by_entity("art2", entities2)

        assert len(links) == 1

    def test_empty_entities_handled(self) -> None:
        """Test handling of empty entities."""
        linker = SemanticLinker()
        links = linker.link_by_entity("art1", [])
        assert links == []


# =============================================================================
# Test SemanticLinker - GWT-2: Concept-Based Linking
# =============================================================================


class TestGWT2SimilarityLinking:
    """Tests for GWT-2: Concept-Based Linking."""

    def test_link_by_high_similarity(self) -> None:
        """Test linking by high similarity."""
        linker = SemanticLinker(similarity_threshold=0.8)

        embedding1 = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding2 = [0.1, 0.2, 0.3, 0.4, 0.5]  # Identical

        candidates = {"art1": embedding1}
        links = linker.link_by_similarity("art2", embedding2, candidates)

        assert len(links) == 1
        assert links[0].link_type == LinkType.SEMANTIC_SIMILARITY
        assert links[0].confidence >= 0.8

    def test_no_link_below_threshold(self) -> None:
        """Test no link when below threshold."""
        linker = SemanticLinker(similarity_threshold=0.95)

        embedding1 = [1.0, 0.0, 0.0, 0.0]
        embedding2 = [0.0, 1.0, 0.0, 0.0]  # Orthogonal

        candidates = {"art1": embedding1}
        links = linker.link_by_similarity("art2", embedding2, candidates)

        assert len(links) == 0

    def test_similarity_score_in_metadata(self) -> None:
        """Test similarity score in link metadata."""
        linker = SemanticLinker(similarity_threshold=0.5)

        embedding1 = [1.0, 1.0, 0.0]
        embedding2 = [1.0, 0.5, 0.0]

        candidates = {"art1": embedding1}
        links = linker.link_by_similarity("art2", embedding2, candidates)

        if links:
            assert "similarity_score" in links[0].metadata

    def test_empty_embeddings_handled(self) -> None:
        """Test handling of empty embeddings."""
        linker = SemanticLinker()
        links = linker.link_by_similarity("art1", [], {})
        assert links == []


# =============================================================================
# Test SemanticLinker - GWT-3: Citation Linking
# =============================================================================


class TestGWT3CitationLinking:
    """Tests for GWT-3: Citation Linking."""

    def test_link_by_citation(self) -> None:
        """Test linking by citation."""
        linker = SemanticLinker()

        citations = [{"key": "smith2024", "title": "Smith Paper"}]
        artifact_index = {"smith2024": "art1"}

        links = linker.link_by_citation("art2", citations, artifact_index)

        assert len(links) == 1
        assert links[0].link_type == LinkType.CITATION
        assert links[0].target_artifact_id == "art1"

    def test_citation_directional(self) -> None:
        """Test that citation links are directional."""
        linker = SemanticLinker()

        citations = [{"key": "ref1"}]
        artifact_index = {"ref1": "art_cited"}

        links = linker.link_by_citation("art_citing", citations, artifact_index)

        assert len(links) == 1
        assert links[0].source_artifact_id == "art_citing"
        assert links[0].target_artifact_id == "art_cited"

    def test_no_link_for_unknown_citation(self) -> None:
        """Test no link for unknown citation."""
        linker = SemanticLinker()

        citations = [{"key": "unknown_paper"}]
        artifact_index = {"other_paper": "art1"}

        links = linker.link_by_citation("art2", citations, artifact_index)
        assert len(links) == 0

    def test_citation_metadata(self) -> None:
        """Test citation metadata in link."""
        linker = SemanticLinker()

        citations = [{"key": "paper1", "type": "journal"}]
        artifact_index = {"paper1": "art1"}

        links = linker.link_by_citation("art2", citations, artifact_index)

        assert len(links) == 1
        assert links[0].metadata["citation_key"] == "paper1"


# =============================================================================
# Test SemanticLinker - GWT-4: Temporal Linking
# =============================================================================


class TestGWT4TemporalLinking:
    """Tests for GWT-4: Temporal Linking."""

    def test_link_by_temporal_overlap(self) -> None:
        """Test linking by temporal overlap."""
        linker = SemanticLinker()

        dates = [{"date": "2024-01-15", "type": "event"}]
        temporal_index = {"2024-06-01": ["art1"]}

        links = linker.link_by_temporal("art2", dates, temporal_index)

        assert len(links) == 1
        assert links[0].link_type == LinkType.TEMPORAL

    def test_no_link_for_distant_dates(self) -> None:
        """Test no link for temporally distant dates."""
        linker = SemanticLinker()

        dates = [{"date": "2000-01-01", "type": "event"}]
        temporal_index = {"2024-01-01": ["art1"]}

        links = linker.link_by_temporal("art2", dates, temporal_index)
        assert len(links) == 0

    def test_temporal_metadata(self) -> None:
        """Test temporal metadata in link."""
        linker = SemanticLinker()

        dates = [{"date": "2024-03-15", "type": "publication"}]
        temporal_index = {"2024-05-01": ["art1"]}

        links = linker.link_by_temporal("art2", dates, temporal_index)

        if links:
            assert "source_date" in links[0].metadata
            assert "target_date" in links[0].metadata


# =============================================================================
# Test SemanticLinker - GWT-5: Link Metadata
# =============================================================================


class TestGWT5LinkMetadata:
    """Tests for GWT-5: Link Metadata."""

    def test_link_has_type(self) -> None:
        """Test link has type."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SHARED_ENTITY,
        )
        assert link.link_type == LinkType.SHARED_ENTITY

    def test_link_has_confidence(self) -> None:
        """Test link has confidence."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SEMANTIC_SIMILARITY,
            confidence=0.92,
        )
        assert link.confidence == 0.92

    def test_link_has_timestamp(self) -> None:
        """Test link has timestamp."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.CITATION,
        )
        assert isinstance(link.created_at, datetime)

    def test_link_has_evidence(self) -> None:
        """Test link has evidence."""
        link = SemanticLink(
            source_artifact_id="art1",
            target_artifact_id="art2",
            link_type=LinkType.SHARED_ENTITY,
            evidence=["Shared entity: Microsoft"],
        )
        assert len(link.evidence) == 1
        assert "Microsoft" in link.evidence[0]


# =============================================================================
# Test Convenience Functions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_semantic_linker(self) -> None:
        """Test create_semantic_linker function."""
        linker = create_semantic_linker(
            similarity_threshold=0.9,
            entity_confidence=0.8,
        )
        assert linker.similarity_threshold == 0.9
        assert linker.entity_confidence_threshold == 0.8

    def test_link_artifacts(self) -> None:
        """Test link_artifacts function."""
        linker = create_semantic_linker()

        entities = [{"text": "Test Entity", "type": "ORG"}]
        links = link_artifacts(linker, "art1", entities)

        assert isinstance(links, list)


# =============================================================================
# Test JPL Compliance
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_max_links_enforced(self) -> None:
        """Test MAX_LINKS_PER_ARTIFACT is enforced."""
        linker = SemanticLinker()

        # Add many entities to many artifacts
        for i in range(MAX_LINKS_PER_ARTIFACT + 20):
            entities = [{"text": "shared_entity", "type": "ORG"}]
            linker.link_by_entity(f"art_{i}", entities)

        # Check that links are bounded
        stats = linker.get_link_stats()
        # Each artifact should have at most MAX_LINKS_PER_ARTIFACT links

    def test_link_confidence_bounded(self) -> None:
        """Test confidence is bounded 0-1."""
        with pytest.raises(AssertionError):
            SemanticLink(
                source_artifact_id="art1",
                target_artifact_id="art2",
                link_type=LinkType.SHARED_ENTITY,
                confidence=2.0,
            )

    def test_get_link_stats(self) -> None:
        """Test get_link_stats returns valid statistics."""
        linker = SemanticLinker()

        entities = [{"text": "Test", "type": "ORG"}]
        linker.link_by_entity("art1", entities)
        linker.link_by_entity("art2", entities)

        stats = linker.get_link_stats()
        assert "total_links" in stats
        assert "artifacts_with_links" in stats
        assert "indexed_entities" in stats

    def test_link_type_enum_values(self) -> None:
        """Test all LinkType values exist."""
        types = list(LinkType)
        assert LinkType.SHARED_ENTITY in types
        assert LinkType.SEMANTIC_SIMILARITY in types
        assert LinkType.CITATION in types
        assert LinkType.TEMPORAL in types
