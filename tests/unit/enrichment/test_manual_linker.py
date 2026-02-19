"""Tests for Manual Graph Linker.

Manual Graph Linker
Tests manual link creation, update, and deletion functionality.
"""

from __future__ import annotations

import pytest

from ingestforge.enrichment.manual_linker import (
    ManualGraphLinker,
    ManualLinkRequest,
    ManualLinkResult,
    create_manual_linker,
    add_manual_link,
    MAX_MANUAL_LINKS,
    MAX_LINKS_PER_ENTITY,
    MAX_RELATION_LENGTH,
)
from ingestforge.enrichment.semantic_linker import LinkType


class TestManualLinkRequest:
    """Tests for ManualLinkRequest dataclass."""

    def test_create_valid_request(self) -> None:
        """Test creating valid request."""
        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )

        assert request.source_entity == "Alice"
        assert request.target_entity == "Bob"
        assert request.relation == "knows"
        assert request.confidence == 1.0

    def test_create_request_with_all_fields(self) -> None:
        """Test creating request with all fields."""
        request = ManualLinkRequest(
            source_entity="Doc1",
            target_entity="Doc2",
            relation="cites",
            confidence=0.9,
            evidence="Found citation on page 5",
            metadata={"page": 5},
        )

        assert request.confidence == 0.9
        assert request.evidence == "Found citation on page 5"
        assert request.metadata["page"] == 5

    def test_invalid_empty_source(self) -> None:
        """Test that empty source fails validation."""
        with pytest.raises(AssertionError):
            ManualLinkRequest(
                source_entity="",
                target_entity="Bob",
                relation="knows",
            )

    def test_invalid_empty_target(self) -> None:
        """Test that empty target fails validation."""
        with pytest.raises(AssertionError):
            ManualLinkRequest(
                source_entity="Alice",
                target_entity="",
                relation="knows",
            )

    def test_invalid_empty_relation(self) -> None:
        """Test that empty relation fails validation."""
        with pytest.raises(AssertionError):
            ManualLinkRequest(
                source_entity="Alice",
                target_entity="Bob",
                relation="",
            )

    def test_invalid_confidence_too_high(self) -> None:
        """Test that confidence > 1.0 fails validation."""
        with pytest.raises(AssertionError):
            ManualLinkRequest(
                source_entity="Alice",
                target_entity="Bob",
                relation="knows",
                confidence=1.5,
            )

    def test_invalid_confidence_negative(self) -> None:
        """Test that negative confidence fails validation."""
        with pytest.raises(AssertionError):
            ManualLinkRequest(
                source_entity="Alice",
                target_entity="Bob",
                relation="knows",
                confidence=-0.1,
            )

    def test_relation_truncated_at_max_length(self) -> None:
        """Test that long relation is truncated."""
        long_relation = "x" * (MAX_RELATION_LENGTH + 50)
        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation=long_relation,
        )

        assert len(request.relation) == MAX_RELATION_LENGTH


class TestManualGraphLinker:
    """Tests for ManualGraphLinker class."""

    def test_create_linker(self) -> None:
        """Test creating linker."""
        linker = ManualGraphLinker()

        assert linker is not None
        assert len(linker.get_manual_links()) == 0

    def test_add_link_success(self) -> None:
        """Test adding a link successfully."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )

        result = linker.add_link(request)

        assert result.success is True
        assert result.link is not None
        assert result.link_id != ""
        assert "Link created" in result.message

    def test_add_link_with_custom_relation(self) -> None:
        """Test adding link with custom relation type."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Doc1",
            target_entity="Doc2",
            relation="my_custom_relation",
            confidence=0.8,
        )

        result = linker.add_link(request)

        assert result.success is True
        assert result.link.metadata["relation_string"] == "my_custom_relation"

    def test_add_link_marked_as_manual(self) -> None:
        """Test that added links are marked as manual."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )

        result = linker.add_link(request)

        assert result.link.metadata["source"] == "manual"
        assert result.link.metadata["created_by"] == "user"

    def test_add_link_with_evidence(self) -> None:
        """Test adding link with evidence."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
            evidence="Met at conference in 2024",
        )

        result = linker.add_link(request)

        assert result.success is True
        assert "Met at conference" in result.link.evidence[0]

    def test_remove_link_success(self) -> None:
        """Test removing a link successfully."""
        linker = ManualGraphLinker()

        # Add link first
        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )
        add_result = linker.add_link(request)

        # Remove it
        remove_result = linker.remove_link(add_result.link_id)

        assert remove_result.success is True
        assert "removed" in remove_result.message.lower()

    def test_remove_nonexistent_link(self) -> None:
        """Test removing a link that doesn't exist."""
        linker = ManualGraphLinker()

        result = linker.remove_link("nonexistent_id")

        assert result.success is False
        assert "not found" in result.message.lower()

    def test_update_link_confidence(self) -> None:
        """Test updating link confidence."""
        linker = ManualGraphLinker()

        # Add link
        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
            confidence=0.5,
        )
        add_result = linker.add_link(request)

        # Update confidence
        update_result = linker.update_link(add_result.link_id, confidence=0.9)

        assert update_result.success is True
        assert update_result.link.confidence == 0.9

    def test_update_link_evidence(self) -> None:
        """Test updating link evidence."""
        linker = ManualGraphLinker()

        # Add link
        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )
        add_result = linker.add_link(request)

        # Update evidence
        update_result = linker.update_link(
            add_result.link_id, evidence="New evidence found"
        )

        assert update_result.success is True
        assert "New evidence found" in update_result.link.evidence[0]

    def test_update_preserves_manual_marker(self) -> None:
        """Test that update preserves manual source marker."""
        linker = ManualGraphLinker()

        # Add link
        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )
        add_result = linker.add_link(request)

        # Update with metadata
        update_result = linker.update_link(
            add_result.link_id, metadata={"extra": "data"}
        )

        assert update_result.link.metadata["source"] == "manual"
        assert update_result.link.metadata["extra"] == "data"

    def test_update_nonexistent_link(self) -> None:
        """Test updating a link that doesn't exist."""
        linker = ManualGraphLinker()

        result = linker.update_link("nonexistent_id", confidence=0.5)

        assert result.success is False
        assert "not found" in result.message.lower()

    def test_get_link_by_id(self) -> None:
        """Test getting link by ID."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="knows",
        )
        add_result = linker.add_link(request)

        link = linker.get_link(add_result.link_id)

        assert link is not None
        assert link.source_artifact_id == "Alice"

    def test_get_nonexistent_link(self) -> None:
        """Test getting a link that doesn't exist."""
        linker = ManualGraphLinker()

        link = linker.get_link("nonexistent_id")

        assert link is None

    def test_get_links_for_entity(self) -> None:
        """Test getting links for a specific entity."""
        linker = ManualGraphLinker()

        # Add multiple links
        linker.add_link(ManualLinkRequest("Alice", "Bob", "knows"))
        linker.add_link(ManualLinkRequest("Alice", "Charlie", "works_with"))
        linker.add_link(ManualLinkRequest("Dave", "Eve", "knows"))

        alice_links = linker.get_links_for_entity("Alice")

        assert len(alice_links) == 2

    def test_get_links_for_entity_as_target(self) -> None:
        """Test getting links where entity is target."""
        linker = ManualGraphLinker()

        linker.add_link(ManualLinkRequest("Alice", "Bob", "knows"))
        linker.add_link(ManualLinkRequest("Charlie", "Bob", "knows"))

        bob_links = linker.get_links_for_entity("Bob")

        assert len(bob_links) == 2

    def test_get_statistics(self) -> None:
        """Test getting link statistics."""
        linker = ManualGraphLinker()

        linker.add_link(ManualLinkRequest("Alice", "Bob", "knows"))
        linker.add_link(ManualLinkRequest("Charlie", "Dave", "works_with"))

        stats = linker.get_statistics()

        assert stats["total_manual_links"] == 2
        assert stats["max_links_allowed"] == MAX_MANUAL_LINKS
        assert "link_type_counts" in stats

    def test_clear_all(self) -> None:
        """Test clearing all manual links."""
        linker = ManualGraphLinker()

        linker.add_link(ManualLinkRequest("Alice", "Bob", "knows"))
        linker.add_link(ManualLinkRequest("Charlie", "Dave", "knows"))

        count = linker.clear_all()

        assert count == 2
        assert len(linker.get_manual_links()) == 0


class TestLinkTypeResolution:
    """Tests for link type resolution."""

    def test_resolve_known_link_type(self) -> None:
        """Test that known link types are resolved."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Doc1",
            target_entity="Doc2",
            relation="citation",
        )

        result = linker.add_link(request)

        assert result.link.link_type == LinkType.CITATION

    def test_resolve_semantic_similarity(self) -> None:
        """Test resolving semantic_similarity type."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Doc1",
            target_entity="Doc2",
            relation="semantic_similarity",
        )

        result = linker.add_link(request)

        assert result.link.link_type == LinkType.SEMANTIC_SIMILARITY

    def test_custom_relation_defaults_to_shared_entity(self) -> None:
        """Test that unknown relations default to SHARED_ENTITY."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest(
            source_entity="Alice",
            target_entity="Bob",
            relation="my_custom_relation",
        )

        result = linker.add_link(request)

        # Uses default but preserves original in metadata
        assert result.link.link_type == LinkType.SHARED_ENTITY
        assert result.link.metadata["relation_string"] == "my_custom_relation"


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_manual_linker(self) -> None:
        """Test create_manual_linker function."""
        linker = create_manual_linker()

        assert isinstance(linker, ManualGraphLinker)

    def test_add_manual_link_function(self) -> None:
        """Test add_manual_link convenience function."""
        linker = create_manual_linker()

        result = add_manual_link(
            linker,
            source="Alice",
            target="Bob",
            relation="knows",
            confidence=0.9,
            evidence="Met at conference",
        )

        assert result.success is True
        assert result.link.confidence == 0.9


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_max_manual_links(self) -> None:
        """Test that MAX_MANUAL_LINKS bound is respected."""
        # Just verify the constant exists and is reasonable
        assert MAX_MANUAL_LINKS > 0
        assert MAX_MANUAL_LINKS <= 100000

    def test_rule_2_max_links_per_entity(self) -> None:
        """Test MAX_LINKS_PER_ENTITY bound."""
        assert MAX_LINKS_PER_ENTITY > 0
        assert MAX_LINKS_PER_ENTITY <= 1000

    def test_rule_5_null_request_rejected(self) -> None:
        """Test that None request is rejected."""
        linker = ManualGraphLinker()

        with pytest.raises(AssertionError):
            linker.add_link(None)

    def test_rule_5_empty_link_id_rejected(self) -> None:
        """Test that empty link_id is rejected."""
        linker = ManualGraphLinker()

        with pytest.raises(AssertionError):
            linker.remove_link("")

    def test_rule_5_empty_entity_id_rejected(self) -> None:
        """Test that empty entity_id is rejected."""
        linker = ManualGraphLinker()

        with pytest.raises(AssertionError):
            linker.get_links_for_entity("")

    def test_rule_7_add_link_returns_result(self) -> None:
        """Test that add_link always returns ManualLinkResult."""
        linker = ManualGraphLinker()

        request = ManualLinkRequest("Alice", "Bob", "knows")
        result = linker.add_link(request)

        assert isinstance(result, ManualLinkResult)
        assert isinstance(result.success, bool)

    def test_rule_9_type_hints(self) -> None:
        """Test that key methods have type hints."""
        import inspect

        linker = ManualGraphLinker()

        # Check add_link
        sig = inspect.signature(linker.add_link)
        assert sig.return_annotation == ManualLinkResult

        # Check remove_link
        sig = inspect.signature(linker.remove_link)
        assert sig.return_annotation == ManualLinkResult
