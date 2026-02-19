"""Manual Graph Linker for user-defined semantic links.

Manual Graph Linker
Epic: EP-12 (Knowledge Graph Foundry)
Feature: FE-12-02 (Real-time Semantic Linkage - Manual Extension)

Enables users to manually create, update, and delete semantic links
between entities when auto-detection is insufficient.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_MANUAL_LINKS)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.enrichment.semantic_linker import (
    LinkIndex,
    LinkType,
    SemanticLink,
)

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_MANUAL_LINKS = 10000  # Maximum total manual links
MAX_LINKS_PER_ENTITY = 100  # Maximum links per entity
MAX_RELATION_LENGTH = 100  # Maximum relation type string length
MAX_EVIDENCE_LENGTH = 500  # Maximum evidence string length


@dataclass
class ManualLinkRequest:
    """Request to create a manual link.

    Attributes:
        source_entity: Source entity name or ID.
        target_entity: Target entity name or ID.
        relation: Relationship type (string or LinkType).
        confidence: Confidence score (0.0 - 1.0).
        evidence: Optional evidence string.
        metadata: Additional metadata.
    """

    source_entity: str
    target_entity: str
    relation: str
    confidence: float = 1.0
    evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request fields."""
        # JPL Rule #5: Assert preconditions
        assert self.source_entity, "source_entity cannot be empty"
        assert self.target_entity, "target_entity cannot be empty"
        assert self.relation, "relation cannot be empty"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be 0-1"

        # JPL Rule #2: Enforce bounds
        self.relation = self.relation[:MAX_RELATION_LENGTH]
        self.evidence = self.evidence[:MAX_EVIDENCE_LENGTH]


@dataclass
class ManualLinkResult:
    """Result of a manual link operation.

    Attributes:
        success: Whether the operation succeeded.
        link: The created/updated link (if successful).
        message: Status or error message.
        link_id: Unique identifier for the link.
    """

    success: bool
    link: Optional[SemanticLink] = None
    message: str = ""
    link_id: str = ""


class ManualGraphLinker:
    """Manual graph linker for user-defined semantic links.

    Enables manual creation, update, and deletion of semantic links
    between entities in the knowledge graph.

    JPL Compliance:
        - No recursion in any method
        - All loops bounded by MAX constants
        - Complete type hints
        - Precondition assertions
    """

    def __init__(self, link_index: Optional[LinkIndex] = None) -> None:
        """Initialize manual graph linker.

        Args:
            link_index: Optional existing link index to use.
        """
        self.link_index = link_index or LinkIndex()
        self._manual_links: Dict[str, SemanticLink] = {}
        self._link_counter = 0

    def _generate_link_id(self, source: str, target: str, relation: str) -> str:
        """Generate unique link ID.

        Args:
            source: Source entity.
            target: Target entity.
            relation: Relationship type.

        Returns:
            Unique link identifier.
        """
        self._link_counter += 1
        return f"manual_{source}_{target}_{relation}_{self._link_counter}"

    def _resolve_link_type(self, relation: str) -> LinkType:
        """Resolve relation string to LinkType.

        Args:
            relation: Relation string.

        Returns:
            Matching LinkType or default.
        """
        # Try to match known LinkTypes
        relation_lower = relation.lower().replace("-", "_").replace(" ", "_")

        for link_type in LinkType:
            if link_type.value == relation_lower:
                return link_type

        # Default to SHARED_ENTITY for custom relations
        return LinkType.SHARED_ENTITY

    def add_link(self, request: ManualLinkRequest) -> ManualLinkResult:
        """Add a manual semantic link.

        Args:
            request: Link creation request.

        Returns:
            Result with created link or error.

        Rule #4: Function <60 lines
        Rule #5: Assert preconditions
        Rule #7: Check return values
        """
        # JPL Rule #5: Assert preconditions
        assert request is not None, "request cannot be None"

        # JPL Rule #2: Enforce bounds
        if len(self._manual_links) >= MAX_MANUAL_LINKS:
            return ManualLinkResult(
                success=False,
                message=f"Maximum manual links ({MAX_MANUAL_LINKS}) reached",
            )

        # Check per-entity limit
        existing_count = len(self.link_index.get_links_from(request.source_entity))
        if existing_count >= MAX_LINKS_PER_ENTITY:
            return ManualLinkResult(
                success=False,
                message=f"Maximum links per entity ({MAX_LINKS_PER_ENTITY}) reached",
            )

        # Resolve link type
        link_type = self._resolve_link_type(request.relation)

        # Build metadata with manual marker
        metadata = dict(request.metadata)
        metadata["source"] = "manual"
        metadata["relation_string"] = request.relation
        metadata["created_by"] = "user"

        # Create link
        link = SemanticLink(
            source_artifact_id=request.source_entity,
            target_artifact_id=request.target_entity,
            link_type=link_type,
            confidence=request.confidence,
            evidence=[request.evidence] if request.evidence else ["Manual link"],
            metadata=metadata,
            created_at=datetime.now(timezone.utc),
        )

        # Add to index
        if not self.link_index.add_link(link):
            return ManualLinkResult(
                success=False,
                message="Link already exists or could not be added",
            )

        # Generate and store link ID
        link_id = self._generate_link_id(
            request.source_entity, request.target_entity, request.relation
        )
        self._manual_links[link_id] = link

        logger.info(
            f"Added manual link: {request.source_entity} --[{request.relation}]--> "
            f"{request.target_entity}"
        )

        return ManualLinkResult(
            success=True,
            link=link,
            message="Link created successfully",
            link_id=link_id,
        )

    def remove_link(self, link_id: str) -> ManualLinkResult:
        """Remove a manual link by ID.

        Args:
            link_id: Link identifier to remove.

        Returns:
            Result indicating success or failure.

        Rule #4: Function <60 lines
        Rule #5: Assert preconditions
        """
        # JPL Rule #5: Assert preconditions
        assert link_id, "link_id cannot be empty"

        if link_id not in self._manual_links:
            return ManualLinkResult(
                success=False,
                message=f"Link not found: {link_id}",
            )

        link = self._manual_links.pop(link_id)

        # Remove from index
        source_links = self.link_index.by_source.get(link.source_artifact_id, [])
        self.link_index.by_source[link.source_artifact_id] = [
            l for l in source_links if l != link
        ]

        target_links = self.link_index.by_target.get(link.target_artifact_id, [])
        self.link_index.by_target[link.target_artifact_id] = [
            l for l in target_links if l != link
        ]

        type_links = self.link_index.by_type.get(link.link_type, [])
        self.link_index.by_type[link.link_type] = [l for l in type_links if l != link]

        logger.info(f"Removed manual link: {link_id}")

        return ManualLinkResult(
            success=True,
            link=link,
            message="Link removed successfully",
            link_id=link_id,
        )

    def update_link(
        self,
        link_id: str,
        confidence: Optional[float] = None,
        evidence: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManualLinkResult:
        """Update an existing manual link.

        Args:
            link_id: Link identifier to update.
            confidence: New confidence score.
            evidence: New evidence string.
            metadata: Additional metadata to merge.

        Returns:
            Result with updated link or error.

        Rule #4: Function <60 lines
        Rule #5: Assert preconditions
        """
        # JPL Rule #5: Assert preconditions
        assert link_id, "link_id cannot be empty"

        if link_id not in self._manual_links:
            return ManualLinkResult(
                success=False,
                message=f"Link not found: {link_id}",
            )

        link = self._manual_links[link_id]

        # Update fields
        if confidence is not None:
            assert 0.0 <= confidence <= 1.0, "confidence must be 0-1"
            link.confidence = confidence

        if evidence is not None:
            evidence = evidence[:MAX_EVIDENCE_LENGTH]
            link.evidence = [evidence]

        if metadata is not None:
            link.metadata.update(metadata)
            link.metadata["source"] = "manual"  # Preserve manual marker

        logger.info(f"Updated manual link: {link_id}")

        return ManualLinkResult(
            success=True,
            link=link,
            message="Link updated successfully",
            link_id=link_id,
        )

    def get_link(self, link_id: str) -> Optional[SemanticLink]:
        """Get a manual link by ID.

        Args:
            link_id: Link identifier.

        Returns:
            SemanticLink if found, None otherwise.
        """
        return self._manual_links.get(link_id)

    def get_manual_links(self) -> List[SemanticLink]:
        """Get all manual links.

        Returns:
            List of all manual links.
        """
        return list(self._manual_links.values())

    def get_links_for_entity(self, entity_id: str) -> List[SemanticLink]:
        """Get all manual links involving an entity.

        Args:
            entity_id: Entity identifier.

        Returns:
            List of links where entity is source or target.
        """
        # JPL Rule #5: Assert preconditions
        assert entity_id, "entity_id cannot be empty"

        result: List[SemanticLink] = []

        for link in self._manual_links.values():
            if link.source_artifact_id == entity_id:
                result.append(link)
            elif link.target_artifact_id == entity_id:
                result.append(link)

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about manual links.

        Returns:
            Dictionary with link statistics.
        """
        type_counts: Dict[str, int] = {}
        for link in self._manual_links.values():
            relation = link.metadata.get("relation_string", link.link_type.value)
            type_counts[relation] = type_counts.get(relation, 0) + 1

        return {
            "total_manual_links": len(self._manual_links),
            "link_type_counts": type_counts,
            "max_links_allowed": MAX_MANUAL_LINKS,
            "links_remaining": MAX_MANUAL_LINKS - len(self._manual_links),
        }

    def clear_all(self) -> int:
        """Remove all manual links.

        Returns:
            Number of links removed.
        """
        count = len(self._manual_links)

        # Clear from index
        for link in self._manual_links.values():
            source_links = self.link_index.by_source.get(link.source_artifact_id, [])
            self.link_index.by_source[link.source_artifact_id] = [
                l for l in source_links if l.metadata.get("source") != "manual"
            ]

        self._manual_links.clear()
        logger.info(f"Cleared {count} manual links")

        return count


# Convenience functions
def create_manual_linker(link_index: Optional[LinkIndex] = None) -> ManualGraphLinker:
    """Create a manual graph linker.

    Args:
        link_index: Optional existing link index.

    Returns:
        Configured ManualGraphLinker instance.
    """
    return ManualGraphLinker(link_index=link_index)


def add_manual_link(
    linker: ManualGraphLinker,
    source: str,
    target: str,
    relation: str,
    confidence: float = 1.0,
    evidence: str = "",
) -> ManualLinkResult:
    """Add a manual link using convenience function.

    Args:
        linker: ManualGraphLinker instance.
        source: Source entity.
        target: Target entity.
        relation: Relationship type.
        confidence: Confidence score.
        evidence: Optional evidence.

    Returns:
        ManualLinkResult with outcome.
    """
    request = ManualLinkRequest(
        source_entity=source,
        target_entity=target,
        relation=relation,
        confidence=confidence,
        evidence=evidence,
    )
    return linker.add_link(request)
