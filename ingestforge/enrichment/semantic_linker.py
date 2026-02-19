"""Real-time semantic linking between documents.

Real-Time Semantic Linkage
Epic: EP-12 (Knowledge Graph Foundry)
Feature: FE-12-02 (Real-time Semantic Linkage)

Enables automatic linking between documents during ingestion based on:
- Shared entities (same normalized entity in multiple docs)
- Semantic similarity (high embedding cosine similarity)
- Citations (one document references another)
- Temporal relationships (overlapping time periods)

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_LINKS, MAX_CANDIDATES)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_LINKS_PER_ARTIFACT = 100  # Maximum links from one artifact
MAX_SIMILARITY_CANDIDATES = 50  # Maximum candidates to check for similarity
MAX_ENTITY_LINKS = 20  # Maximum links per shared entity
MAX_CITATION_DEPTH = 3  # Maximum citation chain depth
MAX_TEMPORAL_WINDOW_DAYS = 365  # Maximum temporal linking window
MAX_EVIDENCE_ITEMS = 10  # Maximum evidence items per link


class LinkType(Enum):
    """Types of semantic links between artifacts."""

    SHARED_ENTITY = "shared_entity"
    SEMANTIC_SIMILARITY = "semantic_similarity"
    CITATION = "citation"
    TEMPORAL = "temporal"
    CAUSAL = "causal"
    CONTRADICTION = "contradiction"
    CONTINUATION = "continuation"


@dataclass
class SemanticLink:
    """A semantic link between two artifacts.

    Attributes:
        source_artifact_id: Source artifact identifier.
        target_artifact_id: Target artifact identifier.
        link_type: Type of semantic relationship.
        confidence: Confidence score (0.0 - 1.0).
        evidence: List of evidence strings supporting the link.
        metadata: Additional link metadata.
        created_at: Timestamp when link was created.
    """

    source_artifact_id: str
    target_artifact_id: str
    link_type: LinkType
    confidence: float = 1.0
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        """Validate link fields."""
        # JPL Rule #5: Assert preconditions
        assert self.source_artifact_id, "source_artifact_id cannot be empty"
        assert self.target_artifact_id, "target_artifact_id cannot be empty"
        assert isinstance(self.link_type, LinkType), "link_type must be LinkType"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be 0-1"
        # JPL Rule #2: Enforce bounds
        self.evidence = self.evidence[:MAX_EVIDENCE_ITEMS]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dictionary with all link fields.
        """
        return {
            "source_artifact_id": self.source_artifact_id,
            "target_artifact_id": self.target_artifact_id,
            "link_type": self.link_type.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticLink":
        """Create from dictionary.

        Args:
            data: Dictionary representation.

        Returns:
            SemanticLink instance.
        """
        return cls(
            source_artifact_id=data["source_artifact_id"],
            target_artifact_id=data["target_artifact_id"],
            link_type=LinkType(data["link_type"]),
            confidence=data.get("confidence", 1.0),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(timezone.utc),
        )


@dataclass
class LinkIndex:
    """Index for efficient link lookup.

    Attributes:
        by_source: Links indexed by source artifact.
        by_target: Links indexed by target artifact.
        by_type: Links indexed by link type.
        by_entity: Links indexed by shared entity.
    """

    by_source: Dict[str, List[SemanticLink]] = field(default_factory=dict)
    by_target: Dict[str, List[SemanticLink]] = field(default_factory=dict)
    by_type: Dict[LinkType, List[SemanticLink]] = field(default_factory=dict)
    by_entity: Dict[str, List[SemanticLink]] = field(default_factory=dict)

    def add_link(self, link: SemanticLink) -> bool:
        """Add link to index.

        Args:
            link: Link to add.

        Returns:
            True if added, False if duplicate.
        """
        # JPL Rule #5: Assert preconditions
        assert link is not None, "link cannot be None"

        # Check for duplicate
        existing = self.by_source.get(link.source_artifact_id, [])
        for existing_link in existing:
            if (
                existing_link.target_artifact_id == link.target_artifact_id
                and existing_link.link_type == link.link_type
            ):
                return False  # Duplicate

        # JPL Rule #2: Enforce bounds
        if len(existing) >= MAX_LINKS_PER_ARTIFACT:
            logger.warning(f"Max links reached for artifact {link.source_artifact_id}")
            return False

        # Add to indexes
        if link.source_artifact_id not in self.by_source:
            self.by_source[link.source_artifact_id] = []
        self.by_source[link.source_artifact_id].append(link)

        if link.target_artifact_id not in self.by_target:
            self.by_target[link.target_artifact_id] = []
        self.by_target[link.target_artifact_id].append(link)

        if link.link_type not in self.by_type:
            self.by_type[link.link_type] = []
        self.by_type[link.link_type].append(link)

        return True

    def get_links_from(self, artifact_id: str) -> List[SemanticLink]:
        """Get all links from an artifact.

        Args:
            artifact_id: Source artifact ID.

        Returns:
            List of outgoing links.
        """
        return self.by_source.get(artifact_id, [])

    def get_links_to(self, artifact_id: str) -> List[SemanticLink]:
        """Get all links to an artifact.

        Args:
            artifact_id: Target artifact ID.

        Returns:
            List of incoming links.
        """
        return self.by_target.get(artifact_id, [])

    def get_links_by_type(self, link_type: LinkType) -> List[SemanticLink]:
        """Get all links of a specific type.

        Args:
            link_type: Type of links to retrieve.

        Returns:
            List of links of the specified type.
        """
        return self.by_type.get(link_type, [])


@dataclass
class EntityIndex:
    """Index mapping entities to artifacts.

    Attributes:
        entity_to_artifacts: Maps normalized entity to artifact IDs.
        artifact_to_entities: Maps artifact ID to its entities.
    """

    entity_to_artifacts: Dict[str, Set[str]] = field(default_factory=dict)
    artifact_to_entities: Dict[str, Set[str]] = field(default_factory=dict)

    def add_entity(self, entity_name: str, artifact_id: str) -> None:
        """Add entity-artifact mapping.

        Args:
            entity_name: Normalized entity name.
            artifact_id: Artifact containing the entity.
        """
        # JPL Rule #5: Assert preconditions
        assert entity_name, "entity_name cannot be empty"
        assert artifact_id, "artifact_id cannot be empty"

        if entity_name not in self.entity_to_artifacts:
            self.entity_to_artifacts[entity_name] = set()
        self.entity_to_artifacts[entity_name].add(artifact_id)

        if artifact_id not in self.artifact_to_entities:
            self.artifact_to_entities[artifact_id] = set()
        self.artifact_to_entities[artifact_id].add(entity_name)

    def get_artifacts_with_entity(self, entity_name: str) -> Set[str]:
        """Get all artifacts containing an entity.

        Args:
            entity_name: Normalized entity name.

        Returns:
            Set of artifact IDs.
        """
        return self.entity_to_artifacts.get(entity_name, set())

    def get_entities_in_artifact(self, artifact_id: str) -> Set[str]:
        """Get all entities in an artifact.

        Args:
            artifact_id: Artifact ID.

        Returns:
            Set of entity names.
        """
        return self.artifact_to_entities.get(artifact_id, set())


class SemanticLinker:
    """Real-time semantic linker for multi-document joins.

    Creates links between artifacts based on shared entities,
    semantic similarity, citations, and temporal relationships.

    JPL Compliance:
        - No recursion in any method
        - All loops bounded by MAX constants
        - Complete type hints
        - Precondition assertions
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        entity_confidence_threshold: float = 0.7,
    ) -> None:
        """Initialize semantic linker.

        Args:
            similarity_threshold: Minimum cosine similarity for linking.
            entity_confidence_threshold: Minimum confidence for entity links.
        """
        # JPL Rule #5: Assert preconditions
        assert 0.0 <= similarity_threshold <= 1.0, "threshold must be 0-1"
        assert 0.0 <= entity_confidence_threshold <= 1.0, "threshold must be 0-1"

        self.similarity_threshold = similarity_threshold
        self.entity_confidence_threshold = entity_confidence_threshold
        self.link_index = LinkIndex()
        self.entity_index = EntityIndex()

    def link_by_entity(
        self,
        artifact_id: str,
        entities: List[Dict[str, Any]],
    ) -> List[SemanticLink]:
        """Create links based on shared entities.

        Args:
            artifact_id: Current artifact ID.
            entities: List of entity dictionaries with 'text' and 'type' keys.

        Returns:
            List of new semantic links.
        """
        # JPL Rule #5: Assert preconditions
        assert artifact_id, "artifact_id cannot be empty"
        assert entities is not None, "entities cannot be None"

        links: List[SemanticLink] = []

        # Process entities (bounded)
        for entity in entities[:MAX_ENTITY_LINKS]:
            entity_name = self._normalize_entity(entity.get("text", ""))
            if not entity_name:
                continue

            # Find other artifacts with this entity
            other_artifacts = self.entity_index.get_artifacts_with_entity(entity_name)

            # Create links to other artifacts (bounded)
            link_count = 0
            for other_id in other_artifacts:
                if other_id == artifact_id:
                    continue
                if link_count >= MAX_ENTITY_LINKS:
                    break

                link = SemanticLink(
                    source_artifact_id=artifact_id,
                    target_artifact_id=other_id,
                    link_type=LinkType.SHARED_ENTITY,
                    confidence=self.entity_confidence_threshold,
                    evidence=[f"Shared entity: {entity_name}"],
                    metadata={"entity": entity_name, "entity_type": entity.get("type")},
                )

                if self.link_index.add_link(link):
                    links.append(link)
                    link_count += 1

            # Add entity to index
            self.entity_index.add_entity(entity_name, artifact_id)

        return links

    def link_by_similarity(
        self,
        artifact_id: str,
        embedding: List[float],
        candidate_embeddings: Dict[str, List[float]],
    ) -> List[SemanticLink]:
        """Create links based on semantic similarity.

        Args:
            artifact_id: Current artifact ID.
            embedding: Embedding vector for current artifact.
            candidate_embeddings: Map of artifact IDs to embeddings.

        Returns:
            List of new semantic links.
        """
        # JPL Rule #5: Assert preconditions
        assert artifact_id, "artifact_id cannot be empty"
        assert embedding is not None, "embedding cannot be None"
        assert candidate_embeddings is not None, "candidates cannot be None"

        links: List[SemanticLink] = []

        # Check candidates (bounded)
        candidates = list(candidate_embeddings.items())[:MAX_SIMILARITY_CANDIDATES]

        for other_id, other_embedding in candidates:
            if other_id == artifact_id:
                continue

            similarity = self._cosine_similarity(embedding, other_embedding)

            if similarity >= self.similarity_threshold:
                link = SemanticLink(
                    source_artifact_id=artifact_id,
                    target_artifact_id=other_id,
                    link_type=LinkType.SEMANTIC_SIMILARITY,
                    confidence=similarity,
                    evidence=[f"Cosine similarity: {similarity:.3f}"],
                    metadata={"similarity_score": similarity},
                )

                if self.link_index.add_link(link):
                    links.append(link)

        return links

    def link_by_citation(
        self,
        artifact_id: str,
        citations: List[Dict[str, Any]],
        artifact_index: Dict[str, str],
    ) -> List[SemanticLink]:
        """Create links based on citations.

        Args:
            artifact_id: Current artifact ID.
            citations: List of citation dictionaries.
            artifact_index: Map of citation keys to artifact IDs.

        Returns:
            List of new semantic links.
        """
        # JPL Rule #5: Assert preconditions
        assert artifact_id, "artifact_id cannot be empty"
        assert citations is not None, "citations cannot be None"

        links: List[SemanticLink] = []

        for citation in citations[:MAX_LINKS_PER_ARTIFACT]:
            citation_key = citation.get("key") or citation.get("title", "")
            if not citation_key:
                continue

            # Look up cited artifact
            cited_artifact = artifact_index.get(citation_key)
            if not cited_artifact or cited_artifact == artifact_id:
                continue

            link = SemanticLink(
                source_artifact_id=artifact_id,
                target_artifact_id=cited_artifact,
                link_type=LinkType.CITATION,
                confidence=citation.get("confidence", 0.9),
                evidence=[f"Citation: {citation_key}"],
                metadata={
                    "citation_key": citation_key,
                    "citation_type": citation.get("type", "reference"),
                },
            )

            if self.link_index.add_link(link):
                links.append(link)

        return links

    def link_by_temporal(
        self,
        artifact_id: str,
        dates: List[Dict[str, Any]],
        temporal_index: Dict[str, List[str]],
    ) -> List[SemanticLink]:
        """Create links based on temporal relationships.

        Args:
            artifact_id: Current artifact ID.
            dates: List of date dictionaries with 'date' and 'type' keys.
            temporal_index: Map of date strings to artifact IDs.

        Returns:
            List of new semantic links.
        """
        # JPL Rule #5: Assert preconditions
        assert artifact_id, "artifact_id cannot be empty"
        assert dates is not None, "dates cannot be None"

        links: List[SemanticLink] = []
        linked_artifacts: Set[str] = set()

        for date_info in dates[:MAX_LINKS_PER_ARTIFACT]:
            date_str = date_info.get("date", "")
            if not date_str:
                continue

            # Find artifacts with overlapping dates
            for indexed_date, artifact_ids in temporal_index.items():
                if not self._dates_overlap(date_str, indexed_date):
                    continue

                for other_id in artifact_ids:
                    if other_id == artifact_id or other_id in linked_artifacts:
                        continue

                    link = SemanticLink(
                        source_artifact_id=artifact_id,
                        target_artifact_id=other_id,
                        link_type=LinkType.TEMPORAL,
                        confidence=0.8,
                        evidence=[f"Temporal overlap: {date_str} ~ {indexed_date}"],
                        metadata={
                            "source_date": date_str,
                            "target_date": indexed_date,
                        },
                    )

                    if self.link_index.add_link(link):
                        links.append(link)
                        linked_artifacts.add(other_id)

                    if len(linked_artifacts) >= MAX_ENTITY_LINKS:
                        break

        return links

    def get_all_links(self, artifact_id: str) -> List[SemanticLink]:
        """Get all links for an artifact (incoming and outgoing).

        Args:
            artifact_id: Artifact ID.

        Returns:
            List of all related links.
        """
        outgoing = self.link_index.get_links_from(artifact_id)
        incoming = self.link_index.get_links_to(artifact_id)
        return outgoing + incoming

    def get_link_stats(self) -> Dict[str, Any]:
        """Get statistics about the link index.

        Returns:
            Dictionary with link statistics.
        """
        total_links = sum(len(links) for links in self.link_index.by_source.values())
        type_counts = {
            link_type.value: len(links)
            for link_type, links in self.link_index.by_type.items()
        }

        return {
            "total_links": total_links,
            "artifacts_with_links": len(self.link_index.by_source),
            "link_type_counts": type_counts,
            "indexed_entities": len(self.entity_index.entity_to_artifacts),
        }

    def _normalize_entity(self, text: str) -> str:
        """Normalize entity text for matching.

        Args:
            text: Raw entity text.

        Returns:
            Normalized entity string.
        """
        if not text:
            return ""

        # Lowercase and strip
        normalized = text.lower().strip()

        # Remove common suffixes
        suffixes = [
            " inc",
            " inc.",
            " incorporated",
            " corp",
            " corp.",
            " corporation",
            " llc",
            " ltd",
            " limited",
            " co",
            " co.",
            " company",
        ]
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[: -len(suffix)]

        return normalized.strip()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            Cosine similarity (0.0 - 1.0).
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _dates_overlap(self, date1: str, date2: str) -> bool:
        """Check if two date strings are close enough to link.

        Args:
            date1: First date string.
            date2: Second date string.

        Returns:
            True if dates are within MAX_TEMPORAL_WINDOW_DAYS.
        """
        # Simple year-based comparison
        try:
            year1 = int(date1[:4]) if len(date1) >= 4 else 0
            year2 = int(date2[:4]) if len(date2) >= 4 else 0

            if year1 == 0 or year2 == 0:
                return False

            return abs(year1 - year2) <= (MAX_TEMPORAL_WINDOW_DAYS // 365)
        except (ValueError, TypeError):
            return False


# Convenience functions
def create_semantic_linker(
    similarity_threshold: float = 0.85,
    entity_confidence: float = 0.7,
) -> SemanticLinker:
    """Create a semantic linker with specified thresholds.

    Args:
        similarity_threshold: Minimum similarity for linking.
        entity_confidence: Minimum confidence for entity links.

    Returns:
        Configured SemanticLinker instance.
    """
    return SemanticLinker(
        similarity_threshold=similarity_threshold,
        entity_confidence_threshold=entity_confidence,
    )


def link_artifacts(
    linker: SemanticLinker,
    artifact_id: str,
    entities: List[Dict[str, Any]],
    embedding: Optional[List[float]] = None,
    candidate_embeddings: Optional[Dict[str, List[float]]] = None,
    citations: Optional[List[Dict[str, Any]]] = None,
    artifact_index: Optional[Dict[str, str]] = None,
) -> List[SemanticLink]:
    """Link an artifact using all available methods.

    Args:
        linker: SemanticLinker instance.
        artifact_id: Artifact to link.
        entities: Extracted entities.
        embedding: Artifact embedding.
        candidate_embeddings: Other artifact embeddings.
        citations: Extracted citations.
        artifact_index: Citation key to artifact ID map.

    Returns:
        All created links.
    """
    all_links: List[SemanticLink] = []

    # Entity-based linking
    if entities:
        entity_links = linker.link_by_entity(artifact_id, entities)
        all_links.extend(entity_links)

    # Similarity-based linking
    if embedding and candidate_embeddings:
        similarity_links = linker.link_by_similarity(
            artifact_id, embedding, candidate_embeddings
        )
        all_links.extend(similarity_links)

    # Citation-based linking
    if citations and artifact_index:
        citation_links = linker.link_by_citation(artifact_id, citations, artifact_index)
        all_links.extend(citation_links)

    return all_links
