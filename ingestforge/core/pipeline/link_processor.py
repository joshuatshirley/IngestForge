"""Pipeline processor for semantic linking.

Real-Time Semantic Linkage
Epic: EP-12 (Knowledge Graph Foundry)
Feature: FE-12-02 (Real-time Semantic Linkage)

Integrates SemanticLinker into the ingestion pipeline to create
links between artifacts during processing.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.artifacts import IFArtifact
from ingestforge.enrichment.semantic_linker import (
    SemanticLink,
    SemanticLinker,
    create_semantic_linker,
)

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_ENTITIES_TO_LINK = 50
MAX_CANDIDATES_TO_CHECK = 100


@dataclass
class LinkProcessorConfig:
    """Configuration for link processor.

    Attributes:
        enable_entity_linking: Enable entity-based linking.
        enable_similarity_linking: Enable similarity-based linking.
        enable_citation_linking: Enable citation-based linking.
        enable_temporal_linking: Enable temporal linking.
        similarity_threshold: Minimum similarity for linking.
        entity_confidence: Minimum confidence for entity links.
    """

    enable_entity_linking: bool = True
    enable_similarity_linking: bool = True
    enable_citation_linking: bool = True
    enable_temporal_linking: bool = True
    similarity_threshold: float = 0.85
    entity_confidence: float = 0.7

    def __post_init__(self) -> None:
        """Validate configuration."""
        # JPL Rule #5: Assert preconditions
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert 0.0 <= self.entity_confidence <= 1.0


@dataclass
class LinkProcessorResult:
    """Result of link processing.

    Attributes:
        artifact_id: Processed artifact ID.
        links_created: Number of links created.
        links_by_type: Count of links by type.
        errors: List of error messages.
    """

    artifact_id: str
    links_created: int = 0
    links_by_type: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "artifact_id": self.artifact_id,
            "links_created": self.links_created,
            "links_by_type": self.links_by_type,
            "errors": self.errors,
        }


class IFLinkProcessor:
    """Pipeline processor for semantic linking.

    Creates links between artifacts during ingestion based on
    shared entities, similarity, citations, and temporal relationships.

    Implements the IFProcessor interface pattern for pipeline integration.

    JPL Compliance:
        - No recursion in any method
        - All loops bounded by MAX constants
        - Complete type hints
        - Precondition assertions
    """

    def __init__(
        self,
        config: Optional[LinkProcessorConfig] = None,
        linker: Optional[SemanticLinker] = None,
    ) -> None:
        """Initialize link processor.

        Args:
            config: Processor configuration.
            linker: Optional pre-configured SemanticLinker.
        """
        self.config = config or LinkProcessorConfig()
        self.linker = linker or create_semantic_linker(
            similarity_threshold=self.config.similarity_threshold,
            entity_confidence=self.config.entity_confidence,
        )

        # State for cross-artifact linking
        self._embedding_cache: Dict[str, List[float]] = {}
        self._artifact_index: Dict[str, str] = {}
        self._temporal_index: Dict[str, List[str]] = {}

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """Process artifact and create semantic links.

        Args:
            artifact: Artifact to process.

        Returns:
            Processed artifact with links in metadata.
        """
        # JPL Rule #5: Assert preconditions
        assert artifact is not None, "artifact cannot be None"
        assert artifact.artifact_id, "artifact must have an ID"

        result = LinkProcessorResult(artifact_id=artifact.artifact_id)
        all_links: List[SemanticLink] = []

        try:
            # Entity-based linking
            if self.config.enable_entity_linking:
                entity_links = self._link_by_entities(artifact)
                all_links.extend(entity_links)

            # Similarity-based linking
            if self.config.enable_similarity_linking:
                similarity_links = self._link_by_similarity(artifact)
                all_links.extend(similarity_links)

            # Citation-based linking
            if self.config.enable_citation_linking:
                citation_links = self._link_by_citations(artifact)
                all_links.extend(citation_links)

            # Temporal linking
            if self.config.enable_temporal_linking:
                temporal_links = self._link_by_temporal(artifact)
                all_links.extend(temporal_links)

        except Exception as e:
            error_msg = f"Link processing error: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)

        # Update result (always runs, even after error)
        result.links_created = len(all_links)
        for link in all_links:
            type_name = link.link_type.value
            result.links_by_type[type_name] = result.links_by_type.get(type_name, 0) + 1

        # Add links to artifact metadata (always runs)
        artifact = self._add_links_to_artifact(artifact, all_links, result)

        return artifact

    def _link_by_entities(self, artifact: IFArtifact) -> List[SemanticLink]:
        """Create entity-based links.

        Args:
            artifact: Artifact to link.

        Returns:
            List of entity links.
        """
        entities = self._extract_entities(artifact)
        if not entities:
            return []

        return self.linker.link_by_entity(
            artifact.artifact_id,
            entities[:MAX_ENTITIES_TO_LINK],
        )

    def _link_by_similarity(self, artifact: IFArtifact) -> List[SemanticLink]:
        """Create similarity-based links.

        Args:
            artifact: Artifact to link.

        Returns:
            List of similarity links.
        """
        embedding = self._get_embedding(artifact)
        if not embedding:
            return []

        # Cache this artifact's embedding
        self._embedding_cache[artifact.artifact_id] = embedding

        # Get candidate embeddings (bounded)
        candidates = dict(list(self._embedding_cache.items())[:MAX_CANDIDATES_TO_CHECK])

        return self.linker.link_by_similarity(
            artifact.artifact_id,
            embedding,
            candidates,
        )

    def _link_by_citations(self, artifact: IFArtifact) -> List[SemanticLink]:
        """Create citation-based links.

        Args:
            artifact: Artifact to link.

        Returns:
            List of citation links.
        """
        # Always index this artifact's title for future citation matching
        title = artifact.metadata.get("title", "")
        if title:
            self._artifact_index[title.lower()] = artifact.artifact_id

        # Extract and process citations
        citations = self._extract_citations(artifact)
        if not citations:
            return []

        return self.linker.link_by_citation(
            artifact.artifact_id,
            citations,
            self._artifact_index,
        )

    def _link_by_temporal(self, artifact: IFArtifact) -> List[SemanticLink]:
        """Create temporal links.

        Args:
            artifact: Artifact to link.

        Returns:
            List of temporal links.
        """
        dates = self._extract_dates(artifact)
        if not dates:
            return []

        # Update temporal index
        for date_info in dates:
            date_str = date_info.get("date", "")
            if date_str:
                if date_str not in self._temporal_index:
                    self._temporal_index[date_str] = []
                self._temporal_index[date_str].append(artifact.artifact_id)

        return self.linker.link_by_temporal(
            artifact.artifact_id,
            dates,
            self._temporal_index,
        )

    def _extract_entities(self, artifact: IFArtifact) -> List[Dict[str, Any]]:
        """Extract entities from artifact metadata.

        Args:
            artifact: Artifact to extract from.

        Returns:
            List of entity dictionaries.
        """
        entities = artifact.metadata.get("entities", [])
        if isinstance(entities, list):
            return entities[:MAX_ENTITIES_TO_LINK]
        return []

    def _get_embedding(self, artifact: IFArtifact) -> Optional[List[float]]:
        """Get embedding from artifact metadata.

        Args:
            artifact: Artifact to get embedding from.

        Returns:
            Embedding vector or None.
        """
        embedding = artifact.metadata.get("embedding")
        if isinstance(embedding, list) and all(
            isinstance(x, (int, float)) for x in embedding
        ):
            return embedding
        return None

    def _extract_citations(self, artifact: IFArtifact) -> List[Dict[str, Any]]:
        """Extract citations from artifact metadata.

        Args:
            artifact: Artifact to extract from.

        Returns:
            List of citation dictionaries.
        """
        citations = artifact.metadata.get("citations", [])
        if isinstance(citations, list):
            return citations
        return []

    def _extract_dates(self, artifact: IFArtifact) -> List[Dict[str, Any]]:
        """Extract dates from artifact metadata.

        Args:
            artifact: Artifact to extract from.

        Returns:
            List of date dictionaries.
        """
        # Check for date entities
        entities = artifact.metadata.get("entities", [])
        dates = [e for e in entities if e.get("type") == "DATE"]

        # Also check for document date
        doc_date = artifact.metadata.get("date")
        if doc_date:
            dates.append({"date": str(doc_date), "type": "document_date"})

        return dates

    def _add_links_to_artifact(
        self,
        artifact: IFArtifact,
        links: List[SemanticLink],
        result: LinkProcessorResult,
    ) -> IFArtifact:
        """Add links to artifact metadata.

        Args:
            artifact: Artifact to update.
            links: Links to add.
            result: Processing result.

        Returns:
            Updated artifact.
        """
        # Add links as metadata
        link_data = [link.to_dict() for link in links]
        artifact.metadata["semantic_links"] = link_data
        artifact.metadata["link_stats"] = result.to_dict()

        return artifact

    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics.

        Returns:
            Dictionary with processing statistics.
        """
        linker_stats = self.linker.get_link_stats()
        return {
            **linker_stats,
            "cached_embeddings": len(self._embedding_cache),
            "indexed_artifacts": len(self._artifact_index),
            "temporal_dates": len(self._temporal_index),
        }

    def clear_cache(self) -> None:
        """Clear internal caches."""
        self._embedding_cache.clear()
        self._artifact_index.clear()
        self._temporal_index.clear()


# Convenience function
def create_link_processor(
    enable_entity: bool = True,
    enable_similarity: bool = True,
    enable_citation: bool = True,
    enable_temporal: bool = True,
    similarity_threshold: float = 0.85,
) -> IFLinkProcessor:
    """Create configured link processor.

    Args:
        enable_entity: Enable entity linking.
        enable_similarity: Enable similarity linking.
        enable_citation: Enable citation linking.
        enable_temporal: Enable temporal linking.
        similarity_threshold: Similarity threshold.

    Returns:
        Configured IFLinkProcessor.
    """
    config = LinkProcessorConfig(
        enable_entity_linking=enable_entity,
        enable_similarity_linking=enable_similarity,
        enable_citation_linking=enable_citation,
        enable_temporal_linking=enable_temporal,
        similarity_threshold=similarity_threshold,
    )
    return IFLinkProcessor(config=config)
