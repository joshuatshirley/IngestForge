"""
Proactive Scout Agent for Knowledge Gap Detection.

Proactive Scout
Analyzes KnowledgeManifest to identify gaps and suggest discovery tasks.

NASA JPL Power of Ten compliant.
Rule #2: All loops bounded.
Rule #4: All functions <60 lines.
Rule #9: 100% type hints.
"""

import uuid
from typing import List
from dataclasses import dataclass

from ingestforge.core.pipeline.knowledge_manifest import (
    IFKnowledgeManifest,
    ManifestEntry,
)
from ingestforge.core.pipeline.artifacts import IFDiscoveryIntentArtifact
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS (JPL Rule #2: Fixed upper bounds)
# =============================================================================

MAX_ENTITIES_TO_ANALYZE = 50  # AC: Top 50 most central entities
MAX_DISCOVERY_INTENTS = 100  # Maximum intents to generate per session
MIN_REFERENCES_FOR_WEAK_NODE = 2  # Threshold for weak connectivity
MIN_CONFIDENCE = 0.6  # Minimum confidence for discovery intent
MAX_SUGGESTED_TERMS = 5  # Maximum search terms per intent

# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class GapAnalysisResult:
    """
    Result of gap analysis for a single entity.

    Rule #9: Complete type hints.
    """

    entity_hash: str
    entity_text: str
    entity_type: str
    reference_count: int
    document_count: int
    is_dangling: bool
    is_weak: bool
    priority_score: float
    rationale: str


# =============================================================================
# PROACTIVE SCOUT IMPLEMENTATION
# =============================================================================


class ProactiveScout:
    """
    Background agent that identifies knowledge gaps and suggests discoveries.

    Proactive Scout
    - Scans KnowledgeManifest for dangling nodes (0 cross-document links)
    - Scans for weak nodes (<2 references)
    - Generates prioritized discovery intents

    NASA JPL Power of Ten compliant.
    """

    def __init__(self) -> None:
        """Initialize scout."""
        self._session_id: str = str(uuid.uuid4())

    def analyze_manifest(
        self, manifest: IFKnowledgeManifest
    ) -> List[IFDiscoveryIntentArtifact]:
        """
        Analyze knowledge manifest for gaps and generate discovery intents.

        AC: Cap analysis to top 50 most central entities.

        Rule #2: Bounded iteration.
        Rule #4: Under 60 lines.

        Args:
            manifest: Active knowledge manifest to analyze.

        Returns:
            List of discovery intent artifacts (max 100).
        """
        # JPL Rule #5: Assert preconditions
        if not manifest.is_active:
            logger.warning("Manifest is not active, cannot analyze")
            return []

        # Get top entities by reference count
        top_entities = self._get_top_entities(manifest)

        # Analyze each entity for gaps
        gap_results = self._analyze_entities(top_entities)

        # Generate discovery intents
        intents = self._generate_intents(gap_results)

        logger.info(
            f"Proactive Scout: Found {len(intents)} discovery intents "
            f"from {len(top_entities)} analyzed entities"
        )

        return intents

    def _get_top_entities(self, manifest: IFKnowledgeManifest) -> List[ManifestEntry]:
        """
        Get top N entities sorted by reference count.

        Rule #2: Bounded to MAX_ENTITIES_TO_ANALYZE.
        Rule #4: Under 60 lines.

        Args:
            manifest: Knowledge manifest.

        Returns:
            List of top entities (max 50).
        """
        # Access manifest entities (thread-safe)
        all_entities = manifest.get_all_entities()

        # Sort by reference count (descending)
        sorted_entities = sorted(
            all_entities, key=lambda e: len(e.references), reverse=True
        )

        # JPL Rule #2: Cap to maximum
        entities_to_analyze = min(len(sorted_entities), MAX_ENTITIES_TO_ANALYZE)

        return sorted_entities[:entities_to_analyze]

    def _analyze_entities(
        self, entities: List[ManifestEntry]
    ) -> List[GapAnalysisResult]:
        """
        Analyze entities for gaps.

        Rule #2: Bounded iteration.
        Rule #4: Under 60 lines.

        Args:
            entities: List of entities to analyze.

        Returns:
            List of gap analysis results.
        """
        results: List[GapAnalysisResult] = []

        # JPL Rule #2: Bounded iteration
        entity_count = min(len(entities), MAX_ENTITIES_TO_ANALYZE)

        for i in range(entity_count):
            entity = entities[i]

            # Check for dangling node (0 cross-document links)
            is_dangling = not entity.is_cross_document

            # Check for weak node (<2 references)
            is_weak = len(entity.references) < MIN_REFERENCES_FOR_WEAK_NODE

            # Calculate priority score
            priority = self._calculate_priority(entity, is_dangling, is_weak)

            # Generate rationale
            rationale = self._generate_rationale(entity, is_dangling, is_weak)

            results.append(
                GapAnalysisResult(
                    entity_hash=entity.entity_hash,
                    entity_text=entity.entity_text,
                    entity_type=entity.entity_type,
                    reference_count=len(entity.references),
                    document_count=entity.document_count,
                    is_dangling=is_dangling,
                    is_weak=is_weak,
                    priority_score=priority,
                    rationale=rationale,
                )
            )

        return results

    def _calculate_priority(
        self, entity: ManifestEntry, is_dangling: bool, is_weak: bool
    ) -> float:
        """
        Calculate priority score for an entity.

        Rule #4: Under 60 lines.

        Args:
            entity: Manifest entry.
            is_dangling: True if entity has 0 cross-document links.
            is_weak: True if entity has <2 references.

        Returns:
            Priority score (0.0-1.0).
        """
        # Base priority on reference count (more refs = higher priority)
        ref_count = len(entity.references)
        base_priority = min(ref_count / 10.0, 0.5)

        # Boost for dangling nodes (need cross-document links)
        if is_dangling:
            base_priority += 0.3

        # Boost for weak nodes (need more references)
        elif is_weak:
            base_priority += 0.2

        # Cap at 1.0
        return min(base_priority, 1.0)

    def _generate_rationale(
        self, entity: ManifestEntry, is_dangling: bool, is_weak: bool
    ) -> str:
        """
        Generate human-readable rationale for discovery.

        Rule #4: Under 60 lines.

        Args:
            entity: Manifest entry.
            is_dangling: True if dangling node.
            is_weak: True if weak node.

        Returns:
            Rationale string.
        """
        ref_count = len(entity.references)
        doc_count = entity.document_count

        if is_dangling:
            return (
                f"'{entity.entity_text}' appears {ref_count} time(s) but only in "
                f"one document. Finding cross-document connections would strengthen "
                f"the knowledge graph."
            )
        elif is_weak:
            return (
                f"'{entity.entity_text}' has only {ref_count} reference(s) across "
                f"{doc_count} document(s). More context would improve understanding."
            )
        else:
            return (
                f"'{entity.entity_text}' appears {ref_count} time(s) in {doc_count} "
                f"document(s). Additional sources could reveal new connections."
            )

    def _generate_intents(
        self, gap_results: List[GapAnalysisResult]
    ) -> List[IFDiscoveryIntentArtifact]:
        """
        Generate discovery intent artifacts from gap analysis.

        Rule #2: Bounded to MAX_DISCOVERY_INTENTS.
        Rule #4: Under 60 lines.

        Args:
            gap_results: Gap analysis results.

        Returns:
            List of discovery intent artifacts.
        """
        intents: List[IFDiscoveryIntentArtifact] = []

        # Filter to gaps with sufficient confidence
        valid_gaps = [g for g in gap_results if g.priority_score >= MIN_CONFIDENCE]

        # Sort by priority (descending)
        sorted_gaps = sorted(valid_gaps, key=lambda g: g.priority_score, reverse=True)

        # JPL Rule #2: Bounded iteration
        intent_count = min(len(sorted_gaps), MAX_DISCOVERY_INTENTS)

        for i in range(intent_count):
            gap = sorted_gaps[i]

            # Determine missing link type
            missing_link = self._determine_missing_link(gap)

            # Generate search terms
            search_terms = self._generate_search_terms(gap)

            # Create discovery intent artifact
            intent = IFDiscoveryIntentArtifact(
                artifact_id=str(uuid.uuid4()),
                content=gap.rationale,
                target_entity=gap.entity_text,
                entity_type=gap.entity_type,
                missing_link_type=missing_link,
                rationale=gap.rationale,
                confidence=gap.priority_score,
                priority_score=gap.priority_score,
                current_reference_count=gap.reference_count,
                suggested_search_terms=search_terms,
                metadata={
                    "session_id": self._session_id,
                    "is_dangling": gap.is_dangling,
                },
            )

            intents.append(intent)

        return intents

    def _determine_missing_link(self, gap: GapAnalysisResult) -> str:
        """
        Determine the type of missing link.

        Rule #4: Under 60 lines.

        Args:
            gap: Gap analysis result.

        Returns:
            Missing link type string.
        """
        if gap.is_dangling:
            return "cross_document_reference"
        elif gap.is_weak:
            return "additional_context"
        else:
            return "related_sources"

    def _generate_search_terms(self, gap: GapAnalysisResult) -> List[str]:
        """
        Generate suggested search terms for discovery.

        Rule #2: Bounded to MAX_SUGGESTED_TERMS.
        Rule #4: Under 60 lines.

        Args:
            gap: Gap analysis result.

        Returns:
            List of search term suggestions.
        """
        terms: List[str] = []

        # Primary term: exact entity
        terms.append(gap.entity_text)

        # Add entity type for context
        if gap.entity_type:
            terms.append(f"{gap.entity_text} {gap.entity_type}")

        # Add related discovery terms
        if gap.is_dangling:
            terms.append(f"{gap.entity_text} research")
            terms.append(f"{gap.entity_text} analysis")
        elif gap.is_weak:
            terms.append(f"{gap.entity_text} overview")

        # JPL Rule #2: Cap to maximum
        return terms[:MAX_SUGGESTED_TERMS]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def run_scout_analysis(
    manifest: IFKnowledgeManifest,
) -> List[IFDiscoveryIntentArtifact]:
    """
    Convenience function to run scout analysis.

    Rule #4: Under 60 lines.

    Args:
        manifest: Knowledge manifest to analyze.

    Returns:
        List of discovery intents.
    """
    scout = ProactiveScout()
    return scout.analyze_manifest(manifest)
