"""Atomic Entity Resolution for Deduplication.

Atomic Entity Resolution.
Follows NASA JPL Power of Ten rules.

Resolves multiple references to the same real-world entity into
a single canonical entity node during ingestion.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CLUSTER_SIZE = 100
MAX_ENTITIES_PER_BATCH = 5000
MAX_ALIASES_PER_CLUSTER = 50
MAX_NAME_LENGTH = 256
MAX_ATTRIBUTE_VALUES = 20
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0
DEFAULT_SIMILARITY_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Entity Normalizer
# ---------------------------------------------------------------------------


class EntityNormalizer:
    """Canonicalizes entity names for matching.

    GWT-1: Name normalization.
    Rule #4: Methods < 60 lines.
    """

    # Common honorifics to remove
    HONORIFICS = frozenset(
        {
            "mr",
            "mrs",
            "ms",
            "miss",
            "dr",
            "prof",
            "professor",
            "sir",
            "madam",
            "dame",
            "lord",
            "lady",
            "rev",
            "reverend",
            "hon",
            "honorable",
            "jr",
            "sr",
            "ii",
            "iii",
            "iv",
            "phd",
            "md",
            "esq",
            "esquire",
            "capt",
            "captain",
            "col",
            "colonel",
            "gen",
            "general",
            "lt",
            "lieutenant",
            "sgt",
            "sergeant",
        }
    )

    # Punctuation to remove (keep apostrophes for names like O'Brien)
    PUNCT_PATTERN = re.compile(r"[^\w\s'-]")

    def __init__(self) -> None:
        """Initialize normalizer."""
        pass

    def normalize(self, name: str) -> str:
        """Normalize a name to canonical form.

        GWT-1: Name normalization.

        Args:
            name: Raw entity name.

        Returns:
            Normalized canonical form.
        """
        if not name:
            return ""

        # Truncate to max length
        name = name[:MAX_NAME_LENGTH]

        # Apply normalization steps
        result = self.normalize_unicode(name)
        result = self.normalize_case(result)
        result = self.normalize_punctuation(result)
        result = self.remove_honorifics(result)
        result = self.normalize_whitespace(result)

        return result

    def normalize_unicode(self, name: str) -> str:
        """Normalize unicode characters.

        Args:
            name: Input name.

        Returns:
            Unicode-normalized name.
        """
        # Normalize to NFKD form and remove diacritics
        normalized = unicodedata.normalize("NFKD", name)
        # Keep only ASCII characters and common punctuation
        return "".join(
            c
            for c in normalized
            if unicodedata.category(c) != "Mn"  # Remove combining marks
        )

    def normalize_case(self, name: str) -> str:
        """Convert to lowercase.

        Args:
            name: Input name.

        Returns:
            Lowercased name.
        """
        return name.lower()

    def normalize_punctuation(self, name: str) -> str:
        """Remove excess punctuation.

        Args:
            name: Input name.

        Returns:
            Name with normalized punctuation.
        """
        # Remove most punctuation except apostrophes and hyphens
        result = self.PUNCT_PATTERN.sub(" ", name)
        # Normalize multiple spaces
        return " ".join(result.split())

    def remove_honorifics(self, name: str) -> str:
        """Remove common honorifics.

        Args:
            name: Input name.

        Returns:
            Name without honorifics.
        """
        words = name.split()
        filtered = [w for w in words if w.rstrip(".").lower() not in self.HONORIFICS]
        return " ".join(filtered)

    def normalize_whitespace(self, name: str) -> str:
        """Collapse whitespace.

        Args:
            name: Input name.

        Returns:
            Name with single spaces.
        """
        return " ".join(name.split()).strip()

    def get_tokens(self, name: str) -> List[str]:
        """Get normalized tokens from a name.

        Args:
            name: Input name.

        Returns:
            List of normalized tokens.
        """
        normalized = self.normalize(name)
        return normalized.split()


# ---------------------------------------------------------------------------
# Fuzzy Matcher
# ---------------------------------------------------------------------------


class FuzzyMatcher:
    """Computes similarity between entity names.

    GWT-2: Fuzzy matching.
    Rule #4: Methods < 60 lines.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        normalizer: Optional[EntityNormalizer] = None,
    ) -> None:
        """Initialize matcher.

        Rule #5: Assert preconditions.

        Args:
            threshold: Similarity threshold for matching (0.0-1.0).
            normalizer: Entity normalizer instance.
        """
        assert (
            MIN_SIMILARITY_THRESHOLD <= threshold <= MAX_SIMILARITY_THRESHOLD
        ), f"threshold must be between {MIN_SIMILARITY_THRESHOLD} and {MAX_SIMILARITY_THRESHOLD}"
        self._threshold = threshold
        self._normalizer = normalizer or EntityNormalizer()

    @property
    def threshold(self) -> float:
        """Get current similarity threshold."""
        return self._threshold

    def compute_similarity(self, name1: str, name2: str) -> float:
        """Compute similarity between two names.

        GWT-2: Fuzzy matching.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        if not name1 or not name2:
            return 0.0

        # Normalize both names
        norm1 = self._normalizer.normalize(name1)
        norm2 = self._normalizer.normalize(name2)

        # Exact match after normalization
        if norm1 == norm2:
            return 1.0

        # Empty after normalization
        if not norm1 or not norm2:
            return 0.0

        # Compute token-sorted Levenshtein ratio
        return self._token_sort_ratio(norm1, norm2)

    def _token_sort_ratio(self, s1: str, s2: str) -> float:
        """Compute token-sorted similarity ratio.

        Sorts tokens alphabetically before comparing to handle
        name order variations like "John Smith" vs "Smith, John".

        Args:
            s1: First normalized string.
            s2: Second normalized string.

        Returns:
            Similarity ratio 0.0-1.0.
        """
        # Sort tokens alphabetically
        sorted1 = " ".join(sorted(s1.split()))
        sorted2 = " ".join(sorted(s2.split()))

        # Compute Levenshtein ratio
        return self._levenshtein_ratio(sorted1, sorted2)

    def _levenshtein_ratio(self, s1: str, s2: str) -> float:
        """Compute Levenshtein similarity ratio.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Ratio = 1 - (edit_distance / max_length).
        """
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 == 0 or len2 == 0:
            return 0.0

        # Compute edit distance using dynamic programming
        # Rule #2: Bounded by MAX_NAME_LENGTH
        distance = self._levenshtein_distance(s1, s2)
        max_len = max(len1, len2)

        return 1.0 - (distance / max_len)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance.

        Uses space-optimized DP with two rows.
        Rule #1: No recursion.

        Args:
            s1: First string.
            s2: Second string.

        Returns:
            Edit distance.
        """
        len1, len2 = len(s1), len(s2)

        # Use shorter string for columns to minimize space
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        # Previous and current rows
        prev_row = list(range(len1 + 1))
        curr_row = [0] * (len1 + 1)

        for j in range(1, len2 + 1):
            curr_row[0] = j
            for i in range(1, len1 + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                curr_row[i] = min(
                    prev_row[i] + 1,  # deletion
                    curr_row[i - 1] + 1,  # insertion
                    prev_row[i - 1] + cost,  # substitution
                )
            prev_row, curr_row = curr_row, prev_row

        return prev_row[len1]

    def is_match(self, name1: str, name2: str) -> bool:
        """Check if two names match above threshold.

        Args:
            name1: First name.
            name2: Second name.

        Returns:
            True if similarity >= threshold.
        """
        return self.compute_similarity(name1, name2) >= self._threshold


# ---------------------------------------------------------------------------
# Entity Cluster
# ---------------------------------------------------------------------------


@dataclass
class EntityReference:
    """Reference to an entity occurrence.

    Rule #9: Complete type hints.
    """

    document_id: str
    artifact_id: str
    original_text: str
    entity_type: str
    start_char: int = 0
    end_char: int = 0
    confidence: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "artifact_id": self.artifact_id,
            "original_text": self.original_text,
            "entity_type": self.entity_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "attributes": self.attributes,
        }


@dataclass
class EntityCluster:
    """Resolved entity group.

    GWT-3: Entity merging.
    GWT-5: Cluster tracking.
    Rule #9: Complete type hints.
    """

    canonical_name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    references: List[EntityReference] = field(default_factory=list)
    confidence: float = 1.0
    merged_attributes: Dict[str, Any] = field(default_factory=dict)
    cluster_id: str = ""

    def __post_init__(self) -> None:
        """Validate cluster.

        Rule #5: Assert preconditions.
        """
        assert (
            len(self.canonical_name) <= MAX_NAME_LENGTH
        ), f"canonical_name exceeds {MAX_NAME_LENGTH} characters"

    @property
    def alias_count(self) -> int:
        """Number of aliases."""
        return len(self.aliases)

    @property
    def reference_count(self) -> int:
        """Number of references."""
        return len(self.references)

    @property
    def document_count(self) -> int:
        """Number of unique documents."""
        return len(set(ref.document_id for ref in self.references))

    @property
    def is_cross_document(self) -> bool:
        """True if entity spans multiple documents."""
        return self.document_count > 1

    def add_alias(self, alias: str) -> bool:
        """Add an alias to the cluster.

        Rule #2: Bounded list size.

        Args:
            alias: Alias to add.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.aliases) >= MAX_ALIASES_PER_CLUSTER:
            return False
        if alias not in self.aliases and alias != self.canonical_name:
            self.aliases.append(alias)
        return True

    def add_reference(self, ref: EntityReference) -> bool:
        """Add a reference to the cluster.

        Rule #2: Bounded list size.

        Args:
            ref: Reference to add.

        Returns:
            True if added, False if at capacity.
        """
        if len(self.references) >= MAX_CLUSTER_SIZE:
            return False
        self.references.append(ref)
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "cluster_id": self.cluster_id,
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "aliases": self.aliases[:MAX_ALIASES_PER_CLUSTER],
            "reference_count": self.reference_count,
            "document_count": self.document_count,
            "confidence": self.confidence,
            "merged_attributes": self.merged_attributes,
            "is_cross_document": self.is_cross_document,
        }


# ---------------------------------------------------------------------------
# Entity Resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Main entity resolution engine.

    GWT-3: Entity merging.
    GWT-4: Conflict resolution.
    GWT-5: Cluster tracking.
    Rule #4: Methods < 60 lines.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        normalizer: Optional[EntityNormalizer] = None,
        matcher: Optional[FuzzyMatcher] = None,
    ) -> None:
        """Initialize resolver.

        Rule #5: Assert preconditions.

        Args:
            threshold: Similarity threshold for matching.
            normalizer: Entity normalizer instance.
            matcher: Fuzzy matcher instance.
        """
        assert (
            MIN_SIMILARITY_THRESHOLD <= threshold <= MAX_SIMILARITY_THRESHOLD
        ), f"threshold must be between {MIN_SIMILARITY_THRESHOLD} and {MAX_SIMILARITY_THRESHOLD}"
        self._normalizer = normalizer or EntityNormalizer()
        self._matcher = matcher or FuzzyMatcher(threshold, self._normalizer)
        self._cluster_counter = 0

    def resolve(
        self,
        entities: List[EntityReference],
    ) -> List[EntityCluster]:
        """Resolve entities into clusters.

        GWT-3: Entity merging.
        GWT-5: Cluster tracking.
        Rule #1: No recursion - uses iterative clustering.

        Args:
            entities: List of entity references to resolve.

        Returns:
            List of resolved entity clusters.
        """
        assert entities is not None, "entities cannot be None"

        # Bound input size
        bounded_entities = entities[:MAX_ENTITIES_PER_BATCH]

        # Group by entity type first
        by_type: Dict[str, List[EntityReference]] = {}
        for entity in bounded_entities:
            entity_type = entity.entity_type
            if entity_type not in by_type:
                by_type[entity_type] = []
            by_type[entity_type].append(entity)

        # Resolve each type independently
        all_clusters: List[EntityCluster] = []
        for entity_type, type_entities in by_type.items():
            clusters = self._resolve_type(type_entities, entity_type)
            all_clusters.extend(clusters)

        return all_clusters

    def _resolve_type(
        self,
        entities: List[EntityReference],
        entity_type: str,
    ) -> List[EntityCluster]:
        """Resolve entities of a single type.

        Rule #1: No recursion - iterative clustering.

        Args:
            entities: Entities of the same type.
            entity_type: The entity type.

        Returns:
            Clusters for this type.
        """
        clusters: List[EntityCluster] = []
        assigned: Set[int] = set()

        # Iterative clustering (no recursion - Rule #1)
        for i, entity in enumerate(entities):
            if i in assigned:
                continue

            # Find or create cluster for this entity
            cluster = self._find_matching_cluster(entity, clusters)

            if cluster is None:
                # Create new cluster
                cluster = self._create_cluster(entity, entity_type)
                clusters.append(cluster)
            else:
                # Add to existing cluster
                self._add_to_cluster(cluster, entity)

            assigned.add(i)

            # Find other entities that match this cluster
            for j, other in enumerate(entities):
                if j in assigned:
                    continue
                if self._matcher.is_match(entity.original_text, other.original_text):
                    self._add_to_cluster(cluster, other)
                    assigned.add(j)

        return clusters

    def _find_matching_cluster(
        self,
        entity: EntityReference,
        clusters: List[EntityCluster],
    ) -> Optional[EntityCluster]:
        """Find a cluster that matches the entity.

        Args:
            entity: Entity to match.
            clusters: Existing clusters.

        Returns:
            Matching cluster or None.
        """
        for cluster in clusters:
            # Check against canonical name
            if self._matcher.is_match(entity.original_text, cluster.canonical_name):
                return cluster
            # Check against aliases
            for alias in cluster.aliases:
                if self._matcher.is_match(entity.original_text, alias):
                    return cluster
        return None

    def _create_cluster(
        self,
        entity: EntityReference,
        entity_type: str,
    ) -> EntityCluster:
        """Create a new cluster from an entity.

        Args:
            entity: Initial entity.
            entity_type: Entity type.

        Returns:
            New cluster.
        """
        self._cluster_counter += 1
        cluster = EntityCluster(
            canonical_name=entity.original_text,
            entity_type=entity_type,
            cluster_id=f"cluster_{self._cluster_counter}",
            confidence=entity.confidence,
            merged_attributes=dict(entity.attributes),
        )
        cluster.add_reference(entity)
        return cluster

    def _add_to_cluster(
        self,
        cluster: EntityCluster,
        entity: EntityReference,
    ) -> None:
        """Add an entity to an existing cluster.

        Args:
            cluster: Target cluster.
            entity: Entity to add.
        """
        cluster.add_reference(entity)
        cluster.add_alias(entity.original_text)

        # Update canonical name if this has higher confidence
        if entity.confidence > cluster.confidence:
            old_canonical = cluster.canonical_name
            cluster.canonical_name = entity.original_text
            cluster.add_alias(old_canonical)
            cluster.confidence = entity.confidence

        # Merge attributes
        self._merge_attributes(cluster, entity.attributes)

    def _merge_attributes(
        self,
        cluster: EntityCluster,
        new_attrs: Dict[str, Any],
    ) -> None:
        """Merge entity attributes into cluster.

        GWT-4: Conflict resolution.

        Args:
            cluster: Target cluster.
            new_attrs: New attributes to merge.
        """
        for key, value in new_attrs.items():
            if key not in cluster.merged_attributes:
                cluster.merged_attributes[key] = value
            else:
                # Track multiple values for conflict resolution
                existing = cluster.merged_attributes[key]
                if isinstance(existing, list):
                    if value not in existing and len(existing) < MAX_ATTRIBUTE_VALUES:
                        existing.append(value)
                elif existing != value:
                    cluster.merged_attributes[key] = [existing, value]

    def merge_clusters(
        self,
        cluster1: EntityCluster,
        cluster2: EntityCluster,
    ) -> EntityCluster:
        """Merge two clusters into one.

        GWT-3: Entity merging.

        Args:
            cluster1: First cluster.
            cluster2: Second cluster.

        Returns:
            Merged cluster.
        """
        # Use higher confidence cluster as base
        if cluster2.confidence > cluster1.confidence:
            cluster1, cluster2 = cluster2, cluster1

        # Create merged cluster
        merged = EntityCluster(
            canonical_name=cluster1.canonical_name,
            entity_type=cluster1.entity_type,
            cluster_id=cluster1.cluster_id,
            confidence=max(cluster1.confidence, cluster2.confidence),
        )

        # Add all aliases
        merged.add_alias(cluster2.canonical_name)
        for alias in cluster1.aliases:
            merged.add_alias(alias)
        for alias in cluster2.aliases:
            merged.add_alias(alias)

        # Add all references
        for ref in cluster1.references:
            merged.add_reference(ref)
        for ref in cluster2.references:
            merged.add_reference(ref)

        # Merge attributes
        merged.merged_attributes = dict(cluster1.merged_attributes)
        for key, value in cluster2.merged_attributes.items():
            self._merge_attributes(merged, {key: value})

        return merged

    def select_canonical(self, names: List[str]) -> str:
        """Select the best canonical name from a list.

        GWT-4: Conflict resolution.

        Args:
            names: List of name variations.

        Returns:
            Best canonical name.
        """
        if not names:
            return ""

        # Prefer the most common variation
        name_counts: Dict[str, int] = {}
        for name in names:
            normalized = self._normalizer.normalize(name)
            name_counts[name] = name_counts.get(name, 0) + 1

        # Return most frequent, preferring longer names for ties
        return max(names, key=lambda n: (name_counts[n], len(n)))

    def resolve_conflicts(
        self,
        attributes: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        """Resolve conflicting attribute values.

        GWT-4: Conflict resolution.

        Args:
            attributes: Dict of attribute name to list of values.

        Returns:
            Dict with resolved single values.
        """
        resolved: Dict[str, Any] = {}

        for key, values in attributes.items():
            if not values:
                continue
            if len(values) == 1:
                resolved[key] = values[0]
            else:
                # Select most frequent value
                value_counts: Dict[Any, int] = {}
                for v in values:
                    # Convert to hashable if needed
                    hashable_v = str(v) if isinstance(v, (dict, list)) else v
                    value_counts[hashable_v] = value_counts.get(hashable_v, 0) + 1
                # Get most frequent
                best_key = max(value_counts.keys(), key=lambda k: value_counts[k])
                # Find original value
                for v in values:
                    hashable_v = str(v) if isinstance(v, (dict, list)) else v
                    if hashable_v == best_key:
                        resolved[key] = v
                        break

        return resolved


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_entity_resolver(
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> EntityResolver:
    """Convenience function to create an EntityResolver.

    Args:
        threshold: Similarity threshold for matching.

    Returns:
        Configured EntityResolver.
    """
    return EntityResolver(threshold=threshold)


def resolve_entities(
    entities: List[EntityReference],
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List[EntityCluster]:
    """Convenience function to resolve a list of entities.

    Args:
        entities: List of entity references.
        threshold: Similarity threshold.

    Returns:
        List of resolved clusters.
    """
    resolver = create_entity_resolver(threshold)
    return resolver.resolve(entities)


def normalize_entity_name(name: str) -> str:
    """Convenience function to normalize an entity name.

    Args:
        name: Raw entity name.

    Returns:
        Normalized name.
    """
    normalizer = EntityNormalizer()
    return normalizer.normalize(name)


def compute_name_similarity(name1: str, name2: str) -> float:
    """Convenience function to compute similarity between names.

    Args:
        name1: First name.
        name2: Second name.

    Returns:
        Similarity score 0.0-1.0.
    """
    matcher = FuzzyMatcher()
    return matcher.compute_similarity(name1, name2)
