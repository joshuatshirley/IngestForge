"""Entity normalization and cross-document linking.

Handles entity variations (e.g., "Microsoft Corp" vs "Microsoft Corporation")
and builds an index for searching entities across documents.

Features:
- Entity normalization (titles, suffixes, abbreviations)
- Fuzzy matching for similar entity detection
- Cross-document entity linking
- Entity co-occurrence tracking
- Entity frequency profiles"""

from __future__ import annotations

import re
from typing import Dict, List, Set, Optional, Any, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.logging import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class EntityMention:
    """A single mention of an entity in a document."""

    text: str  # Original text
    normalized: str  # Normalized form
    entity_type: str  # PERSON, ORG, etc.
    document_id: str  # Document containing this mention
    chunk_id: str  # Chunk containing this mention
    confidence: float = 1.0  # Confidence score
    context: str = ""  # Surrounding text for disambiguation


@dataclass
class LinkedEntity:
    """Result of linking an entity mention to a canonical entity.

    Attributes:
        original: The original Entity object
        canonical_name: The canonical/normalized name
        profile: The linked EntityProfile if found
        similarity_score: How similar to the canonical form (0.0-1.0)
        is_new: Whether this is a newly discovered entity
    """

    original: Any  # Entity dataclass from ner.py
    canonical_name: str
    profile: Optional["EntityProfile"] = None
    similarity_score: float = 1.0
    is_new: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "original_text": self.original.text if self.original else "",
            "canonical_name": self.canonical_name,
            "similarity_score": self.similarity_score,
            "is_new": self.is_new,
            "profile": self.profile.to_dict() if self.profile else None,
        }


@dataclass
class EntityProfile:
    """Aggregated profile of an entity across documents."""

    canonical_name: str  # Canonical form
    entity_type: str  # Entity type
    variations: Set[str] = field(default_factory=set)  # All variations
    documents: Set[str] = field(default_factory=set)  # Document IDs
    chunks: Set[str] = field(default_factory=set)  # Chunk IDs
    mention_count: int = 0  # Total mentions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "canonical_name": self.canonical_name,
            "entity_type": self.entity_type,
            "variations": sorted(list(self.variations)),
            "documents": sorted(list(self.documents)),
            "chunks": sorted(list(self.chunks)),
            "mention_count": self.mention_count,
        }


class EntityLinker:
    """
    Link and normalize entities across documents.

    Handles variations:
    - "Microsoft Corp" → "Microsoft Corporation"
    - "Barack Obama" ←→ "Obama" (contextual)
    - "MIT" → "Massachusetts Institute of Technology"
    """

    def __init__(self) -> None:
        """Initialize entity linker."""
        # Map: normalized_text -> EntityProfile
        self.entity_index: Dict[str, EntityProfile] = {}

        # Common organization suffixes for normalization
        self.org_suffixes = {
            "Corp": "Corporation",
            "Inc": "Incorporated",
            "Ltd": "Limited",
            "Co": "Company",
            "LLC": "Limited Liability Company",
        }

        # Common abbreviations
        self.abbreviations = {
            "MIT": "Massachusetts Institute of Technology",
            "IBM": "International Business Machines",
            "NASA": "National Aeronautics and Space Administration",
            "FBI": "Federal Bureau of Investigation",
            "CIA": "Central Intelligence Agency",
            "UN": "United Nations",
            "EU": "European Union",
            "USA": "United States of America",
            "UK": "United Kingdom",
        }

    def normalize_entity(self, entity_text: str, entity_type: str) -> Tuple[str, str]:
        """
        Normalize entity text to canonical form.

        Args:
            entity_text: Original entity text
            entity_type: Entity type (PERSON, ORG, etc.)

        Returns:
            Tuple of (normalized_key, canonical_form)
            - normalized_key: Used for indexing/lookup (suffix-free)
            - canonical_form: Full, properly formatted name
        """
        # Basic normalization
        text = entity_text.strip()

        # Handle abbreviations
        if text.upper() in self.abbreviations:
            canonical = self.abbreviations[text.upper()]
            # Use abbreviation itself as key (lowercase)
            return text.lower(), canonical

        # Handle organizations
        if entity_type.lower() in ["org", "organization"]:
            canonical, lookup_key = self._normalize_organization(text)
            return lookup_key, canonical

        # Handle persons
        if entity_type.lower() == "person":
            canonical = self._normalize_person(text)
            return canonical.lower(), canonical

        # Default: use as-is
        return text.lower(), text

    def _normalize_organization(self, org_name: str) -> Tuple[str, str]:
        """
        Normalize organization name.

        Returns:
            Tuple of (canonical_name, lookup_key)
            - canonical_name: Full form with expanded suffixes
            - lookup_key: Base name without suffix (for matching)
        """
        # Expand abbreviated suffixes
        canonical = org_name

        for abbrev, full in self.org_suffixes.items():
            # Match with optional period
            pattern = rf"\b{re.escape(abbrev)}\.?\b"
            canonical = re.sub(pattern, full, canonical, flags=re.IGNORECASE)

        # Remove trailing punctuation
        canonical = canonical.rstrip(".,;:").strip()

        # Create lookup key by removing suffixes
        # This ensures "Microsoft Corporation" and "Microsoft Corp"
        # both map to "microsoft" as the key
        lookup_key = canonical.lower()

        for suffix in self.org_suffixes.values():
            # Check if ends with this suffix
            suffix_lower = suffix.lower()
            if lookup_key.endswith(suffix_lower):
                # Remove suffix and any trailing whitespace
                lookup_key = lookup_key[: -len(suffix_lower)].strip()
                break

        return canonical, lookup_key

    def _normalize_person(self, person_name: str) -> str:
        """Normalize person name."""
        # Remove titles
        titles = ["Dr", "Mr", "Mrs", "Ms", "Prof", "Professor"]
        canonical = person_name

        for title in titles:
            # Match with optional period
            pattern = rf"\b{re.escape(title)}\.?\s+"
            canonical = re.sub(pattern, "", canonical, flags=re.IGNORECASE)

        return canonical.strip()

    def add_entity(
        self,
        entity_text: str,
        entity_type: str,
        document_id: str,
        chunk_id: str,
        confidence: float = 1.0,
    ) -> str:
        """
        Add entity to index.

        Args:
            entity_text: Original entity text
            entity_type: Entity type
            document_id: Document ID
            chunk_id: Chunk ID
            confidence: Confidence score

        Returns:
            Canonical entity name
        """
        normalized, canonical = self.normalize_entity(entity_text, entity_type)

        # Create or update profile
        if normalized not in self.entity_index:
            self.entity_index[normalized] = EntityProfile(
                canonical_name=canonical,
                entity_type=entity_type,
            )

        profile = self.entity_index[normalized]
        profile.variations.add(entity_text)
        profile.documents.add(document_id)
        profile.chunks.add(chunk_id)
        profile.mention_count += 1

        return canonical

    def link_chunks(
        self, chunks: List[ChunkRecord], extractor: Any
    ) -> Dict[str, EntityProfile]:
        """
        Build entity index from chunks.

        Args:
            chunks: List of chunks to process
            extractor: EntityExtractor instance

        Returns:
            Dictionary mapping normalized names to EntityProfile objects
        """
        for chunk in chunks:
            # Extract entities from chunk
            entities_dict = extractor.extract(chunk.content)

            # Add each entity to index
            for entity_type, entities in entities_dict.items():
                for entity_text in entities:
                    self.add_entity(
                        entity_text=entity_text,
                        entity_type=entity_type,
                        document_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                    )

        logger.info(f"Linked {len(self.entity_index)} unique entities")
        return self.entity_index

    def search_by_entity(self, entity_name: str) -> Optional[EntityProfile]:
        """
        Find entity profile by name.

        Args:
            entity_name: Entity name to search

        Returns:
            EntityProfile if found, None otherwise
        """
        # Try exact match (case-insensitive)
        normalized = entity_name.lower().strip()

        if normalized in self.entity_index:
            return self.entity_index[normalized]

        # Try partial match
        for key, profile in self.entity_index.items():
            if normalized in key or key in normalized:
                return profile

            # Check variations
            for variation in profile.variations:
                if normalized in variation.lower():
                    return profile

        return None

    def get_documents_with_entity(self, entity_name: str) -> List[str]:
        """
        Get all documents containing entity.

        Args:
            entity_name: Entity name

        Returns:
            List of document IDs
        """
        profile = self.search_by_entity(entity_name)

        if profile:
            return sorted(list(profile.documents))

        return []

    def get_entity_cooccurrences(self, entity_name: str) -> Dict[str, int]:
        """
        Get entities that co-occur with this entity.

        Args:
            entity_name: Entity name

        Returns:
            Dictionary mapping entity names to co-occurrence counts
        """
        profile = self.search_by_entity(entity_name)

        if not profile:
            return {}

        # Find chunks containing this entity
        target_chunks = profile.chunks

        # Count co-occurrences
        cooccurrences: Dict[str, int] = defaultdict(int)

        for other_normalized, other_profile in self.entity_index.items():
            if other_normalized == entity_name.lower():
                continue  # Skip self

            # Count overlapping chunks
            overlap = target_chunks & other_profile.chunks

            if overlap:
                cooccurrences[other_profile.canonical_name] = len(overlap)

        return dict(cooccurrences)

    def export_index(self) -> List[Dict[str, Any]]:
        """
        Export entity index as list of dictionaries.

        Returns:
            List of entity profile dictionaries
        """
        return [profile.to_dict() for profile in self.entity_index.values()]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get entity index statistics.

        Returns:
            Dictionary with statistics
        """
        entity_types = defaultdict(int)
        total_mentions = 0

        for profile in self.entity_index.values():
            entity_types[profile.entity_type] += 1
            total_mentions += profile.mention_count

        return {
            "total_entities": len(self.entity_index),
            "total_mentions": total_mentions,
            "entity_types": dict(entity_types),
            "avg_mentions_per_entity": (
                total_mentions / len(self.entity_index) if self.entity_index else 0
            ),
        }


def link_entities(
    chunks: List[ChunkRecord], extractor: Any
) -> Tuple[EntityLinker, Dict[str, EntityProfile]]:
    """
    Build entity index from chunks.

    Args:
        chunks: List of chunks
        extractor: EntityExtractor instance

    Returns:
        Tuple of (EntityLinker, entity_index)
    """
    linker = EntityLinker()
    entity_index = linker.link_chunks(chunks, extractor)
    return linker, entity_index


# =============================================================================
# EntityIndex - Fast Entity Lookup Structure
# =============================================================================


class EntityIndex:
    """Index for fast entity lookup and similarity search.

    Provides O(1) lookup by normalized key and O(n) fuzzy matching.

            - Rule #4: Functions <60 lines
        - Rule #9: Full type hints
    """

    def __init__(self) -> None:
        """Initialize empty entity index."""
        self._by_normalized: Dict[str, EntityProfile] = {}
        self._by_type: Dict[str, List[EntityProfile]] = defaultdict(list)
        self._all_entities: List[EntityProfile] = []

    def add(self, normalized_key: str, profile: EntityProfile) -> None:
        """Add entity profile to index.

        Args:
            normalized_key: Normalized lookup key
            profile: Entity profile to add
        """
        self._by_normalized[normalized_key] = profile
        self._by_type[profile.entity_type].append(profile)
        self._all_entities.append(profile)

    def get(self, normalized_key: str) -> Optional[EntityProfile]:
        """Get entity by normalized key.

        Args:
            normalized_key: Key to lookup

        Returns:
            EntityProfile if found, None otherwise
        """
        return self._by_normalized.get(normalized_key)

    def get_by_type(self, entity_type: str) -> List[EntityProfile]:
        """Get all entities of a specific type.

        Args:
            entity_type: Type to filter by (PERSON, ORG, etc.)

        Returns:
            List of matching EntityProfile objects
        """
        return self._by_type.get(entity_type, [])

    def all(self) -> List[EntityProfile]:
        """Get all entity profiles.

        Returns:
            List of all EntityProfile objects
        """
        return self._all_entities.copy()

    def __len__(self) -> int:
        """Return number of entities in index."""
        return len(self._all_entities)

    def __contains__(self, key: str) -> bool:
        """Check if key exists in index."""
        return key in self._by_normalized


# =============================================================================
# Fuzzy Matching Utilities
# =============================================================================


def _levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Minimum number of edits to transform s1 to s2

    Rule #4: Function <60 lines
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def _calculate_similarity(s1: str, s2: str) -> float:
    """Calculate normalized similarity between two strings.

    Uses Levenshtein distance normalized by max length.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity score from 0.0 to 1.0

    Rule #4: Function <60 lines
    """
    if not s1 or not s2:
        return 0.0

    s1_lower = s1.lower().strip()
    s2_lower = s2.lower().strip()

    if s1_lower == s2_lower:
        return 1.0

    max_len = max(len(s1_lower), len(s2_lower))
    if max_len == 0:
        return 1.0

    distance = _levenshtein_distance(s1_lower, s2_lower)
    return 1.0 - (distance / max_len)


# =============================================================================
# Enhanced EntityLinker Methods
# =============================================================================


def find_similar_entities(
    linker: EntityLinker,
    entity_text: str,
    entity_type: str,
    threshold: float = 0.8,
) -> List[Tuple[EntityProfile, float]]:
    """Find entities similar to the given text.

    Uses fuzzy matching to find similar entities in the index.

    Args:
        linker: EntityLinker with populated index
        entity_text: Entity text to match
        entity_type: Type to filter by (or "ALL" for any type)
        threshold: Minimum similarity score (0.0 to 1.0)

    Returns:
        List of (EntityProfile, similarity_score) tuples, sorted by score

    Rule #1: Early return for empty input
    Rule #4: Function <60 lines
    """
    if not entity_text:
        return []

    results: List[Tuple[EntityProfile, float]] = []
    normalized_text = entity_text.lower().strip()

    for key, profile in linker.entity_index.items():
        # Filter by type if specified
        if entity_type != "ALL" and profile.entity_type != entity_type:
            continue

        # Calculate similarity with canonical name
        similarity = _calculate_similarity(normalized_text, profile.canonical_name)

        # Check variations too
        for variation in profile.variations:
            var_similarity = _calculate_similarity(normalized_text, variation)
            similarity = max(similarity, var_similarity)

        if similarity >= threshold:
            results.append((profile, similarity))

    # Sort by similarity descending
    return sorted(results, key=lambda x: x[1], reverse=True)


def link_entity(
    linker: EntityLinker,
    entity: Any,  # Entity from ner.py
    context: str = "",
    threshold: float = 0.8,
) -> LinkedEntity:
    """Link an entity mention to a canonical entity.

    Args:
        linker: EntityLinker with populated index
        entity: Entity object from NER extraction
        context: Surrounding text for disambiguation
        threshold: Minimum similarity for matching

    Returns:
        LinkedEntity with linking results

    Rule #1: Early returns for edge cases
    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    if not entity or not entity.text:
        return LinkedEntity(
            original=entity,
            canonical_name="",
            profile=None,
            similarity_score=0.0,
            is_new=True,
        )

    # Try exact match first
    profile = linker.search_by_entity(entity.text)
    if profile:
        return LinkedEntity(
            original=entity,
            canonical_name=profile.canonical_name,
            profile=profile,
            similarity_score=1.0,
            is_new=False,
        )

    # Try fuzzy matching
    similar = find_similar_entities(linker, entity.text, entity.type, threshold)
    if similar:
        best_match, score = similar[0]
        return LinkedEntity(
            original=entity,
            canonical_name=best_match.canonical_name,
            profile=best_match,
            similarity_score=score,
            is_new=False,
        )

    # No match found - normalize and return as new
    normalized, canonical = linker.normalize_entity(entity.text, entity.type)
    return LinkedEntity(
        original=entity,
        canonical_name=canonical,
        profile=None,
        similarity_score=1.0,
        is_new=True,
    )


def build_entity_index(entities: List[Any]) -> EntityIndex:
    """Build an EntityIndex from a list of Entity objects.

    Args:
        entities: List of Entity objects from NER extraction

    Returns:
        Populated EntityIndex

    Rule #4: Function <60 lines
    """
    linker = EntityLinker()
    index = EntityIndex()

    for entity in entities:
        # Add to linker first
        canonical = linker.add_entity(
            entity_text=entity.text,
            entity_type=entity.type,
            document_id=getattr(entity, "document_id", "unknown"),
            chunk_id=getattr(entity, "chunk_id", "unknown"),
            confidence=entity.confidence,
        )

    # Transfer to index
    for key, profile in linker.entity_index.items():
        index.add(key, profile)

    return index
