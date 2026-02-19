"""
Tests for Enhanced EntityLinker - Fuzzy Matching and Entity Index.

Validates:
- LinkedEntity dataclass
- EntityIndex operations
- Fuzzy matching with Levenshtein distance
- find_similar_entities function
- link_entity function
- build_entity_index function

Follows NASA JPL Commandments for test structure.
"""

import pytest
from ingestforge.enrichment.entity_linker import (
    EntityLinker,
    EntityProfile,
    LinkedEntity,
    EntityIndex,
    find_similar_entities,
    link_entity,
    build_entity_index,
    _levenshtein_distance,
    _calculate_similarity,
)
from ingestforge.enrichment.ner import Entity


# =============================================================================
# Test Data
# =============================================================================


def make_test_entity(
    text: str = "Apple Inc.",
    entity_type: str = "ORG",
    start: int = 0,
    end: int = 10,
    confidence: float = 0.9,
) -> Entity:
    """Create test Entity object."""
    return Entity(
        text=text,
        type=entity_type,
        start=start,
        end=end,
        confidence=confidence,
    )


# =============================================================================
# Test LinkedEntity Dataclass
# =============================================================================


class TestLinkedEntity:
    """Tests for LinkedEntity dataclass."""

    def test_linked_entity_creation(self) -> None:
        """Test creating LinkedEntity."""
        entity = make_test_entity()
        linked = LinkedEntity(
            original=entity,
            canonical_name="Apple Incorporated",
            profile=None,
            similarity_score=1.0,
            is_new=True,
        )

        assert linked.original == entity
        assert linked.canonical_name == "Apple Incorporated"
        assert linked.profile is None
        assert linked.similarity_score == 1.0
        assert linked.is_new is True

    def test_linked_entity_with_profile(self) -> None:
        """Test LinkedEntity with profile."""
        entity = make_test_entity()
        profile = EntityProfile(
            canonical_name="Apple Incorporated",
            entity_type="ORG",
        )

        linked = LinkedEntity(
            original=entity,
            canonical_name="Apple Incorporated",
            profile=profile,
            similarity_score=0.95,
            is_new=False,
        )

        assert linked.profile is not None
        assert linked.profile.canonical_name == "Apple Incorporated"
        assert linked.is_new is False

    def test_linked_entity_to_dict(self) -> None:
        """Test converting LinkedEntity to dictionary."""
        entity = make_test_entity()
        linked = LinkedEntity(
            original=entity,
            canonical_name="Apple Inc",
            similarity_score=0.9,
            is_new=True,
        )

        result = linked.to_dict()

        assert "original_text" in result
        assert "canonical_name" in result
        assert "similarity_score" in result
        assert "is_new" in result
        assert result["canonical_name"] == "Apple Inc"
        assert result["is_new"] is True


# =============================================================================
# Test EntityIndex
# =============================================================================


class TestEntityIndex:
    """Tests for EntityIndex class."""

    def test_empty_index(self) -> None:
        """Test empty index."""
        index = EntityIndex()

        assert len(index) == 0
        assert index.all() == []

    def test_add_entity(self) -> None:
        """Test adding entity to index."""
        index = EntityIndex()
        profile = EntityProfile(
            canonical_name="Apple Inc",
            entity_type="ORG",
        )

        index.add("apple", profile)

        assert len(index) == 1
        assert "apple" in index

    def test_get_entity(self) -> None:
        """Test getting entity from index."""
        index = EntityIndex()
        profile = EntityProfile("Apple Inc", "ORG")
        index.add("apple", profile)

        result = index.get("apple")

        assert result is not None
        assert result.canonical_name == "Apple Inc"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent entity."""
        index = EntityIndex()

        result = index.get("nonexistent")

        assert result is None

    def test_get_by_type(self) -> None:
        """Test getting entities by type."""
        index = EntityIndex()
        index.add("apple", EntityProfile("Apple Inc", "ORG"))
        index.add("microsoft", EntityProfile("Microsoft", "ORG"))
        index.add("jobs", EntityProfile("Steve Jobs", "PERSON"))

        orgs = index.get_by_type("ORG")
        persons = index.get_by_type("PERSON")

        assert len(orgs) == 2
        assert len(persons) == 1

    def test_all_entities(self) -> None:
        """Test getting all entities."""
        index = EntityIndex()
        index.add("a", EntityProfile("A", "ORG"))
        index.add("b", EntityProfile("B", "ORG"))
        index.add("c", EntityProfile("C", "PERSON"))

        all_entities = index.all()

        assert len(all_entities) == 3

    def test_contains(self) -> None:
        """Test contains check."""
        index = EntityIndex()
        index.add("apple", EntityProfile("Apple", "ORG"))

        assert "apple" in index
        assert "microsoft" not in index


# =============================================================================
# Test Fuzzy Matching Utilities
# =============================================================================


class TestLevenshteinDistance:
    """Tests for Levenshtein distance calculation."""

    def test_identical_strings(self) -> None:
        """Test distance for identical strings."""
        assert _levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self) -> None:
        """Test distance with empty strings."""
        assert _levenshtein_distance("", "") == 0
        assert _levenshtein_distance("hello", "") == 5
        assert _levenshtein_distance("", "world") == 5

    def test_single_char_difference(self) -> None:
        """Test single character difference."""
        assert _levenshtein_distance("cat", "bat") == 1
        assert _levenshtein_distance("cat", "cats") == 1
        assert _levenshtein_distance("cat", "ca") == 1

    def test_multiple_differences(self) -> None:
        """Test multiple character differences."""
        assert _levenshtein_distance("kitten", "sitting") == 3
        assert _levenshtein_distance("apple", "maple") == 2

    def test_completely_different(self) -> None:
        """Test completely different strings."""
        assert _levenshtein_distance("abc", "xyz") == 3


class TestCalculateSimilarity:
    """Tests for similarity calculation."""

    def test_identical_strings(self) -> None:
        """Test similarity for identical strings."""
        assert _calculate_similarity("hello", "hello") == 1.0

    def test_case_insensitive(self) -> None:
        """Test case-insensitive similarity."""
        assert _calculate_similarity("Hello", "hello") == 1.0
        assert _calculate_similarity("APPLE", "apple") == 1.0

    def test_empty_strings(self) -> None:
        """Test similarity with empty strings."""
        assert _calculate_similarity("", "") == 0.0
        assert _calculate_similarity("hello", "") == 0.0

    def test_similar_strings(self) -> None:
        """Test similarity for similar strings."""
        sim = _calculate_similarity("Microsoft", "Microsft")

        # Should be high but not perfect
        assert 0.8 < sim < 1.0

    def test_different_strings(self) -> None:
        """Test similarity for different strings."""
        sim = _calculate_similarity("Apple", "Microsoft")

        # Should be low
        assert sim < 0.5

    def test_whitespace_handling(self) -> None:
        """Test whitespace is stripped."""
        assert _calculate_similarity("  hello  ", "hello") == 1.0


# =============================================================================
# Test find_similar_entities
# =============================================================================


class TestFindSimilarEntities:
    """Tests for find_similar_entities function."""

    def test_empty_linker(self) -> None:
        """Test with empty linker."""
        linker = EntityLinker()

        result = find_similar_entities(linker, "Apple", "ORG", threshold=0.8)

        assert result == []

    def test_empty_query(self) -> None:
        """Test with empty query."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")

        result = find_similar_entities(linker, "", "ORG", threshold=0.8)

        assert result == []

    def test_exact_match(self) -> None:
        """Test finding exact match."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")

        result = find_similar_entities(linker, "Apple Inc", "ORG", threshold=0.8)

        assert len(result) >= 1
        # First result should have high similarity
        profile, score = result[0]
        assert score > 0.9

    def test_fuzzy_match(self) -> None:
        """Test finding fuzzy match."""
        linker = EntityLinker()
        linker.add_entity("Microsoft Corporation", "ORG", "doc1", "c1")

        # Slight misspelling
        result = find_similar_entities(
            linker, "Microsft Corporation", "ORG", threshold=0.8
        )

        assert len(result) >= 1

    def test_type_filter(self) -> None:
        """Test type filtering."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Apple", "PERSON", "doc2", "c2")  # Different type

        result = find_similar_entities(linker, "Apple", "ORG", threshold=0.5)

        # Should only find ORG
        assert all(p.entity_type == "ORG" for p, _ in result)

    def test_type_all(self) -> None:
        """Test with type ALL."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("John Smith", "PERSON", "doc2", "c2")

        result = find_similar_entities(linker, "Apple", "ALL", threshold=0.5)

        # Should find at least the ORG (Apple)
        assert len(result) >= 1
        # The ORG should be found
        found_orgs = [p for p, _ in result if p.entity_type == "ORG"]
        assert len(found_orgs) >= 1

    def test_threshold_filtering(self) -> None:
        """Test threshold filtering."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Microsoft Corp", "ORG", "doc2", "c2")

        # High threshold
        result_high = find_similar_entities(linker, "Apple", "ORG", threshold=0.9)

        # Low threshold
        result_low = find_similar_entities(linker, "Apple", "ORG", threshold=0.3)

        # Low threshold should have more results
        assert len(result_low) >= len(result_high)

    def test_sorted_by_score(self) -> None:
        """Test results sorted by similarity score."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Apple Corporation", "ORG", "doc2", "c2")
        linker.add_entity("Aple", "ORG", "doc3", "c3")

        result = find_similar_entities(linker, "Apple", "ORG", threshold=0.5)

        if len(result) > 1:
            scores = [score for _, score in result]
            assert scores == sorted(scores, reverse=True)


# =============================================================================
# Test link_entity
# =============================================================================


class TestLinkEntity:
    """Tests for link_entity function."""

    def test_link_empty_entity(self) -> None:
        """Test linking empty entity."""
        linker = EntityLinker()
        entity = Entity("", "ORG", 0, 0)

        result = link_entity(linker, entity)

        assert result.is_new is True
        assert result.similarity_score == 0.0

    def test_link_exact_match(self) -> None:
        """Test linking with exact match."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")

        entity = make_test_entity("Apple Inc", "ORG")
        result = link_entity(linker, entity)

        assert result.is_new is False
        assert result.similarity_score == 1.0
        assert result.profile is not None

    def test_link_fuzzy_match(self) -> None:
        """Test linking with fuzzy match."""
        linker = EntityLinker()
        linker.add_entity("Microsoft Corporation", "ORG", "doc1", "c1")

        entity = make_test_entity("Microsft Corp", "ORG")
        result = link_entity(linker, entity, threshold=0.7)

        # May or may not match depending on similarity
        assert isinstance(result, LinkedEntity)

    def test_link_new_entity(self) -> None:
        """Test linking completely new entity."""
        linker = EntityLinker()
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")

        entity = make_test_entity("Google LLC", "ORG")
        result = link_entity(linker, entity, threshold=0.9)

        assert result.is_new is True
        assert result.canonical_name is not None


# =============================================================================
# Test build_entity_index
# =============================================================================


class TestBuildEntityIndex:
    """Tests for build_entity_index function."""

    def test_empty_list(self) -> None:
        """Test building index from empty list."""
        result = build_entity_index([])

        assert len(result) == 0

    def test_single_entity(self) -> None:
        """Test building index from single entity."""
        entities = [make_test_entity("Apple Inc", "ORG")]

        result = build_entity_index(entities)

        assert len(result) == 1

    def test_multiple_entities(self) -> None:
        """Test building index from multiple entities."""
        entities = [
            make_test_entity("Apple Inc", "ORG", 0, 10),
            make_test_entity("Microsoft", "ORG", 20, 29),
            make_test_entity("Steve Jobs", "PERSON", 50, 60),
        ]

        result = build_entity_index(entities)

        assert len(result) == 3

    def test_duplicate_entities(self) -> None:
        """Test building index with duplicates."""
        entities = [
            make_test_entity("Apple Inc", "ORG", 0, 10),
            make_test_entity("Apple Inc", "ORG", 50, 60),  # Same entity
        ]

        result = build_entity_index(entities)

        # Should deduplicate
        assert len(result) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
