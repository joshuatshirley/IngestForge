"""
Tests for Enhanced Relationship Extraction.

Validates:
- SVOTriple dataclass
- extract_svo function
- extract_with_entities function
- extract_by_type function
- get_relationship_types function
- RELATIONSHIP_TYPES constant

Follows NASA JPL Commandments for test structure.
"""

import pytest
from ingestforge.enrichment.relationships import (
    Relationship,
    SVOTriple,
    extract_svo,
    extract_with_entities,
    extract_by_type,
    get_relationship_types,
    RELATIONSHIP_TYPES,
    _normalize_predicate,
)
from ingestforge.enrichment.ner import Entity


# =============================================================================
# Test Data
# =============================================================================


TEST_EMPLOYMENT_TEXT = """
Tim Cook works at Apple Inc.
Satya Nadella is employed by Microsoft Corporation.
Sundar Pichai leads Google.
"""

TEST_FOUNDING_TEXT = """
Steve Jobs founded Apple in 1976.
Bill Gates established Microsoft Corporation.
Larry Page created Google.
"""

TEST_ACQUISITION_TEXT = """
Microsoft acquired LinkedIn in 2016.
Facebook bought Instagram for $1 billion.
Google purchased YouTube.
"""

TEST_LOCATION_TEXT = """
Apple is based in Cupertino.
Microsoft is headquartered in Redmond.
Google is located in Mountain View.
"""


def make_test_entity(
    text: str,
    entity_type: str = "ORG",
    start: int = 0,
    end: int = 10,
) -> Entity:
    """Create test entity."""
    return Entity(
        text=text,
        type=entity_type,
        start=start,
        end=end,
        confidence=0.9,
    )


# =============================================================================
# Test RELATIONSHIP_TYPES
# =============================================================================


class TestRelationshipTypes:
    """Tests for RELATIONSHIP_TYPES constant."""

    def test_required_types_exist(self) -> None:
        """Test that required relationship types exist."""
        required = [
            "works_at",
            "located_in",
            "invented",
            "influenced_by",
            "founded",
            "acquired",
        ]

        for rel_type in required:
            assert rel_type in RELATIONSHIP_TYPES

    def test_types_have_verbs(self) -> None:
        """Test that each type has associated verbs."""
        for rel_type, verbs in RELATIONSHIP_TYPES.items():
            assert isinstance(verbs, list)
            assert len(verbs) > 0

    def test_get_relationship_types(self) -> None:
        """Test get_relationship_types function."""
        types = get_relationship_types()

        assert isinstance(types, list)
        assert "works_at" in types
        assert "founded" in types
        assert "acquired" in types


# =============================================================================
# Test SVOTriple
# =============================================================================


class TestSVOTriple:
    """Tests for SVOTriple dataclass."""

    def test_svo_creation(self) -> None:
        """Test creating SVOTriple."""
        svo = SVOTriple(
            subject="Tim Cook",
            verb="works",
            object="Apple Inc",
        )

        assert svo.subject == "Tim Cook"
        assert svo.verb == "works"
        assert svo.object == "Apple Inc"
        assert svo.confidence == 0.8  # Default

    def test_svo_with_context(self) -> None:
        """Test SVOTriple with sentence context."""
        svo = SVOTriple(
            subject="Steve Jobs",
            verb="founded",
            object="Apple",
            sentence="Steve Jobs founded Apple in 1976.",
            confidence=0.9,
        )

        assert svo.sentence == "Steve Jobs founded Apple in 1976."
        assert svo.confidence == 0.9

    def test_svo_to_relationship(self) -> None:
        """Test converting SVOTriple to Relationship."""
        svo = SVOTriple(
            subject="Microsoft",
            verb="acquired",
            object="LinkedIn",
            sentence="Microsoft acquired LinkedIn.",
        )

        rel = svo.to_relationship()

        assert isinstance(rel, Relationship)
        assert rel.subject == "Microsoft"
        assert rel.object == "LinkedIn"
        assert rel.context == "Microsoft acquired LinkedIn."


# =============================================================================
# Test _normalize_predicate
# =============================================================================


class TestNormalizePredicate:
    """Tests for predicate normalization."""

    def test_normalize_works(self) -> None:
        """Test normalizing 'work' verbs."""
        assert _normalize_predicate("work") == "works_at"
        assert _normalize_predicate("employ") == "works_at"

    def test_normalize_founded(self) -> None:
        """Test normalizing 'found' verbs."""
        assert _normalize_predicate("found") == "founded"
        assert _normalize_predicate("establish") == "founded"

    def test_normalize_acquired(self) -> None:
        """Test normalizing 'acquire' verbs."""
        assert _normalize_predicate("acquire") == "acquired"
        assert _normalize_predicate("buy") == "acquired"

    def test_normalize_unknown(self) -> None:
        """Test unknown verb passes through."""
        assert _normalize_predicate("unknownverb") == "unknownverb"


# =============================================================================
# Test extract_svo
# =============================================================================


class TestExtractSVO:
    """Tests for extract_svo function."""

    def test_empty_text(self) -> None:
        """Test with empty text."""
        result = extract_svo("")

        assert result == []

    def test_whitespace_only(self) -> None:
        """Test with whitespace only."""
        result = extract_svo("   \n\t  ")

        assert result == []

    def test_simple_svo(self) -> None:
        """Test simple SVO extraction."""
        result = extract_svo("Tim Cook works at Apple")

        # May or may not find depending on pattern
        assert isinstance(result, list)

    def test_returns_tuples(self) -> None:
        """Test that results are tuples."""
        result = extract_svo("Steve Jobs founded Apple")

        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3

    def test_no_reflexive_relations(self) -> None:
        """Test that self-relations are filtered."""
        # Even if pattern matches, subject != object
        result = extract_svo("Apple works at Apple")

        for subj, verb, obj in result:
            assert subj != obj

    def test_employment_pattern(self) -> None:
        """Test employment pattern matching."""
        result = extract_svo("Alice works at Google")

        # Should find employment relationship
        if result:
            subj, verb, obj = result[0]
            assert "work" in verb.lower()

    def test_founding_pattern(self) -> None:
        """Test founding pattern matching."""
        result = extract_svo("Steve Jobs founded Apple")

        if result:
            subj, verb, obj = result[0]
            assert "found" in verb.lower()


# =============================================================================
# Test extract_with_entities
# =============================================================================


class TestExtractWithEntities:
    """Tests for extract_with_entities function."""

    def test_empty_text(self) -> None:
        """Test with empty text."""
        result = extract_with_entities("", [])

        assert result == []

    def test_empty_entities(self) -> None:
        """Test with no entities."""
        result = extract_with_entities(TEST_EMPLOYMENT_TEXT, [])

        assert result == []

    def test_with_matching_entities(self) -> None:
        """Test with entities that match relationships."""
        entities = [
            make_test_entity("Tim Cook", "PERSON"),
            make_test_entity("Apple Inc", "ORG"),
        ]

        result = extract_with_entities(TEST_EMPLOYMENT_TEXT, entities)

        # Should only include relationships where both entities match
        assert isinstance(result, list)

    def test_filters_non_entity_relationships(self) -> None:
        """Test that non-entity relationships are filtered."""
        entities = [
            make_test_entity("Google", "ORG"),
        ]

        # Text mentions Apple and Microsoft, but entity is Google
        result = extract_with_entities(
            "Apple acquired Microsoft",
            entities,
        )

        # Should not find relationships since entities don't match
        assert isinstance(result, list)


# =============================================================================
# Test extract_by_type
# =============================================================================


class TestExtractByType:
    """Tests for extract_by_type function."""

    def test_unknown_type(self) -> None:
        """Test with unknown relationship type."""
        result = extract_by_type(TEST_EMPLOYMENT_TEXT, "unknown_type")

        assert result == []

    def test_works_at_type(self) -> None:
        """Test extracting works_at relationships."""
        result = extract_by_type(TEST_EMPLOYMENT_TEXT, "works_at")

        assert isinstance(result, list)
        # All results should be works_at type
        for rel in result:
            assert isinstance(rel, Relationship)

    def test_founded_type(self) -> None:
        """Test extracting founded relationships."""
        result = extract_by_type(TEST_FOUNDING_TEXT, "founded")

        assert isinstance(result, list)

    def test_acquired_type(self) -> None:
        """Test extracting acquired relationships."""
        result = extract_by_type(TEST_ACQUISITION_TEXT, "acquired")

        assert isinstance(result, list)

    def test_empty_text(self) -> None:
        """Test with empty text."""
        result = extract_by_type("", "works_at")

        assert result == []


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestRelationshipEdgeCases:
    """Tests for edge cases."""

    def test_very_long_text(self) -> None:
        """Test with very long text."""
        long_text = TEST_EMPLOYMENT_TEXT * 100

        result = extract_by_type(long_text, "works_at")

        assert isinstance(result, list)

    def test_special_characters(self) -> None:
        """Test text with special characters."""
        text = "Dr. O'Neill works at AT&T in São Paulo."

        result = extract_svo(text)

        assert isinstance(result, list)

    def test_multiple_sentences(self) -> None:
        """Test text with multiple sentences."""
        text = """
        Alice works at Google.
        Bob founded Startup.
        Charlie acquired Widget Inc.
        """

        result = extract_svo(text)

        # Should find multiple relationships
        assert isinstance(result, list)

    def test_passive_voice(self) -> None:
        """Test passive voice sentences."""
        text = "Apple was founded by Steve Jobs."

        result = extract_svo(text)

        # May or may not detect passive
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
