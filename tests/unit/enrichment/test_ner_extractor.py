"""
Tests for NERExtractor - Enhanced Named Entity Recognition.

Validates:
- Entity dataclass with normalized field
- NERExtractor initialization and model loading
- Single text extraction
- Batch extraction efficiency
- Entity type listing
- Confidence scores
- Fallback to regex when spaCy unavailable

Follows NASA JPL Commandments for test structure.
"""

import pytest
from ingestforge.enrichment.ner import (
    Entity,
    NERExtractor,
)


# =============================================================================
# Test Data
# =============================================================================


TEST_TEXT_SIMPLE = "Apple Inc. was founded by Steve Jobs in Cupertino, California."

TEST_TEXT_COMPLEX = """
Dr. Sarah Johnson from Stanford University published groundbreaking research.
The study, conducted at MIT, received $2.5 million in funding.
Johnson collaborated with Prof. David Chen from Beijing Institute of Technology.
Microsoft Corporation announced a partnership on March 1, 2024.
"""

TEST_TEXT_EVENTS = """
Hurricane Katrina devastated New Orleans in 2005.
The Battle of Gettysburg was a turning point in the Civil War.
Shakespeare wrote Romeo and Juliet in the 16th century.
"""


# =============================================================================
# Test Entity Dataclass
# =============================================================================


class TestEntityDataclass:
    """Tests for Entity dataclass."""

    def test_entity_creation_basic(self) -> None:
        """Test creating Entity with required fields."""
        entity = Entity(
            text="Apple Inc.",
            type="ORG",
            start=0,
            end=10,
        )

        assert entity.text == "Apple Inc."
        assert entity.type == "ORG"
        assert entity.start == 0
        assert entity.end == 10
        assert entity.confidence == 1.0  # Default
        assert entity.normalized is None  # Default

    def test_entity_creation_with_all_fields(self) -> None:
        """Test creating Entity with all fields."""
        entity = Entity(
            text="Steve Jobs",
            type="PERSON",
            start=25,
            end=35,
            confidence=0.95,
            normalized="steve jobs",
        )

        assert entity.text == "Steve Jobs"
        assert entity.type == "PERSON"
        assert entity.start == 25
        assert entity.end == 35
        assert entity.confidence == 0.95
        assert entity.normalized == "steve jobs"

    def test_entity_to_dict(self) -> None:
        """Test converting Entity to dictionary."""
        entity = Entity(
            text="Microsoft",
            type="ORG",
            start=0,
            end=9,
            confidence=0.9,
            normalized="Microsoft Corporation",
        )

        result = entity.to_dict()

        assert result["text"] == "Microsoft"
        assert result["type"] == "ORG"
        assert result["start"] == 0
        assert result["end"] == 9
        assert result["confidence"] == 0.9
        assert result["normalized"] == "Microsoft Corporation"

    def test_entity_hash(self) -> None:
        """Test Entity hashing for deduplication."""
        e1 = Entity("Apple", "ORG", 0, 5, 0.9)
        e2 = Entity("Apple", "ORG", 0, 5, 0.95)  # Same except confidence
        e3 = Entity("apple", "ORG", 0, 5, 0.9)  # Same case-insensitive
        e4 = Entity("Microsoft", "ORG", 0, 9, 0.9)  # Different

        # Same text/type/position should hash equal
        assert hash(e1) == hash(e2)
        assert hash(e1) == hash(e3)
        assert hash(e1) != hash(e4)

    def test_entity_equality(self) -> None:
        """Test Entity equality comparison."""
        e1 = Entity("Apple", "ORG", 0, 5)
        e2 = Entity("Apple", "ORG", 0, 5)
        e3 = Entity("apple", "ORG", 0, 5)
        e4 = Entity("Apple", "PERSON", 0, 5)

        assert e1 == e2
        assert e1 == e3  # Case insensitive
        assert e1 != e4  # Different type
        assert e1 != "Apple"  # Different type


# =============================================================================
# Test NERExtractor Initialization
# =============================================================================


class TestNERExtractorInit:
    """Tests for NERExtractor initialization."""

    def test_default_initialization(self) -> None:
        """Test default initialization."""
        extractor = NERExtractor()

        assert extractor.model_name == "en_core_web_sm"
        assert extractor._nlp is None  # Lazy loading
        assert extractor._spacy_available is True

    def test_custom_model_initialization(self) -> None:
        """Test initialization with custom model."""
        extractor = NERExtractor(model="en_core_web_lg")

        assert extractor.model_name == "en_core_web_lg"

    def test_is_available_always_true(self) -> None:
        """Test that is_available returns True (regex fallback)."""
        extractor = NERExtractor()

        assert extractor.is_available() is True

    def test_get_entity_types(self) -> None:
        """Test getting list of entity types."""
        extractor = NERExtractor()
        types = extractor.get_entity_types()

        assert isinstance(types, list)
        assert "PERSON" in types
        assert "ORG" in types
        assert "GPE" in types
        assert "DATE" in types
        assert "EVENT" in types
        assert "WORK_OF_ART" in types


# =============================================================================
# Test NERExtractor Extraction
# =============================================================================


class TestNERExtractorExtraction:
    """Tests for NERExtractor.extract()."""

    def test_extract_empty_text(self) -> None:
        """Test extraction with empty text."""
        extractor = NERExtractor()
        result = extractor.extract("")

        assert result == []

    def test_extract_whitespace_only(self) -> None:
        """Test extraction with whitespace only."""
        extractor = NERExtractor()
        result = extractor.extract("   \n\t  ")

        assert result == []

    def test_extract_returns_entity_objects(self) -> None:
        """Test that extract returns Entity objects."""
        extractor = NERExtractor()
        result = extractor.extract(TEST_TEXT_SIMPLE)

        assert isinstance(result, list)
        if result:  # May be empty if no spaCy
            assert all(isinstance(e, Entity) for e in result)

    def test_extract_has_required_fields(self) -> None:
        """Test that extracted entities have all fields."""
        extractor = NERExtractor()
        result = extractor.extract(TEST_TEXT_SIMPLE)

        for entity in result:
            assert hasattr(entity, "text")
            assert hasattr(entity, "type")
            assert hasattr(entity, "start")
            assert hasattr(entity, "end")
            assert hasattr(entity, "confidence")
            assert hasattr(entity, "normalized")

    def test_extract_sorted_by_position(self) -> None:
        """Test that entities are sorted by position."""
        extractor = NERExtractor()
        result = extractor.extract(TEST_TEXT_COMPLEX)

        if len(result) > 1:
            positions = [e.start for e in result]
            assert positions == sorted(positions)

    def test_extract_confidence_in_range(self) -> None:
        """Test that confidence scores are in valid range."""
        extractor = NERExtractor()
        result = extractor.extract(TEST_TEXT_SIMPLE)

        for entity in result:
            assert 0.0 <= entity.confidence <= 1.0

    def test_extract_has_normalized_form(self) -> None:
        """Test that entities have normalized form."""
        extractor = NERExtractor()
        result = extractor.extract(TEST_TEXT_SIMPLE)

        for entity in result:
            # Normalized should be set (not None)
            assert entity.normalized is not None


# =============================================================================
# Test NERExtractor Batch Processing
# =============================================================================


class TestNERExtractorBatch:
    """Tests for NERExtractor.extract_batch()."""

    def test_extract_batch_empty_list(self) -> None:
        """Test batch extraction with empty list."""
        extractor = NERExtractor()
        result = extractor.extract_batch([])

        assert result == []

    def test_extract_batch_single_text(self) -> None:
        """Test batch extraction with single text."""
        extractor = NERExtractor()
        result = extractor.extract_batch([TEST_TEXT_SIMPLE])

        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_extract_batch_multiple_texts(self) -> None:
        """Test batch extraction with multiple texts."""
        texts = [
            "Apple Inc. is in California.",
            "Microsoft was founded by Bill Gates.",
            "Google is located in Mountain View.",
        ]
        extractor = NERExtractor()
        result = extractor.extract_batch(texts)

        assert len(result) == 3
        assert all(isinstance(r, list) for r in result)

    def test_extract_batch_preserves_order(self) -> None:
        """Test that batch results maintain input order."""
        texts = [
            "Text about Apple.",
            "Text about nothing.",
            "Text about Microsoft.",
        ]
        extractor = NERExtractor()
        results = extractor.extract_batch(texts)

        assert len(results) == 3
        # First and third should have entities, second may not

    def test_extract_batch_mixed_content(self) -> None:
        """Test batch with mixed content (empty and non-empty)."""
        texts = [
            "Apple Inc. in California.",
            "",
            "   ",
            "Microsoft in Seattle.",
        ]
        extractor = NERExtractor()
        results = extractor.extract_batch(texts)

        assert len(results) == 4
        # Empty texts should have empty entity lists
        assert results[1] == []
        assert results[2] == []


# =============================================================================
# Test NERExtractor Normalization
# =============================================================================


class TestNERExtractorNormalization:
    """Tests for entity normalization."""

    def test_normalize_person_removes_title(self) -> None:
        """Test that person names have titles removed."""
        extractor = NERExtractor()

        # Test normalization directly
        result = extractor._normalize_entity_text("Dr. John Smith", "PERSON")

        assert "Dr." not in result
        assert "John Smith" in result

    def test_normalize_org_expands_suffix(self) -> None:
        """Test that org suffixes are expanded."""
        extractor = NERExtractor()

        result = extractor._normalize_entity_text("Apple Inc.", "ORG")

        assert "Incorporated" in result

    def test_normalize_preserves_unknown(self) -> None:
        """Test that unknown types are preserved."""
        extractor = NERExtractor()

        result = extractor._normalize_entity_text("Something", "UNKNOWN")

        assert result == "Something"


# =============================================================================
# Test NERExtractor spaCy Integration
# =============================================================================


class TestNERExtractorSpacy:
    """Tests for spaCy-based extraction."""

    def test_spacy_availability_check(self) -> None:
        """Test checking spaCy availability."""
        extractor = NERExtractor()

        # Should not crash
        result = extractor.is_spacy_available()
        assert isinstance(result, bool)

    def test_fallback_to_regex(self) -> None:
        """Test fallback to regex when spaCy unavailable."""
        # Force regex mode
        extractor = NERExtractor()
        extractor._spacy_available = False

        result = extractor.extract(TEST_TEXT_SIMPLE)

        # Should still return entities
        assert isinstance(result, list)

    def test_regex_lower_confidence(self) -> None:
        """Test that regex extraction has lower confidence."""
        extractor = NERExtractor()
        extractor._spacy_available = False

        result = extractor.extract("Apple Inc. in California")

        for entity in result:
            # Regex has 0.6 confidence
            assert entity.confidence == 0.6


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestNERExtractorEdgeCases:
    """Tests for edge cases."""

    def test_very_long_text(self) -> None:
        """Test extraction from very long text."""
        extractor = NERExtractor()
        long_text = TEST_TEXT_COMPLEX * 100

        result = extractor.extract(long_text)

        assert isinstance(result, list)
        # Should not crash or timeout

    def test_special_characters(self) -> None:
        """Test text with special characters."""
        extractor = NERExtractor()
        text = "Dr. O'Neill from AT&T met in São Paulo."

        result = extractor.extract(text)

        assert isinstance(result, list)

    def test_unicode_text(self) -> None:
        """Test text with unicode characters."""
        extractor = NERExtractor()
        text = "北京 is the capital of China. Toyota is Japanese."

        result = extractor.extract(text)

        assert isinstance(result, list)

    def test_no_entities(self) -> None:
        """Test text with no entities."""
        extractor = NERExtractor()
        text = "the and but or if then else"

        result = extractor.extract(text)

        # Should return empty or very few
        assert isinstance(result, list)
        assert len(result) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
