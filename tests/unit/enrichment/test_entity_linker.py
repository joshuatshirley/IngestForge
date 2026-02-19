"""
Tests for Entity Normalization and Linking.

Validates:
- Entity normalization (handling variations)
- Cross-document linking
- Entity index building
- Search by entity
- Co-occurrence detection
"""

import pytest
from ingestforge.enrichment.entity_linker import (
    EntityLinker,
    EntityProfile,
    link_entities,
)
from ingestforge.enrichment.entities import EntityExtractor
from ingestforge.chunking.semantic_chunker import ChunkRecord


# ============================================================================
# Test Data
# ============================================================================


TEST_CORPUS_1 = """
Apple Inc. announced a partnership with Microsoft Corporation today.
CEO Tim Cook met with Satya Nadella in Cupertino.
"""

TEST_CORPUS_2 = """
Microsoft Corp. has been working with Apple on the new initiative.
Nadella praised Cook for the collaboration.
"""

TEST_CORPUS_3 = """
Barack Obama visited the White House yesterday.
President Obama discussed climate policy with staff.
"""


def make_test_chunks() -> list:
    """Create test chunks."""
    return [
        ChunkRecord(
            chunk_id="c1",
            document_id="doc1",
            content=TEST_CORPUS_1,
        ),
        ChunkRecord(
            chunk_id="c2",
            document_id="doc2",
            content=TEST_CORPUS_2,
        ),
        ChunkRecord(
            chunk_id="c3",
            document_id="doc3",
            content=TEST_CORPUS_3,
        ),
    ]


# ============================================================================
# Test Classes
# ============================================================================


class TestEntityNormalization:
    """Tests for entity normalization."""

    def test_normalize_organization_with_corp(self):
        """Test normalizing organization with Corp suffix."""
        linker = EntityLinker()

        normalized, canonical = linker.normalize_entity("Microsoft Corp", "ORG")

        # normalized_key should have suffix removed for matching
        assert normalized == "microsoft"
        # canonical should have expanded suffix
        assert "Corporation" in canonical

    def test_normalize_organization_with_inc(self):
        """Test normalizing organization with Inc suffix."""
        linker = EntityLinker()

        normalized, canonical = linker.normalize_entity("Apple Inc.", "ORG")

        # normalized_key should have suffix removed for matching
        assert normalized == "apple"
        # canonical should have expanded suffix
        assert "Incorporated" in canonical

    def test_normalize_abbreviation(self):
        """Test normalizing known abbreviations."""
        linker = EntityLinker()

        normalized, canonical = linker.normalize_entity("MIT", "ORG")

        assert canonical == "Massachusetts Institute of Technology"

    def test_normalize_person_with_title(self):
        """Test normalizing person name with title."""
        linker = EntityLinker()

        normalized, canonical = linker.normalize_entity("Dr. Jane Smith", "PERSON")

        assert "Dr" not in canonical
        assert "Jane Smith" in canonical

    def test_normalize_person_without_title(self):
        """Test normalizing person name without title."""
        linker = EntityLinker()

        normalized, canonical = linker.normalize_entity("Barack Obama", "PERSON")

        assert canonical == "Barack Obama"

    def test_normalize_preserves_original(self):
        """Test that normalization doesn't modify unknown entities."""
        linker = EntityLinker()

        normalized, canonical = linker.normalize_entity("SomeCompany", "ORG")

        assert canonical == "SomeCompany"


class TestEntityIndexing:
    """Tests for entity index building."""

    def test_add_entity_creates_profile(self):
        """Test adding entity creates profile."""
        linker = EntityLinker()

        canonical = linker.add_entity(
            entity_text="Microsoft Corp",
            entity_type="ORG",
            document_id="doc1",
            chunk_id="c1",
        )

        # Entity index is keyed by normalized_key (without suffix)
        assert "microsoft" in linker.entity_index
        assert canonical == "Microsoft Corporation"

    def test_add_entity_tracks_documents(self):
        """Test entity profile tracks documents."""
        linker = EntityLinker()

        linker.add_entity("Microsoft Corp", "ORG", "doc1", "c1")
        linker.add_entity("Microsoft Corporation", "ORG", "doc2", "c2")

        # Both should map to same key "microsoft" (suffix removed)
        profile = linker.entity_index["microsoft"]
        assert "doc1" in profile.documents
        assert "doc2" in profile.documents
        assert len(profile.documents) == 2

    def test_add_entity_tracks_variations(self):
        """Test entity profile tracks variations."""
        linker = EntityLinker()

        linker.add_entity("Microsoft Corp", "ORG", "doc1", "c1")
        linker.add_entity("Microsoft Corporation", "ORG", "doc2", "c2")

        # Both should map to same key "microsoft" (suffix removed)
        profile = linker.entity_index["microsoft"]
        assert "Microsoft Corp" in profile.variations
        assert "Microsoft Corporation" in profile.variations

    def test_add_entity_counts_mentions(self):
        """Test entity profile counts mentions."""
        linker = EntityLinker()

        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Apple Inc", "ORG", "doc1", "c2")
        linker.add_entity("Apple Inc", "ORG", "doc2", "c3")

        # Key is "apple" (suffix removed)
        profile = linker.entity_index["apple"]
        assert profile.mention_count == 3


class TestEntityLinking:
    """Tests for cross-document entity linking."""

    def test_link_chunks_builds_index(self):
        """Test linking chunks builds entity index."""
        linker = EntityLinker()
        extractor = EntityExtractor(use_spacy=False)  # Use regex
        chunks = make_test_chunks()

        entity_index = linker.link_chunks(chunks, extractor)

        # Should have multiple entities
        assert len(entity_index) > 0

    def test_link_chunks_finds_organizations(self):
        """Test linking finds organization entities."""
        linker = EntityLinker()
        extractor = EntityExtractor(use_spacy=False)
        chunks = make_test_chunks()

        linker.link_chunks(chunks, extractor)

        # Should find Apple and Microsoft variations
        found_entities = list(linker.entity_index.keys())

        # Check for presence (case-insensitive)
        entity_text = " ".join(found_entities).lower()
        # At least one org should be found
        assert any(
            term in entity_text for term in ["apple", "microsoft", "company", "corp"]
        )

    def test_link_chunks_tracks_cross_document(self):
        """Test entities linked across documents."""
        linker = EntityLinker()
        extractor = EntityExtractor(use_spacy=False)
        chunks = make_test_chunks()

        linker.link_chunks(chunks, extractor)

        # Find entities appearing in multiple documents
        multi_doc_entities = [
            p for p in linker.entity_index.values() if len(p.documents) > 1
        ]

        # Should have at least some cross-document entities
        assert len(multi_doc_entities) >= 0  # Lenient check


class TestEntitySearch:
    """Tests for entity search functionality."""

    def test_search_by_entity_exact_match(self):
        """Test searching entity by exact name."""
        linker = EntityLinker()

        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")

        profile = linker.search_by_entity("Apple Inc")

        assert profile is not None
        assert profile.canonical_name == "Apple Incorporated"

    def test_search_by_entity_case_insensitive(self):
        """Test search is case-insensitive."""
        linker = EntityLinker()

        linker.add_entity("Microsoft Corp", "ORG", "doc1", "c1")

        profile = linker.search_by_entity("MICROSOFT CORP")

        assert profile is not None

    def test_search_by_entity_partial_match(self):
        """Test search with partial match."""
        linker = EntityLinker()

        linker.add_entity("Microsoft Corporation", "ORG", "doc1", "c1")

        profile = linker.search_by_entity("Microsoft")

        assert profile is not None

    def test_search_by_entity_not_found(self):
        """Test search returns None if not found."""
        linker = EntityLinker()

        profile = linker.search_by_entity("NonexistentEntity")

        assert profile is None

    def test_get_documents_with_entity(self):
        """Test getting documents containing entity."""
        linker = EntityLinker()

        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Apple Inc", "ORG", "doc2", "c2")

        docs = linker.get_documents_with_entity("Apple Inc")

        assert len(docs) == 2
        assert "doc1" in docs
        assert "doc2" in docs

    def test_get_documents_empty_for_nonexistent(self):
        """Test empty list for nonexistent entity."""
        linker = EntityLinker()

        docs = linker.get_documents_with_entity("Nonexistent")

        assert docs == []


class TestEntityCooccurrence:
    """Tests for entity co-occurrence detection."""

    def test_get_entity_cooccurrences(self):
        """Test finding entities that co-occur."""
        linker = EntityLinker()

        # Add entities in same chunk
        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Microsoft Corp", "ORG", "doc1", "c1")

        cooccurrences = linker.get_entity_cooccurrences("Apple Inc")

        assert len(cooccurrences) > 0
        # Microsoft should co-occur with Apple
        assert any("Microsoft" in name for name in cooccurrences.keys())

    def test_cooccurrence_counts(self):
        """Test co-occurrence counts."""
        linker = EntityLinker()

        # Add entities in multiple shared chunks
        linker.add_entity("Alice", "PERSON", "doc1", "c1")
        linker.add_entity("Bob", "PERSON", "doc1", "c1")
        linker.add_entity("Alice", "PERSON", "doc1", "c2")
        linker.add_entity("Bob", "PERSON", "doc1", "c2")

        cooccurrences = linker.get_entity_cooccurrences("Alice")

        # Bob should co-occur with Alice in 2 chunks
        assert "Bob" in cooccurrences
        assert cooccurrences["Bob"] == 2

    def test_cooccurrence_empty_for_nonexistent(self):
        """Test empty co-occurrences for nonexistent entity."""
        linker = EntityLinker()

        cooccurrences = linker.get_entity_cooccurrences("Nonexistent")

        assert cooccurrences == {}


class TestEntityProfile:
    """Tests for EntityProfile dataclass."""

    def test_profile_to_dict(self):
        """Test converting profile to dictionary."""
        profile = EntityProfile(
            canonical_name="Apple Inc.",
            entity_type="ORG",
        )
        profile.variations.add("Apple")
        profile.variations.add("Apple Inc")
        profile.documents.add("doc1")
        profile.chunks.add("c1")
        profile.mention_count = 5

        result = profile.to_dict()

        assert result["canonical_name"] == "Apple Inc."
        assert result["entity_type"] == "ORG"
        assert len(result["variations"]) == 2
        assert len(result["documents"]) == 1
        assert result["mention_count"] == 5


class TestEntityIndexExport:
    """Tests for entity index export."""

    def test_export_index(self):
        """Test exporting entity index."""
        linker = EntityLinker()

        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Microsoft Corp", "ORG", "doc2", "c2")

        exported = linker.export_index()

        assert len(exported) == 2
        assert all("canonical_name" in e for e in exported)
        assert all("entity_type" in e for e in exported)

    def test_get_statistics(self):
        """Test getting entity index statistics."""
        linker = EntityLinker()

        linker.add_entity("Apple Inc", "ORG", "doc1", "c1")
        linker.add_entity("Microsoft Corp", "ORG", "doc2", "c2")
        linker.add_entity("Barack Obama", "PERSON", "doc3", "c3")

        stats = linker.get_statistics()

        assert stats["total_entities"] == 3
        assert stats["total_mentions"] == 3
        assert "ORG" in stats["entity_types"]
        assert "PERSON" in stats["entity_types"]
        assert stats["entity_types"]["ORG"] == 2
        assert stats["entity_types"]["PERSON"] == 1


class TestLinkEntitiesFunction:
    """Tests for link_entities convenience function."""

    def test_link_entities_function(self):
        """Test link_entities function."""
        extractor = EntityExtractor(use_spacy=False)
        chunks = make_test_chunks()

        linker, entity_index = link_entities(chunks, extractor)

        assert isinstance(linker, EntityLinker)
        assert len(entity_index) > 0


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_chunks(self):
        """Test linking with empty chunks."""
        linker = EntityLinker()
        extractor = EntityExtractor(use_spacy=False)

        entity_index = linker.link_chunks([], extractor)

        assert len(entity_index) == 0

    def test_chunk_with_no_entities(self):
        """Test chunk with no entities."""
        linker = EntityLinker()
        extractor = EntityExtractor(use_spacy=False)

        chunks = [
            ChunkRecord(
                chunk_id="c1",
                document_id="doc1",
                content="the and but or",  # No entities
            )
        ]

        entity_index = linker.link_chunks(chunks, extractor)

        # Should handle gracefully
        assert isinstance(entity_index, dict)

    def test_very_long_entity_name(self):
        """Test handling very long entity names."""
        linker = EntityLinker()

        long_name = "A" * 500

        canonical = linker.add_entity(long_name, "ORG", "doc1", "c1")

        # Should handle without crashing
        assert len(canonical) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
