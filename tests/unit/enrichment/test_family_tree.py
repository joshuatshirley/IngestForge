import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.family_tree import FamilyTreeEnricher


class TestFamilyTreeEnricher:
    @pytest.fixture
    def enricher(self):
        return FamilyTreeEnricher()

    def test_enrich_parent_child(self, enricher):
        chunks = [
            ChunkRecord(
                chunk_id="p1",
                document_id="d1",
                content="John",
                chunk_type="INDI",
                metadata={"id": "@I1@"},
                section_title="John Doe",
            ),
            ChunkRecord(
                chunk_id="p2",
                document_id="d1",
                content="Mary",
                chunk_type="INDI",
                metadata={"id": "@I2@"},
                section_title="Mary Smith",
            ),
            ChunkRecord(
                chunk_id="p3",
                document_id="d1",
                content="Bob",
                chunk_type="INDI",
                metadata={"id": "@I3@"},
                section_title="Bob Doe",
            ),
            ChunkRecord(
                chunk_id="f1",
                document_id="d1",
                content="Family",
                chunk_type="FAM",
                metadata={"HUSB": "@I1@", "WIFE": "@I2@", "CHIL": ["@I3@"]},
            ),
        ]

        enriched = enricher.enrich(chunks)

        # John should be parent of Bob
        john = next(c for c in enriched if c.chunk_id == "p1")
        rel_types = [r["predicate"] for r in john.metadata["relationships"]]
        assert "parent_of" in rel_types
        assert "spouse_of" in rel_types

        # Verify Bob is the object of John's parent_of relation
        bob_rel = next(
            r for r in john.metadata["relationships"] if r["predicate"] == "parent_of"
        )
        assert bob_rel["object"] == "Bob Doe"
