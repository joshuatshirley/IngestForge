import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.edu import EduMetadataRefiner


class TestEduMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return EduMetadataRefiner()

    def test_enrich_lesson_plan(self, refiner):
        text = """
        Subject: Mathematics
        Grade: 5
        Standard: CCSS.MATH.CONTENT.5.NBT.A.1
        Objectives: Understand place value systems.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["edu_subject"] == "Mathematics"
        assert enriched.metadata["edu_grade_level"] == "5"
        assert "CCSS.MATH.CONTENT.5.NBT.A.1" in enriched.metadata["edu_standards"]

    def test_higher_ed_syllabus(self, refiner):
        text = "Course: Introduction to Quantum Mechanics. Level: Higher Ed. Standards: NGSS-PHYS-1."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["edu_subject"] == "Introduction to Quantum Mechanics"
        assert enriched.metadata["edu_grade_level"] == "Higher ed"
        assert "NGSS-PHYS-1" in enriched.metadata["edu_standards"]
