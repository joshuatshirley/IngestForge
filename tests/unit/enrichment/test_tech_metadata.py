import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.tech_metadata import TechMetadataRefiner


class TestTechMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return TechMetadataRefiner()

    def test_enrich_python_imports(self, refiner):
        code = """
import os
from datetime import datetime
import numpy as np

def my_func():
    pass
"""
        chunk = ChunkRecord(
            chunk_id="c1",
            document_id="d1",
            content=code,
            chunk_type="code_function",
            source_file="test.py",
        )

        enriched = refiner.enrich(chunk)

        assert "language" in enriched.metadata
        assert enriched.metadata["language"] == "python"
        assert "imports" in enriched.metadata
        assert "import os" in enriched.metadata["imports"]
        assert "from datetime import datetime" in enriched.metadata["imports"]

    def test_non_code_chunk(self, refiner):
        chunk = ChunkRecord(
            chunk_id="c2",
            document_id="d2",
            content="Just some text",
            chunk_type="text",
            source_file="test.txt",
        )

        enriched = refiner.enrich(chunk)
        assert enriched.metadata is None or "language" not in enriched.metadata
