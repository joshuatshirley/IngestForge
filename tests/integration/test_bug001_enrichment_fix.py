import pytest
from typing import List

from ingestforge.core.pipeline.enrichment_stage import IFEnrichmentStage
from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import IFChunkArtifact


class MockEnricher(IFProcessor):
    @property
    def processor_id(self) -> str:
        return "mock-enricher"

    @property
    def capabilities(self) -> List[str]:
        return ["embedding"]

    @property
    def version(self) -> str:
        return "1.0.0"

    @property
    def memory_mb(self) -> int:
        return 10

    def is_available(self) -> bool:
        return True

    def teardown(self) -> bool:
        return True

    def process(self, artifact: IFArtifact) -> IFArtifact:
        if isinstance(artifact, IFChunkArtifact):
            artifact.metadata["enriched"] = True
        return artifact


def test_enrich_batch_backward_compatibility():
    """Verify that IFEnrichmentStage handles legacy ChunkRecord batches (BUG001)."""
    # 1. Setup mock processor and stage
    proc = MockEnricher()
    stage = IFEnrichmentStage(processors=[proc])

    # 2. Create legacy ChunkRecord-like dicts
    legacy_chunks = [
        {"content": "Chunk 1", "metadata": {}},
        {"content": "Chunk 2", "metadata": {}},
    ]

    # 3. Call legacy interface (This was causing AttributeError)
    try:
        enriched = stage.enrich_batch(legacy_chunks)

        # 4. Assertions
        assert len(enriched) == 2
        assert enriched[0]["metadata"]["enriched"] is True
        assert enriched[1]["metadata"]["enriched"] is True
        print("BUG001 Verification: Success")
    except AttributeError as e:
        pytest.fail(f"BUG001 Regression: {e}")


if __name__ == "__main__":
    test_enrich_batch_backward_compatibility()
