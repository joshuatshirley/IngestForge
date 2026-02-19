"""
Tests for IFEnricherAdapter (Convergence - Processor Unification).

Validates that the adapter correctly bridges IEnricher to IFProcessor.
All tests follow GWT (Given-When-Then) behavioral specification.
"""

import pytest
import warnings
from typing import Any

from ingestforge.core.pipeline.interfaces import IFProcessor
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.core.pipeline.enricher_adapter import (
    IFEnricherAdapter,
    adapt_enricher,
    adapt_enrichers,
)
from ingestforge.core.pipeline.enricher_adapter import IEnricher


# =============================================================================
# MOCK ENRICHERS FOR TESTING
# =============================================================================


class MockEmbeddingEnricher(IEnricher):
    """Mock enricher that adds embeddings."""

    def enrich_chunk(self, chunk: Any) -> Any:
        chunk.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        return chunk

    def is_available(self) -> bool:
        return True


class MockEntityEnricher(IEnricher):
    """Mock enricher that adds entities."""

    def enrich_chunk(self, chunk: Any) -> Any:
        chunk.entities = ["PERSON:Alice", "ORG:Acme Corp"]
        return chunk

    def is_available(self) -> bool:
        return True


class UnavailableEnricher(IEnricher):
    """Mock enricher that is not available."""

    def enrich_chunk(self, chunk: Any) -> Any:
        return chunk

    def is_available(self) -> bool:
        return False


class CrashingEnricher(IEnricher):
    """Mock enricher that crashes during processing."""

    def enrich_chunk(self, chunk: Any) -> Any:
        raise RuntimeError("Enricher crashed intentionally")

    def is_available(self) -> bool:
        return True


class MultiFieldEnricher(IEnricher):
    """Mock enricher that adds multiple fields."""

    def enrich_chunk(self, chunk: Any) -> Any:
        chunk.embedding = [0.5, 0.5]
        chunk.entities = ["TEST"]
        chunk.summary = "Test summary"
        chunk.quality_score = 0.95
        chunk.keywords = ["test", "mock"]
        return chunk

    def is_available(self) -> bool:
        return True


# =============================================================================
# ADAPTER INTERFACE TESTS
# =============================================================================


class TestAdapterInterface:
    """Tests verifying the adapter implements IFProcessor correctly."""

    def test_adapter_is_if_processor(self):
        """
        GWT:
        Given a legacy IEnricher wrapped in adapter
        When the adapter type is checked
        Then it is an IFProcessor instance.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert isinstance(adapter, IFProcessor)

    def test_adapter_has_processor_id(self):
        """
        GWT:
        Given an adapter without custom ID
        When processor_id is accessed
        Then it contains the enricher class name.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert "MockEmbeddingEnricher" in adapter.processor_id

    def test_adapter_custom_processor_id(self):
        """
        GWT:
        Given an adapter with custom ID
        When processor_id is accessed
        Then it returns the custom ID.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher, processor_id="custom-embedder")

        assert adapter.processor_id == "custom-embedder"

    def test_adapter_has_version(self):
        """
        GWT:
        Given an adapter
        When version is accessed
        Then it returns a semver string.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert adapter.version == "1.0.0"

    def test_adapter_delegates_is_available(self):
        """
        GWT:
        Given an adapter wrapping an available enricher
        When is_available() is called
        Then it returns True.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert adapter.is_available() is True

    def test_adapter_delegates_unavailable(self):
        """
        GWT:
        Given an adapter wrapping an unavailable enricher
        When is_available() is called
        Then it returns False.
        """
        enricher = UnavailableEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert adapter.is_available() is False

    def test_adapter_teardown_succeeds(self):
        """
        GWT:
        Given an adapter
        When teardown() is called
        Then it returns True.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert adapter.teardown() is True


# =============================================================================
# PROCESS METHOD TESTS
# =============================================================================


class TestAdapterProcess:
    """Tests for the process() method."""

    def test_process_enriches_chunk_artifact(self):
        """
        GWT:
        Given an adapter wrapping an embedding enricher
        When process() is called with IFChunkArtifact
        Then it returns enriched artifact with embedding in metadata.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        artifact = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
        )

        result = adapter.process(artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "embedding" in result.metadata
        assert result.metadata["embedding"] == [0.1, 0.2, 0.3, 0.4, 0.5]

    def test_process_rejects_non_chunk_artifact(self):
        """
        GWT:
        Given an adapter
        When process() is called with non-IFChunkArtifact
        Then it returns IFFailureArtifact.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        text_artifact = IFTextArtifact(artifact_id="text-1", content="Not a chunk")

        result = adapter.process(text_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "requires IFChunkArtifact" in result.error_message

    def test_process_handles_enricher_crash(self):
        """
        GWT:
        Given an adapter wrapping a crashing enricher
        When process() is called
        Then it returns IFFailureArtifact with error details.
        """
        enricher = CrashingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        artifact = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
        )

        result = adapter.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "crashed intentionally" in result.error_message

    def test_process_preserves_lineage(self):
        """
        GWT:
        Given an adapter
        When process() is called
        Then result has correct parent_id and provenance.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        artifact = IFChunkArtifact(
            artifact_id="chunk-original",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
        )

        result = adapter.process(artifact)

        assert result.parent_id == "chunk-original"
        assert adapter.processor_id in result.provenance

    def test_process_extracts_multiple_fields(self):
        """
        GWT:
        Given an adapter wrapping a multi-field enricher
        When process() is called
        Then all enrichment fields are in metadata.
        """
        enricher = MultiFieldEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        artifact = IFChunkArtifact(
            artifact_id="chunk-1",
            document_id="doc-1",
            content="Test content",
            chunk_index=0,
            total_chunks=1,
        )

        result = adapter.process(artifact)

        assert result.metadata["embedding"] == [0.5, 0.5]
        assert result.metadata["entities"] == ["TEST"]
        assert result.metadata["summary"] == "Test summary"
        assert result.metadata["quality_score"] == 0.95
        assert result.metadata["keywords"] == ["test", "mock"]


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunctions:
    """Tests for adapt_enricher and adapt_enrichers functions."""

    def test_adapt_enricher_creates_adapter(self):
        """
        GWT:
        Given a legacy enricher
        When adapt_enricher() is called
        Then it returns an IFEnricherAdapter.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = adapt_enricher(enricher)

        assert isinstance(adapter, IFEnricherAdapter)

    def test_adapt_enricher_with_custom_id(self):
        """
        GWT:
        Given a legacy enricher and custom ID
        When adapt_enricher() is called
        Then adapter has the custom ID.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = adapt_enricher(enricher, processor_id="my-embedder")

        assert adapter.processor_id == "my-embedder"

    def test_adapt_enrichers_batch(self):
        """
        GWT:
        Given multiple legacy enrichers
        When adapt_enrichers() is called
        Then it returns list of adapters.
        """
        enrichers = [
            MockEmbeddingEnricher(),
            MockEntityEnricher(),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapters = adapt_enrichers(enrichers)

        assert len(adapters) == 2
        assert all(isinstance(a, IFEnricherAdapter) for a in adapters)


# =============================================================================
# DEPRECATION WARNING TESTS
# =============================================================================


class TestDeprecationWarnings:
    """Tests for deprecation warning behavior."""

    def test_adapter_emits_deprecation_warning(self):
        """
        GWT:
        Given a legacy enricher
        When wrapped in adapter
        Then DeprecationWarning is emitted.
        """
        enricher = MockEmbeddingEnricher()

        with pytest.warns(DeprecationWarning, match="wrapped via adapter"):
            IFEnricherAdapter(enricher)

    def test_deprecation_mentions_migration(self):
        """
        GWT:
        Given a legacy enricher
        When wrapped in adapter
        Then warning suggests migration to IFProcessor.
        """
        enricher = MockEmbeddingEnricher()

        with pytest.warns(DeprecationWarning, match="IFProcessor"):
            IFEnricherAdapter(enricher)


# =============================================================================
# WRAPPED ENRICHER ACCESS TESTS
# =============================================================================


class TestWrappedEnricherAccess:
    """Tests for accessing the wrapped enricher."""

    def test_wrapped_enricher_accessible(self):
        """
        GWT:
        Given an adapter
        When wrapped_enricher is accessed
        Then the original enricher is returned.
        """
        enricher = MockEmbeddingEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert adapter.wrapped_enricher is enricher

    def test_wrapped_enricher_type_preserved(self):
        """
        GWT:
        Given an adapter wrapping specific enricher type
        When wrapped_enricher is accessed
        Then it has correct type.
        """
        enricher = MockEntityEnricher()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            adapter = IFEnricherAdapter(enricher)

        assert isinstance(adapter.wrapped_enricher, MockEntityEnricher)
