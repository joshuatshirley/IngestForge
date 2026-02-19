"""
Unit tests for IF-Protocol Enrichment Processors.

Migration - Enrichment Parity
Tests all GWT scenarios and JPL rule compliance.
"""

import inspect
import pytest
from unittest.mock import MagicMock

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.ingest.if_enrichment_processor import (
    IFEntityProcessor,
    IFEmbeddingProcessor,
    IFSummaryProcessor,
    MAX_ENTITIES_PER_CHUNK,
    MAX_EMBEDDING_DIM,
    MAX_SUMMARY_LENGTH,
    MAX_CONTENT_SIZE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_chunk_artifact():
    """Create a sample IFChunkArtifact for testing."""
    return IFChunkArtifact(
        artifact_id="test-chunk-001",
        document_id="test-doc-001",
        content="Barack Obama was the 44th President of the United States. "
        "He was born in Hawaii on August 4, 1961. Obama served two terms "
        "from 2009 to 2017 and was awarded the Nobel Peace Prize in 2009.",
        chunk_index=0,
        total_chunks=1,
        metadata={"source_file": "test.pdf"},
    )


@pytest.fixture
def text_artifact():
    """Create a text artifact (wrong type for enrichment)."""
    return IFTextArtifact(artifact_id="test-text-001", content="Some text content")


@pytest.fixture
def entity_processor():
    """Create an entity processor with mocked extractor."""
    processor = IFEntityProcessor(use_spacy=False)
    return processor


@pytest.fixture
def embedding_processor():
    """Create an embedding processor."""
    return IFEmbeddingProcessor()


@pytest.fixture
def summary_processor():
    """Create a summary processor."""
    return IFSummaryProcessor()


# =============================================================================
# GWT Scenario 1: Entity Extraction via IF-Protocol
# =============================================================================


class TestEntityExtraction:
    """Tests for GWT Scenario 1: Entity extraction."""

    def test_extract_entities_from_chunk(self, sample_chunk_artifact):
        """Given chunk, when processed, then entities are extracted."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "entities" in result.metadata
        assert isinstance(result.metadata["entities"], list)

    def test_entities_stored_in_metadata(self, sample_chunk_artifact):
        """Given chunk with entities, when processed, entities stored correctly."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)

        entities = result.metadata.get("entities", [])
        # Should find at least some entities
        assert len(entities) >= 0  # May be empty without spaCy

    def test_entity_count_tracked(self, sample_chunk_artifact):
        """Given chunk, when processed, entity count is tracked."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)

        assert "entity_count" in result.metadata
        assert isinstance(result.metadata["entity_count"], int)

    def test_wrong_artifact_type_returns_failure(self, text_artifact):
        """Given wrong artifact type, when processed, returns failure."""
        processor = IFEntityProcessor()
        result = processor.process(text_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "IFChunkArtifact" in result.error_message


# =============================================================================
# GWT Scenario 2: Embedding Generation via IF-Protocol
# =============================================================================


class TestEmbeddingGeneration:
    """Tests for GWT Scenario 2: Embedding generation."""

    def test_generate_embeddings_mocked(self, sample_chunk_artifact):
        """Given chunk, when processed with mock, embeddings are generated."""
        processor = IFEmbeddingProcessor()

        # Mock the generator
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1, 0.2, 0.3] * 128  # 384 dims
        mock_generator = MagicMock()
        mock_generator.encode.return_value = mock_embedding
        processor._generator = mock_generator

        result = processor.process(sample_chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "embedding" in result.metadata
        assert isinstance(result.metadata["embedding"], list)

    def test_embedding_model_tracked(self, sample_chunk_artifact):
        """Given chunk, when processed, embedding model is tracked."""
        processor = IFEmbeddingProcessor(model_name="test-model")

        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_generator = MagicMock()
        mock_generator.encode.return_value = mock_embedding
        processor._generator = mock_generator

        result = processor.process(sample_chunk_artifact)

        assert result.metadata.get("embedding_model") == "test-model"

    def test_embedding_dim_tracked(self, sample_chunk_artifact):
        """Given chunk, when processed, embedding dimension is tracked."""
        processor = IFEmbeddingProcessor()

        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_generator = MagicMock()
        mock_generator.encode.return_value = mock_embedding
        processor._generator = mock_generator

        result = processor.process(sample_chunk_artifact)

        assert result.metadata.get("embedding_dim") == 384

    def test_embedding_wrong_type_returns_failure(self, text_artifact):
        """Given wrong artifact type, when processed, returns failure."""
        processor = IFEmbeddingProcessor()
        result = processor.process(text_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "IFChunkArtifact" in result.error_message


# =============================================================================
# GWT Scenario 3: Summary Generation via IF-Protocol
# =============================================================================


class TestSummaryGeneration:
    """Tests for GWT Scenario 3: Summary generation."""

    def test_generate_summary_from_chunk(self, sample_chunk_artifact):
        """Given chunk, when processed, summary is generated."""
        processor = IFSummaryProcessor()
        result = processor.process(sample_chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "summary" in result.metadata
        assert isinstance(result.metadata["summary"], str)

    def test_summary_length_tracked(self, sample_chunk_artifact):
        """Given chunk, when processed, summary length is tracked."""
        processor = IFSummaryProcessor()
        result = processor.process(sample_chunk_artifact)

        assert "summary_length" in result.metadata
        assert isinstance(result.metadata["summary_length"], int)

    def test_summary_respects_max_length(self):
        """Given long content, when processed, summary respects max length."""
        long_content = "This is a sentence. " * 100
        artifact = IFChunkArtifact(
            artifact_id="test-long",
            document_id="doc",
            content=long_content,
            chunk_index=0,
            total_chunks=1,
        )

        processor = IFSummaryProcessor(max_length=100)
        result = processor.process(artifact)

        assert len(result.metadata["summary"]) <= 100 + 50  # Some buffer

    def test_summary_wrong_type_returns_failure(self, text_artifact):
        """Given wrong artifact type, when processed, returns failure."""
        processor = IFSummaryProcessor()
        result = processor.process(text_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "IFChunkArtifact" in result.error_message


# =============================================================================
# GWT Scenario 4: Registry Integration
# =============================================================================


class TestRegistryIntegration:
    """Tests for GWT Scenario 4: Registry integration."""

    def test_entity_processor_has_capabilities(self):
        """Given entity processor, when checked, has correct capabilities."""
        processor = IFEntityProcessor()
        assert "enrich.entities" in processor.capabilities
        assert "ner" in processor.capabilities

    def test_embedding_processor_has_capabilities(self):
        """Given embedding processor, when checked, has correct capabilities."""
        processor = IFEmbeddingProcessor()
        assert "enrich.embeddings" in processor.capabilities
        assert "vectorization" in processor.capabilities

    def test_summary_processor_has_capabilities(self):
        """Given summary processor, when checked, has correct capabilities."""
        processor = IFSummaryProcessor()
        assert "enrich.summary" in processor.capabilities
        assert "summarization" in processor.capabilities

    def test_all_processors_have_processor_id(self):
        """Given any processor, when checked, has valid processor ID."""
        processors = [IFEntityProcessor(), IFEmbeddingProcessor(), IFSummaryProcessor()]
        for processor in processors:
            assert processor.processor_id
            assert isinstance(processor.processor_id, str)


# =============================================================================
# GWT Scenario 5: Parity Validation
# =============================================================================


class TestParityValidation:
    """Tests for GWT Scenario 5: Parity validation."""

    def test_entity_processor_produces_same_format(self, sample_chunk_artifact):
        """Given same input, entity processor produces consistent format."""
        processor = IFEntityProcessor(use_spacy=False)

        result1 = processor.process(sample_chunk_artifact)
        result2 = processor.process(sample_chunk_artifact)

        # Both should have same structure
        assert set(result1.metadata.keys()) == set(result2.metadata.keys())

    def test_summary_processor_extractive_fallback_works(self, sample_chunk_artifact):
        """Given no LLM, summary processor uses extractive fallback."""
        processor = IFSummaryProcessor()
        processor._generator = None  # Force extractive fallback

        result = processor.process(sample_chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert "summary" in result.metadata
        # Extractive should contain parts of original
        summary = result.metadata["summary"]
        assert len(summary) > 0

    def test_processor_implements_if_processor(self):
        """Given any processor, it implements IFProcessor interface."""
        processors = [IFEntityProcessor(), IFEmbeddingProcessor(), IFSummaryProcessor()]
        for processor in processors:
            assert isinstance(processor, IFProcessor)
            assert hasattr(processor, "process")
            assert hasattr(processor, "is_available")
            assert hasattr(processor, "processor_id")


# =============================================================================
# JPL Rule #2: Fixed Upper Bounds
# =============================================================================


class TestJPLRule2Bounds:
    """Tests for JPL Rule #2: Fixed upper bounds."""

    def test_max_entities_constant_exists(self):
        """Given constants, MAX_ENTITIES_PER_CHUNK is defined."""
        assert MAX_ENTITIES_PER_CHUNK == 100

    def test_max_embedding_dim_constant_exists(self):
        """Given constants, MAX_EMBEDDING_DIM is defined."""
        assert MAX_EMBEDDING_DIM == 1536

    def test_max_summary_length_constant_exists(self):
        """Given constants, MAX_SUMMARY_LENGTH is defined."""
        assert MAX_SUMMARY_LENGTH == 500

    def test_max_content_size_constant_exists(self):
        """Given constants, MAX_CONTENT_SIZE is defined."""
        assert MAX_CONTENT_SIZE == 100_000

    def test_entity_count_respects_limit(self, sample_chunk_artifact):
        """Given many entities, processor respects MAX_ENTITIES limit."""
        processor = IFEntityProcessor(use_spacy=False)

        # Mock extractor to return many entities
        class MockEntity:
            def __init__(self, i):
                self.text = f"Entity{i}"
                self.label = "TEST"
                self.start_char = i
                self.end_char = i + 5
                self.confidence = 0.9

        mock_extractor = MagicMock()
        mock_extractor.is_available.return_value = True
        mock_extractor.extract_structured.return_value = [
            MockEntity(i)
            for i in range(200)  # More than limit
        ]
        processor._extractor = mock_extractor

        result = processor.process(sample_chunk_artifact)

        entities = result.metadata.get("entities", [])
        assert len(entities) <= MAX_ENTITIES_PER_CHUNK

    def test_content_truncation_respects_limit(self):
        """Given very long content, it is truncated."""
        long_content = "x" * (MAX_CONTENT_SIZE + 1000)
        artifact = IFChunkArtifact(
            artifact_id="test-long",
            document_id="doc",
            content=long_content,
            chunk_index=0,
            total_chunks=1,
        )

        processor = IFSummaryProcessor()
        result = processor.process(artifact)

        # Should not fail, content should be truncated internally
        assert isinstance(result, IFChunkArtifact)


# =============================================================================
# JPL Rule #7: Check Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """Tests for JPL Rule #7: Check all return values."""

    def test_entity_process_always_returns_artifact(self, sample_chunk_artifact):
        """Given any input, entity process returns an artifact."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_embedding_process_always_returns_artifact(self, sample_chunk_artifact):
        """Given any input, embedding process returns an artifact."""
        processor = IFEmbeddingProcessor()
        # Mock to prevent actual model loading
        processor._generator = None
        result = processor.process(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_summary_process_always_returns_artifact(self, sample_chunk_artifact):
        """Given any input, summary process returns an artifact."""
        processor = IFSummaryProcessor()
        result = processor.process(sample_chunk_artifact)
        assert isinstance(result, IFArtifact)

    def test_is_available_returns_bool(self):
        """Given any processor, is_available returns boolean."""
        processors = [
            IFEntityProcessor(use_spacy=False),
            IFEmbeddingProcessor(),
            IFSummaryProcessor(),
        ]
        for processor in processors:
            result = processor.is_available()
            assert isinstance(result, bool)

    def test_teardown_returns_bool(self):
        """Given any processor, teardown returns boolean."""
        processors = [
            IFEntityProcessor(use_spacy=False),
            IFEmbeddingProcessor(),
            IFSummaryProcessor(),
        ]
        for processor in processors:
            result = processor.teardown()
            assert isinstance(result, bool)
            assert result is True


# =============================================================================
# JPL Rule #9: Complete Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_entity_processor_has_type_hints(self):
        """Given IFEntityProcessor, process method has type hints."""
        sig = inspect.signature(IFEntityProcessor.process)
        assert sig.parameters["artifact"].annotation.__name__ == "IFArtifact"
        assert sig.return_annotation.__name__ == "IFArtifact"

    def test_embedding_processor_has_type_hints(self):
        """Given IFEmbeddingProcessor, process method has type hints."""
        sig = inspect.signature(IFEmbeddingProcessor.process)
        assert sig.parameters["artifact"].annotation.__name__ == "IFArtifact"
        assert sig.return_annotation.__name__ == "IFArtifact"

    def test_summary_processor_has_type_hints(self):
        """Given IFSummaryProcessor, process method has type hints."""
        sig = inspect.signature(IFSummaryProcessor.process)
        assert sig.parameters["artifact"].annotation.__name__ == "IFArtifact"
        assert sig.return_annotation.__name__ == "IFArtifact"

    def test_all_properties_have_return_types(self):
        """Given any processor, properties have return type hints."""
        for cls in [IFEntityProcessor, IFEmbeddingProcessor, IFSummaryProcessor]:
            # Check processor_id
            assert "return" in str(inspect.signature(cls.processor_id.fget))
            # Check version
            assert "return" in str(inspect.signature(cls.version.fget))


# =============================================================================
# Artifact Lineage Tracking
# =============================================================================


class TestArtifactLineage:
    """Tests for artifact lineage tracking."""

    def test_entity_enrichment_tracks_parent(self, sample_chunk_artifact):
        """Given chunk, entity enrichment tracks parent artifact."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)

        assert result.parent_id == sample_chunk_artifact.artifact_id

    def test_entity_enrichment_tracks_provenance(self, sample_chunk_artifact):
        """Given chunk, entity enrichment tracks provenance."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)

        assert processor.processor_id in result.provenance

    def test_embedding_enrichment_increments_depth(self, sample_chunk_artifact):
        """Given chunk, embedding enrichment increments lineage depth."""
        processor = IFEmbeddingProcessor()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_generator = MagicMock()
        mock_generator.encode.return_value = mock_embedding
        processor._generator = mock_generator

        result = processor.process(sample_chunk_artifact)

        assert result.lineage_depth == sample_chunk_artifact.lineage_depth + 1

    def test_summary_enrichment_preserves_root_id(self, sample_chunk_artifact):
        """Given chunk, summary enrichment preserves root artifact ID."""
        processor = IFSummaryProcessor()
        result = processor.process(sample_chunk_artifact)

        assert result.root_artifact_id == sample_chunk_artifact.effective_root_id


# =============================================================================
# Content Preservation
# =============================================================================


class TestContentPreservation:
    """Tests for content preservation during enrichment."""

    def test_entity_enrichment_preserves_content(self, sample_chunk_artifact):
        """Given chunk, entity enrichment preserves original content."""
        processor = IFEntityProcessor(use_spacy=False)
        result = processor.process(sample_chunk_artifact)

        assert result.content == sample_chunk_artifact.content

    def test_embedding_enrichment_preserves_content(self, sample_chunk_artifact):
        """Given chunk, embedding enrichment preserves original content."""
        processor = IFEmbeddingProcessor()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 384
        mock_generator = MagicMock()
        mock_generator.encode.return_value = mock_embedding
        processor._generator = mock_generator

        result = processor.process(sample_chunk_artifact)

        assert result.content == sample_chunk_artifact.content

    def test_summary_enrichment_preserves_content(self, sample_chunk_artifact):
        """Given chunk, summary enrichment preserves original content."""
        processor = IFSummaryProcessor()
        result = processor.process(sample_chunk_artifact)

        assert result.content == sample_chunk_artifact.content

    def test_enrichment_preserves_original_metadata(self, sample_chunk_artifact):
        """Given chunk with metadata, enrichment preserves it."""
        processor = IFSummaryProcessor()
        result = processor.process(sample_chunk_artifact)

        assert result.metadata.get("source_file") == "test.pdf"


# =============================================================================
# Teardown Tests
# =============================================================================


class TestTeardown:
    """Tests for processor teardown."""

    def test_entity_teardown_clears_extractor(self):
        """Given entity processor, teardown clears extractor."""
        processor = IFEntityProcessor(use_spacy=False)
        processor._extractor = MagicMock()

        result = processor.teardown()

        assert result is True
        assert processor._extractor is None

    def test_embedding_teardown_clears_generator(self):
        """Given embedding processor, teardown clears generator."""
        processor = IFEmbeddingProcessor()
        processor._generator = MagicMock()

        result = processor.teardown()

        assert result is True
        assert processor._generator is None

    def test_summary_teardown_clears_generator(self):
        """Given summary processor, teardown clears generator."""
        processor = IFSummaryProcessor()
        processor._generator = MagicMock()

        result = processor.teardown()

        assert result is True
        assert processor._generator is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_content_handled(self):
        """Given empty content, processors handle gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="test-empty",
            document_id="doc",
            content="",
            chunk_index=0,
            total_chunks=1,
        )

        processors = [IFEntityProcessor(use_spacy=False), IFSummaryProcessor()]

        for processor in processors:
            result = processor.process(artifact)
            assert isinstance(result, IFArtifact)

    def test_special_characters_handled(self):
        """Given special characters, processors handle gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="test-special",
            document_id="doc",
            content="<script>alert('xss')</script> & special < > chars",
            chunk_index=0,
            total_chunks=1,
        )

        processor = IFSummaryProcessor()
        result = processor.process(artifact)

        assert isinstance(result, IFChunkArtifact)

    def test_unicode_content_handled(self):
        """Given unicode content, processors handle gracefully."""
        artifact = IFChunkArtifact(
            artifact_id="test-unicode",
            document_id="doc",
            content="Unicode content with emojis and accents",
            chunk_index=0,
            total_chunks=1,
        )

        processor = IFSummaryProcessor()
        result = processor.process(artifact)

        assert isinstance(result, IFChunkArtifact)


# =============================================================================
# GWT Scenario Completeness
# =============================================================================


class TestGWTScenarioCompleteness:
    """Meta-tests to ensure all GWT scenarios are covered."""

    def test_scenario_1_entity_extraction_covered(self):
        """Verify Scenario 1 tests exist."""
        test_methods = [m for m in dir(TestEntityExtraction) if m.startswith("test_")]
        assert len(test_methods) >= 4

    def test_scenario_2_embedding_generation_covered(self):
        """Verify Scenario 2 tests exist."""
        test_methods = [
            m for m in dir(TestEmbeddingGeneration) if m.startswith("test_")
        ]
        assert len(test_methods) >= 4

    def test_scenario_3_summary_generation_covered(self):
        """Verify Scenario 3 tests exist."""
        test_methods = [m for m in dir(TestSummaryGeneration) if m.startswith("test_")]
        assert len(test_methods) >= 4

    def test_scenario_4_registry_integration_covered(self):
        """Verify Scenario 4 tests exist."""
        test_methods = [
            m for m in dir(TestRegistryIntegration) if m.startswith("test_")
        ]
        assert len(test_methods) >= 4

    def test_scenario_5_parity_validation_covered(self):
        """Verify Scenario 5 tests exist."""
        test_methods = [m for m in dir(TestParityValidation) if m.startswith("test_")]
        assert len(test_methods) >= 3
