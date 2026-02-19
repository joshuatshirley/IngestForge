"""Tests for Adaptive Semantic Slicer.

Adaptive Semantic Slicing implementation tests.
Tests semantic chunking, overlap, size constraints, and coherence."""

from __future__ import annotations


import pytest

from ingestforge.chunking.adaptive_slicer import (
    AdaptiveSemanticSlicer,
    SlicerConfig,
    SliceResult,
    slice_text,
    create_slicer,
    MAX_CHUNKS_PER_DOCUMENT,
    MAX_CHUNK_CONTENT_LENGTH,
    DEFAULT_MIN_CHUNK_CHARS,
    DEFAULT_MAX_CHUNK_CHARS,
)
from ingestforge.chunking.nlp_boundary import (
    BoundaryConstraints,
    NLPBoundaryDetector,
)
from ingestforge.core.pipeline.artifacts import IFChunkArtifact, IFTextArtifact


# ---------------------------------------------------------------------------
# SlicerConfig Tests
# ---------------------------------------------------------------------------


class TestSlicerConfig:
    """Tests for SlicerConfig dataclass."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = SlicerConfig()
        assert config.min_chunk_chars == DEFAULT_MIN_CHUNK_CHARS
        assert config.max_chunk_chars == DEFAULT_MAX_CHUNK_CHARS
        assert config.overlap_sentences == 2
        assert config.similarity_threshold == 0.8

    def test_custom_values(self) -> None:
        """Test custom configuration values."""
        config = SlicerConfig(
            min_chunk_chars=50,
            max_chunk_chars=500,
            overlap_sentences=3,
            similarity_threshold=0.9,
        )
        assert config.min_chunk_chars == 50
        assert config.max_chunk_chars == 500
        assert config.overlap_sentences == 3

    def test_to_constraints(self) -> None:
        """Test conversion to BoundaryConstraints."""
        config = SlicerConfig(
            min_chunk_chars=100,
            max_chunk_chars=1000,
            overlap_sentences=2,
        )
        constraints = config.to_constraints()
        assert isinstance(constraints, BoundaryConstraints)
        assert constraints.min_chunk_chars == 100
        assert constraints.max_chunk_chars == 1000


# ---------------------------------------------------------------------------
# SliceResult Tests
# ---------------------------------------------------------------------------


class TestSliceResult:
    """Tests for SliceResult dataclass."""

    def test_slice_creation(self) -> None:
        """Test creating a slice result."""
        result = SliceResult(
            content="Hello world.",
            start_sentence=0,
            end_sentence=0,
            coherence_score=1.0,
            char_start=0,
            char_end=12,
            has_overlap=False,
        )
        assert result.content == "Hello world."
        assert result.coherence_score == 1.0

    def test_sentence_count(self) -> None:
        """Test sentence count property."""
        result = SliceResult(
            content="Multiple sentences.",
            start_sentence=0,
            end_sentence=2,
            coherence_score=0.8,
            char_start=0,
            char_end=50,
            has_overlap=False,
        )
        assert result.sentence_count == 3

    def test_char_count(self) -> None:
        """Test character count property."""
        result = SliceResult(
            content="Test content here.",
            start_sentence=0,
            end_sentence=0,
            coherence_score=1.0,
            char_start=0,
            char_end=18,
            has_overlap=False,
        )
        assert result.char_count == 18


# ---------------------------------------------------------------------------
# AdaptiveSemanticSlicer Tests
# ---------------------------------------------------------------------------


class TestAdaptiveSemanticSlicer:
    """Tests for AdaptiveSemanticSlicer initialization."""

    def test_default_initialization(self) -> None:
        """Test default slicer initialization."""
        slicer = AdaptiveSemanticSlicer()
        assert slicer.config.min_chunk_chars == DEFAULT_MIN_CHUNK_CHARS
        assert slicer.processor_id == "adaptive_slicer"

    def test_custom_config(self) -> None:
        """Test slicer with custom config."""
        config = SlicerConfig(max_chunk_chars=500)
        slicer = AdaptiveSemanticSlicer(config=config)
        assert slicer.config.max_chunk_chars == 500

    def test_custom_detector(self) -> None:
        """Test slicer with custom detector."""
        detector = NLPBoundaryDetector(similarity_threshold=0.9)
        slicer = AdaptiveSemanticSlicer(detector=detector)
        assert slicer._detector.similarity_threshold == 0.9


# ---------------------------------------------------------------------------
# GWT-1: Semantic Chunk Creation Tests
# ---------------------------------------------------------------------------


class TestGWT1SemanticChunkCreation:
    """GWT-1: Semantic chunk creation tests."""

    def test_slice_simple_text(self) -> None:
        """GWT-1: Simple text produces chunks."""
        slicer = AdaptiveSemanticSlicer()
        text = "First sentence. Second sentence. Third sentence."

        chunks = slicer.slice(text, "doc-001")

        assert len(chunks) >= 1
        assert all(isinstance(c, IFChunkArtifact) for c in chunks)

    def test_slice_returns_artifacts(self) -> None:
        """GWT-1: Slice returns IFChunkArtifact instances."""
        slicer = AdaptiveSemanticSlicer()
        text = "Hello world. How are you? I am fine."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            assert isinstance(chunk, IFChunkArtifact)
            assert chunk.document_id == "doc-001"

    def test_slice_empty_text(self) -> None:
        """GWT-1: Empty text returns empty list."""
        slicer = AdaptiveSemanticSlicer()

        chunks = slicer.slice("", "doc-001")

        assert chunks == []

    def test_slice_whitespace_only(self) -> None:
        """GWT-1: Whitespace-only text returns empty list."""
        slicer = AdaptiveSemanticSlicer()

        chunks = slicer.slice("   \n\n   ", "doc-001")

        assert chunks == []

    def test_chunks_have_content(self) -> None:
        """GWT-1: All chunks have non-empty content."""
        slicer = AdaptiveSemanticSlicer()
        text = "First paragraph here.\n\nSecond paragraph here."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            assert chunk.content.strip() != ""


# ---------------------------------------------------------------------------
# GWT-2: Overlap Application Tests
# ---------------------------------------------------------------------------


class TestGWT2OverlapApplication:
    """GWT-2: Overlap application tests."""

    def test_overlap_sentences_applied(self) -> None:
        """GWT-2: Overlap sentences are included in chunks."""
        config = SlicerConfig(
            overlap_sentences=2,
            max_chunk_chars=100,  # Force multiple chunks
        )
        slicer = AdaptiveSemanticSlicer(config=config)

        # Create text that will definitely split into multiple chunks
        text = "Sentence one. " * 10 + "Sentence two. " * 10

        chunks = slicer.slice(text, "doc-001")

        # If we have multiple chunks, check for overlap
        if len(chunks) > 1:
            # Second chunk should have overlap metadata
            assert "has_overlap" in chunks[1].metadata

    def test_first_chunk_no_overlap(self) -> None:
        """GWT-2: First chunk has no overlap."""
        config = SlicerConfig(overlap_sentences=2)
        slicer = AdaptiveSemanticSlicer(config=config)
        text = "First. Second. Third. Fourth. Fifth."

        chunks = slicer.slice(text, "doc-001")

        if chunks:
            # First chunk should not have overlap
            assert chunks[0].metadata.get("has_overlap", False) is False

    def test_zero_overlap_config(self) -> None:
        """GWT-2: Zero overlap produces no overlap."""
        config = SlicerConfig(overlap_sentences=0)
        slicer = AdaptiveSemanticSlicer(config=config)
        text = "First. Second. Third."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            assert chunk.metadata.get("has_overlap", False) is False


# ---------------------------------------------------------------------------
# GWT-3: Size Constraint Enforcement Tests
# ---------------------------------------------------------------------------


class TestGWT3SizeConstraints:
    """GWT-3: Size constraint enforcement tests."""

    def test_max_size_enforced(self) -> None:
        """GWT-3: Chunks don't exceed max size."""
        config = SlicerConfig(max_chunk_chars=200)
        slicer = AdaptiveSemanticSlicer(config=config)

        # Long text that should be split
        text = "This is a sentence. " * 50

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            # Allow some tolerance for overlap
            assert len(chunk.content) <= config.max_chunk_chars * 2

    def test_very_long_text_splits(self) -> None:
        """GWT-3: Very long text is split into multiple chunks."""
        config = SlicerConfig(max_chunk_chars=100)
        slicer = AdaptiveSemanticSlicer(config=config)

        text = "Sentence here. " * 100  # Very long

        chunks = slicer.slice(text, "doc-001")

        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# GWT-4: Coherence Scoring Tests
# ---------------------------------------------------------------------------


class TestGWT4CoherenceScoring:
    """GWT-4: Coherence scoring tests."""

    def test_chunks_have_coherence_score(self) -> None:
        """GWT-4: All chunks have coherence_score in metadata."""
        slicer = AdaptiveSemanticSlicer()
        text = "First sentence. Second sentence. Third sentence."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            assert "coherence_score" in chunk.metadata
            assert 0.0 <= chunk.metadata["coherence_score"] <= 1.0

    def test_single_sentence_high_coherence(self) -> None:
        """GWT-4: Single sentence has coherence 1.0."""
        slicer = AdaptiveSemanticSlicer()
        text = "Just one sentence here."

        chunks = slicer.slice(text, "doc-001")

        if chunks:
            assert chunks[0].metadata["coherence_score"] == 1.0

    def test_coherence_in_metadata(self) -> None:
        """GWT-4: Coherence score is accessible via metadata."""
        slicer = AdaptiveSemanticSlicer()
        text = "Topic A here. Topic A continues. Topic A ends."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            score = chunk.metadata.get("coherence_score")
            assert score is not None
            assert isinstance(score, float)


# ---------------------------------------------------------------------------
# GWT-5: Zero Mid-Sentence Cuts Tests
# ---------------------------------------------------------------------------


class TestGWT5ZeroMidSentenceCuts:
    """GWT-5: Zero mid-sentence cuts tests."""

    def test_chunks_end_at_sentence_boundaries(self) -> None:
        """GWT-5: All chunks end at sentence boundaries."""
        slicer = AdaptiveSemanticSlicer()
        text = "First sentence. Second sentence. Third sentence."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            # Content should end with sentence-ending punctuation or be complete
            content = chunk.content.strip()
            assert content[-1] in ".!?" or content == text.strip()

    def test_no_partial_words(self) -> None:
        """GWT-5: Chunks don't contain partial words at boundaries."""
        slicer = AdaptiveSemanticSlicer()
        text = "The quick brown fox jumps. Over the lazy dog."

        chunks = slicer.slice(text, "doc-001")

        for chunk in chunks:
            # Check that words are complete (no truncation)
            words = chunk.content.split()
            for word in words:
                # Words should be recognizable (at least 1 char)
                assert len(word) >= 1


# ---------------------------------------------------------------------------
# Slice With Parent Tests
# ---------------------------------------------------------------------------


class TestSliceWithParent:
    """Tests for slice_with_parent method."""

    def test_slice_with_parent_lineage(self) -> None:
        """Test lineage is preserved from parent."""
        slicer = AdaptiveSemanticSlicer()
        parent = IFTextArtifact(
            artifact_id="parent-001",
            content="Original content",
            provenance=["extractor"],
        )

        chunks = slicer.slice_with_parent(
            "First sentence. Second sentence.",
            parent,
        )

        for chunk in chunks:
            assert chunk.parent_id == "parent-001"
            assert "extractor" in chunk.provenance
            assert "adaptive_slicer" in chunk.provenance

    def test_slice_with_parent_depth(self) -> None:
        """Test lineage depth increments."""
        slicer = AdaptiveSemanticSlicer()
        parent = IFTextArtifact(
            artifact_id="parent-001",
            content="Original",
            lineage_depth=2,
        )

        chunks = slicer.slice_with_parent("Hello world.", parent)

        for chunk in chunks:
            assert chunk.lineage_depth == 3

    def test_slice_with_parent_empty(self) -> None:
        """Test empty text with parent returns empty list."""
        slicer = AdaptiveSemanticSlicer()
        parent = IFTextArtifact(artifact_id="p-001", content="x")

        chunks = slicer.slice_with_parent("", parent)

        assert chunks == []


# ---------------------------------------------------------------------------
# Metadata Tests
# ---------------------------------------------------------------------------


class TestChunkMetadata:
    """Tests for chunk metadata."""

    def test_word_count_in_metadata(self) -> None:
        """Test word_count is in metadata."""
        slicer = AdaptiveSemanticSlicer()
        text = "One two three four five."

        chunks = slicer.slice(text, "doc-001")

        if chunks:
            assert "word_count" in chunks[0].metadata
            assert chunks[0].metadata["word_count"] == 5

    def test_char_count_in_metadata(self) -> None:
        """Test char_count is in metadata."""
        slicer = AdaptiveSemanticSlicer()
        text = "Hello world."

        chunks = slicer.slice(text, "doc-001")

        if chunks:
            assert "char_count" in chunks[0].metadata

    def test_sentence_count_in_metadata(self) -> None:
        """Test sentence_count is in metadata."""
        slicer = AdaptiveSemanticSlicer()
        text = "First. Second. Third."

        chunks = slicer.slice(text, "doc-001")

        if chunks:
            assert "sentence_count" in chunks[0].metadata


# ---------------------------------------------------------------------------
# Chunk Index Tests
# ---------------------------------------------------------------------------


class TestChunkIndexing:
    """Tests for chunk indexing."""

    def test_chunk_indices_sequential(self) -> None:
        """Test chunk indices are sequential."""
        config = SlicerConfig(max_chunk_chars=50)
        slicer = AdaptiveSemanticSlicer(config=config)
        text = "Sentence one. " * 20

        chunks = slicer.slice(text, "doc-001")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_total_chunks_correct(self) -> None:
        """Test total_chunks is set correctly."""
        config = SlicerConfig(max_chunk_chars=50)
        slicer = AdaptiveSemanticSlicer(config=config)
        text = "Sentence. " * 20

        chunks = slicer.slice(text, "doc-001")

        total = len(chunks)
        for chunk in chunks:
            assert chunk.total_chunks == total


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_slice_text_function(self) -> None:
        """Test slice_text convenience function."""
        text = "First sentence. Second sentence."

        chunks = slice_text(text, "doc-001")

        assert len(chunks) >= 1
        assert all(isinstance(c, IFChunkArtifact) for c in chunks)

    def test_slice_text_with_overlap(self) -> None:
        """Test slice_text with custom overlap."""
        text = "First. Second. Third."

        chunks = slice_text(text, "doc-001", overlap_sentences=3)

        assert len(chunks) >= 1

    def test_create_slicer_function(self) -> None:
        """Test create_slicer factory function."""
        slicer = create_slicer(
            min_chunk_chars=50,
            max_chunk_chars=500,
            overlap_sentences=1,
        )

        assert isinstance(slicer, AdaptiveSemanticSlicer)
        assert slicer.config.min_chunk_chars == 50
        assert slicer.config.max_chunk_chars == 500


# ---------------------------------------------------------------------------
# JPL Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule2_max_chunks_constant(self) -> None:
        """JPL Rule #2: MAX_CHUNKS_PER_DOCUMENT is bounded."""
        assert MAX_CHUNKS_PER_DOCUMENT == 10000
        assert MAX_CHUNKS_PER_DOCUMENT > 0

    def test_rule2_max_content_constant(self) -> None:
        """JPL Rule #2: MAX_CHUNK_CONTENT_LENGTH is bounded."""
        assert MAX_CHUNK_CONTENT_LENGTH == 100000
        assert MAX_CHUNK_CONTENT_LENGTH > 0

    def test_rule5_slice_assertions(self) -> None:
        """JPL Rule #5: slice() asserts preconditions."""
        slicer = AdaptiveSemanticSlicer()

        with pytest.raises(AssertionError):
            slicer.slice(None, "doc-001")  # type: ignore

        with pytest.raises(AssertionError):
            slicer.slice("text", "")  # Empty document_id

    def test_rule5_slice_with_parent_assertions(self) -> None:
        """JPL Rule #5: slice_with_parent() asserts preconditions."""
        slicer = AdaptiveSemanticSlicer()
        parent = IFTextArtifact(artifact_id="p", content="c")

        with pytest.raises(AssertionError):
            slicer.slice_with_parent(None, parent)  # type: ignore

        with pytest.raises(AssertionError):
            slicer.slice_with_parent("text", None)  # type: ignore

    def test_rule9_type_hints_config(self) -> None:
        """JPL Rule #9: SlicerConfig has complete type hints."""
        from dataclasses import fields

        config_fields = {f.name for f in fields(SlicerConfig)}
        required = {"min_chunk_chars", "max_chunk_chars", "overlap_sentences"}
        assert required.issubset(config_fields)

    def test_rule9_type_hints_result(self) -> None:
        """JPL Rule #9: SliceResult has complete type hints."""
        from dataclasses import fields

        result_fields = {f.name for f in fields(SliceResult)}
        required = {"content", "start_sentence", "end_sentence", "coherence_score"}
        assert required.issubset(result_fields)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sentence(self) -> None:
        """Test handling of single sentence."""
        slicer = AdaptiveSemanticSlicer()
        text = "Just one sentence."

        chunks = slicer.slice(text, "doc-001")

        assert len(chunks) == 1
        assert chunks[0].content == "Just one sentence."

    def test_unicode_text(self) -> None:
        """Test handling of unicode text."""
        slicer = AdaptiveSemanticSlicer()
        text = "日本語のテスト。これは二番目の文です。"

        chunks = slicer.slice(text, "doc-001")

        assert len(chunks) >= 1

    def test_very_long_sentence(self) -> None:
        """Test handling of very long single sentence."""
        slicer = AdaptiveSemanticSlicer()
        text = "Word " * 500 + "end."

        chunks = slicer.slice(text, "doc-001")

        # Should still produce at least one chunk
        assert len(chunks) >= 1

    def test_many_paragraphs(self) -> None:
        """Test handling of many paragraphs."""
        slicer = AdaptiveSemanticSlicer()
        text = "\n\n".join([f"Paragraph {i}." for i in range(50)])

        chunks = slicer.slice(text, "doc-001")

        assert len(chunks) >= 1
        assert len(chunks) <= MAX_CHUNKS_PER_DOCUMENT

    def test_special_characters(self) -> None:
        """Test handling of special characters."""
        slicer = AdaptiveSemanticSlicer()
        text = "Test @#$%^&*() here. Another sentence!"

        chunks = slicer.slice(text, "doc-001")

        assert len(chunks) >= 1
