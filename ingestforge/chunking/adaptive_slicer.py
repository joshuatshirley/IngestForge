"""Adaptive Semantic Slicer for context-aware chunking.

Adaptive Semantic Slicing implementation.
Follows NASA JPL Power of Ten rules.

Uses NLPBoundaryDetector () to create context-aware chunks
with configurable overlap and semantic cohesion.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestforge.core.pipeline.interfaces import IFArtifact

from ingestforge.chunking.nlp_boundary import (
    BoundaryConstraints,
    NLPBoundaryDetector,
    Sentence,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_OVERLAP_SENTENCES,
)

# JPL Rule #2: Fixed upper bounds
MAX_CHUNKS_PER_DOCUMENT = 10000
MAX_CHUNK_CONTENT_LENGTH = 100000
DEFAULT_MIN_CHUNK_CHARS = 100
DEFAULT_MAX_CHUNK_CHARS = 2000


@dataclass
class SlicerConfig:
    """Configuration for AdaptiveSemanticSlicer.

    Configurable slicing parameters.
    Rule #9: Complete type hints.
    """

    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    min_chunk_sentences: int = 1
    max_chunk_sentences: int = 50

    def to_constraints(self) -> BoundaryConstraints:
        """Convert to BoundaryConstraints for detector."""
        return BoundaryConstraints(
            min_chunk_chars=self.min_chunk_chars,
            max_chunk_chars=self.max_chunk_chars,
            min_chunk_sentences=self.min_chunk_sentences,
            max_chunk_sentences=self.max_chunk_sentences,
            similarity_threshold=self.similarity_threshold,
            overlap_sentences=self.overlap_sentences,
        )


@dataclass
class SliceResult:
    """Result of a slicing operation.

    GWT-4: Includes coherence scoring.
    Rule #9: Complete type hints.
    """

    content: str
    start_sentence: int
    end_sentence: int
    coherence_score: float
    char_start: int
    char_end: int
    has_overlap: bool

    @property
    def sentence_count(self) -> int:
        """Number of sentences in this slice."""
        return self.end_sentence - self.start_sentence + 1

    @property
    def char_count(self) -> int:
        """Character count of content."""
        return len(self.content)


class AdaptiveSemanticSlicer:
    """Creates context-aware chunks using NLP boundary detection.

    GWT-1: Semantic chunk creation.
    GWT-2: Overlap application.
    GWT-3: Size constraint enforcement.
    GWT-4: Coherence scoring.
    GWT-5: Zero mid-sentence cuts.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        config: Optional[SlicerConfig] = None,
        detector: Optional[NLPBoundaryDetector] = None,
    ) -> None:
        """Initialize the slicer.

        Args:
            config: Slicing configuration.
            detector: Optional pre-configured boundary detector.
        """
        self._config = config or SlicerConfig()
        self._detector = detector or NLPBoundaryDetector(
            similarity_threshold=self._config.similarity_threshold,
            overlap_sentences=self._config.overlap_sentences,
        )
        self._processor_id = "adaptive_slicer"

    @property
    def config(self) -> SlicerConfig:
        """Current configuration."""
        return self._config

    @property
    def processor_id(self) -> str:
        """Processor identifier."""
        return self._processor_id

    def slice(
        self,
        text: str,
        document_id: str,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List["IFChunkArtifact"]:
        """Slice text into semantic chunks.

        GWT-1: Semantic chunk creation.
        Rule #2: MAX_CHUNKS_PER_DOCUMENT bound.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            text: Text to slice.
            document_id: Document identifier.
            embeddings: Optional pre-computed sentence embeddings.

        Returns:
            List of IFChunkArtifact instances.
        """
        assert text is not None, "text cannot be None"
        assert document_id, "document_id cannot be empty"

        if not text.strip():
            return []

        # Detect boundaries
        sentences, candidates = self._detector.detect_boundaries(text, embeddings)
        if not sentences:
            return []

        # Select boundaries
        constraints = self._config.to_constraints()
        boundaries = self._detector.select_boundaries(
            sentences, candidates, constraints
        )

        # Create slices with overlap
        slices = self._create_slices(sentences, boundaries)

        # Convert to artifacts
        artifacts = self._slices_to_artifacts(slices, sentences, document_id, None)

        return artifacts[:MAX_CHUNKS_PER_DOCUMENT]

    def slice_with_parent(
        self,
        text: str,
        parent: "IFArtifact",
        embeddings: Optional[List[List[float]]] = None,
    ) -> List["IFChunkArtifact"]:
        """Slice text with parent artifact for lineage tracking.

        GWT-1: Semantic chunk creation with lineage.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            text: Text to slice.
            parent: Parent artifact for lineage.
            embeddings: Optional pre-computed embeddings.

        Returns:
            List of IFChunkArtifact instances with lineage.
        """
        assert text is not None, "text cannot be None"
        assert parent is not None, "parent cannot be None"

        if not text.strip():
            return []

        # Extract document_id from parent
        document_id = getattr(parent, "document_id", parent.artifact_id)

        # Detect and select boundaries
        sentences, candidates = self._detector.detect_boundaries(text, embeddings)
        if not sentences:
            return []

        constraints = self._config.to_constraints()
        boundaries = self._detector.select_boundaries(
            sentences, candidates, constraints
        )

        # Create slices with overlap
        slices = self._create_slices(sentences, boundaries)

        # Convert to artifacts with lineage
        artifacts = self._slices_to_artifacts(slices, sentences, document_id, parent)

        return artifacts[:MAX_CHUNKS_PER_DOCUMENT]

    def _create_slices(
        self,
        sentences: List[Sentence],
        boundaries: List[int],
    ) -> List[SliceResult]:
        """Create slice results from sentences and boundaries.

        GWT-2: Overlap application.
        GWT-4: Coherence scoring.
        Rule #4: Function < 60 lines.
        """
        if not sentences or not boundaries:
            return []

        slices: List[SliceResult] = []
        overlap = self._config.overlap_sentences

        start_idx = 0
        for i, boundary in enumerate(boundaries):
            end_idx = boundary

            # Calculate overlap start for non-first chunks
            overlap_start = start_idx
            has_overlap = False
            if i > 0 and overlap > 0:
                overlap_start = max(0, start_idx - overlap)
                has_overlap = overlap_start < start_idx

            # Get sentences for this slice
            slice_sentences = sentences[overlap_start : end_idx + 1]
            content = " ".join(s.text for s in slice_sentences)

            # Calculate coherence
            coherence = self._calculate_coherence(slice_sentences)

            # Get character positions
            char_start = slice_sentences[0].start_char if slice_sentences else 0
            char_end = slice_sentences[-1].end_char if slice_sentences else 0

            slices.append(
                SliceResult(
                    content=content[:MAX_CHUNK_CONTENT_LENGTH],
                    start_sentence=overlap_start,
                    end_sentence=end_idx,
                    coherence_score=coherence,
                    char_start=char_start,
                    char_end=char_end,
                    has_overlap=has_overlap,
                )
            )

            start_idx = end_idx + 1

        return slices

    def _calculate_coherence(self, sentences: List[Sentence]) -> float:
        """Calculate coherence score for sentences.

        GWT-4: Coherence scoring.
        Rule #4: Function < 60 lines.
        """
        if len(sentences) <= 1:
            return 1.0

        # Word overlap coherence
        all_words: set[str] = set()
        word_counts: Dict[str, int] = {}

        for sentence in sentences:
            words = sentence.text.lower().split()
            all_words.update(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        # Count words appearing in multiple sentences
        shared = sum(1 for count in word_counts.values() if count > 1)
        total = len(all_words)

        return shared / total if total > 0 else 0.5

    def _slices_to_artifacts(
        self,
        slices: List[SliceResult],
        sentences: List[Sentence],
        document_id: str,
        parent: Optional["IFArtifact"],
    ) -> List["IFChunkArtifact"]:
        """Convert slices to IFChunkArtifact instances.

        GWT-5: Zero mid-sentence cuts (verified by construction).
        Rule #4: Function < 60 lines.
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        artifacts: List[IFChunkArtifact] = []
        total_chunks = len(slices)

        for i, slice_result in enumerate(slices):
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(document_id, i, slice_result.content)

            # Build metadata
            metadata: Dict[str, Any] = {
                "coherence_score": slice_result.coherence_score,
                "sentence_count": slice_result.sentence_count,
                "char_start": slice_result.char_start,
                "char_end": slice_result.char_end,
                "has_overlap": slice_result.has_overlap,
                "word_count": len(slice_result.content.split()),
                "char_count": slice_result.char_count,
            }

            # Build lineage from parent
            if parent:
                artifact = IFChunkArtifact(
                    artifact_id=chunk_id,
                    document_id=document_id,
                    content=slice_result.content,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    parent_id=parent.artifact_id,
                    root_artifact_id=parent.effective_root_id,
                    lineage_depth=parent.lineage_depth + 1,
                    provenance=list(parent.provenance) + [self._processor_id],
                    metadata=metadata,
                )
            else:
                artifact = IFChunkArtifact(
                    artifact_id=chunk_id,
                    document_id=document_id,
                    content=slice_result.content,
                    chunk_index=i,
                    total_chunks=total_chunks,
                    provenance=[self._processor_id],
                    metadata=metadata,
                )

            artifacts.append(artifact)

        return artifacts

    def _generate_chunk_id(
        self, document_id: str, chunk_index: int, content: str
    ) -> str:
        """Generate unique chunk ID.

        Rule #4: Function < 60 lines.
        """
        hash_input = f"{document_id}_{chunk_index}_{content[:50]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def slice_text(
    text: str,
    document_id: str,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> List["IFChunkArtifact"]:
    """Convenience function to slice text into semantic chunks.

    Args:
        text: Text to slice.
        document_id: Document identifier.
        overlap_sentences: Number of sentences to overlap.
        similarity_threshold: Similarity threshold for boundaries.

    Returns:
        List of IFChunkArtifact instances.
    """
    config = SlicerConfig(
        overlap_sentences=overlap_sentences,
        similarity_threshold=similarity_threshold,
    )
    slicer = AdaptiveSemanticSlicer(config=config)
    return slicer.slice(text, document_id)


def create_slicer(
    min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
) -> AdaptiveSemanticSlicer:
    """Factory function to create a configured slicer.

    Args:
        min_chunk_chars: Minimum chunk size.
        max_chunk_chars: Maximum chunk size.
        overlap_sentences: Sentence overlap.

    Returns:
        Configured AdaptiveSemanticSlicer.
    """
    config = SlicerConfig(
        min_chunk_chars=min_chunk_chars,
        max_chunk_chars=max_chunk_chars,
        overlap_sentences=overlap_sentences,
    )
    return AdaptiveSemanticSlicer(config=config)
