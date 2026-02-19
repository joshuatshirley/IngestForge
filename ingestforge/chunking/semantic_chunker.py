"""Semantic chunking strategy.

Groups text by semantic similarity rather than fixed size.
Creates chunks that are semantically coherent.

Uses sentence embeddings (all-MiniLM-L6-v2) for true semantic similarity.
Falls back to Jaccard word similarity if embeddings unavailable.

e: Added chunk_to_artifacts() for IFChunkArtifact output."""

from __future__ import annotations

import hashlib
import math
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from ingestforge.core.pipeline.artifacts import IFChunkArtifact
    from ingestforge.core.pipeline.interfaces import IFArtifact


@dataclass
class ChunkRecord:
    """
    DEPRECATED: Use IFChunkArtifact instead.

    This class will be removed in a future major release (v2.0).

    Migration Guide:
        # Convert existing ChunkRecord to IFChunkArtifact
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact
        artifact = IFChunkArtifact.from_chunk_record(record)

        # Or use the factory
        from ingestforge.core.pipeline.artifact_factory import ArtifactFactory
        artifact = ArtifactFactory.chunk_from_record(record)

        # Convert back if needed (during transition)
        record = artifact.to_chunk_record()

    See EPIC-06 for full migration details.

    Original Description:
        Unified chunk representation for IngestForge.
        This is the standard chunk format used throughout the system.
        All chunkers should produce chunks in this format.
    """

    # i: Programmatic deprecation detection
    __deprecated__: ClassVar[bool] = True

    # Core identifiers
    chunk_id: str
    document_id: str

    # Content
    content: str

    # Metadata
    section_title: str = ""
    chunk_type: str = "content"
    source_file: str = ""
    word_count: int = 0
    char_count: int = 0
    section_hierarchy: Optional[List[str]] = None  # For legal/structured documents

    # Position information
    chunk_index: int = 0
    total_chunks: int = 1
    page_start: Optional[int] = None
    page_end: Optional[int] = None

    # Library/collection
    library: str = "default"

    # Source location (for citations)
    source_location: Optional[Any] = None

    # Timestamp
    ingested_at: Optional[str] = None

    # Read/Unread tracking (ORG-001)
    is_read: bool = False

    # Organization tags (ORG-002) - max 50 per chunk
    tags: List[str] = field(default_factory=list)

    # Author/Contributor identity (TICKET-301)
    # Tracks who contributed this content to the knowledge base
    author_id: Optional[str] = None
    author_name: Optional[str] = None

    # Enrichment fields
    embedding: Optional[List[float]] = None
    entities: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    quality_score: float = 0.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Multi-modal metadata (TICKET-V1.3)
    visual_description: Optional[str] = None

    # Unstructured-style spatial and element metadata
    # Bounding box coordinates (x1, y1) = top-left, (x2, y2) = bottom-right
    bbox: Optional[Tuple[int, int, int, int]] = None
    # HTML representation for tables
    table_html: Optional[str] = None
    # Document element type (Title, NarrativeText, ListItem, Table, etc.)
    element_type: str = "NarrativeText"

    def to_dict(self) -> Dict[str, Any]:
        """Convert ChunkRecord to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ChunkRecord":
        """Create ChunkRecord from dictionary."""
        # Filter to only known fields
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    def __post_init__(self) -> None:
        """
        Emit deprecation warning on ChunkRecord instantiation.

        c: Guide migration to IFChunkArtifact.
        Rule #7: No silent deprecation - always warn.
        """
        warnings.warn(
            "ChunkRecord is deprecated and will be removed in a future release. "
            "Use IFChunkArtifact instead. Convert existing records with "
            "IFChunkArtifact.from_chunk_record(record) or "
            "ArtifactFactory.chunk_from_record(record). "
            "See EPIC-06 for migration details.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class SemanticChunk:
    """
    DEPRECATED: Use IFChunkArtifact instead.

    Legacy semantic chunk format (for backward compatibility).
    Will be removed in v2.0.
    """

    # i: Programmatic deprecation detection
    __deprecated__: ClassVar[bool] = True

    text: str
    start_index: int
    end_index: int
    coherence_score: float
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """
        Emit deprecation warning on SemanticChunk instantiation.

        c: Guide migration to IFChunkArtifact.
        """
        warnings.warn(
            "SemanticChunk is deprecated and will be removed in a future release. "
            "Use IFChunkArtifact instead.",
            DeprecationWarning,
            stacklevel=2,
        )


@dataclass
class BoundaryScore:
    """Score for a potential chunk boundary."""

    index: int
    score: float  # Higher = stronger boundary
    similarity_drop: float
    is_paragraph_break: bool


class _Logger:
    """Lazy logger holder."""

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


class SemanticChunker:
    """Chunk text by semantic similarity using sentence embeddings.

    This chunker uses embedding-based similarity to find natural semantic
    boundaries in text. It supports:
    - True semantic boundary detection using embeddings
    - Fallback to Jaccard word similarity
    - Configurable overlap between chunks
    - Multiple embedding models
    """

    def __init__(
        self,
        max_chunk_size: Union[int, Any] = 1000,
        min_chunk_size: int = 100,
        similarity_threshold: float = 0.7,
        use_embeddings: bool = True,
        overlap_sentences: int = 1,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize semantic chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters, or Config object
            min_chunk_size: Minimum chunk size in characters
            similarity_threshold: Threshold for grouping similar sentences (0-1)
            use_embeddings: Use sentence embeddings (True) or word Jaccard (False)
            overlap_sentences: Number of sentences to overlap between chunks
            embedding_model: Embedding model name (default: all-MiniLM-L6-v2)
        """
        # Handle Config object (duck typing)
        if hasattr(max_chunk_size, "chunking"):
            config = max_chunk_size
            self.max_chunk_size = config.chunking.target_size * 4
            self.min_chunk_size = config.chunking.overlap * 2
            self.similarity_threshold = 0.7
            self.config = config
        else:
            self.max_chunk_size = max_chunk_size
            self.min_chunk_size = min_chunk_size
            self.similarity_threshold = similarity_threshold
            self.config = None

        self.use_embeddings = use_embeddings
        self.overlap_sentences = overlap_sentences
        self.embedding_model = embedding_model
        self._embedding_generator = None
        self._sentence_embeddings_cache: Dict[str, List[float]] = {}

    @property
    def embedding_generator(self) -> Any:
        """Lazy-load embedding generator."""
        if self._embedding_generator is not None:
            return self._embedding_generator

        if not self.use_embeddings:
            return None

        try:
            from ingestforge.enrichment.embeddings import EmbeddingGenerator
            from ingestforge.core.config import Config

            config = self.config if self.config else Config()
            self._embedding_generator = EmbeddingGenerator(config)
            return self._embedding_generator
        except ImportError as e:
            _Logger.get().debug(f"Embeddings unavailable, using Jaccard: {e}")
            self.use_embeddings = False
            return None

    def chunk(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ChunkRecord]:
        """Chunk text by semantic similarity.

        Args:
            text: Text to chunk
            document_id: Unique identifier for the source document
            source_file: Path or name of source file
            metadata: Optional metadata to attach to chunks

        Returns:
            List of ChunkRecord objects
        """
        if not text or len(text) < self.min_chunk_size:
            return self._create_single_chunk(text, document_id, source_file, metadata)

        sentences = self._split_into_sentences(text)
        if not sentences:
            return []

        embeddings = self._embed_sentences(sentences)
        boundaries = self._find_boundaries(sentences, embeddings)
        chunks = self._create_chunks(
            sentences, boundaries, text, document_id, source_file, metadata
        )

        return chunks

    def chunk_to_artifacts(
        self,
        text: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        parent: Optional["IFArtifact"] = None,
    ) -> List["IFChunkArtifact"]:
        """
        Chunk text and return IFChunkArtifact instances.

        e: Artifact-based chunking for pipeline migration.
        Rule #2: Bounded by chunk() which enforces limits.
        Rule #7: Explicit return type.
        Rule #9: Complete type hints.

        Args:
            text: Text to chunk.
            document_id: Unique identifier for the source document.
            source_file: Path or name of source file.
            metadata: Optional metadata to attach to chunks.
            parent: Optional parent artifact for lineage tracking.

        Returns:
            List of IFChunkArtifact instances with lineage.
        """
        from ingestforge.core.pipeline.artifacts import IFChunkArtifact

        # Use existing chunk method to get ChunkRecords
        chunk_records = self.chunk(text, document_id, source_file, metadata)

        # Convert each ChunkRecord to IFChunkArtifact
        artifacts: List["IFChunkArtifact"] = []
        for record in chunk_records:
            artifact = IFChunkArtifact.from_chunk_record(record, parent)
            artifacts.append(artifact)

        return artifacts

    def _create_single_chunk(
        self,
        text: str,
        document_id: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Create a single chunk for short or empty text.

        Returns a single chunk even for empty text, as the test expects
        empty input to produce one chunk with empty content.
        """
        return [
            self._create_chunk_record(
                text, 0, len(text), 1.0, document_id, source_file, 0, metadata
            )
        ]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences with improved boundary detection.

        Handles:
        - Standard punctuation (.!?)
        - Abbreviations (Mr., Dr., etc.)
        - Paragraph breaks
        - Quoted text
        """
        # Preserve paragraph structure
        paragraphs = text.split("\n\n")
        sentences = []

        for para in paragraphs:
            if not para.strip():
                continue

            # Split on sentence boundaries, handling abbreviations
            para_sentences = self._split_paragraph(para)
            sentences.extend(para_sentences)

            # Add paragraph marker for boundary detection
            if sentences and para_sentences:
                sentences[-1] = sentences[-1] + " [PARA]"

        # Clean up markers
        sentences = [s.replace(" [PARA]", "") for s in sentences]
        return [s.strip() for s in sentences if s.strip()]

    def _split_paragraph(self, paragraph: str) -> List[str]:
        """Split a paragraph into sentences.

        Uses a simple sentence boundary detection that handles common
        abbreviations without variable-width lookbehind (not supported in Python re).
        """
        # Protect abbreviations by replacing periods temporarily
        protected = paragraph
        abbreviations = [
            "Mr.",
            "Mrs.",
            "Ms.",
            "Dr.",
            "Prof.",
            "Sr.",
            "Jr.",
            "vs.",
            "etc.",
            "i.e.",
            "e.g.",
            "Inc.",
            "Ltd.",
            "Corp.",
        ]
        for abbr in abbreviations:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

        # Split on sentence-ending punctuation followed by space
        pattern = r"(?<=[.!?])\s+"
        parts = re.split(pattern, protected)

        # Restore abbreviations
        parts = [p.replace("<DOT>", ".") for p in parts]

        return [p.strip() for p in parts if p.strip()]

    def _embed_sentences(self, sentences: List[str]) -> List[List[float]]:
        """Generate embeddings for all sentences.

        Args:
            sentences: List of sentences to embed

        Returns:
            List of embedding vectors
        """
        if not self.use_embeddings or not self.embedding_generator:
            return self._word_vectors(sentences)

        try:
            embeddings = self.embedding_generator.embed_batch(sentences)
            return embeddings
        except Exception as e:
            _Logger.get().warning(f"Embedding failed, using Jaccard: {e}")
            self.use_embeddings = False
            return self._word_vectors(sentences)

    def _word_vectors(self, sentences: List[str]) -> List[List[str]]:
        """Create word-based vectors for fallback similarity."""
        vectors = []
        for sentence in sentences:
            words = sentence.lower().split()
            vectors.append(words)
        return vectors

    def _find_boundaries(
        self,
        sentences: List[str],
        embeddings: List[Any],
    ) -> List[int]:
        """Find optimal chunk boundaries using embedding similarity.

        Uses a sliding window approach to detect topic shifts by measuring
        similarity drops between adjacent sentence groups.

        Args:
            sentences: List of sentences
            embeddings: Embedding vectors for each sentence

        Returns:
            List of boundary indices (inclusive end of each chunk)
        """
        if len(sentences) <= 1:
            return [0]

        boundary_scores = self._calculate_boundary_scores(sentences, embeddings)
        boundaries = self._select_boundaries(sentences, boundary_scores)

        return boundaries

    def _calculate_boundary_scores(
        self,
        sentences: List[str],
        embeddings: List[Any],
    ) -> List[BoundaryScore]:
        """Calculate boundary strength scores for each position."""
        scores = []

        for i in range(1, len(sentences)):
            # Calculate similarity drop at this position
            sim_drop = self._calculate_similarity_drop(embeddings, i)

            # Check for paragraph break
            is_para = sentences[i - 1].endswith("[PARA]") if i > 0 else False

            # Combined score (higher = stronger boundary)
            score = sim_drop + (0.3 if is_para else 0.0)

            scores.append(
                BoundaryScore(
                    index=i,
                    score=score,
                    similarity_drop=sim_drop,
                    is_paragraph_break=is_para,
                )
            )

        return scores

    def _calculate_similarity_drop(self, embeddings: List[Any], position: int) -> float:
        """Calculate similarity drop at a given position.

        Compares similarity between sentences before and after the position.
        """
        if position == 0 or position >= len(embeddings):
            return 0.0

        # Compare previous sentence to current
        prev_emb = embeddings[position - 1]
        curr_emb = embeddings[position]

        # Direct similarity
        sim = self._calculate_similarity(prev_emb, curr_emb)

        # Convert to boundary score (1 - sim = dissimilarity)
        return 1.0 - sim

    def _select_boundaries(
        self,
        sentences: List[str],
        scores: List[BoundaryScore],
    ) -> List[int]:
        """Select optimal boundaries based on scores and size constraints."""
        if not scores:
            return [len(sentences) - 1]

        boundaries = []
        current_start = 0
        current_size = 0

        for i, sentence in enumerate(sentences):
            current_size += len(sentence)

            # Check if we should create a boundary
            should_split = self._should_create_boundary(
                i, current_size, scores, sentences
            )

            if should_split:
                boundaries.append(i)
                current_start = i + 1
                current_size = 0

        # Add final boundary if not already included
        if not boundaries or boundaries[-1] != len(sentences) - 1:
            boundaries.append(len(sentences) - 1)

        return boundaries

    def _should_create_boundary(
        self,
        position: int,
        current_size: int,
        scores: List[BoundaryScore],
        sentences: List[str],
    ) -> bool:
        """Determine if a boundary should be created at this position."""
        # Must split if at max size
        if current_size >= self.max_chunk_size:
            return True

        # Don't split if below min size
        if current_size < self.min_chunk_size:
            return False

        # Check boundary score at this position
        if position < len(scores):
            score = scores[position]
            threshold = 1.0 - self.similarity_threshold
            if score.score >= threshold:
                return True

        return False

    def _create_chunks(
        self,
        sentences: List[str],
        boundaries: List[int],
        original_text: str,
        document_id: str,
        source_file: str,
        metadata: Optional[Dict[str, Any]],
    ) -> List[ChunkRecord]:
        """Create ChunkRecord objects from sentences and boundaries."""
        chunks = []
        start_idx = 0

        for chunk_idx, end_idx in enumerate(boundaries):
            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx : end_idx + 1]

            # Add overlap from previous chunk
            if chunk_idx > 0 and self.overlap_sentences > 0:
                overlap_start = max(0, start_idx - self.overlap_sentences)
                overlap = sentences[overlap_start:start_idx]
                chunk_sentences = overlap + chunk_sentences

            chunk_text = " ".join(chunk_sentences)
            coherence = self._calculate_coherence(chunk_sentences)

            # Find position in original text
            text_start = (
                original_text.find(chunk_sentences[0]) if chunk_sentences else 0
            )
            text_end = text_start + len(chunk_text)

            chunk = self._create_chunk_record(
                chunk_text,
                text_start,
                text_end,
                coherence,
                document_id,
                source_file,
                chunk_idx,
                metadata,
            )
            chunks.append(chunk)

            start_idx = end_idx + 1

        # Update total_chunks for all chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _calculate_similarity(
        self,
        vec1: Union[List[float], List[str]],
        vec2: Union[List[float], List[str]],
    ) -> float:
        """Calculate similarity between two vectors.

        Uses cosine similarity for embeddings, Jaccard for word lists.
        """
        if not vec1 or not vec2:
            return 0.0

        # Check if embedding vectors
        if isinstance(vec1[0], (float, int)) and isinstance(vec2[0], (float, int)):
            return self._cosine_similarity(vec1, vec2)

        # Fallback: Jaccard similarity
        return self._jaccard_similarity(vec1, vec2)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between embedding vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _jaccard_similarity(self, vec1: List[str], vec2: List[str]) -> float:
        """Calculate Jaccard similarity between word lists."""
        set1 = set(vec1) if isinstance(vec1, list) else set()
        set2 = set(vec2) if isinstance(vec2, list) else set()

        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _calculate_coherence(self, sentences: List[str]) -> float:
        """Calculate coherence score for a group of sentences."""
        if len(sentences) <= 1:
            return 1.0

        all_words = set()
        word_counts: Dict[str, int] = {}

        for sentence in sentences:
            words = sentence.lower().split()
            all_words.update(words)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1

        shared = sum(1 for count in word_counts.values() if count > 1)
        total = len(all_words)

        return shared / total if total > 0 else 0.5

    def _create_chunk_record(
        self,
        text: str,
        start: int,
        end: int,
        coherence: float,
        document_id: str,
        source_file: str,
        chunk_index: int,
        metadata: Optional[Dict[str, Any]],
    ) -> ChunkRecord:
        """Create ChunkRecord object."""
        chunk_id = hashlib.md5(
            f"{document_id}_{chunk_index}_{text[:50]}".encode()
        ).hexdigest()[:16]

        chunk_meta = metadata.copy() if metadata else {}
        chunk_meta["coherence_score"] = coherence

        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            content=text,
            source_file=source_file,
            word_count=len(text.split()),
            char_count=len(text),
            chunk_index=chunk_index,
            quality_score=coherence,
            metadata=chunk_meta,
        )

    # ===========================================================================
    # Test compatibility methods
    # These methods support the test interface expectations
    # ===========================================================================

    def _detect_topic_boundary(
        self,
        vectors: List[Any],
        current_group: List[int],
        next_idx: int,
    ) -> bool:
        """Detect if there's a topic boundary at the given position.

        Tests expect this method to check if adding next_idx to current_group
        would cross a topic boundary based on coherence drop.

        Args:
            vectors: Embedding vectors for all sentences
            current_group: Indices of sentences in current group
            next_idx: Index of next sentence to potentially add

        Returns:
            True if there's a topic boundary, False otherwise
        """
        if not vectors or not current_group or next_idx >= len(vectors):
            return False

        # Check if vectors are embeddings (list of floats) or word lists
        if not isinstance(vectors[0][0], (float, int)):
            return False  # Requires embeddings

        # Calculate current group coherence
        current_coherence = self._calculate_embedding_coherence(vectors, current_group)

        # Calculate coherence with next sentence added
        extended_group = current_group + [next_idx]
        new_coherence = self._calculate_embedding_coherence(vectors, extended_group)

        # Boundary if coherence drops significantly
        return (current_coherence - new_coherence) > (1.0 - self.similarity_threshold)

    def _update_centroid(
        self,
        vectors: List[Any],
        group_indices: List[int],
    ) -> Any:
        """Calculate the centroid of a group of vectors.

        Args:
            vectors: All vectors (embeddings or word lists)
            group_indices: Indices of vectors in the group

        Returns:
            Centroid vector (average for embeddings, union for word lists)
        """
        if not group_indices:
            return []

        group_vectors = [vectors[i] for i in group_indices if i < len(vectors)]
        if not group_vectors:
            return []

        # Check if embeddings or word lists
        if isinstance(group_vectors[0][0], (float, int)):
            # Numeric embeddings - calculate average
            dim = len(group_vectors[0])
            centroid = [0.0] * dim
            for vec in group_vectors:
                for i in range(dim):
                    centroid[i] += vec[i]
            return [c / len(group_vectors) for c in centroid]
        else:
            # Word lists - return union
            all_words: set[str] = set()
            for vec in group_vectors:
                all_words.update(vec)
            return list(all_words)

    def _calculate_embedding_coherence(
        self,
        vectors: List[List[float]],
        group_indices: List[int],
    ) -> float:
        """Calculate coherence score for a group of embedding vectors.

        Coherence is the average pairwise cosine similarity within the group.

        Args:
            vectors: Embedding vectors
            group_indices: Indices of vectors in the group

        Returns:
            Coherence score between 0.0 and 1.0
        """
        if len(group_indices) <= 1:
            return 1.0

        group_vectors = [vectors[i] for i in group_indices if i < len(vectors)]
        if len(group_vectors) <= 1:
            return 1.0

        # Calculate average pairwise similarity
        total_sim = 0.0
        count = 0
        for i in range(len(group_vectors)):
            for j in range(i + 1, len(group_vectors)):
                total_sim += self._cosine_similarity(group_vectors[i], group_vectors[j])
                count += 1

        return total_sim / count if count > 0 else 1.0

    def _group_by_similarity(
        self,
        sentences: List[str],
        vectors: List[Any],
    ) -> List[List[int]]:
        """Group sentences by similarity using centroid comparison.

        Args:
            sentences: List of sentences
            vectors: Embedding vectors for each sentence

        Returns:
            List of groups, where each group is a list of sentence indices
        """
        if not sentences:
            return []

        groups: List[List[int]] = []
        current_group: List[int] = [0]

        for i in range(1, len(sentences)):
            # Check if we should add to current group
            if self._detect_topic_boundary(vectors, current_group, i):
                groups.append(current_group)
                current_group = [i]
            else:
                current_group.append(i)

        # Add final group
        if current_group:
            groups.append(current_group)

        return groups


def chunk_text(
    text: str,
    max_size: int = 1000,
    min_size: int = 100,
    threshold: float = 0.7,
    document_id: str = "doc",
) -> List[ChunkRecord]:
    """Chunk text by semantic similarity.

    Convenience function for quick chunking.

    Args:
        text: Text to chunk
        max_size: Maximum chunk size in characters
        min_size: Minimum chunk size in characters
        threshold: Similarity threshold (0-1)
        document_id: Document identifier

    Returns:
        List of ChunkRecord objects
    """
    chunker = SemanticChunker(max_size, min_size, threshold)
    return chunker.chunk(text, document_id)
