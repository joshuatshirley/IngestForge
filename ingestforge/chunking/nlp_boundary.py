"""NLP-Driven Boundary Detector for semantic chunking.

NLPBoundaryDetector implementation.
Content hash verification after splitting.
Follows NASA JPL Power of Ten rules.

Identifies optimal chunk boundaries using NLP techniques:
- Sentence integrity (no mid-sentence cuts)
- Semantic cohesion via embedding similarity
- Configurable context overlap
- Content hash verification (JPL Rule #10)
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from ingestforge.core.errors import SafeErrorMessage

if TYPE_CHECKING:
    pass

# JPL Rule #2: Fixed upper bounds
MAX_SENTENCES = 10000
MAX_BOUNDARY_CANDIDATES = 1000
MAX_TEXT_LENGTH = 10_000_000  # 10MB
DEFAULT_SIMILARITY_THRESHOLD = 0.8
DEFAULT_OVERLAP_SENTENCES = 2


class BoundaryReason(Enum):
    """Reason for boundary creation.

    Boundary classification.
    Rule #9: Complete type hints via Enum.
    """

    SIMILARITY_DROP = "similarity_drop"
    PARAGRAPH_BREAK = "paragraph_break"
    SIZE_LIMIT = "size_limit"
    COMBINED = "combined"


@dataclass
class Sentence:
    """A sentence with position metadata.

    GWT-1: Sentence integrity support.
    Rule #9: Complete type hints.
    """

    text: str
    start_char: int
    end_char: int
    index: int
    is_paragraph_end: bool = False

    @property
    def char_count(self) -> int:
        """Character count of sentence."""
        return len(self.text)

    @property
    def word_count(self) -> int:
        """Word count of sentence."""
        return len(self.text.split())


@dataclass
class BoundaryCandidate:
    """Candidate boundary with scores and metadata.

    GWT-2: Semantic boundary detection.
    Rule #9: Complete type hints.
    """

    position: int  # Index in sentence list (boundary AFTER this sentence)
    score: float  # Combined boundary score (higher = stronger)
    similarity_drop: float  # Embedding similarity drop at boundary
    is_paragraph_break: bool  # True if paragraph break at this position
    reason: BoundaryReason
    char_position: int = 0  # Character position in original text

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "position": self.position,
            "score": self.score,
            "similarity_drop": self.similarity_drop,
            "is_paragraph_break": self.is_paragraph_break,
            "reason": self.reason.value,
            "char_position": self.char_position,
        }


@dataclass
class BoundaryConstraints:
    """Constraints for boundary selection.

    GWT-5: Size-bounded boundaries.
    Rule #9: Complete type hints.
    """

    min_chunk_chars: int = 100
    max_chunk_chars: int = 2000
    min_chunk_sentences: int = 1
    max_chunk_sentences: int = 50
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES


class SentenceSegmenter:
    """NLP-aware sentence segmentation.

    GWT-1: Sentence integrity.
    Rule #9: Complete type hints.
    """

    # Common abbreviations that don't end sentences
    ABBREVIATIONS: List[str] = [
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
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Sept.",
        "Oct.",
        "Nov.",
        "Dec.",
        "Ave.",
        "Blvd.",
        "St.",
        "Rd.",
        "Mt.",
        "Ft.",
        "Ph.D.",
        "M.D.",
        "B.A.",
        "M.A.",
        "B.S.",
        "M.S.",
    ]

    def __init__(self) -> None:
        """Initialize segmenter."""
        self._abbr_pattern = self._build_abbr_pattern()

    def _build_abbr_pattern(self) -> str:
        """Build regex pattern for abbreviations."""
        escaped = [re.escape(a) for a in self.ABBREVIATIONS]
        return "|".join(escaped)

    def segment(self, text: str) -> List[Sentence]:
        """Segment text into sentences with position metadata.

        GWT-1: Sentence integrity.
        Rule #2: MAX_SENTENCES bound.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            text: Text to segment.

        Returns:
            List of Sentence objects with positions.
        """
        assert text is not None, "text cannot be None"
        assert len(text) <= MAX_TEXT_LENGTH, f"text exceeds {MAX_TEXT_LENGTH} chars"

        if not text.strip():
            return []

        sentences: List[Sentence] = []
        paragraphs = self._split_paragraphs(text)

        current_char = 0
        sentence_index = 0

        for para_idx, paragraph in enumerate(paragraphs):
            if not paragraph.strip():
                current_char += len(paragraph) + 2  # +2 for \n\n
                continue

            # Find paragraph start in original text
            para_start = text.find(paragraph, current_char)
            if para_start == -1:
                para_start = current_char

            para_sentences = self._segment_paragraph(paragraph)

            for i, sent_text in enumerate(para_sentences):
                if sentence_index >= MAX_SENTENCES:
                    break

                # Find sentence position
                sent_start = text.find(sent_text, para_start)
                if sent_start == -1:
                    sent_start = para_start

                sent_end = sent_start + len(sent_text)
                is_last = i == len(para_sentences) - 1

                sentences.append(
                    Sentence(
                        text=sent_text,
                        start_char=sent_start,
                        end_char=sent_end,
                        index=sentence_index,
                        is_paragraph_end=is_last,
                    )
                )
                sentence_index += 1
                para_start = sent_end

            current_char = para_start

        return sentences[:MAX_SENTENCES]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        return text.split("\n\n")

    def _segment_paragraph(self, paragraph: str) -> List[str]:
        """Segment a paragraph into sentences.

        Rule #4: Function < 60 lines.
        """
        # Protect abbreviations
        protected = paragraph
        for abbr in self.ABBREVIATIONS:
            protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

        # Split on sentence-ending punctuation
        pattern = r"(?<=[.!?])\s+"
        parts = re.split(pattern, protected)

        # Restore abbreviations and clean
        sentences = []
        for part in parts:
            restored = part.replace("<DOT>", ".")
            cleaned = restored.strip()
            if cleaned:
                sentences.append(cleaned)

        return sentences


class NLPBoundaryDetector:
    """Detects optimal chunk boundaries using NLP techniques.

    GWT-1: Sentence integrity.
    GWT-2: Semantic boundary detection.
    GWT-3: Paragraph-aware boundaries.
    GWT-4: Configurable overlap.
    GWT-5: Size-bounded boundaries.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
        paragraph_boost: float = 0.3,
    ) -> None:
        """Initialize boundary detector.

        Args:
            similarity_threshold: Threshold for similarity drop detection.
            overlap_sentences: Sentences to overlap between chunks.
            paragraph_boost: Score boost for paragraph breaks.
        """
        assert 0.0 <= similarity_threshold <= 1.0, "threshold must be 0-1"
        assert overlap_sentences >= 0, "overlap must be non-negative"

        self._similarity_threshold = similarity_threshold
        self._overlap_sentences = overlap_sentences
        self._paragraph_boost = paragraph_boost
        self._segmenter = SentenceSegmenter()
        self._embedding_generator: Any = None

    @property
    def similarity_threshold(self) -> float:
        """Current similarity threshold."""
        return self._similarity_threshold

    @property
    def overlap_sentences(self) -> int:
        """Current overlap setting."""
        return self._overlap_sentences

    def detect_boundaries(
        self,
        text: str,
        embeddings: Optional[List[List[float]]] = None,
    ) -> Tuple[List[Sentence], List[BoundaryCandidate]]:
        """Detect potential chunk boundaries in text.

        GWT-2: Semantic boundary detection.
        Rule #2: MAX_BOUNDARY_CANDIDATES bound.
        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            text: Text to analyze.
            embeddings: Optional pre-computed sentence embeddings.

        Returns:
            Tuple of (sentences, boundary_candidates).
        """
        assert text is not None, "text cannot be None"

        # Segment into sentences
        sentences = self._segmenter.segment(text)
        if len(sentences) <= 1:
            return sentences, []

        # Get or compute embeddings
        if embeddings is None:
            embeddings = self._get_embeddings(sentences)

        # Calculate boundary scores
        candidates = self._calculate_candidates(sentences, embeddings)

        return sentences, candidates[:MAX_BOUNDARY_CANDIDATES]

    def select_boundaries(
        self,
        sentences: List[Sentence],
        candidates: List[BoundaryCandidate],
        constraints: Optional[BoundaryConstraints] = None,
    ) -> List[int]:
        """Select optimal boundaries respecting constraints.

        GWT-5: Size-bounded boundaries.
        Rule #4: Function < 60 lines.

        Args:
            sentences: List of sentences.
            candidates: Candidate boundaries with scores.
            constraints: Size and count constraints.

        Returns:
            List of selected boundary positions (sentence indices).
        """
        if not sentences:
            return []

        if constraints is None:
            constraints = BoundaryConstraints()

        # Build candidate lookup
        candidate_map = {c.position: c for c in candidates}

        selected: List[int] = []
        current_start = 0
        current_chars = 0

        for i, sentence in enumerate(sentences):
            current_chars += sentence.char_count

            should_split = self._should_split(
                i, current_chars, current_start, candidate_map, constraints
            )

            if should_split:
                selected.append(i)
                current_start = i + 1
                current_chars = 0

        # Ensure final boundary
        if not selected or selected[-1] != len(sentences) - 1:
            selected.append(len(sentences) - 1)

        return selected

    def _get_embeddings(self, sentences: List[Sentence]) -> List[List[float]]:
        """Get embeddings for sentences.

        Rule #7: Handle embedding failures gracefully.
        """
        if self._embedding_generator is None:
            try:
                from ingestforge.enrichment.embeddings import EmbeddingGenerator
                from ingestforge.core.config import Config

                self._embedding_generator = EmbeddingGenerator(Config())
            except ImportError:
                return self._fallback_vectors(sentences)

        try:
            texts = [s.text for s in sentences]
            return self._embedding_generator.embed_batch(texts)
        except Exception:
            return self._fallback_vectors(sentences)

    def _fallback_vectors(self, sentences: List[Sentence]) -> List[List[float]]:
        """Create fallback word-frequency vectors."""
        vectors: List[List[float]] = []
        all_words: set[str] = set()

        # Collect vocabulary
        for sentence in sentences:
            words = sentence.text.lower().split()
            all_words.update(words)

        word_list = sorted(all_words)
        word_index = {w: i for i, w in enumerate(word_list)}

        # Create vectors
        for sentence in sentences:
            vec = [0.0] * len(word_list)
            words = sentence.text.lower().split()
            for word in words:
                if word in word_index:
                    vec[word_index[word]] = 1.0
            vectors.append(vec)

        return vectors

    def _calculate_candidates(
        self,
        sentences: List[Sentence],
        embeddings: List[List[float]],
    ) -> List[BoundaryCandidate]:
        """Calculate boundary candidates with scores.

        GWT-2, GWT-3: Similarity drop and paragraph awareness.
        Rule #4: Function < 60 lines.
        """
        candidates: List[BoundaryCandidate] = []

        for i in range(len(sentences) - 1):
            # Calculate similarity drop
            sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
            sim_drop = 1.0 - sim

            # Check paragraph break
            is_para = sentences[i].is_paragraph_end

            # Calculate combined score
            score = sim_drop
            if is_para:
                score += self._paragraph_boost

            # Determine reason
            if is_para and sim_drop >= (1.0 - self._similarity_threshold):
                reason = BoundaryReason.COMBINED
            elif is_para:
                reason = BoundaryReason.PARAGRAPH_BREAK
            elif sim_drop >= (1.0 - self._similarity_threshold):
                reason = BoundaryReason.SIMILARITY_DROP
            else:
                reason = BoundaryReason.SIMILARITY_DROP

            candidates.append(
                BoundaryCandidate(
                    position=i,
                    score=score,
                    similarity_drop=sim_drop,
                    is_paragraph_break=is_para,
                    reason=reason,
                    char_position=sentences[i].end_char,
                )
            )

        return candidates

    def _should_split(
        self,
        position: int,
        current_chars: int,
        current_start: int,
        candidate_map: dict[int, BoundaryCandidate],
        constraints: BoundaryConstraints,
    ) -> bool:
        """Determine if a split should occur at this position.

        Rule #4: Function < 60 lines.
        """
        # Must split if exceeding max size
        if current_chars >= constraints.max_chunk_chars:
            return True

        sentence_count = position - current_start + 1
        if sentence_count >= constraints.max_chunk_sentences:
            return True

        # Don't split if below minimum
        if current_chars < constraints.min_chunk_chars:
            return False
        if sentence_count < constraints.min_chunk_sentences:
            return False

        # Check if this is a good boundary
        if position in candidate_map:
            candidate = candidate_map[position]
            threshold = 1.0 - constraints.similarity_threshold
            if candidate.score >= threshold:
                return True

        return False

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between vectors.

        Rule #4: Function < 60 lines.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0.0 or magnitude2 == 0.0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def detect_boundaries(
    text: str,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> Tuple[List[Sentence], List[BoundaryCandidate]]:
    """Convenience function to detect boundaries.

    Args:
        text: Text to analyze.
        similarity_threshold: Similarity threshold for boundary detection.

    Returns:
        Tuple of (sentences, boundary_candidates).
    """
    detector = NLPBoundaryDetector(similarity_threshold=similarity_threshold)
    return detector.detect_boundaries(text)


def segment_sentences(text: str) -> List[Sentence]:
    """Convenience function to segment text into sentences.

    Args:
        text: Text to segment.

    Returns:
        List of Sentence objects.
    """
    segmenter = SentenceSegmenter()
    return segmenter.segment(text)


# ---------------------------------------------------------------------------
# Content Hash Verification (JPL Rule #10)
# ---------------------------------------------------------------------------


def calculate_content_hash(text: str) -> str:
    """Calculate SHA-256 hash of text content.

    Content hash for verification.
    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.

    Args:
        text: Text to hash.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    assert text is not None, "text cannot be None"
    normalized = text.strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def reconstruct_from_sentences(sentences: List[Sentence]) -> str:
    """Reconstruct original text from sentence objects.

    Text reconstruction for verification.
    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.

    Args:
        sentences: List of Sentence objects.

    Returns:
        Reconstructed text string.
    """
    assert sentences is not None, "sentences cannot be None"

    if not sentences:
        return ""

    # Sort by index to ensure correct order
    sorted_sents = sorted(sentences, key=lambda s: s.index)

    parts: List[str] = []
    for i, sent in enumerate(sorted_sents):
        parts.append(sent.text)
        # Add paragraph break if this sentence ends a paragraph
        if sent.is_paragraph_end and i < len(sorted_sents) - 1:
            parts.append("\n\n")
        elif i < len(sorted_sents) - 1:
            parts.append(" ")

    return "".join(parts)


@dataclass
class SplitVerification:
    """Result of split verification.

    Verification result dataclass.
    Rule #9: Complete type hints.
    """

    is_valid: bool
    original_hash: str
    reconstructed_hash: str
    sentence_count: int
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "original_hash": self.original_hash,
            "reconstructed_hash": self.reconstructed_hash,
            "sentence_count": self.sentence_count,
            "error_message": self.error_message,
        }


def verify_split_integrity(
    original_text: str,
    sentences: List[Sentence],
) -> SplitVerification:
    """Verify that sentences can reconstruct the original text.

    AC: JPL Rule #10 - Verify content hash after splitting.
    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.
    Rule #7: Check all return values.

    Args:
        original_text: Original text before splitting.
        sentences: Sentences produced by splitting.

    Returns:
        SplitVerification with validation result.
    """
    assert original_text is not None, "original_text cannot be None"
    assert sentences is not None, "sentences cannot be None"

    original_hash = calculate_content_hash(original_text)

    try:
        reconstructed = reconstruct_from_sentences(sentences)
        reconstructed_hash = calculate_content_hash(reconstructed)

        is_valid = original_hash == reconstructed_hash

        return SplitVerification(
            is_valid=is_valid,
            original_hash=original_hash,
            reconstructed_hash=reconstructed_hash,
            sentence_count=len(sentences),
            error_message=None if is_valid else "Hash mismatch after reconstruction",
        )
    except Exception as e:
        return SplitVerification(
            is_valid=False,
            original_hash=original_hash,
            reconstructed_hash="",
            sentence_count=len(sentences),
            # SEC-002: Sanitize error message
            error_message=SafeErrorMessage.sanitize(e, "reconstruction_failed", logger),
        )


def segment_and_verify(text: str) -> Tuple[List[Sentence], SplitVerification]:
    """Segment text and verify split integrity.

    Combined segmentation with verification.
    Rule #4: Function < 60 lines.

    Args:
        text: Text to segment.

    Returns:
        Tuple of (sentences, verification_result).
    """
    segmenter = SentenceSegmenter()
    sentences = segmenter.segment(text)
    verification = verify_split_integrity(text, sentences)
    return sentences, verification
