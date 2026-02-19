"""Contradiction detection engine for fact-checking.

Detects semantic contradictions between text passages using
embedding similarity and negation pattern matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Protocol, Set, Tuple

import numpy as np
import numpy.typing as npt


class EmbeddingModel(Protocol):
    """Protocol for embedding models with encode method."""

    def encode(
        self, sentences: List[str], convert_to_numpy: bool = True
    ) -> npt.NDArray[np.float32]:
        """Encode sentences to embeddings."""
        ...


# Max claims to compare to prevent unbounded computation
MAX_CLAIMS_TO_COMPARE = 1000
MAX_CLAIM_LENGTH = 5000

# Negation patterns for contradiction detection
NEGATION_WORDS: Set[str] = {
    "not",
    "no",
    "never",
    "none",
    "nothing",
    "neither",
    "nor",
    "cannot",
    "can not",
    "won not",
    "wouldn not",
    "shouldn not",
    "couldn not",
    "isn not",
    "aren not",
    "wasn not",
    "weren not",
    "hasn not",
    "haven not",
    "hadn not",
    "doesn not",
    "don not",
    "didn not",
    "without",
    "lacks",
    "absent",
    "missing",
    "false",
    "incorrect",
    "untrue",
    "deny",
    "denies",
    "denied",
    "refuse",
    "refuses",
    "refused",
    "opposite",
}

# Antonym pairs for contradiction detection
ANTONYM_PAIRS: Set[Tuple[str, str]] = {
    ("hot", "cold"),
    ("true", "false"),
    ("yes", "no"),
    ("increase", "decrease"),
    ("rise", "fall"),
    ("up", "down"),
    ("more", "less"),
    ("greater", "fewer"),
    ("higher", "lower"),
    ("gain", "loss"),
    ("success", "failure"),
    ("win", "lose"),
    ("start", "stop"),
    ("begin", "end"),
    ("open", "close"),
    ("alive", "dead"),
    ("present", "absent"),
    ("before", "after"),
}


@dataclass
class ContradictionResult:
    """Result of contradiction detection between two claims.

    Attributes:
        claim1: First claim text.
        claim2: Second claim text.
        score: Contradiction score (0.0 to 1.0).
        explanation: Human-readable explanation.
        negation_detected: Whether negation patterns were found.
        antonym_detected: Whether antonym pairs were found.
    """

    claim1: str
    claim2: str
    score: float
    explanation: str
    negation_detected: bool = False
    antonym_detected: bool = False


@dataclass
class ContradictionPair:
    """Pair of contradicting claims from corpus.

    Attributes:
        index1: Index of first claim.
        index2: Index of second claim.
        score: Contradiction score.
        claim1: First claim text.
        claim2: Second claim text.
    """

    index1: int
    index2: int
    score: float
    claim1: str
    claim2: str


class ContradictionDetector:
    """Detect contradictions between text passages.

    Uses semantic similarity (via embeddings) combined with
    negation pattern matching to identify contradicting claims.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        negation_boost: float = 0.3,
        embedding_model: Optional[str] = None,
    ) -> None:
        """Initialize contradiction detector.

        Args:
            similarity_threshold: Minimum similarity for related claims.
            negation_boost: Score boost when negation detected.
            embedding_model: Optional embedding model name.

        Raises:
            ValueError: If thresholds are out of valid range.
        """
        # Validate inputs (Rule #7)
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold must be 0.0-1.0, got {similarity_threshold}"
            )
        if not 0.0 <= negation_boost <= 1.0:
            raise ValueError(f"negation_boost must be 0.0-1.0, got {negation_boost}")

        self.similarity_threshold = similarity_threshold
        self.negation_boost = negation_boost
        self.embedding_model_name = embedding_model or "all-MiniLM-L6-v2"
        self._model: Optional[EmbeddingModel] = None

    @property
    def model(self) -> EmbeddingModel:
        """Lazy-load embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required for contradiction detection. "
                    "Install with: pip install sentence-transformers"
                ) from e

        assert self._model is not None, "Model should be loaded"
        return self._model

    def detect_contradiction(
        self,
        claim1: str,
        claim2: str,
    ) -> ContradictionResult:
        """Detect contradiction between two claims.

        Args:
            claim1: First claim text.
            claim2: Second claim text.

        Returns:
            ContradictionResult with score and explanation.

        Raises:
            ValueError: If claims are empty or too long.
        """
        # Validate inputs (Rule #7)
        self._validate_claims(claim1, claim2)

        # Compute semantic similarity
        similarity = self._compute_similarity(claim1, claim2)

        # Check if claims are semantically related
        if similarity < self.similarity_threshold:
            return ContradictionResult(
                claim1=claim1,
                claim2=claim2,
                score=0.0,
                explanation="Claims are not semantically related.",
            )

        # Analyze contradiction signals
        return self._analyze_contradiction(claim1, claim2, similarity)

    def _validate_claims(self, claim1: str, claim2: str) -> None:
        """Validate claim inputs.

        Args:
            claim1: First claim.
            claim2: Second claim.

        Raises:
            ValueError: If claims are invalid.
        """
        if not claim1 or not claim1.strip():
            raise ValueError("claim1 cannot be empty")
        if not claim2 or not claim2.strip():
            raise ValueError("claim2 cannot be empty")
        if len(claim1) > MAX_CLAIM_LENGTH:
            raise ValueError(
                f"claim1 too long: {len(claim1)} chars (max {MAX_CLAIM_LENGTH})"
            )
        if len(claim2) > MAX_CLAIM_LENGTH:
            raise ValueError(
                f"claim2 too long: {len(claim2)} chars (max {MAX_CLAIM_LENGTH})"
            )

    def _analyze_contradiction(
        self,
        claim1: str,
        claim2: str,
        similarity: float,
    ) -> ContradictionResult:
        """Analyze contradiction signals and generate result.

        Args:
            claim1: First claim.
            claim2: Second claim.
            similarity: Semantic similarity score.

        Returns:
            ContradictionResult with analysis.
        """
        # Check for negation patterns
        negation_score = self._check_negation(claim1, claim2)

        # Check for antonym pairs
        antonym_score = self._check_antonyms(claim1, claim2)

        # Combine scores
        final_score = self._combine_scores(
            similarity,
            negation_score,
            antonym_score,
        )

        # Generate explanation
        explanation = self._generate_explanation(
            similarity,
            negation_score > 0,
            antonym_score > 0,
        )

        return ContradictionResult(
            claim1=claim1,
            claim2=claim2,
            score=final_score,
            explanation=explanation,
            negation_detected=negation_score > 0,
            antonym_detected=antonym_score > 0,
        )

    def find_contradictions_in_corpus(
        self,
        claims: List[str],
        min_score: float = 0.5,
    ) -> List[ContradictionPair]:
        """Find all contradicting pairs in a corpus of claims.

        Args:
            claims: List of claim texts.
            min_score: Minimum contradiction score to include.

        Returns:
            List of ContradictionPair objects.

        Raises:
            ValueError: If claims list is too large or contains invalid items.
        """
        # Validate inputs (Rule #7)
        if not claims:
            return []

        if len(claims) > MAX_CLAIMS_TO_COMPARE:
            raise ValueError(
                f"Too many claims: {len(claims)} (max {MAX_CLAIMS_TO_COMPARE})"
            )

        # Validate each claim
        for i, claim in enumerate(claims):
            if not claim or not claim.strip():
                raise ValueError(f"Claim at index {i} is empty")
            if len(claim) > MAX_CLAIM_LENGTH:
                raise ValueError(f"Claim at index {i} too long: {len(claim)} chars")

        contradictions: List[ContradictionPair] = []

        # Fixed upper bound (Rule #2)
        num_claims = len(claims)
        for i in range(num_claims):
            for j in range(i + 1, num_claims):
                result = self.detect_contradiction(claims[i], claims[j])

                if result.score >= min_score:
                    contradictions.append(
                        ContradictionPair(
                            index1=i,
                            index2=j,
                            score=result.score,
                            claim1=claims[i],
                            claim2=claims[j],
                        )
                    )

        # Sort by score descending
        contradictions.sort(key=lambda x: x.score, reverse=True)
        return contradictions

    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims from text.

        Splits text on sentence boundaries.

        Args:
            text: Input text.

        Returns:
            List of claim strings.
        """
        # Simple sentence splitting on common delimiters
        sentences = re.split(r"[.!?]+", text)

        # Filter empty and whitespace-only sentences
        claims = [s.strip() for s in sentences if s.strip()]

        return claims

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text1: First text.
            text2: Second text.

        Returns:
            Similarity score (0.0 to 1.0).
        """
        embeddings = self.model.encode([text1, text2], convert_to_numpy=True)

        # Compute cosine similarity
        from numpy import dot
        from numpy.linalg import norm

        similarity = float(
            dot(embeddings[0], embeddings[1])
            / (norm(embeddings[0]) * norm(embeddings[1]))
        )

        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))

    def _check_negation(self, claim1: str, claim2: str) -> float:
        """Check for negation patterns between claims.

        Args:
            claim1: First claim.
            claim2: Second claim.

        Returns:
            Negation score (0.0 to 1.0).
        """
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())

        # Count negation words in each claim
        neg1 = len(words1 & NEGATION_WORDS)
        neg2 = len(words2 & NEGATION_WORDS)

        # If one has negation and other doesn't, likely contradiction
        if (neg1 > 0) != (neg2 > 0):
            return self.negation_boost

        return 0.0

    def _check_antonyms(self, claim1: str, claim2: str) -> float:
        """Check for antonym pairs between claims.

        Args:
            claim1: First claim.
            claim2: Second claim.

        Returns:
            Antonym score (0.0 to 1.0).
        """
        words1 = set(claim1.lower().split())
        words2 = set(claim2.lower().split())

        # Check for antonym pairs
        for word1, word2 in ANTONYM_PAIRS:
            if word1 in words1 and word2 in words2:
                return 0.2
            if word2 in words1 and word1 in words2:
                return 0.2

        return 0.0

    def _combine_scores(
        self,
        similarity: float,
        negation: float,
        antonym: float,
    ) -> float:
        """Combine similarity, negation, and antonym scores.

        Args:
            similarity: Semantic similarity.
            negation: Negation score.
            antonym: Antonym score.

        Returns:
            Combined contradiction score (0.0 to 1.0).
        """
        # High similarity + negation/antonym = strong contradiction
        base_score = similarity * (negation + antonym)

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_score))

    def _generate_explanation(
        self,
        similarity: float,
        has_negation: bool,
        has_antonym: bool,
    ) -> str:
        """Generate human-readable explanation.

        Args:
            similarity: Semantic similarity score.
            has_negation: Whether negation was detected.
            has_antonym: Whether antonyms were detected.

        Returns:
            Explanation string.
        """
        parts: List[str] = [f"Claims are {similarity:.1%} similar."]

        if has_negation:
            parts.append("Negation patterns detected.")

        if has_antonym:
            parts.append("Antonym pairs found.")

        if has_negation or has_antonym:
            parts.append("Likely contradiction.")

        return " ".join(parts)
