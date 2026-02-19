"""Evidence linking system for fact-checking.

Connects claims with supporting/refuting evidence from the knowledge base
using semantic similarity and contradiction detection."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from ingestforge.enrichment.contradiction import ContradictionDetector
from ingestforge.storage.base import ChunkRepository, SearchResult

# Max evidence items to prevent unbounded computation (Rule #2)
MAX_EVIDENCE_ITEMS = 100
MAX_CLAIM_LENGTH = 5000
MAX_TOP_K = 50

# Confidence thresholds
STRONG_SUPPORT_THRESHOLD = 0.8
MODERATE_SUPPORT_THRESHOLD = 0.6
WEAK_SUPPORT_THRESHOLD = 0.4


class SupportType(Enum):
    """Classification of how evidence relates to a claim."""

    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


@dataclass
class LinkedEvidence:
    """Evidence linked to a claim.

    Attributes:
        evidence_text: The evidence content.
        source: Source identifier (document ID or URL).
        chunk_id: Unique identifier for the evidence chunk.
        relevance_score: Similarity score (0.0 to 1.0).
        support_type: How evidence relates to claim.
        confidence: Confidence in the classification (0.0 to 1.0).
        metadata: Additional metadata from the chunk.
    """

    evidence_text: str
    source: str
    chunk_id: str
    relevance_score: float
    support_type: SupportType
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "evidence_text": self.evidence_text,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "relevance_score": self.relevance_score,
            "support_type": self.support_type.value,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceLinkResult:
    """Result of evidence linking for a claim.

    Attributes:
        claim: The original claim text.
        linked_evidence: List of linked evidence items.
        total_support: Count of supporting evidence.
        total_refute: Count of refuting evidence.
        total_neutral: Count of neutral evidence.
        key_entities: Entities extracted from the claim.
    """

    claim: str
    linked_evidence: List[LinkedEvidence]
    total_support: int = 0
    total_refute: int = 0
    total_neutral: int = 0
    key_entities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "claim": self.claim,
            "linked_evidence": [e.to_dict() for e in self.linked_evidence],
            "total_support": self.total_support,
            "total_refute": self.total_refute,
            "total_neutral": self.total_neutral,
            "key_entities": self.key_entities,
        }


class EvidenceLinker:
    """Link claims to supporting/refuting evidence from knowledge base.

    Uses semantic search combined with contradiction detection
    to classify evidence as supporting, refuting, or neutral.
    """

    def __init__(
        self,
        contradiction_detector: Optional[ContradictionDetector] = None,
        support_threshold: float = MODERATE_SUPPORT_THRESHOLD,
        refute_threshold: float = MODERATE_SUPPORT_THRESHOLD,
    ) -> None:
        """Initialize evidence linker.

        Args:
            contradiction_detector: Detector for contradictions.
            support_threshold: Minimum similarity for support.
            refute_threshold: Minimum contradiction score for refutation.

        Raises:
            ValueError: If thresholds are out of valid range.
        """
        # Validate inputs (Rule #7)
        if not 0.0 <= support_threshold <= 1.0:
            raise ValueError(
                f"support_threshold must be 0.0-1.0, got {support_threshold}"
            )
        if not 0.0 <= refute_threshold <= 1.0:
            raise ValueError(
                f"refute_threshold must be 0.0-1.0, got {refute_threshold}"
            )

        self.contradiction_detector = contradiction_detector or ContradictionDetector()
        self.support_threshold = support_threshold
        self.refute_threshold = refute_threshold

    def link_evidence(
        self,
        claim: str,
        storage: ChunkRepository,
        top_k: int = 10,
        library_filter: Optional[str] = None,
    ) -> EvidenceLinkResult:
        """Link evidence to a claim from the knowledge base.

        Args:
            claim: The claim to find evidence for.
            storage: Repository to search for evidence.
            top_k: Number of evidence items to retrieve.
            library_filter: Optional library to filter results.

        Returns:
            EvidenceLinkResult with linked evidence.

        Raises:
            ValueError: If inputs are invalid.
        """
        # Validate inputs (Rule #7)
        self._validate_inputs(claim, top_k)

        # Extract key entities for better search
        entities = self._extract_key_entities(claim)

        # Search for relevant chunks
        search_results = storage.search(
            query=claim,
            top_k=min(top_k, MAX_TOP_K),
            library_filter=library_filter,
        )

        # Classify and score each piece of evidence
        linked_evidence = self._classify_evidence(claim, search_results)

        # Count evidence by type
        counts = self._count_by_type(linked_evidence)

        return EvidenceLinkResult(
            claim=claim,
            linked_evidence=linked_evidence,
            total_support=counts["support"],
            total_refute=counts["refute"],
            total_neutral=counts["neutral"],
            key_entities=entities,
        )

    def classify_support(
        self,
        claim: str,
        evidence: str,
    ) -> SupportType:
        """Classify if evidence supports, refutes, or is neutral to claim.

        Args:
            claim: The claim being evaluated.
            evidence: The evidence text.

        Returns:
            SupportType classification.

        Raises:
            ValueError: If inputs are empty.
        """
        # Validate inputs (Rule #7)
        if not claim or not claim.strip():
            raise ValueError("claim cannot be empty")
        if not evidence or not evidence.strip():
            raise ValueError("evidence cannot be empty")

        # First calculate similarity
        similarity = self._calculate_relevance(claim, evidence)

        # Check for contradiction
        result = self.contradiction_detector.detect_contradiction(
            claim,
            evidence,
        )

        # High contradiction score = refutation
        # (even if similarity is high, contradiction takes precedence)
        if result.score >= self.refute_threshold:
            return SupportType.REFUTES

        # High similarity without contradiction = support
        if similarity >= self.support_threshold:
            return SupportType.SUPPORTS

        return SupportType.NEUTRAL

    def _validate_inputs(self, claim: str, top_k: int) -> None:
        """Validate input parameters.

        Args:
            claim: Claim text to validate.
            top_k: Top k parameter to validate.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not claim or not claim.strip():
            raise ValueError("claim cannot be empty")

        if len(claim) > MAX_CLAIM_LENGTH:
            raise ValueError(
                f"claim too long: {len(claim)} chars (max {MAX_CLAIM_LENGTH})"
            )

        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")

        if top_k > MAX_EVIDENCE_ITEMS:
            raise ValueError(
                f"top_k exceeds maximum: {top_k} (max {MAX_EVIDENCE_ITEMS})"
            )

    def _classify_evidence(
        self,
        claim: str,
        search_results: List[SearchResult],
    ) -> List[LinkedEvidence]:
        """Classify search results as supporting/refuting evidence.

        Args:
            claim: The claim being evaluated.
            search_results: Search results from storage.

        Returns:
            List of LinkedEvidence with classifications.
        """
        linked_evidence: List[LinkedEvidence] = []

        # Fixed upper bound (Rule #2)
        num_results = min(len(search_results), MAX_EVIDENCE_ITEMS)

        for i in range(num_results):
            result = search_results[i]

            # Classify support type
            support_type = self.classify_support(claim, result.content)

            # Calculate confidence based on relevance score
            confidence = self._calculate_confidence(
                result.score,
                support_type,
            )

            linked_evidence.append(
                LinkedEvidence(
                    evidence_text=result.content,
                    source=result.document_id,
                    chunk_id=result.chunk_id,
                    relevance_score=result.score,
                    support_type=support_type,
                    confidence=confidence,
                    metadata={
                        "source_file": result.source_file,
                        "section_title": result.section_title,
                        "word_count": result.word_count,
                    },
                )
            )

        return linked_evidence

    def _calculate_relevance(self, claim: str, evidence: str) -> float:
        """Calculate semantic relevance between claim and evidence.

        Args:
            claim: The claim text.
            evidence: The evidence text.

        Returns:
            Relevance score (0.0 to 1.0).
        """
        # Use the contradiction detector's similarity computation
        # but without the negation/antonym checks
        similarity = self.contradiction_detector._compute_similarity(
            claim,
            evidence,
        )

        return similarity

    def _calculate_confidence(
        self,
        relevance_score: float,
        support_type: SupportType,
    ) -> float:
        """Calculate confidence in the support type classification.

        Args:
            relevance_score: Relevance score from search.
            support_type: Classification type.

        Returns:
            Confidence score (0.0 to 1.0).
        """
        # High relevance = high confidence for support/refute
        if support_type == SupportType.NEUTRAL:
            # Low confidence for neutral classifications
            return max(0.0, min(0.5, relevance_score * 0.5))

        # For support/refute, confidence scales with relevance
        base_confidence = relevance_score

        # Clamp to [0, 1]
        return max(0.0, min(1.0, base_confidence))

    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text for search enhancement.

        Uses simple capitalization heuristics. More sophisticated
        NER can be added via the NERExtractor module.

        Args:
            text: Input text.

        Returns:
            List of potential entity strings.
        """
        # Find capitalized words (potential entities)
        # Pattern matches capitalized words (single or multi-word)
        # Matches "Einstein" or "Albert Einstein" anywhere in text
        pattern = r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"

        matches = re.findall(pattern, text)

        # Filter out sentence-start words by checking context
        # Words immediately after '. ' or at start of text are likely not entities
        entities_with_context: List[str] = []

        # Use re.finditer to get positions
        for match in re.finditer(pattern, text):
            entity = match.group(1)
            start_pos = match.start()

            # Skip if at text start or after sentence end
            if start_pos == 0 or (
                start_pos >= 2 and text[start_pos - 2 : start_pos] == ". "
            ):
                # Check if it's a common sentence starter
                if entity in [
                    "The",
                    "A",
                    "An",
                    "This",
                    "That",
                    "These",
                    "Those",
                    "I",
                    "We",
                    "He",
                    "She",
                    "It",
                    "They",
                ]:
                    continue

            entities_with_context.append(entity)

        # Deduplicate while preserving order
        seen: Set[str] = set()
        entities: List[str] = []

        # Fixed upper bound (Rule #2)
        max_entities = 20
        for entity in entities_with_context[:max_entities]:
            if entity not in seen:
                seen.add(entity)
                entities.append(entity)

        return entities

    def _count_by_type(
        self,
        evidence: List[LinkedEvidence],
    ) -> Dict[str, int]:
        """Count evidence items by support type.

        Args:
            evidence: List of linked evidence.

        Returns:
            Dict with counts for each type.
        """
        counts = {
            "support": 0,
            "refute": 0,
            "neutral": 0,
        }

        # Fixed upper bound (Rule #2)
        num_evidence = len(evidence)
        for i in range(num_evidence):
            item = evidence[i]
            if item.support_type == SupportType.SUPPORTS:
                counts["support"] += 1
            elif item.support_type == SupportType.REFUTES:
                counts["refute"] += 1
            else:
                counts["neutral"] += 1

        return counts
