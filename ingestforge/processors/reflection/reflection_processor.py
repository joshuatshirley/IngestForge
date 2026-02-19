"""
Reflection Processor for IngestForge (IF).

Agentic Reflection Loop - Critic pass that reviews extraction accuracy.
Follows NASA JPL Power of Ten rules.

Features:
- Identifies factual contradictions between LLM output and source text
- Triggers re-extraction if confidence < 0.7
- Max 2 reflection passes per document (JPL Rule #2)
"""

from dataclasses import dataclass, field
from typing import Any, List
import uuid

from ingestforge.core.logging import get_logger
from ingestforge.core.pipeline.interfaces import IFArtifact, IFProcessor
from ingestforge.core.pipeline.artifacts import IFTextArtifact

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_REFLECTION_PASSES = 2  # Maximum reflection iterations per document
CONFIDENCE_THRESHOLD = 0.7  # Re-extraction trigger threshold
MAX_CLAIMS_TO_VERIFY = 20  # Maximum claims to verify per pass
MAX_SOURCE_CHUNKS = 10  # Maximum source chunks to compare against


# =============================================================================
# RESULT TYPES
# =============================================================================


@dataclass
class ContradictionResult:
    """
    Result of contradiction check for a single claim.

    Rule #9: Complete type hints.
    """

    claim_text: str
    source_text: str
    entailment_score: float
    is_contradiction: bool
    reasoning: str


@dataclass
class ReflectionResult:
    """
    Aggregate result from reflection analysis.

    Rule #9: Complete type hints.
    """

    pass_number: int
    total_claims: int
    verified_claims: int
    contradicted_claims: int
    average_confidence: float
    should_reextract: bool
    contradictions: List[ContradictionResult] = field(default_factory=list)
    reasoning: str = ""

    @property
    def confidence(self) -> float:
        """Overall confidence score."""
        return self.average_confidence


# =============================================================================
# REFLECTION ARTIFACT
# =============================================================================


class IFReflectionArtifact(IFTextArtifact):
    """
    Artifact containing reflection analysis results.

    Stores reflection pass data and reasoning in provenance.
    Rule #9: Complete type hints.
    """

    reflection_pass: int = 0
    confidence_score: float = 1.0
    contradictions_found: int = 0
    should_reextract: bool = False
    reflection_reasoning: str = ""

    def derive(self, processor_id: str, **kwargs: Any) -> "IFReflectionArtifact":
        """Create a derived reflection artifact."""
        new_provenance = list(self.provenance) + [processor_id]
        return IFReflectionArtifact(
            artifact_id=kwargs.get("artifact_id", str(uuid.uuid4())),
            content=kwargs.get("content", self.content),
            metadata={**self.metadata, **kwargs.get("metadata", {})},
            provenance=new_provenance,
            parent_id=self.artifact_id,
            root_artifact_id=self.effective_root_id,
            lineage_depth=self.lineage_depth + 1,
            reflection_pass=kwargs.get("reflection_pass", self.reflection_pass),
            confidence_score=kwargs.get("confidence_score", self.confidence_score),
            contradictions_found=kwargs.get(
                "contradictions_found", self.contradictions_found
            ),
            should_reextract=kwargs.get("should_reextract", self.should_reextract),
            reflection_reasoning=kwargs.get(
                "reflection_reasoning", self.reflection_reasoning
            ),
        )


# =============================================================================
# REFLECTION PROCESSOR
# =============================================================================


class IFReflectionProcessor(IFProcessor):
    """
    Processor that implements the "Critic" reflection pass.

    Agentic Reflection Loop.

    Features:
    - Extracts claims from LLM output
    - Verifies claims against source text using entailment
    - Identifies contradictions
    - Triggers re-extraction if confidence < threshold

    JPL Rule #2: Max 2 reflection passes enforced.
    JPL Rule #4: All methods < 60 lines.
    JPL Rule #9: Complete type hints.
    """

    def __init__(
        self,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        max_passes: int = MAX_REFLECTION_PASSES,
    ) -> None:
        """
        Initialize the reflection processor.

        Args:
            confidence_threshold: Threshold below which re-extraction triggers.
            max_passes: Maximum reflection passes (capped at MAX_REFLECTION_PASSES).
        """
        # JPL Rule #2: Bounded parameters
        assert 0.0 <= confidence_threshold <= 1.0, "Threshold must be 0.0-1.0"
        self._threshold = confidence_threshold
        self._max_passes = min(max_passes, MAX_REFLECTION_PASSES)

        # Lazy-loaded components
        self._claim_extractor = None
        self._entailment_scorer = None

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "if-reflection-processor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        """Functional capabilities."""
        return ["reflection", "fact-checking", "contradiction-detection"]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement."""
        return 512  # Cross-encoder model requires ~500MB

    def is_available(self) -> bool:
        """Check if dependencies are available."""
        try:
            from ingestforge.agent.critic.claim_extractor import ClaimExtractor
            from ingestforge.agent.critic.entailment import EntailmentScorer

            return True
        except ImportError:
            return False

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Execute reflection pass on artifact.

        AC: Identifies contradictions, triggers re-extraction if needed.
        Rule #4: < 60 lines.
        Rule #7: Check return values.

        Args:
            artifact: Artifact containing extracted content to verify.

        Returns:
            IFReflectionArtifact with analysis results.
        """
        assert artifact is not None, "artifact cannot be None"

        # Get current pass number from metadata
        current_pass = artifact.metadata.get("reflection_pass", 0) + 1

        # JPL Rule #2: Enforce max passes
        if current_pass > self._max_passes:
            logger.info(f"Max reflection passes ({self._max_passes}) reached")
            return self._create_result_artifact(
                artifact,
                current_pass,
                ReflectionResult(
                    pass_number=current_pass,
                    total_claims=0,
                    verified_claims=0,
                    contradicted_claims=0,
                    average_confidence=1.0,
                    should_reextract=False,
                    reasoning=f"Max passes ({self._max_passes}) reached",
                ),
            )

        # Extract content for analysis
        content = self._extract_content(artifact)
        source_text = artifact.metadata.get("source_text", "")

        if not content or not source_text:
            return self._create_skip_artifact(
                artifact, current_pass, "Missing content or source"
            )

        # Perform reflection analysis
        result = self._analyze_claims(content, source_text, current_pass)

        # Store reasoning in provenance (instruction #2)
        return self._create_result_artifact(artifact, current_pass, result)

    def _extract_content(self, artifact: IFArtifact) -> str:
        """Extract text content from artifact."""
        if isinstance(artifact, IFTextArtifact):
            return artifact.content
        return artifact.metadata.get("content", "")

    def _analyze_claims(
        self, content: str, source_text: str, pass_number: int
    ) -> ReflectionResult:
        """
        Analyze claims against source text.

        AC: Identifies factual contradictions.
        Rule #4: < 60 lines.

        Args:
            content: LLM-generated content to verify.
            source_text: Original source text for comparison.
            pass_number: Current reflection pass number.

        Returns:
            ReflectionResult with analysis.
        """
        # Lazy-load components
        extractor = self._get_claim_extractor()
        scorer = self._get_entailment_scorer()

        # Extract claims (bounded by MAX_CLAIMS_TO_VERIFY)
        claims = extractor.extract(content)[:MAX_CLAIMS_TO_VERIFY]

        if not claims:
            return ReflectionResult(
                pass_number=pass_number,
                total_claims=0,
                verified_claims=0,
                contradicted_claims=0,
                average_confidence=1.0,
                should_reextract=False,
                reasoning="No claims to verify",
            )

        # Score each claim against source
        contradictions: List[ContradictionResult] = []
        scores: List[float] = []

        for claim in claims:
            score = scorer.score(claim.text, source_text)
            scores.append(score)

            is_contradiction = score < self._threshold
            if is_contradiction:
                contradictions.append(
                    ContradictionResult(
                        claim_text=claim.text,
                        source_text=source_text[:200],
                        entailment_score=score,
                        is_contradiction=True,
                        reasoning=f"Score {score:.2f} < threshold {self._threshold}",
                    )
                )

        # Calculate aggregate metrics
        avg_confidence = sum(scores) / len(scores) if scores else 1.0
        verified = sum(1 for s in scores if s >= self._threshold)
        should_reextract = avg_confidence < self._threshold

        reasoning = self._build_reasoning(
            pass_number,
            len(claims),
            verified,
            len(contradictions),
            avg_confidence,
            should_reextract,
        )

        return ReflectionResult(
            pass_number=pass_number,
            total_claims=len(claims),
            verified_claims=verified,
            contradicted_claims=len(contradictions),
            average_confidence=avg_confidence,
            should_reextract=should_reextract,
            contradictions=contradictions,
            reasoning=reasoning,
        )

    def _build_reasoning(
        self,
        pass_num: int,
        total: int,
        verified: int,
        contradicted: int,
        confidence: float,
        reextract: bool,
    ) -> str:
        """Build human-readable reasoning string."""
        status = "REEXTRACT RECOMMENDED" if reextract else "PASSED"
        return (
            f"Reflection Pass {pass_num}: {status}. "
            f"Claims: {total} total, {verified} verified, {contradicted} contradicted. "
            f"Confidence: {confidence:.2%}"
        )

    def _create_result_artifact(
        self, source: IFArtifact, pass_number: int, result: ReflectionResult
    ) -> IFReflectionArtifact:
        """Create result artifact with reflection data."""
        content = self._extract_content(source)

        return IFReflectionArtifact(
            artifact_id=str(uuid.uuid4()),
            content=content,
            metadata={
                **source.metadata,
                "reflection_pass": pass_number,
                "reflection_result": {
                    "total_claims": result.total_claims,
                    "verified_claims": result.verified_claims,
                    "contradicted_claims": result.contradicted_claims,
                    "average_confidence": result.average_confidence,
                },
            },
            provenance=list(source.provenance) + [self.processor_id],
            parent_id=source.artifact_id,
            root_artifact_id=source.effective_root_id
            if hasattr(source, "effective_root_id")
            else source.artifact_id,
            lineage_depth=source.lineage_depth + 1
            if hasattr(source, "lineage_depth")
            else 1,
            reflection_pass=pass_number,
            confidence_score=result.average_confidence,
            contradictions_found=result.contradicted_claims,
            should_reextract=result.should_reextract,
            reflection_reasoning=result.reasoning,
        )

    def _create_skip_artifact(
        self, source: IFArtifact, pass_number: int, reason: str
    ) -> IFReflectionArtifact:
        """Create artifact when reflection is skipped."""
        return self._create_result_artifact(
            source,
            pass_number,
            ReflectionResult(
                pass_number=pass_number,
                total_claims=0,
                verified_claims=0,
                contradicted_claims=0,
                average_confidence=1.0,
                should_reextract=False,
                reasoning=f"Skipped: {reason}",
            ),
        )

    def _get_claim_extractor(self):
        """Lazy-load claim extractor."""
        if self._claim_extractor is None:
            from ingestforge.agent.critic.claim_extractor import ClaimExtractor

            self._claim_extractor = ClaimExtractor()
        return self._claim_extractor

    def _get_entailment_scorer(self):
        """Lazy-load entailment scorer."""
        if self._entailment_scorer is None:
            from ingestforge.agent.critic.entailment import EntailmentScorer

            self._entailment_scorer = EntailmentScorer()
        return self._entailment_scorer

    def teardown(self) -> bool:
        """Clean up resources."""
        self._claim_extractor = None
        self._entailment_scorer = None
        return True
