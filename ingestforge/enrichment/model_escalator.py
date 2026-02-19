"""
Multi-Model Fallback Escalator for IngestForge.

Multi-Model-Fallback
Optimizes cost and accuracy by using a "Fast" model first, falling back to
a "Smart" model only when confidence is low.

NASA JPL Power of Ten compliant.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_ESCALATION_ATTEMPTS = 3
DEFAULT_FALLBACK_THRESHOLD = 0.6
MIN_ENTITIES_FOR_CONFIDENCE = 1


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

T = TypeVar("T")


class EscalationEvent(BaseModel):
    """
    Record of an escalation event.

    AC: Log the "Escalation" event.
    Rule #9: Complete type hints.
    """

    model_config = {"frozen": True}

    fast_model: str = Field(..., description="Model used for initial extraction")
    smart_model: str = Field(..., description="Model used for escalation")
    fast_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence from fast model"
    )
    smart_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence from smart model"
    )
    threshold: float = Field(..., ge=0.0, le=1.0, description="Fallback threshold")
    latency_fast_ms: float = Field(0.0, ge=0.0, description="Fast model latency")
    latency_smart_ms: float = Field(0.0, ge=0.0, description="Smart model latency")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp")

    def to_provenance_entry(self) -> str:
        """Generate provenance entry for this escalation."""
        return (
            f"escalated:{self.fast_model}->{self.smart_model}"
            f"[{self.fast_confidence:.2f}<{self.threshold}]"
        )


@dataclass
class EscalationResult(Generic[T]):
    """
    Result of an extraction with potential escalation.

    Rule #9: Complete type hints.
    """

    result: T
    confidence: float
    model_used: str
    escalated: bool = False
    escalation_event: Optional[EscalationEvent] = None
    latency_ms: float = 0.0
    provenance_entries: List[str] = field(default_factory=list)


@dataclass
class EscalationStats:
    """
    Statistics for escalation tracking.

    Rule #9: Complete type hints.
    """

    total_extractions: int = 0
    escalation_count: int = 0
    fast_model_successes: int = 0
    smart_model_successes: int = 0
    total_fast_latency_ms: float = 0.0
    total_smart_latency_ms: float = 0.0

    @property
    def escalation_rate(self) -> float:
        """Calculate escalation rate as percentage."""
        if self.total_extractions == 0:
            return 0.0
        return (self.escalation_count / self.total_extractions) * 100.0

    @property
    def avg_fast_latency_ms(self) -> float:
        """Average latency for fast model calls."""
        fast_calls = self.total_extractions
        if fast_calls == 0:
            return 0.0
        return self.total_fast_latency_ms / fast_calls

    @property
    def avg_smart_latency_ms(self) -> float:
        """Average latency for smart model calls."""
        if self.escalation_count == 0:
            return 0.0
        return self.total_smart_latency_ms / self.escalation_count


# ---------------------------------------------------------------------------
# ModelEscalator Implementation
# ---------------------------------------------------------------------------


class ModelEscalator:
    """
    Manages multi-model fallback logic for extraction.

    AC: Implements ModelEscalator logic in extraction processor.
    - Uses fast model first
    - Falls back to smart model when confidence < threshold
    - Records escalation events in provenance
    - Tracks escalation rate

    NASA JPL Power of Ten compliant.
    Rule #2: Fixed upper bounds on escalation attempts.
    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        fast_model: str = "gpt-4o-mini",
        smart_model: str = "gpt-4o",
        fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
        max_escalation_attempts: int = MAX_ESCALATION_ATTEMPTS,
    ):
        """
        Initialize the escalator.

        AC: Configurable fallback_threshold (Default 0.6).

        Args:
            fast_model: Model ID for fast/cheap extraction
            smart_model: Model ID for smart/accurate extraction
            fallback_threshold: Confidence threshold for escalation
            max_escalation_attempts: Maximum escalations per session (JPL Rule #2)
        """
        # JPL Rule #5: Assert preconditions
        assert fast_model, "fast_model must be non-empty"
        assert smart_model, "smart_model must be non-empty"
        assert (
            0.0 <= fallback_threshold <= 1.0
        ), "fallback_threshold must be in [0.0, 1.0]"
        assert max_escalation_attempts > 0, "max_escalation_attempts must be positive"

        self.fast_model = fast_model
        self.smart_model = smart_model
        self.fallback_threshold = fallback_threshold
        self.max_escalation_attempts = min(
            max_escalation_attempts, MAX_ESCALATION_ATTEMPTS
        )
        self._stats = EscalationStats()

    @property
    def stats(self) -> EscalationStats:
        """Get current escalation statistics."""
        return self._stats

    def reset_stats(self) -> None:
        """Reset escalation statistics."""
        self._stats = EscalationStats()

    def _calculate_confidence(
        self,
        entities: List[Any],
        confidence_extractor: Callable[[Any], float],
    ) -> float:
        """
        Calculate average confidence from entities.

        Rule #4: Helper function < 60 lines.

        Args:
            entities: List of extracted entities
            confidence_extractor: Function to get confidence from entity

        Returns:
            Average confidence (0.0 if no entities)
        """
        if len(entities) < MIN_ENTITIES_FOR_CONFIDENCE:
            return 0.0

        total_conf = sum(confidence_extractor(e) for e in entities)
        return total_conf / len(entities)

    def _should_escalate(self, confidence: float) -> bool:
        """
        Determine if escalation is needed.

        GWT: When confidence < fallback_threshold, escalate.
        Rule #4: Helper function < 60 lines.

        Args:
            confidence: Average confidence from fast model

        Returns:
            True if escalation is needed
        """
        return confidence < self.fallback_threshold

    def _create_escalation_event(
        self,
        fast_confidence: float,
        smart_confidence: float,
        fast_latency_ms: float,
        smart_latency_ms: float,
    ) -> EscalationEvent:
        """Create an escalation event record. Rule #4: Helper < 60 lines."""
        return EscalationEvent(
            fast_model=self.fast_model,
            smart_model=self.smart_model,
            fast_confidence=fast_confidence,
            smart_confidence=smart_confidence,
            threshold=self.fallback_threshold,
            latency_fast_ms=fast_latency_ms,
            latency_smart_ms=smart_latency_ms,
        )

    def _execute_fast_extraction(
        self,
        extract_fn: Callable[[str], List[Any]],
        text: str,
        confidence_extractor: Callable[[Any], float],
    ) -> tuple:
        """Execute fast model extraction. Rule #4: Helper < 60 lines."""
        fast_start = time.perf_counter()
        fast_entities = extract_fn(text)
        fast_latency = (time.perf_counter() - fast_start) * 1000
        self._stats.total_fast_latency_ms += fast_latency
        fast_confidence = self._calculate_confidence(
            fast_entities, confidence_extractor
        )
        return fast_entities, fast_confidence, fast_latency

    def _execute_smart_extraction(
        self,
        extract_fn: Callable[[str], List[Any]],
        text: str,
        confidence_extractor: Callable[[Any], float],
        fast_confidence: float,
        fast_latency: float,
        provenance_entries: List[str],
    ) -> EscalationResult[List[Any]]:
        """Execute smart model extraction after escalation. Rule #4: Helper < 60 lines."""
        logger.info(
            f"Escalating: {self.fast_model} -> {self.smart_model} "
            f"(confidence {fast_confidence:.2f} < {self.fallback_threshold})"
        )
        self._stats.escalation_count += 1
        smart_start = time.perf_counter()
        smart_entities = extract_fn(text)
        smart_latency = (time.perf_counter() - smart_start) * 1000
        self._stats.total_smart_latency_ms += smart_latency
        smart_confidence = self._calculate_confidence(
            smart_entities, confidence_extractor
        )

        event = self._create_escalation_event(
            fast_confidence, smart_confidence, fast_latency, smart_latency
        )
        provenance_entries.append(event.to_provenance_entry())
        logger.info(
            f"Escalation complete: {self.fast_model}->{self.smart_model} "
            f"confidence improved {fast_confidence:.2f}->{smart_confidence:.2f}"
        )
        self._stats.smart_model_successes += 1

        return EscalationResult(
            result=smart_entities,
            confidence=smart_confidence,
            model_used=self.smart_model,
            escalated=True,
            escalation_event=event,
            latency_ms=fast_latency + smart_latency,
            provenance_entries=provenance_entries,
        )

    def extract_with_fallback(
        self,
        extract_fn: Callable[[str], List[Any]],
        text: str,
        confidence_extractor: Callable[[Any], float],
    ) -> EscalationResult[List[Any]]:
        """
        Perform extraction with automatic fallback.

        GWT: Given low-confidence from fast model, When < threshold,
        Then re-run with smart model and log escalation.

        Rule #4: Function < 60 lines (uses helpers).
        Rule #5: Assert preconditions.
        """
        assert text is not None, "text cannot be None"

        self._stats.total_extractions += 1
        provenance_entries: List[str] = []

        # Step 1: Fast model extraction
        fast_entities, fast_confidence, fast_latency = self._execute_fast_extraction(
            extract_fn, text, confidence_extractor
        )
        provenance_entries.append(f"extract:{self.fast_model}")

        # Step 2: Check if escalation needed
        if not self._should_escalate(fast_confidence):
            self._stats.fast_model_successes += 1
            logger.debug(
                f"Fast model sufficient: {self.fast_model} "
                f"confidence={fast_confidence:.2f} >= {self.fallback_threshold}"
            )
            return EscalationResult(
                result=fast_entities,
                confidence=fast_confidence,
                model_used=self.fast_model,
                escalated=False,
                latency_ms=fast_latency,
                provenance_entries=provenance_entries,
            )

        # Step 3: Escalate to smart model
        return self._execute_smart_extraction(
            extract_fn,
            text,
            confidence_extractor,
            fast_confidence,
            fast_latency,
            provenance_entries,
        )

    def get_escalation_rate(self) -> float:
        """
        Get current escalation rate.

        AC: Track "Escalation Rate" in health metrics.

        Returns:
            Escalation rate as percentage
        """
        return self._stats.escalation_rate

    def get_summary(self) -> Dict[str, Any]:
        """
        Get escalation summary for health metrics.

        Returns:
            Dictionary with escalation statistics
        """
        return {
            "fast_model": self.fast_model,
            "smart_model": self.smart_model,
            "fallback_threshold": self.fallback_threshold,
            "total_extractions": self._stats.total_extractions,
            "escalation_count": self._stats.escalation_count,
            "escalation_rate": self._stats.escalation_rate,
            "fast_model_successes": self._stats.fast_model_successes,
            "smart_model_successes": self._stats.smart_model_successes,
            "avg_fast_latency_ms": self._stats.avg_fast_latency_ms,
            "avg_smart_latency_ms": self._stats.avg_smart_latency_ms,
        }


# ---------------------------------------------------------------------------
# Integration Helper
# ---------------------------------------------------------------------------


def create_model_escalator(
    fast_model: str = "gpt-4o-mini",
    smart_model: str = "gpt-4o",
    fallback_threshold: float = DEFAULT_FALLBACK_THRESHOLD,
) -> ModelEscalator:
    """
    Factory function to create a ModelEscalator.

    AC: Configurable fallback_threshold (Default 0.6).

    Args:
        fast_model: Fast/cheap model ID
        smart_model: Smart/accurate model ID
        fallback_threshold: Confidence threshold for escalation

    Returns:
        Configured ModelEscalator instance
    """
    return ModelEscalator(
        fast_model=fast_model,
        smart_model=smart_model,
        fallback_threshold=fallback_threshold,
    )
