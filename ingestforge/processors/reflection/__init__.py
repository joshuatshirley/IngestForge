"""
Reflection Processor Package.

Agentic Reflection Loop - Critic pass for extraction accuracy.
"""

from ingestforge.processors.reflection.reflection_processor import (
    IFReflectionProcessor,
    IFReflectionArtifact,
    ReflectionResult,
    ContradictionResult,
    MAX_REFLECTION_PASSES,
    CONFIDENCE_THRESHOLD,
)
from ingestforge.processors.reflection.stage import ReflectionStage

__all__ = [
    "IFReflectionProcessor",
    "IFReflectionArtifact",
    "ReflectionResult",
    "ContradictionResult",
    "ReflectionStage",
    "MAX_REFLECTION_PASSES",
    "CONFIDENCE_THRESHOLD",
]
