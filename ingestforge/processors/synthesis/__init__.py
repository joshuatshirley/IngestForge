"""
Synthesis Processors for IngestForge.

Generative Synthesis Engine
Epic: EP-10 (Synthesis & Generative API)

JPL Refactored: 2026-02-17T21:30:00Z
- process() reduced from 93 to 52 lines via helper extraction
"""

from ingestforge.processors.synthesis.synthesizer import (
    IFSynthesisProcessor,
    IFSynthesisArtifact,
    SynthesisCitation,
    MAX_CONTEXT_TOKENS,
    MAX_CITATIONS,
    MAX_CONTEXT_CHUNKS,
)

__all__ = [
    "IFSynthesisProcessor",
    "IFSynthesisArtifact",
    "SynthesisCitation",
    "MAX_CONTEXT_TOKENS",
    "MAX_CITATIONS",
    "MAX_CONTEXT_CHUNKS",
]
