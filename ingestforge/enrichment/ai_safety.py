"""
AI Safety and Alignment enrichment.

Extracts model names, parameter counts, and safety benchmarks.
"""

import re
import logging

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class AISafetyMetadataRefiner:
    """
    Enriches chunks with AI safety-specific metadata.
    """

    # AI Safety specific patterns
    MODEL_PATTERN = re.compile(
        r"\b(?:Model|LLM|System)[:\s]+([A-Z][\w\-\.]{2,25}?(?:\s\d+)?)(?=[.,\s]|\Z)"
    )
    PARAM_PATTERN = re.compile(
        r"(\d+(?:\.\d+)?\s*(?:Trillion|Billion|Million|[BMT]))", re.IGNORECASE
    )
    BENCHMARK_KEYWORDS = [
        "TruthfulQA",
        "HellaSwag",
        "MMLU",
        "GSM8K",
        "ARC",
        "HumanEval",
    ]

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich chunk with AI safety metadata."""
        content = chunk.content
        metadata = chunk.metadata or {}

        # Extract Model Name
        model_match = self.MODEL_PATTERN.search(content)
        if model_match:
            metadata["ai_model_name"] = model_match.group(1).strip()

        # Extract Parameter Count
        param_match = self.PARAM_PATTERN.search(content)
        if param_match:
            metadata["ai_param_count"] = " ".join(
                param_match.group(1).strip().upper().split()
            )
        else:
            # Check for fallback Scale: label if pattern above missed it
            fallback = re.search(
                r"\bScale[:\s\-#]*(\d+(?:\.\d+)?\s*[BMT]illion|\d+[BMT])",
                content,
                re.IGNORECASE,
            )
            if fallback:
                metadata["ai_param_count"] = " ".join(
                    fallback.group(1).strip().upper().split()
                )

        # Extract Benchmarks (Score mapping)
        benchmarks = {}
        for bm in self.BENCHMARK_KEYWORDS:
            # Look for "Benchmark: XX.X%" or "Benchmark score of XX.X"
            pattern = rf"\b{bm}\b.*?(\d{{1,3}}(?:\.\d+)?%?)"
            bm_match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if bm_match:
                val_str = bm_match.group(1).replace("%", "")
                benchmarks[bm] = float(val_str)
        if benchmarks:
            metadata["ai_safety_benchmarks"] = benchmarks

        chunk.metadata = metadata
        return chunk
