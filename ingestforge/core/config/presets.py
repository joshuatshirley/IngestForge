"""Performance Presets and Resource Mapping.

Defines hardware-specific configuration overrides for Eco, Balanced, and Performance modes.
Follows NASA JPL Rule #4 (Modular) and Rule #10 (Static Config).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any


@dataclass(frozen=True)
class PresetValues:
    """Immutable values for a specific performance tier."""

    n_threads: int
    embedding_batch_size: int
    context_window: int
    concurrency_limit: int
    use_quantization: bool


PERFORMANCE_TIERS: Dict[str, PresetValues] = {
    "eco": PresetValues(
        n_threads=2,
        embedding_batch_size=1,
        context_window=2048,
        concurrency_limit=1,
        use_quantization=True,
    ),
    "balanced": PresetValues(
        n_threads=4,
        embedding_batch_size=32,
        context_window=4096,
        concurrency_limit=2,
        use_quantization=True,
    ),
    "performance": PresetValues(
        n_threads=0,  # 0 means use all available cores
        embedding_batch_size=128,
        context_window=8192,
        concurrency_limit=4,
        use_quantization=False,
    ),
}


class PerformanceOptimizer:
    """Logic for applying hardware-aware configuration overrides."""

    def apply_preset(self, config: Any, tier: str) -> Any:
        """Override config object values based on selected tier.

        Rule #7: Validate tier name.
        Rule #1: Linear logic.
        """
        if tier not in PERFORMANCE_TIERS:
            tier = "balanced"  # Safe fallback

        preset = PERFORMANCE_TIERS[tier]

        # Apply overrides to the dynamic config object
        # Note: In real use, this modifies config.llm and config.enrichment
        if hasattr(config, "llm") and hasattr(config.llm, "llamacpp"):
            config.llm.llamacpp.n_threads = preset.n_threads
            config.llm.llamacpp.n_ctx = preset.context_window

        if hasattr(config, "enrichment"):
            config.enrichment.embedding_batch_size = preset.embedding_batch_size

        return config
