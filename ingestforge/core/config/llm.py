"""
LLM configuration.

Provides configuration for LLM providers: Claude, OpenAI, Gemini, Ollama, llama.cpp.
Includes per-provider settings, temperature presets by command type, and local model config.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LLMProviderConfig:
    """Individual LLM provider configuration."""

    model: str = ""
    api_key: str = ""
    url: str = ""
    temperature: float = (
        0.3  # Generation temperature (0.0-1.0), low for factual accuracy
    )


@dataclass
class LlamaCppConfig:
    """llama.cpp local model configuration.

    This is the default LLM provider - runs locally without a server.

    Recommended model: Qwen2.5-14B-Instruct (Q4_K_M quantization)
    Download with:
        huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir .data/models

    Then set model_path to: .data/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf
    """

    model_path: str = ""  # Path to GGUF model file (e.g., .data/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf)
    n_ctx: int = 8192  # Context window size (Qwen2.5 supports up to 128k)
    n_gpu_layers: int = (
        0  # 0 = auto-detect from VRAM, -1 = all layers, N = specific count
    )
    n_threads: int = 0  # 0 = auto
    auto_gpu_layers: bool = True  # Enable VRAM-based auto-detection (recommended)


@dataclass
class LLMConfig:
    """LLM providers configuration."""

    default_provider: str = "llamacpp"  # Local LLM via llama.cpp (no server needed)
    gemini: LLMProviderConfig = field(
        default_factory=lambda: LLMProviderConfig(model="gemini-1.5-flash")
    )
    claude: LLMProviderConfig = field(
        default_factory=lambda: LLMProviderConfig(model="claude-3-haiku-20240307")
    )
    openai: LLMProviderConfig = field(
        default_factory=lambda: LLMProviderConfig(model="gpt-4o-mini")
    )
    ollama: LLMProviderConfig = field(
        default_factory=lambda: LLMProviderConfig(
            model="qwen2.5:14b", url="http://localhost:11434"
        )
    )
    llamacpp: LlamaCppConfig = field(default_factory=LlamaCppConfig)

    # Command-specific temperature overrides (auto-applied based on tool type)
    # Users can override in config.yaml under llm.command_temperatures
    command_temperatures: Dict[str, float] = field(default_factory=lambda: {})

    # Built-in temperature presets by command category
    _TEMPERATURE_PRESETS: Dict[str, float] = field(
        default_factory=lambda: {
            # Factual/citation commands - very low temperature for accuracy
            "query": 0.1,
            "cite": 0.1,
            "quote": 0.1,
            "bibliography": 0.1,
            # Study material generation - low temperature for consistency
            "glossary": 0.2,
            "flashcards": 0.2,
            "quiz": 0.3,
            "overview": 0.2,
            "timeline": 0.2,
            # Analysis commands - medium temperature for synthesis
            "explain": 0.3,
            "compare": 0.4,
            "support": 0.4,
            "conflicts": 0.3,
            "gaps": 0.3,
            "scholars": 0.2,
            "concept-map": 0.3,
            # Creative/argumentative commands - higher temperature
            "draft": 0.6,
            "debate": 0.7,
            "thesis": 0.6,
            "counter": 0.6,
            "research": 0.5,
            # Literary analysis - medium-high for interpretation
            "lit-themes": 0.5,
            "lit-symbols": 0.5,
            "lit-character": 0.4,
            "lit-context": 0.3,
        },
        repr=False,
    )

    def get_temperature(self, command: str) -> float:
        """Get temperature for a specific command.

        Priority:
        1. User override in command_temperatures
        2. Built-in preset for the command
        3. Default provider temperature (0.3)
        """
        # Check user override first
        if command in self.command_temperatures:
            return self.command_temperatures[command]
        # Then check built-in presets
        if command in self._TEMPERATURE_PRESETS:
            return self._TEMPERATURE_PRESETS[command]
        # Fall back to default
        return 0.3
