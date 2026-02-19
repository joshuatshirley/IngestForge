"""
LLM provider factory.

Create and configure LLM clients based on configuration.
"""

from typing import Any, Callable, Optional

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient, GenerationConfig

logger = get_logger(__name__)


def get_generation_config(
    config: Config,
    command: Optional[str] = None,
    **overrides,
) -> GenerationConfig:
    """
    Get a GenerationConfig with temperature tuned for the command.

    Different commands need different temperature settings:
    - Factual commands (query, cite) → low temperature (0.1-0.2)
    - Study materials (glossary, quiz) → medium-low (0.2-0.3)
    - Creative commands (draft, debate) → higher (0.6-0.7)

    Args:
        config: IngestForge configuration
        command: Command name (e.g., "query", "draft", "debate")
        **overrides: Additional overrides for GenerationConfig fields

    Returns:
        GenerationConfig with appropriate temperature
    """
    # Get temperature for this command
    temperature = config.llm.get_temperature(command) if command else 0.3

    # Create config with command-specific temperature
    gen_config = GenerationConfig(
        temperature=temperature,
        **overrides,
    )

    return gen_config


def _create_gemini_client(config: Config) -> LLMClient:
    """
    Create Gemini client.

    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints
    """
    from ingestforge.llm.gemini import GeminiClient

    return GeminiClient(
        api_key=config.llm.gemini.api_key or None,
        model=config.llm.gemini.model,
    )


def _create_claude_client(config: Config) -> LLMClient:
    """
    Create Claude client.

    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints
    """
    from ingestforge.llm.claude import ClaudeClient

    return ClaudeClient(
        api_key=config.llm.claude.api_key or None,
        model=config.llm.claude.model,
    )


def _create_openai_client(config: Config) -> LLMClient:
    """
    Create OpenAI client.

    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints
    """
    from ingestforge.llm.openai import OpenAIClient

    return OpenAIClient(
        api_key=config.llm.openai.api_key or None,
        model=config.llm.openai.model,
    )


def _create_ollama_client(config: Config) -> LLMClient:
    """
    Create Ollama client.

    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints
    """
    from ingestforge.llm.ollama import OllamaClient

    return OllamaClient(
        url=config.llm.ollama.url,
        model=config.llm.ollama.model,
    )


def _find_local_model(config: Config) -> str:
    """
    Find local GGUF model in default directory.

    Rule #1: Early return eliminates nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        Path to model file

    Raises:
        ValueError: If no model found
    """

    models_dir = config.data_path / "models"
    if not models_dir.exists():
        raise ValueError(
            "No local model configured. Set llm.llamacpp.model_path in config "
            "or download a model with:\n"
            "  huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF "
            "Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir .data/models"
        )

    gguf_files = list(models_dir.glob("*.gguf"))
    if not gguf_files:
        raise ValueError(
            "No local model configured. Set llm.llamacpp.model_path in config "
            "or download a model with:\n"
            "  huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF "
            "Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir .data/models"
        )

    logger.info(f"Auto-detected local model: {gguf_files[0].name}")
    return str(gguf_files[0])


def _create_llamacpp_client(config: Config) -> LLMClient:
    """
    Create llama.cpp client with auto-detection.

    Rule #1: Early return eliminates nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        Configured LlamaCppClient

    Raises:
        ValueError: If no model configured or found
    """
    from ingestforge.llm.llamacpp import LlamaCppClient

    # Get llamacpp config if available
    llamacpp_config = getattr(config.llm, "llamacpp", None)
    if llamacpp_config and llamacpp_config.model_path:
        return LlamaCppClient(
            model_path=llamacpp_config.model_path,
            n_ctx=getattr(llamacpp_config, "n_ctx", 8192),
            n_gpu_layers=getattr(llamacpp_config, "n_gpu_layers", 0),
            n_threads=getattr(llamacpp_config, "n_threads", None) or None,
            auto_gpu_layers=getattr(llamacpp_config, "auto_gpu_layers", True),
        )

    # Try to auto-detect model in default location
    model_path = _find_local_model(config)
    return LlamaCppClient(
        model_path=model_path,
        n_ctx=8192,
        auto_gpu_layers=True,
    )


def _get_provider_factory(provider: str) -> Optional[Callable[[Config], LLMClient]]:
    """
    Get factory function for LLM provider.

    Rule #1: Dictionary dispatch eliminates nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints

    Args:
        provider: Provider name (gemini, claude, openai, ollama, llamacpp, local)

    Returns:
        Factory function or None if unknown provider
    """
    # Cloud providers require explicit confirmation (logged as warning)
    CLOUD_PROVIDERS = {"gemini", "claude", "openai"}

    if provider in CLOUD_PROVIDERS:
        logger.warning(
            f"[CLOUD] Using cloud provider '{provider}' - API charges may apply!"
        )

    # All providers available
    factories = {
        # Cloud providers (API charges apply)
        "gemini": _create_gemini_client,
        "claude": _create_claude_client,
        "openai": _create_openai_client,
        # Local providers (free)
        "ollama": _create_ollama_client,
        "llamacpp": _create_llamacpp_client,
        "local": _create_llamacpp_client,  # Alias for llamacpp
    }
    return factories.get(provider)


def get_llm_client(
    config: Config,
    provider: Optional[str] = None,
) -> LLMClient:
    """
    Get LLM client based on configuration.

    Rule #1: No nesting - pure dictionary dispatch
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration
        provider: Override provider (gemini, claude, openai, ollama, llamacpp, local)

    Returns:
        Configured LLMClient instance

    Raises:
        ValueError: If provider is unknown
    """
    provider = provider or config.llm.default_provider
    factory = _get_provider_factory(provider)
    if not factory:
        raise ValueError(f"Unknown LLM provider: {provider}")
    return factory(config)


def _check_ollama_available(config: Config) -> bool:
    """
    Check if Ollama is available.

    Rule #1: Extracted helper reduces nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        True if Ollama is available and reachable
    """
    assert config is not None, "Config cannot be None"
    assert hasattr(config, "llm"), "Config must have llm attribute"

    try:
        from ingestforge.llm.ollama import OllamaClient

        client = OllamaClient(url=config.llm.ollama.url)
        return client.is_available()
    except Exception as e:
        logger.debug(f"Failed to check Ollama availability: {e}")
        return False


def _check_llamacpp_configured_model(config: Config) -> bool:
    """
    Check if a configured llama.cpp model exists.

    Rule #1: Early return pattern eliminates nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        True if configured model path exists
    """
    from pathlib import Path

    assert config is not None, "Config cannot be None"

    llamacpp_config = getattr(config.llm, "llamacpp", None)
    if not llamacpp_config:
        logger.debug("No llamacpp config found in config.llm")
        return False
    if not llamacpp_config.model_path:
        logger.debug("llamacpp.model_path is empty or not set")
        return False
    model_path = Path(llamacpp_config.model_path)
    exists = model_path.exists()

    if exists:
        logger.info(f"Found configured llama.cpp model: {model_path}")
    else:
        logger.warning(f"Configured llama.cpp model not found at: {model_path}")

    return exists


def _check_llamacpp_default_models(config: Config) -> bool:
    """
    Check if any GGUF models exist in default directory.

    Rule #1: Early return pattern eliminates nesting
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        True if any .gguf files found in default models directory
    """
    assert config is not None, "Config cannot be None"
    assert hasattr(config, "data_path"), "Config must have data_path"
    models_dir = config.data_path / "models"
    logger.debug(f"Checking for models in default directory: {models_dir}")
    if not models_dir.exists():
        logger.debug(f"Default models directory does not exist: {models_dir}")
        return False
    gguf_files = list(models_dir.glob("*.gguf"))

    if gguf_files:
        logger.info(f"Found {len(gguf_files)} GGUF model(s) in default directory")
    else:
        logger.debug(f"No GGUF models found in: {models_dir}")

    return len(gguf_files) > 0


def _check_llamacpp_available(config: Config) -> bool:
    """
    Check if llama.cpp is available (configured or default model).

    Rule #1: No nesting - pure sequential checks with early returns
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        True if llama.cpp model is available
    """
    assert config is not None, "Config cannot be None"

    try:
        if _check_llamacpp_configured_model(config):
            return True
        return _check_llamacpp_default_models(config)

    except Exception as e:
        logger.debug(f"Failed to check llamacpp availability: {e}")
        return False


def get_available_providers(config: Config) -> list[Any]:
    """
    Get list of available (configured) providers.

    Rule #1: Dictionary dispatch eliminates if chains
    Rule #4: Function <60 lines
    Rule #7: Parameter validation
    Rule #9: Full type hints

    Args:
        config: IngestForge configuration

    Returns:
        List of available provider names

    Raises:
        AssertionError: If config is None or invalid
    """
    assert config is not None, "Config cannot be None"
    assert hasattr(config, "llm"), "Config must have llm attribute"
    PROVIDER_CHECKS = {
        # Cloud providers (check API key)
        "gemini": lambda cfg: bool(
            getattr(cfg.llm, "gemini", None) and cfg.llm.gemini.api_key
        ),
        "claude": lambda cfg: bool(
            getattr(cfg.llm, "claude", None) and cfg.llm.claude.api_key
        ),
        "openai": lambda cfg: bool(
            getattr(cfg.llm, "openai", None) and cfg.llm.openai.api_key
        ),
        # Local providers
        "ollama": _check_ollama_available,
        "llamacpp": _check_llamacpp_available,
    }
    available: list[Any] = []
    for provider, check_fn in PROVIDER_CHECKS.items():
        if check_fn(config):
            available.append(provider)
    assert isinstance(available, list), "Return value must be a list"

    return available


def get_best_available_client(config: Config) -> Optional[LLMClient]:
    """
    Get the best available LLM client.

    Priority: configured default > ollama > llamacpp > gemini > claude > openai
    (Local LLMs are preferred over external providers)

    Args:
        config: IngestForge configuration

    Returns:
        LLMClient or None if none available
    """
    # Log configured default for debugging
    logger.info(f"Configured default LLM provider: {config.llm.default_provider}")

    available = get_available_providers(config)
    logger.info(f"Available LLM providers: {available}")

    if not available:
        logger.warning(
            "No LLM providers available. Download a local model with:\n"
            "  huggingface-cli download bartowski/Qwen2.5-14B-Instruct-GGUF "
            "Qwen2.5-14B-Instruct-Q4_K_M.gguf --local-dir .data/models"
        )
        return None

    # Try configured default first
    if config.llm.default_provider in available:
        logger.info(
            f"Using configured default LLM provider: {config.llm.default_provider}"
        )
        return get_llm_client(config, config.llm.default_provider)

    # Log why default isn't available
    logger.warning(
        f"Configured default '{config.llm.default_provider}' not in available providers: {available}"
    )

    # Prefer local LLMs first to avoid API charges
    priority_order = ["llamacpp", "ollama", "gemini", "claude", "openai"]
    for provider in priority_order:
        if provider in available:
            logger.info(f"Using fallback LLM provider: {provider}")
            return get_llm_client(config, provider)

    # Fall back to first available (shouldn't reach here)
    return get_llm_client(config, available[0])
