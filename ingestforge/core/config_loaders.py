"""
Configuration Loading and Management Functions.

Handles loading, saving, and applying overrides to IngestForge configuration.

This module is part of the config refactoring (Sprint 3, Rule #4)
to reduce config.py from 1,035 lines to <450 lines.
"""

import os
import re
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import yaml

from ingestforge.core.security.env import (
    get_env_int,
    get_env_float,
    get_env_path,
    get_env_whitelist,
    STORAGE_BACKENDS,
    OCR_ENGINES,
    LLM_PROVIDERS,
)

if TYPE_CHECKING:
    from ingestforge.core.config import Config


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls) -> Any:
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


# Import config classes (avoid circular import by importing at module level)


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in config values.

    Handles nested structures including:
    - Strings with ${VAR_NAME} or ${VAR_NAME:default} syntax
    - Nested dictionaries
    - Nested lists (including lists of dicts, lists of lists)

    Args:
        value: Configuration value (string, dict, list, or primitive)

    Returns:
        Value with all environment variables expanded
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} or ${VAR_NAME:default} pattern
        pattern = r"\$\{([^}:]+)(?::([^}]*))?\}"

        def replace_env_var(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default_value)

        return re.sub(pattern, replace_env_var, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        # Handle nested lists correctly (list of lists, list of dicts, etc.)
        return [expand_env_vars(item) for item in value]
    # Return primitives (int, float, bool, None) unchanged
    return value


def _apply_env_overrides(config: "Config") -> "Config":
    """
    Apply environment variable overrides to configuration.

    Environment variables take precedence over config file values.
    This allows deployments to override config without modifying files.
    """
    _apply_api_key_overrides(config)
    _apply_llm_config_overrides(config)
    _apply_embedding_overrides(config)
    _apply_storage_overrides(config)
    _apply_ocr_overrides(config)
    _apply_processing_overrides(config)
    _apply_api_server_overrides(config)
    return config


def _apply_api_key_overrides(config: "Config") -> None:
    """Apply LLM API key overrides from environment."""
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        config.llm.gemini.api_key = gemini_key

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        config.llm.openai.api_key = openai_key

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        config.llm.claude.api_key = anthropic_key


def _apply_llm_config_overrides(config: "Config") -> None:
    """Apply LLM provider, model, and temperature overrides.

    Security:
        Uses get_env_whitelist to validate provider values.
        Uses get_env_float with bounds for temperature.
    """
    # Whitelist validation for LLM provider
    llm_provider = get_env_whitelist("INGESTFORGE_LLM_PROVIDER", LLM_PROVIDERS)
    if llm_provider:
        config.llm.default_provider = llm_provider

    # Model name - basic alphanumeric validation
    llm_model = os.environ.get("INGESTFORGE_LLM_MODEL")
    if llm_model:
        import re

        # Allow common model name patterns (alphanumeric, dash, underscore, dot, colon, slash)
        if re.match(r"^[a-zA-Z0-9._:\-/]+$", llm_model):
            _set_provider_model(config, llm_model)

    # Temperature with bounds validation
    llm_temp = get_env_float(
        "INGESTFORGE_LLM_TEMPERATURE",
        min_value=0.0,
        max_value=2.0,
    )
    if llm_temp is not None:
        _set_provider_temperature_value(config, llm_temp)


def _set_provider_model(config: "Config", model: str) -> None:
    """
    Set[str] model for the current default provider.

    Rule #1: Dictionary dispatch eliminates nesting
    """
    provider = config.llm.default_provider
    provider_configs = {
        "gemini": config.llm.gemini,
        "openai": config.llm.openai,
        "anthropic": config.llm.claude,
        "claude": config.llm.claude,
        "ollama": config.llm.ollama,
    }

    provider_config = provider_configs.get(provider)
    if provider_config:
        provider_config.model = model


def _set_provider_temperature(config: "Config", temp_str: str) -> None:
    """
    Set temperature for the current default provider from string.

    Rule #1: Dictionary dispatch eliminates nesting

    Note: Prefer _set_provider_temperature_value with pre-validated float.
    """
    try:
        temp_value = float(temp_str)
    except ValueError:
        _Logger.get().warning(f"Invalid temperature value '{temp_str}', ignoring")
        return  # Invalid temperature value, ignore

    _set_provider_temperature_value(config, temp_value)


def _set_provider_temperature_value(config: "Config", temp_value: float) -> None:
    """
    Set temperature for the current default provider from validated float.

    Rule #1: Dictionary dispatch eliminates nesting

    Args:
        config: Configuration object
        temp_value: Pre-validated temperature value (0.0-2.0)
    """
    provider = config.llm.default_provider
    provider_configs = {
        "gemini": config.llm.gemini,
        "openai": config.llm.openai,
        "anthropic": config.llm.claude,
        "claude": config.llm.claude,
        "ollama": config.llm.ollama,
    }

    provider_config = provider_configs.get(provider)
    if provider_config:
        provider_config.temperature = temp_value


def _apply_embedding_overrides(config: "Config") -> None:
    """Apply embedding configuration overrides.

    Security:
        Uses get_env_int with bounds for batch size validation.
    """
    # Model name - basic alphanumeric validation
    embedding_model = os.environ.get("INGESTFORGE_EMBEDDING_MODEL")
    if embedding_model:
        import re

        # Allow common model name patterns
        if re.match(r"^[a-zA-Z0-9._:\-/]+$", embedding_model):
            config.enrichment.embedding_model = embedding_model

    # Batch size with reasonable bounds (1 to 1000)
    embedding_batch = get_env_int(
        "INGESTFORGE_EMBEDDING_BATCH_SIZE",
        min_value=1,
        max_value=1000,
    )
    if embedding_batch is not None:
        config.enrichment.embedding_batch_size = embedding_batch


def _apply_storage_overrides(config: "Config") -> None:
    """Apply storage and path configuration overrides.

    Security:
        Uses get_env_path with base_dir for path traversal protection.
        Uses get_env_whitelist to validate storage backend values.
    """
    # Path validation with traversal protection
    data_path = get_env_path(
        "INGESTFORGE_DATA_PATH",
        base_dir=config._base_path,
    )
    if data_path:
        config.project.data_dir = str(data_path)

    pending_path = get_env_path(
        "INGESTFORGE_PENDING_PATH",
        base_dir=config._base_path,
    )
    if pending_path:
        config.ingest.pending_path_override = str(pending_path)

    # Whitelist validation for storage backend
    storage_backend = get_env_whitelist(
        "INGESTFORGE_STORAGE_BACKEND",
        STORAGE_BACKENDS,
    )
    if storage_backend:
        config.storage.backend = storage_backend


def _apply_ocr_overrides(config: "Config") -> None:
    """Apply OCR engine overrides.

    Security:
        Uses get_env_whitelist to validate OCR engine values.
    """
    ocr_engine = get_env_whitelist("INGESTFORGE_OCR_ENGINE", OCR_ENGINES)
    if ocr_engine:
        config.ocr.preferred_engine = ocr_engine


def _apply_processing_overrides(config: "Config") -> None:
    """Apply processing configuration overrides.

    Security:
        Uses get_env_float and get_env_int with bounds validation
        to prevent DoS via extreme configuration values.
    """
    # Float with bounds (0.1 MB to 1000 MB)
    max_inline = get_env_float(
        "INGESTFORGE_MAX_INLINE_SIZE_MB",
        min_value=0.1,
        max_value=1000.0,
    )
    if max_inline is not None:
        config.ingest.max_inline_size_mb = max_inline

    # Integer with bounds (50 to 10000 chars)
    chunk_size = get_env_int(
        "INGESTFORGE_CHUNK_SIZE",
        min_value=50,
        max_value=10000,
    )
    if chunk_size is not None:
        config.chunking.target_size = chunk_size

    # Integer with bounds (0 to 500 chars overlap)
    chunk_overlap = get_env_int(
        "INGESTFORGE_CHUNK_OVERLAP",
        min_value=0,
        max_value=500,
    )
    if chunk_overlap is not None:
        config.chunking.overlap = chunk_overlap


def _validate_cors_origin(origin: str) -> Optional[str]:
    """Validate a single CORS origin.

    Rule #1: Extracted to reduce nesting in _apply_api_server_overrides.

    Args:
        origin: Origin string to validate

    Returns:
        Validated origin or None if invalid
    """
    origin = origin.strip()

    # Accept wildcard
    if origin == "*":
        return origin

    # Must start with http:// or https://
    if not origin.startswith(("http://", "https://")):
        return None

    # Basic URL structure validation
    if re.match(r"^https?://[a-zA-Z0-9.\-:]+$", origin):
        return origin

    return None


def _apply_api_server_overrides(config: "Config") -> None:
    """Apply API server configuration overrides.

    Rule #1: Refactored to max 2 nesting levels (was 4).

    Security:
        Uses get_env_int with port bounds (1-65535).
        Validates CORS origins for valid URL format.
    """
    # Host validation - only allow safe characters
    api_host = os.environ.get("INGESTFORGE_API_HOST")
    if api_host and re.match(r"^[a-zA-Z0-9.\-]+$", api_host):
        config.api.host = api_host

    # Port with valid range bounds
    api_port = get_env_int(
        "INGESTFORGE_API_PORT",
        min_value=1,
        max_value=65535,
    )
    if api_port is not None:
        config.api.port = api_port

    # CORS origins with URL validation (Rule #1: use helper to avoid nesting)
    cors_origins = os.environ.get("INGESTFORGE_CORS_ORIGINS")
    if not cors_origins:
        return

    validated = [_validate_cors_origin(o) for o in cors_origins.split(",")]
    validated = [o for o in validated if o is not None]

    if validated:
        config.api.cors_origins = validated


def load_config(
    config_path: Optional[Path] = None, base_path: Optional[Path] = None
) -> "Config":
    """
    Load configuration from YAML file with environment variable overrides.

    Rule #4: Reduced from 65 → 35 lines

    Configuration precedence: 1. Env vars, 2. YAML file, 3. Defaults
    See module docstring for environment variable details.

    Args:
        config_path: Path to config file. Defaults to config.yaml in base_path.
        base_path: Base path for the project. Defaults to current directory.

    Returns:
        Config object with all settings.
    """
    # Lazy import to avoid circular dependency
    from ingestforge.core.config import Config

    base_path = base_path or Path.cwd()

    # Check for config file (support both config.yaml and ingestforge.yaml)
    if config_path is None:
        for filename in ["config.yaml", "ingestforge.yaml"]:
            candidate = base_path / filename
            if candidate.exists():
                config_path = candidate
                break
        else:
            # No config file found
            return _create_default_config(base_path)
    if not config_path.exists():
        return _create_default_config(base_path)

    # Try to load from YAML
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        config = Config.from_dict(data, base_path)
        return _apply_env_overrides(config)

    except Exception as e:
        print(f"Warning: Could not load config from {config_path}: {e}")
        print("Using default configuration.")
        return _create_default_config(base_path)


def _create_default_config(base_path: Path) -> "Config":
    """
    Create default configuration with environment overrides.

    Rule #4: Extracted to reduce duplication (<60 lines)
    """
    # Lazy import to avoid circular dependency
    from ingestforge.core.config import Config

    config = Config()
    config._base_path = base_path
    return _apply_env_overrides(config)


def save_config(config: "Config", config_path: Optional[Path] = None) -> None:
    """Save configuration to YAML file."""
    if config_path is None:
        config_path = config._base_path / "config.yaml"

    config_dict = config.to_dict()

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def apply_performance_preset(config: "Config") -> "Config":
    """
    Adjust configuration settings based on the performance_mode field.

    Performance modes trade off quality for speed and resource usage:

    - "quality": Maximum accuracy. Enables reranking, entity extraction,
      and uses smaller chunk sizes for precise retrieval.
    - "balanced": Good default. Reranking enabled but entity extraction
      disabled to save processing time.
    - "speed": Minimal resource usage. Disables reranking, entity
      extraction, and question generation. Uses larger chunks to
      reduce the total number of chunks processed.
    - "mobile": Optimized for Android/Termux. Uses JSONL storage with
      compression, BM25-only retrieval, no embeddings, no file watching.
      Eliminates all native C++ dependency requirements.

    Args:
        config: Configuration object with performance_mode set.

    Returns:
        The same Config object with settings adjusted in place.

    Usage:
        config = load_config()
        config.performance_mode = "speed"
        config = apply_performance_preset(config)
    """
    mode = config.performance_mode.lower()

    presets = {
        "quality": _apply_quality_preset,
        "balanced": _apply_balanced_preset,
        "speed": _apply_speed_preset,
        "mobile": _apply_mobile_preset,
    }

    if mode in presets:
        presets[mode](config)
    else:
        raise ValueError(
            f"Unknown performance_mode: '{mode}'. "
            f"Valid options: 'quality', 'balanced', 'speed', 'mobile'."
        )

    return config


def _apply_quality_preset(config: "Config") -> None:
    """Apply quality preset: maximum accuracy with reranking and entity extraction."""
    config.retrieval.rerank = True
    config.enrichment.embedding_model = "all-MiniLM-L6-v2"
    config.chunking.target_size = 300
    config.enrichment.extract_entities = True


def _apply_balanced_preset(config: "Config") -> None:
    """Apply balanced preset: good default with reranking but no entity extraction."""
    config.retrieval.rerank = True
    config.enrichment.embedding_model = "all-MiniLM-L6-v2"
    config.chunking.target_size = 300
    config.enrichment.extract_entities = False


def _apply_speed_preset(config: "Config") -> None:
    """Apply speed preset: minimal resource usage with larger chunks."""
    config.retrieval.rerank = False
    config.enrichment.embedding_model = "all-MiniLM-L6-v2"
    config.chunking.target_size = 500
    config.enrichment.extract_entities = False
    config.enrichment.generate_questions = False


def _apply_mobile_preset(config: "Config") -> None:
    """Apply mobile preset: optimized for Android/Termux with no native C++ dependencies."""
    config.storage.backend = "jsonl"
    config.storage.compression = True
    config.retrieval.strategy = "bm25"
    config.retrieval.rerank = False
    config.enrichment.generate_embeddings = False
    config.enrichment.extract_entities = False
    config.enrichment.generate_questions = False
    config.enrichment.compute_quality = False
    config.chunking.strategy = "fixed"
    config.chunking.target_size = 500
    config.chunking.use_llm = False
    config.ingest.watch_enabled = False
