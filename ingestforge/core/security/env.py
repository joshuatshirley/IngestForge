"""
Safe environment variable parsing with validation.

Provides type-safe functions for reading environment variables with
bounds checking, whitelist validation, and path sanitization.

Threat Model
------------
Environment variables are an attack vector in deployments:

    1. **Integer Overflow**: INGESTFORGE_API_PORT=99999999
       Defense: get_env_int with bounds checking

    2. **Path Traversal**: INGESTFORGE_DATA_PATH=../../../etc
       Defense: get_env_path with PathSanitizer

    3. **Injection via Config**: INGESTFORGE_STORAGE_BACKEND="'; DROP TABLE--"
       Defense: get_env_whitelist validates against allowed values

Usage Pattern
-------------
Instead of unsafe direct environment access:

    # DANGEROUS - no validation
    port = int(os.environ.get("PORT", "8080"))

Use safe getters:

    # SAFE - bounds checked
    from ingestforge.core.security.env import get_env_int
    port = get_env_int("PORT", default=8080, min_value=1, max_value=65535)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import FrozenSet, Optional

from ingestforge.core.security.path import PathSanitizer

logger = logging.getLogger(__name__)


# Whitelists for common configuration values
STORAGE_BACKENDS: FrozenSet[str] = frozenset(
    ["chromadb", "jsonl", "postgres", "sqlite"]
)

OCR_ENGINES: FrozenSet[str] = frozenset(["tesseract", "easyocr", "paddleocr", "none"])

LLM_PROVIDERS: FrozenSet[str] = frozenset(
    ["gemini", "openai", "anthropic", "ollama", "llamacpp"]
)

CHUNKING_STRATEGIES: FrozenSet[str] = frozenset(
    ["semantic", "fixed", "paragraph", "header"]
)

PERFORMANCE_MODES: FrozenSet[str] = frozenset(
    ["quality", "balanced", "speed", "mobile"]
)


def get_env_int(
    name: str,
    default: Optional[int] = None,
    min_value: Optional[int] = None,
    max_value: Optional[int] = None,
) -> Optional[int]:
    """
    Get integer from environment variable with bounds validation.

    Args:
        name: Environment variable name.
        default: Default value if not set or invalid.
        min_value: Minimum allowed value (clamped if exceeded).
        max_value: Maximum allowed value (clamped if exceeded).

    Returns:
        Validated integer or default.

    Example:
        >>> get_env_int("PORT", default=8080, min_value=1, max_value=65535)
        8080  # If PORT not set

        >>> # With PORT=99999
        >>> get_env_int("PORT", default=8080, min_value=1, max_value=65535)
        65535  # Clamped to max
    """
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        int_value = int(value)
    except ValueError:
        logger.warning(
            f"Invalid integer value for {name}={value}: Returning default {default}"
        )
        return default

    # Clamp to bounds
    if min_value is not None and int_value < min_value:
        return min_value
    if max_value is not None and int_value > max_value:
        return max_value

    return int_value


def get_env_float(
    name: str,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> Optional[float]:
    """
    Get float from environment variable with bounds validation.

    Args:
        name: Environment variable name.
        default: Default value if not set or invalid.
        min_value: Minimum allowed value (clamped if exceeded).
        max_value: Maximum allowed value (clamped if exceeded).

    Returns:
        Validated float or default.

    Example:
        >>> get_env_float("TEMPERATURE", default=0.7, min_value=0.0, max_value=2.0)
        0.7
    """
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        float_value = float(value)
    except ValueError:
        logger.warning(
            f"Invalid float value for {name}={value}: Returning default {default}"
        )
        return default

    # Check for special float values
    if float_value != float_value:  # NaN check
        return default

    # Clamp to bounds
    if min_value is not None and float_value < min_value:
        return min_value
    if max_value is not None and float_value > max_value:
        return max_value

    return float_value


def get_env_path(
    name: str,
    base_dir: Optional[Path] = None,
    default: Optional[Path] = None,
    must_exist: bool = False,
) -> Optional[Path]:
    """
    Get path from environment variable with traversal protection.

    When base_dir is provided, uses PathSanitizer to ensure the path
    doesn't escape the base directory via traversal attacks.

    Args:
        name: Environment variable name.
        base_dir: Base directory for path validation. If provided,
                  paths are validated to stay within this directory.
        default: Default path if not set or invalid.
        must_exist: If True, returns default if path doesn't exist.

    Returns:
        Validated Path or default.

    Example:
        >>> get_env_path("DATA_DIR", base_dir=Path("/app"), default=Path("/app/data"))
        Path('/app/data')

        >>> # With DATA_DIR="../../../etc"
        >>> get_env_path("DATA_DIR", base_dir=Path("/app"))
        None  # Traversal blocked, returns default
    """
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        if base_dir is not None:
            # Use PathSanitizer for traversal protection
            sanitizer = PathSanitizer(base_dir)
            path = sanitizer.sanitize_path(value)
        else:
            # No base directory, just resolve
            path = Path(value).resolve()

        # Optionally check existence
        if must_exist and not path.exists():
            return default

        return path

    except Exception:
        # PathTraversalError or other issues
        logger.exception(
            f"Error parsing environment variable {name}={value}: "
            f"Returning default {default}"
        )
        return default


def get_env_whitelist(
    name: str,
    allowed: FrozenSet[str],
    default: Optional[str] = None,
    case_sensitive: bool = False,
) -> Optional[str]:
    """
    Get string from environment variable with whitelist validation.

    Only returns value if it matches one of the allowed values.

    Args:
        name: Environment variable name.
        allowed: Set of allowed values.
        default: Default value if not set or not in whitelist.
        case_sensitive: If False (default), comparison is case-insensitive.

    Returns:
        Validated string or default.

    Example:
        >>> get_env_whitelist("STORAGE", STORAGE_BACKENDS, default="chromadb")
        'chromadb'

        >>> # With STORAGE="malicious_value"
        >>> get_env_whitelist("STORAGE", STORAGE_BACKENDS, default="chromadb")
        'chromadb'  # Returns default, not the malicious value
    """
    value = os.environ.get(name)
    if value is None:
        return default

    if case_sensitive:
        if value in allowed:
            return value
    else:
        normalized = value.lower()
        for allowed_value in allowed:
            if normalized == allowed_value.lower():
                return allowed_value

    return default


def get_env_bool(
    name: str,
    default: bool = False,
) -> bool:
    """
    Get boolean from environment variable.

    Recognizes common truthy/falsy values:
    - True: "true", "yes", "1", "on"
    - False: "false", "no", "0", "off", ""

    Args:
        name: Environment variable name.
        default: Default value if not set or unrecognized.

    Returns:
        Boolean value.

    Example:
        >>> get_env_bool("DEBUG", default=False)
        False

        >>> # With DEBUG="true"
        >>> get_env_bool("DEBUG", default=False)
        True
    """
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.lower().strip()

    if normalized in ("true", "yes", "1", "on"):
        return True
    if normalized in ("false", "no", "0", "off", ""):
        return False

    return default
