"""
Scoped Configuration Injection for IngestForge (IF).

DI - Scoped Sub-Config Injection.
Provides type-safe, scoped configuration to processors.
Follows NASA JPL Power of Ten rules.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Type, Callable

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds for configuration
MAX_CONFIG_KEYS = 64
MAX_CONFIG_VALUE_SIZE = 16384  # 16KB per value
MAX_CONFIG_SCOPES = 32


@dataclass(frozen=True)
class ConfigKey:
    """
    Definition of a configuration key.

    DI - Scoped Sub-Config Injection.
    Rule #9: Complete type hints.

    Attributes:
        name: The configuration key name.
        required: Whether this key is required.
        default: Default value if not required and not provided.
        description: Human-readable description.
    """

    name: str
    required: bool = True
    default: Any = None
    description: str = ""


@dataclass(frozen=True)
class ConfigScope:
    """
    Defines the configuration scope for a processor.

    DI - Scoped Sub-Config Injection.
    Rule #2: Bounded key count.
    Rule #9: Complete type hints.

    Attributes:
        scope_id: Unique identifier for this scope.
        keys: Set of required/optional configuration keys.
        prefix: Optional prefix for namespacing (e.g., "ocr.", "llm.").
    """

    scope_id: str
    keys: FrozenSet[ConfigKey] = field(default_factory=frozenset)
    prefix: str = ""

    def __post_init__(self) -> None:
        """Validate scope constraints."""
        if len(self.keys) > MAX_CONFIG_KEYS:
            raise ValueError(
                f"ConfigScope '{self.scope_id}' exceeds {MAX_CONFIG_KEYS} keys"
            )

    @property
    def required_keys(self) -> FrozenSet[str]:
        """Get names of required keys."""
        return frozenset(k.name for k in self.keys if k.required)

    @property
    def optional_keys(self) -> FrozenSet[str]:
        """Get names of optional keys."""
        return frozenset(k.name for k in self.keys if not k.required)

    @property
    def all_key_names(self) -> FrozenSet[str]:
        """Get all key names (required and optional)."""
        return frozenset(k.name for k in self.keys)

    def get_key(self, name: str) -> Optional[ConfigKey]:
        """Get a ConfigKey by name."""
        for key in self.keys:
            if key.name == name:
                return key
        return None


@dataclass(frozen=True)
class ScopedConfig:
    """
    Immutable configuration subset for a specific scope.

    DI - Scoped Sub-Config Injection.
    Rule #6: Data in smallest scope.
    Rule #9: Complete type hints.

    Attributes:
        scope_id: The scope this config belongs to.
        values: Frozen dictionary of configuration values.
    """

    scope_id: str
    values: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure values dict is converted to immutable form."""
        if not isinstance(self.values, dict):
            raise TypeError("ScopedConfig values must be a dict")
        # Shallow freeze - values dict itself is already frozen by dataclass

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Rule #7: Explicit return (default if not found).

        Args:
            key: The configuration key.
            default: Default value if key not found.

        Returns:
            The configuration value or default.
        """
        return self.values.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get a configuration value (raises KeyError if not found)."""
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the config."""
        return key in self.values

    @property
    def key_count(self) -> int:
        """Get the number of keys in this config."""
        return len(self.values)


@dataclass(frozen=True)
class ConfigValidationError:
    """
    Represents a configuration validation error.

    DI - Scoped Sub-Config Injection.
    Rule #7: Explicit error reporting.

    Attributes:
        scope_id: The scope where the error occurred.
        key: The problematic key (if applicable).
        message: Human-readable error message.
        error_type: Type of validation error.
    """

    scope_id: str
    key: Optional[str]
    message: str
    error_type: str = "validation_error"


@dataclass(frozen=True)
class ConfigValidationResult:
    """
    Result of configuration validation.

    DI - Scoped Sub-Config Injection.
    Rule #7: Explicit return values.

    Attributes:
        valid: True if validation passed.
        errors: List of validation errors.
        scoped_config: The validated config (if valid).
    """

    valid: bool
    errors: tuple  # Tuple[ConfigValidationError, ...]
    scoped_config: Optional[ScopedConfig] = None


class IFConfigProvider(ABC):
    """
    Interface for configuration providers.

    DI - Scoped Sub-Config Injection.
    Processors receive configuration through this interface.

    Rule #9: Complete type hints.
    """

    @abstractmethod
    def get_config(self, scope: ConfigScope) -> ConfigValidationResult:
        """
        Get scoped configuration.

        Args:
            scope: The configuration scope definition.

        Returns:
            Validation result with scoped config if valid.
        """
        pass

    @abstractmethod
    def has_key(self, key: str, prefix: str = "") -> bool:
        """
        Check if a configuration key exists.

        Args:
            key: The key name.
            prefix: Optional prefix.

        Returns:
            True if the key exists.
        """
        pass


class DictConfigProvider(IFConfigProvider):
    """
    Dictionary-based configuration provider.

    DI - Scoped Sub-Config Injection.
    Simple implementation for testing and basic use cases.

    Rule #2: Bounded configuration size.
    Rule #7: Validates all inputs.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with a configuration dictionary.

        Args:
            config: The source configuration dictionary.

        Raises:
            ValueError: If config exceeds bounds.
        """
        if len(config) > MAX_CONFIG_KEYS * MAX_CONFIG_SCOPES:
            raise ValueError(
                f"Configuration exceeds maximum size: "
                f"{len(config)} > {MAX_CONFIG_KEYS * MAX_CONFIG_SCOPES}"
            )
        self._config = dict(config)  # Defensive copy

    def get_config(self, scope: ConfigScope) -> ConfigValidationResult:
        """
        Extract and validate scoped configuration.

        Rule #7: Explicit validation with detailed errors.

        Args:
            scope: The configuration scope definition.

        Returns:
            Validation result with scoped config if valid.
        """
        errors: List[ConfigValidationError] = []
        values: Dict[str, Any] = {}

        # Extract values for each defined key
        for key_def in scope.keys:
            full_key = f"{scope.prefix}{key_def.name}" if scope.prefix else key_def.name

            if full_key in self._config:
                value = self._config[full_key]
                try:
                    import json

                    serialized = json.dumps(value)
                    if len(serialized) > MAX_CONFIG_VALUE_SIZE:
                        errors.append(
                            ConfigValidationError(
                                scope_id=scope.scope_id,
                                key=key_def.name,
                                message=f"Value exceeds {MAX_CONFIG_VALUE_SIZE} bytes",
                                error_type="size_exceeded",
                            )
                        )
                        continue
                except (TypeError, ValueError):
                    errors.append(
                        ConfigValidationError(
                            scope_id=scope.scope_id,
                            key=key_def.name,
                            message="Value is not JSON-serializable",
                            error_type="invalid_value",
                        )
                    )
                    continue

                values[key_def.name] = value
            elif key_def.required:
                errors.append(
                    ConfigValidationError(
                        scope_id=scope.scope_id,
                        key=key_def.name,
                        message=f"Required key '{full_key}' not found",
                        error_type="missing_required",
                    )
                )
            else:
                # Use default for optional keys
                values[key_def.name] = key_def.default

        if errors:
            return ConfigValidationResult(
                valid=False, errors=tuple(errors), scoped_config=None
            )

        return ConfigValidationResult(
            valid=True,
            errors=(),
            scoped_config=ScopedConfig(scope_id=scope.scope_id, values=values),
        )

    def has_key(self, key: str, prefix: str = "") -> bool:
        """Check if a key exists in the configuration."""
        full_key = f"{prefix}{key}" if prefix else key
        return full_key in self._config


def config_scope(
    scope_id: str, keys: List[ConfigKey], prefix: str = ""
) -> Callable[[Type], Type]:
    """
    Decorator to declare configuration scope for a processor class.

    DI - Scoped Sub-Config Injection.
    Rule #9: Complete type hints.

    Args:
        scope_id: Unique scope identifier.
        keys: List of ConfigKey definitions.
        prefix: Optional key prefix.

    Returns:
        Decorated class with _config_scope attribute.

    Example:
        @config_scope("ocr", [
            ConfigKey("engine", required=True),
            ConfigKey("language", required=False, default="en"),
        ])
        class OCRProcessor(IFProcessor):
            ...
    """

    def decorator(cls: Type) -> Type:
        if len(keys) > MAX_CONFIG_KEYS:
            raise ValueError(
                f"ConfigScope for {cls.__name__} exceeds {MAX_CONFIG_KEYS} keys"
            )

        scope = ConfigScope(scope_id=scope_id, keys=frozenset(keys), prefix=prefix)
        cls._config_scope = scope
        return cls

    return decorator


def get_processor_scope(processor: Any) -> Optional[ConfigScope]:
    """
    Get the configuration scope for a processor.

    DI - Scoped Sub-Config Injection.
    Rule #7: Returns None if no scope defined.

    Args:
        processor: The processor instance or class.

    Returns:
        ConfigScope if defined, None otherwise.
    """
    return getattr(processor, "_config_scope", None)


def inject_config(processor: Any, provider: IFConfigProvider) -> ConfigValidationResult:
    """
    Inject configuration into a processor.

    DI - Scoped Sub-Config Injection.
    Rule #7: Returns explicit validation result.

    Args:
        processor: The processor to configure.
        provider: The configuration provider.

    Returns:
        Validation result with injected config.
    """
    scope = get_processor_scope(processor)

    if scope is None:
        # No scope defined - return empty valid config
        return ConfigValidationResult(
            valid=True,
            errors=(),
            scoped_config=ScopedConfig(scope_id="default", values={}),
        )

    result = provider.get_config(scope)

    if result.valid and result.scoped_config is not None:
        # Attach config to processor instance
        object.__setattr__(processor, "_injected_config", result.scoped_config)
        logger.debug(
            f"Injected config for {scope.scope_id}: "
            f"{result.scoped_config.key_count} keys"
        )

    return result


def get_injected_config(processor: Any) -> Optional[ScopedConfig]:
    """
    Get the injected configuration from a processor.

    DI - Scoped Sub-Config Injection.
    Rule #7: Returns None if no config injected.

    Args:
        processor: The processor instance.

    Returns:
        ScopedConfig if injected, None otherwise.
    """
    return getattr(processor, "_injected_config", None)
