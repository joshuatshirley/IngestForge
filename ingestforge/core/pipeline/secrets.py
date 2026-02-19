"""
Secure Secrets Injection for IngestForge (IF).

Secrets - Dynamic Secret Propagation.
Provides secure, scoped secret injection to processors.
Secrets are never persisted in artifacts, checkpoints, or logs.
Follows NASA JPL Power of Ten rules.
"""

import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    NoReturn,
    Optional,
    Set,
    Tuple,
    Type,
)

logger = logging.getLogger(__name__)

# JPL Rule #2: Fixed upper bounds for secrets
MAX_SECRET_KEYS = 16
MAX_SECRET_VALUE_SIZE = 4096  # 4KB per secret value
MAX_SECRET_SCOPES = 16

# Pattern for masking secrets in logs
SECRET_MASK = "****"


@dataclass(frozen=True)
class SecretKey:
    """
    Definition of a secret requirement.

    Secrets - Dynamic Secret Propagation.
    Rule #9: Complete type hints.

    Attributes:
        name: The secret key name.
        required: Whether this secret is required.
        description: Human-readable description.
        env_var: Optional environment variable name override.
    """

    name: str
    required: bool = True
    description: str = ""
    env_var: Optional[str] = None


@dataclass(frozen=True)
class SecretScope:
    """
    Defines the secrets scope for a processor.

    Secrets - Dynamic Secret Propagation.
    Rule #2: Bounded key count.
    Rule #9: Complete type hints.

    Attributes:
        scope_id: Unique identifier for this scope.
        keys: Set of required/optional secret keys.
        prefix: Optional prefix for namespacing (e.g., "llm_", "api_").
    """

    scope_id: str
    keys: FrozenSet[SecretKey] = field(default_factory=frozenset)
    prefix: str = ""

    def __post_init__(self) -> None:
        """Validate scope constraints."""
        if len(self.keys) > MAX_SECRET_KEYS:
            raise ValueError(
                f"SecretScope '{self.scope_id}' exceeds {MAX_SECRET_KEYS} keys"
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

    def get_key(self, name: str) -> Optional[SecretKey]:
        """Get a SecretKey by name."""
        for key in self.keys:
            if key.name == name:
                return key
        return None


class ScopedSecrets:
    """
    Runtime container for secrets (NOT frozen - intentionally mutable for rotation).

    Secrets - Dynamic Secret Propagation.
    Rule #6: Data in smallest scope.

    SECURITY: This class is NOT serializable and secrets cannot be extracted
    except through the get() method which is monitored.

    Attributes:
        scope_id: The scope these secrets belong to.
    """

    __slots__ = ("_scope_id", "_values", "_accessed_keys")

    def __init__(self, scope_id: str, values: Dict[str, str]) -> None:
        """
        Initialize scoped secrets.

        Args:
            scope_id: The scope identifier.
            values: Secret values (will be copied defensively).
        """
        self._scope_id = scope_id
        self._values: Dict[str, str] = dict(values)
        self._accessed_keys: Set[str] = set()

    @property
    def scope_id(self) -> str:
        """Get the scope ID."""
        return self._scope_id

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value.

        Rule #7: Explicit return (default if not found).

        Args:
            key: The secret key.
            default: Default value if key not found.

        Returns:
            The secret value or default.
        """
        self._accessed_keys.add(key)
        return self._values.get(key, default)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the secrets."""
        return key in self._values

    @property
    def key_count(self) -> int:
        """Get the number of keys in this scope."""
        return len(self._values)

    def rotate(self, key: str, new_value: str) -> bool:
        """
        Rotate a secret value.

        Args:
            key: The secret key to rotate.
            new_value: The new secret value.

        Returns:
            True if rotation succeeded, False if key doesn't exist.
        """
        if key not in self._values:
            return False
        if len(new_value) > MAX_SECRET_VALUE_SIZE:
            return False
        self._values[key] = new_value
        logger.debug(f"Secret rotated: {self._scope_id}.{key}")
        return True

    def __repr__(self) -> str:
        """Safe repr that never exposes secrets."""
        return f"ScopedSecrets(scope_id={self._scope_id!r}, key_count={self.key_count})"

    def __str__(self) -> str:
        """Safe str that never exposes secrets."""
        return f"ScopedSecrets({self._scope_id}, {self.key_count} keys)"

    # SECURITY: Prevent serialization
    def __getstate__(self) -> NoReturn:
        """Prevent pickling of secrets."""
        raise TypeError("ScopedSecrets cannot be serialized")

    def __reduce__(self) -> NoReturn:
        """Prevent pickling of secrets."""
        raise TypeError("ScopedSecrets cannot be serialized")


@dataclass(frozen=True)
class SecretValidationError:
    """
    Represents a secret validation error.

    Secrets - Dynamic Secret Propagation.
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
class SecretValidationResult:
    """
    Result of secret validation.

    Secrets - Dynamic Secret Propagation.
    Rule #7: Explicit return values.

    Attributes:
        valid: True if validation passed.
        errors: List of validation errors.
        scoped_secrets: The validated secrets (if valid).
    """

    valid: bool
    errors: Tuple[SecretValidationError, ...]
    scoped_secrets: Optional[ScopedSecrets] = None


class IFSecretProvider(ABC):
    """
    Interface for secret providers.

    Secrets - Dynamic Secret Propagation.
    Processors receive secrets through this interface.

    Rule #9: Complete type hints.
    """

    @abstractmethod
    def get_secrets(self, scope: SecretScope) -> SecretValidationResult:
        """
        Get scoped secrets.

        Args:
            scope: The secret scope definition.

        Returns:
            Validation result with scoped secrets if valid.
        """
        pass

    @abstractmethod
    def has_secret(self, key: str, prefix: str = "") -> bool:
        """
        Check if a secret exists.

        Args:
            key: The key name.
            prefix: Optional prefix.

        Returns:
            True if the secret exists.
        """
        pass

    @abstractmethod
    def rotate_secret(self, scope_id: str, key: str, new_value: str) -> bool:
        """
        Rotate a secret value.

        Args:
            scope_id: The scope identifier.
            key: The secret key.
            new_value: The new secret value.

        Returns:
            True if rotation succeeded.
        """
        pass


def _build_validation_result(
    scope_id: str, errors: List[SecretValidationError], values: Dict[str, str]
) -> Tuple[SecretValidationResult, Optional[ScopedSecrets]]:
    """
    Build validation result tuple.

    Helper function to reduce duplication (JPL Rule #4).

    Returns:
        Tuple of (SecretValidationResult, Optional[ScopedSecrets]).
    """
    if errors:
        return SecretValidationResult(
            valid=False, errors=tuple(errors), scoped_secrets=None
        ), None

    scoped_secrets = ScopedSecrets(scope_id=scope_id, values=values)
    return SecretValidationResult(
        valid=True, errors=(), scoped_secrets=scoped_secrets
    ), scoped_secrets


class EnvSecretProvider(IFSecretProvider):
    """
    Environment variable-based secret provider.

    Secrets - Dynamic Secret Propagation.
    Reads secrets from environment variables.

    Rule #2: Bounded secret size.
    Rule #7: Validates all inputs.
    """

    def __init__(self, prefix: str = "") -> None:
        """
        Initialize with optional environment variable prefix.

        Args:
            prefix: Prefix for environment variables (e.g., "INGESTFORGE_").
        """
        self._prefix = prefix
        self._injected_scopes: Dict[str, ScopedSecrets] = {}

    def _get_env_key(self, key_def: SecretKey, scope: SecretScope) -> str:
        """Get environment variable name for a secret key."""
        if key_def.env_var:
            return key_def.env_var
        full_key = f"{scope.prefix}{key_def.name}" if scope.prefix else key_def.name
        return f"{self._prefix}{full_key}".upper()

    def get_secrets(self, scope: SecretScope) -> SecretValidationResult:
        """Extract and validate secrets from environment."""
        errors: List[SecretValidationError] = []
        values: Dict[str, str] = {}

        for key_def in scope.keys:
            env_key = self._get_env_key(key_def, scope)
            value = os.environ.get(env_key)

            if value is not None:
                if len(value) > MAX_SECRET_VALUE_SIZE:
                    errors.append(
                        SecretValidationError(
                            scope_id=scope.scope_id,
                            key=key_def.name,
                            message=f"Secret exceeds {MAX_SECRET_VALUE_SIZE} bytes",
                            error_type="size_exceeded",
                        )
                    )
                else:
                    values[key_def.name] = value
            elif key_def.required:
                errors.append(
                    SecretValidationError(
                        scope_id=scope.scope_id,
                        key=key_def.name,
                        message=f"Required secret '{env_key}' not found in environment",
                        error_type="missing_required",
                    )
                )

        result, scoped_secrets = _build_validation_result(
            scope.scope_id, errors, values
        )
        if scoped_secrets:
            self._injected_scopes[scope.scope_id] = scoped_secrets
        return result

    def has_secret(self, key: str, prefix: str = "") -> bool:
        """Check if a secret exists in environment."""
        full_key = f"{prefix}{key}" if prefix else key
        env_key = f"{self._prefix}{full_key}".upper()
        return env_key in os.environ

    def rotate_secret(self, scope_id: str, key: str, new_value: str) -> bool:
        """
        Rotate a secret in an injected scope.

        Args:
            scope_id: The scope identifier.
            key: The secret key.
            new_value: The new secret value.

        Returns:
            True if rotation succeeded.
        """
        if scope_id not in self._injected_scopes:
            return False
        return self._injected_scopes[scope_id].rotate(key, new_value)


class DictSecretProvider(IFSecretProvider):
    """
    Dictionary-based secret provider for testing.

    Secrets - Dynamic Secret Propagation.
    Simple implementation for testing purposes.

    WARNING: Not for production use - secrets in memory.
    """

    def __init__(self, secrets: Dict[str, str]) -> None:
        """
        Initialize with a secrets dictionary.

        Args:
            secrets: The source secrets dictionary.
        """
        self._secrets: Dict[str, str] = dict(secrets)
        self._injected_scopes: Dict[str, ScopedSecrets] = {}

    def get_secrets(self, scope: SecretScope) -> SecretValidationResult:
        """Extract and validate secrets from dictionary."""
        errors: List[SecretValidationError] = []
        values: Dict[str, str] = {}

        for key_def in scope.keys:
            full_key = f"{scope.prefix}{key_def.name}" if scope.prefix else key_def.name

            if full_key in self._secrets:
                value = self._secrets[full_key]
                if len(value) > MAX_SECRET_VALUE_SIZE:
                    errors.append(
                        SecretValidationError(
                            scope_id=scope.scope_id,
                            key=key_def.name,
                            message=f"Secret exceeds {MAX_SECRET_VALUE_SIZE} bytes",
                            error_type="size_exceeded",
                        )
                    )
                else:
                    values[key_def.name] = value
            elif key_def.required:
                errors.append(
                    SecretValidationError(
                        scope_id=scope.scope_id,
                        key=key_def.name,
                        message=f"Required secret '{full_key}' not found",
                        error_type="missing_required",
                    )
                )

        result, scoped_secrets = _build_validation_result(
            scope.scope_id, errors, values
        )
        if scoped_secrets:
            self._injected_scopes[scope.scope_id] = scoped_secrets
        return result

    def has_secret(self, key: str, prefix: str = "") -> bool:
        """Check if a secret exists."""
        full_key = f"{prefix}{key}" if prefix else key
        return full_key in self._secrets

    def rotate_secret(self, scope_id: str, key: str, new_value: str) -> bool:
        """Rotate a secret in an injected scope."""
        if scope_id not in self._injected_scopes:
            return False
        return self._injected_scopes[scope_id].rotate(key, new_value)


def secret_scope(
    scope_id: str, keys: List[SecretKey], prefix: str = ""
) -> Callable[[Type[Any]], Type[Any]]:
    """
    Decorator to declare secret scope for a processor class.

    Secrets - Dynamic Secret Propagation.
    Rule #9: Complete type hints.

    Args:
        scope_id: Unique scope identifier.
        keys: List of SecretKey definitions.
        prefix: Optional key prefix.

    Returns:
        Decorated class with _secret_scope attribute.

    Example:
        @secret_scope("llm", [
            SecretKey("api_key", required=True),
            SecretKey("org_id", required=False),
        ])
        class LLMProcessor(IFProcessor):
            ...
    """

    def decorator(cls: Type[Any]) -> Type[Any]:
        if len(keys) > MAX_SECRET_KEYS:
            raise ValueError(
                f"SecretScope for {cls.__name__} exceeds {MAX_SECRET_KEYS} keys"
            )

        scope = SecretScope(scope_id=scope_id, keys=frozenset(keys), prefix=prefix)
        cls._secret_scope = scope
        return cls

    return decorator


def get_processor_secret_scope(processor: Any) -> Optional[SecretScope]:
    """
    Get the secret scope for a processor.

    Secrets - Dynamic Secret Propagation.
    Rule #7: Returns None if no scope defined.

    Args:
        processor: The processor instance or class.

    Returns:
        SecretScope if defined, None otherwise.
    """
    return getattr(processor, "_secret_scope", None)


def inject_secrets(
    processor: Any, provider: IFSecretProvider
) -> SecretValidationResult:
    """
    Inject secrets into a processor.

    Secrets - Dynamic Secret Propagation.
    Rule #7: Returns explicit validation result.

    Args:
        processor: The processor to configure.
        provider: The secret provider.

    Returns:
        Validation result with injected secrets.
    """
    scope = get_processor_secret_scope(processor)

    if scope is None:
        # No scope defined - return empty valid result
        return SecretValidationResult(
            valid=True,
            errors=(),
            scoped_secrets=ScopedSecrets(scope_id="default", values={}),
        )

    result = provider.get_secrets(scope)

    if result.valid and result.scoped_secrets is not None:
        # Attach secrets to processor instance
        object.__setattr__(processor, "_injected_secrets", result.scoped_secrets)
        logger.debug(
            f"Injected secrets for {scope.scope_id}: "
            f"{result.scoped_secrets.key_count} keys"
        )

    return result


def get_injected_secrets(processor: Any) -> Optional[ScopedSecrets]:
    """
    Get the injected secrets from a processor.

    Secrets - Dynamic Secret Propagation.
    Rule #7: Returns None if no secrets injected.

    Args:
        processor: The processor instance.

    Returns:
        ScopedSecrets if injected, None otherwise.
    """
    return getattr(processor, "_injected_secrets", None)


class SecretMaskingFilter(logging.Filter):
    """
    Logging filter that masks secret values.

    Secrets - Dynamic Secret Propagation.
    Prevents accidental secret exposure in logs.
    """

    def __init__(self, secret_patterns: Optional[List[str]] = None) -> None:
        """
        Initialize with optional secret patterns.

        Args:
            secret_patterns: List of regex patterns to mask.
        """
        super().__init__()
        self._patterns: List[re.Pattern[str]] = []
        if secret_patterns:
            for pattern in secret_patterns:
                try:
                    self._patterns.append(re.compile(pattern))
                except re.error:
                    logger.warning(f"Invalid secret pattern: {pattern}")

    def add_secret(self, secret_value: str) -> None:
        """
        Add a secret value to be masked.

        Args:
            secret_value: The secret to mask in logs.
        """
        if secret_value and len(secret_value) >= 4:
            # Escape regex special characters
            escaped = re.escape(secret_value)
            self._patterns.append(re.compile(escaped))

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Filter log record by masking secrets.

        Args:
            record: The log record.

        Returns:
            True (always allow, but mask secrets).
        """
        if hasattr(record, "msg") and isinstance(record.msg, str):
            msg = record.msg
            for pattern in self._patterns:
                msg = pattern.sub(SECRET_MASK, msg)
            record.msg = msg

        if hasattr(record, "args") and record.args:
            masked_args: List[Any] = []
            for arg in record.args:
                if isinstance(arg, str):
                    masked = arg
                    for pattern in self._patterns:
                        masked = pattern.sub(SECRET_MASK, masked)
                    masked_args.append(masked)
                else:
                    masked_args.append(arg)
            record.args = tuple(masked_args)

        return True


def create_masking_filter_for_secrets(
    scoped_secrets: ScopedSecrets,
) -> SecretMaskingFilter:
    """
    Create a log filter that masks all secrets in a scope.

    Args:
        scoped_secrets: The secrets to mask.

    Returns:
        Configured SecretMaskingFilter.
    """
    filter_instance = SecretMaskingFilter()
    # Access internal values safely
    for key in scoped_secrets._values:
        value = scoped_secrets._values[key]
        filter_instance.add_secret(value)
    return filter_instance
