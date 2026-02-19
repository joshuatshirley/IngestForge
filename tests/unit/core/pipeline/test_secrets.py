"""
Unit tests for Secrets - Dynamic Secret Propagation.

Tests follow GWT (Given-When-Then) behavioral specification format.
Validates NASA JPL Power of Ten rule compliance.
"""

import logging
import os
import pickle
import pytest
from unittest.mock import MagicMock, patch

from ingestforge.core.pipeline.secrets import (
    MAX_SECRET_KEYS,
    MAX_SECRET_VALUE_SIZE,
    SECRET_MASK,
    SecretKey,
    SecretScope,
    ScopedSecrets,
    SecretValidationError,
    SecretValidationResult,
    EnvSecretProvider,
    DictSecretProvider,
    secret_scope,
    get_processor_secret_scope,
    inject_secrets,
    get_injected_secrets,
    SecretMaskingFilter,
    create_masking_filter_for_secrets,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockProcessor:
    """Mock processor for testing secret injection."""

    pass


@secret_scope(
    "test_llm",
    [
        SecretKey("api_key", required=True, description="API key for LLM"),
        SecretKey("org_id", required=False, description="Organization ID"),
    ],
)
class LLMProcessor:
    """Mock LLM processor with secret requirements."""

    pass


@secret_scope(
    "test_storage",
    [
        SecretKey("access_key", required=True),
        SecretKey("secret_key", required=True),
        SecretKey("region", required=False),
    ],
)
class StorageProcessor:
    """Mock storage processor with secret requirements."""

    pass


# =============================================================================
# GWT Scenario 1: Secret Injection at Runtime
# =============================================================================


class TestSecretInjection:
    """Tests for GWT Scenario 1: Secret Injection at Runtime."""

    def test_given_processor_with_secrets_when_injected_then_secrets_available(self):
        """
        Given a processor registered with secret requirements,
        When the pipeline runner invokes the processor,
        Then secrets are injected from the secure provider.
        """
        # Given
        processor = LLMProcessor()
        provider = DictSecretProvider({"api_key": "sk-test-123"})

        # When
        result = inject_secrets(processor, provider)

        # Then
        assert result.valid is True
        secrets = get_injected_secrets(processor)
        assert secrets is not None
        assert secrets.get("api_key") == "sk-test-123"

    def test_given_processor_with_optional_secrets_when_missing_then_still_valid(self):
        """
        Given a processor with optional secret keys,
        When optional secrets are not provided,
        Then injection succeeds without errors.
        """
        # Given
        processor = LLMProcessor()
        provider = DictSecretProvider({"api_key": "sk-test-123"})  # org_id is optional

        # When
        result = inject_secrets(processor, provider)

        # Then
        assert result.valid is True
        secrets = get_injected_secrets(processor)
        assert secrets is not None
        assert secrets.get("api_key") == "sk-test-123"
        assert secrets.get("org_id") is None

    def test_given_processor_without_scope_when_injected_then_empty_valid_result(self):
        """
        Given a processor without secret scope defined,
        When secrets are injected,
        Then an empty valid result is returned.
        """
        # Given
        processor = MockProcessor()
        provider = DictSecretProvider({})

        # When
        result = inject_secrets(processor, provider)

        # Then
        assert result.valid is True
        assert result.scoped_secrets is not None
        assert result.scoped_secrets.key_count == 0


# =============================================================================
# GWT Scenario 2: Secrets Never Persisted in Artifacts
# =============================================================================


class TestSecretsNotInArtifacts:
    """Tests for GWT Scenario 2: Secrets Never Persisted in Artifacts."""

    def test_given_scoped_secrets_when_repr_then_no_values_exposed(self):
        """
        Given a processor with injected secrets,
        When the secrets repr is examined,
        Then no secret values appear.
        """
        # Given
        secrets = ScopedSecrets("test", {"api_key": "super-secret-value"})

        # When
        repr_output = repr(secrets)
        str_output = str(secrets)

        # Then
        assert "super-secret-value" not in repr_output
        assert "super-secret-value" not in str_output
        assert "key_count=1" in repr_output
        assert "1 keys" in str_output

    def test_given_scoped_secrets_when_pickle_then_raises_error(self):
        """
        Given a ScopedSecrets object,
        When serialization is attempted,
        Then a TypeError is raised.
        """
        # Given
        secrets = ScopedSecrets("test", {"api_key": "secret-123"})

        # When/Then
        with pytest.raises(TypeError, match="cannot be serialized"):
            pickle.dumps(secrets)


# =============================================================================
# GWT Scenario 3: Secrets Never Persisted in Checkpoints
# =============================================================================


class TestSecretsNotInCheckpoints:
    """Tests for GWT Scenario 3: Secrets Never Persisted in Checkpoints."""

    def test_given_scoped_secrets_when_getstate_then_raises_error(self):
        """
        Given a ScopedSecrets object,
        When __getstate__ is called (for pickling),
        Then a TypeError is raised.
        """
        # Given
        secrets = ScopedSecrets("test", {"password": "my-password"})

        # When/Then
        with pytest.raises(TypeError, match="cannot be serialized"):
            secrets.__getstate__()

    def test_given_scoped_secrets_when_reduce_then_raises_error(self):
        """
        Given a ScopedSecrets object,
        When __reduce__ is called (for pickling),
        Then a TypeError is raised.
        """
        # Given
        secrets = ScopedSecrets("test", {"token": "bearer-token"})

        # When/Then
        with pytest.raises(TypeError, match="cannot be serialized"):
            secrets.__reduce__()


# =============================================================================
# GWT Scenario 4: Secrets Masked in Logs
# =============================================================================


class TestSecretMasking:
    """Tests for GWT Scenario 4: Secrets Masked in Logs."""

    def test_given_secret_filter_when_log_contains_secret_then_masked(self):
        """
        Given a SecretMaskingFilter with a registered secret,
        When a log message contains the secret,
        Then the secret is masked in the output.
        """
        # Given
        filter_instance = SecretMaskingFilter()
        filter_instance.add_secret("my-secret-api-key")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Using API key: my-secret-api-key for request",
            args=(),
            exc_info=None,
        )

        # When
        filter_instance.filter(record)

        # Then
        assert "my-secret-api-key" not in record.msg
        assert SECRET_MASK in record.msg

    def test_given_secret_filter_when_log_args_contain_secret_then_masked(self):
        """
        Given a SecretMaskingFilter,
        When log arguments contain secrets,
        Then secrets in args are masked.
        """
        # Given
        filter_instance = SecretMaskingFilter()
        filter_instance.add_secret("secret-token-value")

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Token: %s",
            args=("secret-token-value",),
            exc_info=None,
        )

        # When
        filter_instance.filter(record)

        # Then
        assert "secret-token-value" not in record.args[0]
        assert SECRET_MASK in record.args[0]

    def test_given_scoped_secrets_when_filter_created_then_all_masked(self):
        """
        Given scoped secrets with multiple values,
        When a masking filter is created,
        Then all secret values are registered for masking.
        """
        # Given
        secrets = ScopedSecrets(
            "test",
            {
                "key1": "secret-value-1",
                "key2": "secret-value-2",
            },
        )

        # When
        filter_instance = create_masking_filter_for_secrets(secrets)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Secrets: secret-value-1 and secret-value-2",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)

        # Then
        assert "secret-value-1" not in record.msg
        assert "secret-value-2" not in record.msg
        assert record.msg.count(SECRET_MASK) == 2


# =============================================================================
# GWT Scenario 5: Missing Required Secret Validation
# =============================================================================


class TestMissingSecretValidation:
    """Tests for GWT Scenario 5: Missing Required Secret Validation."""

    def test_given_required_secret_when_missing_then_validation_error(self):
        """
        Given a processor requiring a secret key,
        When the secret is not available in the provider,
        Then a validation error is raised.
        """
        # Given
        processor = LLMProcessor()
        provider = DictSecretProvider({})  # No secrets provided

        # When
        result = inject_secrets(processor, provider)

        # Then
        assert result.valid is False
        assert len(result.errors) == 1
        error = result.errors[0]
        assert error.key == "api_key"
        assert error.error_type == "missing_required"

    def test_given_multiple_required_secrets_when_all_missing_then_multiple_errors(
        self,
    ):
        """
        Given a processor with multiple required secrets,
        When all are missing,
        Then validation errors for each are returned.
        """
        # Given
        processor = StorageProcessor()
        provider = DictSecretProvider({})

        # When
        result = inject_secrets(processor, provider)

        # Then
        assert result.valid is False
        assert len(result.errors) == 2  # access_key and secret_key
        error_keys = {e.key for e in result.errors}
        assert error_keys == {"access_key", "secret_key"}


# =============================================================================
# GWT Scenario 6: Secret Rotation Support
# =============================================================================


class TestSecretRotation:
    """Tests for GWT Scenario 6: Secret Rotation Support."""

    def test_given_injected_secrets_when_rotated_then_new_value_used(self):
        """
        Given an active pipeline with injected secrets,
        When the secret provider's values are updated,
        Then subsequent accesses receive the updated secrets.
        """
        # Given
        processor = LLMProcessor()
        provider = DictSecretProvider({"api_key": "old-key-123"})
        inject_secrets(processor, provider)
        secrets = get_injected_secrets(processor)

        # When
        success = provider.rotate_secret("test_llm", "api_key", "new-key-456")

        # Then
        assert success is True
        assert secrets.get("api_key") == "new-key-456"

    def test_given_secrets_when_rotate_nonexistent_key_then_fails(self):
        """
        Given scoped secrets,
        When rotating a key that doesn't exist,
        Then rotation fails.
        """
        # Given
        processor = LLMProcessor()
        provider = DictSecretProvider({"api_key": "key-123"})
        inject_secrets(processor, provider)

        # When
        success = provider.rotate_secret("test_llm", "nonexistent_key", "value")

        # Then
        assert success is False


# =============================================================================
# JPL Rule #2: Fixed Bounds
# =============================================================================


class TestJPLRule2FixedBounds:
    """Tests for JPL Rule #2: Fixed upper bounds."""

    def test_max_secret_keys_constant_defined(self):
        """Verify MAX_SECRET_KEYS constant is defined and reasonable."""
        assert MAX_SECRET_KEYS == 16
        assert MAX_SECRET_KEYS > 0
        assert MAX_SECRET_KEYS < 100  # Sanity check

    def test_given_too_many_keys_when_scope_created_then_error(self):
        """
        Given more than MAX_SECRET_KEYS,
        When a SecretScope is created,
        Then a ValueError is raised.
        """
        # Given
        keys = [SecretKey(f"key_{i}") for i in range(MAX_SECRET_KEYS + 1)]

        # When/Then
        with pytest.raises(ValueError, match=f"exceeds {MAX_SECRET_KEYS}"):
            SecretScope(scope_id="test", keys=frozenset(keys))

    def test_given_value_exceeds_size_when_injected_then_error(self):
        """
        Given a secret value exceeding MAX_SECRET_VALUE_SIZE,
        When secrets are injected,
        Then a validation error is returned.
        """
        # Given
        processor = LLMProcessor()
        large_value = "x" * (MAX_SECRET_VALUE_SIZE + 1)
        provider = DictSecretProvider({"api_key": large_value})

        # When
        result = inject_secrets(processor, provider)

        # Then
        assert result.valid is False
        assert any(e.error_type == "size_exceeded" for e in result.errors)


# =============================================================================
# JPL Rule #7: Explicit Return Values
# =============================================================================


class TestJPLRule7ReturnValues:
    """Tests for JPL Rule #7: Explicit return values."""

    def test_validation_result_has_explicit_fields(self):
        """Verify SecretValidationResult has explicit valid/errors fields."""
        result = SecretValidationResult(valid=True, errors=())
        assert hasattr(result, "valid")
        assert hasattr(result, "errors")
        assert hasattr(result, "scoped_secrets")

    def test_get_injected_secrets_returns_none_when_not_injected(self):
        """
        Given a processor without injected secrets,
        When get_injected_secrets is called,
        Then None is explicitly returned.
        """
        # Given
        processor = MockProcessor()

        # When
        result = get_injected_secrets(processor)

        # Then
        assert result is None


# =============================================================================
# JPL Rule #9: Type Hints
# =============================================================================


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_secret_key_has_type_hints(self):
        """Verify SecretKey class has type annotations."""
        annotations = SecretKey.__dataclass_fields__
        assert "name" in annotations
        assert "required" in annotations
        assert "description" in annotations

    def test_inject_secrets_returns_typed_result(self):
        """Verify inject_secrets returns properly typed result."""
        processor = MockProcessor()
        provider = DictSecretProvider({})
        result = inject_secrets(processor, provider)

        assert isinstance(result, SecretValidationResult)
        assert isinstance(result.valid, bool)
        assert isinstance(result.errors, tuple)


# =============================================================================
# SecretScope Tests
# =============================================================================


class TestSecretScope:
    """Tests for SecretScope class."""

    def test_required_keys_property(self):
        """Test required_keys returns only required keys."""
        scope = SecretScope(
            scope_id="test",
            keys=frozenset(
                [
                    SecretKey("required_1", required=True),
                    SecretKey("optional_1", required=False),
                ]
            ),
        )
        assert scope.required_keys == frozenset(["required_1"])

    def test_optional_keys_property(self):
        """Test optional_keys returns only optional keys."""
        scope = SecretScope(
            scope_id="test",
            keys=frozenset(
                [
                    SecretKey("required_1", required=True),
                    SecretKey("optional_1", required=False),
                ]
            ),
        )
        assert scope.optional_keys == frozenset(["optional_1"])

    def test_get_key_returns_matching_key(self):
        """Test get_key returns the matching SecretKey."""
        key = SecretKey("my_key", required=True, description="Test key")
        scope = SecretScope(scope_id="test", keys=frozenset([key]))

        result = scope.get_key("my_key")
        assert result == key

    def test_get_key_returns_none_for_missing(self):
        """Test get_key returns None for non-existent key."""
        scope = SecretScope(scope_id="test", keys=frozenset())
        assert scope.get_key("missing") is None


# =============================================================================
# ScopedSecrets Tests
# =============================================================================


class TestScopedSecrets:
    """Tests for ScopedSecrets class."""

    def test_contains_check(self):
        """Test __contains__ for key existence check."""
        secrets = ScopedSecrets("test", {"key1": "value1"})
        assert "key1" in secrets
        assert "key2" not in secrets

    def test_key_count_property(self):
        """Test key_count returns correct count."""
        secrets = ScopedSecrets("test", {"a": "1", "b": "2", "c": "3"})
        assert secrets.key_count == 3

    def test_get_with_default(self):
        """Test get returns default for missing key."""
        secrets = ScopedSecrets("test", {"key1": "value1"})
        assert secrets.get("missing", "default") == "default"

    def test_rotate_oversized_value_fails(self):
        """Test rotate fails for oversized values."""
        secrets = ScopedSecrets("test", {"key1": "value1"})
        large_value = "x" * (MAX_SECRET_VALUE_SIZE + 1)
        assert secrets.rotate("key1", large_value) is False


# =============================================================================
# EnvSecretProvider Tests
# =============================================================================


class TestEnvSecretProvider:
    """Tests for EnvSecretProvider class."""

    def test_reads_from_environment(self):
        """Test reading secrets from environment variables."""
        with patch.dict(os.environ, {"TEST_API_KEY": "env-secret-123"}):
            provider = EnvSecretProvider(prefix="TEST_")

            @secret_scope("env_test", [SecretKey("api_key", required=True)])
            class TestProcessor:
                pass

            processor = TestProcessor()
            result = inject_secrets(processor, provider)

            assert result.valid is True
            secrets = get_injected_secrets(processor)
            assert secrets.get("api_key") == "env-secret-123"

    def test_custom_env_var_override(self):
        """Test using custom env_var in SecretKey."""
        with patch.dict(os.environ, {"CUSTOM_VAR_NAME": "custom-value"}):
            provider = EnvSecretProvider()

            @secret_scope(
                "custom_test",
                [SecretKey("api_key", required=True, env_var="CUSTOM_VAR_NAME")],
            )
            class TestProcessor:
                pass

            processor = TestProcessor()
            result = inject_secrets(processor, provider)

            assert result.valid is True
            secrets = get_injected_secrets(processor)
            assert secrets.get("api_key") == "custom-value"

    def test_has_secret_checks_environment(self):
        """Test has_secret method checks environment."""
        with patch.dict(os.environ, {"PREFIX_MY_SECRET": "value"}):
            provider = EnvSecretProvider(prefix="PREFIX_")
            assert provider.has_secret("my_secret") is True
            assert provider.has_secret("nonexistent") is False


# =============================================================================
# secret_scope Decorator Tests
# =============================================================================


class TestSecretScopeDecorator:
    """Tests for @secret_scope decorator."""

    def test_decorator_attaches_scope(self):
        """Test decorator attaches _secret_scope to class."""

        @secret_scope("decorated", [SecretKey("key1", required=True)])
        class DecoratedProcessor:
            pass

        assert hasattr(DecoratedProcessor, "_secret_scope")
        scope = DecoratedProcessor._secret_scope
        assert scope.scope_id == "decorated"
        assert "key1" in scope.all_key_names

    def test_decorator_validates_key_count(self):
        """Test decorator rejects too many keys."""
        keys = [SecretKey(f"key_{i}") for i in range(MAX_SECRET_KEYS + 1)]

        with pytest.raises(ValueError, match=f"exceeds {MAX_SECRET_KEYS}"):

            @secret_scope("too_many", keys)
            class TooManyKeysProcessor:
                pass

    def test_decorator_with_prefix(self):
        """Test decorator with prefix parameter."""

        @secret_scope("prefixed", [SecretKey("key1")], prefix="my_")
        class PrefixedProcessor:
            pass

        scope = PrefixedProcessor._secret_scope
        assert scope.prefix == "my_"


# =============================================================================
# SecretMaskingFilter Tests
# =============================================================================


class TestSecretMaskingFilterEdgeCases:
    """Additional edge case tests for SecretMaskingFilter."""

    def test_filter_with_invalid_regex_pattern(self):
        """Test filter handles invalid regex patterns gracefully."""
        # Should not raise, just log warning
        filter_instance = SecretMaskingFilter(secret_patterns=["[invalid"])
        assert filter_instance is not None

    def test_filter_with_short_secret_ignored(self):
        """Test secrets shorter than 4 chars are ignored."""
        filter_instance = SecretMaskingFilter()
        filter_instance.add_secret("abc")  # Too short

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Secret: abc",
            args=(),
            exc_info=None,
        )
        filter_instance.filter(record)

        # Should still contain "abc" since it's too short to mask
        assert "abc" in record.msg

    def test_filter_always_returns_true(self):
        """Test filter always allows log records through."""
        filter_instance = SecretMaskingFilter()
        record = MagicMock(spec=logging.LogRecord)
        record.msg = "Test message"
        record.args = None

        result = filter_instance.filter(record)
        assert result is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecretInjectionIntegration:
    """Integration tests for secret injection workflow."""

    def test_full_workflow_with_rotation(self):
        """Test complete workflow: inject, use, rotate, verify."""
        # Setup
        processor = LLMProcessor()
        provider = DictSecretProvider({"api_key": "initial-key"})

        # Inject
        result = inject_secrets(processor, provider)
        assert result.valid is True

        # Use
        secrets = get_injected_secrets(processor)
        assert secrets.get("api_key") == "initial-key"

        # Rotate
        assert provider.rotate_secret("test_llm", "api_key", "rotated-key") is True

        # Verify
        assert secrets.get("api_key") == "rotated-key"

    def test_multiple_processors_isolated(self):
        """Test secrets are isolated between processors."""
        provider = DictSecretProvider(
            {
                "api_key": "llm-key",
                "access_key": "storage-access",
                "secret_key": "storage-secret",
            }
        )

        llm_processor = LLMProcessor()
        storage_processor = StorageProcessor()

        inject_secrets(llm_processor, provider)
        inject_secrets(storage_processor, provider)

        llm_secrets = get_injected_secrets(llm_processor)
        storage_secrets = get_injected_secrets(storage_processor)

        # LLM processor should only see api_key
        assert llm_secrets.get("api_key") == "llm-key"
        assert "access_key" not in llm_secrets

        # Storage processor should only see storage keys
        assert storage_secrets.get("access_key") == "storage-access"
        assert "api_key" not in storage_secrets


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestAdditionalCoverage:
    """Additional tests to improve code coverage."""

    def test_scoped_secrets_scope_id_property(self):
        """Test ScopedSecrets.scope_id property."""
        secrets = ScopedSecrets("my_scope", {"key": "value"})
        assert secrets.scope_id == "my_scope"

    def test_secret_scope_all_key_names(self):
        """Test SecretScope.all_key_names property."""
        scope = SecretScope(
            scope_id="test",
            keys=frozenset(
                [
                    SecretKey("key1", required=True),
                    SecretKey("key2", required=False),
                ]
            ),
        )
        assert scope.all_key_names == frozenset(["key1", "key2"])

    def test_env_provider_rotate_nonexistent_scope(self):
        """Test EnvSecretProvider.rotate_secret with nonexistent scope."""
        provider = EnvSecretProvider()
        result = provider.rotate_secret("nonexistent", "key", "value")
        assert result is False

    def test_dict_provider_has_secret_with_prefix(self):
        """Test DictSecretProvider.has_secret with prefix."""
        provider = DictSecretProvider({"prefix_key": "value"})
        assert provider.has_secret("key", prefix="prefix_") is True
        assert provider.has_secret("missing", prefix="prefix_") is False

    def test_get_processor_secret_scope_returns_scope(self):
        """Test get_processor_secret_scope returns attached scope."""
        processor = LLMProcessor()
        scope = get_processor_secret_scope(processor)
        assert scope is not None
        assert scope.scope_id == "test_llm"

    def test_get_processor_secret_scope_no_scope(self):
        """Test get_processor_secret_scope returns None when no scope."""
        processor = MockProcessor()
        scope = get_processor_secret_scope(processor)
        assert scope is None

    def test_secret_validation_error_fields(self):
        """Test SecretValidationError with all fields."""
        error = SecretValidationError(
            scope_id="test",
            key="api_key",
            message="Missing required secret",
            error_type="missing_required",
        )
        assert error.scope_id == "test"
        assert error.key == "api_key"
        assert error.message == "Missing required secret"
        assert error.error_type == "missing_required"

    def test_scoped_secrets_rotate_success_logs(self):
        """Test ScopedSecrets.rotate logs on success."""
        secrets = ScopedSecrets("test", {"key1": "value1"})
        result = secrets.rotate("key1", "new_value")
        assert result is True
        assert secrets.get("key1") == "new_value"

    def test_secret_scope_prefix_applied(self):
        """Test SecretScope with prefix."""
        scope = SecretScope(
            scope_id="prefixed", keys=frozenset([SecretKey("key1")]), prefix="app_"
        )
        assert scope.prefix == "app_"

    def test_dict_provider_rotate_success(self):
        """Test DictSecretProvider.rotate_secret success case."""
        provider = DictSecretProvider({"api_key": "old"})

        @secret_scope("rotate_test", [SecretKey("api_key")])
        class RotateProcessor:
            pass

        processor = RotateProcessor()
        inject_secrets(processor, provider)

        result = provider.rotate_secret("rotate_test", "api_key", "new")
        assert result is True

        secrets = get_injected_secrets(processor)
        assert secrets.get("api_key") == "new"
