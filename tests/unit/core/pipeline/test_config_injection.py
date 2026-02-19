"""
Unit tests for DI - Scoped Sub-Config Injection.

Tests follow GWT (Given-When-Then) format and verify NASA JPL Power of Ten compliance.
"""

import pytest
from ingestforge.core.pipeline.config_injection import (
    ConfigKey,
    ConfigScope,
    ScopedConfig,
    IFConfigProvider,
    DictConfigProvider,
    config_scope,
    get_processor_scope,
    inject_config,
    get_injected_config,
    MAX_CONFIG_KEYS,
    MAX_CONFIG_VALUE_SIZE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockProcessor:
    """Simple mock processor for testing."""

    pass


@config_scope(
    "test-scope",
    [
        ConfigKey("api_key", required=True, description="API key"),
        ConfigKey(
            "timeout", required=False, default=30, description="Timeout in seconds"
        ),
    ],
)
class DecoratedProcessor:
    """Processor with config scope decorator."""

    pass


# =============================================================================
# GWT Scenario 1: Processor Receives Scoped Config
# =============================================================================


class TestProcessorReceivesScopedConfig:
    """Tests for Scenario 1: Processor receives scoped config."""

    def test_processor_receives_only_requested_keys(self):
        """
        GWT:
        Given a processor registered with specific config requirements,
        When the pipeline runner invokes the processor,
        Then only the relevant config subset is injected.
        """
        scope = ConfigScope(
            scope_id="ocr",
            keys=frozenset(
                [
                    ConfigKey("engine", required=True),
                    ConfigKey("language", required=False, default="en"),
                ]
            ),
        )

        provider = DictConfigProvider(
            {
                "engine": "tesseract",
                "language": "de",
                "unrelated_key": "should_not_appear",
                "another_key": 12345,
            }
        )

        result = provider.get_config(scope)

        assert result.valid is True
        assert result.scoped_config is not None
        assert result.scoped_config.get("engine") == "tesseract"
        assert result.scoped_config.get("language") == "de"
        assert "unrelated_key" not in result.scoped_config.values
        assert "another_key" not in result.scoped_config.values

    def test_scoped_config_uses_prefix(self):
        """
        GWT:
        Given a config scope with prefix "ocr.",
        When extracting config,
        Then keys are looked up with the prefix.
        """
        scope = ConfigScope(
            scope_id="ocr",
            keys=frozenset(
                [
                    ConfigKey("engine", required=True),
                ]
            ),
            prefix="ocr.",
        )

        provider = DictConfigProvider(
            {
                "ocr.engine": "easyocr",
                "engine": "wrong_engine",  # Should not be used
            }
        )

        result = provider.get_config(scope)

        assert result.valid is True
        assert result.scoped_config.get("engine") == "easyocr"


# =============================================================================
# GWT Scenario 2: Config Isolation Between Processors
# =============================================================================


class TestConfigIsolation:
    """Tests for Scenario 2: Config isolation between processors."""

    def test_two_processors_get_different_scopes(self):
        """
        GWT:
        Given two processors with different config scopes,
        When each processor accesses its config,
        Then neither can see the other's configuration.
        """
        scope_a = ConfigScope(
            scope_id="processor_a", keys=frozenset([ConfigKey("key_a", required=True)])
        )
        scope_b = ConfigScope(
            scope_id="processor_b", keys=frozenset([ConfigKey("key_b", required=True)])
        )

        provider = DictConfigProvider(
            {
                "key_a": "value_a",
                "key_b": "value_b",
            }
        )

        result_a = provider.get_config(scope_a)
        result_b = provider.get_config(scope_b)

        # Processor A only sees key_a
        assert result_a.scoped_config.get("key_a") == "value_a"
        assert "key_b" not in result_a.scoped_config.values

        # Processor B only sees key_b
        assert result_b.scoped_config.get("key_b") == "value_b"
        assert "key_a" not in result_b.scoped_config.values

    def test_prefixed_scopes_are_isolated(self):
        """
        GWT:
        Given two processors with prefixed scopes,
        When each extracts config,
        Then they only see their prefixed keys.
        """
        scope_ocr = ConfigScope(
            scope_id="ocr",
            keys=frozenset([ConfigKey("model", required=True)]),
            prefix="ocr.",
        )
        scope_llm = ConfigScope(
            scope_id="llm",
            keys=frozenset([ConfigKey("model", required=True)]),
            prefix="llm.",
        )

        provider = DictConfigProvider(
            {
                "ocr.model": "tesseract",
                "llm.model": "gpt-4",
            }
        )

        ocr_result = provider.get_config(scope_ocr)
        llm_result = provider.get_config(scope_llm)

        assert ocr_result.scoped_config.get("model") == "tesseract"
        assert llm_result.scoped_config.get("model") == "gpt-4"


# =============================================================================
# GWT Scenario 3: Missing Config Validation
# =============================================================================


class TestMissingConfigValidation:
    """Tests for Scenario 3: Missing config validation."""

    def test_missing_required_key_fails_validation(self):
        """
        GWT:
        Given a processor requiring specific config keys,
        When required keys are missing,
        Then a validation error is raised before execution.
        """
        scope = ConfigScope(
            scope_id="test",
            keys=frozenset(
                [
                    ConfigKey("required_key", required=True),
                ]
            ),
        )

        provider = DictConfigProvider(
            {
                "other_key": "value",
            }
        )

        result = provider.get_config(scope)

        assert result.valid is False
        assert len(result.errors) == 1
        assert result.errors[0].error_type == "missing_required"
        assert "required_key" in result.errors[0].message

    def test_multiple_missing_keys_all_reported(self):
        """
        GWT:
        Given multiple required keys missing,
        When validation runs,
        Then all missing keys are reported.
        """
        scope = ConfigScope(
            scope_id="test",
            keys=frozenset(
                [
                    ConfigKey("key1", required=True),
                    ConfigKey("key2", required=True),
                    ConfigKey("key3", required=True),
                ]
            ),
        )

        provider = DictConfigProvider({})

        result = provider.get_config(scope)

        assert result.valid is False
        assert len(result.errors) == 3
        missing_keys = {e.key for e in result.errors}
        assert missing_keys == {"key1", "key2", "key3"}


# =============================================================================
# GWT Scenario 4: Config Immutability
# =============================================================================


class TestConfigImmutability:
    """Tests for Scenario 4: Config immutability."""

    def test_scoped_config_is_frozen(self):
        """
        GWT:
        Given an injected config object,
        When a processor attempts to modify its attributes,
        Then the modification is rejected (frozen dataclass).
        """
        config = ScopedConfig(scope_id="test", values={"key": "value"})

        # Frozen dataclass prevents attribute reassignment
        with pytest.raises(Exception):  # FrozenInstanceError
            config.scope_id = "modified"

        # Note: dict contents can still be mutated (shallow freeze)
        # but this is acceptable as the config is meant to be read-only by convention

    def test_config_scope_is_frozen(self):
        """
        GWT:
        Given a ConfigScope,
        When attempting to modify it,
        Then modification is rejected.
        """
        scope = ConfigScope(
            scope_id="test", keys=frozenset([ConfigKey("key", required=True)])
        )

        with pytest.raises(Exception):  # FrozenInstanceError
            scope.scope_id = "modified"

    def test_config_key_is_frozen(self):
        """
        GWT:
        Given a ConfigKey,
        When attempting to modify it,
        Then modification is rejected.
        """
        key = ConfigKey("test", required=True, default=None)

        with pytest.raises(Exception):  # FrozenInstanceError
            key.name = "modified"


# =============================================================================
# GWT Scenario 5: Default Config Values
# =============================================================================


class TestDefaultConfigValues:
    """Tests for Scenario 5: Default config values."""

    def test_optional_keys_use_defaults(self):
        """
        GWT:
        Given a processor with optional config keys,
        When optional keys are not provided,
        Then default values are used.
        """
        scope = ConfigScope(
            scope_id="test",
            keys=frozenset(
                [
                    ConfigKey("required", required=True),
                    ConfigKey("optional_int", required=False, default=42),
                    ConfigKey("optional_str", required=False, default="default"),
                ]
            ),
        )

        provider = DictConfigProvider(
            {
                "required": "provided",
            }
        )

        result = provider.get_config(scope)

        assert result.valid is True
        assert result.scoped_config.get("required") == "provided"
        assert result.scoped_config.get("optional_int") == 42
        assert result.scoped_config.get("optional_str") == "default"

    def test_provided_optional_overrides_default(self):
        """
        GWT:
        Given an optional key with default,
        When the key is provided,
        Then the provided value is used.
        """
        scope = ConfigScope(
            scope_id="test",
            keys=frozenset(
                [
                    ConfigKey("timeout", required=False, default=30),
                ]
            ),
        )

        provider = DictConfigProvider(
            {
                "timeout": 60,
            }
        )

        result = provider.get_config(scope)

        assert result.scoped_config.get("timeout") == 60


# =============================================================================
# Decorator Tests
# =============================================================================


class TestConfigScopeDecorator:
    """Tests for the @config_scope decorator."""

    def test_decorator_attaches_scope(self):
        """
        GWT:
        Given a class decorated with @config_scope,
        When inspecting the class,
        Then it has _config_scope attribute.
        """
        scope = get_processor_scope(DecoratedProcessor)

        assert scope is not None
        assert scope.scope_id == "test-scope"
        assert "api_key" in scope.all_key_names
        assert "timeout" in scope.all_key_names

    def test_undecorated_processor_has_no_scope(self):
        """
        GWT:
        Given an undecorated class,
        When inspecting it,
        Then get_processor_scope returns None.
        """
        scope = get_processor_scope(MockProcessor)

        assert scope is None

    def test_inject_config_with_decorated_processor(self):
        """
        GWT:
        Given a decorated processor,
        When inject_config is called,
        Then config is attached to the instance.
        """
        processor = DecoratedProcessor()
        provider = DictConfigProvider(
            {
                "api_key": "secret123",
                "timeout": 45,
            }
        )

        result = inject_config(processor, provider)

        assert result.valid is True
        injected = get_injected_config(processor)
        assert injected is not None
        assert injected.get("api_key") == "secret123"
        assert injected.get("timeout") == 45


# =============================================================================
# JPL Power of Ten Compliance Tests
# =============================================================================


class TestJPLRule2FixedBounds:
    """Tests for JPL Rule #2: Fixed upper bounds."""

    def test_max_config_keys_constant_exists(self):
        """
        GWT:
        Given the config_injection module,
        When importing,
        Then MAX_CONFIG_KEYS constant is defined.
        """
        assert isinstance(MAX_CONFIG_KEYS, int)
        assert MAX_CONFIG_KEYS > 0

    def test_config_scope_rejects_too_many_keys(self):
        """
        GWT:
        Given a scope with more than MAX_CONFIG_KEYS,
        When creating the scope,
        Then ValueError is raised.
        """
        too_many_keys = frozenset(
            [ConfigKey(f"key_{i}", required=False) for i in range(MAX_CONFIG_KEYS + 1)]
        )

        with pytest.raises(ValueError) as exc_info:
            ConfigScope(scope_id="test", keys=too_many_keys)

        assert str(MAX_CONFIG_KEYS) in str(exc_info.value)

    def test_decorator_rejects_too_many_keys(self):
        """
        GWT:
        Given @config_scope with too many keys,
        When decorating a class,
        Then ValueError is raised.
        """
        too_many_keys = [
            ConfigKey(f"key_{i}", required=False) for i in range(MAX_CONFIG_KEYS + 1)
        ]

        with pytest.raises(ValueError):

            @config_scope("test", too_many_keys)
            class TooManyKeysProcessor:
                pass

    def test_value_size_limit_enforced(self):
        """
        GWT:
        Given a config value exceeding MAX_CONFIG_VALUE_SIZE,
        When validating config,
        Then error is reported.
        """
        scope = ConfigScope(
            scope_id="test", keys=frozenset([ConfigKey("large_value", required=True)])
        )

        # Create value larger than limit
        large_value = "x" * (MAX_CONFIG_VALUE_SIZE + 100)
        provider = DictConfigProvider({"large_value": large_value})

        result = provider.get_config(scope)

        assert result.valid is False
        assert result.errors[0].error_type == "size_exceeded"


class TestJPLRule7ReturnValues:
    """Tests for JPL Rule #7: Check return values."""

    def test_validation_result_always_has_valid_field(self):
        """
        GWT:
        Given any validation operation,
        When completed,
        Then result always has explicit valid field.
        """
        scope = ConfigScope(
            scope_id="test", keys=frozenset([ConfigKey("key", required=True)])
        )

        provider = DictConfigProvider({"key": "value"})
        result = provider.get_config(scope)
        assert hasattr(result, "valid")
        assert isinstance(result.valid, bool)

        empty_provider = DictConfigProvider({})
        fail_result = empty_provider.get_config(scope)
        assert hasattr(fail_result, "valid")
        assert isinstance(fail_result.valid, bool)

    def test_errors_tuple_always_present(self):
        """
        GWT:
        Given a validation result,
        When accessing errors,
        Then it's always a tuple (never None).
        """
        scope = ConfigScope(
            scope_id="test",
            keys=frozenset([ConfigKey("key", required=False, default=1)]),
        )

        provider = DictConfigProvider({})
        result = provider.get_config(scope)

        assert result.errors is not None
        assert isinstance(result.errors, tuple)

    def test_scoped_config_get_returns_default(self):
        """
        GWT:
        Given a ScopedConfig,
        When get() is called with missing key,
        Then default is returned (never raises).
        """
        config = ScopedConfig(scope_id="test", values={"a": 1})

        result = config.get("nonexistent", "default_value")

        assert result == "default_value"


class TestJPLRule9TypeHints:
    """Tests for JPL Rule #9: Complete type hints."""

    def test_config_key_has_type_hints(self):
        """
        GWT:
        Given ConfigKey class,
        When inspecting type hints,
        Then all fields are annotated.
        """
        import typing

        hints = typing.get_type_hints(ConfigKey)

        assert "name" in hints
        assert "required" in hints
        assert "default" in hints
        assert "description" in hints

    def test_scoped_config_has_type_hints(self):
        """
        GWT:
        Given ScopedConfig class,
        When inspecting type hints,
        Then all fields are annotated.
        """
        import typing

        hints = typing.get_type_hints(ScopedConfig)

        assert "scope_id" in hints
        assert "values" in hints

    def test_config_provider_has_type_hints(self):
        """
        GWT:
        Given IFConfigProvider methods,
        When inspecting type hints,
        Then return types are annotated.
        """
        import typing

        hints = typing.get_type_hints(IFConfigProvider.get_config)

        assert "return" in hints
        assert "scope" in hints


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_scope_is_valid(self):
        """
        GWT:
        Given a scope with no keys,
        When validating,
        Then it succeeds with empty config.
        """
        scope = ConfigScope(scope_id="empty", keys=frozenset())
        provider = DictConfigProvider({"irrelevant": "value"})

        result = provider.get_config(scope)

        assert result.valid is True
        assert result.scoped_config.key_count == 0

    def test_non_serializable_value_fails(self):
        """
        GWT:
        Given a config value that's not JSON-serializable,
        When validating,
        Then error is reported.
        """
        scope = ConfigScope(
            scope_id="test", keys=frozenset([ConfigKey("func", required=True)])
        )

        # Functions are not JSON-serializable
        provider = DictConfigProvider({"func": lambda x: x})

        result = provider.get_config(scope)

        assert result.valid is False
        assert result.errors[0].error_type == "invalid_value"

    def test_inject_config_without_scope_succeeds(self):
        """
        GWT:
        Given a processor without scope decorator,
        When inject_config is called,
        Then it succeeds with empty config.
        """
        processor = MockProcessor()
        provider = DictConfigProvider({"key": "value"})

        result = inject_config(processor, provider)

        assert result.valid is True
        assert result.scoped_config.scope_id == "default"

    def test_config_scope_required_keys_property(self):
        """
        GWT:
        Given a scope with mixed required/optional keys,
        When accessing required_keys,
        Then only required key names are returned.
        """
        scope = ConfigScope(
            scope_id="test",
            keys=frozenset(
                [
                    ConfigKey("req1", required=True),
                    ConfigKey("req2", required=True),
                    ConfigKey("opt1", required=False),
                ]
            ),
        )

        assert scope.required_keys == frozenset(["req1", "req2"])
        assert scope.optional_keys == frozenset(["opt1"])

    def test_has_key_with_prefix(self):
        """
        GWT:
        Given a config provider,
        When checking has_key with prefix,
        Then correct result is returned.
        """
        provider = DictConfigProvider(
            {
                "prefix.key": "value",
                "other_key": "value2",
            }
        )

        assert provider.has_key("key", prefix="prefix.") is True
        assert provider.has_key("key", prefix="") is False
        assert provider.has_key("other_key") is True
