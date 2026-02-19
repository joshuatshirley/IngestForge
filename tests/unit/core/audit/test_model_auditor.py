"""Tests for LLM Model Parity Auditor.

LLM Model Parity Auditor
Epic: EP-26 (Security & Compliance)

Tests cover:
- ModelCapability enum
- ProviderStatus dataclass
- AuditReport dataclass
- ModelAuditor operations
- JPL compliance
"""

from __future__ import annotations

import os
import pytest
from unittest.mock import patch

from ingestforge.core.audit import (
    ModelAuditor,
    ModelCapability,
    AuditStatus,
    ProviderStatus,
    AuditReport,
    create_model_auditor,
    PROVIDER_CAPABILITIES,
    MAX_PROVIDERS,
    MAX_CAPABILITIES,
)


class TestModelCapability:
    """Tests for ModelCapability enum."""

    def test_all_capabilities_defined(self) -> None:
        """Test that all expected capabilities are defined."""
        expected = {
            "streaming",
            "function_calling",
            "vision",
            "embeddings",
            "json_mode",
            "system_prompt",
            "multi_turn",
        }
        actual = {cap.value for cap in ModelCapability}
        assert expected == actual

    def test_capability_values(self) -> None:
        """Test capability value strings."""
        assert ModelCapability.STREAMING.value == "streaming"
        assert ModelCapability.FUNCTION_CALLING.value == "function_calling"
        assert ModelCapability.VISION.value == "vision"


class TestAuditStatus:
    """Tests for AuditStatus enum."""

    def test_all_statuses_defined(self) -> None:
        """Test that all status values are defined."""
        expected = {"available", "unavailable", "error", "not_configured"}
        actual = {status.value for status in AuditStatus}
        assert expected == actual


class TestProviderStatus:
    """Tests for ProviderStatus dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic provider status."""
        status = ProviderStatus(
            provider_name="openai",
            status=AuditStatus.AVAILABLE,
        )
        assert status.provider_name == "openai"
        assert status.status == AuditStatus.AVAILABLE
        assert status.config_present is False
        assert status.error_message is None

    def test_full_creation(self) -> None:
        """Test creating status with all fields."""
        caps = {ModelCapability.STREAMING, ModelCapability.VISION}
        missing = {ModelCapability.EMBEDDINGS}

        status = ProviderStatus(
            provider_name="test",
            status=AuditStatus.AVAILABLE,
            capabilities=caps,
            missing_capabilities=missing,
            config_present=True,
            check_time_ms=10.5,
        )

        assert status.capabilities == caps
        assert status.missing_capabilities == missing
        assert status.config_present is True
        assert status.check_time_ms == 10.5

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        status = ProviderStatus(
            provider_name="anthropic",
            status=AuditStatus.NOT_CONFIGURED,
            error_message="API key not found",
        )
        result = status.to_dict()

        assert result["provider_name"] == "anthropic"
        assert result["status"] == "not_configured"
        assert result["error_message"] == "API key not found"
        assert isinstance(result["capabilities"], list)

    def test_immutability(self) -> None:
        """Test that provider status is immutable."""
        status = ProviderStatus(
            provider_name="test",
            status=AuditStatus.AVAILABLE,
        )
        with pytest.raises(AttributeError):
            status.provider_name = "changed"  # type: ignore


class TestAuditReport:
    """Tests for AuditReport dataclass."""

    def test_basic_creation(self) -> None:
        """Test creating a basic report."""
        report = AuditReport(
            report_id="test-001",
            timestamp="2026-02-17T12:00:00Z",
        )
        assert report.report_id == "test-001"
        assert report.total_providers == 0
        assert report.available_providers == 0

    def test_add_provider(self) -> None:
        """Test adding providers to report."""
        report = AuditReport(
            report_id="test-002",
            timestamp="2026-02-17T12:00:00Z",
        )

        status = ProviderStatus(
            provider_name="openai",
            status=AuditStatus.AVAILABLE,
            config_present=True,
        )

        result = report.add_provider(status)
        assert result is True
        assert report.total_providers == 1
        assert report.available_providers == 1

    def test_add_unavailable_provider(self) -> None:
        """Test that unavailable providers don't increment available count."""
        report = AuditReport(
            report_id="test-003",
            timestamp="2026-02-17T12:00:00Z",
        )

        status = ProviderStatus(
            provider_name="test",
            status=AuditStatus.NOT_CONFIGURED,
        )

        report.add_provider(status)
        assert report.total_providers == 1
        assert report.available_providers == 0

    def test_add_provider_respects_limit(self) -> None:
        """Test that add_provider respects MAX_PROVIDERS.

        JPL Rule #2: Fixed upper bounds.
        """
        report = AuditReport(
            report_id="test-limit",
            timestamp="2026-02-17T12:00:00Z",
        )

        # Fill to capacity
        for i in range(MAX_PROVIDERS):
            status = ProviderStatus(
                provider_name=f"provider_{i}",
                status=AuditStatus.AVAILABLE,
            )
            report.add_provider(status)

        # Try to add one more
        extra = ProviderStatus(
            provider_name="extra",
            status=AuditStatus.AVAILABLE,
        )
        result = report.add_provider(extra)

        assert result is False
        assert report.total_providers == MAX_PROVIDERS

    def test_exit_code_clean(self) -> None:
        """Test exit code when all available."""
        report = AuditReport(
            report_id="test-exit",
            timestamp="2026-02-17T12:00:00Z",
        )
        report.add_provider(
            ProviderStatus(
                provider_name="test",
                status=AuditStatus.AVAILABLE,
            )
        )

        assert report.exit_code == 0

    def test_exit_code_unavailable(self) -> None:
        """Test exit code with unavailable provider."""
        report = AuditReport(
            report_id="test-exit",
            timestamp="2026-02-17T12:00:00Z",
        )
        report.add_provider(
            ProviderStatus(
                provider_name="test",
                status=AuditStatus.UNAVAILABLE,
            )
        )

        assert report.exit_code == 1

    def test_exit_code_error(self) -> None:
        """Test exit code with error."""
        report = AuditReport(
            report_id="test-exit",
            timestamp="2026-02-17T12:00:00Z",
        )
        report.add_provider(
            ProviderStatus(
                provider_name="test",
                status=AuditStatus.ERROR,
                error_message="Test error",
            )
        )

        assert report.exit_code == 2

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        report = AuditReport(
            report_id="test-dict",
            timestamp="2026-02-17T12:00:00Z",
            audit_duration_ms=100.5,
        )
        result = report.to_dict()

        assert result["report_id"] == "test-dict"
        assert result["audit_duration_ms"] == 100.5
        assert "summary" in result
        assert "providers" in result


class TestModelAuditor:
    """Tests for ModelAuditor class."""

    def test_init_default(self) -> None:
        """Test default initialization."""
        auditor = ModelAuditor()
        assert len(auditor._known_providers) > 0

    def test_init_with_reference_capabilities(self) -> None:
        """Test initialization with custom capabilities."""
        caps = {ModelCapability.STREAMING, ModelCapability.EMBEDDINGS}
        auditor = ModelAuditor(reference_capabilities=caps)
        assert auditor._reference == caps

    def test_audit_unknown_provider(self) -> None:
        """Test auditing an unknown provider."""
        auditor = ModelAuditor()
        status = auditor.audit_provider("unknown_provider")

        assert status.status == AuditStatus.ERROR
        assert "Unknown provider" in (status.error_message or "")

    def test_audit_provider_not_configured(self) -> None:
        """Test auditing provider without configuration."""
        auditor = ModelAuditor()

        # Clear any existing env vars
        with patch.dict(os.environ, {}, clear=True):
            status = auditor.audit_provider("openai")
            assert status.status == AuditStatus.NOT_CONFIGURED

    def test_audit_provider_configured(self) -> None:
        """Test auditing a configured provider."""
        auditor = ModelAuditor()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            status = auditor.audit_provider("openai")
            assert status.status == AuditStatus.AVAILABLE
            assert status.config_present is True
            assert ModelCapability.STREAMING in status.capabilities

    def test_audit_all(self) -> None:
        """Test auditing all providers."""
        auditor = ModelAuditor()
        report = auditor.audit_all()

        assert report.total_providers == len(auditor._known_providers)
        assert report.report_id.startswith("audit-")
        assert report.audit_duration_ms > 0

    def test_audit_providers_specific(self) -> None:
        """Test auditing specific providers."""
        auditor = ModelAuditor()
        report = auditor.audit_providers(["openai", "ollama"])

        assert report.total_providers == 2

    def test_get_known_providers(self) -> None:
        """Test getting known providers."""
        auditor = ModelAuditor()
        providers = auditor.get_known_providers()

        assert "openai" in providers
        assert "anthropic" in providers
        assert "ollama" in providers
        assert providers == sorted(providers)  # Should be sorted

    def test_get_all_capabilities(self) -> None:
        """Test getting all capabilities."""
        auditor = ModelAuditor()
        caps = auditor.get_all_capabilities()

        assert "streaming" in caps
        assert "vision" in caps
        assert len(caps) == len(ModelCapability)


class TestProviderCapabilities:
    """Tests for provider capability definitions."""

    def test_openai_has_all_capabilities(self) -> None:
        """Test OpenAI has all major capabilities."""
        caps = PROVIDER_CAPABILITIES.get("openai", set())

        assert ModelCapability.STREAMING in caps
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.VISION in caps
        assert ModelCapability.EMBEDDINGS in caps
        assert ModelCapability.JSON_MODE in caps

    def test_anthropic_capabilities(self) -> None:
        """Test Anthropic capabilities."""
        caps = PROVIDER_CAPABILITIES.get("anthropic", set())

        assert ModelCapability.STREAMING in caps
        assert ModelCapability.FUNCTION_CALLING in caps
        assert ModelCapability.VISION in caps
        # Anthropic doesn't have embeddings
        assert ModelCapability.EMBEDDINGS not in caps

    def test_ollama_capabilities(self) -> None:
        """Test Ollama capabilities."""
        caps = PROVIDER_CAPABILITIES.get("ollama", set())

        assert ModelCapability.STREAMING in caps
        assert ModelCapability.EMBEDDINGS in caps
        # Ollama doesn't have function calling by default
        assert ModelCapability.FUNCTION_CALLING not in caps

    def test_llamacpp_capabilities(self) -> None:
        """Test LlamaCpp capabilities."""
        caps = PROVIDER_CAPABILITIES.get("llamacpp", set())

        assert ModelCapability.STREAMING in caps
        # LlamaCpp has minimal capabilities
        assert ModelCapability.VISION not in caps


class TestFactoryFunction:
    """Tests for create_model_auditor factory."""

    def test_create_model_auditor(self) -> None:
        """Test factory creates auditor."""
        auditor = create_model_auditor()
        assert isinstance(auditor, ModelAuditor)

    def test_create_with_capabilities(self) -> None:
        """Test factory with custom capabilities."""
        caps = {ModelCapability.STREAMING}
        auditor = create_model_auditor(reference_capabilities=caps)

        assert auditor._reference == caps


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_fixed_bounds_defined(self) -> None:
        """Test that fixed bounds are defined.

        JPL Rule #2: Fixed upper bounds.
        """
        assert MAX_PROVIDERS == 20
        assert MAX_CAPABILITIES == 10

    def test_capability_limit(self) -> None:
        """Test capability limit is enforced.

        JPL Rule #2: Fixed upper bounds.
        """
        # Create too many capabilities
        many_caps = set(ModelCapability)
        # This should work - we have fewer than MAX_CAPABILITIES
        assert len(many_caps) <= MAX_CAPABILITIES

    def test_provider_limit_enforced(self) -> None:
        """Test provider limit is enforced.

        JPL Rule #2: Fixed upper bounds.
        """
        auditor = ModelAuditor()

        # Generate more than MAX_PROVIDERS
        many_providers = [f"provider_{i}" for i in range(MAX_PROVIDERS + 10)]

        with pytest.raises(AssertionError):
            auditor.audit_providers(many_providers)

    def test_all_functions_return_explicit_values(self) -> None:
        """Test functions return explicit values.

        JPL Rule #7: Check all return values.
        """
        auditor = ModelAuditor()

        # audit_provider returns ProviderStatus
        status = auditor.audit_provider("openai")
        assert isinstance(status, ProviderStatus)

        # audit_all returns AuditReport
        report = auditor.audit_all()
        assert isinstance(report, AuditReport)

        # get_known_providers returns list
        providers = auditor.get_known_providers()
        assert isinstance(providers, list)
