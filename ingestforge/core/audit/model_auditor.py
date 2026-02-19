"""LLM Model Parity Auditor.

LLM Model Parity Auditor
Epic: EP-26 (Security & Compliance)

Provides auditing framework for validating LLM provider configurations
and checking feature parity across different models.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_PROVIDERS, MAX_CAPABILITIES)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_PROVIDERS = 20
MAX_CAPABILITIES = 10
MAX_AUDIT_TIME_SECONDS = 60


class ModelCapability(Enum):
    """LLM model capabilities for parity checking."""

    STREAMING = "streaming"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    EMBEDDINGS = "embeddings"
    JSON_MODE = "json_mode"
    SYSTEM_PROMPT = "system_prompt"
    MULTI_TURN = "multi_turn"


class AuditStatus(Enum):
    """Audit result status."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


# Provider capability definitions (what each provider supports)
PROVIDER_CAPABILITIES: Dict[str, Set[ModelCapability]] = {
    "openai": {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.VISION,
        ModelCapability.EMBEDDINGS,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.MULTI_TURN,
    },
    "anthropic": {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.VISION,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.MULTI_TURN,
    },
    "ollama": {
        ModelCapability.STREAMING,
        ModelCapability.EMBEDDINGS,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.MULTI_TURN,
    },
    "llamacpp": {
        ModelCapability.STREAMING,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.MULTI_TURN,
    },
    "gemini": {
        ModelCapability.STREAMING,
        ModelCapability.FUNCTION_CALLING,
        ModelCapability.VISION,
        ModelCapability.EMBEDDINGS,
        ModelCapability.JSON_MODE,
        ModelCapability.SYSTEM_PROMPT,
        ModelCapability.MULTI_TURN,
    },
}


@dataclass(frozen=True)
class ProviderStatus:
    """Status of a single LLM provider.

    Rule #9: Complete type hints.
    """

    provider_name: str
    status: AuditStatus
    capabilities: Set[ModelCapability] = field(default_factory=set)
    missing_capabilities: Set[ModelCapability] = field(default_factory=set)
    config_present: bool = False
    error_message: Optional[str] = None
    check_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "provider_name": self.provider_name,
            "status": self.status.value,
            "capabilities": [c.value for c in self.capabilities],
            "missing_capabilities": [c.value for c in self.missing_capabilities],
            "config_present": self.config_present,
            "error_message": self.error_message,
            "check_time_ms": self.check_time_ms,
        }


@dataclass
class AuditReport:
    """Complete audit report for all providers.

    Rule #9: Complete type hints.
    """

    report_id: str
    timestamp: str
    providers: List[ProviderStatus] = field(default_factory=list)
    total_providers: int = 0
    available_providers: int = 0
    audit_duration_ms: float = 0.0

    def __post_init__(self) -> None:
        """Validate report.

        Rule #5: Assert preconditions.
        """
        assert len(self.providers) <= MAX_PROVIDERS, "Too many providers"

    def add_provider(self, status: ProviderStatus) -> bool:
        """Add provider status to report.

        Args:
            status: Provider status to add.

        Returns:
            True if added, False if at capacity.

        Rule #7: Check return values.
        """
        if len(self.providers) >= MAX_PROVIDERS:
            return False

        self.providers.append(status)
        self.total_providers = len(self.providers)

        if status.status == AuditStatus.AVAILABLE:
            self.available_providers += 1

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "report_id": self.report_id,
            "timestamp": self.timestamp,
            "providers": [p.to_dict() for p in self.providers],
            "total_providers": self.total_providers,
            "available_providers": self.available_providers,
            "audit_duration_ms": self.audit_duration_ms,
            "summary": self._generate_summary(),
        }

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Summary dictionary.
        """
        full_parity = []
        partial_parity = []
        unavailable = []

        for provider in self.providers:
            if provider.status == AuditStatus.AVAILABLE:
                if not provider.missing_capabilities:
                    full_parity.append(provider.provider_name)
                else:
                    partial_parity.append(provider.provider_name)
            else:
                unavailable.append(provider.provider_name)

        return {
            "full_parity": full_parity,
            "partial_parity": partial_parity,
            "unavailable": unavailable,
        }

    @property
    def exit_code(self) -> int:
        """Get exit code for CI integration.

        Returns:
            0 if all available, 1 if some unavailable, 2 if errors.
        """
        has_errors = any(p.status == AuditStatus.ERROR for p in self.providers)
        if has_errors:
            return 2

        has_unavailable = any(
            p.status == AuditStatus.UNAVAILABLE for p in self.providers
        )
        if has_unavailable:
            return 1

        return 0


class ModelAuditor:
    """LLM model parity auditor.

    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        reference_capabilities: Optional[Set[ModelCapability]] = None,
    ) -> None:
        """Initialize auditor.

        Args:
            reference_capabilities: Capabilities to check against (default: all).

        Rule #5: Assert preconditions.
        """
        if reference_capabilities:
            assert (
                len(reference_capabilities) <= MAX_CAPABILITIES
            ), "Too many capabilities"

        self._reference = reference_capabilities or set(ModelCapability)
        self._known_providers = set(PROVIDER_CAPABILITIES.keys())

    def audit_provider(self, provider_name: str) -> ProviderStatus:
        """Audit a single provider.

        Args:
            provider_name: Name of the provider to audit.

        Returns:
            ProviderStatus with audit results.

        Rule #7: Return explicit result.
        """
        start_time = time.perf_counter()
        provider_lower = provider_name.lower()

        # Check if provider is known
        if provider_lower not in self._known_providers:
            return self._create_unknown_status(provider_name, start_time)

        # Check configuration
        config_present = self._check_provider_config(provider_lower)
        if not config_present:
            return self._create_not_configured_status(provider_name, start_time)

        # Check capabilities
        return self._check_capabilities(provider_lower, start_time)

    def _create_unknown_status(
        self, provider_name: str, start_time: float
    ) -> ProviderStatus:
        """Create status for unknown provider.

        Args:
            provider_name: Provider name.
            start_time: Audit start time.

        Returns:
            ProviderStatus for unknown provider.
        """
        elapsed = (time.perf_counter() - start_time) * 1000
        return ProviderStatus(
            provider_name=provider_name,
            status=AuditStatus.ERROR,
            error_message=f"Unknown provider: {provider_name}",
            check_time_ms=elapsed,
        )

    def _create_not_configured_status(
        self, provider_name: str, start_time: float
    ) -> ProviderStatus:
        """Create status for unconfigured provider.

        Args:
            provider_name: Provider name.
            start_time: Audit start time.

        Returns:
            ProviderStatus for unconfigured provider.
        """
        elapsed = (time.perf_counter() - start_time) * 1000
        return ProviderStatus(
            provider_name=provider_name,
            status=AuditStatus.NOT_CONFIGURED,
            config_present=False,
            check_time_ms=elapsed,
        )

    def _check_capabilities(
        self, provider_name: str, start_time: float
    ) -> ProviderStatus:
        """Check provider capabilities against reference.

        Args:
            provider_name: Provider name.
            start_time: Audit start time.

        Returns:
            ProviderStatus with capability check results.
        """
        provider_caps = PROVIDER_CAPABILITIES.get(provider_name, set())
        present_caps = provider_caps & self._reference
        missing_caps = self._reference - provider_caps

        elapsed = (time.perf_counter() - start_time) * 1000

        return ProviderStatus(
            provider_name=provider_name,
            status=AuditStatus.AVAILABLE,
            capabilities=present_caps,
            missing_capabilities=missing_caps,
            config_present=True,
            check_time_ms=elapsed,
        )

    def _check_provider_config(self, provider_name: str) -> bool:
        """Check if provider is configured.

        Args:
            provider_name: Provider to check.

        Returns:
            True if configured.

        Rule #7: Return explicit result.
        """
        import os

        config_vars = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "gemini": "GOOGLE_API_KEY",
            "ollama": "OLLAMA_HOST",
            "llamacpp": "LLAMACPP_MODEL_PATH",
        }

        var_name = config_vars.get(provider_name)
        if not var_name:
            return False

        # Ollama and LlamaCpp can work without env vars
        if provider_name in ("ollama", "llamacpp"):
            return True

        return bool(os.environ.get(var_name))

    def audit_all(self) -> AuditReport:
        """Audit all known providers.

        Returns:
            AuditReport with all provider statuses.

        Rule #7: Return explicit result.
        """
        start_time = time.perf_counter()

        report = AuditReport(
            report_id=f"audit-{int(time.time())}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        for provider_name in sorted(self._known_providers):
            status = self.audit_provider(provider_name)
            report.add_provider(status)

        report.audit_duration_ms = (time.perf_counter() - start_time) * 1000
        return report

    def audit_providers(self, provider_names: List[str]) -> AuditReport:
        """Audit specific providers.

        Args:
            provider_names: List of provider names to audit.

        Returns:
            AuditReport with specified provider statuses.

        Rule #5: Assert preconditions.
        Rule #7: Return explicit result.
        """
        assert len(provider_names) <= MAX_PROVIDERS, "Too many providers"

        start_time = time.perf_counter()

        report = AuditReport(
            report_id=f"audit-{int(time.time())}",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

        for provider_name in provider_names[:MAX_PROVIDERS]:
            status = self.audit_provider(provider_name)
            report.add_provider(status)

        report.audit_duration_ms = (time.perf_counter() - start_time) * 1000
        return report

    def get_known_providers(self) -> List[str]:
        """Get list of known providers.

        Returns:
            Sorted list of provider names.
        """
        return sorted(self._known_providers)

    def get_all_capabilities(self) -> List[str]:
        """Get list of all capabilities.

        Returns:
            List of capability names.
        """
        return [c.value for c in ModelCapability]


def create_model_auditor(
    reference_capabilities: Optional[Set[ModelCapability]] = None,
) -> ModelAuditor:
    """Factory function to create a model auditor.

    Args:
        reference_capabilities: Capabilities to check against.

    Returns:
        Configured ModelAuditor instance.
    """
    return ModelAuditor(reference_capabilities=reference_capabilities)
