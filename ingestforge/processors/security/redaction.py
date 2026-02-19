"""
IFRedactionProcessor - Security Redaction for Knowledge Graph.

Security-Redaction-Guardrails
Prevents sensitive information from leaking into the Knowledge Graph via
automated redaction layer.

NASA JPL Power of Ten compliant.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set

from ingestforge.core.pipeline.interfaces import IFProcessor, IFArtifact
from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFailureArtifact,
)
from ingestforge.ingest.refiners.redaction import (
    PIIRedactor,
    PIIType,
    RedactionConfig,
    RedactionResult,
)
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_CUSTOM_PATTERNS = 50
MAX_TEXT_LENGTH = 1_000_000  # 1MB


class IFRedactionProcessor(IFProcessor):
    """
    Processor that redacts sensitive information before Knowledge Graph indexing.

    Security-Redaction-Guardrails
    - Implements IFProcessor interface
    - Supports regex-based pattern matching
    - Replaces sensitive data with [REDACTED]

    NASA JPL Power of Ten compliant.
    Rule #4: Methods < 60 lines.
    Rule #7: Check return values of redaction functions.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        config: Optional[RedactionConfig] = None,
        config_path: Optional[str] = None,
        enabled_types: Optional[Set[PIIType]] = None,
    ) -> None:
        """
        Initialize the redaction processor.

        Args:
            config: Optional RedactionConfig instance
            config_path: Path to redaction.yaml configuration file
            enabled_types: Set of PIITypes to enable (defaults to all)
        """
        self._config = config
        self._config_path = config_path
        self._enabled_types = enabled_types
        self._redactor: Optional[PIIRedactor] = None
        self._initialized = False

    def _initialize(self) -> bool:
        """
        Initialize the redactor with configuration.

        Rule #4: Helper function < 60 lines.
        Rule #7: Check return values.

        Returns:
            True if initialization successful.
        """
        if self._initialized:
            return True

        # Load config from file if path provided
        if self._config_path and not self._config:
            loaded_config = self._load_config_from_yaml(self._config_path)
            if loaded_config:
                self._config = loaded_config

        # Create default config if none provided
        if not self._config:
            self._config = RedactionConfig(
                enabled_types=self._enabled_types
                or {
                    PIIType.EMAIL,
                    PIIType.PHONE,
                    PIIType.SSN,
                    PIIType.CREDIT_CARD,
                    PIIType.PERSON_NAME,
                },
                show_type=False,  # Use [REDACTED] instead of [TYPE]
            )

        # Override show_type to use [REDACTED]
        self._config.show_type = False

        self._redactor = PIIRedactor(self._config)
        self._initialized = True
        return True

    def _load_config_from_yaml(self, path: str) -> Optional[RedactionConfig]:
        """
        Load redaction configuration from YAML file.

        Rule #4: Helper function < 60 lines.
        Rule #7: Handle errors gracefully.

        Args:
            path: Path to YAML configuration file

        Returns:
            RedactionConfig if loaded successfully, None otherwise
        """
        try:
            import yaml
        except ImportError:
            logger.warning("PyYAML not installed, cannot load redaction.yaml")
            return None

        config_file = Path(path)
        if not config_file.exists():
            logger.warning(f"Redaction config not found: {path}")
            return None

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load redaction config: {e}")
            return None

        if not data:
            return None

        # Parse enabled types
        enabled_types: Set[PIIType] = set()
        if "enabled_types" in data:
            for type_name in data["enabled_types"]:
                try:
                    enabled_types.add(PIIType(type_name.lower()))
                except ValueError:
                    logger.warning(f"Unknown PII type: {type_name}")

        # Parse custom patterns
        custom_patterns: Dict[str, str] = {}
        if "custom_patterns" in data:
            patterns = data["custom_patterns"]
            if isinstance(patterns, dict) and len(patterns) <= MAX_CUSTOM_PATTERNS:
                custom_patterns = patterns

        # Parse whitelist
        whitelist: Set[str] = set()
        if "whitelist" in data:
            whitelist = set(data["whitelist"][:1000])  # JPL Rule #2

        return RedactionConfig(
            enabled_types=enabled_types or {PIIType.SSN, PIIType.CREDIT_CARD},
            whitelist=whitelist,
            custom_patterns=custom_patterns,
            show_type=False,  # Always use [REDACTED]
        )

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "security-redaction-processor"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return "1.0.0"

    @property
    def capabilities(self) -> List[str]:
        """Capabilities provided by this processor."""
        return [
            "pii-redaction",
            "security",
            "data-protection",
        ]

    @property
    def memory_mb(self) -> int:
        """Estimated memory requirement in MB."""
        return 50

    def is_available(self) -> bool:
        """
        Check if processor is available.

        Always returns True as regex redaction has no external dependencies.
        """
        return True

    def _redact_text(self, text: str) -> RedactionResult:
        """
        Redact sensitive information from text.

        Rule #4: Helper function < 60 lines.
        Rule #7: Check return values explicitly.

        Args:
            text: Text to redact

        Returns:
            RedactionResult with redacted text and statistics
        """
        # JPL Rule #2: Enforce text length limit
        if len(text) > MAX_TEXT_LENGTH:
            logger.warning(f"Text truncated from {len(text)} to {MAX_TEXT_LENGTH}")
            text = text[:MAX_TEXT_LENGTH]

        # JPL Rule #7: Explicitly check redaction result
        result = self._redactor.redact(text)

        if result is None:
            # This shouldn't happen, but handle defensively
            logger.error("Redaction returned None unexpectedly")
            return RedactionResult(
                original_text=text,
                redacted_text=text,
            )

        return result

    def _apply_redacted_replacement(self, result: RedactionResult) -> str:
        """
        Replace redacted tokens with [REDACTED].

        AC: Redacted data is replaced with [REDACTED].
        Rule #4: Helper function < 60 lines.

        Args:
            result: RedactionResult from PIIRedactor

        Returns:
            Text with all sensitive data replaced by [REDACTED]
        """
        text = result.original_text

        # Sort matches by position (reverse for safe replacement)
        sorted_matches = sorted(result.matches, key=lambda m: m.start, reverse=True)

        for match in sorted_matches:
            text = text[: match.start] + "[REDACTED]" + text[match.end :]

        return text

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """
        Process artifact to redact sensitive information.

        Implements IFProcessor.process().
        Runs before storage in the pipeline loop.

        Rule #4: Function < 60 lines.
        Rule #7: Handle all input types.

        Args:
            artifact: Input artifact (IFTextArtifact or IFChunkArtifact)

        Returns:
            Derived artifact with redacted content
        """
        # Initialize on first use
        if not self._initialize():
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-redaction-failure",
                error_message="Failed to initialize redaction processor",
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Determine content based on artifact type
        content: Optional[str] = None

        if isinstance(artifact, IFTextArtifact):
            content = artifact.content
        elif isinstance(artifact, IFChunkArtifact):
            content = artifact.content
        else:
            return IFFailureArtifact(
                artifact_id=f"{artifact.artifact_id}-redaction-failure",
                error_message=(
                    f"IFRedactionProcessor requires IFTextArtifact or IFChunkArtifact, "
                    f"got {type(artifact).__name__}"
                ),
                failed_processor_id=self.processor_id,
                parent_id=artifact.artifact_id,
                root_artifact_id=artifact.effective_root_id,
                lineage_depth=artifact.lineage_depth + 1,
                provenance=artifact.provenance + [self.processor_id],
            )

        # Perform redaction
        result = self._redact_text(content)

        # Apply [REDACTED] replacement
        redacted_content = self._apply_redacted_replacement(result)

        # Build metadata
        new_metadata = dict(artifact.metadata)
        new_metadata["redaction_applied"] = result.has_redactions
        new_metadata["redaction_count"] = result.total_redactions
        new_metadata["redaction_stats"] = result.stats
        new_metadata["redaction_skipped"] = len(result.skipped)

        # Return derived artifact with redacted content
        return artifact.derive(
            self.processor_id,
            artifact_id=f"{artifact.artifact_id}-redacted",
            content=redacted_content,
            metadata=new_metadata,
        )

    def teardown(self) -> bool:
        """Clean up resources."""
        self._redactor = None
        self._initialized = False
        return True
