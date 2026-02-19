"""
Unit tests for IFRedactionProcessor.

Security-Redaction-Guardrails
Tests GWT scenarios and NASA JPL Power of Ten compliance.
"""

import pytest
from pathlib import Path

from ingestforge.processors.security.redaction import (
    IFRedactionProcessor,
)
from ingestforge.ingest.refiners.redaction import (
    PIIType,
    RedactionConfig,
)
from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFileArtifact,
    IFFailureArtifact,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def processor() -> IFRedactionProcessor:
    """Create a default redaction processor."""
    return IFRedactionProcessor()


@pytest.fixture
def text_artifact_with_pii() -> IFTextArtifact:
    """Text artifact containing sensitive PII."""
    return IFTextArtifact(
        artifact_id="test-doc-001",
        content=(
            "Contact John Smith at john.smith@example.com or call 555-123-4567. "
            "SSN: 123-45-6789. Credit card: 4111-1111-1111-1111."
        ),
    )


@pytest.fixture
def chunk_artifact_with_pii() -> IFChunkArtifact:
    """Chunk artifact containing sensitive PII."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content="Patient SSN: 987-65-4321, DOB: 01/15/1980",
        chunk_index=0,
        total_chunks=1,
    )


@pytest.fixture
def text_artifact_no_pii() -> IFTextArtifact:
    """Text artifact with no sensitive information."""
    return IFTextArtifact(
        artifact_id="test-doc-002",
        content="The quick brown fox jumps over the lazy dog.",
    )


# ---------------------------------------------------------------------------
# GWT Scenario 1: SSN Redaction
# ---------------------------------------------------------------------------


class TestSSNRedaction:
    """Given SSN in text, When redaction runs, Then SSN is replaced with [REDACTED]."""

    def test_given_ssn_when_processed_then_redacted(
        self, processor: IFRedactionProcessor
    ):
        """Given text with SSN, When processed, Then SSN is [REDACTED]."""
        artifact = IFTextArtifact(
            artifact_id="ssn-test",
            content="SSN: 123-45-6789 is sensitive",
        )

        result = processor.process(artifact)

        assert "[REDACTED]" in result.content
        assert "123-45-6789" not in result.content
        assert result.metadata["redaction_applied"] is True
        assert result.metadata["redaction_count"] >= 1

    def test_given_multiple_ssns_when_processed_then_all_redacted(
        self, processor: IFRedactionProcessor
    ):
        """Given multiple SSNs, When processed, Then all are [REDACTED]."""
        artifact = IFTextArtifact(
            artifact_id="multi-ssn",
            content="SSN1: 111-22-3333, SSN2: 444-55-6666",
        )

        result = processor.process(artifact)

        assert result.content.count("[REDACTED]") >= 2
        assert "111-22-3333" not in result.content
        assert "444-55-6666" not in result.content


# ---------------------------------------------------------------------------
# GWT Scenario 2: Credit Card Redaction
# ---------------------------------------------------------------------------


class TestCreditCardRedaction:
    """Given credit card in text, When redaction runs, Then card is [REDACTED]."""

    def test_given_credit_card_when_processed_then_redacted(
        self, processor: IFRedactionProcessor
    ):
        """Given text with credit card, When processed, Then card is [REDACTED]."""
        processor = IFRedactionProcessor(enabled_types={PIIType.CREDIT_CARD})
        artifact = IFTextArtifact(
            artifact_id="cc-test",
            content="Card: 4111-1111-1111-1111",
        )

        result = processor.process(artifact)

        assert "[REDACTED]" in result.content
        assert "4111-1111-1111-1111" not in result.content


# ---------------------------------------------------------------------------
# GWT Scenario 3: Email Redaction
# ---------------------------------------------------------------------------


class TestEmailRedaction:
    """Given email in text, When redaction runs, Then email is [REDACTED]."""

    def test_given_email_when_processed_then_redacted(
        self, processor: IFRedactionProcessor
    ):
        """Given text with email, When processed, Then email is [REDACTED]."""
        artifact = IFTextArtifact(
            artifact_id="email-test",
            content="Contact: user@example.com",
        )

        result = processor.process(artifact)

        assert "[REDACTED]" in result.content
        assert "user@example.com" not in result.content


# ---------------------------------------------------------------------------
# GWT Scenario 4: Phone Redaction
# ---------------------------------------------------------------------------


class TestPhoneRedaction:
    """Given phone number in text, When redaction runs, Then phone is [REDACTED]."""

    def test_given_phone_when_processed_then_redacted(
        self, processor: IFRedactionProcessor
    ):
        """Given text with phone, When processed, Then phone is [REDACTED]."""
        artifact = IFTextArtifact(
            artifact_id="phone-test",
            content="Call: 555-123-4567",
        )

        result = processor.process(artifact)

        assert "[REDACTED]" in result.content
        assert "555-123-4567" not in result.content


# ---------------------------------------------------------------------------
# IFProcessor Interface Tests
# ---------------------------------------------------------------------------


class TestIFProcessorInterface:
    """Verify IFRedactionProcessor implements IFProcessor interface."""

    def test_processor_id(self, processor: IFRedactionProcessor):
        """Processor has valid processor_id."""
        assert processor.processor_id == "security-redaction-processor"

    def test_version(self, processor: IFRedactionProcessor):
        """Processor has valid version."""
        assert processor.version == "1.0.0"

    def test_capabilities(self, processor: IFRedactionProcessor):
        """Processor declares capabilities."""
        caps = processor.capabilities
        assert "pii-redaction" in caps
        assert "security" in caps

    def test_is_available(self, processor: IFRedactionProcessor):
        """Processor is always available."""
        assert processor.is_available() is True

    def test_memory_mb(self, processor: IFRedactionProcessor):
        """Processor declares memory requirement."""
        assert processor.memory_mb > 0

    def test_teardown(self, processor: IFRedactionProcessor):
        """Teardown returns True."""
        processor._initialize()
        result = processor.teardown()
        assert result is True
        assert processor._initialized is False


# ---------------------------------------------------------------------------
# Artifact Type Handling Tests
# ---------------------------------------------------------------------------


class TestArtifactTypeHandling:
    """Test handling of different artifact types."""

    def test_processes_text_artifact(
        self, processor: IFRedactionProcessor, text_artifact_with_pii: IFTextArtifact
    ):
        """IFTextArtifact is processed correctly."""
        result = processor.process(text_artifact_with_pii)

        assert not isinstance(result, IFFailureArtifact)
        assert result.metadata["redaction_applied"] is True

    def test_processes_chunk_artifact(
        self, processor: IFRedactionProcessor, chunk_artifact_with_pii: IFChunkArtifact
    ):
        """IFChunkArtifact is processed correctly."""
        result = processor.process(chunk_artifact_with_pii)

        assert not isinstance(result, IFFailureArtifact)
        assert "987-65-4321" not in result.content

    def test_unsupported_artifact_returns_failure(
        self, processor: IFRedactionProcessor
    ):
        """Unsupported artifact type returns IFFailureArtifact."""
        from pathlib import Path

        artifact = IFFileArtifact(
            artifact_id="file-001",
            file_path=Path("/path/to/file.pdf"),
            mime_type="application/pdf",
        )

        result = processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "IFRedactionProcessor requires" in result.error_message


# ---------------------------------------------------------------------------
# Metadata Tests
# ---------------------------------------------------------------------------


class TestRedactionMetadata:
    """Test redaction metadata in processed artifacts."""

    def test_metadata_includes_redaction_stats(
        self, processor: IFRedactionProcessor, text_artifact_with_pii: IFTextArtifact
    ):
        """Processed artifact includes redaction statistics."""
        result = processor.process(text_artifact_with_pii)

        assert "redaction_applied" in result.metadata
        assert "redaction_count" in result.metadata
        assert "redaction_stats" in result.metadata
        assert "redaction_skipped" in result.metadata

    def test_no_redactions_metadata(
        self, processor: IFRedactionProcessor, text_artifact_no_pii: IFTextArtifact
    ):
        """Non-sensitive text has redaction_applied=False."""
        result = processor.process(text_artifact_no_pii)

        assert result.metadata["redaction_applied"] is False
        assert result.metadata["redaction_count"] == 0


# ---------------------------------------------------------------------------
# Lineage Tests
# ---------------------------------------------------------------------------


class TestLineageTracking:
    """Test artifact lineage is maintained."""

    def test_parent_id_set(
        self, processor: IFRedactionProcessor, text_artifact_with_pii: IFTextArtifact
    ):
        """Processed artifact has correct parent_id."""
        result = processor.process(text_artifact_with_pii)

        assert result.parent_id == text_artifact_with_pii.artifact_id

    def test_provenance_updated(
        self, processor: IFRedactionProcessor, text_artifact_with_pii: IFTextArtifact
    ):
        """Processor ID is added to provenance."""
        result = processor.process(text_artifact_with_pii)

        assert processor.processor_id in result.provenance

    def test_lineage_depth_incremented(
        self, processor: IFRedactionProcessor, text_artifact_with_pii: IFTextArtifact
    ):
        """Lineage depth is incremented."""
        result = processor.process(text_artifact_with_pii)

        assert result.lineage_depth == text_artifact_with_pii.lineage_depth + 1


# ---------------------------------------------------------------------------
# JPL Rule #7: Return Value Checks
# ---------------------------------------------------------------------------


class TestJPLRule7ReturnValueChecks:
    """AC: JPL Rule #7 - Check return values of redaction functions."""

    def test_redaction_result_always_returned(self, processor: IFRedactionProcessor):
        """Redaction always returns a result, never None."""
        artifact = IFTextArtifact(
            artifact_id="test",
            content="Test content",
        )

        result = processor.process(artifact)

        assert result is not None

    def test_empty_content_handled(self, processor: IFRedactionProcessor):
        """Empty content is handled gracefully."""
        artifact = IFTextArtifact(
            artifact_id="empty-test",
            content="",
        )

        result = processor.process(artifact)

        assert result is not None
        assert result.content == ""

    def test_whitespace_content_handled(self, processor: IFRedactionProcessor):
        """Whitespace-only content is handled gracefully."""
        artifact = IFTextArtifact(
            artifact_id="ws-test",
            content="   \n\t  ",
        )

        result = processor.process(artifact)

        assert result is not None


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------


class TestRedactionConfiguration:
    """Test configuration options."""

    def test_custom_enabled_types(self):
        """Custom enabled types are respected."""
        processor = IFRedactionProcessor(enabled_types={PIIType.SSN})
        artifact = IFTextArtifact(
            artifact_id="config-test",
            content="SSN: 123-45-6789, Email: test@example.com",
        )

        result = processor.process(artifact)

        # SSN should be redacted
        assert "123-45-6789" not in result.content
        # Email should NOT be redacted (not in enabled_types)
        assert "test@example.com" in result.content

    def test_config_object_used(self):
        """Provided RedactionConfig is used."""
        config = RedactionConfig(
            enabled_types={PIIType.EMAIL},
            show_type=False,
        )
        processor = IFRedactionProcessor(config=config)
        artifact = IFTextArtifact(
            artifact_id="config-obj-test",
            content="Email: test@example.com, SSN: 123-45-6789",
        )

        result = processor.process(artifact)

        # Email should be redacted
        assert "test@example.com" not in result.content
        # SSN should NOT be redacted
        assert "123-45-6789" in result.content


# ---------------------------------------------------------------------------
# YAML Configuration Tests
# ---------------------------------------------------------------------------


class TestYAMLConfiguration:
    """Test loading configuration from YAML."""

    def test_yaml_config_loading(self, tmp_path: Path):
        """YAML configuration is loaded correctly."""
        yaml_content = """
enabled_types:
  - ssn
  - email
whitelist:
  - example.com
custom_patterns:
  test_pattern: "TEST-\\d{4}"
"""
        config_file = tmp_path / "redaction.yaml"
        config_file.write_text(yaml_content)

        processor = IFRedactionProcessor(config_path=str(config_file))
        processor._initialize()

        assert processor._config is not None
        assert PIIType.SSN in processor._config.enabled_types

    def test_missing_yaml_uses_defaults(self):
        """Missing YAML file uses default configuration."""
        processor = IFRedactionProcessor(config_path="/nonexistent/path.yaml")
        processor._initialize()

        assert processor._config is not None
        assert processor._initialized is True
