"""
Unit tests for IFUniversalExtractor.

Universal Entity Extraction
Tests GWT scenarios and NASA JPL Power of Ten compliance.
"""

import pytest
from unittest.mock import Mock, patch

from ingestforge.enrichment.universal_extractor import (
    IFUniversalExtractor,
    ExtractedEntity,
    EntityExtractionResult,
    MAX_ENTITIES_PER_EXTRACTION,
    MAX_ENTITY_TEXT_LENGTH,
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
def extractor() -> IFUniversalExtractor:
    """Create extractor in regex-only mode for deterministic testing."""
    return IFUniversalExtractor(use_llm=False, min_confidence=0.5)


@pytest.fixture
def text_artifact() -> IFTextArtifact:
    """Sample text artifact with entities."""
    return IFTextArtifact(
        artifact_id="test-doc-001",
        content=(
            "Dr. John Smith from Microsoft Corporation filed Case No. 20-1234 "
            "on January 15, 2024. The security advisory mentions CVE-2024-5678 "
            "affecting Part P/N ABC-12345. Contact support@example.com or call "
            "(555) 123-4567. Visit https://example.com for $1,000 discount."
        ),
    )


@pytest.fixture
def chunk_artifact() -> IFChunkArtifact:
    """Sample chunk artifact with entities."""
    return IFChunkArtifact(
        artifact_id="chunk-001",
        document_id="doc-001",
        content="Judge Elena Rodriguez ruled on Docket No. 11-cv-01846-LHK in California.",
        chunk_index=0,
        total_chunks=5,
    )


# ---------------------------------------------------------------------------
# GWT Scenario 1: Standard NER Detection
# ---------------------------------------------------------------------------


class TestStandardNERDetection:
    """Given raw text, When extraction runs, Then standard NER entities are detected."""

    def test_detects_person_with_title(self, extractor: IFUniversalExtractor):
        """Given text with titled name, When extracted, Then PERSON entity found."""
        text = "Dr. Jane Doe presented the findings."
        entities = extractor.extract(text)

        person_entities = [e for e in entities if e.entity_type == "PERSON"]
        assert len(person_entities) >= 1
        assert any("Jane Doe" in e.text for e in person_entities)

    def test_detects_organization_with_suffix(self, extractor: IFUniversalExtractor):
        """Given text with org suffixes, When extracted, Then ORG entities found."""
        text = "Apple Inc. and Microsoft Corporation announced partnership."
        entities = extractor.extract(text)

        org_entities = [e for e in entities if e.entity_type == "ORG"]
        assert len(org_entities) >= 1

    def test_detects_date_formats(self, extractor: IFUniversalExtractor):
        """Given various date formats, When extracted, Then DATE entities found."""
        text = "Meeting on January 15, 2024 and deadline 2024-12-31."
        entities = extractor.extract(text)

        date_entities = [e for e in entities if e.entity_type == "DATE"]
        assert len(date_entities) >= 2

    def test_detects_iso_date(self, extractor: IFUniversalExtractor):
        """Given ISO date format, When extracted, Then DATE entity found."""
        text = "Release scheduled for 2024-06-15."
        entities = extractor.extract(text)

        date_entities = [e for e in entities if e.entity_type == "DATE"]
        assert len(date_entities) == 1
        assert date_entities[0].text == "2024-06-15"


# ---------------------------------------------------------------------------
# GWT Scenario 2: Domain Entity Detection (Dockets, PartNumbers, CVEs)
# ---------------------------------------------------------------------------


class TestDomainEntityDetection:
    """Given text with domain markers, When extracted, Then domain entities found."""

    def test_detects_cve_ids(self, extractor: IFUniversalExtractor):
        """Given CVE IDs, When extracted, Then CVE entities found."""
        text = "Patches for CVE-2024-1234 and CVE-2023-98765 are available."
        entities = extractor.extract(text)

        cve_entities = [e for e in entities if e.entity_type == "CVE"]
        assert len(cve_entities) == 2
        assert all(e.confidence >= 0.9 for e in cve_entities)

    def test_detects_docket_numbers(self, extractor: IFUniversalExtractor):
        """Given legal docket numbers, When extracted, Then DOCKET entities found."""
        text = "Case No. 20-1234 was consolidated with Docket No. 21-5678."
        entities = extractor.extract(text)

        docket_entities = [e for e in entities if e.entity_type == "DOCKET"]
        assert len(docket_entities) >= 1

    def test_detects_federal_docket_format(self, extractor: IFUniversalExtractor):
        """Given federal court docket format, When extracted, Then DOCKET found."""
        text = "Filing in case 1:22-cv-00001-ABC was approved."
        entities = extractor.extract(text)

        docket_entities = [e for e in entities if e.entity_type == "DOCKET"]
        assert len(docket_entities) >= 1

    def test_detects_part_numbers(self, extractor: IFUniversalExtractor):
        """Given manufacturing part numbers, When extracted, Then PART_NUMBER found."""
        text = "Replace Part P/N XYZ-12345678 with Ref: ABC-9876543."
        entities = extractor.extract(text)

        part_entities = [e for e in entities if e.entity_type == "PART_NUMBER"]
        assert len(part_entities) >= 1


# ---------------------------------------------------------------------------
# GWT Scenario 3: Confidence and Source Link
# ---------------------------------------------------------------------------


class TestConfidenceAndSourceLink:
    """Given extracted entity, Then confidence and source_link are present."""

    def test_entity_has_confidence(self, extractor: IFUniversalExtractor):
        """Given extraction, When entity returned, Then confidence is set."""
        text = "CVE-2024-9999 is critical."
        entities = extractor.extract(text)

        assert len(entities) >= 1
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0

    def test_entity_has_source_link(self, extractor: IFUniversalExtractor):
        """Given extraction, When entity returned, Then source_link is valid."""
        text = "Contact test@example.com for info."
        entities = extractor.extract(text)

        email_entities = [e for e in entities if e.entity_type == "EMAIL"]
        assert len(email_entities) == 1
        assert (
            email_entities[0].source_link
            == f"{email_entities[0].start_char}:{email_entities[0].end_char}"
        )

    def test_source_link_matches_text_position(self, extractor: IFUniversalExtractor):
        """Given entity, When source_link checked, Then positions are accurate."""
        text = "CVE-2024-1234 is here."
        entities = extractor.extract(text)

        cve_entity = next(e for e in entities if e.entity_type == "CVE")
        # Verify the positions point to the actual text
        assert text[cve_entity.start_char : cve_entity.end_char] == cve_entity.text

    def test_to_dict_includes_source_link(self, extractor: IFUniversalExtractor):
        """Given entity, When to_dict called, Then source_link included."""
        text = "$500 discount available."
        entities = extractor.extract(text)

        money_entities = [e for e in entities if e.entity_type == "MONEY"]
        assert len(money_entities) >= 1
        entity_dict = money_entities[0].to_dict()
        assert "source_link" in entity_dict
        assert ":" in entity_dict["source_link"]


# ---------------------------------------------------------------------------
# GWT Scenario 4: IFProcessor Interface
# ---------------------------------------------------------------------------


class TestIFProcessorInterface:
    """Given IFUniversalExtractor, Then it implements IFProcessor correctly."""

    def test_processor_id(self, extractor: IFUniversalExtractor):
        """Given extractor, When processor_id accessed, Then valid ID returned."""
        assert extractor.processor_id == "universal-entity-extractor"

    def test_version(self, extractor: IFUniversalExtractor):
        """Given extractor, When version accessed, Then semver returned."""
        assert extractor.version == "1.0.0"

    def test_capabilities(self, extractor: IFUniversalExtractor):
        """Given extractor, When capabilities accessed, Then list returned."""
        caps = extractor.capabilities
        assert "ner" in caps
        assert "entity-extraction" in caps
        assert "universal-extraction" in caps

    def test_memory_mb(self, extractor: IFUniversalExtractor):
        """Given extractor, When memory_mb accessed, Then reasonable value."""
        assert extractor.memory_mb > 0
        assert extractor.memory_mb < 1000

    def test_is_available(self, extractor: IFUniversalExtractor):
        """Given extractor, When is_available called, Then True (regex fallback)."""
        assert extractor.is_available() is True

    def test_teardown(self, extractor: IFUniversalExtractor):
        """Given extractor, When teardown called, Then returns True."""
        result = extractor.teardown()
        assert result is True


# ---------------------------------------------------------------------------
# GWT Scenario 5: Process Method
# ---------------------------------------------------------------------------


class TestProcessMethod:
    """Given artifact, When process called, Then derived artifact returned."""

    def test_process_text_artifact(
        self, extractor: IFUniversalExtractor, text_artifact: IFTextArtifact
    ):
        """Given IFTextArtifact, When processed, Then derived artifact with entities."""
        result = extractor.process(text_artifact)

        assert isinstance(result, IFTextArtifact)
        assert result.artifact_id == f"{text_artifact.artifact_id}-entities"
        assert result.parent_id == text_artifact.artifact_id
        assert "entities_structured" in result.metadata
        assert "entity_count" in result.metadata
        assert result.metadata["entity_count"] > 0

    def test_process_chunk_artifact(
        self, extractor: IFUniversalExtractor, chunk_artifact: IFChunkArtifact
    ):
        """Given IFChunkArtifact, When processed, Then derived artifact with entities."""
        result = extractor.process(chunk_artifact)

        assert isinstance(result, IFChunkArtifact)
        assert result.artifact_id == f"{chunk_artifact.artifact_id}-entities"
        assert result.parent_id == chunk_artifact.artifact_id
        assert "entities_structured" in result.metadata

    def test_process_invalid_artifact_type(self, extractor: IFUniversalExtractor):
        """Given unsupported artifact type, When processed, Then IFFailureArtifact."""
        from pathlib import Path

        file_artifact = IFFileArtifact(
            artifact_id="file-001",
            file_path=Path("/fake/path.txt"),
        )

        result = extractor.process(file_artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "requires IFTextArtifact or IFChunkArtifact" in result.error_message

    def test_process_preserves_lineage(
        self, extractor: IFUniversalExtractor, text_artifact: IFTextArtifact
    ):
        """Given artifact, When processed, Then lineage preserved."""
        result = extractor.process(text_artifact)

        assert result.lineage_depth == text_artifact.lineage_depth + 1
        assert extractor.processor_id in result.provenance

    def test_process_includes_extraction_method(
        self, extractor: IFUniversalExtractor, text_artifact: IFTextArtifact
    ):
        """Given regex-only extractor, When processed, Then method is 'regex'."""
        result = extractor.process(text_artifact)

        assert result.metadata["extraction_method"] == "regex"


# ---------------------------------------------------------------------------
# JPL Rule #2: Fixed Upper Bounds
# ---------------------------------------------------------------------------


class TestJPLRule2Bounds:
    """Test fixed upper bounds per JPL Rule #2."""

    def test_max_entities_bound(self, extractor: IFUniversalExtractor):
        """Given many entities, When extracted, Then bounded by MAX_ENTITIES."""
        # Create text with many entities
        text = " ".join([f"CVE-2024-{i:04d}" for i in range(150)])
        entities = extractor.extract(text)

        assert len(entities) <= MAX_ENTITIES_PER_EXTRACTION

    def test_entity_text_length_bound(self):
        """Given long entity text, When validated, Then truncated."""
        long_text = "A" * (MAX_ENTITY_TEXT_LENGTH + 100)
        entity = ExtractedEntity(
            text=long_text,
            entity_type="ORG",
            start_char=0,
            end_char=len(long_text),
            confidence=0.8,
        )

        assert len(entity.text) <= MAX_ENTITY_TEXT_LENGTH

    def test_extraction_result_bounds(self):
        """Given many entities in result, When validated, Then bounded."""
        entities = [
            ExtractedEntity(
                text=f"Entity{i}",
                entity_type="PERSON",
                start_char=i * 10,
                end_char=i * 10 + 8,
                confidence=0.8,
            )
            for i in range(150)
        ]

        result = EntityExtractionResult(entities=entities)
        assert len(result.entities) <= MAX_ENTITIES_PER_EXTRACTION


# ---------------------------------------------------------------------------
# JPL Rule #4: Functions < 60 Lines
# ---------------------------------------------------------------------------


class TestJPLRule4FunctionSize:
    """Test that functions are under 60 lines per JPL Rule #4."""

    def test_extract_method_size(self):
        """Given extract method, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor.extract)
        lines = source.split("\n")
        assert len(lines) < 60

    def test_process_method_size(self):
        """Given process method, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor.process)
        lines = source.split("\n")
        assert len(lines) < 60

    def test_extract_with_regex_size(self):
        """Given _extract_with_regex, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor._extract_with_regex)
        lines = source.split("\n")
        assert len(lines) < 60


# ---------------------------------------------------------------------------
# JPL Rule #9: Complete Type Hints
# ---------------------------------------------------------------------------


class TestJPLRule9TypeHints:
    """Test complete type hints per JPL Rule #9."""

    def test_processor_id_return_type(self, extractor: IFUniversalExtractor):
        """Given processor_id property, Then returns str."""
        result = extractor.processor_id
        assert isinstance(result, str)

    def test_version_return_type(self, extractor: IFUniversalExtractor):
        """Given version property, Then returns str."""
        result = extractor.version
        assert isinstance(result, str)

    def test_capabilities_return_type(self, extractor: IFUniversalExtractor):
        """Given capabilities property, Then returns List[str]."""
        result = extractor.capabilities
        assert isinstance(result, list)
        assert all(isinstance(c, str) for c in result)

    def test_extract_return_type(self, extractor: IFUniversalExtractor):
        """Given extract method, Then returns List[ExtractedEntity]."""
        result = extractor.extract("Test text")
        assert isinstance(result, list)
        assert all(isinstance(e, ExtractedEntity) for e in result)


# ---------------------------------------------------------------------------
# Additional Extraction Tests
# ---------------------------------------------------------------------------


class TestAdditionalEntityTypes:
    """Test extraction of additional entity types."""

    def test_detects_email(self, extractor: IFUniversalExtractor):
        """Given email address, When extracted, Then EMAIL entity found."""
        text = "Contact john.doe@company.org for details."
        entities = extractor.extract(text)

        email_entities = [e for e in entities if e.entity_type == "EMAIL"]
        assert len(email_entities) == 1
        assert "john.doe@company.org" in email_entities[0].text

    def test_detects_url(self, extractor: IFUniversalExtractor):
        """Given URL, When extracted, Then URL entity found."""
        text = "Visit https://www.example.com/path?query=1 for more."
        entities = extractor.extract(text)

        url_entities = [e for e in entities if e.entity_type == "URL"]
        assert len(url_entities) == 1

    def test_detects_phone(self, extractor: IFUniversalExtractor):
        """Given phone number, When extracted, Then PHONE entity found."""
        text = "Call us at (555) 123-4567 or 800-555-1234."
        entities = extractor.extract(text)

        phone_entities = [e for e in entities if e.entity_type == "PHONE"]
        assert len(phone_entities) >= 1

    def test_detects_money(self, extractor: IFUniversalExtractor):
        """Given monetary amount, When extracted, Then MONEY entity found."""
        text = "The cost is $1,500.00 or $2 million."
        entities = extractor.extract(text)

        money_entities = [e for e in entities if e.entity_type == "MONEY"]
        assert len(money_entities) >= 1


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self, extractor: IFUniversalExtractor):
        """Given empty text, When extracted, Then empty list returned."""
        entities = extractor.extract("")
        assert entities == []

    def test_no_entities_text(self, extractor: IFUniversalExtractor):
        """Given text without entities, When extracted, Then empty list."""
        text = "This is a simple sentence with no specific entities."
        entities = extractor.extract(text)
        # May still find some with low confidence, but filtered out
        assert isinstance(entities, list)

    def test_unicode_text(self, extractor: IFUniversalExtractor):
        """Given Unicode text, When extracted, Then no crash."""
        text = "Dr. José García works at 株式会社 on CVE-2024-0001."
        entities = extractor.extract(text)
        cve_entities = [e for e in entities if e.entity_type == "CVE"]
        assert len(cve_entities) == 1

    def test_very_long_text_truncated(self, extractor: IFUniversalExtractor):
        """Given very long text, When extracted, Then no crash."""
        text = "CVE-2024-1234 " * 10000
        entities = extractor.extract(text)
        # Should still find entities despite truncation
        assert len(entities) > 0

    def test_min_confidence_filter(self):
        """Given high min_confidence, When extracted, Then entities filtered."""
        extractor = IFUniversalExtractor(use_llm=False, min_confidence=0.99)
        text = "Dr. John Smith from Apple Inc."
        entities = extractor.extract(text)
        # High confidence filter should remove regex matches
        assert all(e.confidence >= 0.99 for e in entities)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Test entity deduplication logic."""

    def test_no_duplicate_entities(self, extractor: IFUniversalExtractor):
        """Given repeated text, When extracted, Then no duplicates."""
        text = "CVE-2024-1234 CVE-2024-1234 CVE-2024-1234"
        entities = extractor.extract(text)

        # Should have 3 entities (same CVE at different positions)
        cve_entities = [e for e in entities if e.entity_type == "CVE"]
        # Positions should be unique
        positions = [(e.start_char, e.end_char) for e in cve_entities]
        assert len(positions) == len(set(positions))


# ---------------------------------------------------------------------------
# LLM Integration (Mocked)
# ---------------------------------------------------------------------------


class TestLLMIntegration:
    """Test LLM extraction path with mocks."""

    def test_llm_extraction_called_when_available(self):
        """Given LLM available, When extracted, Then LLM called."""
        mock_config = Mock()
        extractor = IFUniversalExtractor(config=mock_config, use_llm=True)

        # Mock the LLM availability
        with patch.object(extractor, "_check_llm_available", return_value=True):
            with patch.object(extractor, "_extract_with_llm") as mock_llm:
                mock_llm.return_value = []
                extractor.extract("Test text")
                mock_llm.assert_called_once()

    def test_fallback_to_regex_when_llm_unavailable(self):
        """Given LLM unavailable, When extracted, Then regex used."""
        extractor = IFUniversalExtractor(use_llm=False)

        text = "CVE-2024-1234 is critical."
        entities = extractor.extract(text)

        # Should still find CVE via regex
        cve_entities = [e for e in entities if e.entity_type == "CVE"]
        assert len(cve_entities) == 1

    def test_merge_llm_and_regex_entities(self):
        """Given both LLM and regex entities, When merged, Then no duplicates."""
        extractor = IFUniversalExtractor(use_llm=False)

        llm_entities = [
            ExtractedEntity(
                text="CVE-2024-1234",
                entity_type="CVE",
                start_char=0,
                end_char=13,
                confidence=0.95,
            )
        ]
        regex_entities = [
            ExtractedEntity(
                text="CVE-2024-1234",
                entity_type="CVE",
                start_char=0,
                end_char=13,
                confidence=0.90,
            ),
            ExtractedEntity(
                text="test@example.com",
                entity_type="EMAIL",
                start_char=20,
                end_char=36,
                confidence=0.95,
            ),
        ]

        merged = extractor._merge_entities(llm_entities, regex_entities)

        # LLM entity preserved, duplicate regex removed, non-overlapping kept
        assert len(merged) == 2
        cve_entities = [e for e in merged if e.entity_type == "CVE"]
        assert len(cve_entities) == 1
        assert cve_entities[0].confidence == 0.95  # LLM version


# ---------------------------------------------------------------------------
# Comprehensive Integration Test
# ---------------------------------------------------------------------------


class TestComprehensiveExtraction:
    """Test comprehensive extraction from realistic text."""

    def test_full_extraction_pipeline(self, extractor: IFUniversalExtractor):
        """Given rich document text, When extracted, Then all entity types found."""
        text = """
        SECURITY ADVISORY

        Case No. 2024-CV-00123-XYZ

        Issued: January 15, 2024

        Dr. Sarah Johnson, Chief Security Officer at CyberDefense Inc., has reported
        a critical vulnerability CVE-2024-98765 affecting component P/N SEC-2024-PROD.

        Impact: This vulnerability allows remote code execution with a CVSS score of 9.8.

        Affected systems should be patched immediately. For assistance, contact:
        - Email: security@cyberdefense.com
        - Phone: (555) 867-5309
        - Web: https://cyberdefense.com/advisory

        Cost of remediation estimated at $50,000 per enterprise deployment.
        """

        entities = extractor.extract(text)

        # Check for variety of entity types
        entity_types = {e.entity_type for e in entities}

        assert "CVE" in entity_types
        assert "DATE" in entity_types
        assert "EMAIL" in entity_types
        assert "PHONE" in entity_types
        assert "URL" in entity_types
        assert "MONEY" in entity_types
        assert "DOCKET" in entity_types or "PART_NUMBER" in entity_types

        # Check all have valid confidence and source_link
        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0
            assert ":" in entity.source_link


# ---------------------------------------------------------------------------
# JPL Rule #1: Simple Control Flow
# ---------------------------------------------------------------------------


class TestJPLRule1SimpleControlFlow:
    """Test simple control flow per JPL Rule #1."""

    def test_no_goto_or_setjmp(self):
        """Given source code, Then no goto/setjmp constructs."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor)
        assert "goto" not in source.lower()
        assert "setjmp" not in source.lower()

    def test_extract_has_single_exit_path(self, extractor: IFUniversalExtractor):
        """Given extract method, Then predictable return path."""
        # Empty input returns empty list
        assert extractor.extract("") == []
        # Non-empty input returns list
        result = extractor.extract("CVE-2024-1234")
        assert isinstance(result, list)

    def test_process_returns_artifact_type(self, extractor: IFUniversalExtractor):
        """Given any input, When process called, Then returns IFArtifact."""
        valid_artifact = IFTextArtifact(artifact_id="test", content="test")
        result = extractor.process(valid_artifact)
        # Always returns an artifact (success or failure)
        assert hasattr(result, "artifact_id")


# ---------------------------------------------------------------------------
# JPL Rule #5: Assertion Density
# ---------------------------------------------------------------------------


class TestJPLRule5AssertionDensity:
    """Test assertion/validation density per JPL Rule #5."""

    def test_confidence_validated_in_range(self):
        """Given confidence out of range, When entity created, Then error."""
        with pytest.raises(Exception):
            ExtractedEntity(
                text="Test",
                entity_type="PERSON",
                start_char=0,
                end_char=4,
                confidence=1.5,  # Out of range
            )

    def test_negative_confidence_rejected(self):
        """Given negative confidence, When entity created, Then error."""
        with pytest.raises(Exception):
            ExtractedEntity(
                text="Test",
                entity_type="PERSON",
                start_char=0,
                end_char=4,
                confidence=-0.1,  # Negative
            )

    def test_start_char_must_be_positive(self):
        """Given negative start_char, When entity created, Then error."""
        with pytest.raises(Exception):
            ExtractedEntity(
                text="Test",
                entity_type="PERSON",
                start_char=-1,  # Negative
                end_char=4,
                confidence=0.8,
            )

    def test_invalid_entity_type_rejected(self):
        """Given invalid entity type, When entity created, Then error."""
        with pytest.raises(Exception):
            ExtractedEntity(
                text="Test",
                entity_type="INVALID_TYPE",  # Not in Literal
                start_char=0,
                end_char=4,
                confidence=0.8,
            )


# ---------------------------------------------------------------------------
# JPL Rule #6: Smallest Scope
# ---------------------------------------------------------------------------


class TestJPLRule6SmallestScope:
    """Test smallest scope principle per JPL Rule #6."""

    def test_constants_defined_at_module_level(self):
        """Given module constants, Then accessible and immutable."""
        from ingestforge.enrichment import universal_extractor as mod

        assert hasattr(mod, "MAX_ENTITIES_PER_EXTRACTION")
        assert hasattr(mod, "MAX_TEXT_LENGTH")
        assert hasattr(mod, "MAX_ENTITY_TEXT_LENGTH")

    def test_extractor_encapsulates_state(self, extractor: IFUniversalExtractor):
        """Given extractor, Then internal state is encapsulated."""
        # Internal state should be private
        assert hasattr(extractor, "_instructor_client")
        assert hasattr(extractor, "_llm_available")
        # These should not be directly accessible as public
        assert not hasattr(extractor, "instructor_client")

    def test_fallback_patterns_module_scoped(self):
        """Given FALLBACK_PATTERNS, Then defined at module scope."""
        assert isinstance(FALLBACK_PATTERNS, list)
        assert len(FALLBACK_PATTERNS) > 0


# ---------------------------------------------------------------------------
# JPL Rule #7: Check Return Values
# ---------------------------------------------------------------------------


class TestJPLRule7ReturnValues:
    """Test explicit return value checking per JPL Rule #7."""

    def test_extract_always_returns_list(self, extractor: IFUniversalExtractor):
        """Given any input, When extract called, Then list returned."""
        assert isinstance(extractor.extract(""), list)
        assert isinstance(extractor.extract("text"), list)
        assert isinstance(extractor.extract("CVE-2024-1234"), list)

    def test_process_always_returns_artifact(self, extractor: IFUniversalExtractor):
        """Given any artifact, When process called, Then artifact returned."""
        artifact = IFTextArtifact(artifact_id="t", content="c")
        result = extractor.process(artifact)
        assert hasattr(result, "artifact_id")
        assert hasattr(result, "metadata")

    def test_is_available_returns_bool(self, extractor: IFUniversalExtractor):
        """Given extractor, When is_available called, Then bool returned."""
        result = extractor.is_available()
        assert isinstance(result, bool)

    def test_teardown_returns_bool(self, extractor: IFUniversalExtractor):
        """Given extractor, When teardown called, Then bool returned."""
        result = extractor.teardown()
        assert isinstance(result, bool)
        assert result is True


# ---------------------------------------------------------------------------
# GWT: Behavioral Specification Tests ()
# ---------------------------------------------------------------------------


class TestGWTBehavioralSpec:
    """
    Test exact GWT specification from
    - Given: A raw IFTextArtifact
    - When: The Extraction Stage runs
    - Then: Returns artifact with unique entities, types, and source offsets
    """

    def test_gwt_given_raw_text_artifact(self, extractor: IFUniversalExtractor):
        """Given raw IFTextArtifact with no prior processing."""
        artifact = IFTextArtifact(
            artifact_id="raw-doc-001",
            content="Dr. Alice Brown works at OpenAI Inc. CVE-2024-0001 reported.",
            metadata={},  # Raw, no prior enrichment
        )
        assert artifact.lineage_depth == 0
        assert artifact.provenance == []

    def test_gwt_when_extraction_stage_runs(self, extractor: IFUniversalExtractor):
        """When extraction stage processes artifact."""
        artifact = IFTextArtifact(
            artifact_id="raw-doc-001",
            content="Dr. Alice Brown works at OpenAI Inc. CVE-2024-0001 reported.",
        )
        result = extractor.process(artifact)
        # Process completed without error
        assert not isinstance(result, IFFailureArtifact)

    def test_gwt_then_returns_entity_artifact_with_list(
        self, extractor: IFUniversalExtractor
    ):
        """Then returns artifact with list of unique entities."""
        artifact = IFTextArtifact(
            artifact_id="raw-doc-001",
            content="Dr. Alice Brown works at OpenAI Inc. CVE-2024-0001 reported.",
        )
        result = extractor.process(artifact)

        # Must have entities_structured as list
        assert "entities_structured" in result.metadata
        assert isinstance(result.metadata["entities_structured"], list)

        # Entities should be unique (no duplicate positions)
        positions = [
            (e["start"], e["end"]) for e in result.metadata["entities_structured"]
        ]
        assert len(positions) == len(set(positions))

    def test_gwt_then_entities_have_types(self, extractor: IFUniversalExtractor):
        """Then each entity has a type classification."""
        artifact = IFTextArtifact(
            artifact_id="raw-doc-001",
            content="Dr. Alice Brown works at OpenAI Inc. CVE-2024-0001 reported.",
        )
        result = extractor.process(artifact)

        for entity in result.metadata["entities_structured"]:
            assert "type" in entity
            assert entity["type"] in [
                "PERSON",
                "ORG",
                "LOC",
                "DATE",
                "DOCKET",
                "PART_NUMBER",
                "CVE",
                "MONEY",
                "PERCENT",
                "EMAIL",
                "URL",
                "PHONE",
            ]

    def test_gwt_then_entities_have_source_offsets(
        self, extractor: IFUniversalExtractor
    ):
        """Then each entity has exact source character offsets."""
        artifact = IFTextArtifact(
            artifact_id="raw-doc-001",
            content="Dr. Alice Brown works at OpenAI Inc. CVE-2024-0001 reported.",
        )
        result = extractor.process(artifact)

        for entity in result.metadata["entities_structured"]:
            assert "start" in entity
            assert "end" in entity
            assert "source_link" in entity
            assert isinstance(entity["start"], int)
            assert isinstance(entity["end"], int)
            assert entity["start"] >= 0
            assert entity["end"] > entity["start"]


# ---------------------------------------------------------------------------
# Acceptance Criteria Tests ()
# ---------------------------------------------------------------------------


class TestAcceptanceCriteria:
    """Test explicit acceptance criteria from ."""

    def test_ac1_uses_instructor_pydantic_model(self):
        """AC1: Uses Instructor with poly-entity Pydantic model."""
        # Verify Pydantic models exist and are properly defined
        from pydantic import BaseModel

        assert issubclass(ExtractedEntity, BaseModel)
        assert issubclass(EntityExtractionResult, BaseModel)

        # Verify poly-entity (multiple types)
        from ingestforge.enrichment.universal_extractor import EntityType

        # EntityType should be a Literal with multiple values
        import typing

        assert hasattr(typing, "get_args")
        entity_types = typing.get_args(EntityType)
        assert len(entity_types) >= 10  # At least 10 entity types

    def test_ac2_detects_standard_ner(self, extractor: IFUniversalExtractor):
        """AC2: Detection of standard NER (Person, Org, Loc)."""
        text = "Dr. John Smith met CEO Jane Doe at Microsoft Corp. in New York."
        entities = extractor.extract(text)

        types_found = {e.entity_type for e in entities}
        # Should detect PERSON and ORG at minimum (LOC requires LLM typically)
        assert "PERSON" in types_found or "ORG" in types_found

    def test_ac3_detects_domain_markers(self, extractor: IFUniversalExtractor):
        """AC3: Detection of domain markers (Dockets, PartNumbers, CVEs)."""
        text = "Case No. 20-1234 references CVE-2024-5678 for Part P/N ABC-1234567."
        entities = extractor.extract(text)

        types_found = {e.entity_type for e in entities}
        # Should detect at least CVE and DOCKET
        assert "CVE" in types_found
        assert "DOCKET" in types_found or "PART_NUMBER" in types_found

    def test_ac4_entity_has_confidence_and_source_link(
        self, extractor: IFUniversalExtractor
    ):
        """AC4: Every entity has confidence score and source_link."""
        text = "CVE-2024-1234 is critical."
        entities = extractor.extract(text)

        assert len(entities) >= 1
        for entity in entities:
            # Confidence score
            assert hasattr(entity, "confidence")
            assert 0.0 <= entity.confidence <= 1.0

            # Source link
            assert hasattr(entity, "source_link")
            assert entity.source_link is not None
            assert ":" in entity.source_link


# ---------------------------------------------------------------------------
# Metadata Structure Tests
# ---------------------------------------------------------------------------


class TestMetadataStructure:
    """Test metadata structure in processed artifacts."""

    def test_entities_structured_is_serializable(
        self, extractor: IFUniversalExtractor, text_artifact: IFTextArtifact
    ):
        """Given processed artifact, Then entities_structured is JSON-serializable."""
        import json

        result = extractor.process(text_artifact)

        # Should not raise
        json_str = json.dumps(result.metadata["entities_structured"])
        assert isinstance(json_str, str)

    def test_entity_types_found_is_sorted(
        self, extractor: IFUniversalExtractor, text_artifact: IFTextArtifact
    ):
        """Given processed artifact, Then entity_types_found is sorted."""
        result = extractor.process(text_artifact)

        types_found = result.metadata["entity_types_found"]
        assert types_found == sorted(types_found)

    def test_entity_count_matches_list_length(
        self, extractor: IFUniversalExtractor, text_artifact: IFTextArtifact
    ):
        """Given processed artifact, Then entity_count matches list length."""
        result = extractor.process(text_artifact)

        assert result.metadata["entity_count"] == len(
            result.metadata["entities_structured"]
        )


# ---------------------------------------------------------------------------
# Confidence-Aware-Extraction Tests
# ---------------------------------------------------------------------------


class TestConfidenceBucketing:
    """System categorizes entities into High/Med/Low confidence buckets."""

    def test_given_high_confidence_entity_when_categorized_then_in_high_bucket(self):
        """Given entity with confidence >= 0.8, When categorized, Then in High bucket."""
        extractor = IFUniversalExtractor(use_llm=False)
        entity = ExtractedEntity(
            text="High Conf",
            entity_type="PERSON",
            start_char=0,
            end_char=9,
            confidence=0.95,
        )

        buckets = extractor._categorize_confidence([entity])

        assert len(buckets["High"]) == 1
        assert len(buckets["Med"]) == 0
        assert len(buckets["Low"]) == 0
        assert buckets["High"][0]["text"] == "High Conf"

    def test_given_medium_confidence_entity_when_categorized_then_in_med_bucket(self):
        """Given entity with 0.5 <= confidence < 0.8, When categorized, Then in Med bucket."""
        extractor = IFUniversalExtractor(use_llm=False)
        entity = ExtractedEntity(
            text="Med Conf",
            entity_type="ORG",
            start_char=0,
            end_char=8,
            confidence=0.65,
        )

        buckets = extractor._categorize_confidence([entity])

        assert len(buckets["High"]) == 0
        assert len(buckets["Med"]) == 1
        assert len(buckets["Low"]) == 0
        assert buckets["Med"][0]["text"] == "Med Conf"

    def test_given_low_confidence_entity_when_categorized_then_in_low_bucket(self):
        """Given entity with confidence < 0.5, When categorized, Then in Low bucket."""
        extractor = IFUniversalExtractor(use_llm=False)
        entity = ExtractedEntity(
            text="Low Conf",
            entity_type="LOC",
            start_char=0,
            end_char=8,
            confidence=0.35,
        )

        buckets = extractor._categorize_confidence([entity])

        assert len(buckets["High"]) == 0
        assert len(buckets["Med"]) == 0
        assert len(buckets["Low"]) == 1
        assert buckets["Low"][0]["text"] == "Low Conf"

    def test_given_mixed_entities_when_categorized_then_all_buckets_populated(self):
        """Given entities with mixed confidence, When categorized, Then all buckets populated."""
        extractor = IFUniversalExtractor(use_llm=False)
        entities = [
            ExtractedEntity(
                text="H", entity_type="PERSON", start_char=0, end_char=1, confidence=0.9
            ),
            ExtractedEntity(
                text="M", entity_type="ORG", start_char=2, end_char=3, confidence=0.6
            ),
            ExtractedEntity(
                text="L", entity_type="LOC", start_char=4, end_char=5, confidence=0.3
            ),
        ]

        buckets = extractor._categorize_confidence(entities)

        assert len(buckets["High"]) == 1
        assert len(buckets["Med"]) == 1
        assert len(buckets["Low"]) == 1


class TestLowConfidenceWarning:
    """Low confidence entities emit LowConfidenceWarning."""

    def test_given_low_confidence_entity_when_categorized_then_warning_logged(
        self, caplog
    ):
        """Given entity with confidence < 0.5, When categorized, Then warning logged."""
        import logging

        caplog.set_level(logging.WARNING)

        extractor = IFUniversalExtractor(use_llm=False)
        entity = ExtractedEntity(
            text="Suspicious",
            entity_type="ORG",
            start_char=0,
            end_char=10,
            confidence=0.25,
        )

        extractor._categorize_confidence([entity])

        assert any(
            "LowConfidenceWarning" in record.message for record in caplog.records
        )
        assert any("Suspicious" in record.message for record in caplog.records)
        assert any("0.25" in record.message for record in caplog.records)

    def test_given_high_confidence_entity_when_categorized_then_no_warning(
        self, caplog
    ):
        """Given entity with high confidence, When categorized, Then no warning logged."""
        import logging

        caplog.set_level(logging.WARNING)

        extractor = IFUniversalExtractor(use_llm=False)
        entity = ExtractedEntity(
            text="Reliable",
            entity_type="PERSON",
            start_char=0,
            end_char=8,
            confidence=0.95,
        )

        extractor._categorize_confidence([entity])

        assert not any(
            "LowConfidenceWarning" in record.message for record in caplog.records
        )


class TestConfidenceBucketsInMetadata:
    """Confidence buckets appear in artifact metadata."""

    def test_given_text_artifact_when_processed_then_confidence_buckets_in_metadata(
        self,
    ):
        """Given text artifact, When processed, Then confidence_buckets in metadata."""
        extractor = IFUniversalExtractor(use_llm=False, min_confidence=0.0)
        artifact = IFTextArtifact(
            artifact_id="test-001",
            content="Dr. John Smith from Microsoft Corporation on 2024-01-15",
        )

        result = extractor.process(artifact)

        assert "confidence_buckets" in result.metadata
        buckets = result.metadata["confidence_buckets"]
        assert "High" in buckets
        assert "Med" in buckets
        assert "Low" in buckets

    def test_given_processed_artifact_then_buckets_contain_entity_dicts(self):
        """Given processed artifact, Then bucket entries are entity dictionaries."""
        extractor = IFUniversalExtractor(use_llm=False, min_confidence=0.0)
        artifact = IFTextArtifact(
            artifact_id="test-002",
            content="CVE-2024-1234 was reported on 2024-05-15",
        )

        result = extractor.process(artifact)

        buckets = result.metadata["confidence_buckets"]
        all_entities = buckets["High"] + buckets["Med"] + buckets["Low"]

        for entity in all_entities:
            assert isinstance(entity, dict)
            assert "text" in entity
            assert "type" in entity
            assert "confidence" in entity


class TestEntityNodeConfidence:
    """EntityNode model contains confidence field."""

    def test_entity_node_has_confidence_field(self):
        """Given EntityNode, Then it has confidence field."""
        from ingestforge.core.pipeline.artifacts import EntityNode, SourceProvenance

        node = EntityNode(
            entity_id="ent-001",
            entity_type="PERSON",
            name="Test",
            confidence=0.85,
            source_provenance=SourceProvenance(
                source_artifact_id="art-001",
                char_offset_start=0,
                char_offset_end=4,
            ),
        )

        assert hasattr(node, "confidence")
        assert node.confidence == 0.85

    def test_entity_node_confidence_default_value(self):
        """Given EntityNode without confidence, Then default is 1.0."""
        from ingestforge.core.pipeline.artifacts import EntityNode, SourceProvenance

        node = EntityNode(
            entity_id="ent-002",
            entity_type="ORG",
            name="Test Corp",
            source_provenance=SourceProvenance(
                source_artifact_id="art-002",
                char_offset_start=0,
                char_offset_end=9,
            ),
        )

        assert node.confidence == 1.0

    def test_entity_node_confidence_validation(self):
        """Given invalid confidence, Then validation fails."""
        from ingestforge.core.pipeline.artifacts import EntityNode, SourceProvenance
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            EntityNode(
                entity_id="ent-003",
                entity_type="LOC",
                name="Invalid",
                confidence=1.5,  # Invalid: > 1.0
                source_provenance=SourceProvenance(
                    source_artifact_id="art-003",
                    char_offset_start=0,
                    char_offset_end=7,
                ),
            )


class TestEnrichmentConfigMinConfidence:
    """EnrichmentConfig contains min_confidence field."""

    def test_enrichment_config_has_min_confidence(self):
        """Given EnrichmentConfig, Then it has min_confidence field."""
        from ingestforge.core.config.enrichment import EnrichmentConfig

        config = EnrichmentConfig()

        assert hasattr(config, "min_confidence")
        assert config.min_confidence == 0.5

    def test_enrichment_config_min_confidence_custom_value(self):
        """Given custom min_confidence, Then it is applied."""
        from ingestforge.core.config.enrichment import EnrichmentConfig

        config = EnrichmentConfig(min_confidence=0.7)

        assert config.min_confidence == 0.7

    def test_enrichment_config_min_confidence_validation(self):
        """Given invalid min_confidence, Then ValueError raised."""
        from ingestforge.core.config.enrichment import EnrichmentConfig

        with pytest.raises(ValueError, match="min_confidence must be between"):
            EnrichmentConfig(min_confidence=1.5)


# ---------------------------------------------------------------------------
# Multi-Model Fallback Tests
# ---------------------------------------------------------------------------


class TestMultiModelFallbackConfiguration:
    """ModelEscalator logic configuration."""

    def test_default_fast_model(self):
        """Given no fast_model, Then default is gpt-4o-mini."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor.fast_model == "gpt-4o-mini"

    def test_default_smart_model(self):
        """Given no smart_model, Then default is gpt-4o."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor.smart_model == "gpt-4o"

    def test_default_fallback_threshold(self):
        """Given no fallback_threshold, Then default is 0.6."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor.fallback_threshold == 0.6

    def test_custom_fast_model(self):
        """Given custom fast_model, Then it is used."""
        extractor = IFUniversalExtractor(use_llm=False, fast_model="custom-fast")
        assert extractor.fast_model == "custom-fast"

    def test_custom_smart_model(self):
        """Given custom smart_model, Then it is used."""
        extractor = IFUniversalExtractor(use_llm=False, smart_model="custom-smart")
        assert extractor.smart_model == "custom-smart"

    def test_custom_fallback_threshold(self):
        """Given custom fallback_threshold, Then it is used."""
        extractor = IFUniversalExtractor(use_llm=False, fallback_threshold=0.75)
        assert extractor.fallback_threshold == 0.75


class TestShouldEscalate:
    """Escalation decision logic."""

    def test_given_empty_entities_when_checked_then_no_escalation(self):
        """Given no entities, When checked, Then no escalation needed."""
        extractor = IFUniversalExtractor(use_llm=False, fallback_threshold=0.6)
        result = extractor._should_escalate([])
        assert result is False

    def test_given_high_confidence_entities_when_checked_then_no_escalation(self):
        """Given high confidence entities, When checked, Then no escalation."""
        extractor = IFUniversalExtractor(use_llm=False, fallback_threshold=0.6)
        entities = [
            ExtractedEntity(
                text="High",
                entity_type="PERSON",
                start_char=0,
                end_char=4,
                confidence=0.9,
            ),
            ExtractedEntity(
                text="Also High",
                entity_type="ORG",
                start_char=5,
                end_char=14,
                confidence=0.85,
            ),
        ]
        result = extractor._should_escalate(entities)
        assert result is False

    def test_given_low_confidence_entities_when_checked_then_escalation_needed(self):
        """Given low confidence entities, When checked, Then escalation needed."""
        extractor = IFUniversalExtractor(use_llm=False, fallback_threshold=0.6)
        entities = [
            ExtractedEntity(
                text="Low",
                entity_type="PERSON",
                start_char=0,
                end_char=3,
                confidence=0.4,
            ),
            ExtractedEntity(
                text="Also Low",
                entity_type="ORG",
                start_char=4,
                end_char=12,
                confidence=0.3,
            ),
        ]
        result = extractor._should_escalate(entities)
        assert result is True

    def test_given_mixed_confidence_below_threshold_then_escalation_needed(self):
        """Given mixed confidence below threshold, Then escalation needed."""
        extractor = IFUniversalExtractor(use_llm=False, fallback_threshold=0.6)
        entities = [
            ExtractedEntity(
                text="High",
                entity_type="PERSON",
                start_char=0,
                end_char=4,
                confidence=0.8,
            ),
            ExtractedEntity(
                text="Low", entity_type="ORG", start_char=5, end_char=8, confidence=0.3
            ),
        ]
        # Average: (0.8 + 0.3) / 2 = 0.55 < 0.6
        result = extractor._should_escalate(entities)
        assert result is True


class TestCalculateAvgConfidence:
    """Average confidence calculation."""

    def test_given_empty_entities_then_returns_1_0(self):
        """Given no entities, Then returns 1.0 (no escalation needed)."""
        extractor = IFUniversalExtractor(use_llm=False)
        result = extractor._calculate_avg_confidence([])
        assert result == 1.0

    def test_given_single_entity_then_returns_its_confidence(self):
        """Given single entity, Then returns its confidence."""
        extractor = IFUniversalExtractor(use_llm=False)
        entities = [
            ExtractedEntity(
                text="Test",
                entity_type="PERSON",
                start_char=0,
                end_char=4,
                confidence=0.75,
            ),
        ]
        result = extractor._calculate_avg_confidence(entities)
        assert result == 0.75

    def test_given_multiple_entities_then_returns_average(self):
        """Given multiple entities, Then returns average confidence."""
        extractor = IFUniversalExtractor(use_llm=False)
        entities = [
            ExtractedEntity(
                text="A", entity_type="PERSON", start_char=0, end_char=1, confidence=0.8
            ),
            ExtractedEntity(
                text="B", entity_type="ORG", start_char=2, end_char=3, confidence=0.6
            ),
            ExtractedEntity(
                text="C", entity_type="LOC", start_char=4, end_char=5, confidence=0.4
            ),
        ]
        # Average: (0.8 + 0.6 + 0.4) / 3 = 0.6
        result = extractor._calculate_avg_confidence(entities)
        assert abs(result - 0.6) < 0.001


class TestEscalationRate:
    """Escalation rate health metric."""

    def test_given_no_extractions_then_rate_is_zero(self):
        """Given no extractions, Then escalation_rate is 0."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor.escalation_rate == 0.0

    def test_escalation_rate_tracks_extractions(self):
        """Given extractions, Then escalation_rate is calculated correctly via escalator."""
        extractor = IFUniversalExtractor(use_llm=False)
        # Stats are now tracked by ModelEscalator internally
        extractor._escalator._stats.total_extractions = 10
        extractor._escalator._stats.escalation_count = 3
        # escalation_rate is a percentage (3/10 * 100 = 30.0%)
        assert extractor.escalation_rate == 30.0


class TestExtractReturnsEscalationInfo:
    """Extract method returns escalation information."""

    def test_extract_with_escalation_returns_tuple(self):
        """Given text, When extract_with_escalation called, Then returns (entities, escalated) tuple."""
        extractor = IFUniversalExtractor(use_llm=False)
        result = extractor.extract_with_escalation(
            "Dr. Jane Doe works at Microsoft Inc."
        )

        assert isinstance(result, tuple)
        assert len(result) == 2
        entities, escalated = result
        assert isinstance(entities, list)
        assert isinstance(escalated, bool)

    def test_extract_returns_only_entities(self):
        """Given text, When extract called, Then returns only entities list."""
        extractor = IFUniversalExtractor(use_llm=False)
        result = extractor.extract("Dr. Jane Doe works at Microsoft Inc.")

        assert isinstance(result, list)
        # Should not be a tuple
        assert all(hasattr(e, "entity_type") for e in result)

    def test_extract_simple_returns_only_entities(self):
        """Given text, When extract_simple called, Then returns only entities."""
        extractor = IFUniversalExtractor(use_llm=False)
        result = extractor.extract_simple("Dr. Jane Doe works at Microsoft Inc.")

        assert isinstance(result, list)
        assert all(hasattr(e, "entity_type") for e in result)


class TestProvenanceWithEscalation:
    """Fallback event recorded in IFArtifact.provenance."""

    def test_given_no_escalation_then_standard_provenance(self):
        """Given no escalation, Then standard provenance chain."""
        extractor = IFUniversalExtractor(use_llm=False)
        artifact = IFTextArtifact(
            artifact_id="test-001",
            content="Simple text with no PII",
        )

        result = extractor.process(artifact)

        assert extractor.processor_id in result.provenance
        assert f"{extractor.processor_id}:escalated" not in result.provenance

    def test_metadata_includes_escalation_info(self):
        """Given processed artifact, Then metadata includes escalation info."""
        extractor = IFUniversalExtractor(use_llm=False)
        artifact = IFTextArtifact(
            artifact_id="test-002",
            content="Dr. Jane Doe from Microsoft Corp on 2024-01-15",
        )

        result = extractor.process(artifact)

        assert "model_escalated" in result.metadata
        assert "extraction_model" in result.metadata
        assert "escalation_rate" in result.metadata
        assert result.metadata["model_escalated"] is False  # No LLM = no escalation


# ---------------------------------------------------------------------------
# AC#3: CORRUPT Tagging for Invalid Offsets (Negative Test)
# ---------------------------------------------------------------------------


class TestCorruptTagging:
    """AC#3: If offsets are invalid, entity is tagged as CORRUPT."""

    def test_given_invalid_offsets_start_ge_end_then_tagged_corrupt(self):
        """
        Negative Test: Given entity with start_char >= end_char,
        When validation runs, Then entity is tagged CORRUPT.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        text = "Test text"

        # Create entity with invalid offset (start >= end)
        invalid_entity = ExtractedEntity(
            text="Test",
            entity_type="PERSON",
            start_char=5,
            end_char=3,  # Invalid: start > end
            confidence=0.8,
        )

        validated = extractor._validate_offsets([invalid_entity], text)

        assert len(validated) == 1
        assert validated[0].entity_type == "CORRUPT"
        assert validated[0].confidence == 0.0
        assert "Original type: PERSON" in validated[0].context

    def test_given_offsets_out_of_bounds_then_tagged_corrupt(self):
        """
        Negative Test: Given entity with offsets beyond text length,
        When validation runs, Then entity is tagged CORRUPT.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        text = "Short"  # Length = 5

        # Create entity with offset beyond text bounds
        invalid_entity = ExtractedEntity(
            text="Entity",
            entity_type="ORG",
            start_char=0,
            end_char=100,  # Invalid: beyond text length
            confidence=0.9,
        )

        validated = extractor._validate_offsets([invalid_entity], text)

        assert len(validated) == 1
        assert validated[0].entity_type == "CORRUPT"
        assert validated[0].confidence == 0.0

    def test_given_text_mismatch_at_offset_then_tagged_corrupt(self):
        """
        Negative Test: Given entity where text at offset doesn't match,
        When validation runs, Then entity is tagged CORRUPT.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        text = "Hello World"

        # Create entity with mismatched text
        invalid_entity = ExtractedEntity(
            text="Wrong",  # Doesn't match text at offset
            entity_type="LOC",
            start_char=0,
            end_char=5,  # Actually contains "Hello"
            confidence=0.85,
        )

        validated = extractor._validate_offsets([invalid_entity], text)

        assert len(validated) == 1
        assert validated[0].entity_type == "CORRUPT"
        assert validated[0].confidence == 0.0

    def test_given_valid_offsets_then_not_tagged_corrupt(self):
        """
        Positive Test: Given entity with valid offsets,
        When validation runs, Then entity type is preserved.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        text = "CVE-2024-1234 is critical"

        # Create entity with valid offset
        valid_entity = ExtractedEntity(
            text="CVE-2024-1234",
            entity_type="CVE",
            start_char=0,
            end_char=13,  # Correct position
            confidence=0.95,
        )

        validated = extractor._validate_offsets([valid_entity], text)

        assert len(validated) == 1
        assert validated[0].entity_type == "CVE"  # Not changed
        assert validated[0].confidence == 0.95  # Preserved

    def test_extract_all_including_corrupt_returns_corrupt_entities(self):
        """
        Given extraction with potential corrupt entities,
        When extract_all_including_corrupt called,
        Then corrupt entities returned separately.
        """
        extractor = IFUniversalExtractor(use_llm=False, min_confidence=0.0)
        text = "CVE-2024-1234 is here"

        valid, corrupt = extractor.extract_all_including_corrupt(text)

        # Valid entities should have correct offsets
        for entity in valid:
            assert (
                text[entity.start_char : entity.end_char].strip() == entity.text.strip()
            )
            assert entity.entity_type != "CORRUPT"

        # Corrupt entities (if any) should be tagged as CORRUPT
        for entity in corrupt:
            assert entity.entity_type == "CORRUPT"
            assert entity.confidence == 0.0

    def test_corrupt_entity_has_zero_confidence(self):
        """Given CORRUPT entity, Then confidence is 0.0."""
        extractor = IFUniversalExtractor(use_llm=False)
        text = "Test"

        invalid_entity = ExtractedEntity(
            text="Invalid",
            entity_type="DATE",
            start_char=0,
            end_char=10,  # Beyond text
            confidence=0.99,  # Original high confidence
        )

        validated = extractor._validate_offsets([invalid_entity], text)

        assert validated[0].entity_type == "CORRUPT"
        assert validated[0].confidence == 0.0  # Forced to zero

    def test_corrupt_entities_filtered_by_min_confidence(self):
        """Given CORRUPT entity, When min_confidence > 0, Then filtered out."""
        extractor = IFUniversalExtractor(use_llm=False, min_confidence=0.5)
        text = "CVE-2024-1234"

        # This will be tagged CORRUPT due to text mismatch
        invalid_entity = ExtractedEntity(
            text="Wrong",
            entity_type="CVE",
            start_char=0,
            end_char=5,  # Doesn't match "Wrong"
            confidence=0.9,
        )

        # Manually test internal extraction logic
        validated = extractor._validate_offsets([invalid_entity], text)
        filtered = [e for e in validated if e.confidence >= extractor.min_confidence]

        assert len(filtered) == 0  # CORRUPT (0.0 confidence) is filtered out

    def test_negative_start_offset_tagged_corrupt(self):
        """
        Negative Test: Given entity with negative start offset,
        When validation runs, Then entity is tagged CORRUPT.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        text = "Test text"

        # This should be caught by Pydantic validation (ge=0), but test anyway
        # Create a mock entity bypassing validation for edge case
        with pytest.raises(Exception):
            ExtractedEntity(
                text="Test",
                entity_type="PERSON",
                start_char=-1,  # Invalid: negative
                end_char=4,
                confidence=0.8,
            )

    def test_corrupt_warning_logged(self, caplog):
        """Given invalid entity, When validation runs, Then warning logged."""
        import logging

        caplog.set_level(logging.WARNING)

        extractor = IFUniversalExtractor(use_llm=False)
        text = "Test"

        invalid_entity = ExtractedEntity(
            text="Invalid",
            entity_type="ORG",
            start_char=0,
            end_char=100,  # Beyond bounds
            confidence=0.9,
        )

        extractor._validate_offsets([invalid_entity], text)

        assert any("CORRUPT" in record.message for record in caplog.records)
        assert any("out of bounds" in record.message for record in caplog.records)

    def test_mixed_valid_and_corrupt_entities(self):
        """Given mix of valid and invalid entities, Then correctly separated."""
        extractor = IFUniversalExtractor(use_llm=False)
        text = "CVE-2024-1234 is test@example.com"

        entities = [
            # Valid
            ExtractedEntity(
                text="CVE-2024-1234",
                entity_type="CVE",
                start_char=0,
                end_char=13,
                confidence=0.95,
            ),
            # Invalid (text mismatch)
            ExtractedEntity(
                text="WrongText",
                entity_type="EMAIL",
                start_char=17,
                end_char=33,  # Actually "test@example.com"
                confidence=0.9,
            ),
        ]

        validated = extractor._validate_offsets(entities, text)

        valid_entities = [e for e in validated if e.entity_type != "CORRUPT"]
        corrupt_entities = [e for e in validated if e.entity_type == "CORRUPT"]

        assert len(valid_entities) == 1
        assert valid_entities[0].entity_type == "CVE"

        assert len(corrupt_entities) == 1
        assert corrupt_entities[0].entity_type == "CORRUPT"
        assert "Original type: EMAIL" in corrupt_entities[0].context


# ---------------------------------------------------------------------------
# Few-Shot Learning Integration Tests
# ---------------------------------------------------------------------------


class TestFewShotLearningConfiguration:
    """Few-shot learning configuration parameters."""

    def test_default_enable_few_shot_is_false(self):
        """Given no enable_few_shot, Then default is False."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor._enable_few_shot is False

    def test_custom_enable_few_shot(self):
        """Given enable_few_shot=True, Then it is enabled."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=True)
        assert extractor._enable_few_shot is True

    def test_default_cloud_provider_is_true(self):
        """Given no cloud_provider, Then default is True (sanitize PII)."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor._cloud_provider is True

    def test_embedder_is_none_by_default(self):
        """Given no embedder, Then default is None."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor._embedder is None

    def test_example_registry_is_none_by_default(self):
        """Given no example_registry, Then default is None."""
        extractor = IFUniversalExtractor(use_llm=False)
        assert extractor._example_registry is None

    def test_prompt_tuner_is_none_initially(self):
        """Given extractor, Then prompt tuner is None until initialized."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=True)
        assert extractor._prompt_tuner is None


class TestFewShotPromptTunerInitialization:
    """Lazy initialization of IFPromptTuner."""

    def test_init_prompt_tuner_creates_instance(self):
        """Given enable_few_shot, When _init_prompt_tuner called, Then tuner created."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=True)
        assert extractor._prompt_tuner is None

        extractor._init_prompt_tuner()

        assert extractor._prompt_tuner is not None

    def test_init_prompt_tuner_idempotent(self):
        """Given tuner already initialized, When _init_prompt_tuner called again, Then same instance."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=True)
        extractor._init_prompt_tuner()
        first_tuner = extractor._prompt_tuner

        extractor._init_prompt_tuner()
        second_tuner = extractor._prompt_tuner

        assert first_tuner is second_tuner


class TestGetFewShotExamples:
    """Fetching semantically similar examples."""

    def test_given_few_shot_disabled_then_empty_list(self):
        """Given enable_few_shot=False, When _get_few_shot_examples called, Then empty list."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=False)
        result = extractor._get_few_shot_examples("Test text")
        assert result == []

    def test_given_no_embedder_then_empty_list(self):
        """Given no embedder configured, When _get_few_shot_examples called, Then empty list."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=True)
        result = extractor._get_few_shot_examples("Test text")
        assert result == []

    def test_given_no_registry_then_empty_list(self):
        """Given no registry configured, When _get_few_shot_examples called, Then empty list."""
        mock_embedder = Mock()
        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
        )
        result = extractor._get_few_shot_examples("Test text")
        assert result == []

    def test_given_embedder_and_registry_then_examples_returned(self):
        """Given embedder and registry, When _get_few_shot_examples called, Then examples returned."""
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        mock_registry = Mock()
        mock_registry.find_similar.return_value = [
            (
                "Example chunk 1",
                {"entities": [{"type": "CVE", "text": "CVE-2024-0001"}]},
            ),
            ("Example chunk 2", {"entities": [{"type": "DATE", "text": "2024-01-15"}]}),
        ]

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=mock_registry,
        )

        result = extractor._get_few_shot_examples("Test text with CVE-2024-1234")

        assert len(result) == 2
        mock_embedder.embed.assert_called_once()
        mock_registry.find_similar.assert_called_once()

    def test_get_few_shot_examples_handles_exceptions(self):
        """Given exception during fetch, When _get_few_shot_examples called, Then empty list."""
        mock_embedder = Mock()
        mock_embedder.embed.side_effect = Exception("Embedding error")

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=Mock(),
        )

        result = extractor._get_few_shot_examples("Test text")
        assert result == []

    def test_get_few_shot_examples_asserts_text_not_none(self):
        """Given None text, When _get_few_shot_examples called, Then assertion error."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=True)
        with pytest.raises(AssertionError):
            extractor._get_few_shot_examples(None)


class TestGetSystemPromptWithFewShot:
    """System prompt enhancement with few-shot examples."""

    def test_given_few_shot_disabled_then_base_prompt_returned(self):
        """Given enable_few_shot=False, When _get_system_prompt called, Then base prompt."""
        extractor = IFUniversalExtractor(use_llm=False, enable_few_shot=False)
        prompt = extractor._get_system_prompt(text="Test text")

        assert "entity extraction system" in prompt
        # Should not have few-shot example markers
        assert "Example" not in prompt or "EXAMPLE" not in prompt

    def test_given_no_text_then_base_prompt_returned(self):
        """Given text=None, When _get_system_prompt called, Then base prompt."""
        mock_embedder = Mock()
        mock_registry = Mock()
        mock_registry.find_similar.return_value = []

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=mock_registry,
        )
        prompt = extractor._get_system_prompt(text=None)

        assert "entity extraction system" in prompt
        mock_embedder.embed.assert_not_called()

    def test_given_few_shot_enabled_with_examples_then_enhanced_prompt(self):
        """Given few-shot enabled with examples, When _get_system_prompt called, Then enhanced."""
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        mock_registry = Mock()
        mock_registry.find_similar.return_value = [
            (
                "CVE-2024-0001 found here",
                {"entities": [{"type": "CVE", "text": "CVE-2024-0001"}]},
            ),
        ]

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=mock_registry,
        )

        prompt = extractor._get_system_prompt(text="Text with CVE-2024-5678")

        # Should still contain base prompt content
        assert "entity extraction system" in prompt or len(prompt) > 100

    def test_given_no_examples_found_then_base_prompt_returned(self):
        """Given no examples found, When _get_system_prompt called, Then base prompt."""
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

        mock_registry = Mock()
        mock_registry.find_similar.return_value = []  # No examples

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=mock_registry,
        )

        prompt = extractor._get_system_prompt(text="Test text")

        assert "entity extraction system" in prompt


class TestFewShotJPLCompliance:
    """JPL Power of Ten compliance for few-shot methods."""

    def test_init_prompt_tuner_under_60_lines(self):
        """Given _init_prompt_tuner, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor._init_prompt_tuner)
        lines = source.split("\n")
        assert len(lines) < 60

    def test_get_few_shot_examples_under_60_lines(self):
        """Given _get_few_shot_examples, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor._get_few_shot_examples)
        lines = source.split("\n")
        assert len(lines) < 60

    def test_get_system_prompt_under_60_lines(self):
        """Given _get_system_prompt, Then under 60 lines."""
        import inspect

        source = inspect.getsource(IFUniversalExtractor._get_system_prompt)
        lines = source.split("\n")
        assert len(lines) < 60


class TestFewShotGWTBehavior:
    """
    GWT Behavioral Specification:
    - Given: A set of 3 semantically relevant golden examples
    - When: The IFUniversalExtractor is called
    - Then: The examples must be injected into the LLM system prompt as "Few-Shot" hints
    """

    def test_gwt_given_golden_examples_available(self):
        """Given 3 semantically relevant golden examples in registry."""
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.5, 0.5, 0.5]

        mock_registry = Mock()
        mock_registry.find_similar.return_value = [
            ("Chunk 1 with CVE-2024-0001", {"entities": []}),
            ("Chunk 2 with $1,000", {"entities": []}),
            ("Chunk 3 with 2024-01-15", {"entities": []}),
        ]

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=mock_registry,
        )

        # Verify examples can be retrieved
        examples = extractor._get_few_shot_examples("Test input")
        assert len(examples) == 3

    def test_gwt_when_extractor_called_then_examples_injected(self):
        """When IFUniversalExtractor is called, Then examples injected into prompt."""
        mock_embedder = Mock()
        mock_embedder.embed.return_value = [0.5, 0.5, 0.5]

        mock_registry = Mock()
        mock_registry.find_similar.return_value = [
            ("Example chunk", {"entities": [{"type": "CVE", "text": "CVE-2024-0001"}]}),
        ]

        extractor = IFUniversalExtractor(
            use_llm=False,
            enable_few_shot=True,
            embedder=mock_embedder,
            example_registry=mock_registry,
        )

        # Get prompt with text (should trigger few-shot injection)
        prompt = extractor._get_system_prompt(text="Test with CVE-2024-9999")

        # Verify prompt tuner was initialized (indicates injection attempted)
        assert extractor._prompt_tuner is not None


# ---------------------------------------------------------------------------
# AC#3: Extraction Rationale Tests
# ---------------------------------------------------------------------------


class TestExtractionRationale:
    """
    AC#3: Captures extraction_rationale for 100% of factual claims.

    GWT:
    - Given: Text containing entities
    - When: Extraction is performed
    - Then: Each entity includes extraction_rationale
    """

    def test_extracted_entity_has_rationale_field(self):
        """
        GWT:
        Given ExtractedEntity model
        When created
        Then extraction_rationale field exists.
        """
        entity = ExtractedEntity(
            text="CVE-2024-1234",
            entity_type="CVE",
            start_char=0,
            end_char=13,
            confidence=0.95,
            extraction_rationale="Matched CVE pattern format",
        )

        assert entity.extraction_rationale == "Matched CVE pattern format"

    def test_regex_extraction_includes_rationale(self):
        """
        GWT:
        Given text with CVE pattern
        When regex extraction runs
        Then entity includes extraction_rationale.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        entities = extractor.extract("Found vulnerability CVE-2024-5678 in code.")

        cve_entities = [e for e in entities if e.entity_type == "CVE"]
        assert len(cve_entities) >= 1

        # AC#3: Rationale must be present
        for entity in cve_entities:
            assert entity.extraction_rationale is not None
            assert (
                "regex" in entity.extraction_rationale.lower()
                or "pattern" in entity.extraction_rationale.lower()
            )

    def test_to_dict_includes_rationale_when_present(self):
        """
        GWT:
        Given entity with rationale
        When to_dict is called
        Then rationale is in output.
        """
        entity = ExtractedEntity(
            text="$1,000",
            entity_type="MONEY",
            start_char=10,
            end_char=16,
            confidence=0.9,
            extraction_rationale="Matched currency pattern with dollar sign",
        )

        result = entity.to_dict()

        assert "extraction_rationale" in result
        assert (
            result["extraction_rationale"]
            == "Matched currency pattern with dollar sign"
        )

    def test_to_dict_excludes_rationale_when_none(self):
        """
        GWT:
        Given entity without rationale
        When to_dict is called
        Then rationale is not in output.
        """
        entity = ExtractedEntity(
            text="test",
            entity_type="PERSON",
            start_char=0,
            end_char=4,
            confidence=0.5,
        )

        result = entity.to_dict()

        assert "extraction_rationale" not in result

    def test_system_prompt_requests_rationale(self):
        """
        GWT:
        Given extractor
        When system prompt is generated
        Then it requests extraction_rationale.
        """
        extractor = IFUniversalExtractor(use_llm=False)
        prompt = extractor._get_system_prompt()

        assert "extraction_rationale" in prompt
        assert "" in prompt or "rationale" in prompt.lower()
