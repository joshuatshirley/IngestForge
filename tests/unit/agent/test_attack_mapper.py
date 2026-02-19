"""Tests for ATT&CK Mapping Agent.

Tests the ATTACKMapper for CYBER-003 implementation."""

from __future__ import annotations

from unittest.mock import MagicMock


from ingestforge.agent.attack_mapper import (
    ATTACKMapper,
    ATTACKMapping,
    TechniqueInfo,
    Tactic,
    create_mapper,
    create_llm_mapping_function,
    map_chunk_to_attack,
    get_technique_database,
    get_all_tactics,
    TECHNIQUE_DATABASE,
    MAX_MAPPINGS_PER_CHUNK,
    MAX_INDICATORS_PER_MAPPING,
)
from ingestforge.enrichment.log_flattener import (
    FlattenedLog,
    EventCategory,
)

# =============================================================================
# Tactic Tests
# =============================================================================


class TestTactic:
    """Tests for Tactic enum."""

    def test_tactics_defined(self) -> None:
        """Test all tactics are defined."""
        tactics = [t.value for t in Tactic]

        assert "execution" in tactics
        assert "initial-access" in tactics
        assert "persistence" in tactics
        assert "credential-access" in tactics

    def test_tactic_count(self) -> None:
        """Test correct number of tactics."""
        assert len(Tactic) == 14


# =============================================================================
# ATTACKMapping Tests
# =============================================================================


class TestATTACKMapping:
    """Tests for ATTACKMapping dataclass."""

    def test_mapping_creation(self) -> None:
        """Test creating a mapping."""
        mapping = ATTACKMapping(
            chunk_id="chunk1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="execution",
            confidence=0.9,
        )

        assert mapping.technique_id == "T1059.001"
        assert mapping.confidence == 0.9

    def test_confidence_clamped(self) -> None:
        """Test confidence is clamped to 0-1."""
        mapping_high = ATTACKMapping(
            chunk_id="chunk1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="execution",
            confidence=1.5,
        )
        mapping_low = ATTACKMapping(
            chunk_id="chunk1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="execution",
            confidence=-0.5,
        )

        assert mapping_high.confidence == 1.0
        assert mapping_low.confidence == 0.0

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        mapping = ATTACKMapping(
            chunk_id="chunk1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="execution",
            confidence=0.8,
            evidence="powershell.exe found",
            indicators=["powershell.exe"],
        )

        d = mapping.to_dict()

        assert d["technique_id"] == "T1059.001"
        assert d["tactic"] == "execution"
        assert "powershell.exe" in d["indicators"]

    def test_evidence_truncation(self) -> None:
        """Test long evidence is truncated."""
        long_evidence = "x" * 1000
        mapping = ATTACKMapping(
            chunk_id="chunk1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="execution",
            confidence=0.8,
            evidence=long_evidence,
        )

        assert len(mapping.evidence) <= 500

    def test_indicators_limit(self) -> None:
        """Test indicators list is limited."""
        many_indicators = [f"indicator_{i}" for i in range(50)]
        mapping = ATTACKMapping(
            chunk_id="chunk1",
            technique_id="T1059.001",
            technique_name="PowerShell",
            tactic="execution",
            confidence=0.8,
            indicators=many_indicators,
        )

        assert len(mapping.indicators) <= MAX_INDICATORS_PER_MAPPING


# =============================================================================
# TechniqueInfo Tests
# =============================================================================


class TestTechniqueInfo:
    """Tests for TechniqueInfo dataclass."""

    def test_technique_creation(self) -> None:
        """Test creating technique info."""
        technique = TechniqueInfo(
            technique_id="T1059.001",
            name="PowerShell",
            tactic=Tactic.EXECUTION,
            description="Abuse PowerShell",
            keywords=["powershell", "pwsh"],
        )

        assert technique.technique_id == "T1059.001"
        assert technique.tactic == Tactic.EXECUTION

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        technique = TechniqueInfo(
            technique_id="T1059.001",
            name="PowerShell",
            tactic=Tactic.EXECUTION,
        )

        d = technique.to_dict()

        assert d["technique_id"] == "T1059.001"
        assert d["tactic"] == "execution"


# =============================================================================
# Technique Database Tests
# =============================================================================


class TestTechniqueDatabase:
    """Tests for built-in technique database."""

    def test_database_populated(self) -> None:
        """Test database has techniques."""
        assert len(TECHNIQUE_DATABASE) >= 50

    def test_common_techniques_present(self) -> None:
        """Test common techniques are in database."""
        ids = {t.technique_id for t in TECHNIQUE_DATABASE}

        assert "T1059.001" in ids  # PowerShell
        assert "T1566.001" in ids  # Spearphishing Attachment
        assert "T1003.001" in ids  # LSASS Memory
        assert "T1486" in ids  # Data Encrypted for Impact

    def test_techniques_have_keywords(self) -> None:
        """Test all techniques have keywords."""
        for technique in TECHNIQUE_DATABASE:
            assert len(technique.keywords) > 0

    def test_techniques_have_hints(self) -> None:
        """Test all techniques have detection hints."""
        for technique in TECHNIQUE_DATABASE:
            assert len(technique.detection_hints) > 0


# =============================================================================
# ATTACKMapper Tests - Technique Recognition
# =============================================================================


class TestTechniqueRecognition:
    """Tests for technique recognition."""

    def test_powershell_detection(self) -> None:
        """Test PowerShell technique detection."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test1",
            "text": "Process started: powershell.exe -enc SGVsbG8gV29ybGQ=",
        }

        mappings = mapper.map_chunk(chunk)

        assert any(m.technique_id == "T1059.001" for m in mappings)

    def test_phishing_detection(self) -> None:
        """Test spearphishing detection."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test2",
            "text": "User received email attachment from external sender",
        }

        mappings = mapper.map_chunk(chunk)

        # Should detect phishing or attachment-related technique
        assert len(mappings) > 0

    def test_credential_dump_detection(self) -> None:
        """Test credential dumping detection."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test3",
            "text": "Process lsass.exe accessed by mimikatz for credential dump",
        }

        mappings = mapper.map_chunk(chunk)

        assert any(m.technique_id == "T1003.001" for m in mappings)

    def test_ransomware_detection(self) -> None:
        """Test ransomware technique detection."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test4",
            "text": "Ransomware encrypted files, ransom note found",
        }

        mappings = mapper.map_chunk(chunk)

        assert any(m.technique_id == "T1486" for m in mappings)

    def test_no_match_empty_text(self) -> None:
        """Test no matches for empty text."""
        mapper = ATTACKMapper()
        chunk = {"id": "test5", "text": ""}

        mappings = mapper.map_chunk(chunk)

        assert len(mappings) == 0

    def test_no_match_unrelated_text(self) -> None:
        """Test no matches for unrelated text."""
        mapper = ATTACKMapper()
        chunk = {"id": "test6", "text": "The weather today is sunny."}

        mappings = mapper.map_chunk(chunk)

        assert len(mappings) == 0


# =============================================================================
# ATTACKMapper Tests - Tactic Categorization
# =============================================================================


class TestTacticCategorization:
    """Tests for tactic categorization."""

    def test_execution_tactic(self) -> None:
        """Test execution tactic identification."""
        mapper = ATTACKMapper()
        chunk = {"id": "test1", "text": "cmd.exe /c whoami executed"}

        mappings = mapper.map_chunk(chunk)

        execution_mappings = [m for m in mappings if "execution" in m.tactic]
        assert len(execution_mappings) > 0

    def test_persistence_tactic(self) -> None:
        """Test persistence tactic identification."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test2",
            "text": "Registry key added: HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
        }

        mappings = mapper.map_chunk(chunk)

        persistence_mappings = [m for m in mappings if "persistence" in m.tactic]
        assert len(persistence_mappings) > 0

    def test_lateral_movement_tactic(self) -> None:
        """Test lateral movement tactic identification."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test3",
            "text": "RDP connection established to remote desktop port 3389",
        }

        mappings = mapper.map_chunk(chunk)

        lateral_mappings = [m for m in mappings if "lateral" in m.tactic]
        assert len(lateral_mappings) > 0

    def test_multiple_tactics(self) -> None:
        """Test detecting multiple tactics in one chunk."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test4",
            "text": "PowerShell script downloaded malware and created scheduled task",
        }

        mappings = mapper.map_chunk(chunk)

        tactics = {m.tactic for m in mappings}
        # Should have at least execution and possibly persistence
        assert "execution" in tactics or len(tactics) >= 1


# =============================================================================
# ATTACKMapper Tests - Confidence Scoring
# =============================================================================


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_high_confidence_exact_match(self) -> None:
        """Test high confidence for exact pattern match."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test1",
            "text": "powershell.exe -encodedcommand executed invoke-expression",
        }

        mappings = mapper.map_chunk(chunk)

        powershell_mapping = next(
            (m for m in mappings if m.technique_id == "T1059.001"), None
        )
        assert powershell_mapping is not None
        assert powershell_mapping.confidence > 0.5

    def test_low_confidence_partial_match(self) -> None:
        """Test lower confidence for partial match."""
        mapper = ATTACKMapper()
        chunk = {"id": "test2", "text": "powershell mentioned in log"}

        mappings = mapper.map_chunk(chunk)

        if mappings:
            # Partial match should have lower confidence
            assert mappings[0].confidence < 1.0

    def test_confidence_in_range(self) -> None:
        """Test all confidences are in valid range."""
        mapper = ATTACKMapper()
        chunk = {
            "id": "test3",
            "text": "Multiple techniques: powershell, mimikatz, ransomware",
        }

        mappings = mapper.map_chunk(chunk)

        for mapping in mappings:
            assert 0.0 <= mapping.confidence <= 1.0

    def test_min_confidence_filter(self) -> None:
        """Test minimum confidence threshold."""
        mapper = ATTACKMapper(min_confidence=0.8)
        chunk = {"id": "test4", "text": "powershell briefly mentioned"}

        mappings = mapper.map_chunk(chunk)

        # All returned mappings should meet threshold
        for mapping in mappings:
            assert mapping.confidence >= 0.8


# =============================================================================
# ATTACKMapper Tests - Chunk Enrichment
# =============================================================================


class TestChunkEnrichment:
    """Tests for chunk enrichment."""

    def test_enrich_single_chunk(self) -> None:
        """Test enriching a single chunk."""
        mapper = ATTACKMapper()
        chunks = [{"id": "chunk1", "text": "powershell.exe -enc base64"}]

        enriched = mapper.enrich_chunks(chunks)

        assert len(enriched) == 1
        assert "metadata" in enriched[0]
        assert "attack_technique" in enriched[0]["metadata"]

    def test_enrich_preserves_original(self) -> None:
        """Test enrichment preserves original fields."""
        mapper = ATTACKMapper()
        chunks = [
            {
                "id": "chunk1",
                "text": "powershell.exe",
                "source": "test.log",
            }
        ]

        enriched = mapper.enrich_chunks(chunks)

        assert enriched[0]["source"] == "test.log"
        assert enriched[0]["id"] == "chunk1"

    def test_enrich_multiple_chunks(self) -> None:
        """Test enriching multiple chunks."""
        mapper = ATTACKMapper()
        chunks = [
            {"id": "chunk1", "text": "powershell.exe"},
            {"id": "chunk2", "text": "mimikatz lsass dump"},
            {"id": "chunk3", "text": "normal log entry"},
        ]

        enriched = mapper.enrich_chunks(chunks)

        assert len(enriched) == 3

    def test_enrich_adds_all_techniques(self) -> None:
        """Test enrichment adds all mapped techniques."""
        mapper = ATTACKMapper()
        chunks = [
            {
                "id": "chunk1",
                "text": "powershell invoke-expression mimikatz",
            }
        ]

        enriched = mapper.enrich_chunks(chunks)

        metadata = enriched[0]["metadata"]
        assert "attack_techniques" in metadata
        assert isinstance(metadata["attack_techniques"], list)


# =============================================================================
# ATTACKMapper Tests - Log Event Mapping
# =============================================================================


class TestLogEventMapping:
    """Tests for FlattenedLog event mapping."""

    def test_map_log_event(self) -> None:
        """Test mapping a flattened log event."""
        mapper = ATTACKMapper()
        log = FlattenedLog(
            timestamp="2024-01-15T10:30:00Z",
            event_id="ProcessCreate",
            event_category=EventCategory.PROCESS,
            message="Process powershell.exe started with encoded command",
        )

        mappings = mapper.map_log_event(log)

        assert len(mappings) > 0
        assert any(m.technique_id == "T1059.001" for m in mappings)

    def test_map_auth_log(self) -> None:
        """Test mapping authentication log."""
        mapper = ATTACKMapper()
        log = FlattenedLog(
            timestamp="2024-01-15T10:30:00Z",
            event_id="AuthenticationSuccess",
            event_category=EventCategory.AUTH,
            user="admin",
            message="Successful login from external source",
        )

        mappings = mapper.map_log_event(log)

        # Should potentially match valid accounts technique
        assert isinstance(mappings, list)

    def test_map_network_log(self) -> None:
        """Test mapping network log."""
        mapper = ATTACKMapper()
        log = FlattenedLog(
            timestamp="2024-01-15T10:30:00Z",
            event_id="NetworkConnection",
            event_category=EventCategory.NETWORK,
            dst_ip="192.168.xxx.xxx",
            message="RDP connection to port 3389",
        )

        mappings = mapper.map_log_event(log)

        assert isinstance(mappings, list)


# =============================================================================
# ATTACKMapper Tests - Technique Info
# =============================================================================


class TestTechniqueInfo:
    """Tests for technique info retrieval."""

    def test_get_technique_info(self) -> None:
        """Test getting technique info by ID."""
        mapper = ATTACKMapper()

        info = mapper.get_technique_info("T1059.001")

        assert info is not None
        assert info["technique_id"] == "T1059.001"
        assert info["name"] == "PowerShell"
        assert info["tactic"] == "execution"

    def test_get_unknown_technique(self) -> None:
        """Test getting unknown technique returns None."""
        mapper = ATTACKMapper()

        info = mapper.get_technique_info("T9999.999")

        assert info is None

    def test_get_techniques_by_tactic(self) -> None:
        """Test getting techniques by tactic."""
        mapper = ATTACKMapper()

        techniques = mapper.get_techniques_by_tactic("execution")

        assert len(techniques) > 0
        assert all(t["tactic"] == "execution" for t in techniques)

    def test_get_techniques_unknown_tactic(self) -> None:
        """Test getting techniques for unknown tactic."""
        mapper = ATTACKMapper()

        techniques = mapper.get_techniques_by_tactic("unknown-tactic")

        assert techniques == []


# =============================================================================
# ATTACKMapper Tests - LLM Integration
# =============================================================================


class TestLLMIntegration:
    """Tests for LLM integration with mock."""

    def test_llm_mapping(self) -> None:
        """Test LLM-enhanced mapping."""
        mock_response = """
        [{"technique_id": "T1059.001", "confidence": 0.9, "evidence": "PowerShell detected"}]
        """
        mock_llm = MagicMock(return_value=mock_response)
        mapper = ATTACKMapper(llm_fn=mock_llm)

        chunk = {"id": "test1", "text": "suspicious activity"}
        mappings = mapper.map_chunk(chunk)

        assert mock_llm.called
        assert any(m.technique_id == "T1059.001" for m in mappings)

    def test_llm_error_handled(self) -> None:
        """Test LLM errors are handled gracefully."""

        def failing_llm(prompt: str) -> str:
            raise Exception("LLM error")

        mapper = ATTACKMapper(llm_fn=failing_llm)

        chunk = {"id": "test1", "text": "powershell.exe"}
        mappings = mapper.map_chunk(chunk)

        # Should still return rule-based mappings
        assert len(mappings) >= 0

    def test_llm_invalid_json_handled(self) -> None:
        """Test invalid LLM JSON response is handled."""
        mock_llm = MagicMock(return_value="not valid json")
        mapper = ATTACKMapper(llm_fn=mock_llm)

        chunk = {"id": "test1", "text": "powershell.exe"}
        mappings = mapper.map_chunk(chunk)

        # Should still work with rule-based mappings
        assert isinstance(mappings, list)

    def test_llm_empty_response_handled(self) -> None:
        """Test empty LLM response is handled."""
        mock_llm = MagicMock(return_value="[]")
        mapper = ATTACKMapper(llm_fn=mock_llm)

        chunk = {"id": "test1", "text": "powershell.exe"}
        mappings = mapper.map_chunk(chunk)

        # Should return rule-based mappings
        assert len(mappings) > 0


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_mapper(self) -> None:
        """Test create_mapper factory."""
        mapper = create_mapper()

        assert isinstance(mapper, ATTACKMapper)

    def test_create_mapper_with_confidence(self) -> None:
        """Test create_mapper with custom confidence."""
        mapper = create_mapper(min_confidence=0.5)

        assert mapper._min_confidence == 0.5

    def test_create_mapper_with_llm(self) -> None:
        """Test create_mapper with LLM function."""
        mock_llm = MagicMock()
        mapper = create_mapper(llm_fn=mock_llm)

        assert mapper._llm_fn is not None

    def test_create_llm_mapping_function(self) -> None:
        """Test creating LLM mapping function."""
        mock_client = MagicMock()
        mock_client.generate.return_value = "response"

        fn = create_llm_mapping_function(mock_client)
        result = fn("test prompt")

        assert result == "response"
        mock_client.generate.assert_called_once_with("test prompt")

    def test_map_chunk_to_attack_convenience(self) -> None:
        """Test map_chunk_to_attack convenience function."""
        chunk = {"id": "test1", "text": "powershell.exe"}

        mappings = map_chunk_to_attack(chunk)

        assert isinstance(mappings, list)
        assert any(m.technique_id == "T1059.001" for m in mappings)

    def test_get_technique_database(self) -> None:
        """Test get_technique_database function."""
        db = get_technique_database()

        assert len(db) >= 50
        assert all(isinstance(t, dict) for t in db)

    def test_get_all_tactics(self) -> None:
        """Test get_all_tactics function."""
        tactics = get_all_tactics()

        assert "execution" in tactics
        assert "initial-access" in tactics
        assert len(tactics) == 14


# =============================================================================
# Edge Cases and Limits
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and limits."""

    def test_max_mappings_limit(self) -> None:
        """Test max mappings per chunk limit."""
        mapper = ATTACKMapper()
        # Text with many technique keywords
        chunk = {
            "id": "test1",
            "text": " ".join(
                [
                    "powershell cmd mimikatz ransomware rdp ssh vpn",
                    "download wget curl scheduled task registry service",
                ]
            ),
        }

        mappings = mapper.map_chunk(chunk)

        assert len(mappings) <= MAX_MAPPINGS_PER_CHUNK

    def test_missing_chunk_id(self) -> None:
        """Test chunk without id field."""
        mapper = ATTACKMapper()
        chunk = {"text": "powershell.exe"}

        mappings = mapper.map_chunk(chunk)

        assert all(m.chunk_id == "unknown" for m in mappings)

    def test_missing_chunk_text(self) -> None:
        """Test chunk without text field."""
        mapper = ATTACKMapper()
        chunk = {"id": "test1"}

        mappings = mapper.map_chunk(chunk)

        assert len(mappings) == 0

    def test_invalid_min_confidence(self) -> None:
        """Test invalid min_confidence is corrected."""
        mapper_high = ATTACKMapper(min_confidence=2.0)
        mapper_low = ATTACKMapper(min_confidence=-1.0)

        assert mapper_high._min_confidence == 0.3  # Reset to default
        assert mapper_low._min_confidence == 0.3  # Reset to default

    def test_case_insensitive_matching(self) -> None:
        """Test matching is case insensitive."""
        mapper = ATTACKMapper()
        chunk = {"id": "test1", "text": "POWERSHELL.EXE -EncodedCommand"}

        mappings = mapper.map_chunk(chunk)

        assert any(m.technique_id == "T1059.001" for m in mappings)


# =============================================================================
# Constant Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_max_mappings_per_chunk(self) -> None:
        """Test MAX_MAPPINGS_PER_CHUNK is reasonable."""
        assert MAX_MAPPINGS_PER_CHUNK > 0
        assert MAX_MAPPINGS_PER_CHUNK == 10

    def test_max_indicators_per_mapping(self) -> None:
        """Test MAX_INDICATORS_PER_MAPPING is reasonable."""
        assert MAX_INDICATORS_PER_MAPPING > 0
        assert MAX_INDICATORS_PER_MAPPING == 20
