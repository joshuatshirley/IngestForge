"""Comprehensive GWT unit tests for the Cyber Vertical.

Cyber CVE Blueprint.
Verifies CVE detection, CVSS validation, and security report synthesis.
"""

from unittest.mock import MagicMock, patch
from ingestforge.verticals.cyber.models import CyberVulnerabilityModel
from ingestforge.verticals.cyber.extractor import CyberExtractor
from ingestforge.verticals.cyber.generator import CyberSecurityReportGenerator
from ingestforge.verticals.cyber.validator import CyberVulnerabilityValidator
from ingestforge.verticals.cyber.aggregator import CyberVulnerabilityAggregator

# =============================================================================
# MODELS TESTS
# =============================================================================


def test_severity_color_mapping():
    """GIVEN a vulnerability model with varying CVSS scores
    WHEN get_severity_color is called
    THEN it returns the correct color code for UI rendering.
    """
    critical = CyberVulnerabilityModel(
        cve_id="CVE-2024-1234", cvss_score=9.8, summary="Crit"
    )
    high = CyberVulnerabilityModel(
        cve_id="CVE-2024-5678", cvss_score=7.5, summary="High"
    )
    low = CyberVulnerabilityModel(cve_id="CVE-2024-9012", cvss_score=2.0, summary="Low")

    assert critical.get_severity_color() == "red"
    assert high.get_severity_color() == "orange"
    assert low.get_severity_color() == "green"


# =============================================================================
# EXTRACTOR TESTS
# =============================================================================


def test_cve_pattern_detection():
    """GIVEN text containing CVE identifiers
    WHEN extract_from_text is called
    THEN it correctly identifies all CVE patterns using regex.
    """
    text = "Found CVE-2021-44228 and also CVE-2023-1234567 in the logs."

    with patch("ingestforge.verticals.cyber.extractor.load_config"), patch(
        "ingestforge.llm.factory.get_llm_client"
    ), patch.object(CyberExtractor, "_llm_extract_cve_details") as mock_llm:
        extractor = CyberExtractor()
        extractor.extract_from_text(text)

        # Verify both CVEs were passed to the detail extractor
        assert mock_llm.call_count == 2
        calls = [c.args[0] for c in mock_llm.call_args_list]
        assert "CVE-2021-44228" in [c.upper() for c in calls]


# =============================================================================
# AGGREGATOR TESTS
# =============================================================================


def test_vulnerability_aggregation_from_metadata():
    """GIVEN search results with cyber metadata
    WHEN aggregate_mission_vulnerabilities is called
    THEN it extracts CVE models from the chunk metadata.
    """
    with patch("ingestforge.verticals.cyber.aggregator.load_config"), patch(
        "ingestforge.verticals.cyber.aggregator.Pipeline"
    ), patch("ingestforge.verticals.cyber.aggregator.HybridRetriever") as MockRetriever:
        mock_res = MagicMock()
        mock_res.content = "Vuln details"
        mock_res.metadata = {"cyber_cve_id": "CVE-2024-0001", "cyber_cvss_score": 8.5}
        MockRetriever.return_value.search.return_value = [mock_res]

        aggregator = CyberVulnerabilityAggregator()
        results = aggregator.aggregate_mission_vulnerabilities("test")

        assert len(results) == 1
        assert results[0].cve_id == "CVE-2024-0001"
        assert results[0].severity == "HIGH"


# =============================================================================
# GENERATOR TESTS
# =============================================================================


def test_security_bulletin_generation():
    """GIVEN a list of vulnerabilities
    WHEN generate_markdown_report is called
    THEN it produces a professional security bulletin header and details.
    """
    vulns = [
        CyberVulnerabilityModel(
            cve_id="CVE-2024-1111",
            cvss_score=9.1,
            severity="CRITICAL",
            summary="Exploit",
        )
    ]
    generator = CyberSecurityReportGenerator()
    report = generator.generate_markdown_report(vulns, "M-123")

    assert "# SECURITY INTELLIGENCE BULLETIN" in report
    assert "CVE-2024-1111" in report
    assert "CRITICAL RISK" in report


# =============================================================================
# VALIDATOR TESTS
# =============================================================================


def test_cyber_validation_logic():
    """GIVEN cyber data inputs
    WHEN validator methods are called
    THEN they enforce strict formatting and range constraints.
    """
    validator = CyberVulnerabilityValidator()

    assert validator.validate_cve_id("CVE-2024-1234") is True
    assert validator.validate_cve_id("invalid-cve") is False

    ok, _ = validator.validate_cvss(7.5)
    assert ok is True

    fail, _ = validator.validate_cvss(11.0)
    assert fail is False
