import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.cyber import CyberMetadataRefiner


class TestCyberMetadataRefiner:
    @pytest.fixture
    def refiner(self):
        return CyberMetadataRefiner()

    def test_enrich_security_advisory(self, refiner):
        text = """
        Security Alert: CVE-2024-12345 discovered in web server.
        CVSS Score: 9.8 (Critical)
        Affected Software: Apache HTTP Server 2.4.58, Nginx 1.25.3
        Remediation: Update to latest version immediately.
        """
        chunk = ChunkRecord(chunk_id="c1", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert enriched.metadata["cyber_cve_id"] == "CVE-2024-12345"
        assert enriched.metadata["cyber_cvss_score"] == 9.8
        assert "Apache HTTP Server 2.4.58" in enriched.metadata["cyber_affected_sw"]
        assert "Nginx 1.25.3" in enriched.metadata["cyber_affected_sw"]

    def test_multiple_cves(self, refiner):
        text = "This patch addresses CVE-2023-5555 and CVE-2023-6666."
        chunk = ChunkRecord(chunk_id="c2", document_id="d1", content=text)
        enriched = refiner.enrich(chunk)

        assert "CVE-2023-5555" in enriched.metadata["cyber_all_cves"]
        assert "CVE-2023-6666" in enriched.metadata["cyber_all_cves"]
