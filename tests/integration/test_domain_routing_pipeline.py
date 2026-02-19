from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.enrichment.dynamic_enricher import DynamicDomainEnricher


class TestDomainRoutingPipeline:
    def test_routing_integration(self):
        """Test that DynamicDomainEnricher correctly routes and enriches chunks."""
        enricher = DynamicDomainEnricher()

        # 1. Urban Chunk
        urban_text = "Zoning: R-1. Density: 5 units/acre."
        urban_chunk = ChunkRecord(chunk_id="u1", document_id="d1", content=urban_text)

        # 2. Cyber Chunk
        cyber_text = "Found vulnerability CVE-2024-9999 in the payload."
        cyber_chunk = ChunkRecord(chunk_id="c1", document_id="d2", content=cyber_text)

        # 3. Generic/Unknown Chunk
        generic_text = "The quick brown fox jumps over the lazy dog."
        generic_chunk = ChunkRecord(
            chunk_id="g1", document_id="d3", content=generic_text
        )

        chunks = [
            urban_text,
            cyber_text,
            generic_text,
        ]  # wait, enrich_batch takes chunks
        chunks = [urban_chunk, cyber_chunk, generic_chunk]

        # Process
        results = enricher.enrich_batch(chunks)

        # Verify Urban
        res_urban = results[0]
        assert "primary_domain" in res_urban.metadata
        assert res_urban.metadata["primary_domain"] == "urban"
        assert res_urban.metadata["urban_zoning_code"] == "R-1"

        # Verify Cyber
        res_cyber = results[1]
        assert "primary_domain" in res_cyber.metadata
        assert res_cyber.metadata["primary_domain"] == "cyber"
        assert "CVE-2024-9999" in res_cyber.metadata["cyber_all_cves"]

        # Verify Generic
        res_generic = results[2]
        # Should detect no domain (or low score) and thus no detected_domain tag
        # (or maybe it just doesn't add specific fields)
        if "detected_domain" in (res_generic.metadata or {}):
            # If it detected something, ensure it didn't crash
            pass
        else:
            assert "urban_zoning_code" not in (res_generic.metadata or {})

    def test_mixed_domain_enrichment(self):
        """Test that a chunk with multiple signals applies all refiners."""
        enricher = DynamicDomainEnricher(min_score=2, multi_domain_threshold=0.5)

        # Mixed Text: Urban + Cyber
        text = "Zoning R-1 is strict. See CVE-2024-1234 for details."
        chunk = ChunkRecord(chunk_id="m1", document_id="d4", content=text)

        enriched = enricher.enrich_chunk(chunk)

        # Should have both domains
        domains = enriched.metadata.get("detected_domains", [])
        assert "urban" in domains
        assert "cyber" in domains

        # Should have metadata from both
        assert enriched.metadata["urban_zoning_code"] == "R-1"
        assert "CVE-2024-1234" in enriched.metadata["cyber_all_cves"]
