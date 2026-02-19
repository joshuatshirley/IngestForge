import pytest
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.analysis.lateral_linker import LateralLinker


class TestLateralLinker:
    @pytest.fixture
    def linker(self):
        return LateralLinker()

    def test_find_shared_entity_connection(self, linker):
        # Chunk 1: Cyber domain mentioning Microsoft
        c1 = ChunkRecord(
            chunk_id="c1", document_id="d1", content="Vuln in Microsoft Windows."
        )
        c1.entities = ["Microsoft"]
        c1.metadata = {"detected_domains": ["cyber"], "primary_domain": "cyber"}

        # Chunk 2: Legal domain mentioning Microsoft
        c2 = ChunkRecord(
            chunk_id="c2",
            document_id="d2",
            content="Microsoft Corp vs Department of Justice.",
        )
        c2.entities = ["Microsoft"]
        c2.metadata = {"detected_domains": ["legal"], "primary_domain": "legal"}

        connections = linker.find_connections([c1, c2])

        assert len(connections) == 1
        assert connections[0]["type"] == "lateral_entity_link"
        assert connections[0]["entity"] == "microsoft"
        assert set(connections[0]["domains"]) == {"cyber", "legal"}

    def test_find_identifier_collision(self, linker):
        # Chunk 1: Cyber report
        c1 = ChunkRecord(
            chunk_id="c1", document_id="d1", content="Exploit CVE-2024-1234."
        )
        c1.metadata = {
            "detected_domains": ["cyber"],
            "primary_domain": "cyber",
            "cyber_cve_id": "CVE-2024-1234",
        }

        # Chunk 2: Urban Planning / Smart City doc
        c2 = ChunkRecord(
            chunk_id="c2",
            document_id="d2",
            content="Smart traffic lights risk CVE-2024-1234.",
        )
        c2.metadata = {
            "detected_domains": ["urban"],
            "primary_domain": "urban",
            "cyber_cve_id": "CVE-2024-1234",  # Extracted by Cyber refiner acting on Urban doc
        }

        connections = linker.find_connections([c1, c2])

        assert any(conn["type"] == "cross_domain_id_collision" for conn in connections)
        collision = [
            conn for conn in connections if conn["type"] == "cross_domain_id_collision"
        ][0]
        assert collision["id"] == "CVE-2024-1234"
        assert set(collision["domains"]) == {"cyber", "urban"}
