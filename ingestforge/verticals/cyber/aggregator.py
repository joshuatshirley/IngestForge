"""
Cyber Vulnerability Aggregator.

Phase 2 - Evidence Tagging & Aggregation.
Retrieves detected CVEs and vulnerabilities from the Knowledge Graph.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List, Optional
from ingestforge.verticals.cyber.models import CyberVulnerabilityModel
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.core.pipeline.pipeline import Pipeline
from ingestforge.core.config_loaders import load_config


class CyberVulnerabilityAggregator:
    """Aggregates vulnerability data from the knowledge base."""

    def __init__(self):
        self.config = load_config()
        self.pipeline = Pipeline(self.config)
        self.retriever = HybridRetriever(self.config, self.pipeline.storage)

    def aggregate_mission_vulnerabilities(
        self, mission_query: str, limit: int = 20
    ) -> List[CyberVulnerabilityModel]:
        """
        Retrieves document chunks related to vulnerabilities and maps them to models.

        JPL Rule #4: Concise retrieval.
        """
        # Search for chunks with high cyber relevance
        results = self.retriever.search(
            query=f"vulnerability CVE {mission_query}", top_k=limit
        )

        vulnerabilities: List[CyberVulnerabilityModel] = []
        for res in results:
            # Check metadata for already extracted CVEs (from enrichment/cyber.py)
            cve_id = res.metadata.get("cyber_cve_id")
            if cve_id:
                vulnerabilities.append(
                    CyberVulnerabilityModel(
                        cve_id=cve_id,
                        cvss_score=res.metadata.get("cyber_cvss_score"),
                        severity=self._score_to_severity(
                            res.metadata.get("cyber_cvss_score")
                        ),
                        summary=res.content[:200] + "...",
                    )
                )
        return vulnerabilities

    def _score_to_severity(self, score: Optional[float]) -> str:
        """Maps CVSS score to severity string."""
        if score is None:
            return "UNKNOWN"
        if score >= 9.0:
            return "CRITICAL"
        if score >= 7.0:
            return "HIGH"
        if score >= 4.0:
            return "MEDIUM"
        return "LOW"
