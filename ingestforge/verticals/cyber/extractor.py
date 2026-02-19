"""
Cyber Intelligence Extractor.

Cyber CVE Blueprint
Specialized extractor for vulnerability metadata.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: 100% type hints.
"""

import re
from typing import List, Optional
from ingestforge.verticals.cyber.models import CyberVulnerabilityModel
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.base import GenerationConfig
from ingestforge.core.config_loaders import load_config
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class CyberExtractor:
    """Extracts structured vulnerability data from security reports."""

    # High-precision patterns (extending enrichment/cyber.py)
    CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)

    def __init__(self):
        """Initialize with LLM client."""
        self.config = load_config()
        self.llm = get_llm_client(self.config)

    def extract_from_text(self, text: str) -> List[CyberVulnerabilityModel]:
        """
        Scan text for CVEs and extract structured models using LLM.

        Rule #2: Bounded by found CVEs (limited to 5 per chunk for safety).
        """
        cves = list(set(self.CVE_PATTERN.findall(text)))
        results: List[CyberVulnerabilityModel] = []

        for cve_id in cves[:5]:  # JPL Rule #2: Strict iteration bound
            try:
                model = self._llm_extract_cve_details(cve_id, text)
                if model:
                    results.append(model)
            except Exception as e:
                logger.error(f"Failed to extract details for {cve_id}: {e}")

        return results

    def _llm_extract_cve_details(
        self, cve_id: str, context: str
    ) -> Optional[CyberVulnerabilityModel]:
        """Uses LLM to fill in details for a specific CVE identifier."""
        prompt = f"""You are a cybersecurity analyst. Extract structured details for the vulnerability {cve_id} from the context below.
If details are missing, use your knowledge base or state 'Information not available in context'.

CONTEXT:
{context}

JSON FORMAT:
{{
  "cve_id": "{cve_id}",
  "cvss_score": float,
  "severity": "CRITICAL|HIGH|MEDIUM|LOW",
  "summary": "brief summary",
  "affected_systems": [{{ "product": "name", "version_range": "versions" }}],
  "remediation": "steps to fix"
}}

ANSWER (JSON ONLY):"""

        gen_config = GenerationConfig(max_tokens=512, temperature=0.1)
        response = self.llm.generate(prompt, gen_config)

        if not response or not response.text:
            return None

        return self._parse_llm_response(response.text)

    def _parse_llm_response(self, raw_text: str) -> Optional[CyberVulnerabilityModel]:
        """Parses JSON response from LLM into a Pydantic model.

        Rule #4: Decomposed parser.
        """
        import json

        try:
            # Clean possible markdown formatting
            clean_json = raw_text.strip().strip("`").replace("json\n", "")
            data = json.loads(clean_json)
            return CyberVulnerabilityModel(**data)
        except Exception as e:
            logger.error(f"JSON parsing failed for cyber extraction: {e}")
            return None
