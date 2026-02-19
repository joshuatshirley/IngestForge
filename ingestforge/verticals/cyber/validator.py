"""
Cyber Vulnerability Validator.

Phase 4 - Compliance & Validation.
Ensures CVEs and CVSS scores meet strict formatting standards.

JPL Compliance:
- Rule #7: Return explicit validation status.
- Rule #9: 100% type hints.
"""

import re
from typing import Tuple


class CyberVulnerabilityValidator:
    """Validates cybersecurity intelligence data."""

    CVE_REGEX = re.compile(r"^CVE-\d{4}-\d{4,7}$")

    def validate_cve_id(self, cve_id: str) -> bool:
        """Checks if a string is a valid CVE identifier."""
        return bool(self.CVE_REGEX.match(cve_id))

    def validate_cvss(self, score: float) -> Tuple[bool, str]:
        """
        Validates CVSS v3 score range.

        JPL Rule #7: Return explicit validation status.
        """
        if 0.0 <= score <= 10.0:
            return True, "Valid"
        return False, f"CVSS score {score} out of bounds (0.0-10.0)"
