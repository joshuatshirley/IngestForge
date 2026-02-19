"""
Cyber Vertical Models.

Cyber CVE Blueprint
Defines the structure for vulnerability intelligence outputs.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class VulnerabilityReference(BaseModel):
    """Reference to external vulnerability data."""

    url: str
    description: Optional[str] = None


class AffectedSystem(BaseModel):
    """System or software affected by a vulnerability."""

    product: str
    version_range: str
    platform: Optional[str] = None


class CyberVulnerabilityModel(BaseModel):
    """Structured intelligence for a single vulnerability (CVE)."""

    cve_id: str = Field(..., pattern=r"^CVE-\d{4}-\d{4,7}$")
    cvss_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    severity: str = Field(
        default="UNKNOWN", description="Critical, High, Medium, Low, or Info"
    )
    summary: str
    affected_systems: List[AffectedSystem] = Field(default_factory=list)
    remediation: Optional[str] = None
    references: List[VulnerabilityReference] = Field(default_factory=list)

    def get_severity_color(self) -> str:
        """Returns a UI-friendly color code based on CVSS score."""
        if self.cvss_score is None:
            return "gray"
        if self.cvss_score >= 9.0:
            return "red"
        if self.cvss_score >= 7.0:
            return "orange"
        if self.cvss_score >= 4.0:
            return "yellow"
        return "green"


class CyberMissionReport(BaseModel):
    """Aggregate security report for a research mission."""

    mission_id: str
    vulnerabilities: List[CyberVulnerabilityModel]
    total_count: int
    critical_count: int
    risk_assessment: str
