"""
Cyber Vertical API Router.

Cyber CVE Blueprint
Exposes vulnerability intelligence capabilities.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: Complete type hints.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ingestforge.verticals.cyber.models import CyberVulnerabilityModel
from ingestforge.verticals.cyber.extractor import CyberExtractor
from ingestforge.verticals.cyber.generator import CyberSecurityReportGenerator
from ingestforge.verticals.cyber.aggregator import CyberVulnerabilityAggregator

router = APIRouter(prefix="/v1/cyber", tags=["cyber"])


class ScanRequest(BaseModel):
    """API request for security scanning."""

    text: str


class ReportRequest(BaseModel):
    """API request for report generation."""

    mission_id: str
    vulnerabilities: List[CyberVulnerabilityModel] = []
    context_query: Optional[str] = None


class ReportResponse(BaseModel):
    """API response for report generation."""

    markdown: str
    success: bool = True


@router.post("/scan", response_model=List[CyberVulnerabilityModel])
async def scan_text(request: ScanRequest) -> List[CyberVulnerabilityModel]:
    """
    Scan text for vulnerabilities and extract structured data.

    Intelligence extraction endpoint.
    """
    try:
        extractor = CyberExtractor()
        return extractor.extract_from_text(request.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest) -> ReportResponse:
    """
    Generate a security bulletin from vulnerability data.

    Intelligence reporting endpoint.
    """
    try:
        generator = CyberSecurityReportGenerator()
        aggregator = CyberVulnerabilityAggregator()

        vulnerabilities = request.vulnerabilities

        # If no vulnerabilities provided, auto-aggregate from Knowledge Graph
        if not vulnerabilities and request.context_query:
            vulnerabilities = aggregator.aggregate_mission_vulnerabilities(
                request.context_query
            )

        md = generator.generate_markdown_report(vulnerabilities, request.mission_id)
        return ReportResponse(markdown=md)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
