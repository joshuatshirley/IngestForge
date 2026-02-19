"""
Cyber Security Report Generator.

Cyber CVE Blueprint
Generates professional security bulletins from extracted vulnerability data.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List
from ingestforge.verticals.cyber.models import CyberVulnerabilityModel


class CyberSecurityReportGenerator:
    """Service for generating boardroom-ready security reports."""

    def generate_markdown_report(
        self, vulnerabilities: List[CyberVulnerabilityModel], mission_id: str
    ) -> str:
        """
        Transforms vulnerability data into a formatted Markdown report.

        Rule #4: Concise aggregator.
        """
        critical_count = sum(1 for v in vulnerabilities if (v.cvss_score or 0) >= 9.0)

        sections = [
            f"# SECURITY INTELLIGENCE BULLETIN: Mission {mission_id}",
            self._render_executive_summary(len(vulnerabilities), critical_count),
            "## VULNERABILITY DETAILS",
            self._render_vulnerability_list(vulnerabilities),
            "## AGGREGATE RISK ASSESSMENT",
            self._calculate_risk_level(vulnerabilities),
        ]

        return "\n\n".join(sections)

    def _render_executive_summary(self, total: int, critical: int) -> str:
        """Renders the executive summary section."""
        return f"""**Total Vulnerabilities Identified**: {total}
**Critical Severity (CVSS >= 9.0)**: {critical}

---
"""

    def _render_vulnerability_list(
        self, vulnerabilities: List[CyberVulnerabilityModel]
    ) -> str:
        """Formats the list of vulnerabilities with citations and severity."""
        lines = []
        for v in vulnerabilities[:20]:  # JPL Rule #2: Bound output
            status = f"[{v.severity}]" if v.severity else ""
            score = f"(Score: {v.cvss_score})" if v.cvss_score else ""
            lines.append(f"### {v.cve_id} {status} {score}")
            lines.append(f"**Summary**: {v.summary}")
            if v.remediation:
                lines.append(f"**Remediation**: {v.remediation}")
            lines.append("")
        return "\n".join(lines)

    def _calculate_risk_level(
        self, vulnerabilities: List[CyberVulnerabilityModel]
    ) -> str:
        """Calculates aggregate risk based on CVSS scores."""
        if not vulnerabilities:
            return "No vulnerabilities detected. Risk Level: LOW"

        max_score = max((v.cvss_score or 0) for v in vulnerabilities)

        if max_score >= 9.0:
            return "🔥 **CRITICAL RISK**: Immediate action required for one or more critical vulnerabilities."
        if max_score >= 7.0:
            return "⚠️ **HIGH RISK**: Significant security gaps identified."
        return "ℹ️ **MEDIUM/LOW RISK**: Routine patching recommended."
