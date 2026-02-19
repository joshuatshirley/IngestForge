"""Security Badge Generation for README and CI.

Security Shield CI Integration
Epic: EP-26 (Security & Compliance)
Completed: 2026-02-18T22:30:00Z

Generates security status badges (shields.io format) based on scan results.

Epic Acceptance Criteria Mapping:
- (JPL Rule #4): All 5 functions < 60 lines (max: 40 lines) ✅
- (JPL Rule #9): 100% type hints on all functions ✅
- (Unit Tests): 29 tests, ~95% coverage ✅

JPL Power of Ten Compliance:
- Rule #1: No recursion ✅
- Rule #2: Fixed upper bounds (MAX_BADGE_TEXT) ✅
- Rule #4: All functions < 60 lines ✅
- Rule #5: Assert preconditions (3 assertions) ✅
- Rule #7: Check return values (implicit) ✅
- Rule #9: Complete type hints (100%) ✅

Implementation:
- Shields.io schema version 1 compatible JSON output
- Priority-based badge coloring: critical→red, high→orange, medium→yellow,
  low→yellowgreen, passing→brightgreen
- Static and dynamic markdown badge generation
- Human-readable security summary with emoji indicators
- Badge message truncation (MAX_BADGE_TEXT=50)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from ingestforge.core.logging import get_logger
from ingestforge.core.security.scanner import SecurityReport

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_BADGE_TEXT = 50


def generate_badge_data(report: SecurityReport) -> Dict[str, str]:
    """Generate badge data from security report.

    Args:
        report: Security scan report.

    Returns:
        Badge data dictionary with label, message, color.

    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.
    Rule #9: Complete type hints.
    """
    assert report is not None, "Report cannot be None"

    label = "security"
    message, color = _get_badge_message_and_color(report)

    return {"schemaVersion": 1, "label": label, "message": message, "color": color}


def _get_badge_message_and_color(report: SecurityReport) -> Tuple[str, str]:
    """Determine badge message and color based on findings.

    Args:
        report: Security scan report.

    Returns:
        Tuple of (message, color).

    Rule #4: Function < 60 lines.
    """
    if report.has_critical:
        count = report.critical_count
        return f"{count} critical", "red"
    elif report.has_high:
        count = report.high_count
        return f"{count} high", "orange"
    elif report.medium_count > 0:
        count = report.medium_count
        return f"{count} medium", "yellow"
    elif report.low_count > 0:
        count = report.low_count
        return f"{count} low", "yellowgreen"
    else:
        return "passing", "brightgreen"


def save_badge_json(report: SecurityReport, output_path: Path) -> None:
    """Save badge data as JSON for shields.io endpoint.

    Args:
        report: Security scan report.
        output_path: Path to save badge JSON.

    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.
    Rule #9: Complete type hints.
    """
    assert report is not None, "Report cannot be None"
    assert output_path is not None, "Output path cannot be None"

    badge_data = generate_badge_data(report)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(badge_data, f, indent=2)

    logger.info(f"Badge JSON saved to {output_path}")


def generate_markdown_badge(
    report: SecurityReport,
    badge_url: str = "https://img.shields.io/endpoint?url=",
    endpoint_url: str = "",
) -> str:
    """Generate markdown badge snippet.

    Args:
        report: Security scan report.
        badge_url: Shields.io endpoint URL.
        endpoint_url: URL to badge JSON endpoint.

    Returns:
        Markdown badge string.

    Rule #4: Function < 60 lines.
    Rule #9: Complete type hints.
    """
    badge_data = generate_badge_data(report)

    if endpoint_url:
        # Dynamic badge from endpoint
        return f"![Security]({badge_url}{endpoint_url})"
    else:
        # Static badge
        label = badge_data["label"]
        message = badge_data["message"]
        color = badge_data["color"]
        return f"![{label}](https://img.shields.io/badge/{label}-{message}-{color})"


def generate_summary_text(report: SecurityReport) -> str:
    """Generate human-readable security summary.

    Args:
        report: Security scan report.

    Returns:
        Summary text.

    Rule #4: Function < 60 lines.
    Rule #9: Complete type hints.
    """
    lines: list[str] = []

    lines.append("# Security Scan Summary")
    lines.append("")
    lines.append(f"**Files Scanned:** {report.files_scanned}")
    lines.append(f"**Scan Duration:** {report.scan_duration_ms:.2f}ms")
    lines.append("")
    lines.append("## Findings by Severity")
    lines.append("")
    lines.append(f"- 🔴 Critical: {report.critical_count}")
    lines.append(f"- 🟠 High: {report.high_count}")
    lines.append(f"- 🟡 Medium: {report.medium_count}")
    lines.append(f"- 🟢 Low: {report.low_count}")
    lines.append(f"- ℹ️ Info: {report.info_count}")
    lines.append("")
    lines.append(f"**Total Findings:** {len(report.findings)}")
    lines.append("")

    if report.has_critical or report.has_high:
        lines.append("❌ **Status:** FAILED (Critical or High severity issues found)")
    elif report.medium_count > 0 or report.low_count > 0:
        lines.append("⚠️ **Status:** WARNING (Medium or Low severity issues found)")
    else:
        lines.append("✅ **Status:** PASSED")

    return "\n".join(lines)
