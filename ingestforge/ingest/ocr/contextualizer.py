"""Multi-Modal Contextualizer.

Combines AI visual summaries with spatially-linked captions.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from typing import Optional
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class MultiModalContextualizer:
    """Logic for synthesizing a unified visual evidence description."""

    def contextualize(self, ai_summary: str, caption: Optional[str]) -> str:
        """Merge AI analysis with document captions.

        Rule #1: Linear logic with early returns.
        Rule #7: Handle None values gracefully.
        """
        if not ai_summary and not caption:
            return ""
        if not caption:
            return ai_summary

        if not ai_summary:
            return f"[CAPTURED CAPTION]: {caption}"

        return f"ANALYSIS: {ai_summary}\n" f"SOURCE CAPTION: {caption}"
