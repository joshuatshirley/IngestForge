"""
Intelligence Brief Models.

Intelligence Briefs
Defines the structure for boardroom-ready research summaries.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class EvidenceLink(BaseModel):
    """Hard link between a claim and a source chunk."""

    doc_id: str
    chunk_id: str
    offset: Optional[int] = Field(None, description="Character offset or page number")
    snippet: str = Field(..., max_length=500)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class KeyEntity(BaseModel):
    """Important entity identified in the brief."""

    name: str
    type: str  # e.g., 'Person', 'Org', 'Concept'
    description: str
    relevance: float = Field(default=1.0, ge=0.0, le=1.0)


from datetime import timezone

# ... existing code ...


class IntelligenceBrief(BaseModel):
    """Boardroom-ready research brief with citations."""

    mission_id: str
    title: str
    summary: str = Field(..., description="Executive summary of the research")
    key_entities: List[KeyEntity] = Field(default_factory=list)
    evidence: List[EvidenceLink] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_markdown(self) -> str:
        """Renders the brief as a structured Markdown document."""
        sections = [
            f"# INTELLIGENCE BRIEF: {self.title}",
            f"**Mission ID**: {self.mission_id} | **Date**: {self.created_at.strftime('%Y-%m-%d')}",
            "## EXECUTIVE SUMMARY",
            self.summary,
            "## KEY ENTITIES",
            "\n".join(
                [
                    f"- **{e.name}** ({e.type}): {e.description}"
                    for e in self.key_entities
                ]
            ),
            "## SUPPORTING EVIDENCE",
            "\n".join(
                [
                    f"- [{ev.doc_id}:{ev.offset or 'N/A'}] \"{ev.snippet}...\""
                    for ev in self.evidence
                ]
            ),
        ]
        return "\n\n".join(sections)
