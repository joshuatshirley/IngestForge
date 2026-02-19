"""
Legal Vertical Models.

Legal Pleading Template
Defines the structure for court-ready research outputs.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PleadingParty(BaseModel):
    """Represents a plaintiff or defendant."""

    name: str
    role: str  # 'Plaintiff', 'Defendant', 'Petitioner', etc.


class LegalFact(BaseModel):
    """A specific fact extracted from evidence with citation."""

    text: str
    source_id: str
    page_number: Optional[int] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class LegalPleadingModel(BaseModel):
    """Structured data for a legal pleading document."""

    court_name: str = Field(
        ..., description="Name of the court (e.g., Superior Court of California)"
    )
    jurisdiction: str
    case_number: Optional[str] = None
    plaintiffs: List[PleadingParty]
    defendants: List[PleadingParty]
    title: str = Field(..., description="Document title (e.g., COMPLAINT FOR DAMAGES)")

    statement_of_facts: List[LegalFact] = Field(default_factory=list)
    legal_argument: str = ""

    def get_caption(self) -> str:
        """Generates the standard court caption header."""
        p_names = " & ".join([p.name for p in self.plaintiffs])
        d_names = " & ".join([d.name for d in self.defendants])

        caption = f"{self.court_name}\n"
        caption += f"{self.jurisdiction.upper()}\n\n"
        caption += f"{p_names},\n    Plaintiffs,\n\nv.\n\n"
        caption += f"{d_names},\n    Defendants.\n"
        caption += "-" * 40 + "\n"
        caption += f"CASE NO: {self.case_number or 'PENDING'}\n"
        caption += f"{self.title.upper()}\n"
        caption += "-" * 40 + "\n"

        return caption
