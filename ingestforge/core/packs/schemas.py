"""Forge Pack Schemas.

Defines the structure for pack-specific settings.
Follows NASA JPL Rule #7 (Validation) and Rule #9 (Type Hints).
"""

from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field


class BasePackConfig(BaseModel):
    """Common fields for all Forge Packs."""

    enabled: bool = True
    version: str = "1.0.0"


class LegalPackConfig(BasePackConfig):
    """Specific settings for the Legal Forge."""

    preferred_citation_style: str = Field(
        "bluebook", description="Default citation format"
    )
    pii_redaction_level: int = Field(2, ge=0, le=3, description="0=None, 3=Strict")
    jurisdiction_filter: List[str] = Field(
        default_factory=lambda: ["federal"], description="Scope of case law search"
    )


class ResearchPackConfig(BasePackConfig):
    """Specific settings for the Academic Research Forge."""

    arxiv_max_results: int = Field(10, ge=1, le=50)
    include_preprints: bool = True
    semantic_scholar_api_key: Optional[str] = None


class CyberPackConfig(BasePackConfig):
    """Specific settings for the Cybersecurity Forge."""

    log_format_detect: bool = True
    mitre_attack_version: str = "v14"
    vulnerability_database_url: str = "https://cve.mitre.org"


class GlobalPacksConfig(BaseModel):
    """Container for all pack configurations."""

    legal: LegalPackConfig = Field(default_factory=LegalPackConfig)
    research: ResearchPackConfig = Field(default_factory=ResearchPackConfig)
    cyber: CyberPackConfig = Field(default_factory=CyberPackConfig)
