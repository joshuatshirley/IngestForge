"""
Code Intelligence Vertical Models.

Task 313: Code understanding with precise citations and AST analysis.
Defines structured outputs for code repository analysis.

JPL Compliance:
- Rule #9: 100% type hints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from ingestforge.core.pipeline.artifacts import SymbolKind


class CodeSymbolLink(BaseModel):
    """Link between a symbol and its location in code."""

    name: str
    kind: SymbolKind
    file_path: str
    line_start: int
    line_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None


class DependencyLink(BaseModel):
    """Link to an external or internal dependency."""

    module: str
    names: List[str] = Field(default_factory=list)
    line_number: int


class CodeFileAnalysis(BaseModel):
    """Detailed intelligence for a single source file."""

    file_path: str
    language: str
    symbols: List[CodeSymbolLink] = Field(default_factory=list)
    imports: List[DependencyLink] = Field(default_factory=list)
    line_count: int
    complexity_score: Optional[int] = None
    quality_notes: List[str] = Field(default_factory=list)


class CodeIntelligenceModel(BaseModel):
    """Aggregate intelligence for a code repository or module."""

    module_name: str
    files: List[CodeFileAnalysis] = Field(default_factory=list)
    total_lines: int
    entry_points: List[str] = Field(default_factory=list)
    architecture_summary: str
    jpl_compliance_issues: List[Dict[str, Any]] = Field(default_factory=list)
