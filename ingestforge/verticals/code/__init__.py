"""
Code Intelligence Vertical.
Task 313: Precise code understanding and citation.
"""

from ingestforge.verticals.code.models import CodeIntelligenceModel, CodeFileAnalysis
from ingestforge.verticals.code.extractor import CodeIntelligenceExtractor
from ingestforge.verticals.code.generator import CodeIntelligenceGenerator

__all__ = [
    "CodeIntelligenceModel",
    "CodeFileAnalysis",
    "CodeIntelligenceExtractor",
    "CodeIntelligenceGenerator",
]
