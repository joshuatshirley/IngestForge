"""
Code Intelligence Vertical Extractor.

Task 313: Code understanding with AST analysis.
Integrates TreeSitterExtractor into the Code Intelligence vertical.

JPL Compliance:
- Rule #4: Extraction functions < 60 lines.
- Rule #9: Complete type hints.
"""

from typing import List
from ingestforge.core.pipeline.artifacts import IFCodeArtifact
from ingestforge.verticals.code.models import (
    CodeFileAnalysis,
    CodeSymbolLink,
    DependencyLink,
)


class CodeIntelligenceExtractor:
    """
    Vertical-specific extractor for code intelligence.

    Rule #4: Focused methods for symbol and import conversion.
    """

    def extract_from_artifact(self, artifact: IFCodeArtifact) -> CodeFileAnalysis:
        """
        Convert IFCodeArtifact to a structured CodeFileAnalysis.

        Args:
            artifact: The IFCodeArtifact containing raw AST extraction data.

        Returns:
            CodeFileAnalysis structured for synthesis and citations.
        """
        symbols = self._convert_symbols(artifact)
        imports = self._convert_imports(artifact)

        return CodeFileAnalysis(
            file_path=artifact.file_path or "unknown",
            language=artifact.language,
            symbols=symbols,
            imports=imports,
            line_count=artifact.line_count,
            complexity_score=artifact.complexity,
            quality_notes=self._generate_quality_notes(artifact),
        )

    def _convert_symbols(self, artifact: IFCodeArtifact) -> List[CodeSymbolLink]:
        """Convert IFCodeArtifact symbols to CodeSymbolLink."""
        links: List[CodeSymbolLink] = []
        for symbol in artifact.symbols:
            links.append(
                CodeSymbolLink(
                    name=symbol.name,
                    kind=symbol.kind,
                    file_path=artifact.file_path or "unknown",
                    line_start=symbol.line_start,
                    line_end=symbol.line_end,
                    signature=symbol.signature,
                    docstring=symbol.docstring,
                )
            )
        return links

    def _convert_imports(self, artifact: IFCodeArtifact) -> List[DependencyLink]:
        """Convert IFCodeArtifact imports to DependencyLink."""
        links: List[DependencyLink] = []
        for imp in artifact.imports:
            links.append(
                DependencyLink(
                    module=imp.module,
                    names=imp.names,
                    line_number=imp.line_number,
                )
            )
        return links

    def _generate_quality_notes(self, artifact: IFCodeArtifact) -> List[str]:
        """Simple rule-based quality analysis for the vertical."""
        notes: List[str] = []
        for symbol in artifact.symbols:
            if symbol.line_end - symbol.line_start > 60:
                notes.append(
                    f"Rule #4 Violation: Function '{symbol.name}' exceeds 60 lines "
                    f"({symbol.line_end - symbol.line_start} lines)."
                )
        if not artifact.symbols and artifact.line_count > 10:
            notes.append(
                "No symbols extracted - possibly missing parser or structural issues."
            )
        return notes
