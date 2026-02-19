"""
Code Intelligence Vertical Generator.

Task 313: Code understanding assistant with precise citations.
Synthesizes answers based on CodeIntelligenceModel AST analysis.

JPL Compliance:
- Rule #4: Generation methods < 60 lines.
- Rule #9: Complete type hints.
"""

from typing import List
from ingestforge.verticals.code.models import CodeIntelligenceModel


class CodeIntelligenceGenerator:
    """
    Assistant logic for synthesizing code repository answers.

    Uses AST-aware metadata for high-fidelity citations.
    """

    def synthesize_response(self, query: str, analysis: CodeIntelligenceModel) -> str:
        """
        Synthesize a response based on the analysis.

        Args:
            query: User's natural language query.
            analysis: The structured CodeIntelligenceModel for the repository.

        Returns:
            A synthesized answer with precise citations.
        """
        # Logic for high-level synthesis would go here.
        # For now, we provide the architectural summary.
        response = [
            f"### Code Analysis: {analysis.module_name}",
            analysis.architecture_summary,
            "",
            "#### Precise Citations:",
        ]

        # Link symbols to citations
        for file in analysis.files:
            for symbol in file.symbols:
                if symbol.name.lower() in query.lower():
                    response.append(
                        f"- **{symbol.name}** ({symbol.kind}): "
                        f"`{file.file_path}` lines {symbol.line_start}-{symbol.line_end}"
                    )

        if len(response) <= 3:
            response.append("- No specific symbols found matching your query.")

        return "\n".join(response)

    def identify_entry_points(self, analysis: CodeIntelligenceModel) -> List[str]:
        """Identify potential application entry points."""
        return [f for f in analysis.entry_points]
