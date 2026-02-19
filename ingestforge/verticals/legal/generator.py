"""
Legal Pleading Generator.

Legal Pleading Template
Orchestrates LLM synthesis for court-ready documents.

JPL Compliance:
- Rule #2: Bounded loops (MAX_PLEADING_FACTS).
- Rule #4: Functions < 60 lines.
- Rule #9: 100% type hints.
"""

from typing import List
from ingestforge.verticals.legal.models import LegalPleadingModel, LegalFact
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.base import GenerationConfig
from ingestforge.core.config_loaders import load_config

# JPL Rule #2: Fixed upper bounds
MAX_PLEADING_FACTS = 50


class LegalPleadingGenerator:
    """Service for generating structured legal pleadings."""

    def __init__(self):
        """Initialize generator with default LLM client."""
        self.config = load_config()
        self.llm = get_llm_client(self.config)

    def generate_markdown(self, model: LegalPleadingModel) -> str:
        """Generates a complete pleading in Markdown format."""
        sections = [
            model.get_caption(),
            "## I. INTRODUCTION",
            "The Plaintiffs hereby submit this pleading based on the following facts and research.",
            "## II. STATEMENT OF FACTS",
            self._format_facts(model.statement_of_facts),
            "## III. ARGUMENT",
            model.legal_argument
            or "The law supports the Plaintiffs' position based on the evidence presented.",
            "## IV. CONCLUSION",
            "WHEREFORE, Plaintiffs pray for judgment as requested.",
            "\nRespectfully submitted,",
            "____________________",
            "Counsel for Plaintiffs",
        ]
        return "\n\n".join(sections)

    def _format_facts(self, facts: List[LegalFact]) -> str:
        """Formats a list of facts with citations.

        Rule #2: Bounded loop.
        Rule #4: Concise helper.
        """
        if not facts:
            return "No specific facts have been asserted at this time."

        formatted = []
        # JPL Rule #2: Strict upper bound on iteration
        for i, fact in enumerate(facts[:MAX_PLEADING_FACTS]):
            cite = f" [Doc: {fact.source_id}"
            if fact.page_number:
                cite += f", p. {fact.page_number}"
            cite += "]"
            formatted.append(f"{i+1}. {fact.text}{cite}")

        return "\n".join(formatted)

    def synthesize_argument(self, query: str, context_chunks: List[str]) -> str:
        """Uses LLM to synthesize a legal argument from context.

        Rule #4: Under 60 lines.
        """
        context_text = "\n\n".join(context_chunks)
        prompt = f"""You are a senior litigation attorney. Synthesize a legal argument for a court pleading based on the following research context. Use professional, persuasive language.

RESEARCH CONTEXT:
{context_text}

LEGAL ISSUE: {query}

ARGUMENT:"""

        gen_config = GenerationConfig(max_tokens=1024, temperature=0.3)
        response = self.llm.generate(prompt, gen_config)
        return response.text if response else "Error generating argument."
