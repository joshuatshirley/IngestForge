"""HyDE (Hypothetical Document Embeddings) Generator.

Generates fake answers to queries to improve semantic retrieval recall.
Follows NASA JPL Rule #4 (Modular) and Rule #7 (Validation).
"""

from __future__ import annotations
from ingestforge.core.logging import get_logger
from ingestforge.llm.base import LLMClient

logger = get_logger(__name__)


class HyDEGenerator:
    """Logic for generating hypothetical documents."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def generate_hypothetical_doc(self, query: str) -> str:
        """Create a hypothetical answer to the query.

        Rule #1: Flat logic with early returns.
        Rule #7: Validate query.
        """
        if not query or len(query.strip()) < 5:
            return ""

        prompt = (
            "You are an expert researcher. Provide a brief, one-paragraph technical "
            "answer to the following question. Focus on factual statements and "
            "standard terminology. Do not cite sources.\n\n"
            f"QUESTION: {query}\n\n"
            "ANSWER:"
        )

        try:
            # Generate the 'fake' document
            response = self._llm.generate(prompt)
            logger.info(f"HyDE document generated for query: {query[:30]}...")
            return response.strip()
        except Exception as e:
            logger.debug(f"HyDE generation failed: {e}")
            return query  # Fallback to original query
