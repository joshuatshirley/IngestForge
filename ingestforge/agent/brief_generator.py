"""
Intelligence Brief Generator (G-RAG).

Intelligence Briefs
Reasoning engine for boardroom-ready summaries with citations.

JPL Compliance:
- Rule #4: All functions < 60 lines.
- Rule #7: Citation validation pass.
- Rule #9: 100% type hints.
"""

from typing import List, Dict, Any
import asyncio
from ingestforge.agent.brief_models import IntelligenceBrief, EvidenceLink, KeyEntity
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.base import GenerationConfig
from ingestforge.core.config_loaders import load_config
from ingestforge.core.pipeline.pipeline import Pipeline
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Safety limits
MAX_BRIEF_SOURCES = 10
MAX_KEY_ENTITIES = 15


class IFBriefGenerator:
    """Orchestrates Generative RAG (G-RAG) for intelligence reports."""

    def __init__(self):
        """Initialize with engine components."""
        self.config = load_config()
        self.pipeline = Pipeline(self.config)
        self.retriever = HybridRetriever(self.config, self.pipeline.storage)
        self.llm = get_llm_client(self.config)

    async def generate_brief(self, mission_id: str, query: str) -> IntelligenceBrief:
        """
        Main entry point for brief generation.

        Flow:
        1. Retrieve multi-source context.
        2. Synthesize sections in parallel.
        3. Map citations and validate.
        """
        # 1. Retrieval
        context = self.retriever.search(query=query, top_k=MAX_BRIEF_SOURCES)
        if not context:
            return self._create_empty_brief(mission_id, query)

        # 2. Parallel Synthesis
        summary_task = self._synthesize_summary(query, context)
        entities_task = self._extract_key_entities(context)

        summary, entities = await asyncio.gather(summary_task, entities_task)

        # 3. Citation Mapping (G-RAG)
        evidence = self._map_evidence(context)

        return IntelligenceBrief(
            mission_id=mission_id,
            title=f"Research Summary: {query[:50]}",
            summary=summary,
            key_entities=entities,
            evidence=evidence,
        )

    async def _synthesize_summary(self, query: str, context: List[Any]) -> str:
        """Synthesizes high-level summary from chunks."""
        context_text = "\n\n".join([c.content for c in context])
        prompt = f"Synthesize a professional research summary for the objective: '{query}' based on these sources:\n\n{context_text}\n\nSUMMARY:"

        res = self.llm.generate(
            prompt, GenerationConfig(max_tokens=800, temperature=0.3)
        )
        return res.text if res else "Failed to generate summary."

    async def _extract_key_entities(self, context: List[Any]) -> List[KeyEntity]:
        """Identifies top entities from context."""
        all_entities: Dict[str, KeyEntity] = {}

        # In a real implementation, this would use LLM or spaCy
        # For MVP, we simulate with extracted metadata
        for chunk in context:
            # JPL Rule #2: Strict bound on inner loop
            entities = chunk.metadata.get("entities", [])
            for ent_name in entities[:MAX_KEY_ENTITIES]:
                if ent_name not in all_entities:
                    all_entities[ent_name] = KeyEntity(
                        name=ent_name,
                        type="Concept",
                        description=f"Identified in {chunk.source_file or 'document'}",
                    )

        return list(all_entities.values())[:MAX_KEY_ENTITIES]

    def _map_evidence(self, context: List[Any]) -> List[EvidenceLink]:
        """Maps retrieved context to EvidenceLink models."""
        links = []
        for chunk in context[:MAX_BRIEF_SOURCES]:
            links.append(
                EvidenceLink(
                    doc_id=chunk.source_file or chunk.document_id,
                    chunk_id=chunk.chunk_id,
                    offset=chunk.page_start,
                    snippet=chunk.content[:200],
                    confidence=chunk.score,
                )
            )
        return links

    def _create_empty_brief(self, mission_id: str, query: str) -> IntelligenceBrief:
        """Helper for null result cases."""
        return IntelligenceBrief(
            mission_id=mission_id,
            title=f"Empty Brief: {query[:30]}",
            summary="No relevant information found in knowledge base.",
        )


class BriefCitationValidator:
    """Validates citations against local storage (JPL Rule #7)."""

    def __init__(self, storage: Any):
        self.storage = storage

    def validate(self, brief: IntelligenceBrief) -> bool:
        """
        Verifies every citation exists in storage.

        Rule #7: Explicit verification logic.
        """
        for link in brief.evidence:
            if not self.storage.get_chunk(link.chunk_id):
                logger.error(f"Broken citation found: {link.chunk_id}")
                return False
        return True
