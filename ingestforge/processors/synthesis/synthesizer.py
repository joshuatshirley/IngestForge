"""
Generative Synthesis Processor for IngestForge.

Turns retrieved Knowledge Graph artifacts into a human-readable,
cited summary.

Follows NASA JPL Power of Ten rules.
"""

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import Field

from ingestforge.core.pipeline.interfaces import IFArtifact, IFProcessor
from ingestforge.core.pipeline.artifacts import (
    IFTextArtifact,
    IFChunkArtifact,
    IFFailureArtifact,
)

if TYPE_CHECKING:
    from ingestforge.llm.base import BaseLLMClient

logger = logging.getLogger(__name__)

# =============================================================================
# JPL Rule #2: Fixed upper bounds
# =============================================================================

MAX_CONTEXT_TOKENS = 16000
MAX_CONTEXT_CHUNKS = 20
MAX_CITATIONS = 50
MAX_ANSWER_LENGTH = 8000
MAX_SNIPPET_LENGTH = 500
CHARS_PER_TOKEN = 4  # Rough estimate


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class SynthesisCitation:
    """Citation linking a fact to its source artifact.

    Rule #9: Complete type hints.
    """

    artifact_id: str
    document_id: str
    content_snippet: str
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    source_file: Optional[str] = None
    relevance_score: float = 0.0


@dataclass
class SynthesisContext:
    """Context for synthesis operation.

    Rule #9: Complete type hints.
    """

    query: str
    chunks: List[IFChunkArtifact] = field(default_factory=list)
    max_tokens: int = 1024


class IFSynthesisArtifact(IFArtifact):
    """Artifact containing synthesis result with citations.

    Stores synthesized answer with provenance tracking.

    Rule #9: Complete type hints.
    """

    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Synthesized answer")
    citations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Citations supporting the answer",
    )
    source_chunks: int = Field(
        default=0,
        description="Number of source chunks used",
    )
    context_tokens: int = Field(
        default=0,
        description="Estimated tokens in context",
    )

    def derive(self, processor_id: str, **kwargs: Any) -> "IFSynthesisArtifact":
        """Create a derived synthesis artifact."""
        new_provenance = self.provenance + [processor_id]
        new_root_id = (
            self.root_artifact_id if self.root_artifact_id else self.artifact_id
        )
        new_depth = self.lineage_depth + 1
        return self.model_copy(
            update={
                "parent_id": self.artifact_id,
                "provenance": new_provenance,
                "root_artifact_id": new_root_id,
                "lineage_depth": new_depth,
                **kwargs,
            }
        )


# =============================================================================
# HELPER FUNCTIONS (Rule #1: Extracted for clarity)
# =============================================================================


def _estimate_tokens(text: str) -> int:
    """Estimate token count from text length.

    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.
    """
    assert text is not None, "text cannot be None"
    return len(text) // CHARS_PER_TOKEN


def _create_failure(
    processor_id: str,
    parent_id: str,
    message: str,
) -> IFFailureArtifact:
    """Create a failure artifact.

    Rule #1: Extracted helper.
    """
    return IFFailureArtifact(
        artifact_id=str(uuid.uuid4()),
        error_message=message,
        failed_processor_id=processor_id,
        parent_id=parent_id,
    )


def _extract_chunks_from_artifact(artifact: IFArtifact) -> List[IFChunkArtifact]:
    """Extract chunk artifacts from input.

    Rule #4: Function < 60 lines.
    Rule #5: Assert preconditions.

    Supports:
    - IFChunkArtifact: Returns as single-item list
    - IFTextArtifact with 'chunks' metadata: Extracts chunk data
    """
    assert artifact is not None, "artifact cannot be None"

    if isinstance(artifact, IFChunkArtifact):
        return [artifact]

    # Check for chunks in metadata
    chunks_data = artifact.metadata.get("synthesis_chunks", [])
    if not chunks_data:
        return []

    result: List[IFChunkArtifact] = []
    for i, chunk_data in enumerate(chunks_data[:MAX_CONTEXT_CHUNKS]):
        if isinstance(chunk_data, IFChunkArtifact):
            result.append(chunk_data)
        elif isinstance(chunk_data, dict):
            # Reconstruct from dict
            chunk = IFChunkArtifact(
                artifact_id=chunk_data.get("artifact_id", str(uuid.uuid4())),
                document_id=chunk_data.get("document_id", "unknown"),
                content=chunk_data.get("content", ""),
                chunk_index=chunk_data.get("chunk_index", i),
                metadata=chunk_data.get("metadata", {}),
            )
            result.append(chunk)

    return result


def _build_citation_prompt(
    query: str,
    chunks: List[IFChunkArtifact],
) -> str:
    """Build synthesis prompt with hard-citation requirements.

    AC: Prompt enforces hard-citations.
    Rule #4: Function < 60 lines.
    """
    assert query, "query cannot be empty"

    context_parts: List[str] = []

    for i, chunk in enumerate(chunks[:MAX_CONTEXT_CHUNKS]):
        # Extract source info from metadata
        source_file = chunk.metadata.get("source_file", chunk.document_id)
        page = chunk.metadata.get("page_start", "")
        section = chunk.metadata.get("section_title", "")

        # Build source reference
        source_ref = f"[Doc:{source_file}"
        if page:
            source_ref += f":Page {page}"
        source_ref += "]"

        header = f"--- SOURCE {i + 1} {source_ref} ---"
        if section:
            header += f"\nSection: {section}"

        context_parts.append(f"{header}\n{chunk.content}\n")

    context = "\n".join(context_parts)

    prompt = f"""You are a precise research assistant. Answer the question based ONLY on the provided sources.

CRITICAL CITATION RULES:
1. Every factual claim MUST include a citation in the format [Doc:filename:Page N]
2. If a fact cannot be attributed to a source, DO NOT include it
3. If sources conflict, explicitly state the discrepancy
4. Never add information not present in the sources

SOURCES:
{context}

QUESTION: {query}

ANSWER (every fact must cite [Doc:filename:Page N]):"""

    return prompt


def _extract_citations(
    answer: str,
    chunks: List[IFChunkArtifact],
) -> List[SynthesisCitation]:
    """Extract citation references from generated answer.

    Rule #4: Function < 60 lines.
    Rule #2: Bounded by MAX_CITATIONS.
    """
    citations: List[SynthesisCitation] = []
    seen: set = set()

    # Pattern: [Doc:filename] or [Doc:filename:Page N]
    pattern = r"\[Doc:([^\]]+)\]"
    matches = re.findall(pattern, answer)

    for match in matches[:MAX_CITATIONS]:
        parts = match.split(":")
        doc_ref = parts[0] if parts else ""
        page = None

        # Extract page number if present
        if len(parts) > 1 and "Page" in parts[-1]:
            page_match = re.search(r"Page\s*(\d+)", parts[-1])
            if page_match:
                page = int(page_match.group(1))

        # Find matching chunk
        for chunk in chunks:
            source_file = chunk.metadata.get("source_file", chunk.document_id)
            chunk_page = chunk.metadata.get("page_start")

            # Match by filename
            if doc_ref in source_file or source_file in doc_ref:
                cite_key = f"{chunk.artifact_id}:{page}"
                if cite_key not in seen:
                    citation = SynthesisCitation(
                        artifact_id=chunk.artifact_id,
                        document_id=chunk.document_id,
                        content_snippet=chunk.content[:MAX_SNIPPET_LENGTH],
                        page_number=page or chunk_page,
                        section_title=chunk.metadata.get("section_title"),
                        source_file=source_file,
                    )
                    citations.append(citation)
                    seen.add(cite_key)
                break

    return citations


def _extract_query(artifact: IFArtifact) -> str:
    """Extract synthesis query from artifact.

    Rule #1: Extracted helper for process().
    Rule #4: Function < 60 lines.
    """
    query = artifact.metadata.get("synthesis_query", "")
    if not query and isinstance(artifact, IFTextArtifact):
        query = artifact.content
    return query


def _format_citations(citations: List[SynthesisCitation]) -> List[Dict[str, Any]]:
    """Format citations for artifact storage.

    Rule #1: Extracted helper for process().
    Rule #4: Function < 60 lines.
    """
    return [
        {
            "artifact_id": c.artifact_id,
            "document_id": c.document_id,
            "snippet": c.content_snippet,
            "page": c.page_number,
            "section": c.section_title,
            "source_file": c.source_file,
        }
        for c in citations
    ]


def _build_synthesis_result(
    artifact: IFArtifact,
    processor_id: str,
    query: str,
    answer: str,
    citations: List[SynthesisCitation],
    chunks: List[IFChunkArtifact],
    context_tokens: int,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> IFSynthesisArtifact:
    """Build synthesis result artifact.

    Rule #1: Extracted helper for process().
    Rule #4: Function < 60 lines.
    """
    return IFSynthesisArtifact(
        artifact_id=str(uuid.uuid4()),
        parent_id=artifact.artifact_id,
        provenance=artifact.provenance + [processor_id],
        root_artifact_id=artifact.effective_root_id,
        lineage_depth=artifact.lineage_depth + 1,
        query=query,
        answer=answer,
        citations=_format_citations(citations),
        source_chunks=len(chunks),
        context_tokens=context_tokens,
        metadata={
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )


# =============================================================================
# MAIN PROCESSOR
# =============================================================================


class IFSynthesisProcessor(IFProcessor):
    """Processor that synthesizes answers from retrieved chunks.

    Generative Synthesis Engine
    Epic: EP-10 (Synthesis & Generative API)

    GWT Specification:
    - Given: A set of linked IFChunkArtifacts
    - When: Synthesis is requested
    - Then: Return a narrative summary where every fact includes a [Doc:Page] citation

    JPL Power of Ten Compliance:
    - Rule #1: No recursion
    - Rule #2: Fixed bounds (MAX_CONTEXT_TOKENS, MAX_CITATIONS)
    - Rule #4: All methods < 60 lines
    - Rule #5: Assert preconditions
    - Rule #7: Check all return values
    - Rule #9: Complete type hints
    """

    def __init__(
        self,
        llm_client: Optional["BaseLLMClient"] = None,
        model_name: str = "default",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> None:
        """Initialize synthesis processor.

        Rule #4: Function < 60 lines.

        Args:
            llm_client: LLM client for generation (lazy-loaded if None).
            model_name: Model identifier for lazy loading.
            max_tokens: Maximum tokens in generated answer.
            temperature: Generation temperature.
        """
        self._llm_client = llm_client
        self._model_name = model_name
        self._max_tokens = min(max_tokens, MAX_ANSWER_LENGTH)
        self._temperature = temperature
        self._initialized = llm_client is not None

    @property
    def processor_id(self) -> str:
        """Unique identifier for this processor."""
        return "if.synthesis.v1"

    @property
    def version(self) -> str:
        """SemVer version of this processor."""
        return "1.0.0"

    def is_available(self) -> bool:
        """Check if processor dependencies are available.

        Rule #7: Return explicit boolean.
        """
        try:
            from ingestforge.llm.factory import get_llm_client

            return True
        except ImportError:
            return False

    def _ensure_llm(self) -> None:
        """Lazy-initialize LLM client.

        Rule #4: Function < 60 lines.
        """
        if self._llm_client is not None:
            return

        from ingestforge.llm.factory import get_llm_client
        from ingestforge.core.config_loaders import load_config

        config = load_config()
        self._llm_client = get_llm_client(config)
        self._initialized = True

    def _validate_context_length(
        self,
        chunks: List[IFChunkArtifact],
    ) -> int:
        """Validate total context is within bounds.

        AC: JPL Rule #5 - Assert context length < 16k tokens.
        Rule #4: Function < 60 lines.

        Args:
            chunks: Context chunks to validate.

        Returns:
            Estimated token count.

        Raises:
            ValueError: If context exceeds MAX_CONTEXT_TOKENS.
        """
        total_chars = sum(len(c.content) for c in chunks)
        estimated_tokens = total_chars // CHARS_PER_TOKEN

        assert estimated_tokens < MAX_CONTEXT_TOKENS, (
            f"Context length ({estimated_tokens} tokens) exceeds "
            f"maximum ({MAX_CONTEXT_TOKENS} tokens)"
        )

        return estimated_tokens

    def _generate_answer(
        self,
        prompt: str,
    ) -> str:
        """Generate answer using LLM.

        Rule #4: Function < 60 lines.
        Rule #7: Check return value.
        """
        self._ensure_llm()

        from ingestforge.llm.base import GenerationConfig

        gen_config = GenerationConfig(
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        response = self._llm_client.generate(prompt, gen_config)

        if response is None or not response.text:
            logger.warning("LLM returned empty response")
            return "Unable to generate a response from the provided sources."

        return response.text[:MAX_ANSWER_LENGTH]

    def process(self, artifact: IFArtifact) -> IFArtifact:
        """Process artifact to generate synthesis.

        Rule #4: Function < 60 lines (refactored with helpers).
        Rule #5: Assert preconditions.
        Rule #7: Check return values.

        Args:
            artifact: Input artifact containing query and chunks.

        Returns:
            IFSynthesisArtifact with answer and citations, or IFFailureArtifact.
        """
        assert artifact is not None, "artifact cannot be None"

        # Extract and validate query
        query = _extract_query(artifact)
        if not query:
            return _create_failure(
                self.processor_id, artifact.artifact_id, "No synthesis query provided"
            )

        # Extract and validate chunks
        chunks = _extract_chunks_from_artifact(artifact)
        if not chunks:
            return _create_failure(
                self.processor_id, artifact.artifact_id, "No context chunks provided"
            )

        try:
            # Validate context length (JPL Rule #5)
            context_tokens = self._validate_context_length(chunks)

            # Build prompt, generate answer, extract citations
            prompt = _build_citation_prompt(query, chunks)
            answer = self._generate_answer(prompt)
            citations = _extract_citations(answer, chunks)

            # AC: JPL Rule #7 - Fail-fast if LLM output lacks citations
            if not citations:
                logger.warning("LLM output contains no valid citations")
                return _create_failure(
                    self.processor_id,
                    artifact.artifact_id,
                    "Synthesis failed: LLM output lacks required citations. "
                    "All claims must be supported by [Doc:Page] references.",
                )

            # Build result artifact using helper
            return _build_synthesis_result(
                artifact,
                self.processor_id,
                query,
                answer,
                citations,
                chunks,
                context_tokens,
                self._model_name,
                self._max_tokens,
                self._temperature,
            )

        except AssertionError as e:
            return _create_failure(self.processor_id, artifact.artifact_id, str(e))
        except Exception as e:
            logger.exception(f"Synthesis failed: {e}")
            return _create_failure(
                self.processor_id, artifact.artifact_id, f"Synthesis error: {e}"
            )

    def teardown(self) -> bool:
        """Clean up resources.

        Rule #7: Return explicit boolean.
        """
        self._llm_client = None
        self._initialized = False
        return True
