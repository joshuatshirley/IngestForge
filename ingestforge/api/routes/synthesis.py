"""RAG Synthesis Router.

RAG Synthesis API Endpoint
Epic: EP-10 (Synthesis & Generative API)
Feature: FE-10-02 (RESTful RAG API)

Implements /v1/synthesize endpoint that combines retrieval with LLM synthesis.

JPL Power of Ten Compliance:
- Rule #1: No recursion
- Rule #2: Fixed upper bounds (MAX_* constants)
- Rule #4: All functions < 60 lines
- Rule #5: Assert preconditions
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    pass

# =============================================================================
# JPL Rule #2: Fixed upper bounds
# =============================================================================

MAX_QUERY_LENGTH = 10_000
MAX_CONTEXT_TOKENS = 8_000
MAX_CITATIONS = 20
MAX_SOURCES = 50
MAX_ANSWER_LENGTH = 4_000
MAX_SNIPPET_LENGTH = 500
MAX_TOP_K = 100


# =============================================================================
# REQUEST/RESPONSE MODELS (Rule #7: Validation)
# =============================================================================


class SynthesizeRequest(BaseModel):
    """Request model for synthesis endpoint.

    Rule #9: Complete type hints for all fields.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Query to synthesize answer for",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=MAX_TOP_K,
        description="Number of chunks to retrieve",
    )
    library: Optional[str] = Field(
        default=None,
        max_length=256,
        description="Library filter",
    )
    threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum relevance threshold",
    )
    stream: bool = Field(
        default=False,
        description="Stream response as SSE",
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    max_tokens: int = Field(
        default=1024,
        ge=100,
        le=MAX_ANSWER_LENGTH,
        description="Maximum tokens in generated answer",
    )

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        """Validate query is not just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace-only")
        return v.strip()


class Citation(BaseModel):
    """Citation linking answer to source artifact.

    Rule #9: Complete type hints.
    """

    artifact_id: str = Field(..., description="Source artifact ID")
    chunk_id: str = Field(..., description="Source chunk ID")
    content_snippet: str = Field(
        ...,
        max_length=MAX_SNIPPET_LENGTH,
        description="Relevant content snippet",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score",
    )
    source_file: Optional[str] = Field(
        default=None,
        description="Source file path",
    )
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number if available",
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section title if available",
    )


class SynthesizeResponse(BaseModel):
    """Response model for synthesis endpoint.

    Rule #9: Complete type hints.
    """

    success: bool = Field(..., description="Whether synthesis succeeded")
    answer: str = Field(default="", description="Synthesized answer")
    citations: List[Citation] = Field(
        default_factory=list,
        description="Citations supporting the answer",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Unique source files used",
    )
    chunks_retrieved: int = Field(
        default=0,
        ge=0,
        description="Number of chunks retrieved",
    )
    retrieval_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on retrieval",
    )
    synthesis_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time spent on synthesis",
    )
    total_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Total processing time",
    )
    message: str = Field(default="", description="Status message")


class SynthesisError(BaseModel):
    """Error response for synthesis failures.

    Rule #9: Complete type hints.
    """

    success: bool = Field(default=False, description="Always false for errors")
    error: str = Field(..., description="Error message")
    error_type: str = Field(default="synthesis_error", description="Error type")
    total_time_ms: float = Field(default=0.0, description="Time before failure")


# =============================================================================
# LAZY IMPORTS (Rule #6: Avoid slow startup)
# =============================================================================


class _LazyDeps:
    """Lazy loader for heavy dependencies."""

    _router = None
    _logger = None

    @classmethod
    def get_router(cls):
        """Get FastAPI router (lazy-loaded)."""
        if cls._router is None:
            try:
                from fastapi import APIRouter

                cls._router = APIRouter(prefix="/v1", tags=["synthesis"])
            except ImportError:
                raise ImportError(
                    "FastAPI is required for API functionality. "
                    "Install with: pip install fastapi uvicorn"
                )
        return cls._router

    @classmethod
    def get_logger(cls):
        """Get logger (lazy-loaded)."""
        if cls._logger is None:
            from ingestforge.core.logging import get_logger

            cls._logger = get_logger(__name__)
        return cls._logger


# Initialize router
router = _LazyDeps.get_router()


# =============================================================================
# SYNTHESIS SERVICE (Rule #4: <60 lines per method)
# =============================================================================


@dataclass
class RetrievedChunk:
    """Chunk retrieved for synthesis context.

    Rule #9: Complete type hints.
    """

    chunk_id: str
    document_id: str
    content: str
    score: float
    source_file: str = ""
    page_start: Optional[int] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result of synthesis operation.

    Rule #9: Complete type hints.
    """

    answer: str
    citations: List[Citation]
    success: bool = True
    error: Optional[str] = None


class SynthesisService:
    """Service for RAG synthesis operations.

    Combines retrieval and LLM generation with provenance tracking.

    Rule #1: No recursion.
    Rule #2: All loops bounded.
    Rule #9: Complete type hints.
    """

    def __init__(self) -> None:
        """Initialize synthesis service."""
        self._pipeline = None
        self._retriever = None
        self._llm_client = None

    def _ensure_initialized(self) -> None:
        """Lazy initialization of dependencies.

        Rule #4: Function <60 lines.
        """
        if self._pipeline is None:
            from ingestforge.core.pipeline.pipeline import Pipeline

            self._pipeline = Pipeline()

        if self._retriever is None:
            from ingestforge.retrieval import HybridRetriever

            self._retriever = HybridRetriever(
                self._pipeline.config, self._pipeline.storage
            )

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        library: Optional[str] = None,
        threshold: float = 0.7,
    ) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for synthesis.

        Rule #4: Function <60 lines.
        Rule #5: Assert preconditions.

        Args:
            query: Search query.
            top_k: Maximum chunks to retrieve.
            library: Optional library filter.
            threshold: Minimum relevance threshold.

        Returns:
            List of retrieved chunks.
        """
        assert query, "query cannot be empty"
        assert 1 <= top_k <= MAX_TOP_K, "top_k out of bounds"

        self._ensure_initialized()

        raw_results = self._retriever.search(
            query=query,
            top_k=top_k,
            library_filter=library,
        )

        chunks: List[RetrievedChunk] = []
        for result in raw_results[:MAX_SOURCES]:
            if result.score < threshold:
                continue

            chunk = RetrievedChunk(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                score=result.score,
                source_file=result.source_file or "",
                page_start=result.page_start,
                section_title=result.section_title,
                metadata=result.metadata or {},
            )
            chunks.append(chunk)

        return chunks

    def _build_context_prompt(
        self,
        query: str,
        chunks: List[RetrievedChunk],
    ) -> str:
        """Build synthesis prompt from retrieved chunks.

        Rule #4: Function <60 lines.

        Args:
            query: User query.
            chunks: Retrieved context chunks.

        Returns:
            Formatted prompt for LLM.
        """
        context_parts: List[str] = []

        for i, chunk in enumerate(chunks[:MAX_CITATIONS]):
            source_info = f"[Source {i + 1}: {chunk.source_file or chunk.document_id}]"
            if chunk.page_start:
                source_info += f" (Page {chunk.page_start})"
            context_parts.append(f"{source_info}\n{chunk.content}\n")

        context = "\n---\n".join(context_parts)

        prompt = f"""Based on the following sources, answer the question.
Cite sources using [Source N] notation for each claim.

SOURCES:
{context}

QUESTION: {query}

ANSWER (with citations):"""

        return prompt

    def synthesize(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        max_tokens: int = 1024,
    ) -> SynthesisResult:
        """Synthesize answer from retrieved chunks.

        Rule #4: Function <60 lines.
        Rule #5: Assert preconditions.

        Args:
            query: User query.
            chunks: Retrieved context chunks.
            max_tokens: Maximum answer length.

        Returns:
            SynthesisResult with answer and citations.
        """
        assert query, "query cannot be empty"

        if not chunks:
            return SynthesisResult(
                answer="No relevant information found to answer this query.",
                citations=[],
                success=True,
            )

        try:
            self._ensure_initialized()

            # Build prompt
            prompt = self._build_context_prompt(query, chunks)

            # Get LLM client
            from ingestforge.llm.factory import get_llm_client
            from ingestforge.llm.base import GenerationConfig

            llm = get_llm_client(self._pipeline.config)
            gen_config = GenerationConfig(
                max_tokens=max_tokens,
                temperature=0.3,
            )

            # Generate answer
            response = llm.generate(prompt, gen_config)
            answer = response.text if response else ""

            # Extract citations
            citations = self._extract_citations(answer, chunks)

            return SynthesisResult(
                answer=answer[:MAX_ANSWER_LENGTH],
                citations=citations,
                success=True,
            )

        except Exception as e:
            logger = _LazyDeps.get_logger()
            logger.error(f"Synthesis failed: {e}")
            return SynthesisResult(
                answer="",
                citations=[],
                success=False,
                error=str(e),
            )

    def _extract_citations(
        self,
        answer: str,
        chunks: List[RetrievedChunk],
    ) -> List[Citation]:
        """Extract citations from answer text.

        Rule #4: Function <60 lines.
        Rule #2: Bounded by MAX_CITATIONS.

        Args:
            answer: Generated answer text.
            chunks: Source chunks.

        Returns:
            List of citations found in answer.
        """
        import re

        citations: List[Citation] = []
        seen_chunks: set = set()

        # Find [Source N] references
        pattern = r"\[Source\s*(\d+)\]"
        matches = re.findall(pattern, answer)

        for match in matches[:MAX_CITATIONS]:
            try:
                idx = int(match) - 1
                if 0 <= idx < len(chunks) and idx not in seen_chunks:
                    chunk = chunks[idx]
                    citation = Citation(
                        artifact_id=chunk.document_id,
                        chunk_id=chunk.chunk_id,
                        content_snippet=chunk.content[:MAX_SNIPPET_LENGTH],
                        relevance_score=chunk.score,
                        source_file=chunk.source_file or None,
                        page_number=chunk.page_start,
                        section_title=chunk.section_title,
                    )
                    citations.append(citation)
                    seen_chunks.add(idx)
            except (ValueError, IndexError):
                continue

        return citations


# =============================================================================
# ENDPOINT HANDLERS (Rule #4: <60 lines each)
# =============================================================================

from fastapi import HTTPException
from fastapi.responses import StreamingResponse


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest) -> SynthesizeResponse:
    """Synthesize answer from retrieved context.

    Rule #4: Function body <60 lines.
    Rule #7: Input validated via Pydantic.
    Rule #9: Complete type hints.

    Args:
        request: Synthesis request parameters.

    Returns:
        SynthesizeResponse with answer and citations.

    Raises:
        HTTPException: If synthesis fails.
    """
    start_time = time.perf_counter()
    logger = _LazyDeps.get_logger()

    try:
        service = SynthesisService()

        # Retrieve context
        retrieval_start = time.perf_counter()
        chunks = service.retrieve_context(
            query=request.query,
            top_k=request.top_k,
            library=request.library,
            threshold=request.threshold,
        )
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000

        # Synthesize answer
        synthesis_start = time.perf_counter()
        result = service.synthesize(
            query=request.query,
            chunks=chunks,
            max_tokens=request.max_tokens,
        )
        synthesis_time = (time.perf_counter() - synthesis_start) * 1000

        total_time = (time.perf_counter() - start_time) * 1000

        if not result.success:
            raise HTTPException(status_code=500, detail=result.error)

        # Extract unique sources
        sources: List[str] = []
        if request.include_sources:
            seen = set()
            for chunk in chunks[:MAX_SOURCES]:
                source = chunk.source_file or chunk.document_id
                if source and source not in seen:
                    sources.append(source)
                    seen.add(source)

        logger.info(
            f"Synthesis completed: {len(chunks)} chunks, "
            f"{len(result.citations)} citations, {total_time:.0f}ms"
        )

        return SynthesizeResponse(
            success=True,
            answer=result.answer,
            citations=result.citations,
            sources=sources,
            chunks_retrieved=len(chunks),
            retrieval_time_ms=retrieval_time,
            synthesis_time_ms=synthesis_time,
            total_time_ms=total_time,
            message=f"Synthesized from {len(chunks)} sources",
        )

    except HTTPException:
        raise

    except Exception as e:
        total_time = (time.perf_counter() - start_time) * 1000
        logger.exception(f"Synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Synthesis failed: {e}",
        ) from e


@router.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest) -> StreamingResponse:
    """Stream synthesis response as Server-Sent Events.

    Rule #4: Function body <60 lines.
    Rule #9: Complete type hints.

    Args:
        request: Synthesis request parameters.

    Returns:
        StreamingResponse with SSE events.
    """
    logger = _LazyDeps.get_logger()

    async def event_generator() -> AsyncIterator[str]:
        """Generate SSE events for streaming response."""
        start_time = time.perf_counter()

        try:
            service = SynthesisService()

            # Send retrieval start event
            yield "event: status\ndata: Retrieving context...\n\n"

            # Retrieve context
            chunks = service.retrieve_context(
                query=request.query,
                top_k=request.top_k,
                library=request.library,
                threshold=request.threshold,
            )

            yield f"event: status\ndata: Retrieved {len(chunks)} chunks\n\n"

            # Synthesize
            yield "event: status\ndata: Generating answer...\n\n"

            result = service.synthesize(
                query=request.query,
                chunks=chunks,
                max_tokens=request.max_tokens,
            )

            if result.success:
                # Send answer
                import json

                answer_data = json.dumps({"answer": result.answer})
                yield f"event: answer\ndata: {answer_data}\n\n"

                # Send citations
                for citation in result.citations:
                    citation_data = json.dumps(citation.model_dump())
                    yield f"event: citation\ndata: {citation_data}\n\n"

                total_time = (time.perf_counter() - start_time) * 1000
                done_data = json.dumps({"total_time_ms": total_time})
                yield f"event: done\ndata: {done_data}\n\n"
            else:
                error_data = json.dumps({"error": result.error})
                yield f"event: error\ndata: {error_data}\n\n"

        except Exception as e:
            import json

            logger.exception(f"Stream synthesis failed: {e}")
            error_data = json.dumps({"error": str(e)})
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def create_synthesis_service() -> SynthesisService:
    """Create a synthesis service instance.

    Returns:
        Configured SynthesisService.
    """
    return SynthesisService()
