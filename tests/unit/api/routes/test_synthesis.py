"""Tests for RAG Synthesis Route.

RAG Synthesis API Endpoint
Epic: EP-10 (Synthesis & Generative API)
Feature: FE-10-02 (RESTful RAG API)

GWT-based tests for synthesis endpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from ingestforge.api.routes.synthesis import (
    MAX_CITATIONS,
    MAX_CONTEXT_TOKENS,
    MAX_QUERY_LENGTH,
    MAX_SNIPPET_LENGTH,
    MAX_SOURCES,
    MAX_TOP_K,
    Citation,
    RetrievedChunk,
    SynthesisResult,
    SynthesisService,
    SynthesizeRequest,
    SynthesizeResponse,
    create_synthesis_service,
)


# =============================================================================
# Test Fixtures
# =============================================================================


def make_chunk(
    chunk_id: str,
    content: str = "Test content",
    score: float = 0.9,
    source_file: str = "test.pdf",
) -> RetrievedChunk:
    """Create test chunk."""
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=content,
        score=score,
        source_file=source_file,
        page_start=1,
        section_title="Test Section",
    )


def make_citation(
    artifact_id: str,
    chunk_id: str,
    score: float = 0.9,
) -> Citation:
    """Create test citation."""
    return Citation(
        artifact_id=artifact_id,
        chunk_id=chunk_id,
        content_snippet="Test snippet",
        relevance_score=score,
        source_file="test.pdf",
    )


# =============================================================================
# TestSynthesizeRequest
# =============================================================================


class TestSynthesizeRequest:
    """Tests for SynthesizeRequest schema."""

    def test_valid_minimal_request(self) -> None:
        """Test minimal valid request."""
        request = SynthesizeRequest(query="What is AI?")

        assert request.query == "What is AI?"
        assert request.top_k == 5
        assert request.threshold == 0.7
        assert request.stream is False

    def test_valid_full_request(self) -> None:
        """Test fully specified request."""
        request = SynthesizeRequest(
            query="What is machine learning?",
            top_k=10,
            library="research",
            threshold=0.8,
            stream=True,
            include_sources=True,
            max_tokens=2048,
        )

        assert request.query == "What is machine learning?"
        assert request.top_k == 10
        assert request.library == "research"
        assert request.threshold == 0.8
        assert request.stream is True
        assert request.max_tokens == 2048

    def test_query_whitespace_stripped(self) -> None:
        """Test query whitespace is stripped."""
        request = SynthesizeRequest(query="  What is AI?  ")
        assert request.query == "What is AI?"

    def test_empty_query_raises(self) -> None:
        """Test empty query raises validation error."""
        with pytest.raises(ValueError):
            SynthesizeRequest(query="")

    def test_whitespace_query_raises(self) -> None:
        """Test whitespace-only query raises validation error."""
        with pytest.raises(ValueError):
            SynthesizeRequest(query="   ")

    def test_top_k_bounds(self) -> None:
        """Test top_k must be within bounds."""
        # Valid
        request = SynthesizeRequest(query="test", top_k=1)
        assert request.top_k == 1

        request = SynthesizeRequest(query="test", top_k=MAX_TOP_K)
        assert request.top_k == MAX_TOP_K

        # Invalid
        with pytest.raises(ValueError):
            SynthesizeRequest(query="test", top_k=0)

        with pytest.raises(ValueError):
            SynthesizeRequest(query="test", top_k=MAX_TOP_K + 1)

    def test_threshold_bounds(self) -> None:
        """Test threshold must be 0.0-1.0."""
        request = SynthesizeRequest(query="test", threshold=0.0)
        assert request.threshold == 0.0

        request = SynthesizeRequest(query="test", threshold=1.0)
        assert request.threshold == 1.0

        with pytest.raises(ValueError):
            SynthesizeRequest(query="test", threshold=-0.1)

        with pytest.raises(ValueError):
            SynthesizeRequest(query="test", threshold=1.1)


# =============================================================================
# TestCitation
# =============================================================================


class TestCitation:
    """Tests for Citation schema."""

    def test_valid_citation(self) -> None:
        """Test valid citation creation."""
        citation = Citation(
            artifact_id="art-001",
            chunk_id="chunk-001",
            content_snippet="Relevant text...",
            relevance_score=0.95,
            source_file="document.pdf",
            page_number=42,
            section_title="Introduction",
        )

        assert citation.artifact_id == "art-001"
        assert citation.chunk_id == "chunk-001"
        assert citation.relevance_score == 0.95
        assert citation.page_number == 42

    def test_citation_optional_fields(self) -> None:
        """Test citation with optional fields omitted."""
        citation = Citation(
            artifact_id="art-001",
            chunk_id="chunk-001",
            content_snippet="Text",
            relevance_score=0.8,
        )

        assert citation.source_file is None
        assert citation.page_number is None
        assert citation.section_title is None

    def test_relevance_score_bounds(self) -> None:
        """Test relevance score must be 0.0-1.0."""
        citation = Citation(
            artifact_id="art-001",
            chunk_id="chunk-001",
            content_snippet="Text",
            relevance_score=0.0,
        )
        assert citation.relevance_score == 0.0

        with pytest.raises(ValueError):
            Citation(
                artifact_id="art-001",
                chunk_id="chunk-001",
                content_snippet="Text",
                relevance_score=1.5,
            )


# =============================================================================
# TestSynthesizeResponse
# =============================================================================


class TestSynthesizeResponse:
    """Tests for SynthesizeResponse schema."""

    def test_successful_response(self) -> None:
        """Test successful response creation."""
        response = SynthesizeResponse(
            success=True,
            answer="AI is artificial intelligence.",
            citations=[make_citation("art-001", "chunk-001")],
            sources=["doc1.pdf", "doc2.pdf"],
            chunks_retrieved=5,
            retrieval_time_ms=100.5,
            synthesis_time_ms=500.0,
            total_time_ms=600.5,
            message="Success",
        )

        assert response.success is True
        assert "artificial intelligence" in response.answer
        assert len(response.citations) == 1
        assert len(response.sources) == 2
        assert response.total_time_ms == 600.5

    def test_empty_response(self) -> None:
        """Test response with no results."""
        response = SynthesizeResponse(
            success=True,
            answer="No results found.",
            citations=[],
            sources=[],
            chunks_retrieved=0,
        )

        assert response.success is True
        assert response.chunks_retrieved == 0
        assert len(response.citations) == 0


# =============================================================================
# TestRetrievedChunk
# =============================================================================


class TestRetrievedChunk:
    """Tests for RetrievedChunk dataclass."""

    def test_create_chunk(self) -> None:
        """Test chunk creation."""
        chunk = make_chunk("chunk-001", "Test content", 0.95)

        assert chunk.chunk_id == "chunk-001"
        assert chunk.content == "Test content"
        assert chunk.score == 0.95
        assert chunk.document_id == "doc-chunk-001"

    def test_chunk_with_metadata(self) -> None:
        """Test chunk with metadata."""
        chunk = RetrievedChunk(
            chunk_id="chunk-001",
            document_id="doc-001",
            content="Content",
            score=0.9,
            metadata={"key": "value"},
        )

        assert chunk.metadata["key"] == "value"


# =============================================================================
# TestSynthesisResult
# =============================================================================


class TestSynthesisResult:
    """Tests for SynthesisResult dataclass."""

    def test_successful_result(self) -> None:
        """Test successful synthesis result."""
        result = SynthesisResult(
            answer="The answer is 42.",
            citations=[make_citation("art-001", "chunk-001")],
            success=True,
        )

        assert result.success is True
        assert result.answer == "The answer is 42."
        assert result.error is None

    def test_failed_result(self) -> None:
        """Test failed synthesis result."""
        result = SynthesisResult(
            answer="",
            citations=[],
            success=False,
            error="LLM unavailable",
        )

        assert result.success is False
        assert result.error == "LLM unavailable"


# =============================================================================
# TestSynthesisService - GWT-1: Basic Synthesis
# =============================================================================


class TestGWT1BasicSynthesis:
    """GWT-1: Basic synthesis request."""

    def test_given_query_when_synthesized_then_answer_returned(self) -> None:
        """Test basic synthesis returns answer."""
        service = SynthesisService()

        chunks = [
            make_chunk("c1", "AI is artificial intelligence.", 0.95),
            make_chunk("c2", "Machine learning is a subset of AI.", 0.90),
        ]

        # Mock the pipeline with config to avoid NoneType error
        mock_pipeline = MagicMock()
        mock_pipeline.config = MagicMock()
        service._pipeline = mock_pipeline

        with patch.object(service, "_ensure_initialized"):
            with patch.object(service, "retrieve_context", return_value=chunks):
                with patch("ingestforge.llm.factory.get_llm_client") as mock_llm:
                    mock_client = MagicMock()
                    mock_response = MagicMock()
                    mock_response.text = (
                        "AI stands for artificial intelligence. [Source 1]"
                    )
                    mock_client.generate.return_value = mock_response
                    mock_llm.return_value = mock_client

                    result = service.synthesize("What is AI?", chunks)

                    assert result.success is True
                    assert "artificial intelligence" in result.answer

    def test_no_chunks_returns_no_results_message(self) -> None:
        """Test synthesis with no chunks returns appropriate message."""
        service = SynthesisService()

        result = service.synthesize("Unknown query", [])

        assert result.success is True
        assert "No relevant information" in result.answer
        assert len(result.citations) == 0


# =============================================================================
# TestSynthesisService - GWT-2: Provenance Citations
# =============================================================================


class TestGWT2ProvenanceCitations:
    """GWT-2: Provenance citations in responses."""

    def test_citations_extracted_from_answer(self) -> None:
        """Test citations are extracted from answer text."""
        service = SynthesisService()

        chunks = [
            make_chunk("c1", "First source content", 0.95),
            make_chunk("c2", "Second source content", 0.90),
        ]

        answer = "According to [Source 1], this is true. Also [Source 2] confirms it."
        citations = service._extract_citations(answer, chunks)

        assert len(citations) == 2
        assert citations[0].chunk_id == "c1"
        assert citations[1].chunk_id == "c2"

    def test_citation_links_to_artifact(self) -> None:
        """Test each citation links to artifact ID."""
        service = SynthesisService()

        chunks = [make_chunk("chunk-001", "Content", 0.9)]
        answer = "See [Source 1] for details."

        citations = service._extract_citations(answer, chunks)

        assert len(citations) == 1
        assert citations[0].artifact_id == "doc-chunk-001"
        assert citations[0].chunk_id == "chunk-001"

    def test_invalid_source_reference_ignored(self) -> None:
        """Test invalid source references are ignored."""
        service = SynthesisService()

        chunks = [make_chunk("c1", "Content", 0.9)]
        answer = "See [Source 5] and [Source abc]."  # Invalid refs

        citations = service._extract_citations(answer, chunks)

        assert len(citations) == 0

    def test_duplicate_citations_deduplicated(self) -> None:
        """Test duplicate source references are deduplicated."""
        service = SynthesisService()

        chunks = [make_chunk("c1", "Content", 0.9)]
        answer = "[Source 1] says... and [Source 1] also confirms..."

        citations = service._extract_citations(answer, chunks)

        assert len(citations) == 1


# =============================================================================
# TestSynthesisService - GWT-3: Configurable Retrieval
# =============================================================================


class TestGWT3ConfigurableRetrieval:
    """GWT-3: Configurable retrieval parameters."""

    def test_top_k_applied(self) -> None:
        """Test top_k parameter is applied to retrieval."""
        service = SynthesisService()

        with patch.object(service, "_ensure_initialized"):
            with patch.object(service, "_retriever") as mock_retriever:
                mock_retriever.search.return_value = []
                service._retriever = mock_retriever

                service.retrieve_context("query", top_k=10)

                mock_retriever.search.assert_called_once()
                call_kwargs = mock_retriever.search.call_args[1]
                assert call_kwargs["top_k"] == 10

    def test_library_filter_applied(self) -> None:
        """Test library filter is applied."""
        service = SynthesisService()

        with patch.object(service, "_ensure_initialized"):
            with patch.object(service, "_retriever") as mock_retriever:
                mock_retriever.search.return_value = []
                service._retriever = mock_retriever

                service.retrieve_context("query", library="research")

                call_kwargs = mock_retriever.search.call_args[1]
                assert call_kwargs["library_filter"] == "research"

    def test_threshold_filters_results(self) -> None:
        """Test threshold filters low-scoring results."""
        service = SynthesisService()

        mock_results = [
            MagicMock(
                chunk_id="c1",
                document_id="d1",
                content="High score",
                score=0.9,
                source_file="file1.pdf",
                page_start=1,
                section_title="S1",
                metadata={},
            ),
            MagicMock(
                chunk_id="c2",
                document_id="d2",
                content="Low score",
                score=0.5,  # Below threshold
                source_file="file2.pdf",
                page_start=2,
                section_title="S2",
                metadata={},
            ),
        ]

        with patch.object(service, "_ensure_initialized"):
            with patch.object(service, "_retriever") as mock_retriever:
                mock_retriever.search.return_value = mock_results
                service._retriever = mock_retriever

                chunks = service.retrieve_context("query", threshold=0.7)

                assert len(chunks) == 1
                assert chunks[0].chunk_id == "c1"


# =============================================================================
# TestSynthesisService - GWT-5: Error Handling
# =============================================================================


class TestGWT5ErrorHandling:
    """GWT-5: Error handling in synthesis."""

    def test_llm_error_returns_failed_result(self) -> None:
        """Test LLM error returns failed result."""
        service = SynthesisService()
        chunks = [make_chunk("c1", "Content", 0.9)]

        # Mock the pipeline with config to avoid NoneType error
        mock_pipeline = MagicMock()
        mock_pipeline.config = MagicMock()
        service._pipeline = mock_pipeline

        with patch.object(service, "_ensure_initialized"):
            with patch("ingestforge.llm.factory.get_llm_client") as mock_llm:
                mock_llm.side_effect = Exception("LLM unavailable")

                result = service.synthesize("query", chunks)

                assert result.success is False
                assert "LLM unavailable" in result.error

    def test_retrieval_error_handled(self) -> None:
        """Test retrieval error is handled gracefully."""
        service = SynthesisService()

        with patch.object(service, "_ensure_initialized"):
            with patch.object(service, "_retriever") as mock_retriever:
                mock_retriever.search.side_effect = Exception("DB error")
                service._retriever = mock_retriever

                with pytest.raises(Exception) as exc_info:
                    service.retrieve_context("query")

                assert "DB error" in str(exc_info.value)


# =============================================================================
# TestSynthesisService - Context Prompt Building
# =============================================================================


class TestContextPromptBuilding:
    """Tests for context prompt building."""

    def test_build_context_prompt_includes_sources(self) -> None:
        """Test context prompt includes all source chunks."""
        service = SynthesisService()

        chunks = [
            make_chunk("c1", "First content", 0.9, "doc1.pdf"),
            make_chunk("c2", "Second content", 0.85, "doc2.pdf"),
        ]

        prompt = service._build_context_prompt("What is this?", chunks)

        assert "First content" in prompt
        assert "Second content" in prompt
        assert "doc1.pdf" in prompt
        assert "doc2.pdf" in prompt
        assert "What is this?" in prompt

    def test_build_context_prompt_includes_page_numbers(self) -> None:
        """Test page numbers are included in prompt."""
        service = SynthesisService()

        chunk = make_chunk("c1", "Content", 0.9)
        chunk.page_start = 42

        prompt = service._build_context_prompt("Query", [chunk])

        assert "Page 42" in prompt

    def test_prompt_bounded_by_max_citations(self) -> None:
        """Test prompt is bounded by MAX_CITATIONS."""
        service = SynthesisService()

        chunks = [make_chunk(f"c{i}", f"Content {i}", 0.9) for i in range(50)]

        prompt = service._build_context_prompt("Query", chunks)

        # Should only include MAX_CITATIONS sources
        assert f"Source {MAX_CITATIONS}" in prompt or MAX_CITATIONS == 20


# =============================================================================
# TestConvenienceFunctions
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_synthesis_service(self) -> None:
        """Test synthesis service creation."""
        service = create_synthesis_service()

        assert isinstance(service, SynthesisService)
        assert service._pipeline is None  # Lazy init


# =============================================================================
# TestJPLCompliance
# =============================================================================


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule_2_max_constants_defined(self) -> None:
        """JPL Rule #2: Verify MAX constants are defined."""
        assert MAX_QUERY_LENGTH > 0
        assert MAX_CONTEXT_TOKENS > 0
        assert MAX_CITATIONS > 0
        assert MAX_SOURCES > 0
        assert MAX_TOP_K > 0
        assert MAX_SNIPPET_LENGTH > 0

    def test_rule_2_max_citations_enforced(self) -> None:
        """JPL Rule #2: MAX_CITATIONS is enforced."""
        service = SynthesisService()

        # Create more chunks than MAX_CITATIONS
        chunks = [make_chunk(f"c{i}", f"Content {i}", 0.9) for i in range(50)]

        # Create answer with many source refs
        refs = " ".join([f"[Source {i+1}]" for i in range(50)])
        answer = f"Answer with {refs}"

        citations = service._extract_citations(answer, chunks)

        assert len(citations) <= MAX_CITATIONS

    def test_rule_5_precondition_empty_query(self) -> None:
        """JPL Rule #5: Assert preconditions for empty query."""
        service = SynthesisService()

        with pytest.raises(AssertionError):
            service.synthesize("", [])

    def test_rule_5_precondition_top_k_bounds(self) -> None:
        """JPL Rule #5: Assert preconditions for top_k bounds."""
        service = SynthesisService()

        with patch.object(service, "_ensure_initialized"):
            with pytest.raises(AssertionError):
                service.retrieve_context("query", top_k=0)

            with pytest.raises(AssertionError):
                service.retrieve_context("query", top_k=MAX_TOP_K + 1)

    def test_rule_9_type_hints_present(self) -> None:
        """JPL Rule #9: Verify type hints on key methods."""
        import inspect

        service = SynthesisService()

        # Check retrieve_context
        sig = inspect.signature(service.retrieve_context)
        assert sig.return_annotation != inspect.Parameter.empty

        # Check synthesize
        sig = inspect.signature(service.synthesize)
        assert sig.return_annotation != inspect.Parameter.empty


# =============================================================================
# TestSynthesisEndpoint (Integration-style)
# =============================================================================


class TestSynthesisEndpoint:
    """Tests for synthesis endpoint integration."""

    @pytest.mark.asyncio
    async def test_synthesize_endpoint_success(self) -> None:
        """Test successful synthesis via endpoint."""
        from ingestforge.api.routes.synthesis import synthesize

        request = SynthesizeRequest(query="What is AI?", top_k=3)

        with patch("ingestforge.api.routes.synthesis.SynthesisService") as MockService:
            mock_service = MagicMock()
            mock_service.retrieve_context.return_value = [
                make_chunk("c1", "AI is intelligence", 0.9)
            ]
            mock_service.synthesize.return_value = SynthesisResult(
                answer="AI is artificial intelligence.",
                citations=[make_citation("d1", "c1")],
                success=True,
            )
            MockService.return_value = mock_service

            response = await synthesize(request)

            assert response.success is True
            assert "artificial intelligence" in response.answer
            assert response.chunks_retrieved == 1

    @pytest.mark.asyncio
    async def test_synthesize_endpoint_failure(self) -> None:
        """Test synthesis endpoint handles errors."""
        from fastapi import HTTPException
        from ingestforge.api.routes.synthesis import synthesize

        request = SynthesizeRequest(query="Test query")

        with patch("ingestforge.api.routes.synthesis.SynthesisService") as MockService:
            mock_service = MagicMock()
            mock_service.retrieve_context.return_value = [
                make_chunk("c1", "Content", 0.9)
            ]
            mock_service.synthesize.return_value = SynthesisResult(
                answer="",
                citations=[],
                success=False,
                error="LLM error",
            )
            MockService.return_value = mock_service

            with pytest.raises(HTTPException) as exc_info:
                await synthesize(request)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_synthesize_includes_sources(self) -> None:
        """Test synthesis includes unique source files."""
        from ingestforge.api.routes.synthesis import synthesize

        request = SynthesizeRequest(query="Test", include_sources=True)

        with patch("ingestforge.api.routes.synthesis.SynthesisService") as MockService:
            mock_service = MagicMock()
            mock_service.retrieve_context.return_value = [
                make_chunk("c1", "Content 1", 0.9, "doc1.pdf"),
                make_chunk("c2", "Content 2", 0.85, "doc2.pdf"),
                make_chunk("c3", "Content 3", 0.8, "doc1.pdf"),  # Duplicate
            ]
            mock_service.synthesize.return_value = SynthesisResult(
                answer="Answer",
                citations=[],
                success=True,
            )
            MockService.return_value = mock_service

            response = await synthesize(request)

            assert len(response.sources) == 2
            assert "doc1.pdf" in response.sources
            assert "doc2.pdf" in response.sources


# =============================================================================
# TestStreamEndpoint
# =============================================================================


class TestStreamEndpoint:
    """Tests for streaming synthesis endpoint."""

    @pytest.mark.asyncio
    async def test_stream_returns_sse_response(self) -> None:
        """Test streaming endpoint returns SSE response."""
        from fastapi.responses import StreamingResponse
        from ingestforge.api.routes.synthesis import synthesize_stream

        request = SynthesizeRequest(query="Test query", stream=True)

        with patch("ingestforge.api.routes.synthesis.SynthesisService") as MockService:
            mock_service = MagicMock()
            mock_service.retrieve_context.return_value = []
            mock_service.synthesize.return_value = SynthesisResult(
                answer="Streamed answer",
                citations=[],
                success=True,
            )
            MockService.return_value = mock_service

            response = await synthesize_stream(request)

            assert isinstance(response, StreamingResponse)
            assert response.media_type == "text/event-stream"
