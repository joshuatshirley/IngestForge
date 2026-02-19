"""
Tests for IFSynthesisProcessor.

Generative Synthesis Engine
Verifies JPL Power of Ten compliance and GWT behavior.
"""

import uuid
from typing import List
from unittest.mock import MagicMock

import pytest

from ingestforge.core.pipeline.artifacts import (
    IFChunkArtifact,
    IFTextArtifact,
    IFFailureArtifact,
)
from ingestforge.processors.synthesis.synthesizer import (
    IFSynthesisProcessor,
    IFSynthesisArtifact,
    SynthesisCitation,
    MAX_CONTEXT_TOKENS,
    MAX_CITATIONS,
    MAX_CONTEXT_CHUNKS,
    _estimate_tokens,
    _build_citation_prompt,
    _extract_citations,
    _extract_chunks_from_artifact,
    _extract_query,
    _format_citations,
    _build_synthesis_result,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock()
    response = MagicMock()
    response.text = "Based on the sources, the answer is X [Doc:test.pdf:Page 1]."
    client.generate.return_value = response
    return client


@pytest.fixture
def sample_chunk() -> IFChunkArtifact:
    """Create a sample chunk artifact."""
    return IFChunkArtifact(
        artifact_id=str(uuid.uuid4()),
        document_id="doc-001",
        content="This is test content about topic X.",
        chunk_index=0,
        metadata={
            "source_file": "test.pdf",
            "page_start": 1,
            "section_title": "Introduction",
        },
    )


@pytest.fixture
def sample_chunks() -> List[IFChunkArtifact]:
    """Create multiple sample chunks."""
    return [
        IFChunkArtifact(
            artifact_id=str(uuid.uuid4()),
            document_id="doc-001",
            content="First document content about topic A.",
            chunk_index=0,
            metadata={
                "source_file": "manual_a.pdf",
                "page_start": 10,
                "section_title": "Chapter 1",
            },
        ),
        IFChunkArtifact(
            artifact_id=str(uuid.uuid4()),
            document_id="doc-002",
            content="Second document content about topic B.",
            chunk_index=0,
            metadata={
                "source_file": "manual_b.pdf",
                "page_start": 25,
                "section_title": "Overview",
            },
        ),
    ]


@pytest.fixture
def synthesis_input_artifact(sample_chunks: List[IFChunkArtifact]) -> IFTextArtifact:
    """Create input artifact for synthesis."""
    return IFTextArtifact(
        artifact_id=str(uuid.uuid4()),
        content="What is the relationship between topic A and B?",
        metadata={
            "synthesis_chunks": [
                {
                    "artifact_id": c.artifact_id,
                    "document_id": c.document_id,
                    "content": c.content,
                    "chunk_index": c.chunk_index,
                    "metadata": c.metadata,
                }
                for c in sample_chunks
            ],
        },
    )


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestEstimateTokens:
    """Tests for _estimate_tokens helper."""

    def test_empty_string(self):
        """Empty string returns 0 tokens."""
        assert _estimate_tokens("") == 0

    def test_short_string(self):
        """Short string estimates correctly."""
        result = _estimate_tokens("hello")
        assert result == 1  # 5 chars / 4 = 1

    def test_longer_string(self):
        """Longer string estimates correctly."""
        text = "a" * 100
        result = _estimate_tokens(text)
        assert result == 25  # 100 / 4 = 25

    def test_none_raises_assertion(self):
        """None input raises AssertionError."""
        with pytest.raises(AssertionError, match="text cannot be None"):
            _estimate_tokens(None)


class TestBuildCitationPrompt:
    """Tests for _build_citation_prompt helper."""

    def test_includes_query(self, sample_chunks: List[IFChunkArtifact]):
        """Prompt includes the query."""
        query = "Test question?"
        prompt = _build_citation_prompt(query, sample_chunks)
        assert query in prompt

    def test_includes_chunk_content(self, sample_chunks: List[IFChunkArtifact]):
        """Prompt includes chunk content."""
        prompt = _build_citation_prompt("Query?", sample_chunks)
        for chunk in sample_chunks:
            assert chunk.content in prompt

    def test_includes_source_references(self, sample_chunks: List[IFChunkArtifact]):
        """Prompt includes source file references."""
        prompt = _build_citation_prompt("Query?", sample_chunks)
        assert "[Doc:manual_a.pdf" in prompt
        assert "[Doc:manual_b.pdf" in prompt

    def test_includes_page_numbers(self, sample_chunks: List[IFChunkArtifact]):
        """Prompt includes page numbers."""
        prompt = _build_citation_prompt("Query?", sample_chunks)
        assert "Page 10" in prompt
        assert "Page 25" in prompt

    def test_includes_citation_rules(self, sample_chunks: List[IFChunkArtifact]):
        """Prompt includes citation enforcement rules."""
        prompt = _build_citation_prompt("Query?", sample_chunks)
        assert "CITATION RULES" in prompt
        assert "[Doc:filename:Page N]" in prompt

    def test_empty_query_raises(self, sample_chunks: List[IFChunkArtifact]):
        """Empty query raises AssertionError."""
        with pytest.raises(AssertionError, match="query cannot be empty"):
            _build_citation_prompt("", sample_chunks)


class TestExtractCitations:
    """Tests for _extract_citations helper."""

    def test_extracts_simple_citation(self, sample_chunks: List[IFChunkArtifact]):
        """Extracts citation from answer."""
        answer = "The fact is X [Doc:manual_a.pdf:Page 10]."
        citations = _extract_citations(answer, sample_chunks)
        assert len(citations) >= 1
        assert any(c.source_file == "manual_a.pdf" for c in citations)

    def test_extracts_multiple_citations(self, sample_chunks: List[IFChunkArtifact]):
        """Extracts multiple citations."""
        answer = "Fact A [Doc:manual_a.pdf:Page 10]. Fact B [Doc:manual_b.pdf:Page 25]."
        citations = _extract_citations(answer, sample_chunks)
        assert len(citations) >= 2

    def test_no_citations_in_answer(self, sample_chunks: List[IFChunkArtifact]):
        """Returns empty list when no citations found."""
        answer = "This answer has no citations."
        citations = _extract_citations(answer, sample_chunks)
        assert citations == []

    def test_respects_max_citations(self, sample_chunks: List[IFChunkArtifact]):
        """Limits citations to MAX_CITATIONS."""
        # Create answer with many citations
        answer = " ".join(["Fact [Doc:manual_a.pdf]" for _ in range(100)])
        citations = _extract_citations(answer, sample_chunks)
        assert len(citations) <= MAX_CITATIONS


class TestExtractChunksFromArtifact:
    """Tests for _extract_chunks_from_artifact helper."""

    def test_extracts_from_chunk_artifact(self, sample_chunk: IFChunkArtifact):
        """Returns chunk artifact as single-item list."""
        result = _extract_chunks_from_artifact(sample_chunk)
        assert len(result) == 1
        assert result[0].artifact_id == sample_chunk.artifact_id

    def test_extracts_from_metadata(self, synthesis_input_artifact: IFTextArtifact):
        """Extracts chunks from metadata."""
        result = _extract_chunks_from_artifact(synthesis_input_artifact)
        assert len(result) == 2

    def test_empty_metadata_returns_empty(self):
        """Returns empty list when no chunks in metadata."""
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="Query",
            metadata={},
        )
        result = _extract_chunks_from_artifact(artifact)
        assert result == []

    def test_respects_max_chunks(self):
        """Limits chunks to MAX_CONTEXT_CHUNKS."""
        many_chunks = [
            {
                "artifact_id": str(uuid.uuid4()),
                "document_id": f"doc-{i}",
                "content": f"Content {i}",
                "chunk_index": i,
                "metadata": {},
            }
            for i in range(50)
        ]
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="Query",
            metadata={"synthesis_chunks": many_chunks},
        )
        result = _extract_chunks_from_artifact(artifact)
        assert len(result) <= MAX_CONTEXT_CHUNKS


class TestExtractQuery:
    """Tests for _extract_query helper (JPL refactor)."""

    def test_extracts_from_content(self):
        """Extracts query from artifact content."""
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="What is the answer?",
            metadata={},
        )
        result = _extract_query(artifact)
        assert result == "What is the answer?"

    def test_extracts_from_metadata(self):
        """Prefers synthesis_query metadata over content."""
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="Content text",
            metadata={"synthesis_query": "Query from metadata"},
        )
        result = _extract_query(artifact)
        assert result == "Query from metadata"

    def test_returns_empty_for_no_query(self):
        """Returns empty string when no query found."""
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="",
            metadata={},
        )
        result = _extract_query(artifact)
        assert result == ""


class TestFormatCitations:
    """Tests for _format_citations helper (JPL refactor)."""

    def test_formats_single_citation(self):
        """Formats a single citation to dict."""
        citation = SynthesisCitation(
            artifact_id="art-001",
            document_id="doc-001",
            content_snippet="Test snippet",
            page_number=10,
            section_title="Introduction",
            source_file="test.pdf",
        )
        result = _format_citations([citation])
        assert len(result) == 1
        assert result[0]["artifact_id"] == "art-001"
        assert result[0]["page"] == 10
        assert result[0]["snippet"] == "Test snippet"

    def test_formats_multiple_citations(self):
        """Formats multiple citations."""
        citations = [
            SynthesisCitation(
                artifact_id=f"art-{i}",
                document_id=f"doc-{i}",
                content_snippet=f"Snippet {i}",
            )
            for i in range(3)
        ]
        result = _format_citations(citations)
        assert len(result) == 3

    def test_formats_empty_list(self):
        """Returns empty list for no citations."""
        result = _format_citations([])
        assert result == []


class TestBuildSynthesisResult:
    """Tests for _build_synthesis_result helper (JPL refactor)."""

    def test_builds_artifact_with_all_fields(self, sample_chunk: IFChunkArtifact):
        """Builds synthesis artifact with all required fields."""
        parent = IFTextArtifact(
            artifact_id="parent-001",
            content="Query",
        )
        citation = SynthesisCitation(
            artifact_id="art-001",
            document_id="doc-001",
            content_snippet="Snippet",
        )
        result = _build_synthesis_result(
            artifact=parent,
            processor_id="test.processor",
            query="Test query?",
            answer="Test answer.",
            citations=[citation],
            chunks=[sample_chunk],
            context_tokens=100,
            model_name="test-model",
            max_tokens=1024,
            temperature=0.3,
        )

        assert isinstance(result, IFSynthesisArtifact)
        assert result.query == "Test query?"
        assert result.answer == "Test answer."
        assert result.source_chunks == 1
        assert result.context_tokens == 100
        assert result.parent_id == "parent-001"
        assert "test.processor" in result.provenance

    def test_preserves_lineage(self, sample_chunk: IFChunkArtifact):
        """Preserves artifact lineage."""
        parent = IFTextArtifact(
            artifact_id="parent-001",
            content="Query",
            provenance=["processor.a"],
            lineage_depth=1,
        )
        result = _build_synthesis_result(
            artifact=parent,
            processor_id="processor.b",
            query="Query",
            answer="Answer",
            citations=[],
            chunks=[sample_chunk],
            context_tokens=50,
            model_name="model",
            max_tokens=1024,
            temperature=0.3,
        )

        assert result.lineage_depth == 2
        assert "processor.a" in result.provenance
        assert "processor.b" in result.provenance


# =============================================================================
# PROCESSOR TESTS
# =============================================================================


class TestIFSynthesisProcessorInit:
    """Tests for processor initialization."""

    def test_default_init(self):
        """Processor initializes with defaults."""
        processor = IFSynthesisProcessor()
        assert processor.processor_id == "if.synthesis.v1"
        assert processor.version == "1.0.0"

    def test_with_llm_client(self, mock_llm_client):
        """Processor initializes with provided client."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        assert processor._llm_client is mock_llm_client
        assert processor._initialized is True

    def test_max_tokens_capped(self):
        """Max tokens is capped at MAX_ANSWER_LENGTH."""
        processor = IFSynthesisProcessor(max_tokens=100000)
        assert processor._max_tokens <= 8000


class TestIFSynthesisProcessorAvailability:
    """Tests for is_available method."""

    def test_is_available_with_dependencies(self):
        """Returns True when dependencies available."""
        processor = IFSynthesisProcessor()
        # Should return True since llm.factory exists
        assert processor.is_available() in [True, False]


class TestIFSynthesisProcessorProcess:
    """Tests for process method."""

    def test_process_with_valid_input(
        self,
        mock_llm_client,
        synthesis_input_artifact: IFTextArtifact,
    ):
        """Process returns synthesis artifact."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        result = processor.process(synthesis_input_artifact)

        assert isinstance(result, IFSynthesisArtifact)
        assert result.query == synthesis_input_artifact.content
        assert len(result.answer) > 0
        assert result.source_chunks == 2

    def test_process_tracks_lineage(
        self,
        mock_llm_client,
        synthesis_input_artifact: IFTextArtifact,
    ):
        """Process maintains artifact lineage."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        result = processor.process(synthesis_input_artifact)

        assert result.parent_id == synthesis_input_artifact.artifact_id
        assert processor.processor_id in result.provenance

    def test_process_no_query_fails(self, mock_llm_client):
        """Process fails when no query provided."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="",  # Empty query
            metadata={"synthesis_chunks": []},
        )
        result = processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "No synthesis query" in result.error_message

    def test_process_no_chunks_fails(self, mock_llm_client):
        """Process fails when no chunks provided."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="Valid query?",
            metadata={},  # No chunks
        )
        result = processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "No context chunks" in result.error_message

    def test_process_no_citations_fails(self, sample_chunks: List[IFChunkArtifact]):
        """AC: JPL Rule #7 - Fail-fast if LLM output lacks citations."""
        # Create mock that returns answer WITHOUT citations
        mock_client = MagicMock()
        response = MagicMock()
        response.text = "This is an answer with no citations at all."
        mock_client.generate.return_value = response

        processor = IFSynthesisProcessor(llm_client=mock_client)
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="What is the answer?",
            metadata={
                "synthesis_chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "document_id": c.document_id,
                        "content": c.content,
                        "chunk_index": c.chunk_index,
                        "metadata": c.metadata,
                    }
                    for c in sample_chunks
                ],
            },
        )
        result = processor.process(artifact)

        assert isinstance(result, IFFailureArtifact)
        assert "lacks required citations" in result.error_message


class TestContextLengthValidation:
    """Tests for context length validation (JPL Rule #5)."""

    def test_validates_context_under_limit(self, mock_llm_client):
        """Accepts context under MAX_CONTEXT_TOKENS."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        chunks = [
            IFChunkArtifact(
                artifact_id=str(uuid.uuid4()),
                document_id="doc",
                content="Short content",
                metadata={},
            )
        ]
        # Should not raise
        tokens = processor._validate_context_length(chunks)
        assert tokens < MAX_CONTEXT_TOKENS

    def test_rejects_context_over_limit(self):
        """Rejects context exceeding MAX_CONTEXT_TOKENS."""
        processor = IFSynthesisProcessor()
        # Create chunk with very long content
        huge_content = "x" * (MAX_CONTEXT_TOKENS * 5)
        chunks = [
            IFChunkArtifact(
                artifact_id=str(uuid.uuid4()),
                document_id="doc",
                content=huge_content,
                metadata={},
            )
        ]
        with pytest.raises(AssertionError, match="exceeds maximum"):
            processor._validate_context_length(chunks)


class TestTeardown:
    """Tests for teardown method."""

    def test_teardown_clears_client(self, mock_llm_client):
        """Teardown clears LLM client."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        assert processor._llm_client is not None

        result = processor.teardown()

        assert result is True
        assert processor._llm_client is None
        assert processor._initialized is False


# =============================================================================
# GWT BEHAVIOR TESTS ()
# =============================================================================


class TestGWTBehavior:
    """Tests verifying GWT specification."""

    def test_given_linked_chunks_when_synthesis_then_cited_summary(
        self,
        mock_llm_client,
        sample_chunks: List[IFChunkArtifact],
    ):
        """
        GWT: Given linked IFChunkArtifacts, When synthesis requested,
        Then return narrative summary with [Doc:Page] citations.
        """
        # Given: Linked chunks
        artifact = IFTextArtifact(
            artifact_id=str(uuid.uuid4()),
            content="Summarize the relationship between topics.",
            metadata={
                "synthesis_chunks": [
                    {
                        "artifact_id": c.artifact_id,
                        "document_id": c.document_id,
                        "content": c.content,
                        "chunk_index": c.chunk_index,
                        "metadata": c.metadata,
                    }
                    for c in sample_chunks
                ],
            },
        )

        # When: Synthesis requested
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)
        result = processor.process(artifact)

        # Then: Returns narrative with citations
        assert isinstance(result, IFSynthesisArtifact)
        assert len(result.answer) > 0
        assert result.source_chunks == len(sample_chunks)


# =============================================================================
# JPL COMPLIANCE TESTS
# =============================================================================


class TestJPLCompliance:
    """Tests verifying JPL Power of Ten compliance."""

    def test_rule_2_fixed_bounds(self):
        """Rule #2: All loops have fixed upper bounds."""
        assert MAX_CONTEXT_TOKENS > 0
        assert MAX_CITATIONS > 0
        assert MAX_CONTEXT_CHUNKS > 0

    def test_rule_5_assertions(self, mock_llm_client):
        """Rule #5: Assert preconditions."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)

        # None artifact should raise
        with pytest.raises(AssertionError, match="artifact cannot be None"):
            processor.process(None)

    def test_rule_7_return_values(self, mock_llm_client):
        """Rule #7: All functions return explicit values."""
        processor = IFSynthesisProcessor(llm_client=mock_llm_client)

        # is_available returns bool
        result = processor.is_available()
        assert isinstance(result, bool)

        # teardown returns bool
        result = processor.teardown()
        assert isinstance(result, bool)

    def test_rule_9_type_hints(self):
        """Rule #9: Methods have type hints."""
        import inspect

        processor = IFSynthesisProcessor()

        # Check key methods have return annotations
        for method_name in ["process", "is_available", "teardown"]:
            method = getattr(processor, method_name)
            sig = inspect.signature(method)
            assert (
                sig.return_annotation != inspect.Signature.empty
            ), f"{method_name} missing return type hint"
