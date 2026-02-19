"""Tests for Citation Enforcer.

Citation-Enforcement Prompting implementation tests.
Tests prompt generation, citation validation, and contradiction detection."""

from __future__ import annotations

from typing import List

import pytest

from ingestforge.query.citation_enforcer import (
    Citation,
    CitationEnforcer,
    ValidationResult,
    build_citation_prompt,
    create_citation_enforcer,
    validate_response_citations,
    MAX_SOURCES,
    DEFAULT_MAX_SOURCES,
    MAX_PROMPT_LENGTH,
    MAX_CITATIONS_PER_RESPONSE,
)
from ingestforge.query.context_aggregator import ContextChunk, ContextWindow


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


def make_chunk(
    artifact_id: str = "art-001",
    document_id: str = "doc-001",
    content: str = "Test content.",
    chunk_index: int = 0,
    relevance_score: float = 1.0,
    page_number: int = None,
) -> ContextChunk:
    """Create a test ContextChunk."""
    return ContextChunk(
        content=content,
        artifact_id=artifact_id,
        document_id=document_id,
        chunk_index=chunk_index,
        relevance_score=relevance_score,
        page_number=page_number,
    )


def make_window(chunks: List[ContextChunk] = None) -> ContextWindow:
    """Create a test ContextWindow."""
    if chunks is None:
        chunks = [make_chunk()]
    return ContextWindow(
        chunks=chunks,
        total_tokens=sum(c.estimated_tokens for c in chunks),
        token_budget=8000,
        source_documents=sorted(set(c.document_id for c in chunks)),
    )


# ---------------------------------------------------------------------------
# Citation Tests
# ---------------------------------------------------------------------------


class TestCitation:
    """Tests for Citation dataclass."""

    def test_citation_creation(self) -> None:
        """Test creating a citation."""
        citation = Citation(
            artifact_id="art-001",
            page_number=42,
            text_before="The pressure is",
            position=100,
        )
        assert citation.artifact_id == "art-001"
        assert citation.page_number == 42
        assert citation.text_before == "The pressure is"
        assert citation.position == 100

    def test_citation_no_page(self) -> None:
        """Test citation without page number."""
        citation = Citation(artifact_id="art-001")
        assert citation.artifact_id == "art-001"
        assert citation.page_number is None

    def test_citation_to_dict(self) -> None:
        """Test citation dictionary conversion."""
        citation = Citation(
            artifact_id="art-001",
            page_number=10,
        )
        d = citation.to_dict()
        assert d["artifact_id"] == "art-001"
        assert d["page_number"] == 10


# ---------------------------------------------------------------------------
# ValidationResult Tests
# ---------------------------------------------------------------------------


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self) -> None:
        """Test creating a valid result."""
        result = ValidationResult(
            valid=True,
            citations_found=[Citation(artifact_id="art-001")],
            coverage=0.8,
        )
        assert result.valid is True
        assert result.citation_count == 1
        assert result.coverage == 0.8

    def test_invalid_result(self) -> None:
        """Test creating an invalid result."""
        result = ValidationResult(
            valid=False,
            invalid_citations=["bad-id"],
        )
        assert result.valid is False
        assert result.invalid_count == 1

    def test_result_to_dict(self) -> None:
        """Test result dictionary conversion."""
        result = ValidationResult(
            valid=True,
            citations_found=[Citation(artifact_id="art-001")],
            coverage=0.75,
            contradiction_count=1,
        )
        d = result.to_dict()
        assert d["valid"] is True
        assert d["citation_count"] == 1
        assert d["coverage"] == 0.75
        assert d["contradiction_count"] == 1


# ---------------------------------------------------------------------------
# CitationEnforcer Init Tests
# ---------------------------------------------------------------------------


class TestCitationEnforcerInit:
    """Tests for CitationEnforcer initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        enforcer = CitationEnforcer()
        assert enforcer._max_sources == DEFAULT_MAX_SOURCES
        assert enforcer._include_contradictions is True
        assert enforcer._include_hallucination is True

    def test_custom_max_sources(self) -> None:
        """Test custom max sources."""
        enforcer = CitationEnforcer(max_sources=25)
        assert enforcer._max_sources == 25

    def test_disable_contradiction_detection(self) -> None:
        """Test disabling contradiction detection."""
        enforcer = CitationEnforcer(include_contradiction_detection=False)
        assert enforcer._include_contradictions is False

    def test_disable_hallucination_prevention(self) -> None:
        """Test disabling hallucination prevention."""
        enforcer = CitationEnforcer(include_hallucination_prevention=False)
        assert enforcer._include_hallucination is False

    def test_invalid_max_sources_zero(self) -> None:
        """Test rejection of zero max_sources."""
        with pytest.raises(AssertionError):
            CitationEnforcer(max_sources=0)

    def test_invalid_max_sources_too_high(self) -> None:
        """Test rejection of too-high max_sources."""
        with pytest.raises(AssertionError):
            CitationEnforcer(max_sources=MAX_SOURCES + 1)


# ---------------------------------------------------------------------------
# GWT-1: Citation Prompt Generation Tests
# ---------------------------------------------------------------------------


class TestGWT1CitationPromptGeneration:
    """GWT-1: Citation prompt generation tests."""

    def test_prompt_includes_citation_instructions(self) -> None:
        """GWT-1: Prompt includes citation format instructions."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "What is the pressure?")

        assert "Citation Requirements" in prompt
        assert "[SOURCE_ID]" in prompt

    def test_prompt_includes_source_list(self) -> None:
        """GWT-1: Prompt includes source reference list."""
        chunks = [
            make_chunk(artifact_id="manual-001", document_id="doc-A"),
            make_chunk(artifact_id="spec-002", document_id="doc-B"),
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "manual-001" in prompt
        assert "spec-002" in prompt
        assert "Available Sources" in prompt

    def test_prompt_includes_context(self) -> None:
        """GWT-1: Prompt includes context text."""
        chunks = [make_chunk(content="The pressure limit is 50 psi.")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "The pressure limit is 50 psi" in prompt

    def test_prompt_includes_query(self) -> None:
        """GWT-1: Prompt includes the user query."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "What is the maximum pressure?")

        assert "What is the maximum pressure?" in prompt

    def test_prompt_assertion_none_context(self) -> None:
        """GWT-1: Prompt asserts context is not None."""
        enforcer = CitationEnforcer()
        with pytest.raises(AssertionError):
            enforcer.build_prompt(None, "Query")  # type: ignore

    def test_prompt_assertion_none_query(self) -> None:
        """GWT-1: Prompt asserts query is not None."""
        window = make_window()
        enforcer = CitationEnforcer()
        with pytest.raises(AssertionError):
            enforcer.build_prompt(window, None)  # type: ignore

    def test_prompt_assertion_empty_query(self) -> None:
        """GWT-1: Prompt asserts query is not empty."""
        window = make_window()
        enforcer = CitationEnforcer()
        with pytest.raises(AssertionError):
            enforcer.build_prompt(window, "   ")


# ---------------------------------------------------------------------------
# GWT-2: Citation Format Enforcement Tests
# ---------------------------------------------------------------------------


class TestGWT2CitationFormatEnforcement:
    """GWT-2: Citation format enforcement tests."""

    def test_format_instructions_present(self) -> None:
        """GWT-2: Citation format instructions are present."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "[artifact_id, p.PAGE]" in prompt
        assert "[artifact_id]" in prompt

    def test_format_examples_present(self) -> None:
        """GWT-2: Format examples are included."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "50 psi [doc-001, p.12]" in prompt

    def test_every_claim_must_cite(self) -> None:
        """GWT-2: Instructions require citing every claim."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "Every factual statement MUST have a citation" in prompt


# ---------------------------------------------------------------------------
# GWT-3: Contradiction Detection Tests
# ---------------------------------------------------------------------------


class TestGWT3ContradictionDetection:
    """GWT-3: Contradiction detection tests."""

    def test_contradiction_instructions_included(self) -> None:
        """GWT-3: Contradiction instructions are included."""
        window = make_window()
        enforcer = CitationEnforcer(include_contradiction_detection=True)
        prompt = enforcer.build_prompt(window, "Query")

        assert "Contradiction Detection" in prompt
        assert "**Contradiction**:" in prompt

    def test_contradiction_format_example(self) -> None:
        """GWT-3: Contradiction format example is provided."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "manual-a" in prompt or "Source A" in prompt

    def test_contradiction_instructions_can_disable(self) -> None:
        """GWT-3: Contradiction instructions can be disabled."""
        window = make_window()
        enforcer = CitationEnforcer(include_contradiction_detection=False)
        prompt = enforcer.build_prompt(window, "Query")

        assert "Contradiction Detection" not in prompt

    def test_contradiction_count_in_validation(self) -> None:
        """GWT-3: Validation counts contradictions in response."""
        window = make_window()
        enforcer = CitationEnforcer()
        response = """
        The pressure is 50 psi [art-001].
        **Contradiction**: [manual-a] says 50 psi while [manual-b] says 60 psi.
        """
        result = enforcer.validate_citations(response, window)

        assert result.contradiction_count == 1

    def test_multiple_contradictions_counted(self) -> None:
        """GWT-3: Multiple contradictions are counted."""
        window = make_window()
        enforcer = CitationEnforcer()
        response = """
        **Contradiction**: Source A vs Source B.
        **Contradiction**: Source C vs Source D.
        """
        result = enforcer.validate_citations(response, window)

        assert result.contradiction_count == 2


# ---------------------------------------------------------------------------
# GWT-4: Hallucination Prevention Tests
# ---------------------------------------------------------------------------


class TestGWT4HallucinationPrevention:
    """GWT-4: Hallucination prevention tests."""

    def test_hallucination_prevention_included(self) -> None:
        """GWT-4: Hallucination prevention instructions included."""
        window = make_window()
        enforcer = CitationEnforcer(include_hallucination_prevention=True)
        prompt = enforcer.build_prompt(window, "Query")

        assert "Strict Context Adherence" in prompt
        assert "ONLY use information from the provided context" in prompt

    def test_no_external_knowledge_instruction(self) -> None:
        """GWT-4: Instructs not to use external knowledge."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "Do NOT use any external knowledge" in prompt

    def test_hallucination_prevention_can_disable(self) -> None:
        """GWT-4: Hallucination prevention can be disabled."""
        window = make_window()
        enforcer = CitationEnforcer(include_hallucination_prevention=False)
        prompt = enforcer.build_prompt(window, "Query")

        assert "Strict Context Adherence" not in prompt

    def test_uncertainty_instruction(self) -> None:
        """GWT-4: Instructs to indicate uncertainty."""
        window = make_window()
        enforcer = CitationEnforcer()
        prompt = enforcer.build_prompt(window, "Query")

        assert "uncertain" in prompt.lower()


# ---------------------------------------------------------------------------
# GWT-5: Citation Validation Tests
# ---------------------------------------------------------------------------


class TestGWT5CitationValidation:
    """GWT-5: Citation validation tests."""

    def test_validate_valid_citations(self) -> None:
        """GWT-5: Valid citations are recognized."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "The pressure is 50 psi [art-001]."
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 1
        assert result.invalid_count == 0

    def test_validate_invalid_citations(self) -> None:
        """GWT-5: Invalid citations are flagged."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "The pressure is 50 psi [unknown-id]."
        result = enforcer.validate_citations(response, window)

        assert "unknown-id" in result.invalid_citations
        assert result.valid is False

    def test_validate_with_page_numbers(self) -> None:
        """GWT-5: Citations with page numbers are parsed."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "The pressure is 50 psi [art-001, p.12]."
        result = enforcer.validate_citations(response, window)

        assert result.citations_found[0].page_number == 12

    def test_validate_multiple_citations(self) -> None:
        """GWT-5: Multiple citations are validated."""
        chunks = [
            make_chunk(artifact_id="art-001"),
            make_chunk(artifact_id="art-002"),
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "Fact A [art-001]. Fact B [art-002]. Fact C [art-001, p.5]."
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 3
        assert result.invalid_count == 0

    def test_validate_mixed_citations(self) -> None:
        """GWT-5: Mix of valid and invalid citations."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "Valid [art-001]. Invalid [bad-id]."
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 1  # Only valid counted
        assert result.invalid_count == 1
        assert result.valid is False

    def test_validate_no_citations(self) -> None:
        """GWT-5: Response with no citations."""
        window = make_window()
        enforcer = CitationEnforcer()

        response = "This response has no citations at all."
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 0
        assert result.valid is False

    def test_validate_coverage_calculation(self) -> None:
        """GWT-5: Coverage is calculated."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "Fact one [art-001]. Fact two [art-001]. Fact three [art-001]."
        result = enforcer.validate_citations(response, window)

        assert result.coverage > 0


# ---------------------------------------------------------------------------
# Source Reference List Tests
# ---------------------------------------------------------------------------


class TestSourceReferenceList:
    """Tests for source reference list generation."""

    def test_build_source_list(self) -> None:
        """Test building source reference list."""
        chunks = [
            make_chunk(artifact_id="manual-001", document_id="doc-A", page_number=5),
            make_chunk(artifact_id="spec-002", document_id="doc-B"),
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        source_list = enforcer.build_source_reference_list(window)

        assert "manual-001" in source_list
        assert "spec-002" in source_list
        assert "doc-A" in source_list
        assert "Page: 5" in source_list

    def test_source_list_empty_window(self) -> None:
        """Test source list with empty window."""
        window = ContextWindow()
        enforcer = CitationEnforcer()

        source_list = enforcer.build_source_reference_list(window)

        assert "No sources available" in source_list

    def test_source_list_deduplicates(self) -> None:
        """Test source list deduplicates artifacts."""
        chunks = [
            make_chunk(artifact_id="art-001", chunk_index=0),
            make_chunk(artifact_id="art-001", chunk_index=1),  # Same artifact
            make_chunk(artifact_id="art-002", chunk_index=0),
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        source_list = enforcer.build_source_reference_list(window)

        # Should only appear once
        assert source_list.count("art-001") == 1


# ---------------------------------------------------------------------------
# Contradiction Check Tests
# ---------------------------------------------------------------------------


class TestContradictionCheck:
    """Tests for contradiction checking."""

    def test_check_contradictions_multiple_docs(self) -> None:
        """Test checking for contradictions with multiple documents."""
        chunks = [
            make_chunk(artifact_id="a1", document_id="doc-A"),
            make_chunk(artifact_id="a2", document_id="doc-B"),
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        conflicts = enforcer.check_contradictions(window)

        assert len(conflicts) >= 1

    def test_check_contradictions_single_doc(self) -> None:
        """Test no contradictions flagged for single document."""
        chunks = [
            make_chunk(artifact_id="a1", document_id="doc-A", chunk_index=0),
            make_chunk(artifact_id="a2", document_id="doc-A", chunk_index=1),
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        conflicts = enforcer.check_contradictions(window)

        assert len(conflicts) == 0

    def test_check_contradictions_empty(self) -> None:
        """Test contradiction check with empty window."""
        window = ContextWindow()
        enforcer = CitationEnforcer()

        conflicts = enforcer.check_contradictions(window)

        assert len(conflicts) == 0


# ---------------------------------------------------------------------------
# Convenience Function Tests
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_build_citation_prompt_function(self) -> None:
        """Test build_citation_prompt convenience function."""
        window = make_window()
        prompt = build_citation_prompt(window, "Test query")

        assert "Test query" in prompt
        assert "Citation Requirements" in prompt

    def test_validate_response_citations_function(self) -> None:
        """Test validate_response_citations convenience function."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)

        result = validate_response_citations("Test [art-001].", window)

        assert result.citation_count == 1

    def test_create_citation_enforcer_function(self) -> None:
        """Test create_citation_enforcer factory function."""
        enforcer = create_citation_enforcer(
            max_sources=30,
            include_contradictions=False,
        )

        assert enforcer._max_sources == 30
        assert enforcer._include_contradictions is False


# ---------------------------------------------------------------------------
# JPL Compliance Tests
# ---------------------------------------------------------------------------


class TestJPLCompliance:
    """Tests for JPL Power of Ten compliance."""

    def test_rule2_max_sources_constant(self) -> None:
        """JPL Rule #2: MAX_SOURCES is bounded."""
        assert MAX_SOURCES == 100
        assert MAX_SOURCES > 0

    def test_rule2_max_prompt_length_constant(self) -> None:
        """JPL Rule #2: MAX_PROMPT_LENGTH is bounded."""
        assert MAX_PROMPT_LENGTH == 100000
        assert MAX_PROMPT_LENGTH > 0

    def test_rule2_max_citations_constant(self) -> None:
        """JPL Rule #2: MAX_CITATIONS_PER_RESPONSE is bounded."""
        assert MAX_CITATIONS_PER_RESPONSE == 500
        assert MAX_CITATIONS_PER_RESPONSE > 0

    def test_rule5_build_prompt_assertions(self) -> None:
        """JPL Rule #5: build_prompt asserts preconditions."""
        enforcer = CitationEnforcer()

        with pytest.raises(AssertionError):
            enforcer.build_prompt(None, "Query")  # type: ignore

    def test_rule5_validate_assertions(self) -> None:
        """JPL Rule #5: validate_citations asserts preconditions."""
        enforcer = CitationEnforcer()

        with pytest.raises(AssertionError):
            enforcer.validate_citations(None, make_window())  # type: ignore

    def test_rule9_citation_type_hints(self) -> None:
        """JPL Rule #9: Citation has complete type hints."""
        from dataclasses import fields

        citation_fields = {f.name for f in fields(Citation)}
        required = {"artifact_id", "page_number", "text_before", "position"}
        assert required.issubset(citation_fields)

    def test_rule9_validation_result_type_hints(self) -> None:
        """JPL Rule #9: ValidationResult has complete type hints."""
        from dataclasses import fields

        result_fields = {f.name for f in fields(ValidationResult)}
        required = {"valid", "citations_found", "invalid_citations", "coverage"}
        assert required.issubset(result_fields)


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases."""

    def test_very_long_prompt_truncated(self) -> None:
        """Test that very long prompts are truncated."""
        # Create chunks with lots of content
        chunks = [
            make_chunk(
                artifact_id=f"art-{i}",
                content="X" * 5000,
            )
            for i in range(30)
        ]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        prompt = enforcer.build_prompt(window, "Query")

        assert len(prompt) <= MAX_PROMPT_LENGTH

    def test_unicode_in_citations(self) -> None:
        """Test unicode characters in content."""
        chunks = [make_chunk(content="Unicode: cafe")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        prompt = enforcer.build_prompt(window, "Test query")

        assert "cafe" in prompt

    def test_special_characters_in_artifact_id(self) -> None:
        """Test special characters in artifact IDs."""
        chunks = [make_chunk(artifact_id="doc-v1.2_final")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "Fact [doc-v1.2_final]."
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 1

    def test_citation_at_start_of_response(self) -> None:
        """Test citation at the very start."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "[art-001] states this fact."
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 1

    def test_citation_at_end_of_response(self) -> None:
        """Test citation at the very end."""
        chunks = [make_chunk(artifact_id="art-001")]
        window = make_window(chunks)
        enforcer = CitationEnforcer()

        response = "This is a fact [art-001]"
        result = enforcer.validate_citations(response, window)

        assert result.citation_count == 1

    def test_empty_response_validation(self) -> None:
        """Test validating empty response."""
        window = make_window()
        enforcer = CitationEnforcer()

        result = enforcer.validate_citations("", window)

        assert result.citation_count == 0
        assert result.valid is False

    def test_whitespace_only_response(self) -> None:
        """Test validating whitespace-only response."""
        window = make_window()
        enforcer = CitationEnforcer()

        result = enforcer.validate_citations("   \n\t   ", window)

        assert result.citation_count == 0
