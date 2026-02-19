"""Tests for multi-agent paper summarizer.

Tests 3-panel summary generation with mock LLM responses."""

from __future__ import annotations

import pytest

from ingestforge.agent.paper_summarizer import (
    AgentRole,
    AgentOutput,
    PaperSummary,
    SummarizationPrompts,
    PaperSummarizer,
    create_paper_summarizer,
    summarize_paper,
    MAX_DOCUMENT_LENGTH,
    MAX_SUMMARY_LENGTH,
    MAX_FINDINGS,
    MAX_LIMITATIONS,
)

# Import mock LLM from test fixtures
from tests.fixtures.agents import MockLLM

# AgentRole tests


class TestAgentRole:
    """Tests for AgentRole enum."""

    def test_roles_defined(self) -> None:
        """Test all roles are defined."""
        roles = [r.value for r in AgentRole]

        assert "abstract" in roles
        assert "methodology" in roles
        assert "critique" in roles

    def test_role_count(self) -> None:
        """Test correct number of roles."""
        assert len(AgentRole) == 3


# PaperSummary tests


class TestPaperSummary:
    """Tests for PaperSummary dataclass."""

    def test_summary_creation(self) -> None:
        """Test creating a paper summary."""
        summary = PaperSummary(
            title="Test Paper",
            abstract_summary="Main thesis is X",
            methodology_summary="Used method Y",
            critique_summary="Limitation Z",
        )

        assert summary.title == "Test Paper"
        assert summary.abstract_summary == "Main thesis is X"

    def test_summary_truncation(self) -> None:
        """Test long summaries are truncated."""
        long_text = "x" * 3000
        summary = PaperSummary(
            title="Test",
            abstract_summary=long_text,
            methodology_summary=long_text,
            critique_summary=long_text,
        )

        assert len(summary.abstract_summary) == MAX_SUMMARY_LENGTH
        assert len(summary.methodology_summary) == MAX_SUMMARY_LENGTH

    def test_findings_truncation(self) -> None:
        """Test key findings list is truncated."""
        many_findings = [f"Finding {i}" for i in range(20)]
        summary = PaperSummary(
            title="Test",
            abstract_summary="Test",
            methodology_summary="Test",
            critique_summary="Test",
            key_findings=many_findings,
        )

        assert len(summary.key_findings) == MAX_FINDINGS

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        summary = PaperSummary(
            title="Test Paper",
            abstract_summary="Abstract",
            methodology_summary="Methods",
            critique_summary="Critique",
            key_findings=["Finding 1", "Finding 2"],
            limitations=["Limitation 1"],
        )

        d = summary.to_dict()

        assert d["title"] == "Test Paper"
        assert d["abstract_summary"] == "Abstract"
        assert len(d["key_findings"]) == 2
        assert len(d["limitations"]) == 1

    def test_to_markdown(self) -> None:
        """Test converting to markdown."""
        summary = PaperSummary(
            title="Research Paper",
            abstract_summary="The main thesis.",
            methodology_summary="The methodology.",
            critique_summary="The critique.",
            key_findings=["Finding A", "Finding B"],
            limitations=["Limitation A"],
        )

        md = summary.to_markdown()

        assert "# Research Paper" in md
        assert "## Abstract & Thesis" in md
        assert "## Methodology" in md
        assert "## Critical Analysis" in md
        assert "- Finding A" in md
        assert "- Limitation A" in md


# AgentOutput tests


class TestAgentOutput:
    """Tests for AgentOutput dataclass."""

    def test_output_creation(self) -> None:
        """Test creating agent output."""
        output = AgentOutput(
            role=AgentRole.ABSTRACT,
            summary="Main thesis summary",
            key_points=["Point 1", "Point 2"],
            confidence=0.9,
        )

        assert output.role == AgentRole.ABSTRACT
        assert output.summary == "Main thesis summary"
        assert output.confidence == 0.9

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        output = AgentOutput(
            role=AgentRole.METHODOLOGY,
            summary="Methods used",
            key_points=["Method A"],
        )

        d = output.to_dict()

        assert d["role"] == "methodology"
        assert d["summary"] == "Methods used"


# SummarizationPrompts tests


class TestSummarizationPrompts:
    """Tests for SummarizationPrompts class."""

    def test_abstract_prompt(self) -> None:
        """Test abstract prompt generation."""
        prompt = SummarizationPrompts.abstract_prompt(
            document="Paper content here",
            title="Test Paper",
        )

        assert "Test Paper" in prompt
        assert "Paper content here" in prompt
        assert "thesis" in prompt.lower()
        assert "contribution" in prompt.lower()

    def test_methodology_prompt(self) -> None:
        """Test methodology prompt generation."""
        prompt = SummarizationPrompts.methodology_prompt(
            document="Paper content",
            title="Test Paper",
        )

        assert "methodology" in prompt.lower()
        assert "research design" in prompt.lower()

    def test_critique_prompt(self) -> None:
        """Test critique prompt generation."""
        prompt = SummarizationPrompts.critique_prompt(
            document="Paper content",
            title="Test Paper",
        )

        assert "critical" in prompt.lower()
        assert "limitations" in prompt.lower()


# PaperSummarizer tests


class TestPaperSummarizer:
    """Tests for PaperSummarizer class."""

    def test_summarizer_creation(self) -> None:
        """Test creating a summarizer."""
        llm = MockLLM()
        summarizer = PaperSummarizer(llm_client=llm)

        assert summarizer is not None

    def test_summarizer_requires_llm(self) -> None:
        """Test summarizer requires LLM client."""
        with pytest.raises(ValueError, match="llm_client cannot be None"):
            PaperSummarizer(llm_client=None)  # type: ignore

    def test_summarize_paper(self) -> None:
        """Test summarizing a paper with mock responses."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Summary: This paper presents a novel approach to X.
Key Points:
- Key contribution A
- Key contribution B
Confidence: 0.9""",
                """Summary: The methodology involves Y analysis.
Key Points:
- Method step 1
- Method step 2
Confidence: 0.85""",
                """Summary: Limitations include Z.
Key Points:
- Limitation A
- Limitation B
Confidence: 0.8""",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(
            document="Test paper content about research.",
            title="Test Research Paper",
        )

        assert summary.title == "Test Research Paper"
        assert "novel approach" in summary.abstract_summary
        assert "methodology" in summary.methodology_summary.lower()
        assert "limitations" in summary.critique_summary.lower()
        assert llm.call_count == 3

    def test_summarize_empty_document(self) -> None:
        """Test summarizing empty document."""
        llm = MockLLM()
        summarizer = PaperSummarizer(llm_client=llm)

        summary = summarizer.summarize(document="", title="Empty Paper")

        assert summary.title == "Empty Paper"
        assert "No content" in summary.abstract_summary
        assert llm.call_count == 0

    def test_summarize_whitespace_document(self) -> None:
        """Test summarizing whitespace-only document."""
        llm = MockLLM()
        summarizer = PaperSummarizer(llm_client=llm)

        summary = summarizer.summarize(document="   \n\t  ", title="Blank Paper")

        assert summary.title == "Blank Paper"
        assert "No content" in summary.abstract_summary

    def test_three_agents_called(self) -> None:
        """Test that all three agents are called."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: Abstract\nKey Points:\n- A\nConfidence: 0.9",
                "Summary: Methods\nKey Points:\n- B\nConfidence: 0.9",
                "Summary: Critique\nKey Points:\n- C\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summarizer.summarize(
            document="Paper content",
            title="Test",
        )

        # Verify all 3 agents were called
        assert llm.call_count == 3

        # Check prompts contain expected keywords
        prompts = llm.call_history
        assert any("thesis" in p.lower() for p in prompts)
        assert any("methodology" in p.lower() for p in prompts)
        assert any("critical" in p.lower() for p in prompts)

    def test_key_findings_combined(self) -> None:
        """Test key findings are combined from abstract and methodology."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Summary: Abstract summary.
Key Points:
- Abstract point 1
- Abstract point 2
Confidence: 0.9""",
                """Summary: Methods summary.
Key Points:
- Method point 1
Confidence: 0.85""",
                """Summary: Critique summary.
Key Points:
- Limitation 1
Confidence: 0.8""",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(document="Content", title="Test")

        # Key findings from abstract + methodology
        assert len(summary.key_findings) >= 2

    def test_limitations_from_critique(self) -> None:
        """Test limitations come from critique agent."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: A\nKey Points:\n- Point\nConfidence: 0.9",
                "Summary: B\nKey Points:\n- Point\nConfidence: 0.9",
                """Summary: Critical analysis here.
Key Points:
- Major limitation X
- Minor limitation Y
Confidence: 0.8""",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(document="Content", title="Test")

        assert "Major limitation X" in summary.limitations
        assert "Minor limitation Y" in summary.limitations

    def test_source_chunks_included(self) -> None:
        """Test source chunks are included in summary."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: A\nConfidence: 0.9",
                "Summary: B\nConfidence: 0.9",
                "Summary: C\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(
            document="Content",
            title="Test",
            source_chunks=["chunk_001", "chunk_002"],
        )

        assert "chunk_001" in summary.source_chunks
        assert "chunk_002" in summary.source_chunks


class TestSummarizerParsing:
    """Tests for response parsing in summarizer."""

    def test_parse_standard_response(self) -> None:
        """Test parsing standard formatted response."""
        llm = MockLLM()
        llm.set_responses(
            [
                """Summary: The main thesis of this paper is about machine learning.
Key Points:
- Uses neural networks
- Novel architecture
- State of the art results
Confidence: 0.92""",
                "Summary: Methods\nConfidence: 0.9",
                "Summary: Critique\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(document="Content", title="Test")

        assert "machine learning" in summary.abstract_summary
        assert "neural networks" in summary.key_findings[0]

    def test_parse_missing_confidence(self) -> None:
        """Test parsing response without confidence."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: No confidence provided here.",
                "Summary: Methods",
                "Summary: Critique",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(document="Content", title="Test")

        # Should still work with default confidence
        assert summary.abstract_summary is not None

    def test_parse_empty_response(self) -> None:
        """Test handling empty LLM response."""
        llm = MockLLM()
        llm.set_responses(["", "Summary: B", "Summary: C"])

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(document="Content", title="Test")

        assert "No response" in summary.abstract_summary

    def test_parse_malformed_response(self) -> None:
        """Test handling malformed response."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Random text without proper formatting",
                "Summary: B\nConfidence: 0.9",
                "Summary: C\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(document="Content", title="Test")

        # Should fallback to using first paragraph
        assert "Random text" in summary.abstract_summary


class TestSummarizerEdgeCases:
    """Tests for edge cases in summarizer."""

    def test_very_long_document(self) -> None:
        """Test handling very long document."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: Handled long doc\nConfidence: 0.9",
                "Summary: Methods\nConfidence: 0.9",
                "Summary: Critique\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        long_doc = "x" * 200000  # Exceeds MAX_DOCUMENT_LENGTH
        summary = summarizer.summarize(document=long_doc, title="Long Paper")

        assert summary.title == "Long Paper"
        # Should succeed by truncating
        assert "Handled long doc" in summary.abstract_summary

    def test_special_characters_in_title(self) -> None:
        """Test handling special characters in title."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: A\nConfidence: 0.9",
                "Summary: B\nConfidence: 0.9",
                "Summary: C\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(
            document="Content",
            title='Paper "Title" with <special> & chars',
        )

        assert "special" in summary.title

    def test_unicode_content(self) -> None:
        """Test handling unicode content."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: Unicode handled\nConfidence: 0.9",
                "Summary: Methods\nConfidence: 0.9",
                "Summary: Critique\nConfidence: 0.9",
            ]
        )

        summarizer = PaperSummarizer(llm_client=llm)
        summary = summarizer.summarize(
            document="Content with symbols",
            title="Test",
        )

        assert summary.abstract_summary is not None


# Factory function tests


class TestCreatePaperSummarizer:
    """Tests for create_paper_summarizer factory."""

    def test_create(self) -> None:
        """Test creating summarizer via factory."""
        llm = MockLLM()
        summarizer = create_paper_summarizer(llm_client=llm)

        assert isinstance(summarizer, PaperSummarizer)

    def test_create_with_config(self) -> None:
        """Test creating with custom config."""
        from ingestforge.llm.base import GenerationConfig

        llm = MockLLM()
        config = GenerationConfig(max_tokens=2000, temperature=0.5)
        summarizer = create_paper_summarizer(llm_client=llm, config=config)

        assert summarizer._config.max_tokens == 2000
        assert summarizer._config.temperature == 0.5


class TestSummarizePaperFunction:
    """Tests for summarize_paper convenience function."""

    def test_summarize_paper(self) -> None:
        """Test convenience function."""
        llm = MockLLM()
        llm.set_responses(
            [
                "Summary: Abstract\nConfidence: 0.9",
                "Summary: Methods\nConfidence: 0.9",
                "Summary: Critique\nConfidence: 0.9",
            ]
        )

        summary = summarize_paper(
            document="Test content",
            llm_client=llm,
            title="Quick Test",
        )

        assert summary.title == "Quick Test"
        assert "Abstract" in summary.abstract_summary


# Constant tests


class TestConstants:
    """Tests for module constants."""

    def test_max_document_length(self) -> None:
        """Test MAX_DOCUMENT_LENGTH is reasonable."""
        assert MAX_DOCUMENT_LENGTH > 0
        assert MAX_DOCUMENT_LENGTH == 100000

    def test_max_summary_length(self) -> None:
        """Test MAX_SUMMARY_LENGTH is reasonable."""
        assert MAX_SUMMARY_LENGTH > 0
        assert MAX_SUMMARY_LENGTH == 2000

    def test_max_findings(self) -> None:
        """Test MAX_FINDINGS is reasonable."""
        assert MAX_FINDINGS > 0
        assert MAX_FINDINGS == 10

    def test_max_limitations(self) -> None:
        """Test MAX_LIMITATIONS is reasonable."""
        assert MAX_LIMITATIONS > 0
        assert MAX_LIMITATIONS == 10
