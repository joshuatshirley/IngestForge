"""Comprehensive GWT unit tests for the Intelligence Brief Generator.

Intelligence Briefs.
Verifies G-RAG synthesis, citation mapping, and JPL Rule #7 validation.
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestforge.agent.brief_models import IntelligenceBrief, EvidenceLink, KeyEntity
from ingestforge.agent.brief_generator import IFBriefGenerator, BriefCitationValidator

# =============================================================================
# MODELS TESTS
# =============================================================================


def test_brief_markdown_rendering():
    """GIVEN a structured IntelligenceBrief
    WHEN to_markdown is called
    THEN it returns a properly formatted Markdown document with all sections.
    """
    brief = IntelligenceBrief(
        mission_id="M-123",
        title="Test Brief",
        summary="This is a summary.",
        key_entities=[
            KeyEntity(name="Einstein", type="Person", description="Physicist")
        ],
        evidence=[
            EvidenceLink(
                doc_id="doc1", chunk_id="c1", snippet="Gravity", confidence=0.9
            )
        ],
    )

    md = brief.to_markdown()

    assert "# INTELLIGENCE BRIEF: Test Brief" in md
    assert "## EXECUTIVE SUMMARY" in md
    assert "- **Einstein** (Person): Physicist" in md
    assert '[doc1:N/A] "Gravity..."' in md


# =============================================================================
# GENERATOR TESTS
# =============================================================================


@pytest.mark.asyncio
async def test_brief_generation_flow():
    """GIVEN a research mission query
    WHEN generate_brief is called
    THEN it retrieves context, synthesizes sections, and returns a complete brief.
    """
    with patch("ingestforge.agent.brief_generator.load_config"), patch(
        "ingestforge.agent.brief_generator.Pipeline"
    ), patch(
        "ingestforge.agent.brief_generator.HybridRetriever"
    ) as MockRetriever, patch(
        "ingestforge.agent.brief_generator.get_llm_client"
    ) as MockLLM:
        # Setup mock retrieval
        mock_chunk = MagicMock()
        mock_chunk.content = "Relevant text about physics."
        mock_chunk.chunk_id = "c1"
        mock_chunk.document_id = "d1"
        mock_chunk.source_file = "physics.pdf"
        mock_chunk.page_start = 5
        mock_chunk.score = 0.95
        mock_chunk.metadata = {"entities": ["Quantum"]}

        MockRetriever.return_value.search.return_value = [mock_chunk]

        # Setup mock LLM
        MockLLM.return_value.generate.return_value = MagicMock(
            text="Synthesized summary."
        )

        generator = IFBriefGenerator()
        brief = await generator.generate_brief("M-123", "Tell me about physics")

        assert brief.summary == "Synthesized summary."
        assert len(brief.evidence) == 1
        assert brief.evidence[0].doc_id == "physics.pdf"
        assert brief.evidence[0].offset == 5
        assert brief.key_entities[0].name == "Quantum"


@pytest.mark.asyncio
async def test_empty_brief_fallback():
    """GIVEN a query with no relevant sources
    WHEN generate_brief is called
    THEN it returns a helpful 'empty' brief instead of failing.
    """
    with patch("ingestforge.agent.brief_generator.load_config"), patch(
        "ingestforge.agent.brief_generator.Pipeline"
    ), patch("ingestforge.agent.brief_generator.HybridRetriever") as MockRetriever:
        MockRetriever.return_value.search.return_value = []

        generator = IFBriefGenerator()
        brief = await generator.generate_brief("M-123", "Unknown query")

        assert "No relevant information found" in brief.summary
        assert len(brief.evidence) == 0


# =============================================================================
# VALIDATOR TESTS (Rule #7)
# =============================================================================


def test_citation_validator_success():
    """GIVEN a brief with valid citations
    WHEN validate is called
    THEN it returns True (JPL Rule #7 compliance).
    """
    mock_storage = MagicMock()
    mock_storage.get_chunk.return_value = {"id": "c1"}  # Found

    brief = IntelligenceBrief(
        mission_id="M",
        title="T",
        summary="S",
        evidence=[EvidenceLink(doc_id="d1", chunk_id="c1", snippet="X")],
    )

    validator = BriefCitationValidator(mock_storage)
    assert validator.validate(brief) is True


def test_citation_validator_failure():
    """GIVEN a brief with a broken citation
    WHEN validate is called
    THEN it returns False and logs an error.
    """
    mock_storage = MagicMock()
    mock_storage.get_chunk.return_value = None  # Not found

    brief = IntelligenceBrief(
        mission_id="M",
        title="T",
        summary="S",
        evidence=[EvidenceLink(doc_id="d1", chunk_id="broken_id", snippet="X")],
    )

    validator = BriefCitationValidator(mock_storage)
    assert validator.validate(brief) is False
