"""Comprehensive GWT unit tests for the Legal Vertical.

Legal Pleading Template.
Verifies model integrity, synthesis logic, and evidence aggregation.
"""

from unittest.mock import MagicMock, patch
from ingestforge.verticals.legal.models import (
    LegalPleadingModel,
    PleadingParty,
    LegalFact,
)
from ingestforge.verticals.legal.generator import LegalPleadingGenerator
from ingestforge.verticals.legal.validator import LegalCitationValidator
from ingestforge.verticals.legal.aggregator import LegalFactAggregator

# =============================================================================
# MODELS TESTS
# =============================================================================


def test_legal_caption_generation():
    """GIVEN a LegalPleadingModel with parties
    WHEN get_caption is called
    THEN it returns a properly formatted court header.
    """
    model = LegalPleadingModel(
        court_name="Superior Court of California",
        jurisdiction="County of Los Angeles",
        plaintiffs=[PleadingParty(name="John Doe", role="Plaintiff")],
        defendants=[PleadingParty(name="Acme Corp", role="Defendant")],
        title="Complaint for Damages",
    )

    caption = model.get_caption()

    assert "Superior Court of California" in caption
    assert "John Doe" in caption
    assert "Acme Corp" in caption
    assert "V." in caption or "v." in caption
    assert "CASE NO: PENDING" in caption


# =============================================================================
# GENERATOR TESTS
# =============================================================================


def test_pleading_markdown_generation():
    """GIVEN a model with facts
    WHEN generate_markdown is called
    THEN it includes sections for facts and arguments with citations.
    """
    model = LegalPleadingModel(
        court_name="Court",
        jurisdiction="State",
        plaintiffs=[PleadingParty(name="P", role="P")],
        defendants=[PleadingParty(name="D", role="D")],
        title="Title",
        statement_of_facts=[
            LegalFact(text="The sky is blue.", source_id="doc_1", page_number=5)
        ],
    )

    with patch("ingestforge.verticals.legal.generator.load_config"), patch(
        "ingestforge.llm.factory.get_llm_client"
    ):
        generator = LegalPleadingGenerator()
        md = generator.generate_markdown(model)

        assert "## II. STATEMENT OF FACTS" in md
        assert "1. The sky is blue. [Doc: doc_1, p. 5]" in md


# =============================================================================
# AGGREGATOR TESTS
# =============================================================================


def test_fact_aggregation_filtering():
    """GIVEN search results from the knowledge base
    WHEN aggregate_evidence is called
    THEN it filters results by confidence and maps them to LegalFacts.
    """
    with patch("ingestforge.verticals.legal.aggregator.load_config"), patch(
        "ingestforge.verticals.legal.aggregator.Pipeline"
    ), patch("ingestforge.verticals.legal.aggregator.HybridRetriever") as MockRetriever:
        # Setup mock results
        mock_res = MagicMock()
        mock_res.content = "Evidence"
        mock_res.source_file = "file.pdf"
        mock_res.page_start = 10
        mock_res.score = 0.9

        low_score_res = MagicMock()
        low_score_res.score = 0.2

        MockRetriever.return_value.search.return_value = [mock_res, low_score_res]

        aggregator = LegalFactAggregator()
        facts = aggregator.aggregate_evidence("test query")

        # Should only have the high score one
        assert len(facts) == 1
        assert facts[0].text == "Evidence"
        assert facts[0].source_id == "file.pdf"


# =============================================================================
# VALIDATOR TESTS
# =============================================================================


def test_citation_validation():
    """GIVEN text with legal citations
    WHEN validate_citations is called
    THEN it correctly identifies Bluebook-style patterns.
    """
    validator = LegalCitationValidator()

    valid_text = "As seen in 410 U.S. 113, the court held..."
    invalid_text = "No citations here."

    ok, matches = validator.validate_citations(valid_text)
    assert ok is True
    assert "410 U.S. 113" in matches

    fail, _ = validator.validate_citations(invalid_text)
    assert fail is False
