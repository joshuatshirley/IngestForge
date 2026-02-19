"""Tests for Bluebook Citation Parser.

Tests legal citation extraction and parsing following the Bluebook
citation format used in US legal documents.

Coverage:
- Federal reporter recognition (U.S., F.2d, F.3d, F. Supp.)
- State reporter recognition (Cal., N.Y., Tex., etc.)
- Pin cite extraction
- Year and court extraction
- Edge cases (malformed citations, multiple citations)
"""

import pytest

from ingestforge.enrichment.bluebook_parser import (
    BluebookParser,
    LegalCitation,
    extract_citations,
    parse_citation,
    enrich_with_citations,
    FEDERAL_REPORTERS,
    STATE_REPORTERS,
    COURT_ABBREVIATIONS,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def parser() -> BluebookParser:
    """Create parser instance."""
    return BluebookParser()


# =============================================================================
# Basic Extraction Tests
# =============================================================================


class TestBasicExtraction:
    """Tests for basic citation extraction functionality."""

    def test_extract_empty_text(self, parser: BluebookParser):
        """Test extraction from empty text."""
        citations = parser.extract_citations("")
        assert citations == []

    def test_extract_none_text(self, parser: BluebookParser):
        """Test extraction from None text."""
        citations = parser.extract_citations(None)  # type: ignore
        assert citations == []

    def test_extract_whitespace_only(self, parser: BluebookParser):
        """Test extraction from whitespace-only text."""
        citations = parser.extract_citations("   \n\t  ")
        assert citations == []

    def test_extract_no_citations(self, parser: BluebookParser):
        """Test extraction from text without citations."""
        text = "This is a paragraph about law without any case citations."
        citations = parser.extract_citations(text)
        assert citations == []


# =============================================================================
# Federal Reporter Tests
# =============================================================================


class TestFederalReporters:
    """Tests for federal reporter recognition."""

    def test_us_reports_citation(self, parser: BluebookParser):
        """Test U.S. Reports citation."""
        text = "See 347 U.S. 483 for the holding."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].volume == 347
        assert citations[0].reporter == "U.S."
        assert citations[0].page == 483

    def test_us_reports_with_year(self, parser: BluebookParser):
        """Test U.S. Reports with year in parenthetical."""
        text = "The case was decided in 347 U.S. 483 (1954)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].year == 1954

    def test_federal_reporter_second_series(self, parser: BluebookParser):
        """Test F.2d citation."""
        text = "In 456 F.2d 123, the court held..."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].volume == 456
        assert citations[0].reporter == "F.2d"
        assert citations[0].page == 123

    def test_federal_reporter_third_series(self, parser: BluebookParser):
        """Test F.3d citation."""
        text = "See 789 F.3d 456 (9th Cir. 2015)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].volume == 789
        assert citations[0].reporter == "F.3d"
        assert citations[0].page == 456
        assert citations[0].court == "9th Cir."
        assert citations[0].year == 2015

    def test_federal_reporter_fourth_series(self, parser: BluebookParser):
        """Test F.4th citation."""
        text = "The recent case at 12 F.4th 345 (2d Cir. 2021)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "F.4th"

    def test_federal_supplement(self, parser: BluebookParser):
        """Test F. Supp. citation."""
        text = "See 456 F. Supp. 789 (S.D.N.Y. 2010)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "F. Supp."
        assert citations[0].court == "S.D.N.Y."

    def test_federal_supplement_second(self, parser: BluebookParser):
        """Test F. Supp. 2d citation."""
        text = "In 789 F. Supp. 2d 123 (N.D. Cal. 2011)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "F. Supp. 2d"

    def test_federal_supplement_third(self, parser: BluebookParser):
        """Test F. Supp. 3d citation."""
        text = "See 321 F. Supp. 3d 456 (D. Mass. 2018)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "F. Supp. 3d"

    def test_supreme_court_reporter(self, parser: BluebookParser):
        """Test S. Ct. citation."""
        text = "See 138 S. Ct. 1719 (2018)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "S. Ct."

    def test_lawyers_edition(self, parser: BluebookParser):
        """Test L. Ed. citation."""
        text = "See 98 L. Ed. 873 (1954)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "L. Ed."


# =============================================================================
# State Reporter Tests
# =============================================================================


class TestStateReporters:
    """Tests for state reporter recognition."""

    def test_california_reports_third(self, parser: BluebookParser):
        """Test Cal. 3d citation."""
        text = "See 123 Cal. 3d 456 (1982)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "Cal. 3d"

    def test_california_reports_fourth(self, parser: BluebookParser):
        """Test Cal. 4th citation."""
        text = "In 45 Cal. 4th 789 (2009)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "Cal. 4th"

    def test_california_appellate_reports(self, parser: BluebookParser):
        """Test Cal. App. citation."""
        text = "See 67 Cal. App. 4th 890 (1998)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "Cal. App. 4th"

    def test_california_reporter(self, parser: BluebookParser):
        """Test Cal. Rptr. citation."""
        text = "See 234 Cal. Rptr. 567 (1991)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "Cal. Rptr."

    def test_new_york_reports(self, parser: BluebookParser):
        """Test N.Y. citation."""
        text = "See 89 N.Y.2d 123 (1996)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "N.Y.2d"

    def test_new_york_third(self, parser: BluebookParser):
        """Test N.Y.3d citation."""
        text = "In 12 N.Y.3d 456 (2009)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "N.Y.3d"

    def test_new_york_supplement(self, parser: BluebookParser):
        """Test N.Y.S. citation."""
        text = "See 456 N.Y.S.2d 789 (1982)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "N.Y.S.2d"

    def test_appellate_division(self, parser: BluebookParser):
        """Test A.D. citation."""
        text = "See 78 A.D.3d 901 (2010)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "A.D.3d"

    def test_texas_southwestern(self, parser: BluebookParser):
        """Test S.W.3d citation."""
        text = "See 345 S.W.3d 678 (Tex. 2011)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "S.W.3d"

    def test_pacific_reporter(self, parser: BluebookParser):
        """Test P.3d citation."""
        text = "See 123 P.3d 456 (Cal. 2006)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "P.3d"

    def test_atlantic_reporter(self, parser: BluebookParser):
        """Test A.3d citation."""
        text = "See 78 A.3d 901 (Pa. 2013)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "A.3d"

    def test_northeastern_reporter(self, parser: BluebookParser):
        """Test N.E.2d citation."""
        text = "See 456 N.E.2d 789 (Ill. 1983)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "N.E.2d"

    def test_southeastern_reporter(self, parser: BluebookParser):
        """Test S.E.2d citation."""
        text = "See 234 S.E.2d 567 (Ga. 1976)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "S.E.2d"

    def test_southern_reporter(self, parser: BluebookParser):
        """Test So. 3d citation."""
        text = "See 123 So. 3d 456 (Fla. 2012)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].reporter == "So. 3d"


# =============================================================================
# Full Citation Tests (with Case Names)
# =============================================================================


class TestFullCitations:
    """Tests for full citations with case names."""

    def test_supreme_court_full_citation(self, parser: BluebookParser):
        """Test Brown v. Board of Education citation."""
        text = "Brown v. Board of Education, 347 U.S. 483 (1954)"
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].case_name == "Brown v. Board of Education"
        assert citations[0].volume == 347
        assert citations[0].reporter == "U.S."
        assert citations[0].page == 483
        assert citations[0].year == 1954

    def test_circuit_court_full_citation(self, parser: BluebookParser):
        """Test circuit court citation with court identifier."""
        text = "Smith v. Jones, 123 F.3d 456 (9th Cir. 1999)"
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].case_name == "Smith v. Jones"
        assert citations[0].court == "9th Cir."
        assert citations[0].year == 1999

    def test_district_court_full_citation(self, parser: BluebookParser):
        """Test district court citation."""
        text = "Doe v. Roe, 456 F. Supp. 2d 789 (S.D.N.Y. 2006)"
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].case_name == "Doe v. Roe"
        assert citations[0].court == "S.D.N.Y."

    def test_case_name_with_hyphen(self, parser: BluebookParser):
        """Test case name with hyphenated party."""
        text = "Smith-Johnson v. State, 123 F.3d 456 (5th Cir. 2000)"
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert "Smith-Johnson" in citations[0].case_name

    def test_case_name_with_initials(self, parser: BluebookParser):
        """Test case name with initials."""
        text = "J.M. v. School Board, 789 F.3d 123 (11th Cir. 2015)"
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert "J.M." in citations[0].case_name


# =============================================================================
# Pin Cite Tests
# =============================================================================


class TestPinCites:
    """Tests for pin cite extraction."""

    def test_pin_cite_at_format(self, parser: BluebookParser):
        """Test 'at' format pin cite."""
        text = "See 347 U.S. at 490."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].pin_cite == 490

    def test_pin_cite_comma_format(self, parser: BluebookParser):
        """Test comma format pin cite in full citation."""
        text = "See 347 U.S. 483, 490 (1954)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].page == 483
        assert citations[0].pin_cite == 490

    def test_pin_cite_with_case_name(self, parser: BluebookParser):
        """Test pin cite in full citation with case name."""
        text = "Brown v. Board of Education, 347 U.S. 483, 495 (1954)"
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].pin_cite == 495


# =============================================================================
# Court Extraction Tests
# =============================================================================


class TestCourtExtraction:
    """Tests for court identifier extraction."""

    def test_ninth_circuit(self, parser: BluebookParser):
        """Test 9th Circuit extraction."""
        text = "See 123 F.3d 456 (9th Cir. 1999)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].court == "9th Cir."

    def test_second_circuit(self, parser: BluebookParser):
        """Test 2d Circuit extraction."""
        text = "See 456 F.3d 789 (2d Cir. 2006)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].court == "2d Cir."

    def test_dc_circuit(self, parser: BluebookParser):
        """Test D.C. Circuit extraction."""
        text = "See 789 F.3d 123 (D.C. Cir. 2015)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].court == "D.C. Cir."

    def test_federal_circuit(self, parser: BluebookParser):
        """Test Federal Circuit extraction."""
        text = "See 234 F.3d 567 (Fed. Cir. 2000)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].court == "Fed. Cir."

    def test_sdny_district(self, parser: BluebookParser):
        """Test S.D.N.Y. extraction."""
        text = "See 456 F. Supp. 2d 789 (S.D.N.Y. 2006)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].court == "S.D.N.Y."

    def test_nd_cal_district(self, parser: BluebookParser):
        """Test N.D. Cal. extraction."""
        text = "See 123 F. Supp. 3d 456 (N.D. Cal. 2015)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].court == "N.D. Cal."


# =============================================================================
# Year Extraction Tests
# =============================================================================


class TestYearExtraction:
    """Tests for year extraction."""

    def test_year_extraction_1950s(self, parser: BluebookParser):
        """Test year extraction from 1950s."""
        text = "See 347 U.S. 483 (1954)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].year == 1954

    def test_year_extraction_2000s(self, parser: BluebookParser):
        """Test year extraction from 2000s."""
        text = "See 456 F.3d 789 (9th Cir. 2006)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].year == 2006

    def test_year_extraction_2020s(self, parser: BluebookParser):
        """Test year extraction from 2020s."""
        text = "See 123 F.4th 456 (2d Cir. 2022)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].year == 2022

    def test_no_year_in_citation(self, parser: BluebookParser):
        """Test citation without year."""
        text = "See 347 U.S. 483."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].year is None


# =============================================================================
# Multiple Citations Tests
# =============================================================================


class TestMultipleCitations:
    """Tests for multiple citation extraction."""

    def test_two_citations_same_paragraph(self, parser: BluebookParser):
        """Test extracting two citations from same paragraph."""
        text = "See 347 U.S. 483 (1954) and 384 U.S. 436 (1966)."
        citations = parser.extract_citations(text)

        assert len(citations) == 2
        volumes = [c.volume for c in citations]
        assert 347 in volumes
        assert 384 in volumes

    def test_multiple_citations_different_reporters(self, parser: BluebookParser):
        """Test citations from different reporters."""
        text = "Compare 347 U.S. 483 with 123 F.3d 456 and 456 Cal. 4th 789."
        citations = parser.extract_citations(text)

        assert len(citations) == 3
        reporters = [c.reporter for c in citations]
        assert "U.S." in reporters
        assert "F.3d" in reporters
        assert "Cal. 4th" in reporters

    def test_citations_sorted_by_position(self, parser: BluebookParser):
        """Test citations are sorted by position in text."""
        text = "First 100 U.S. 200, then 300 F.3d 400, finally 500 Cal. 4th 600."
        citations = parser.extract_citations(text)

        assert len(citations) == 3
        assert citations[0].start < citations[1].start
        assert citations[1].start < citations[2].start


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and malformed citations."""

    def test_citation_with_extra_spaces(self, parser: BluebookParser):
        """Test citation with extra spaces."""
        text = "See  347  U.S.  483  (1954)."
        # May or may not match depending on pattern strictness
        citations = parser.extract_citations(text)
        # Should handle gracefully
        assert isinstance(citations, list)

    def test_very_long_text(self, parser: BluebookParser):
        """Test extraction from very long text."""
        long_text = "word " * 10000 + "347 U.S. 483 (1954)" + " word" * 10000
        citations = parser.extract_citations(long_text)

        assert len(citations) >= 1

    def test_citation_in_footnote_context(self, parser: BluebookParser):
        """Test citation that might appear in footnote."""
        text = "[1] See Brown v. Board of Education, 347 U.S. 483 (1954)."
        citations = parser.extract_citations(text)

        assert len(citations) >= 1
        assert citations[0].volume == 347

    def test_citation_with_quotation_marks(self, parser: BluebookParser):
        """Test citation near quotation marks."""
        text = 'The Court stated, "separate but equal," 347 U.S. 483, 495 (1954).'
        citations = parser.extract_citations(text)

        assert len(citations) >= 1

    def test_large_volume_number(self, parser: BluebookParser):
        """Test citation with large volume number."""
        text = "See 999 F.3d 123 (9th Cir. 2021)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].volume == 999

    def test_large_page_number(self, parser: BluebookParser):
        """Test citation with large page number."""
        text = "See 123 F.3d 9999 (9th Cir. 1997)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].page == 9999

    def test_single_digit_volume(self, parser: BluebookParser):
        """Test citation with single digit volume."""
        text = "See 1 U.S. 123 (1789)."
        citations = parser.extract_citations(text)

        assert len(citations) == 1
        assert citations[0].volume == 1


# =============================================================================
# Reporter Normalization Tests
# =============================================================================


class TestReporterNormalization:
    """Tests for reporter abbreviation normalization."""

    def test_normalize_us_reports(self, parser: BluebookParser):
        """Test U.S. normalization."""
        assert parser.normalize_reporter("U.S.") == "U.S."
        assert parser.normalize_reporter("U. S.") == "U.S."

    def test_normalize_f3d(self, parser: BluebookParser):
        """Test F.3d normalization."""
        assert parser.normalize_reporter("F.3d") == "F.3d"
        assert parser.normalize_reporter("F. 3d") == "F.3d"

    def test_normalize_fsupp(self, parser: BluebookParser):
        """Test F. Supp. normalization."""
        assert parser.normalize_reporter("F. Supp.") == "F. Supp."
        assert parser.normalize_reporter("F.Supp.") == "F. Supp."

    def test_normalize_ny(self, parser: BluebookParser):
        """Test N.Y. normalization."""
        assert parser.normalize_reporter("N.Y.2d") == "N.Y.2d"
        assert parser.normalize_reporter("N. Y. 2d") == "N.Y.2d"

    def test_normalize_unknown_reporter(self, parser: BluebookParser):
        """Test unknown reporter returns as-is."""
        assert parser.normalize_reporter("Unknown") == "Unknown"


# =============================================================================
# Bluebook Format Output Tests
# =============================================================================


class TestBluebookFormat:
    """Tests for Bluebook format output."""

    def test_format_basic_citation(self, parser: BluebookParser):
        """Test basic citation formatting."""
        citation = LegalCitation(
            raw_text="347 U.S. 483",
            volume=347,
            reporter="U.S.",
            page=483,
        )
        formatted = parser.to_bluebook_format(citation)

        assert "347" in formatted
        assert "U.S." in formatted
        assert "483" in formatted

    def test_format_with_case_name(self, parser: BluebookParser):
        """Test formatting with case name."""
        citation = LegalCitation(
            raw_text="Brown v. Board of Education, 347 U.S. 483 (1954)",
            volume=347,
            reporter="U.S.",
            page=483,
            year=1954,
            case_name="Brown v. Board of Education",
        )
        formatted = parser.to_bluebook_format(citation)

        assert "Brown v. Board of Education" in formatted
        assert "347" in formatted
        assert "1954" in formatted

    def test_format_with_court_and_year(self, parser: BluebookParser):
        """Test formatting with court and year."""
        citation = LegalCitation(
            raw_text="123 F.3d 456 (9th Cir. 1999)",
            volume=123,
            reporter="F.3d",
            page=456,
            court="9th Cir.",
            year=1999,
        )
        formatted = parser.to_bluebook_format(citation)

        assert "9th Cir." in formatted
        assert "1999" in formatted

    def test_format_with_pin_cite(self, parser: BluebookParser):
        """Test formatting with pin cite."""
        citation = LegalCitation(
            raw_text="347 U.S. 483, 490 (1954)",
            volume=347,
            reporter="U.S.",
            page=483,
            pin_cite=490,
            year=1954,
        )
        formatted = parser.to_bluebook_format(citation)

        assert "490" in formatted


# =============================================================================
# Enrichment Integration Tests
# =============================================================================


class TestEnrichment:
    """Tests for chunk enrichment functionality."""

    def test_enrich_chunk_with_citations(self, parser: BluebookParser):
        """Test enriching chunk with citations."""
        chunk = {"text": "See 347 U.S. 483 (1954) for the holding."}

        enriched = parser.enrich(chunk)

        assert "legal_citations" in enriched
        assert len(enriched["legal_citations"]) == 1
        assert enriched["citation_count"] == 1

    def test_enrich_chunk_empty_text(self, parser: BluebookParser):
        """Test enriching chunk with empty text."""
        chunk = {"text": ""}

        enriched = parser.enrich(chunk)

        assert enriched["legal_citations"] == []
        assert enriched["citation_count"] == 0

    def test_enrich_chunk_no_citations(self, parser: BluebookParser):
        """Test enriching chunk without citations."""
        chunk = {"text": "This is a paragraph without legal citations."}

        enriched = parser.enrich(chunk)

        assert enriched["legal_citations"] == []
        assert enriched["citation_count"] == 0

    def test_enrich_chunk_multiple_citations(self, parser: BluebookParser):
        """Test enriching chunk with multiple citations."""
        chunk = {"text": "Compare 347 U.S. 483 (1954) with 384 U.S. 436 (1966)."}

        enriched = parser.enrich(chunk)

        assert enriched["citation_count"] == 2
        assert "reporter_counts" in enriched
        assert enriched["reporter_counts"]["U.S."] == 2

    def test_enrich_chunk_with_content_key(self, parser: BluebookParser):
        """Test enriching chunk using 'content' key instead of 'text'."""
        chunk = {"content": "See 123 F.3d 456 (9th Cir. 1999)."}

        enriched = parser.enrich(chunk)

        assert len(enriched["legal_citations"]) == 1

    def test_enrich_preserves_original_fields(self, parser: BluebookParser):
        """Test that enrichment preserves original chunk fields."""
        chunk = {
            "id": "doc-123",
            "source": "legal_brief.pdf",
            "text": "See 347 U.S. 483.",
        }

        enriched = parser.enrich(chunk)

        assert enriched["id"] == "doc-123"
        assert enriched["source"] == "legal_brief.pdf"


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_extract_citations_function(self):
        """Test extract_citations convenience function."""
        text = "See 347 U.S. 483 (1954)."
        citations = extract_citations(text)

        assert len(citations) == 1
        assert citations[0].volume == 347

    def test_parse_citation_function(self):
        """Test parse_citation convenience function."""
        text = "347 U.S. 483 (1954)"
        citation = parse_citation(text)

        assert citation is not None
        assert citation.volume == 347
        assert citation.year == 1954

    def test_parse_citation_invalid(self):
        """Test parse_citation with invalid input."""
        citation = parse_citation("not a citation")

        assert citation is None

    def test_enrich_with_citations_function(self):
        """Test enrich_with_citations convenience function."""
        chunk = {"text": "See 123 F.3d 456."}
        enriched = enrich_with_citations(chunk)

        assert "legal_citations" in enriched


# =============================================================================
# LegalCitation Dataclass Tests
# =============================================================================


class TestLegalCitationDataclass:
    """Tests for LegalCitation dataclass."""

    def test_to_dict(self):
        """Test to_dict method."""
        citation = LegalCitation(
            raw_text="347 U.S. 483",
            volume=347,
            reporter="U.S.",
            page=483,
            year=1954,
        )
        data = citation.to_dict()

        assert data["volume"] == 347
        assert data["reporter"] == "U.S."
        assert data["page"] == 483
        assert data["year"] == 1954

    def test_hash_equality(self):
        """Test hash and equality for deduplication."""
        cite1 = LegalCitation(
            raw_text="347 U.S. 483",
            volume=347,
            reporter="U.S.",
            page=483,
        )
        cite2 = LegalCitation(
            raw_text="347 U.S. 483 (1954)",
            volume=347,
            reporter="U.S.",
            page=483,
            year=1954,
        )

        # Same volume/reporter/page should be equal
        assert cite1 == cite2
        assert hash(cite1) == hash(cite2)

    def test_hash_inequality(self):
        """Test hash inequality for different citations."""
        cite1 = LegalCitation(
            raw_text="347 U.S. 483",
            volume=347,
            reporter="U.S.",
            page=483,
        )
        cite2 = LegalCitation(
            raw_text="384 U.S. 436",
            volume=384,
            reporter="U.S.",
            page=436,
        )

        assert cite1 != cite2
        assert hash(cite1) != hash(cite2)


# =============================================================================
# Reporter Dictionary Tests
# =============================================================================


class TestReporterDictionaries:
    """Tests for reporter dictionary completeness."""

    def test_federal_reporters_not_empty(self):
        """Test federal reporters dictionary is populated."""
        assert len(FEDERAL_REPORTERS) > 0

    def test_state_reporters_not_empty(self):
        """Test state reporters dictionary is populated."""
        assert len(STATE_REPORTERS) > 0

    def test_court_abbreviations_not_empty(self):
        """Test court abbreviations dictionary is populated."""
        assert len(COURT_ABBREVIATIONS) > 0

    def test_all_circuit_courts_present(self):
        """Test all circuit courts are present."""
        circuits = [
            "1st Cir.",
            "2d Cir.",
            "3d Cir.",
            "4th Cir.",
            "5th Cir.",
            "6th Cir.",
            "7th Cir.",
            "8th Cir.",
            "9th Cir.",
            "10th Cir.",
            "11th Cir.",
            "D.C. Cir.",
            "Fed. Cir.",
        ]

        for circuit in circuits:
            assert circuit in COURT_ABBREVIATIONS.values()
