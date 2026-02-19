"""Tests for Citation Provenance Engine (RES-003).

Validates:
- APA citation extraction
- MLA citation extraction
- Chicago citation extraction
- Numbered citation extraction
- Citation style detection
- Chunk linking
- Confidence scoring
- Edge cases
"""

import pytest
from unittest.mock import Mock

from ingestforge.enrichment.citation_provenance import (
    CitationProvenanceEngine,
    CitationProvenance,
    CitationStyle,
    extract_academic_citations,
    detect_citation_style,
    link_citations_to_chunks,
    MAX_TEXT_LENGTH,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
)
from ingestforge.storage.base import SearchResult


# ============================================================================
# Test Data
# ============================================================================

APA_TEXT = """
Research shows significant findings (Smith, 2020). Further studies
(Jones & Brown, 2019) confirmed these results. According to multiple
authors (Williams et al., 2021), the effect is widespread. The specific
page reference (Johnson, 2018, pp. 42-45) provides more detail.
"""

MLA_TEXT = """
The novel demonstrates this theme clearly (Fitzgerald 42). Multiple
scholars (Miller and Taylor 128-130) have analyzed this passage
extensively. Some argue (Anderson 15) that the interpretation varies.
"""

CHICAGO_TEXT = """
Smith (2020) argues that the methodology is sound. Both Jones and Brown
(2019, 45) provide supporting evidence. Williams (2021) extends this work.
"""

NUMBERED_TEXT = """
Previous studies [1] have shown similar results. Multiple sources [2-5]
confirm these findings. Selected references [1,3,7] are particularly relevant.
"""

MIXED_TEXT = """
According to (Smith, 2020) and (Jones 42), the evidence supports this
claim. Williams (2021) and [1] provide additional context.
"""


# ============================================================================
# Mock Storage
# ============================================================================


def create_mock_storage() -> Mock:
    """Create a mock storage backend."""
    storage = Mock()

    def mock_search(query: str, top_k: int = 10, library_filter=None):
        # Return mock results based on query content
        results = []

        if "smith" in query.lower():
            results.append(
                SearchResult(
                    chunk_id="chunk_smith_1",
                    content="Smith discusses the methodology in 2020.",
                    score=0.9,
                    document_id="doc_smith",
                    section_title="Methods",
                    chunk_type="content",
                    source_file="smith_2020.pdf",
                    word_count=50,
                )
            )

        if "jones" in query.lower():
            results.append(
                SearchResult(
                    chunk_id="chunk_jones_1",
                    content="Jones and Brown published findings in 2019.",
                    score=0.85,
                    document_id="doc_jones",
                    section_title="Results",
                    chunk_type="content",
                    source_file="jones_2019.pdf",
                    word_count=60,
                )
            )

        if "williams" in query.lower():
            results.append(
                SearchResult(
                    chunk_id="chunk_williams_1",
                    content="Williams et al. (2021) extended prior work.",
                    score=0.88,
                    document_id="doc_williams",
                    section_title="Discussion",
                    chunk_type="content",
                    source_file="williams_2021.pdf",
                    word_count=45,
                )
            )

        return results

    storage.search = mock_search
    return storage


# ============================================================================
# Test Classes
# ============================================================================


class TestAPACitationExtraction:
    """Tests for APA-style citation extraction."""

    def test_extract_single_author_apa(self):
        """Test extracting single author APA citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("As noted (Smith, 2020).")

        assert len(citations) == 1
        assert citations[0].author == "Smith"
        assert citations[0].year == 2020
        assert citations[0].citation_style == CitationStyle.APA.value

    def test_extract_two_authors_apa(self):
        """Test extracting two-author APA citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Research (Jones & Brown, 2019).")

        assert len(citations) == 1
        assert citations[0].author == "Jones"
        assert citations[0].year == 2019

    def test_extract_et_al_apa(self):
        """Test extracting et al. APA citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Studies (Williams et al., 2021).")

        assert len(citations) == 1
        assert citations[0].author == "Williams"
        assert citations[0].year == 2021

    def test_extract_apa_with_page(self):
        """Test extracting APA citation with page reference."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Text (Smith, 2020, pp. 42-45).")

        assert len(citations) == 1
        assert citations[0].page_ref == "42-45"
        assert citations[0].year == 2020

    def test_extract_multiple_apa(self):
        """Test extracting multiple APA citations from text."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations(APA_TEXT)

        # Should find multiple citations
        assert len(citations) >= 3

        # Check authors are extracted
        authors = [c.author for c in citations]
        assert "Smith" in authors
        assert "Jones" in authors

    def test_apa_confidence_high(self):
        """Test APA citations have high confidence."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("(Smith, 2020)")

        assert citations[0].confidence >= HIGH_CONFIDENCE


class TestMLACitationExtraction:
    """Tests for MLA-style citation extraction."""

    def test_extract_single_author_mla(self):
        """Test extracting single author MLA citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("The text states (Fitzgerald 42).")

        assert len(citations) == 1
        assert citations[0].author == "Fitzgerald"
        assert citations[0].page_ref == "42"
        assert citations[0].year is None
        assert citations[0].citation_style == CitationStyle.MLA.value

    def test_extract_two_authors_mla(self):
        """Test extracting two-author MLA citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("As noted (Miller and Taylor 128-130).")

        assert len(citations) == 1
        assert citations[0].author == "Miller"
        assert citations[0].page_ref == "128-130"

    def test_extract_multiple_mla(self):
        """Test extracting multiple MLA citations from text."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations(MLA_TEXT)

        # Should find multiple citations
        assert len(citations) >= 2

        # Check page refs extracted
        assert any(c.page_ref is not None for c in citations)

    def test_mla_page_range(self):
        """Test MLA citation with page range."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("(Author 10-15)")

        assert len(citations) == 1
        assert citations[0].page_ref == "10-15"


class TestChicagoCitationExtraction:
    """Tests for Chicago-style citation extraction."""

    def test_extract_chicago_author_date(self):
        """Test extracting Chicago author-date citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Smith (2020) argues that...")

        assert len(citations) == 1
        assert citations[0].author == "Smith"
        assert citations[0].year == 2020
        assert citations[0].citation_style == CitationStyle.CHICAGO.value

    def test_extract_chicago_with_page(self):
        """Test extracting Chicago citation with page."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Jones (2019, 45) states that...")

        assert len(citations) == 1
        assert citations[0].year == 2019
        assert citations[0].page_ref == "45"

    def test_extract_chicago_two_authors(self):
        """Test extracting Chicago citation with two authors."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Smith and Jones (2020) found...")

        assert len(citations) == 1
        assert citations[0].author == "Smith"
        assert citations[0].year == 2020

    def test_extract_chicago_et_al(self):
        """Test extracting Chicago citation with et al."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Williams et al. (2021) demonstrates...")

        assert len(citations) == 1
        assert "Williams" in citations[0].author
        assert citations[0].year == 2021
        assert citations[0].citation_style == CitationStyle.CHICAGO.value

    def test_extract_multiple_chicago(self):
        """Test extracting multiple Chicago citations."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations(CHICAGO_TEXT)

        assert len(citations) >= 2


class TestNumberedCitationExtraction:
    """Tests for numbered citation extraction."""

    def test_extract_single_numbered(self):
        """Test extracting single numbered citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Previous work [1] showed...")

        assert len(citations) == 1
        assert citations[0].author == "ref_1"
        assert citations[0].citation_style == CitationStyle.NUMBERED.value

    def test_extract_numbered_range(self):
        """Test extracting numbered range citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Multiple studies [2-5] confirm...")

        assert len(citations) == 1
        assert citations[0].author == "ref_2-5"

    def test_extract_numbered_list(self):
        """Test extracting numbered list citation."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("Selected works [1,3,7] are relevant...")

        assert len(citations) == 1
        assert "refs_" in citations[0].author

    def test_extract_multiple_numbered(self):
        """Test extracting multiple numbered citations."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations(NUMBERED_TEXT)

        assert len(citations) >= 3


class TestCitationStyleDetection:
    """Tests for citation style detection."""

    def test_detect_apa_style(self):
        """Test detecting APA style."""
        engine = CitationProvenanceEngine()

        style = engine.detect_style("(Smith, 2020)")

        assert style == CitationStyle.APA.value

    def test_detect_mla_style(self):
        """Test detecting MLA style."""
        engine = CitationProvenanceEngine()

        style = engine.detect_style("(Smith 42)")

        assert style == CitationStyle.MLA.value

    def test_detect_chicago_style(self):
        """Test detecting Chicago style."""
        engine = CitationProvenanceEngine()

        style = engine.detect_style("Smith (2020)")

        assert style == CitationStyle.CHICAGO.value

    def test_detect_numbered_style(self):
        """Test detecting numbered style."""
        engine = CitationProvenanceEngine()

        style = engine.detect_style("[1]")

        assert style == CitationStyle.NUMBERED.value

    def test_detect_unknown_style(self):
        """Test detecting unknown style."""
        engine = CitationProvenanceEngine()

        style = engine.detect_style("random text without citation")

        assert style == CitationStyle.UNKNOWN.value

    def test_detect_empty_text(self):
        """Test detecting style on empty text."""
        engine = CitationProvenanceEngine()

        style = engine.detect_style("")

        assert style == CitationStyle.UNKNOWN.value


class TestChunkLinking:
    """Tests for chunk linking functionality."""

    def test_link_to_chunks_finds_matches(self):
        """Test linking citations to chunks."""
        engine = CitationProvenanceEngine()
        storage = create_mock_storage()

        citation = CitationProvenance(
            raw_text="(Smith, 2020)",
            author="Smith",
            year=2020,
            page_ref=None,
            citation_style=CitationStyle.APA.value,
            confidence=HIGH_CONFIDENCE,
        )

        matched = engine.link_to_chunks(citation, storage)

        assert len(matched) > 0
        assert "chunk_smith_1" in matched

    def test_link_updates_citation(self):
        """Test that linking updates citation object."""
        engine = CitationProvenanceEngine()
        storage = create_mock_storage()

        citation = CitationProvenance(
            raw_text="(Smith, 2020)",
            author="Smith",
            year=2020,
            page_ref=None,
            citation_style=CitationStyle.APA.value,
            confidence=HIGH_CONFIDENCE,
        )

        engine.link_to_chunks(citation, storage)

        assert len(citation.matched_chunks) > 0
        assert citation.confidence > 0

    def test_link_no_author_returns_empty(self):
        """Test linking with no author returns empty list."""
        engine = CitationProvenanceEngine()
        storage = create_mock_storage()

        citation = CitationProvenance(
            raw_text="[1]",
            author="",
            year=None,
            page_ref=None,
            citation_style=CitationStyle.NUMBERED.value,
            confidence=HIGH_CONFIDENCE,
        )

        matched = engine.link_to_chunks(citation, storage)

        assert matched == []

    def test_link_with_year_filter(self):
        """Test linking filters by year when available."""
        engine = CitationProvenanceEngine()
        storage = create_mock_storage()

        # Mock search to return result without the target year in content
        # (The year 2020 should NOT appear in the content for filtering to work)
        storage.search = Mock(
            return_value=[
                SearchResult(
                    chunk_id="chunk_wrong_year",
                    content="Smith published something in 2015.",
                    score=0.9,
                    document_id="doc_old",
                    section_title="Test",
                    chunk_type="content",
                    source_file="old.pdf",
                    word_count=30,
                )
            ]
        )

        citation = CitationProvenance(
            raw_text="(Smith, 2020)",
            author="Smith",
            year=2020,
            page_ref=None,
            citation_style=CitationStyle.APA.value,
            confidence=HIGH_CONFIDENCE,
        )

        matched = engine.link_to_chunks(citation, storage)

        # Should not match because year 2020 is not in the content
        assert len(matched) == 0


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_high_confidence_apa(self):
        """Test APA citations get high confidence."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("(Smith, 2020)")

        assert citations[0].confidence >= HIGH_CONFIDENCE

    def test_medium_confidence_mla_single(self):
        """Test single MLA citation gets medium confidence."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("(Author 42)")

        # Single author MLA has slightly lower confidence due to ambiguity
        assert citations[0].confidence >= MEDIUM_CONFIDENCE

    def test_confidence_increases_with_matches(self):
        """Test confidence increases with more chunk matches."""
        engine = CitationProvenanceEngine()

        citation = CitationProvenance(
            raw_text="(Smith, 2020)",
            author="Smith",
            year=2020,
            page_ref=None,
            citation_style=CitationStyle.APA.value,
            confidence=HIGH_CONFIDENCE,
        )

        # Test with 0, 1, and 3 matches
        conf_0 = engine._calculate_link_confidence(citation, 0)
        conf_1 = engine._calculate_link_confidence(citation, 1)
        conf_3 = engine._calculate_link_confidence(citation, 3)

        assert conf_0 == 0.0
        assert conf_1 > 0
        assert conf_3 >= conf_1


class TestCitationGraph:
    """Tests for citation graph building."""

    def test_build_citation_graph_without_storage(self):
        """Test building citation graph without storage."""
        engine = CitationProvenanceEngine()

        graph = engine.build_citation_graph(
            document_id="test_doc",
            text="As (Smith, 2020) and (Jones, 2019) show...",
            storage=None,
        )

        assert len(graph) == 2
        assert "(Smith, 2020)" in graph
        assert "(Jones, 2019)" in graph
        # Without storage, no chunks linked
        assert graph["(Smith, 2020)"] == []

    def test_build_citation_graph_with_storage(self):
        """Test building citation graph with storage."""
        engine = CitationProvenanceEngine()
        storage = create_mock_storage()

        graph = engine.build_citation_graph(
            document_id="test_doc",
            text="Smith (2020) and Williams (2021) argue...",
            storage=storage,
        )

        assert len(graph) >= 2
        # Should have linked chunks
        assert any(len(chunks) > 0 for chunks in graph.values())


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_extract_academic_citations_function(self):
        """Test extract_academic_citations convenience function."""
        citations = extract_academic_citations("(Smith, 2020)")

        assert len(citations) == 1
        assert citations[0].author == "Smith"

    def test_detect_citation_style_function(self):
        """Test detect_citation_style convenience function."""
        style = detect_citation_style("(Smith, 2020)")

        assert style == CitationStyle.APA.value

    def test_link_citations_to_chunks_function(self):
        """Test link_citations_to_chunks convenience function."""
        storage = create_mock_storage()

        citations = link_citations_to_chunks(
            "Smith (2020) and Williams (2021) argue...",
            storage,
        )

        assert len(citations) >= 1
        # At least one should have matched chunks
        assert any(len(c.matched_chunks) > 0 for c in citations)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_text(self):
        """Test extraction from empty text."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations("")

        assert citations == []

    def test_no_citations(self):
        """Test extraction from text without citations."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations(
            "This is plain text without any citations or references."
        )

        assert len(citations) == 0

    def test_text_too_long(self):
        """Test that overly long text raises error."""
        engine = CitationProvenanceEngine()

        long_text = "x" * (MAX_TEXT_LENGTH + 1)

        with pytest.raises(ValueError) as exc_info:
            engine.extract_citations(long_text)

        assert "too long" in str(exc_info.value)

    def test_deduplication(self):
        """Test that duplicate citations are removed."""
        engine = CitationProvenanceEngine()

        text = "(Smith, 2020) and again (Smith, 2020) cited."
        citations = engine.extract_citations(text)

        # Should deduplicate
        raw_texts = [c.raw_text for c in citations]
        assert raw_texts.count("(Smith, 2020)") == 1

    def test_mixed_styles(self):
        """Test extraction from mixed citation styles."""
        engine = CitationProvenanceEngine()

        citations = engine.extract_citations(MIXED_TEXT)

        styles = set(c.citation_style for c in citations)
        assert len(styles) >= 2  # At least 2 different styles

    def test_citation_to_dict(self):
        """Test CitationProvenance to_dict method."""
        citation = CitationProvenance(
            raw_text="(Smith, 2020)",
            author="Smith",
            year=2020,
            page_ref="42",
            citation_style=CitationStyle.APA.value,
            matched_chunks=["chunk1", "chunk2"],
            confidence=0.9,
        )

        result = citation.to_dict()

        assert result["raw_text"] == "(Smith, 2020)"
        assert result["author"] == "Smith"
        assert result["year"] == 2020
        assert result["page_ref"] == "42"
        assert result["matched_chunks"] == ["chunk1", "chunk2"]
        assert result["confidence"] == 0.9

    def test_year_boundaries(self):
        """Test citation year boundaries (1900-2099)."""
        engine = CitationProvenanceEngine()

        # Valid years
        citations_1900 = engine.extract_citations("(Author, 1900)")
        citations_2099 = engine.extract_citations("(Author, 2099)")

        assert len(citations_1900) == 1
        assert len(citations_2099) == 1

    def test_special_characters_in_text(self):
        """Test extraction with special characters around citations."""
        engine = CitationProvenanceEngine()

        text = '"As (Smith, 2020) noted...' + "and (Jones, 2019)."
        citations = engine.extract_citations(text)

        assert len(citations) == 2


class TestCitationStyleEnum:
    """Tests for CitationStyle enum."""

    def test_style_values(self):
        """Test CitationStyle enum values."""
        assert CitationStyle.APA.value == "apa"
        assert CitationStyle.MLA.value == "mla"
        assert CitationStyle.CHICAGO.value == "chicago"
        assert CitationStyle.NUMBERED.value == "numbered"
        assert CitationStyle.UNKNOWN.value == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
