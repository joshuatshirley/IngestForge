"""Tests for link_extractor module (CITE-001.1).

Tests the internal reference extractor:
- Reference pattern matching
- Reference types detection
- Link map construction
- Reference resolution
"""


from ingestforge.core.citation.link_extractor import (
    ReferenceType,
    InternalReference,
    LinkMap,
    LinkExtractor,
    extract_internal_references,
)


class TestReferenceType:
    """Test ReferenceType enum."""

    def test_all_types_defined(self) -> None:
        """All reference types should be defined."""
        assert ReferenceType.EXPLICIT_CITATION
        assert ReferenceType.SECTION_REF
        assert ReferenceType.FIGURE_REF
        assert ReferenceType.TABLE_REF
        assert ReferenceType.EQUATION_REF
        assert ReferenceType.CHAPTER_REF
        assert ReferenceType.PAGE_REF
        assert ReferenceType.FOOTNOTE_REF
        assert ReferenceType.DOCUMENT_REF
        assert ReferenceType.LEGAL_CITATION
        assert ReferenceType.WIKILINK


class TestInternalReference:
    """Test InternalReference dataclass."""

    def test_unresolved_reference(self) -> None:
        """Unresolved reference should not be marked resolved."""
        ref = InternalReference(
            source_chunk_id="chunk1",
            reference_text="[Smith, 2020]",
        )

        assert not ref.is_resolved
        assert ref.target_chunk_id is None

    def test_resolved_reference(self) -> None:
        """Resolved reference should be marked resolved."""
        ref = InternalReference(
            source_chunk_id="chunk1",
            target_chunk_id="chunk2",
            reference_text="[Smith, 2020]",
            resolved=True,
            confidence=0.8,
        )

        assert ref.is_resolved
        assert ref.target_chunk_id == "chunk2"


class TestLinkMap:
    """Test LinkMap dataclass."""

    def test_empty_link_map(self) -> None:
        """Empty link map should have zero counts."""
        link_map = LinkMap()

        assert link_map.node_count == 0
        assert link_map.edge_count == 0

    def test_add_reference(self) -> None:
        """Should add references to link map."""
        link_map = LinkMap()

        ref = InternalReference(
            source_chunk_id="chunk1",
            target_chunk_id="chunk2",
            resolved=True,
        )

        link_map.add_reference(ref)

        assert "chunk2" in link_map.get_outgoing("chunk1")
        assert "chunk1" in link_map.get_incoming("chunk2")

    def test_skip_unresolved(self) -> None:
        """Should skip unresolved references."""
        link_map = LinkMap()

        ref = InternalReference(
            source_chunk_id="chunk1",
            reference_text="[Smith, 2020]",
            resolved=False,
        )

        link_map.add_reference(ref)

        assert link_map.edge_count == 0

    def test_node_count(self) -> None:
        """Should count unique nodes."""
        link_map = LinkMap()

        ref1 = InternalReference(
            source_chunk_id="A",
            target_chunk_id="B",
            resolved=True,
        )
        ref2 = InternalReference(
            source_chunk_id="A",
            target_chunk_id="C",
            resolved=True,
        )

        link_map.add_reference(ref1)
        link_map.add_reference(ref2)

        assert link_map.node_count == 3
        assert link_map.edge_count == 2


class TestLinkExtractorPatterns:
    """Test reference pattern matching."""

    def test_extract_explicit_citation_brackets(self) -> None:
        """Should extract [Author, Year] citations."""
        extractor = LinkExtractor()
        text = "As shown by [Smith, 2020], the results are significant."

        refs = extractor.extract_references(text, "chunk1")

        assert len(refs) == 1
        assert refs[0].reference_type == ReferenceType.EXPLICIT_CITATION

    def test_extract_section_reference(self) -> None:
        """Should extract section references."""
        extractor = LinkExtractor()
        text = "See Section 3.2 for details."

        refs = extractor.extract_references(text, "chunk1")

        assert len(refs) == 1
        assert refs[0].reference_type == ReferenceType.SECTION_REF

    def test_extract_figure_reference(self) -> None:
        """Should extract figure references."""
        extractor = LinkExtractor()
        text = "As shown in Figure 5, the trend is clear."

        refs = extractor.extract_references(text, "chunk1")

        assert len(refs) == 1
        assert refs[0].reference_type == ReferenceType.FIGURE_REF

    def test_extract_table_reference(self) -> None:
        """Should extract table references."""
        extractor = LinkExtractor()
        text = "Results are summarized in Table 2."

        refs = extractor.extract_references(text, "chunk1")

        assert len(refs) == 1
        assert refs[0].reference_type == ReferenceType.TABLE_REF

    def test_extract_legal_citation(self) -> None:
        """Should extract legal citations."""
        extractor = LinkExtractor()
        text = "Established in Roe v. Wade, 410 U.S. 113 (1973)."

        refs = extractor.extract_references(text, "chunk1")

        assert len(refs) == 1
        assert refs[0].reference_type == ReferenceType.LEGAL_CITATION
        assert refs[0].reference_text == "410 U.S. 113"

    def test_extract_wikilink(self) -> None:
        """Should extract wikilinks."""
        extractor = LinkExtractor()
        text = "See [[Target Page]] or [[Other Page|With Alias]]."

        refs = extractor.extract_references(text, "chunk1")

        assert len(refs) == 2
        assert refs[0].reference_type == ReferenceType.WIKILINK
        assert refs[0].reference_text == "[[Target Page]]"
        assert refs[1].reference_type == ReferenceType.WIKILINK
        assert refs[1].reference_text == "[[Other Page|With Alias]]"


class TestLinkExtractorValidation:
    """Test input validation."""

    def test_empty_text(self) -> None:
        """Should handle empty text."""
        extractor = LinkExtractor()
        refs = extractor.extract_references("", "chunk1")

        assert refs == []

    def test_no_references(self) -> None:
        """Should handle text with no references."""
        extractor = LinkExtractor()
        text = "This is a simple paragraph with no citations."

        refs = extractor.extract_references(text, "chunk1")

        assert refs == []


class TestConvenienceFunction:
    """Test extract_internal_references convenience function."""

    def test_extract_convenience(self) -> None:
        """Should work as standalone function."""
        text = "According to [Smith, 2020], the results are valid."

        refs = extract_internal_references(text, "chunk1")

        assert len(refs) == 1
        assert refs[0].source_chunk_id == "chunk1"
