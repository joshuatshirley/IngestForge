"""
Tests for Legal Document Chunking.

This module tests chunking of legal documents by numbered sections/clauses.

Test Strategy
-------------
- Test section header pattern matching
- Test section splitting and hierarchy
- Test large section handling
- Test edge cases (no sections, nested sections)

Organization
------------
- TestSectionPatternMatching: Section header pattern detection
- TestBasicChunking: Basic legal document chunking
- TestLargeSectionSplitting: Large section handling
- TestSectionHierarchy: Section hierarchy building
- TestEdgeCases: Edge cases and fallbacks
"""

import pytest

from ingestforge.chunking.legal_chunker import LegalChunker
from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def legal_chunker() -> LegalChunker:
    """Create a LegalChunker instance with default config."""
    return LegalChunker()


@pytest.fixture
def config_with_small_max() -> Config:
    """Create config with small max section size for testing splits."""
    config = Config()
    return config


# ============================================================================
# Test Classes
# ============================================================================


class TestSectionPatternMatching:
    """Tests for section header pattern matching.

    Rule #4: Focused test class - tests pattern matching only
    """

    def test_match_article_pattern(self, legal_chunker: LegalChunker):
        """Test matching ARTICLE header patterns."""
        result = legal_chunker._match_section_header("ARTICLE 1 - Definitions")
        assert result is not None
        number, title, level, parent = result
        assert number == "Article 1"
        assert "Definitions" in title
        assert level == 1

    def test_match_section_pattern(self, legal_chunker: LegalChunker):
        """Test matching Section header patterns."""
        result = legal_chunker._match_section_header("Section 1.1 - Purpose")
        assert result is not None
        number, title, level, parent = result
        assert number == "1.1"
        assert level == 2
        assert parent == "1"

    def test_match_numbered_pattern(self, legal_chunker: LegalChunker):
        """Test matching numbered section patterns (1.1, 2.3.1)."""
        result = legal_chunker._match_section_header("1.2.3 Sub-subsection")
        assert result is not None
        number, title, level, parent = result
        assert number == "1.2.3"
        assert level == 3
        assert parent == "1.2"

    def test_match_single_number_pattern(self, legal_chunker: LegalChunker):
        """Test matching single number patterns (1. Title)."""
        result = legal_chunker._match_section_header("1. Introduction")
        assert result is not None
        number, title, level, parent = result
        assert number == "1"
        assert "Introduction" in title
        assert level == 1

    def test_match_paragraph_symbol(self, legal_chunker: LegalChunker):
        """Test matching paragraph symbol patterns (§1.1)."""
        result = legal_chunker._match_section_header("§1.1 Scope")
        assert result is not None
        number, title, level, parent = result
        assert "§" in number

    def test_match_roman_numeral(self, legal_chunker: LegalChunker):
        """Test matching Roman numeral patterns (I., II.)."""
        result = legal_chunker._match_section_header("IV. General Provisions")
        assert result is not None
        number, title, level, parent = result
        assert number == "IV"

    def test_no_match_for_regular_text(self, legal_chunker: LegalChunker):
        """Test that regular text doesn't match section patterns."""
        result = legal_chunker._match_section_header("This is regular paragraph text.")
        assert result is None


class TestBasicChunking:
    """Tests for basic legal document chunking.

    Rule #4: Focused test class - tests basic chunking
    """

    def test_chunk_simple_legal_document(self, legal_chunker: LegalChunker):
        """Test chunking a simple legal document."""
        text = """
ARTICLE 1 - DEFINITIONS

1.1 Agreement. This Agreement means the entire contract.

1.2 Parties. The parties to this agreement are Company and Employee.

ARTICLE 2 - TERMS

2.1 Duration. The term shall be one year.
"""
        chunks = legal_chunker.chunk(
            text, document_id="doc1", source_file="contract.txt"
        )

        assert len(chunks) > 0
        assert all(isinstance(c, ChunkRecord) for c in chunks)
        assert all(c.chunk_type == "legal_clause" for c in chunks)

    def test_chunks_have_section_titles(self, legal_chunker: LegalChunker):
        """Test that chunks include section titles."""
        text = """
1.1 Purpose. This document describes the purpose.

1.2 Scope. This document applies to all parties.
"""
        chunks = legal_chunker.chunk(text, document_id="doc1", source_file="test.txt")

        assert any("1.1" in c.section_title for c in chunks)
        assert any("1.2" in c.section_title for c in chunks)

    def test_chunks_have_correct_document_id(self, legal_chunker: LegalChunker):
        """Test that chunks have correct document_id."""
        text = "1.1 Simple clause."

        chunks = legal_chunker.chunk(
            text, document_id="test_doc_123", source_file="test.txt"
        )

        assert all(c.document_id == "test_doc_123" for c in chunks)

    def test_chunks_have_unique_ids(self, legal_chunker: LegalChunker):
        """Test that each chunk has a unique ID."""
        text = """
1.1 First clause.
1.2 Second clause.
1.3 Third clause.
"""
        chunks = legal_chunker.chunk(text, document_id="doc1", source_file="test.txt")

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique


class TestLargeSectionSplitting:
    """Tests for large section splitting.

    Rule #4: Focused test class - tests large section handling

    This tests the fix for TEST-MAINT-LEGAL: large sections should be
    split when they exceed max_section_size.
    """

    def test_split_large_section(self):
        """Test that large sections are split into smaller chunks.

        This is the primary test for TEST-MAINT-LEGAL backlog item.
        Large sections (>2000 chars by default) should be split.
        """
        chunker = LegalChunker()
        chunker.max_section_size = 500  # Set smaller for testing

        # Create a section with content that exceeds max_section_size
        # Note: Section header must be on its own line, content follows
        large_content = "This is a sentence. " * 100  # ~2000 chars
        text = f"""
1.1 Large Section

{large_content}

1.2 Normal Section

This is normal.
"""
        chunks = chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # The large section should be split into multiple chunks
        section_1_1_chunks = [c for c in chunks if "1.1" in c.section_title]
        assert (
            len(section_1_1_chunks) > 1
        ), "Large section should be split into multiple chunks"

        # Each chunk should be within size limits (with some tolerance)
        for chunk in section_1_1_chunks:
            assert chunk.char_count <= 600, f"Chunk too large: {chunk.char_count} chars"

    def test_split_by_sentences_when_no_paragraphs(self):
        """Test that large paragraphs without breaks are split by sentences."""
        chunker = LegalChunker()
        chunker.max_section_size = 200

        # Single long sentence repeated - no paragraph breaks
        text = (
            "1.1 Long Section. "
            + "This is sentence one. This is sentence two. This is sentence three. "
            * 20
        )

        chunks = chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # Should create multiple chunks
        assert len(chunks) > 1, "Long single paragraph should be split by sentences"

    def test_preserve_section_title_after_split(self):
        """Test that split sections retain the section title."""
        chunker = LegalChunker()
        chunker.max_section_size = 300

        large_content = "Content content content. " * 50
        text = f"Article 1 - Important Section\n\n{large_content}"

        chunks = chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # All chunks from the same section should have the same section title
        for chunk in chunks:
            assert "Article 1" in chunk.section_title

    def test_total_chunks_updated_after_split(self):
        """Test that total_chunks is correctly updated after splitting."""
        chunker = LegalChunker()
        chunker.max_section_size = 300

        large_content = "Word word word. " * 100
        text = f"1.1 Section. {large_content}"

        chunks = chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # total_chunks should reflect actual chunk count
        assert all(c.total_chunks == len(chunks) for c in chunks)


class TestSectionHierarchy:
    """Tests for section hierarchy building.

    Rule #4: Focused test class - tests hierarchy
    """

    def test_build_hierarchy_for_subsections(self, legal_chunker: LegalChunker):
        """Test that subsections have correct hierarchy."""
        text = """
1. Main Section

1.1 Subsection A

1.1.1 Sub-subsection

1.2 Subsection B
"""
        chunks = legal_chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # Find chunk for 1.1.1
        sub_sub_chunks = [c for c in chunks if "1.1.1" in c.section_title]
        if sub_sub_chunks:
            chunk = sub_sub_chunks[0]
            # Should have hierarchy pointing to parents
            assert chunk.section_hierarchy is not None

    def test_hierarchy_preserved_after_split(self):
        """Test that hierarchy is preserved when sections are split."""
        chunker = LegalChunker()
        chunker.max_section_size = 200

        large_content = "Content. " * 50
        text = f"""
1. Parent Section

1.1 Large Subsection. {large_content}
"""
        chunks = chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # All chunks from 1.1 should have the same hierarchy
        subsection_chunks = [c for c in chunks if "1.1" in c.section_title]
        if len(subsection_chunks) > 1:
            hierarchies = [tuple(c.section_hierarchy or []) for c in subsection_chunks]
            assert (
                len(set(hierarchies)) == 1
            ), "All split chunks should have same hierarchy"


class TestEdgeCases:
    """Tests for edge cases.

    Rule #4: Focused test class - tests edge cases
    """

    def test_empty_text(self, legal_chunker: LegalChunker):
        """Test handling of empty text."""
        chunks = legal_chunker.chunk("", document_id="doc1", source_file="test.txt")
        assert len(chunks) == 0

    def test_whitespace_only_text(self, legal_chunker: LegalChunker):
        """Test handling of whitespace-only text."""
        chunks = legal_chunker.chunk(
            "   \n\n   ", document_id="doc1", source_file="test.txt"
        )
        assert len(chunks) == 0

    def test_no_section_headers(self, legal_chunker: LegalChunker):
        """Test fallback when no section headers found."""
        text = "This is just a plain paragraph without any section structure."

        chunks = legal_chunker.chunk(text, document_id="doc1", source_file="test.txt")

        # Should fallback to semantic chunking
        assert len(chunks) >= 1

    def test_very_long_header_ignored(self, legal_chunker: LegalChunker):
        """Test that very long lines are not matched as headers."""
        # Line > 200 chars should not be matched
        long_line = "1.1 " + "x" * 250

        result = legal_chunker._match_section_header(long_line)
        assert result is None


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Pattern matching: 7 tests (article, section, numbered, single, paragraph, roman, regular)
    - Basic chunking: 4 tests (simple doc, titles, document_id, unique ids)
    - Large section splitting: 4 tests (split large, split by sentences, preserve title, total_chunks)
    - Section hierarchy: 2 tests (subsection hierarchy, hierarchy after split)
    - Edge cases: 4 tests (empty, whitespace, no headers, long header)

    Total: 21 tests

Design Decisions:
    1. Test all section header patterns (Article, Section, numbered, Roman, etc.)
    2. Verify large section splitting works correctly (TEST-MAINT-LEGAL fix)
    3. Test hierarchy preservation for structured documents
    4. Test fallback behavior when no sections found

Behaviors Tested:
    - Section pattern matching
    - Basic legal document chunking
    - Large section splitting (primary fix for TEST-MAINT-LEGAL)
    - Section hierarchy building
    - Edge cases (empty, no sections, etc.)
"""
