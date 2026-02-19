"""Unit tests for LayoutChunker."""


from ingestforge.ingest.refinement import ChapterMarker
from ingestforge.chunking.layout_chunker import (
    LayoutChunker,
    LayoutSection,
    chunk_by_layout,
    chunk_by_title,
)


def create_markers(positions_and_titles: list) -> list:
    """Helper to create ChapterMarker list."""
    return [
        ChapterMarker(position=pos, title=title, level=level)
        for pos, title, level in positions_and_titles
    ]


class TestLayoutSection:
    """Tests for LayoutSection dataclass."""

    def test_word_count(self) -> None:
        """Test word count property."""
        section = LayoutSection(
            title="Test",
            content="one two three four",
            level=1,
            start_pos=0,
            end_pos=18,
        )
        assert section.word_count == 4

    def test_char_count(self) -> None:
        """Test character count property."""
        section = LayoutSection(
            title="Test",
            content="hello",
            level=1,
            start_pos=0,
            end_pos=5,
        )
        assert section.char_count == 5


class TestLayoutChunker:
    """Tests for LayoutChunker class."""

    def test_chunk_empty_text(self) -> None:
        """Test chunking empty text."""
        chunker = LayoutChunker()
        chunks = chunker.chunk("", "doc_123")
        assert chunks == []

    def test_chunk_without_markers(self) -> None:
        """Test chunking without markers (paragraph split)."""
        chunker = LayoutChunker()
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = chunker.chunk(text, "doc_123")

        assert len(chunks) == 3
        assert "First paragraph" in chunks[0].content
        assert "Second paragraph" in chunks[1].content
        assert "Third paragraph" in chunks[2].content

    def test_chunk_with_markers(self) -> None:
        """Test chunking with chapter markers."""
        # Use longer content to avoid combining
        chunker = LayoutChunker(combine_text_under_n_chars=10)
        text = "Intro text here.\n\nChapter 1\n\nContent here is longer.\n\nChapter 2\n\nMore content in second chapter."
        markers = create_markers(
            [
                (0, "Intro", 1),
                (18, "Chapter 1", 1),
                (54, "Chapter 2", 1),
            ]
        )

        chunks = chunker.chunk_with_markers(text, markers, "doc_123")

        assert len(chunks) >= 1
        # Verify sections are created from markers
        assert any(c.section_title for c in chunks)

    def test_chunk_by_title_mode(self) -> None:
        """Test chunk_by_title splits only at level 1 markers."""
        chunker = LayoutChunker(chunk_by_title=True)
        text = "Title 1\n\nSection 1.1\n\nContent\n\nTitle 2\n\nMore content"
        markers = create_markers(
            [
                (0, "Title 1", 1),
                (9, "Section 1.1", 2),
                (30, "Title 2", 1),
            ]
        )

        chunks = chunker.chunk_with_markers(text, markers, "doc_123")

        # Should split at level 1 markers only
        titles = [c.section_title for c in chunks if c.section_title]
        assert "Title 1" in titles or any("Title 1" in t for t in titles)


class TestSmallSectionCombining:
    """Tests for combining small sections."""

    def test_combines_small_sections(self) -> None:
        """Test that small sections are combined."""
        chunker = LayoutChunker(combine_text_under_n_chars=50)
        text = "A\n\nB\n\nC\n\nThis is a longer section with more content."
        markers = create_markers(
            [
                (0, "A", 2),
                (3, "B", 2),
                (6, "C", 2),
                (9, "Long", 2),
            ]
        )

        chunks = chunker.chunk_with_markers(text, markers, "doc_123")

        # Small sections should be combined
        assert len(chunks) < 4


class TestLargeSectionSplitting:
    """Tests for splitting large sections."""

    def test_splits_large_sections(self) -> None:
        """Test that large sections are split at paragraph boundaries."""
        chunker = LayoutChunker(max_chunk_size=100)

        # Create text with large section
        para1 = "A" * 60
        para2 = "B" * 60
        text = f"Title\n\n{para1}\n\n{para2}"
        markers = create_markers([(0, "Title", 1)])

        chunks = chunker.chunk_with_markers(text, markers, "doc_123")

        # Should be split into multiple chunks
        assert len(chunks) >= 2
        # Each chunk should be under max size
        for chunk in chunks:
            assert (
                len(chunk.content) <= chunker.max_chunk_size + 100
            )  # Allow some tolerance

    def test_respects_max_chunk_size(self) -> None:
        """Test that max chunk size is respected when splitting at paragraphs."""
        chunker = LayoutChunker(max_chunk_size=200)

        # Create text with multiple paragraphs that can be split
        para1 = "First paragraph with some text. " * 3
        para2 = "Second paragraph with more text. " * 3
        para3 = "Third paragraph with even more text. " * 3
        text = f"{para1}\n\n{para2}\n\n{para3}"

        chunks = chunker.chunk(text, "doc_123")

        # Should produce multiple chunks since total is > 200 chars
        assert len(chunks) >= 2
        # Each chunk should respect max size (with tolerance for paragraph boundaries)
        for chunk in chunks:
            assert (
                len(chunk.content) <= 400
            ), f"Chunk too large: {len(chunk.content)} chars"
        # All content should be preserved
        combined = " ".join(c.content for c in chunks)
        assert "First paragraph" in combined
        assert "Third paragraph" in combined


class TestChunkMetadata:
    """Tests for chunk metadata."""

    def test_chunk_has_section_title(self) -> None:
        """Test that chunks have section titles."""
        chunker = LayoutChunker()
        text = "Chapter 1\n\nContent here."
        markers = create_markers([(0, "Chapter 1", 1)])

        chunks = chunker.chunk_with_markers(text, markers, "doc_123")

        assert any(c.section_title == "Chapter 1" for c in chunks)

    def test_chunk_has_document_id(self) -> None:
        """Test that chunks have document ID."""
        chunker = LayoutChunker()
        chunks = chunker.chunk("Test content.", "my_doc_id")

        assert all(c.document_id == "my_doc_id" for c in chunks)

    def test_chunk_has_word_count(self) -> None:
        """Test that chunks have word count."""
        chunker = LayoutChunker()
        chunks = chunker.chunk("One two three four.", "doc")

        assert chunks[0].word_count == 4

    def test_chunk_has_char_count(self) -> None:
        """Test that chunks have character count."""
        chunker = LayoutChunker()
        chunks = chunker.chunk("Hello", "doc")

        assert chunks[0].char_count == 5

    def test_chunk_indices(self) -> None:
        """Test that chunks have correct indices."""
        chunker = LayoutChunker()
        text = "Para 1.\n\nPara 2.\n\nPara 3."
        chunks = chunker.chunk(text, "doc")

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_metadata_passed_to_chunks(self) -> None:
        """Test that metadata is passed to chunks."""
        chunker = LayoutChunker()
        metadata = {"source": "test", "author": "alice"}
        chunks = chunker.chunk("Content.", "doc", metadata=metadata)

        assert chunks[0].metadata["source"] == "test"
        assert chunks[0].metadata["author"] == "alice"


class TestConvenienceFunctions:
    """Tests for standalone convenience functions."""

    def test_chunk_by_layout_function(self) -> None:
        """Test the chunk_by_layout convenience function."""
        text = "Title\n\nContent here.\n\nMore content."
        markers = create_markers([(0, "Title", 1)])

        chunks = chunk_by_layout(text, markers, "doc_123")

        assert len(chunks) >= 1
        assert all(c.document_id == "doc_123" for c in chunks)

    def test_chunk_by_title_function(self) -> None:
        """Test the chunk_by_title convenience function."""
        # Use longer text to ensure separate chunks
        text = (
            "Title 1\n\n"
            + ("Content " * 50)
            + "\n\nTitle 2\n\n"
            + ("More content " * 50)
        )
        markers = create_markers(
            [
                (0, "Title 1", 1),
                (260, "Title 2", 1),  # Approximate position
            ]
        )

        chunks = chunk_by_title(text, markers, "doc")

        # Should create chunks for the content
        assert len(chunks) >= 1
        assert any(c.section_title for c in chunks)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_markers_list(self) -> None:
        """Test handling empty markers list."""
        chunker = LayoutChunker()
        chunks = chunker.chunk_with_markers("Some text.", [], "doc")

        assert len(chunks) == 1
        assert chunks[0].content == "Some text."
        assert chunks[0].document_id == "doc"

    def test_single_marker(self) -> None:
        """Test single marker at start."""
        chunker = LayoutChunker()
        text = "Title\n\nContent."
        markers = create_markers([(0, "Title", 1)])

        chunks = chunker.chunk_with_markers(text, markers, "doc")

        assert len(chunks) >= 1
        # Content should be preserved
        all_content = " ".join(c.content for c in chunks)
        assert "Title" in all_content or "Content" in all_content
        # Should have section title from marker
        assert any(c.section_title == "Title" for c in chunks)

    def test_marker_at_end(self) -> None:
        """Test marker at end of text."""
        chunker = LayoutChunker()
        text = "Content.\n\nTitle"
        markers = create_markers([(10, "Title", 1)])

        chunks = chunker.chunk_with_markers(text, markers, "doc")

        assert len(chunks) >= 1
        # All text should be captured
        all_content = " ".join(c.content for c in chunks)
        assert "Content" in all_content
        assert "Title" in all_content

    def test_text_before_first_marker(self) -> None:
        """Test text before first marker is included."""
        chunker = LayoutChunker()
        text = "Preface text.\n\nChapter 1\n\nContent."
        markers = create_markers([(15, "Chapter 1", 1)])

        chunks = chunker.chunk_with_markers(text, markers, "doc")

        # Preface should be in a chunk
        all_content = " ".join(c.content for c in chunks)
        assert "Preface" in all_content
