"""Unit tests for TextCleanerRefiner."""


from ingestforge.ingest.refiners.text_cleaners import (
    TextCleanerRefiner,
    clean_bullets,
    clean_prefix_postfix,
    group_broken_paragraphs,
)


class TestTextCleanerRefiner:
    """Tests for TextCleanerRefiner class."""

    def test_is_available(self) -> None:
        """Test that refiner is always available."""
        cleaner = TextCleanerRefiner()
        assert cleaner.is_available() is True

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        cleaner = TextCleanerRefiner()
        result = cleaner.refine("")
        assert result.refined == ""
        assert not result.changes

    def test_no_changes_needed(self) -> None:
        """Test text that doesn't need cleaning."""
        cleaner = TextCleanerRefiner()
        text = "This is a complete sentence. It has proper formatting."
        result = cleaner.refine(text)
        assert result.refined == text


class TestGroupBrokenParagraphs:
    """Tests for paragraph grouping functionality."""

    def test_joins_broken_line(self) -> None:
        """Test joining a line that was broken mid-sentence."""
        text = "This is a broken\nline of text."
        result = group_broken_paragraphs(text)
        assert result == "This is a broken line of text."

    def test_preserves_paragraph_breaks(self) -> None:
        """Test that paragraph breaks are preserved."""
        text = "First paragraph.\n\nSecond paragraph."
        result = group_broken_paragraphs(text)
        assert result == text

    def test_preserves_sentence_breaks(self) -> None:
        """Test that sentence boundaries are preserved."""
        text = "First sentence.\nSecond sentence."
        result = group_broken_paragraphs(text)
        assert result == text

    def test_joins_continuation_lowercase(self) -> None:
        """Test joining when next line starts with lowercase."""
        text = "This sentence continues\non the next line."
        result = group_broken_paragraphs(text)
        assert result == "This sentence continues on the next line."

    def test_disabled_grouping(self) -> None:
        """Test that grouping can be disabled."""
        cleaner = TextCleanerRefiner(group_paragraphs=False)
        text = "This is a broken\nline of text."
        result = cleaner.refine(text)
        assert result.refined == text


class TestCleanBullets:
    """Tests for bullet normalization functionality."""

    def test_normalizes_unicode_bullets(self) -> None:
        """Test converting Unicode bullets to standard dash."""
        text = "\u2022 Item one\n\u2022 Item two"
        result = clean_bullets(text)
        assert result == "- Item one\n- Item two"

    def test_normalizes_multiple_bullet_types(self) -> None:
        """Test normalizing different bullet types."""
        text = "\u25cf First\n\u25cb Second\n\u2219 Third"
        result = clean_bullets(text)
        assert result == "- First\n- Second\n- Third"

    def test_preserves_regular_dashes(self) -> None:
        """Test that regular dashes are not affected."""
        text = "- Regular bullet\n- Another item"
        result = clean_bullets(text)
        assert result == text

    def test_disabled_bullet_cleaning(self) -> None:
        """Test that bullet cleaning can be disabled."""
        cleaner = TextCleanerRefiner(clean_bullets=False)
        text = "\u2022 Item one"
        result = cleaner.refine(text)
        assert result.refined == text


class TestCleanPrefixPostfix:
    """Tests for page number and header removal."""

    def test_removes_standalone_page_numbers(self) -> None:
        """Test removing standalone page numbers."""
        text = "Some content.\n\n42\n\nMore content."
        result = clean_prefix_postfix(text)
        assert "42" not in result
        assert "Some content." in result
        assert "More content." in result

    def test_removes_page_x_of_y(self) -> None:
        """Test removing 'Page X of Y' format."""
        text = "Content here.\n\nPage 3 of 10\n\nMore content."
        result = clean_prefix_postfix(text)
        assert "Page 3 of 10" not in result

    def test_preserves_numbers_in_text(self) -> None:
        """Test that numbers within text are preserved."""
        text = "There are 42 items in the list."
        result = clean_prefix_postfix(text)
        assert result == text

    def test_disabled_prefix_postfix_cleaning(self) -> None:
        """Test that prefix/postfix cleaning can be disabled."""
        cleaner = TextCleanerRefiner(clean_prefix_postfix=False)
        text = "Content.\n\n42\n\nMore."
        result = cleaner.refine(text)
        assert "42" in result.refined


class TestCombinedCleaning:
    """Tests for combined cleaning operations."""

    def test_all_operations_together(self) -> None:
        """Test all cleaning operations applied together."""
        text = "\u2022 Item one\n42\n\nBroken\nline here."
        cleaner = TextCleanerRefiner()
        result = cleaner.refine(text)

        # Bullet should be normalized
        assert "\u2022" not in result.refined
        # Page number should be removed
        assert "42" not in result.refined
        # Broken line should be joined
        assert "Broken line here" in result.refined

    def test_changes_tracked(self) -> None:
        """Test that changes are tracked in the result."""
        text = "\u2022 Item\n42\n\nBroken\nline."
        cleaner = TextCleanerRefiner()
        result = cleaner.refine(text)

        assert len(result.changes) >= 2  # At least bullets and page numbers
        assert result.was_modified


class TestConvenienceFunctions:
    """Tests for standalone convenience functions."""

    def test_group_broken_paragraphs_function(self) -> None:
        """Test the standalone group_broken_paragraphs function."""
        text = "This is\na test."
        result = group_broken_paragraphs(text)
        assert "This is a test" in result

    def test_clean_bullets_function(self) -> None:
        """Test the standalone clean_bullets function."""
        text = "\u2022 Item"
        result = clean_bullets(text)
        assert result == "- Item"

    def test_clean_prefix_postfix_function(self) -> None:
        """Test the standalone clean_prefix_postfix function."""
        text = "Content.\n\n42\n\nMore."
        result = clean_prefix_postfix(text)
        assert "42" not in result
