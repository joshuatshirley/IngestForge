"""Unit tests for ElementClassifier."""


from ingestforge.ingest.refinement import DocumentElementType
from ingestforge.ingest.refiners.element_classifier import (
    ElementClassifier,
    ClassifiedElement,
    classify_elements,
    get_element_type,
)


class TestElementClassifier:
    """Tests for ElementClassifier class."""

    def test_is_available(self) -> None:
        """Test that classifier is always available."""
        classifier = ElementClassifier()
        assert classifier.is_available() is True

    def test_empty_text(self) -> None:
        """Test handling of empty text."""
        classifier = ElementClassifier()
        result = classifier.refine("")
        assert result.refined == ""
        assert not result.chapter_markers


class TestTitleDetection:
    """Tests for title element detection."""

    def test_markdown_header(self) -> None:
        """Test detecting markdown headers."""
        classifier = ElementClassifier()
        assert classifier.classify_line("# Introduction") == DocumentElementType.TITLE
        assert classifier.classify_line("## Section") == DocumentElementType.TITLE
        assert classifier.classify_line("### Subsection") == DocumentElementType.TITLE

    def test_all_caps_heading(self) -> None:
        """Test detecting ALL CAPS headings."""
        classifier = ElementClassifier()
        assert classifier.classify_line("CHAPTER ONE") == DocumentElementType.TITLE
        assert classifier.classify_line("INTRODUCTION") == DocumentElementType.TITLE

    def test_numbered_section(self) -> None:
        """Test detecting numbered sections."""
        classifier = ElementClassifier()
        assert classifier.classify_line("1.2 Methods") == DocumentElementType.TITLE
        assert classifier.classify_line("3.4.5 Results") == DocumentElementType.TITLE

    def test_chapter_keyword(self) -> None:
        """Test detecting Chapter/Section keywords."""
        classifier = ElementClassifier()
        assert classifier.classify_line("Chapter 1") == DocumentElementType.TITLE
        assert classifier.classify_line("Section 2.3") == DocumentElementType.TITLE
        assert classifier.classify_line("Part III") == DocumentElementType.TITLE

    def test_short_title_without_punctuation(self) -> None:
        """Test detecting short lines without punctuation as titles."""
        classifier = ElementClassifier()
        result = classifier.classify_line("Introduction")
        assert result == DocumentElementType.TITLE

    def test_not_title_with_period(self) -> None:
        """Test that lines with periods are not titles."""
        classifier = ElementClassifier()
        result = classifier.classify_line("This is a sentence.")
        assert result == DocumentElementType.NARRATIVE_TEXT


class TestListItemDetection:
    """Tests for list item detection."""

    def test_bullet_points(self) -> None:
        """Test detecting bullet points."""
        classifier = ElementClassifier()
        assert classifier.classify_line("- Item one") == DocumentElementType.LIST_ITEM
        assert (
            classifier.classify_line("• Bullet item") == DocumentElementType.LIST_ITEM
        )
        assert (
            classifier.classify_line("● Another item") == DocumentElementType.LIST_ITEM
        )

    def test_numbered_list(self) -> None:
        """Test detecting numbered lists."""
        classifier = ElementClassifier()
        assert (
            classifier.classify_line("1. First item") == DocumentElementType.LIST_ITEM
        )
        assert (
            classifier.classify_line("2) Second item") == DocumentElementType.LIST_ITEM
        )

    def test_lettered_list(self) -> None:
        """Test detecting lettered lists."""
        classifier = ElementClassifier()
        assert classifier.classify_line("a. Item A") == DocumentElementType.LIST_ITEM
        assert classifier.classify_line("b) Item B") == DocumentElementType.LIST_ITEM


class TestCodeDetection:
    """Tests for code block detection."""

    def test_fenced_code_block(self) -> None:
        """Test detecting fenced code blocks."""
        classifier = ElementClassifier()
        code = "```python\nprint('hello')\n```"
        assert classifier.classify_line(code) == DocumentElementType.CODE

    def test_indented_code(self) -> None:
        """Test detecting indented code."""
        classifier = ElementClassifier()
        assert classifier.classify_line("    def foo():") == DocumentElementType.CODE

    def test_function_definition(self) -> None:
        """Test detecting function definitions."""
        classifier = ElementClassifier()
        assert (
            classifier.classify_line("def my_function():") == DocumentElementType.CODE
        )
        assert classifier.classify_line("class MyClass:") == DocumentElementType.CODE


class TestTableDetection:
    """Tests for table detection."""

    def test_markdown_table(self) -> None:
        """Test detecting markdown tables."""
        classifier = ElementClassifier()
        table = "| A | B |\n|---|---|\n| 1 | 2 |"
        assert classifier.classify_line(table) == DocumentElementType.TABLE

    def test_table_row(self) -> None:
        """Test detecting table rows."""
        classifier = ElementClassifier()
        assert classifier.classify_line("| Col1 | Col2 |") == DocumentElementType.TABLE


class TestImageDetection:
    """Tests for image/figure detection."""

    def test_markdown_image(self) -> None:
        """Test detecting markdown images."""
        classifier = ElementClassifier()
        assert (
            classifier.classify_line("![Alt text](image.png)")
            == DocumentElementType.IMAGE
        )

    def test_figure_reference(self) -> None:
        """Test detecting figure references."""
        classifier = ElementClassifier()
        assert classifier.classify_line("[Figure 1]") == DocumentElementType.IMAGE
        assert classifier.classify_line("[Figure 12]") == DocumentElementType.IMAGE


class TestHeaderFooterDetection:
    """Tests for header/footer detection."""

    def test_page_number(self) -> None:
        """Test detecting page numbers."""
        classifier = ElementClassifier()
        assert classifier.classify_line("Page 42") == DocumentElementType.HEADER
        assert classifier.classify_line("Page 1 of 10") == DocumentElementType.HEADER
        assert classifier.classify_line("42") == DocumentElementType.HEADER


class TestNarrativeText:
    """Tests for narrative text classification."""

    def test_regular_paragraph(self) -> None:
        """Test that regular paragraphs are classified as narrative."""
        classifier = ElementClassifier()
        text = "This is a regular paragraph with some text content."
        assert classifier.classify_line(text) == DocumentElementType.NARRATIVE_TEXT

    def test_multi_sentence(self) -> None:
        """Test multi-sentence paragraphs."""
        classifier = ElementClassifier()
        text = "First sentence. Second sentence. Third one too."
        assert classifier.classify_line(text) == DocumentElementType.NARRATIVE_TEXT


class TestClassifyText:
    """Tests for full text classification."""

    def test_classify_mixed_content(self) -> None:
        """Test classifying text with mixed content types."""
        text = """# Introduction

This is narrative text explaining the topic.

- First list item
- Second list item

## Conclusion

Final paragraph here."""

        classifier = ElementClassifier()
        elements = classifier.classify_text(text)

        types = [e.element_type for e in elements]
        assert DocumentElementType.TITLE in types
        assert DocumentElementType.NARRATIVE_TEXT in types
        assert DocumentElementType.LIST_ITEM in types

    def test_element_positions(self) -> None:
        """Test that element positions are tracked correctly."""
        text = "Title\n\nParagraph"
        classifier = ElementClassifier()
        elements = classifier.classify_text(text)

        assert len(elements) == 2
        assert elements[0].start_pos == 0
        assert elements[0].end_pos == 5
        assert elements[1].start_pos > elements[0].end_pos


class TestConvenienceFunctions:
    """Tests for standalone convenience functions."""

    def test_classify_elements_function(self) -> None:
        """Test the classify_elements convenience function."""
        text = "# Title\n\nParagraph text here."
        elements = classify_elements(text)

        assert len(elements) >= 2
        assert isinstance(elements[0], ClassifiedElement)

    def test_get_element_type_function(self) -> None:
        """Test the get_element_type convenience function."""
        assert get_element_type("# Header") == "Title"
        assert get_element_type("Regular text here.") == "NarrativeText"
        assert get_element_type("- List item") == "ListItem"


class TestConfidenceScores:
    """Tests for classification confidence scores."""

    def test_narrative_text_high_confidence(self) -> None:
        """Test that narrative text has high confidence."""
        classifier = ElementClassifier()
        elements = classifier.classify_text("This is regular paragraph text.")

        assert elements[0].confidence >= 0.8

    def test_markdown_header_high_confidence(self) -> None:
        """Test that markdown headers have high confidence."""
        classifier = ElementClassifier()
        elements = classifier.classify_text("# Clear Header")

        assert elements[0].element_type == DocumentElementType.TITLE
        assert elements[0].confidence >= 0.9


class TestClassifierOptions:
    """Tests for classifier configuration options."""

    def test_disable_title_detection(self) -> None:
        """Test disabling title detection."""
        classifier = ElementClassifier(detect_titles=False)
        result = classifier.classify_line("# Header")
        # Should fall through to narrative text
        assert result != DocumentElementType.TITLE

    def test_disable_list_detection(self) -> None:
        """Test disabling list detection."""
        classifier = ElementClassifier(detect_lists=False)
        result = classifier.classify_line("- Item")
        assert result != DocumentElementType.LIST_ITEM

    def test_disable_code_detection(self) -> None:
        """Test disabling code detection."""
        classifier = ElementClassifier(detect_code=False)
        result = classifier.classify_line("def function():")
        assert result != DocumentElementType.CODE
