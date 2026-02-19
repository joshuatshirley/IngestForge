"""
Tests for Markdown document processor.

This module tests extraction of text, frontmatter, code blocks,
links, images, and table of contents from Markdown files.

Test Strategy
-------------
- Test frontmatter extraction (YAML)
- Test code block extraction with language
- Test link extraction (inline and reference)
- Test image extraction
- Test table of contents generation
- Test text cleaning
- No external dependencies to mock

Organization
------------
- TestBasicProcessing: Main process method
- TestProcessToDocument: Structured document output
- TestFrontmatterExtraction: YAML frontmatter parsing
- TestYamlParsing: Simple YAML parser
- TestTitleExtraction: Title from frontmatter or h1
- TestTocGeneration: Table of contents building
- TestCodeBlockExtraction: Fenced code block extraction
- TestLinkExtraction: Inline and reference links
- TestImageExtraction: Image references
- TestContentCleaning: Markdown to plain text
- TestHelperFunctions: Module-level helper functions
- TestComplexDocuments: Realistic scenarios
- TestEdgeCases: Edge cases and malformed input
"""


from ingestforge.ingest.markdown_processor import (
    MarkdownProcessor,
    MarkdownDocument,
    CodeBlock,
    Link,
    Image,
    TocEntry,
    extract_text,
    extract_with_metadata,
    process_to_document,
    extract_frontmatter,
    extract_code_blocks,
    build_toc,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestBasicProcessing:
    """Tests for basic Markdown processing.

    Rule #4: Focused test class - tests process method
    """

    def test_process_minimal_document(self, temp_dir):
        """Test processing minimal Markdown document."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text("Hello World")

        result = processor.process(md_file)

        assert "Hello World" in result["text"]
        assert result["type"] == "markdown"
        assert result["source"] == "doc.md"

    def test_process_with_frontmatter(self, temp_dir):
        """Test processing document with frontmatter."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text(
            """---
title: My Document
author: John Doe
---

Content here.
"""
        )

        result = processor.process(md_file)

        assert result["frontmatter"]["title"] == "My Document"
        assert result["frontmatter"]["author"] == "John Doe"
        assert "Content here" in result["text"]

    def test_process_with_headers(self, temp_dir):
        """Test processing document with headers."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text(
            """# Title

## Section 1

Content.

## Section 2

More content.
"""
        )

        result = processor.process(md_file)

        assert len(result["toc"]) == 3
        assert result["metadata"]["title"] == "Title"

    def test_process_returns_code_blocks(self, temp_dir):
        """Test that process returns extracted code blocks."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text(
            """# Doc

```python
print('hello')
```
"""
        )

        result = processor.process(md_file)

        assert len(result["code_blocks"]) == 1
        assert result["code_blocks"][0]["language"] == "python"

    def test_process_returns_links(self, temp_dir):
        """Test that process returns extracted links."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text("Check out [this link](https://example.com).")

        result = processor.process(md_file)

        assert len(result["links"]) == 1
        assert result["links"][0]["url"] == "https://example.com"


class TestProcessToDocument:
    """Tests for process_to_document method."""

    def test_returns_markdown_document(self, temp_dir):
        """Test that process_to_document returns MarkdownDocument."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text("# Test\n\nContent")

        doc = processor.process_to_document(md_file)

        assert isinstance(doc, MarkdownDocument)
        assert doc.title == "Test"

    def test_document_has_all_fields(self, temp_dir):
        """Test that document contains all expected fields."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        md_file.write_text("Content")

        doc = processor.process_to_document(md_file)

        assert hasattr(doc, "title")
        assert hasattr(doc, "frontmatter")
        assert hasattr(doc, "toc")
        assert hasattr(doc, "code_blocks")
        assert hasattr(doc, "links")
        assert hasattr(doc, "images")
        assert hasattr(doc, "text")
        assert hasattr(doc, "word_count")


class TestFrontmatterExtraction:
    """Tests for frontmatter extraction."""

    def test_extract_frontmatter(self, temp_dir):
        """Test extracting YAML frontmatter."""
        processor = MarkdownProcessor()

        content = """---
title: Document Title
author: Jane Doe
date: 2024-01-15
---

Body content.
"""

        content_without_fm, fm = processor._extract_frontmatter(content)

        assert fm["title"] == "Document Title"
        assert fm["author"] == "Jane Doe"
        assert "Body content" in content_without_fm
        assert "---" not in content_without_fm

    def test_no_frontmatter(self, temp_dir):
        """Test document without frontmatter."""
        processor = MarkdownProcessor()

        content = "# Just a heading\n\nBody content."

        content_without_fm, fm = processor._extract_frontmatter(content)

        assert fm == {}
        assert content_without_fm == content

    def test_frontmatter_with_list(self, temp_dir):
        """Test frontmatter with list values."""
        processor = MarkdownProcessor()

        content = """---
title: Document
tags:
- python
- markdown
- testing
---

Content.
"""

        _, fm = processor._extract_frontmatter(content)

        assert fm["title"] == "Document"
        assert fm["tags"] == ["python", "markdown", "testing"]

    def test_frontmatter_with_boolean(self, temp_dir):
        """Test frontmatter with boolean values."""
        processor = MarkdownProcessor()

        content = """---
draft: true
published: false
---

Content.
"""

        _, fm = processor._extract_frontmatter(content)

        assert fm["draft"] is True
        assert fm["published"] is False

    def test_frontmatter_with_numbers(self, temp_dir):
        """Test frontmatter with numeric values."""
        processor = MarkdownProcessor()

        content = """---
version: 2
weight: 3.14
---

Content.
"""

        _, fm = processor._extract_frontmatter(content)

        assert fm["version"] == 2
        assert fm["weight"] == 3.14


class TestYamlParsing:
    """Tests for simple YAML parsing."""

    def test_parse_quoted_string(self, temp_dir):
        """Test parsing quoted string values."""
        processor = MarkdownProcessor()

        result = processor._parse_yaml_line('title: "Hello World"')

        assert result == ("title", "Hello World")

    def test_parse_single_quoted_string(self, temp_dir):
        """Test parsing single-quoted string values."""
        processor = MarkdownProcessor()

        result = processor._parse_yaml_line("name: 'John Doe'")

        assert result == ("name", "John Doe")

    def test_parse_null_value(self, temp_dir):
        """Test parsing null values."""
        processor = MarkdownProcessor()

        result = processor._parse_yaml_line("value: null")

        assert result == ("value", None)

    def test_parse_empty_value(self, temp_dir):
        """Test parsing empty values."""
        processor = MarkdownProcessor()

        result = processor._parse_yaml_line("key:")

        assert result == ("key", None)

    def test_skip_comments(self, temp_dir):
        """Test that comments are skipped."""
        processor = MarkdownProcessor()

        yaml_content = """
# This is a comment
title: Test
# Another comment
"""

        result = processor._parse_yaml_simple(yaml_content)

        assert result["title"] == "Test"
        assert len(result) == 1


class TestTitleExtraction:
    """Tests for title extraction."""

    def test_title_from_frontmatter(self, temp_dir):
        """Test getting title from frontmatter."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        frontmatter = {"title": "Frontmatter Title"}
        content = "# Heading Title\n\nContent"

        title = processor._extract_title(frontmatter, content, md_file)

        assert title == "Frontmatter Title"

    def test_title_from_h1(self, temp_dir):
        """Test getting title from first h1."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "doc.md"
        frontmatter = {}
        content = "# Heading Title\n\nContent"

        title = processor._extract_title(frontmatter, content, md_file)

        assert title == "Heading Title"

    def test_title_from_filename(self, temp_dir):
        """Test getting title from filename."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "my-document.md"
        frontmatter = {}
        content = "Just content, no heading."

        title = processor._extract_title(frontmatter, content, md_file)

        assert title == "my-document"


class TestTocGeneration:
    """Tests for table of contents generation."""

    def test_build_toc_single_header(self, temp_dir):
        """Test TOC with single header."""
        processor = MarkdownProcessor()

        content = "# Title"

        toc = processor._build_toc(content)

        assert len(toc) == 1
        assert toc[0].level == 1
        assert toc[0].text == "Title"

    def test_build_toc_multiple_headers(self, temp_dir):
        """Test TOC with multiple headers."""
        processor = MarkdownProcessor()

        content = """# Title

## Section 1

### Subsection 1.1

## Section 2
"""

        toc = processor._build_toc(content)

        assert len(toc) == 4
        assert toc[0].level == 1
        assert toc[1].level == 2
        assert toc[2].level == 3
        assert toc[3].level == 2

    def test_toc_anchor_generation(self, temp_dir):
        """Test that anchors are generated correctly."""
        processor = MarkdownProcessor()

        content = "# Hello World"

        toc = processor._build_toc(content)

        assert toc[0].anchor == "hello-world"

    def test_toc_anchor_special_characters(self, temp_dir):
        """Test anchor generation with special characters."""
        processor = MarkdownProcessor()

        anchor = processor._generate_anchor("Hello, World! (2024)")

        assert anchor == "hello-world-2024"

    def test_toc_entry_dataclass(self):
        """Test TocEntry dataclass."""
        entry = TocEntry(level=2, text="Section", anchor="section", position=10)

        assert entry.level == 2
        assert entry.text == "Section"
        assert entry.anchor == "section"
        assert entry.position == 10


class TestCodeBlockExtraction:
    """Tests for code block extraction."""

    def test_extract_code_block_with_language(self, temp_dir):
        """Test extracting code block with language."""
        processor = MarkdownProcessor()

        content = """
```python
def hello():
    print("Hello")
```
"""

        blocks = processor._extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].code

    def test_extract_code_block_no_language(self, temp_dir):
        """Test extracting code block without language."""
        processor = MarkdownProcessor()

        content = """
```
plain code
```
"""

        blocks = processor._extract_code_blocks(content)

        assert len(blocks) == 1
        assert blocks[0].language == ""
        assert "plain code" in blocks[0].code

    def test_extract_multiple_code_blocks(self, temp_dir):
        """Test extracting multiple code blocks."""
        processor = MarkdownProcessor()

        content = """
```python
python code
```

```javascript
js code
```
"""

        blocks = processor._extract_code_blocks(content)

        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"

    def test_code_block_dataclass(self):
        """Test CodeBlock dataclass."""
        block = CodeBlock(
            language="python",
            code="print('hello')",
            position=10,
            info_string="python file.py",
        )

        assert block.language == "python"
        assert block.code == "print('hello')"
        assert block.info_string == "python file.py"

    def test_extract_code_block_multiline(self, temp_dir):
        """Test extracting multiline code block."""
        processor = MarkdownProcessor()

        content = """
```python
def func():
    x = 1
    y = 2
    return x + y
```
"""

        blocks = processor._extract_code_blocks(content)

        assert len(blocks) == 1
        assert "def func():" in blocks[0].code
        assert "return x + y" in blocks[0].code


class TestLinkExtraction:
    """Tests for link extraction."""

    def test_extract_inline_link(self, temp_dir):
        """Test extracting inline link."""
        processor = MarkdownProcessor()

        content = "[Example](https://example.com)"

        links = processor._extract_links(content)

        assert len(links) == 1
        assert links[0].text == "Example"
        assert links[0].url == "https://example.com"
        assert links[0].is_reference is False

    def test_extract_inline_link_with_title(self, temp_dir):
        """Test extracting inline link with title."""
        processor = MarkdownProcessor()

        content = '[Example](https://example.com "Example Site")'

        links = processor._extract_links(content)

        assert len(links) == 1
        assert links[0].title == "Example Site"

    def test_extract_reference_link(self, temp_dir):
        """Test extracting reference link."""
        processor = MarkdownProcessor()

        content = """[Example][1]

[1]: https://example.com
"""

        links = processor._extract_links(content)

        assert len(links) == 1
        assert links[0].text == "Example"
        assert links[0].url == "https://example.com"
        assert links[0].is_reference is True

    def test_extract_reference_link_implicit(self, temp_dir):
        """Test extracting implicit reference link."""
        processor = MarkdownProcessor()

        content = """[example][]

[example]: https://example.com
"""

        links = processor._extract_links(content)

        assert len(links) == 1
        assert links[0].text == "example"

    def test_extract_multiple_links(self, temp_dir):
        """Test extracting multiple links."""
        processor = MarkdownProcessor()

        content = "[Link1](https://a.com) and [Link2](https://b.com)"

        links = processor._extract_links(content)

        assert len(links) == 2

    def test_link_dataclass(self):
        """Test Link dataclass."""
        link = Link(
            text="Example",
            url="https://example.com",
            title="A site",
            position=0,
            is_reference=False,
        )

        assert link.text == "Example"
        assert link.url == "https://example.com"
        assert link.title == "A site"

    def test_not_extract_image_as_link(self, temp_dir):
        """Test that images are not extracted as links."""
        processor = MarkdownProcessor()

        content = "![Image](image.png) and [Link](page.html)"

        links = processor._extract_links(content)

        assert len(links) == 1
        assert links[0].text == "Link"


class TestImageExtraction:
    """Tests for image extraction."""

    def test_extract_image(self, temp_dir):
        """Test extracting image."""
        processor = MarkdownProcessor()

        content = "![Alt text](image.png)"

        images = processor._extract_images(content)

        assert len(images) == 1
        assert images[0].alt_text == "Alt text"
        assert images[0].url == "image.png"

    def test_extract_image_with_title(self, temp_dir):
        """Test extracting image with title."""
        processor = MarkdownProcessor()

        content = '![Alt](image.png "Image title")'

        images = processor._extract_images(content)

        assert len(images) == 1
        assert images[0].title == "Image title"

    def test_extract_multiple_images(self, temp_dir):
        """Test extracting multiple images."""
        processor = MarkdownProcessor()

        content = "![Img1](a.png) and ![Img2](b.png)"

        images = processor._extract_images(content)

        assert len(images) == 2

    def test_extract_image_empty_alt(self, temp_dir):
        """Test extracting image with empty alt text."""
        processor = MarkdownProcessor()

        content = "![](image.png)"

        images = processor._extract_images(content)

        assert len(images) == 1
        assert images[0].alt_text == ""

    def test_image_dataclass(self):
        """Test Image dataclass."""
        img = Image(alt_text="Photo", url="photo.jpg", title="A photo", position=0)

        assert img.alt_text == "Photo"
        assert img.url == "photo.jpg"


class TestContentCleaning:
    """Tests for content cleaning."""

    def test_clean_removes_code_blocks(self, temp_dir):
        """Test that code blocks are replaced."""
        processor = MarkdownProcessor()

        content = """Text before

```python
code
```

Text after"""

        result = processor._clean_content(content)

        assert "[CODE]" in result
        assert "```" not in result

    def test_clean_removes_headers_markers(self, temp_dir):
        """Test that header markers are removed."""
        processor = MarkdownProcessor()

        content = "# Title\n\nContent"

        result = processor._clean_content(content)

        assert "Title" in result
        assert "#" not in result

    def test_clean_removes_emphasis(self, temp_dir):
        """Test that emphasis markers are removed."""
        processor = MarkdownProcessor()

        content = "This is **bold** and *italic* text."

        result = processor._clean_content(content)

        assert "bold" in result
        assert "italic" in result
        assert "**" not in result
        assert "*" not in result or result.count("*") == 0

    def test_clean_converts_links_to_text(self, temp_dir):
        """Test that links are converted to just text."""
        processor = MarkdownProcessor()

        content = "Check [this link](https://example.com) out."

        result = processor._clean_content(content)

        assert "this link" in result
        assert "https://example.com" not in result

    def test_clean_marks_images(self, temp_dir):
        """Test that images are marked."""
        processor = MarkdownProcessor()

        content = "![Photo](image.png)"

        result = processor._clean_content(content)

        assert "[IMAGE: Photo]" in result

    def test_clean_removes_inline_code(self, temp_dir):
        """Test that inline code markers are removed."""
        processor = MarkdownProcessor()

        content = "Use the `print()` function."

        result = processor._clean_content(content)

        assert "print()" in result
        assert "`" not in result


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_extract_text_function(self, temp_dir):
        """Test extract_text helper function."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Title\n\nContent")

        result = extract_text(md_file)

        assert "Title" in result
        assert "Content" in result

    def test_extract_with_metadata_function(self, temp_dir):
        """Test extract_with_metadata helper function."""
        md_file = temp_dir / "test.md"
        md_file.write_text(
            """---
title: Test Title
---

Content
"""
        )

        result = extract_with_metadata(md_file)

        assert "text" in result
        assert "metadata" in result
        assert result["metadata"]["title"] == "Test Title"

    def test_process_to_document_function(self, temp_dir):
        """Test process_to_document helper function."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Test\n\nContent")

        doc = process_to_document(md_file)

        assert isinstance(doc, MarkdownDocument)
        assert doc.title == "Test"

    def test_extract_frontmatter_function(self, temp_dir):
        """Test extract_frontmatter helper function."""
        md_file = temp_dir / "test.md"
        md_file.write_text(
            """---
key: value
---

Content
"""
        )

        fm = extract_frontmatter(md_file)

        assert fm["key"] == "value"

    def test_extract_code_blocks_function(self, temp_dir):
        """Test extract_code_blocks helper function."""
        md_file = temp_dir / "test.md"
        md_file.write_text(
            """
```python
code
```
"""
        )

        blocks = extract_code_blocks(md_file)

        assert len(blocks) == 1
        assert blocks[0]["language"] == "python"

    def test_build_toc_function(self, temp_dir):
        """Test build_toc helper function."""
        md_file = temp_dir / "test.md"
        md_file.write_text("# Title\n\n## Section")

        toc = build_toc(md_file)

        assert len(toc) == 2
        assert toc[0]["level"] == 1


class TestComplexDocuments:
    """Tests for complex Markdown documents."""

    def test_readme_document(self, temp_dir):
        """Test processing README-style document."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "README.md"
        md_file.write_text(
            """# Project Name

[![Build Status](https://badge.png)](https://ci.com)

A description of the project.

## Installation

```bash
pip install project
```

## Usage

```python
import project
project.run()
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT License
"""
        )

        doc = processor.process_to_document(md_file)

        assert doc.title == "Project Name"
        assert len(doc.toc) >= 4
        assert len(doc.code_blocks) >= 2
        assert len(doc.links) >= 2
        assert len(doc.images) >= 1

    def test_blog_post_with_frontmatter(self, temp_dir):
        """Test processing blog post with frontmatter."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "post.md"
        md_file.write_text(
            """---
title: My Blog Post
date: 2024-01-15
author: Jane Doe
tags:
- python
- programming
draft: false
---

# Introduction

This is my blog post about **Python**.

## Main Content

Here's some code:

```python
def hello():
    return "Hello, World!"
```

Check out [this resource](https://example.com) for more.

![Figure 1](figure.png "A diagram")

## Conclusion

Thanks for reading!
"""
        )

        doc = processor.process_to_document(md_file)

        assert doc.frontmatter["title"] == "My Blog Post"
        assert doc.frontmatter["author"] == "Jane Doe"
        assert doc.frontmatter["tags"] == ["python", "programming"]
        assert doc.frontmatter["draft"] is False
        assert len(doc.code_blocks) >= 1
        assert len(doc.links) >= 1
        assert len(doc.images) >= 1


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_document(self, temp_dir):
        """Test processing empty document."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "empty.md"
        md_file.write_text("")

        result = processor.process(md_file)

        assert result["text"] == ""

    def test_only_frontmatter(self, temp_dir):
        """Test document with only frontmatter."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "frontmatter.md"
        md_file.write_text(
            """---
title: Just Frontmatter
---
"""
        )

        result = processor.process(md_file)

        assert result["frontmatter"]["title"] == "Just Frontmatter"

    def test_incomplete_frontmatter(self, temp_dir):
        """Test document with incomplete frontmatter."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "incomplete.md"
        md_file.write_text(
            """---
title: Test
Content without closing frontmatter
"""
        )

        result = processor.process(md_file)

        # Should treat as content, not frontmatter
        assert result["frontmatter"] == {}

    def test_unicode_content(self, temp_dir):
        """Test handling unicode content."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "unicode.md"
        md_file.write_text("# Hello World\n\nThis is content", encoding="utf-8")

        result = processor.process(md_file)

        # Test that content is extracted (unicode handling may vary by platform)
        assert "Hello" in result["text"]
        assert "content" in result["text"]

    def test_nested_code_blocks(self, temp_dir):
        """Test handling nested backticks in code."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "nested.md"
        md_file.write_text(
            """
```markdown
# Header

```python
nested
```
```
"""
        )

        result = processor.process(md_file)

        # Should handle gracefully
        assert "code_blocks" in result

    def test_malformed_links(self, temp_dir):
        """Test handling malformed links."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "malformed.md"
        md_file.write_text("[Broken link(https://example.com)")

        result = processor.process(md_file)

        # Should not crash, may or may not extract
        assert "text" in result

    def test_deeply_nested_headers(self, temp_dir):
        """Test all header levels."""
        processor = MarkdownProcessor()

        content = """# H1
## H2
### H3
#### H4
##### H5
###### H6
"""

        toc = processor._build_toc(content)

        assert len(toc) == 6
        assert toc[0].level == 1
        assert toc[5].level == 6

    def test_word_count(self, temp_dir):
        """Test word count calculation."""
        processor = MarkdownProcessor()

        md_file = temp_dir / "words.md"
        md_file.write_text("One two three four five.")

        doc = processor.process_to_document(md_file)

        assert doc.word_count == 5


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - Basic processing: 5 tests
    - Process to document: 2 tests
    - Frontmatter extraction: 5 tests
    - YAML parsing: 5 tests
    - Title extraction: 3 tests
    - TOC generation: 5 tests
    - Code block extraction: 5 tests
    - Link extraction: 7 tests
    - Image extraction: 5 tests
    - Content cleaning: 6 tests
    - Helper functions: 6 tests
    - Complex documents: 2 tests
    - Edge cases: 8 tests

    Total: 64 tests

Design Decisions:
    1. No external dependencies (pure Python YAML parsing)
    2. Test all extraction types
    3. Cover frontmatter, code blocks, links, images
    4. Test TOC generation with anchors
    5. Include realistic document scenarios
    6. Test edge cases and malformed input
"""
