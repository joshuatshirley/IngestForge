"""Markdown document processor.

Processes Markdown (.md) files, extracting text, structure, code blocks,
links, images, and frontmatter."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class CodeBlock:
    """Represents a fenced code block."""

    language: str
    code: str
    position: int = 0
    info_string: str = ""  # Full info string after ```


@dataclass
class Link:
    """Represents a markdown link."""

    text: str
    url: str
    title: Optional[str] = None
    position: int = 0
    is_reference: bool = False  # [text][ref] style


@dataclass
class Image:
    """Represents a markdown image."""

    alt_text: str
    url: str
    title: Optional[str] = None
    position: int = 0


@dataclass
class TocEntry:
    """Represents a table of contents entry."""

    level: int  # 1-6 for h1-h6
    text: str
    anchor: str
    position: int = 0


@dataclass
class Footnote:
    """Represents a markdown footnote."""

    label: str  # Footnote label/id (e.g., "1" or "note")
    content: str  # Footnote content text
    position: int = 0


@dataclass
class MarkdownDocument:
    """Represents a processed Markdown document."""

    title: str = ""
    frontmatter: Dict[str, Any] = field(default_factory=dict)
    toc: List[TocEntry] = field(default_factory=list)
    code_blocks: List[CodeBlock] = field(default_factory=list)
    links: List[Link] = field(default_factory=list)
    wikilinks: List[Link] = field(default_factory=list)
    images: List[Image] = field(default_factory=list)
    footnotes: List[Footnote] = field(default_factory=list)
    text: str = ""
    raw_content: str = ""
    word_count: int = 0


# ============================================================================
# Processor Class
# ============================================================================


class MarkdownProcessor:
    """Process Markdown document files.

    Extracts text, frontmatter, code blocks, links, images, and
    table of contents from Markdown files.
    """

    # Regex patterns compiled at class level
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#*)?\s*$", re.MULTILINE)
    FENCED_CODE_PATTERN = re.compile(
        r"^```(\w*)[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL
    )
    INDENTED_CODE_PATTERN = re.compile(r"^(?:(?:    |\t).+\n?)+", re.MULTILINE)
    INLINE_LINK_PATTERN = re.compile(r'\[([^\]]+)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)')
    REFERENCE_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\[([^\]]*)\]")
    LINK_DEFINITION_PATTERN = re.compile(
        r'^\[([^\]]+)\]:\s*(\S+)(?:\s+"([^"]+)")?\s*$', re.MULTILINE
    )
    IMAGE_PATTERN = re.compile(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)')
    # Footnote patterns: [^label] for references, [^label]: content for definitions
    FOOTNOTE_REF_PATTERN = re.compile(r"\[\^([^\]]+)\](?!\:)")
    FOOTNOTE_DEF_PATTERN = re.compile(
        r"^\[\^([^\]]+)\]:\s*(.+?)(?=\n\[\^|\n\n|\Z)", re.MULTILINE | re.DOTALL
    )
    WIKILINK_PATTERN = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]")

    def process(self, file_path: Path) -> Dict[str, Any]:
        """Process Markdown file.

        Args:
            file_path: Path to .md file

        Returns:
            Dictionary with extracted content and metadata
        """
        content = self._read_file(file_path)
        document = self._build_document(content, file_path)

        return {
            "text": document.text,
            "metadata": self._build_metadata(document, file_path),
            "frontmatter": document.frontmatter,
            "toc": [
                {"level": e.level, "text": e.text, "anchor": e.anchor}
                for e in document.toc
            ],
            "code_blocks": [
                {"language": c.language, "code": c.code} for c in document.code_blocks
            ],
            "links": [{"text": l.text, "url": l.url} for l in document.links],
            "wikilinks": [{"text": l.text, "url": l.url} for l in document.wikilinks],
            "images": [{"alt": i.alt_text, "url": i.url} for i in document.images],
            "footnotes": [
                {"label": f.label, "content": f.content} for f in document.footnotes
            ],
            "type": "markdown",
            "source": str(file_path.name),
        }

    def process_to_document(self, file_path: Path) -> MarkdownDocument:
        """Process Markdown file to structured document.

        Args:
            file_path: Path to .md file

        Returns:
            MarkdownDocument with all extracted content
        """
        content = self._read_file(file_path)
        return self._build_document(content, file_path)

    def _read_file(self, file_path: Path) -> str:
        """Read file with proper encoding handling."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except OSError as e:
            logger.error(f"Failed to read Markdown file {file_path}: {e}")
            raise

    def _build_document(self, content: str, file_path: Path) -> MarkdownDocument:
        """Build complete document structure.

        Rule #4: Orchestrator function - delegates to extractors
        """
        document = MarkdownDocument()
        document.raw_content = content

        # Extract frontmatter first (removes it from content)
        content_without_fm, frontmatter = self._extract_frontmatter(content)
        document.frontmatter = frontmatter

        # Get title from frontmatter or first heading
        document.title = self._extract_title(frontmatter, content_without_fm, file_path)

        # Extract structure and content
        document.toc = self._build_toc(content_without_fm)
        document.code_blocks = self._extract_code_blocks(content_without_fm)
        document.links = self._extract_links(content_without_fm)
        document.wikilinks = self._extract_wikilinks(content_without_fm)
        document.images = self._extract_images(content_without_fm)
        document.footnotes = self._extract_footnotes(content_without_fm)

        # Build clean text content
        document.text = self._clean_content(content_without_fm)
        document.word_count = len(document.text.split())

        return document

    def _build_metadata(
        self, document: MarkdownDocument, file_path: Path
    ) -> Dict[str, Any]:
        """Build metadata dictionary from document."""
        metadata: Dict[str, Any] = {
            "filename": file_path.name,
            "title": document.title,
            "word_count": document.word_count,
            "heading_count": len(document.toc),
            "code_block_count": len(document.code_blocks),
            "link_count": len(document.links),
            "image_count": len(document.images),
            "footnote_count": len(document.footnotes),
        }

        # Add frontmatter fields to metadata
        for key, value in document.frontmatter.items():
            if key not in metadata:
                metadata[key] = value

        return metadata

    # ========================================================================
    # Frontmatter Extraction
    # ========================================================================

    def _extract_frontmatter(self, content: str) -> Tuple[str, Dict[str, Any]]:
        """Extract YAML frontmatter from content.

        Returns:
            Tuple of (content without frontmatter, frontmatter dict)
        """
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            return content, {}

        yaml_content = match.group(1)
        content_without_fm = content[match.end() :]

        frontmatter = self._parse_yaml_simple(yaml_content)
        return content_without_fm, frontmatter

    def _parse_yaml_simple(self, yaml_content: str) -> Dict[str, Any]:
        """Parse simple YAML frontmatter without external dependencies.

        Handles basic key: value pairs, simple lists, and multi-line strings
        using literal (|) and folded (>) block scalars.
        """
        result: Dict[str, Any] = {}

        current_key: Optional[str] = None
        current_list: Optional[List[str]] = None
        multiline_mode: Optional[str] = None  # '|' for literal, '>' for folded
        multiline_lines: List[str] = []
        base_indent: int = 0

        lines = yaml_content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments (unless in multiline mode)
            if not stripped or stripped.startswith("#"):
                if multiline_mode and current_key:
                    multiline_lines.append("")
                i += 1
                continue

            # Check if we're in multiline mode
            if multiline_mode and current_key:
                if self._is_multiline_continuation(line, base_indent):
                    multiline_lines.append(line.lstrip())
                    i += 1
                    continue

                # End of multiline - save and reset
                self._save_multiline_content(
                    result, current_key, multiline_mode, multiline_lines
                )
                multiline_mode = None
                multiline_lines = []

            # Check for list item
            if stripped.startswith("- ") and current_key:
                if current_list is None:
                    current_list = []
                current_list.append(stripped[2:].strip())
                result[current_key] = current_list
                i += 1
                continue

            # Parse key: value
            parsed = self._parse_yaml_line(stripped)
            if parsed:
                key, value = parsed
                current_key = key
                current_list = None

                # Check for multiline indicators
                if value == "|" or value == ">":
                    multiline_mode = value
                    multiline_lines = []
                    base_indent = len(line) - len(line.lstrip())
                else:
                    result[key] = value

            i += 1

        # Handle any remaining multiline content
        if multiline_mode and current_key and multiline_lines:
            self._save_multiline_content(
                result, current_key, multiline_mode, multiline_lines
            )

        return result

    def _is_multiline_continuation(self, line: str, base_indent: int) -> bool:
        """Check if line is continuation of multiline content.

        Args:
            line: Current line
            base_indent: Base indentation level

        Returns:
            True if continuation
        """
        indent = len(line) - len(line.lstrip())
        return indent > base_indent or (indent == base_indent and line.startswith(" "))

    def _save_multiline_content(
        self,
        result: Dict[str, Any],
        key: str,
        mode: str,
        lines: List[str],
    ) -> None:
        """Save multiline content to result.

        Args:
            result: Result dictionary (modified in place)
            key: Key to save under
            mode: '|' for literal, '>' for folded
            lines: Content lines
        """
        if mode == "|":
            result[key] = "\n".join(lines)
        else:  # '>' folded
            result[key] = " ".join(ln for ln in lines if ln)

    def _parse_yaml_line(self, line: str) -> Optional[Tuple[str, Any]]:
        """Parse a single YAML line.

        Rule #1: Early return for invalid lines
        """
        if ":" not in line:
            return None

        parts = line.split(":", 1)
        if len(parts) != 2:
            return None

        key = parts[0].strip()
        value = parts[1].strip()

        # Remove quotes from value
        if value.startswith('"') and value.endswith('"'):
            value = value[1:-1]
        elif value.startswith("'") and value.endswith("'"):
            value = value[1:-1]

        # Handle boolean/null
        value_lower = value.lower()
        if value_lower == "true":
            return key, True
        if value_lower == "false":
            return key, False
        if value_lower in ("null", "~", ""):
            return key, None

        # Try to parse as number, fall back to string
        try:
            if "." in value:
                return key, float(value)
            return key, int(value)
        except ValueError:
            return key, value  # Return as string if not a number

    def _extract_title(
        self, frontmatter: Dict[str, Any], content: str, file_path: Path
    ) -> str:
        """Extract document title.

        Priority: frontmatter title > first h1 > filename
        """
        # Check frontmatter
        if "title" in frontmatter:
            return str(frontmatter["title"])

        # Check for first h1
        h1_match = re.search(r"^#\s+(.+?)(?:\s+#*)?\s*$", content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()

        # Fall back to filename
        return file_path.stem

    # ========================================================================
    # Table of Contents
    # ========================================================================

    def _build_toc(self, content: str) -> List[TocEntry]:
        """Build table of contents from headers.

        Args:
            content: Markdown content

        Returns:
            List of TocEntry objects
        """
        entries: List[TocEntry] = []

        for match in self.HEADER_PATTERN.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            anchor = self._generate_anchor(text)

            entries.append(
                TocEntry(
                    level=level,
                    text=text,
                    anchor=anchor,
                    position=match.start(),
                )
            )

        return entries

    def _generate_anchor(self, text: str) -> str:
        """Generate GitHub-style anchor from header text."""
        # Lowercase
        anchor = text.lower()

        # Remove special characters except spaces and hyphens
        anchor = re.sub(r"[^\w\s-]", "", anchor)

        # Replace spaces with hyphens
        anchor = re.sub(r"\s+", "-", anchor)

        # Remove consecutive hyphens
        anchor = re.sub(r"-+", "-", anchor)

        return anchor.strip("-")

    # ========================================================================
    # Code Block Extraction
    # ========================================================================

    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract all fenced code blocks.

        Args:
            content: Markdown content

        Returns:
            List of CodeBlock objects
        """
        blocks: List[CodeBlock] = []

        for match in self.FENCED_CODE_PATTERN.finditer(content):
            language = match.group(1) or ""
            code = match.group(2).rstrip("\n") if match.group(2) else ""

            blocks.append(
                CodeBlock(
                    language=language,
                    code=code,
                    position=match.start(),
                    info_string="",
                )
            )

        return blocks

    # ========================================================================
    # Link Extraction
    # ========================================================================

    def _extract_links(self, content: str) -> List[Link]:
        """Extract all links from content.

        Handles both inline and reference-style links.
        """
        links: List[Link] = []

        # Build reference definitions map
        ref_definitions = self._extract_link_definitions(content)

        # Extract inline links
        links.extend(self._extract_inline_links(content))

        # Extract reference links
        links.extend(self._extract_reference_links(content, ref_definitions))

        # Sort by position
        return sorted(links, key=lambda l: l.position)

    def _extract_wikilinks(self, content: str) -> List[Link]:
        """Extract Obsidian-style wikilinks [[Page Name|Alias]]."""
        links: List[Link] = []

        for match in self.WIKILINK_PATTERN.finditer(content):
            target = match.group(1).strip()
            alias = match.group(2).strip() if match.group(2) else target

            links.append(
                Link(
                    text=alias,
                    url=target,  # For wikilinks, URL is the target page name
                    position=match.start(),
                    is_reference=False,
                )
            )

        return links

    def _extract_inline_links(self, content: str) -> List[Link]:
        """Extract inline-style links [text](url)."""
        links: List[Link] = []

        for match in self.INLINE_LINK_PATTERN.finditer(content):
            # Skip if this is actually an image
            if match.start() > 0 and content[match.start() - 1] == "!":
                continue

            links.append(
                Link(
                    text=match.group(1),
                    url=match.group(2),
                    title=match.group(3),
                    position=match.start(),
                    is_reference=False,
                )
            )

        return links

    def _extract_reference_links(
        self, content: str, definitions: Dict[str, Tuple[str, Optional[str]]]
    ) -> List[Link]:
        """Extract reference-style links [text][ref]."""
        links: List[Link] = []

        for match in self.REFERENCE_LINK_PATTERN.finditer(content):
            text = match.group(1)
            ref = match.group(2) or text  # Empty ref means use text as ref

            # Look up reference definition
            ref_lower = ref.lower()
            if ref_lower not in definitions:
                continue

            url, title = definitions[ref_lower]

            links.append(
                Link(
                    text=text,
                    url=url,
                    title=title,
                    position=match.start(),
                    is_reference=True,
                )
            )

        return links

    def _extract_link_definitions(
        self, content: str
    ) -> Dict[str, Tuple[str, Optional[str]]]:
        """Extract link reference definitions."""
        definitions: Dict[str, Tuple[str, Optional[str]]] = {}

        for match in self.LINK_DEFINITION_PATTERN.finditer(content):
            ref = match.group(1).lower()
            url = match.group(2)
            title = match.group(3)
            definitions[ref] = (url, title)

        return definitions

    # ========================================================================
    # Image Extraction
    # ========================================================================

    def _extract_images(self, content: str) -> List[Image]:
        """Extract all images from content.

        Args:
            content: Markdown content

        Returns:
            List of Image objects
        """
        images: List[Image] = []

        for match in self.IMAGE_PATTERN.finditer(content):
            images.append(
                Image(
                    alt_text=match.group(1),
                    url=match.group(2),
                    title=match.group(3),
                    position=match.start(),
                )
            )

        return images

    # ========================================================================
    # Footnote Extraction
    # ========================================================================

    def _extract_footnotes(self, content: str) -> List[Footnote]:
        """Extract footnote definitions from content.

        Handles standard markdown footnote syntax: [^label]: content

        Args:
            content: Markdown content

        Returns:
            List of Footnote objects
        """
        footnotes: List[Footnote] = []

        for match in self.FOOTNOTE_DEF_PATTERN.finditer(content):
            label = match.group(1)
            # Clean up multi-line content (remove leading spaces from continuation)
            raw_content = match.group(2)
            # Normalize continuation lines (indented with spaces/tabs)
            lines = raw_content.split("\n")
            cleaned_lines = [lines[0].strip()]
            for line in lines[1:]:
                # Continuation lines are typically indented
                stripped = line.strip()
                if stripped:
                    cleaned_lines.append(stripped)
            content_text = " ".join(cleaned_lines)

            footnotes.append(
                Footnote(
                    label=label,
                    content=content_text,
                    position=match.start(),
                )
            )

        return footnotes

    # ========================================================================
    # Content Cleaning
    # ========================================================================

    def _clean_content(self, content: str) -> str:
        """Clean markdown content to plain text.

        Args:
            content: Markdown content

        Returns:
            Cleaned plain text
        """
        text = content

        # Remove code blocks
        text = self.FENCED_CODE_PATTERN.sub("[CODE]", text)
        text = self.INDENTED_CODE_PATTERN.sub("[CODE]", text)

        # Convert headers to plain text
        text = self.HEADER_PATTERN.sub(r"\2", text)

        # Remove images but note them
        text = self.IMAGE_PATTERN.sub("[IMAGE: \\1]", text)

        # Convert links to just text
        text = self.INLINE_LINK_PATTERN.sub(r"\1", text)
        text = self.REFERENCE_LINK_PATTERN.sub(r"\1", text)

        # Convert wikilinks to just alias text
        # [[Page|Alias]] -> Alias, [[Page]] -> Page
        text = self.WIKILINK_PATTERN.sub(
            lambda m: m.group(2) if m.group(2) else m.group(1), text
        )

        # Remove link definitions
        text = self.LINK_DEFINITION_PATTERN.sub("", text)

        # Remove emphasis markers
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        text = re.sub(r"__(.+?)__", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)

        # Remove inline code markers
        text = re.sub(r"`(.+?)`", r"\1", text)

        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()


# ============================================================================
# Module-level Functions
# ============================================================================


def extract_text(file_path: Path) -> str:
    """Extract text from Markdown file.

    Args:
        file_path: Path to .md file

    Returns:
        Extracted text content
    """
    processor = MarkdownProcessor()
    result = processor.process(file_path)
    return result.get("text", "")


def extract_with_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract text and metadata from Markdown file.

    Args:
        file_path: Path to .md file

    Returns:
        Dictionary with text and metadata
    """
    processor = MarkdownProcessor()
    return processor.process(file_path)


def process_to_document(file_path: Path) -> MarkdownDocument:
    """Process Markdown file to structured document.

    Args:
        file_path: Path to .md file

    Returns:
        MarkdownDocument with all extracted content
    """
    processor = MarkdownProcessor()
    return processor.process_to_document(file_path)


def extract_frontmatter(file_path: Path) -> Dict[str, Any]:
    """Extract only frontmatter from Markdown file.

    Args:
        file_path: Path to .md file

    Returns:
        Frontmatter dictionary
    """
    processor = MarkdownProcessor()
    document = processor.process_to_document(file_path)
    return document.frontmatter


def extract_code_blocks(file_path: Path) -> List[Dict[str, str]]:
    """Extract code blocks from Markdown file.

    Args:
        file_path: Path to .md file

    Returns:
        List of code block dictionaries with 'language' and 'code'
    """
    processor = MarkdownProcessor()
    document = processor.process_to_document(file_path)
    return [{"language": b.language, "code": b.code} for b in document.code_blocks]


def build_toc(file_path: Path) -> List[Dict[str, Any]]:
    """Build table of contents from Markdown file.

    Args:
        file_path: Path to .md file

    Returns:
        List of TOC entry dictionaries
    """
    processor = MarkdownProcessor()
    document = processor.process_to_document(file_path)
    return [
        {"level": e.level, "text": e.text, "anchor": e.anchor} for e in document.toc
    ]
