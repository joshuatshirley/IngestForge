"""
Source Link Generation for Audit Trail.

Generates clickable file:// URIs and navigation hints for source documents.
Supports multiple link formats for different environments (file, vscode, http).

Architecture Context
--------------------
The SourceLinker creates deep links to source documents:

    SourceLocation → SourceLinker → SourceLink
                                    ├── file:///path/doc.pdf#page=47
                                    ├── vscode://file/path:47:1
                                    └── Navigation: Chapter 3 → Section 2 → Page 47

Platform-Specific Formats
-------------------------
| Platform | File URI Format              | Example                          |
|----------|------------------------------|----------------------------------|
| Windows  | file:///C:/path/file.pdf     | file:///C:/docs/thesis.pdf#page=5|
| macOS    | file:///Users/name/file.pdf  | file:///Users/john/doc.pdf       |
| Linux    | file:///home/user/file.pdf   | file:///home/user/paper.pdf      |

Deep Link Formats
-----------------
| Format   | URI Template                  | Support        |
|----------|-------------------------------|----------------|
| PDF      | file:///path/doc.pdf#page=47  | Most viewers   |
| VS Code  | vscode://file/path:47:1       | Full           |
| HTML     | file:///path/article.html#id  | Universal      |
| Text     | file:///path/notes.md         | Universal      |
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict
from urllib.parse import quote
import sys


@dataclass
class SourceLink:
    """
    A clickable link to a source document.

    Attributes:
        uri: The full URI (file:///path/doc.pdf#page=47)
        file_path: The original file path
        file_exists: Whether the file currently exists on disk
        navigation_hint: Human-readable location (Chapter 3 → Section 2 → Page 47)
        link_type: Type of link generated ("file", "vscode", "http")
    """

    uri: str
    file_path: Path
    file_exists: bool
    navigation_hint: str
    link_type: str

    def __str__(self) -> str:
        """Return the URI for display."""
        return self.uri

    @property
    def status_symbol(self) -> str:
        """Get a status symbol for file existence."""
        return "+" if self.file_exists else "x"

    def to_markdown(self, text: Optional[str] = None) -> str:
        """
        Format as a markdown link.

        Args:
            text: Optional link text (defaults to navigation_hint)

        Returns:
            Markdown formatted link: [text](uri)
        """
        display = text or self.navigation_hint or str(self.file_path.name)
        return f"[{display}]({self.uri})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "uri": self.uri,
            "file_path": str(self.file_path),
            "file_exists": self.file_exists,
            "navigation_hint": self.navigation_hint,
            "link_type": self.link_type,
        }


class SourceLinker:
    """
    Generate clickable links to source documents.

    Creates file:// URIs with appropriate fragments for deep linking
    into documents (PDF pages, HTML anchors, line numbers).

    Example:
        linker = SourceLinker()

        # From a SourceLocation
        link = linker.generate_link(source_location)
        print(link.uri)  # file:///C:/docs/thesis.pdf#page=47
        print(link.navigation_hint)  # Chapter 3 → Section 2 → Page 47

        # Check if file exists
        if not link.file_exists:
            print("Warning: Source file not found")
    """

    def __init__(self, prefer_vscode: bool = False) -> None:
        """
        Initialize the linker.

        Args:
            prefer_vscode: If True, generate vscode:// URIs instead of file://
        """
        self.prefer_vscode = prefer_vscode

    def generate_link(
        self,
        file_path: Optional[str] = None,
        page_start: Optional[int] = None,
        line_start: Optional[int] = None,
        chapter: Optional[str] = None,
        chapter_number: Optional[int] = None,
        section: Optional[str] = None,
        section_number: Optional[str] = None,
        subsection: Optional[str] = None,
        paragraph_number: Optional[int] = None,
        url: Optional[str] = None,
        source_type: Optional[str] = None,
        **kwargs: Any,
    ) -> SourceLink:
        """
        Generate clickable link from source location data.

        Rule #4: Reduced from 64 → 51 lines (shortened docstring)
        """
        # Determine the path and check existence
        path = Path(file_path) if file_path else None
        file_exists = path.exists() if path else False
        uri, link_type = self._determine_uri_and_type(
            path, line_start, url, page_start, source_type, file_path
        )

        # Build navigation hint
        navigation_hint = self._build_navigation_hint(
            chapter=chapter,
            chapter_number=chapter_number,
            section=section,
            section_number=section_number,
            subsection=subsection,
            page_start=page_start,
            paragraph_number=paragraph_number,
        )

        return SourceLink(
            uri=uri,
            file_path=path or Path(""),
            file_exists=file_exists,
            navigation_hint=navigation_hint,
            link_type=link_type,
        )

    def _determine_uri_and_type(
        self,
        path: Optional[Path],
        line_start: Optional[int],
        url: Optional[str],
        page_start: Optional[int],
        source_type: Optional[str],
        file_path: Optional[str],
    ) -> tuple[str, str]:
        """
        Determine URI and link type.

        Rule #4: Extracted to reduce function size

        Returns:
            (uri, link_type) tuple
        """
        if self.prefer_vscode and path:
            return self._generate_vscode_uri(path, line_start), "vscode"
        if url and not file_path:
            return url, "http"
        if path:
            return self._generate_file_uri(path, page_start, source_type), "file"
        return "", "none"

    def _generate_file_uri(
        self,
        path: Path,
        page: Optional[int] = None,
        source_type: Optional[str] = None,
    ) -> str:
        """
        Generate a file:// URI with optional fragment.

        Args:
            path: Path to the file
            page: Page number for PDF documents
            source_type: Type of source for format-specific fragments

        Returns:
            File URI like file:///C:/path/doc.pdf#page=47
        """
        # Resolve to absolute path
        abs_path = path.resolve()

        # Convert to URI format
        if sys.platform == "win32":
            # Windows: file:///C:/path/file.pdf
            # Convert backslashes to forward slashes
            path_str = str(abs_path).replace("\\", "/")
            # Ensure drive letter has proper format
            if len(path_str) >= 2 and path_str[1] == ":":
                path_str = "/" + path_str
            # URL-encode spaces and special characters but not slashes
            parts = path_str.split("/")
            encoded_parts = [quote(p, safe="") for p in parts]
            encoded_path = "/".join(encoded_parts)
            uri = f"file://{encoded_path}"
        else:
            # Unix: file:///path/file.pdf
            path_str = str(abs_path)
            parts = path_str.split("/")
            encoded_parts = [quote(p, safe="") for p in parts]
            encoded_path = "/".join(encoded_parts)
            uri = f"file://{encoded_path}"

        # Add fragment for deep linking
        suffix = path.suffix.lower()
        if page is not None and suffix == ".pdf":
            uri += f"#page={page}"

        return uri

    def _generate_vscode_uri(
        self,
        path: Path,
        line: Optional[int] = None,
        column: int = 1,
    ) -> str:
        """
        Generate a VS Code URI for opening files.

        Args:
            path: Path to the file
            line: Line number to jump to
            column: Column number (default 1)

        Returns:
            VS Code URI like vscode://file/C:/path/file.py:47:1
        """
        abs_path = path.resolve()

        # VS Code uses forward slashes on all platforms
        if sys.platform == "win32":
            path_str = str(abs_path).replace("\\", "/")
        else:
            path_str = str(abs_path)

        # URL-encode the path
        parts = path_str.split("/")
        encoded_parts = [quote(p, safe="") for p in parts]
        encoded_path = "/".join(encoded_parts)

        uri = f"vscode://file/{encoded_path}"

        if line is not None:
            uri += f":{line}:{column}"

        return uri

    def _build_navigation_hint(
        self,
        chapter: Optional[str] = None,
        chapter_number: Optional[int] = None,
        section: Optional[str] = None,
        section_number: Optional[str] = None,
        subsection: Optional[str] = None,
        page_start: Optional[int] = None,
        paragraph_number: Optional[int] = None,
    ) -> str:
        """
        Build a human-readable navigation path.

        Rule #4: Reduced from 70 lines to <60 lines via helper extraction

        Args:
            chapter: Chapter title
            chapter_number: Chapter number
            section: Section title
            section_number: Section number
            subsection: Subsection title
            page_start: Starting page
            paragraph_number: Paragraph number

        Returns:
            Navigation string like "Chapter 3 → Section 2 → Page 47"
        """
        parts = []
        chapter_part = self._format_chapter_part(chapter, chapter_number)
        if chapter_part:
            parts.append(chapter_part)

        section_part = self._format_section_part(section, section_number)
        if section_part:
            parts.append(section_part)

        # Subsection
        if subsection:
            short_sub = subsection[:20] + "..." if len(subsection) > 20 else subsection
            parts.append(short_sub)

        # Page
        if page_start is not None:
            parts.append(f"Page {page_start}")

        # Paragraph
        if paragraph_number is not None:
            parts.append(f"Para {paragraph_number}")

        return " -> ".join(parts) if parts else ""

    def _format_chapter_part(
        self, chapter: Optional[str], chapter_number: Optional[int]
    ) -> str:
        """
        Format chapter navigation part.

        Rule #4: Extracted to reduce function size
        """
        if chapter_number is not None:
            return f"Chapter {chapter_number}"

        if not chapter:
            return ""

        # Try to extract number from chapter title
        import re

        match = re.search(r"Chapter\s+(\d+)", chapter, re.IGNORECASE)
        if match:
            return f"Chapter {match.group(1)}"

        # Use shortened chapter title
        short_chapter = chapter[:30] + "..." if len(chapter) > 30 else chapter
        return short_chapter

    def _format_section_part(
        self, section: Optional[str], section_number: Optional[str]
    ) -> str:
        """
        Format section navigation part.

        Rule #4: Extracted to reduce function size
        """
        if section_number:
            return f"Section {section_number}"

        if not section:
            return ""

        # Try to extract number from section
        import re

        match = re.search(r"^(\d+(?:\.\d+)*)", section)
        if match:
            return f"Section {match.group(1)}"

        short_section = section[:25] + "..." if len(section) > 25 else section
        return short_section


def generate_source_link(
    file_path: Optional[str] = None,
    page_start: Optional[int] = None,
    line_start: Optional[int] = None,
    prefer_vscode: bool = False,
    **kwargs: Any,
) -> SourceLink:
    """
    Convenience function to generate a source link.

    Args:
        file_path: Path to the source file
        page_start: Starting page number
        line_start: Starting line number
        prefer_vscode: Generate VS Code URIs instead of file://
        **kwargs: Additional location data passed to SourceLinker

    Returns:
        SourceLink with URI and metadata
    """
    linker = SourceLinker(prefer_vscode=prefer_vscode)
    return linker.generate_link(
        file_path=file_path,
        page_start=page_start,
        line_start=line_start,
        **kwargs,
    )
